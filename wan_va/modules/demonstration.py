from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DemonstrationEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: tuple[int, int, int] | list[int],
        attention_dim: int,
    ) -> None:
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.patch_projection = nn.Linear(
            in_channels * math.prod(self.patch_size), attention_dim
        )
        self.position_projection = nn.Sequential(
            nn.Linear(3, attention_dim),
            nn.SiLU(),
            nn.Linear(attention_dim, attention_dim),
        )

    def forward(
        self,
        demo_latents: torch.Tensor,
        demo_positions: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        demo_latents = demo_latents.to(dtype=self.patch_projection.weight.dtype)
        batch_size, _, frames, height, width = demo_latents.shape
        temporal_patch, height_patch, width_patch = self.patch_size
        frame_padding = (-frames) % temporal_patch
        height_padding = (-height) % height_patch
        width_padding = (-width) % width_patch
        masked_latents = demo_latents * demo_mask[:, None, :, None, None]
        padded = F.pad(
            masked_latents,
            (0, width_padding, 0, height_padding, 0, frame_padding),
        )
        if frame_padding:
            last_position = demo_positions[:, -1:].expand(-1, frame_padding)
            demo_positions = torch.cat((demo_positions, last_position), dim=1)
            demo_mask = F.pad(demo_mask, (0, frame_padding), value=False)

        tokens = rearrange(
            padded,
            "b c (f pf) (h ph) (w pw) -> b f h w (c pf ph pw)",
            pf=temporal_patch,
            ph=height_patch,
            pw=width_patch,
        )
        tokens = self.patch_projection(tokens)
        patch_frames, patch_height, patch_width = tokens.shape[1:4]
        position_groups = demo_positions.reshape(
            batch_size, patch_frames, temporal_patch
        )
        mask_groups = demo_mask.reshape(batch_size, patch_frames, temporal_patch)
        temporal_positions = (
            position_groups * mask_groups
        ).sum(dim=-1) / mask_groups.sum(dim=-1).clamp_min(1)
        y_positions = torch.linspace(
            -1, 1, patch_height, device=tokens.device, dtype=tokens.dtype
        )
        x_positions = torch.linspace(
            -1, 1, patch_width, device=tokens.device, dtype=tokens.dtype
        )
        temporal_grid = temporal_positions.to(tokens.dtype)[:, :, None, None].expand(
            -1, -1, patch_height, patch_width
        )
        y_grid = y_positions[None, None, :, None].expand(
            batch_size, patch_frames, -1, patch_width
        )
        x_grid = x_positions[None, None, None, :].expand(
            batch_size, patch_frames, patch_height, -1
        )
        coordinates = torch.stack((temporal_grid, y_grid, x_grid), dim=-1)
        tokens = tokens + self.position_projection(coordinates)

        frame_mask = demo_mask.reshape(
            batch_size, patch_frames, temporal_patch
        ).any(dim=-1)
        token_mask = frame_mask[:, :, None, None].expand(
            -1, -1, patch_height, patch_width
        )
        return tokens.flatten(1, 3), token_mask.flatten(1, 3)


class DemonstrationAttention(nn.Module):
    def __init__(self, model_dim: int, attention_dim: int, num_heads: int) -> None:
        super().__init__()
        if attention_dim <= 0 or num_heads <= 0 or attention_dim % num_heads:
            raise ValueError("demonstration attention_dim must divide num_heads")
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.query = nn.Linear(model_dim, attention_dim)
        self.key = nn.Linear(attention_dim, attention_dim)
        self.value = nn.Linear(attention_dim, attention_dim)
        self.query_norm = nn.LayerNorm(attention_dim)
        self.key_norm = nn.LayerNorm(attention_dim)
        self.output = nn.Linear(attention_dim, model_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self._demo_caches: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _project_demo(
        self, demo_tokens: torch.Tensor, demo_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.key_norm(self.key(demo_tokens)).unflatten(
            -1, (self.num_heads, self.head_dim)
        )
        value = self.value(demo_tokens).unflatten(
            -1, (self.num_heads, self.head_dim)
        )
        return key.transpose(1, 2), value.transpose(1, 2), demo_mask

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        projected_query = self.query_norm(self.query(query)).unflatten(
            -1, (self.num_heads, self.head_dim)
        )
        attended = F.scaled_dot_product_attention(
            projected_query.transpose(1, 2),
            key,
            value,
            attn_mask=valid[None, None, None, :],
        )
        output = self.output(attended.transpose(1, 2).flatten(2))
        has_valid_token = valid.any()
        return torch.where(has_valid_token, output, torch.zeros_like(output))

    def forward(
        self,
        query: torch.Tensor | None,
        demo_tokens: torch.Tensor | None = None,
        demo_mask: torch.Tensor | None = None,
        *,
        query_sample_ids: torch.Tensor | None = None,
        cache_name: str = "pos",
        prepare_cache: bool = False,
    ) -> torch.Tensor | None:
        if demo_tokens is not None and demo_mask is not None:
            key, value, valid_mask = self._project_demo(demo_tokens, demo_mask)
            if prepare_cache:
                self._demo_caches[cache_name] = (
                    key.detach(),
                    value.detach(),
                    valid_mask.detach(),
                )
                return None
        else:
            if cache_name not in self._demo_caches:
                raise RuntimeError(f"demonstration cache {cache_name!r} is not prepared")
            key, value, valid_mask = self._demo_caches[cache_name]

        if query is None:
            raise RuntimeError("query is required outside demonstration cache preparation")
        query_batch = query.shape[0]
        demo_batch = key.shape[0]
        if query_sample_ids is not None:
            if query_batch != 1 or query_sample_ids.shape != (query.shape[1],):
                raise ValueError("packed query sample ids must match a single query sequence")
            output = torch.zeros_like(query)
            for sample_id in range(demo_batch):
                selector = query_sample_ids == sample_id
                output[:, selector] = self._attend(
                    query[:, selector],
                    key[sample_id : sample_id + 1],
                    value[sample_id : sample_id + 1],
                    valid_mask[sample_id],
                )
            return output

        if demo_batch not in (1, query_batch):
            raise ValueError("demonstration batch must be one or match the query batch")
        outputs = []
        for query_id in range(query_batch):
            demo_id = 0 if demo_batch == 1 else query_id
            outputs.append(
                self._attend(
                    query[query_id : query_id + 1],
                    key[demo_id : demo_id + 1],
                    value[demo_id : demo_id + 1],
                    valid_mask[demo_id],
                )
            )
        return torch.cat(outputs, dim=0)

    def clear_pred_cache(self, cache_name: str) -> None:
        return None

    def has_cache(self, cache_name: str) -> bool:
        return cache_name in self._demo_caches

    def clear_cache(self, cache_name: str) -> None:
        self._demo_caches.pop(cache_name, None)
