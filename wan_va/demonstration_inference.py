from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

import torch


class DemoModelInput(TypedDict):
    demo_latents: torch.Tensor
    demo_positions: torch.Tensor
    demo_mask: torch.Tensor


class DemonstrationCacheModel(Protocol):
    def __call__(
        self,
        input_dict: DemoModelInput,
        *,
        prepare_demo_cache: bool,
        cache_name: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class DemonstrationPayload:
    demo_latents: torch.Tensor
    demo_positions: torch.Tensor
    demo_mask: torch.Tensor


def _batched_payload(payload: DemonstrationPayload) -> DemonstrationPayload:
    latents = payload.demo_latents
    positions = payload.demo_positions
    mask = payload.demo_mask
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if latents.ndim != 5 or positions.ndim != 2 or mask.ndim != 2:
        raise ValueError("demonstration tensors must be [B,C,F,H,W], [B,F], [B,F]")
    if latents.shape[0] != positions.shape[0] or positions.shape != mask.shape:
        raise ValueError("demonstration tensor batch dimensions must match")
    if latents.shape[2] != positions.shape[1]:
        raise ValueError("demonstration tensor frame dimensions must match")
    return DemonstrationPayload(latents, positions, mask.to(dtype=torch.bool))


def shuffle_temporal_content(
    payload: DemonstrationPayload, *, seed: int
) -> DemonstrationPayload:
    batched = _batched_payload(payload)
    shuffled = batched.demo_latents.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for batch_index in range(shuffled.shape[0]):
        valid_indices = torch.nonzero(
            batched.demo_mask[batch_index].cpu(), as_tuple=False
        ).flatten()
        permutation = valid_indices[
            torch.randperm(valid_indices.numel(), generator=generator)
        ].to(shuffled.device)
        target_indices = valid_indices.to(shuffled.device)
        shuffled[batch_index, :, target_indices] = batched.demo_latents[
            batch_index, :, permutation
        ]
    result = DemonstrationPayload(
        shuffled, batched.demo_positions, batched.demo_mask
    )
    if payload.demo_latents.ndim == 4:
        return DemonstrationPayload(
            result.demo_latents[0], result.demo_positions[0], result.demo_mask[0]
        )
    return result


def prepare_demonstration_cache(
    model: DemonstrationCacheModel,
    payload: DemonstrationPayload,
    *,
    cache_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    batched = _batched_payload(payload)
    model(
        DemoModelInput(
            demo_latents=batched.demo_latents.to(device=device, dtype=dtype),
            demo_positions=batched.demo_positions.to(
                device=device, dtype=torch.float32
            ),
            demo_mask=batched.demo_mask.to(device=device),
        ),
        prepare_demo_cache=True,
        cache_name=cache_name,
    )
