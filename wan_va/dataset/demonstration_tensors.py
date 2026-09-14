from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch


class PreparedDemonstration(Protocol):
    prepared_tensor_path: Path


@dataclass(frozen=True, slots=True)
class DemoTensors:
    demo_latents: torch.Tensor
    demo_positions: torch.Tensor
    demo_mask: torch.Tensor

    def as_payload(self) -> dict[str, torch.Tensor]:
        return {
            "demo_latents": self.demo_latents,
            "demo_positions": self.demo_positions,
            "demo_mask": self.demo_mask,
        }


def prepare_demonstration_conditioning(
    batch: dict[str, torch.Tensor],
    dropout_probability: float,
    draws: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    required = {"demo_latents", "demo_positions", "demo_mask"}
    if set(batch) != required:
        raise RuntimeError("demonstration batch must contain exactly three tensors")
    mask = batch["demo_mask"]
    latents = batch["demo_latents"]
    positions = batch["demo_positions"]
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise RuntimeError("batched demo_mask must have shape [B,F] and boolean dtype")
    if (
        latents.ndim != 5
        or positions.ndim != 2
        or latents.shape[0] != mask.shape[0]
        or latents.shape[2] != mask.shape[1]
        or positions.shape != mask.shape
    ):
        raise RuntimeError("batched demonstration tensor shapes are inconsistent")
    if not 0 <= dropout_probability <= 1:
        raise RuntimeError("demonstration dropout probability must be in [0,1]")
    dropout_draws = draws
    if dropout_draws is None:
        dropout_draws = torch.rand(mask.shape[0], device=mask.device)
    if dropout_draws.shape != (mask.shape[0],):
        raise RuntimeError("demonstration dropout draws must have shape [B]")
    keep = dropout_draws >= dropout_probability
    return {
        "demo_latents": batch["demo_latents"],
        "demo_positions": batch["demo_positions"],
        "demo_mask": mask & keep[:, None],
    }


def prepare_demo_tensors(
    full_video_latents: torch.Tensor, max_frames: int = 17
) -> DemoTensors:
    if full_video_latents.ndim != 4 or full_video_latents.shape[1] < 1:
        raise RuntimeError("demo latent source must have shape [C,F,H,W]")
    if not full_video_latents.is_floating_point() or not torch.isfinite(full_video_latents).all():
        raise RuntimeError("demo latent source must contain finite floating-point values")
    if max_frames < 2:
        raise RuntimeError("max_frames must be at least two")
    source_frames = full_video_latents.shape[1]
    sampled_frames = min(source_frames, max_frames)
    frame_ids = torch.linspace(0, source_frames - 1, sampled_frames).round().long()
    sampled = full_video_latents.index_select(1, frame_ids).detach().cpu()
    positions = frame_ids.float() / max(source_frames - 1, 1)
    latents = torch.zeros(
        (sampled.shape[0], max_frames, sampled.shape[2], sampled.shape[3]),
        dtype=sampled.dtype,
    )
    padded_positions = torch.zeros(max_frames, dtype=torch.float32)
    mask = torch.zeros(max_frames, dtype=torch.bool)
    latents[:, :sampled_frames] = sampled
    padded_positions[:sampled_frames] = positions
    mask[:sampled_frames] = True
    return DemoTensors(latents, padded_positions, mask)


def load_prepared_demonstration(
    source: PreparedDemonstration | str | Path, max_frames: int = 17
) -> DemoTensors:
    path = source if isinstance(source, (str, Path)) else source.prepared_tensor_path
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return _validate_prepared_payload(payload, max_frames)


def load_prepared_demonstration_with_digest(
    source: PreparedDemonstration | str | Path, max_frames: int = 17
) -> tuple[DemoTensors, str]:
    path = Path(
        source if isinstance(source, (str, Path)) else source.prepared_tensor_path
    )
    try:
        snapshot = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"failed to read prepared demonstration: {path}") from error
    digest = hashlib.sha256(snapshot).hexdigest()
    payload = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    return _validate_prepared_payload(payload, max_frames), digest


def _validate_prepared_payload(payload: object, max_frames: int) -> DemoTensors:
    if not isinstance(payload, dict) or set(payload) != {
        "demo_latents",
        "demo_positions",
        "demo_mask",
    }:
        raise RuntimeError("prepared payload must contain only demo tensors")
    latents = payload["demo_latents"]
    positions = payload["demo_positions"]
    mask = payload["demo_mask"]
    if not all(isinstance(value, torch.Tensor) for value in (latents, positions, mask)):
        raise RuntimeError("prepared payload values must be tensors")
    if (
        latents.ndim != 4
        or any(dimension < 1 for dimension in latents.shape)
        or positions.shape != (max_frames,)
        or mask.shape != (max_frames,)
    ):
        raise RuntimeError("prepared demo tensor shapes are invalid")
    if (
        latents.shape[1] != max_frames
        or not latents.is_floating_point()
        or not positions.is_floating_point()
        or mask.dtype != torch.bool
    ):
        raise RuntimeError("prepared demo tensor dtypes or frame count are invalid")
    if not torch.isfinite(latents).all() or not torch.isfinite(positions).all():
        raise RuntimeError("prepared demo tensors must be finite")
    if not mask.any():
        raise RuntimeError("prepared demo must contain at least one valid frame")
    valid_count = int(mask.sum().item())
    expected_mask = torch.arange(max_frames) < valid_count
    if not torch.equal(mask.cpu(), expected_mask):
        raise RuntimeError("prepared demo mask must be a valid prefix followed by padding")
    valid_positions = positions[mask]
    if torch.any(valid_positions < 0) or torch.any(valid_positions > 1):
        raise RuntimeError("demo positions must be normalized")
    if valid_positions.numel() > 1 and torch.any(valid_positions[1:] < valid_positions[:-1]):
        raise RuntimeError("valid demo positions must be monotonic")
    if valid_positions[0] != 0 or (valid_count > 1 and valid_positions[-1] != 1):
        raise RuntimeError("valid demo positions must retain source endpoints")
    if torch.any(latents[:, ~mask] != 0):
        raise RuntimeError("padded demo latents must be zero")
    if torch.any(positions[~mask] != 0):
        raise RuntimeError("padded demo positions must be zero")
    return DemoTensors(latents, positions.float(), mask)
