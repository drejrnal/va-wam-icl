from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as functional

ROBOTWIN_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _patchify(video: torch.Tensor, patch_size: int | tuple[int, int]) -> torch.Tensor:
    if patch_size == 1:
        return video
    spatial_patch = patch_size[0] if isinstance(patch_size, tuple) else patch_size
    batch, channels, frames, height, width = video.shape
    video = video.view(
        batch,
        channels,
        frames,
        height // spatial_patch,
        spatial_patch,
        width // spatial_patch,
        spatial_patch,
    )
    video = video.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return video.view(
        batch,
        channels * spatial_patch * spatial_patch,
        frames,
        height // spatial_patch,
        width // spatial_patch,
    )


class _StreamingVAEEncoder:
    def __init__(self, vae) -> None:
        self.vae = vae
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        conv_count = (
            vae._cached_conv_counts["encoder"]
            if hasattr(vae, "_cached_conv_counts")
            else sum(
                module.__class__.__name__ == "WanCausalConv3d"
                for module in self.encoder.modules()
            )
        )
        self.feature_cache = [None] * conv_count

    def encode_chunk(self, video: torch.Tensor) -> torch.Tensor:
        patch_size = getattr(self.vae.config, "patch_size", None)
        if patch_size is not None:
            video = _patchify(video, patch_size)
        feature_index = [0]
        encoded = self.encoder(
            video, feat_cache=self.feature_cache, feat_idx=feature_index)
        return self.quant_conv(encoded)


@dataclass(frozen=True, slots=True)
class VideoEncodingOptions:
    vae_path: Path
    layout: str
    image_size: tuple[int, int]
    chunk_frames: int = 17
    device: str = "cuda"


def parse_camera_assignments(values: Sequence[str]) -> dict[str, Path]:
    assignments: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("camera inputs must use CAMERA=PATH")
        if name in assignments:
            raise ValueError(f"duplicate camera input: {name}")
        assignments[name] = Path(raw_path)
    return assignments


def combine_camera_latents(
    camera_latents: Mapping[str, torch.Tensor], layout: str
) -> torch.Tensor:
    if tuple(camera_latents) != ROBOTWIN_CAMERAS:
        raise ValueError(f"camera order must be {ROBOTWIN_CAMERAS}")
    high, left, right = (camera_latents[name] for name in ROBOTWIN_CAMERAS)
    if any(latent.ndim != 4 for latent in (high, left, right)):
        raise ValueError("camera latents must have shape [C,F,H,W]")
    if any(not torch.isfinite(latent).all() for latent in (high, left, right)):
        raise ValueError("camera latents must contain finite normalized values")
    if len({latent.shape[:2] for latent in (high, left, right)}) != 1:
        raise ValueError("camera latent channels and frame counts must match")
    match layout:
        case "robotwin_tshape":
            wrists = torch.cat((left, right), dim=-1)
            if wrists.shape[-1] != high.shape[-1]:
                raise ValueError("combined wrist width must equal high camera width")
            return torch.cat((wrists, high), dim=-2)
        case "horizontal":
            if len({latent.shape[-2] for latent in (high, left, right)}) != 1:
                raise ValueError("horizontal camera latent heights must match")
            return torch.cat((high, left, right), dim=-1)
        case _:
            raise ValueError(f"unsupported layout: {layout}")


def load_encoded_camera_latents(paths: Mapping[str, Path]) -> dict[str, torch.Tensor]:
    loaded: dict[str, torch.Tensor] = {}
    for name in ROBOTWIN_CAMERAS:
        if name not in paths:
            raise ValueError(f"missing camera input: {name}")
        value = torch.load(paths[name], map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"encoded camera input must be a tensor: {name}")
        loaded[name] = value
    return loaded


def _encode_video_chunk(
    frames: list[np.ndarray], wrapper: _StreamingVAEEncoder, size: tuple[int, int]
) -> torch.Tensor:
    video = torch.from_numpy(np.stack(frames)).float().permute(3, 0, 1, 2)
    video = functional.interpolate(
        video, size=size, mode="bilinear", align_corners=False
    ).unsqueeze(0)
    device = next(wrapper.vae.parameters()).device
    dtype = next(wrapper.vae.parameters()).dtype
    encoded = wrapper.encode_chunk(
        (video / 255.0 * 2.0 - 1.0).to(device=device, dtype=dtype))
    mean, _ = torch.chunk(encoded, 2, dim=1)
    latent_mean = torch.tensor(wrapper.vae.config.latents_mean, device=device)
    latent_std = torch.tensor(wrapper.vae.config.latents_std, device=device)
    return ((mean.float() - latent_mean.view(1, -1, 1, 1, 1)) / latent_std.view(1, -1, 1, 1, 1)).cpu()


@torch.inference_mode()
def encode_synchronized_videos(
    paths: Mapping[str, Path],
    options: VideoEncodingOptions,
) -> torch.Tensor:
    from diffusers import AutoencoderKLWan

    if tuple(paths) != ROBOTWIN_CAMERAS:
        raise ValueError(f"camera order must be {ROBOTWIN_CAMERAS}")
    if options.chunk_frames < 5 or (options.chunk_frames - 1) % 4:
        raise ValueError("chunk_frames must equal 4n+1 and be at least five")
    if any(dimension < 16 or dimension % 16 for dimension in options.image_size):
        raise ValueError("image height and width must be positive multiples of 16")
    dtype = torch.bfloat16 if options.device.startswith("cuda") else torch.float32
    vae = AutoencoderKLWan.from_pretrained(
        str(options.vae_path), torch_dtype=dtype).to(options.device)
    wrappers = {name: _StreamingVAEEncoder(vae) for name in ROBOTWIN_CAMERAS}
    iterators = {name: iter(iio.imiter(paths[name])) for name in ROBOTWIN_CAMERAS}
    camera_outputs: dict[str, list[torch.Tensor]] = {name: [] for name in ROBOTWIN_CAMERAS}
    pending: dict[str, list[np.ndarray]] = {name: [] for name in ROBOTWIN_CAMERAS}
    current_chunk_frames = options.chunk_frames
    while True:
        frame_set: dict[str, np.ndarray] = {}
        ended: list[str] = []
        for name in ROBOTWIN_CAMERAS:
            try:
                frame_set[name] = next(iterators[name])
            except StopIteration:
                ended.append(name)
        if ended:
            if len(ended) != len(ROBOTWIN_CAMERAS):
                raise ValueError("camera videos must have equal frame counts")
            break
        for name, frame in frame_set.items():
            pending[name].append(frame)
        if len(pending[ROBOTWIN_CAMERAS[0]]) == current_chunk_frames:
            for index, name in enumerate(ROBOTWIN_CAMERAS):
                height, width = options.image_size
                size = (height, width) if index == 0 else (height // 2, width // 2)
                camera_outputs[name].append(_encode_video_chunk(pending[name], wrappers[name], size))
                pending[name] = []
            current_chunk_frames = options.chunk_frames - 1
    if pending[ROBOTWIN_CAMERAS[0]]:
        valid_tail_frames = len(pending[ROBOTWIN_CAMERAS[0]])
        tail_remainder = (
            (valid_tail_frames - 1) % 4
            if not camera_outputs[ROBOTWIN_CAMERAS[0]]
            else valid_tail_frames % 4
        )
        padding_frames = (4 - tail_remainder) % 4
        for name in ROBOTWIN_CAMERAS:
            pending[name].extend([pending[name][-1]] * padding_frames)
        for index, name in enumerate(ROBOTWIN_CAMERAS):
            height, width = options.image_size
            size = (height, width) if index == 0 else (height // 2, width // 2)
            camera_outputs[name].append(_encode_video_chunk(pending[name], wrappers[name], size))
    if not camera_outputs[ROBOTWIN_CAMERAS[0]]:
        raise ValueError("camera videos must contain at least one frame")
    camera_latents = {
        name: torch.cat(chunks, dim=2).squeeze(0) for name, chunks in camera_outputs.items()
    }
    return combine_camera_latents(camera_latents, options.layout)
