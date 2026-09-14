# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.9.0", "diffusers==0.36.0", "imageio[ffmpeg]", "numpy>=1.26.4,<2"]
# ///
# ─── How to run ───
# uv run script/prepare_robot_demonstrations.py --help

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wan_va.dataset.demonstration import MAX_DEMO_FRAMES, prepare_demo_tensors
from wan_va.dataset.demonstration_prepare import (
    ROBOTWIN_CAMERAS,
    VideoEncodingOptions,
    combine_camera_latents,
    encode_synchronized_videos,
    load_encoded_camera_latents,
    parse_camera_assignments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full-episode robot video demonstrations."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera-latent", action="append", default=[])
    source.add_argument("--camera-video", action="append", default=[])
    parser.add_argument("--vae-path", type=Path)
    parser.add_argument(
        "--layout", choices=("robotwin_tshape", "horizontal"), required=True
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--chunk-frames", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--camera-latents-normalized",
        action="store_true",
        help="Confirm --camera-latent tensors already use Wan latent normalization.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_assignments = args.camera_latent or args.camera_video
    assignments = parse_camera_assignments(raw_assignments)
    if set(assignments) != set(ROBOTWIN_CAMERAS):
        raise ValueError(f"camera inputs must equal {ROBOTWIN_CAMERAS}")
    assignments = {name: assignments[name] for name in ROBOTWIN_CAMERAS}
    if args.camera_latent:
        if not args.camera_latents_normalized:
            raise ValueError(
                "--camera-latents-normalized is required with --camera-latent"
            )
        full_latents = combine_camera_latents(
            load_encoded_camera_latents(assignments), args.layout
        )
    else:
        if args.vae_path is None:
            raise ValueError("--vae-path is required with --camera-video")
        full_latents = encode_synchronized_videos(
            assignments,
            VideoEncodingOptions(
                vae_path=args.vae_path,
                layout=args.layout,
                image_size=(args.height, args.width),
                chunk_frames=args.chunk_frames,
                device=args.device,
            ),
        )
    prepared = prepare_demo_tensors(full_latents, MAX_DEMO_FRAMES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prepared.as_payload(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
