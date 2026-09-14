from __future__ import annotations

import torch

from wan_va.demonstration_inference import (
    DemonstrationPayload,
    prepare_demonstration_cache,
    shuffle_temporal_content,
)


class RecordingTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, torch.Tensor], bool, str]] = []

    def __call__(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        prepare_demo_cache: bool,
        cache_name: str,
    ) -> None:
        self.calls.append((input_dict, prepare_demo_cache, cache_name))


def _payload() -> DemonstrationPayload:
    return DemonstrationPayload(
        demo_latents=torch.arange(5, dtype=torch.float32).reshape(1, 5, 1, 1),
        demo_positions=torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
        demo_mask=torch.tensor([True, True, True, False, False]),
    )


def test_prepare_demonstration_cache_when_payload_is_unbatched() -> None:
    # Given
    transformer = RecordingTransformer()

    # When
    prepare_demonstration_cache(
        transformer,
        _payload(),
        cache_name="episode",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Then
    prepared, prepare_demo_cache, cache_name = transformer.calls[0]
    assert prepared["demo_latents"].shape == (1, 1, 5, 1, 1)
    assert prepared["demo_positions"].shape == (1, 5)
    assert prepared["demo_mask"].shape == (1, 5)
    assert prepare_demo_cache is True
    assert cache_name == "episode"


def test_shuffle_temporal_content_when_frames_are_padded() -> None:
    # Given
    payload = _payload()

    # When
    shuffled = shuffle_temporal_content(payload, seed=0)

    # Then
    assert shuffled.demo_positions.equal(payload.demo_positions)
    assert shuffled.demo_mask.equal(payload.demo_mask)
    assert shuffled.demo_latents[:, :3].flatten().tolist() == [2.0, 0.0, 1.0]
    assert shuffled.demo_latents[:, 3:].equal(payload.demo_latents[:, 3:])


def test_shuffle_temporal_content_when_payload_is_batched() -> None:
    # Given
    base = _payload()
    payload = DemonstrationPayload(
        demo_latents=torch.stack(
            [base.demo_latents, base.demo_latents + 10], dim=0
        ),
        demo_positions=torch.stack([base.demo_positions, base.demo_positions]),
        demo_mask=torch.stack([base.demo_mask, base.demo_mask]),
    )

    # When
    shuffled = shuffle_temporal_content(payload, seed=0)

    # Then
    assert shuffled.demo_latents[0, :, :3].flatten().tolist() == [2.0, 0.0, 1.0]
    assert sorted(shuffled.demo_latents[1, :, :3].flatten().tolist()) == [10.0, 11.0, 12.0]
    assert shuffled.demo_positions.equal(payload.demo_positions)
