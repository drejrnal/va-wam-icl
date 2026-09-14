import torch
from torch.utils.data import default_collate

from wan_va.dataset.demonstration_tensors import (
    prepare_demo_tensors,
    prepare_demonstration_conditioning,
)


def test_demo_dropout_is_independent_per_example_and_preserves_tensors() -> None:
    # Given a collated demonstration batch and deterministic per-example draws.
    batch = {
        "demo_latents": torch.ones(3, 2, 17, 2, 2),
        "demo_positions": torch.linspace(0, 1, 17).repeat(3, 1),
        "demo_mask": torch.ones(3, 17, dtype=torch.bool),
    }

    # When dropout draws mask examples zero and two.
    result = prepare_demonstration_conditioning(
        batch, dropout_probability=0.2, draws=torch.tensor([0.1, 0.9, 0.0])
    )

    # Then only those examples have all frames masked.
    assert not result["demo_mask"][0].any()
    assert result["demo_mask"][1].all()
    assert not result["demo_mask"][2].any()
    assert result["demo_latents"] is batch["demo_latents"]
    assert result["demo_positions"] is batch["demo_positions"]


def test_prepared_demonstrations_collate_with_variable_source_lengths() -> None:
    # Given demonstrations prepared from different full-video lengths.
    short = prepare_demo_tensors(torch.ones(2, 3, 2, 2)).as_payload()
    long = prepare_demo_tensors(torch.ones(2, 20, 2, 2)).as_payload()

    # When the standard data loader collates them.
    batch = default_collate([short, long])

    # Then batching adds B while preserving the fixed frame axis and pad masks.
    assert batch["demo_latents"].shape == (2, 2, 17, 2, 2)
    assert batch["demo_positions"].shape == (2, 17)
    assert batch["demo_mask"].sum(dim=1).tolist() == [3, 17]
