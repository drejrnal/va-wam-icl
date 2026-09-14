# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lerobot_latent_dataset import MultiLatentLeRobotDataset

__all__ = [
    'MultiLatentLeRobotDataset',
    'dataset_indexes_ready',
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from .lerobot_latent_dataset import (
        MultiLatentLeRobotDataset,
        dataset_indexes_ready,
    )

    exports = {
        'MultiLatentLeRobotDataset': MultiLatentLeRobotDataset,
        'dataset_indexes_ready': dataset_indexes_ready,
    }
    return exports[name]
