from __future__ import annotations

from typing import Protocol, TypedDict


class ProvenanceRecord(Protocol):
    demo_id: str
    repo_id: str
    episode_id: str
    episode_index: int
    task: str
    family: str
    goal: str
    embodiment: str
    split: str
    source_seed: int


class ProvenanceRegistry(Protocol):
    registry_sha256: str


class DemonstrationProvenance(TypedDict):
    registry_sha256: str
    demo_id: str
    repo_id: str
    episode_id: str
    episode_index: int
    task: str
    family: str
    goal: str
    embodiment: str
    split: str
    source_seed: int
    prepared_tensor_sha256: str


def build_demonstration_provenance(
    registry: ProvenanceRegistry,
    record: ProvenanceRecord,
    prepared_tensor_sha256: str,
) -> DemonstrationProvenance:
    if (
        len(prepared_tensor_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in prepared_tensor_sha256
        )
    ):
        raise ValueError("prepared_tensor_sha256 must be a lowercase SHA-256 digest")
    return {
        "registry_sha256": registry.registry_sha256,
        "demo_id": record.demo_id,
        "repo_id": record.repo_id,
        "episode_id": record.episode_id,
        "episode_index": record.episode_index,
        "task": record.task,
        "family": record.family,
        "goal": record.goal,
        "embodiment": record.embodiment,
        "split": record.split,
        "source_seed": record.source_seed,
        "prepared_tensor_sha256": prepared_tensor_sha256,
    }
