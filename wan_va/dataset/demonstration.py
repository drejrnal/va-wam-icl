from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import torch

from .demonstration_tensors import (
    DemoTensors,
    load_prepared_demonstration,
    load_prepared_demonstration_with_digest,
    prepare_demo_tensors,
)
from .demonstration_manifest import (
    DemonstrationManifestError,
    DemonstrationRecord,
    EPISODE_FIELDS,
    SCHEMA_VERSION,
    manifest_error as _manifest_error,
    parse_record as _parse_record,
)
from .demonstration_provenance import (
    DemonstrationProvenance,
    build_demonstration_provenance,
)


MAX_DEMO_FRAMES: Final = 17


@dataclass(frozen=True, slots=True)
class DemonstrationRegistry:
    records: tuple[DemonstrationRecord, ...]
    registry_sha256: str

    @classmethod
    def from_file(
        cls, path: str | Path, required_cameras: Sequence[str] | Mapping[str, str]
    ) -> DemonstrationRegistry:
        manifest_path = Path(path).resolve()
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise _manifest_error(str(error)) from error
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "episodes"}:
            raise _manifest_error("top level must contain only schema_version and episodes")
        if raw["schema_version"] != SCHEMA_VERSION:
            raise _manifest_error(f"schema_version must be {SCHEMA_VERSION}")
        episodes = raw["episodes"]
        if not isinstance(episodes, list) or not episodes:
            raise _manifest_error("episodes must be a non-empty list")
        registry_sha256 = hashlib.sha256(
            json.dumps(
                raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        records = tuple(
            _parse_record(item, manifest_path.parent, frozenset(required_cameras))
            for item in episodes
        )
        demo_ids = [record.demo_id for record in records]
        if len(demo_ids) != len(set(demo_ids)):
            raise _manifest_error("duplicate demo_id")
        episode_ids = [(record.repo_path, record.episode_id) for record in records]
        if len(episode_ids) != len(set(episode_ids)):
            raise _manifest_error("duplicate repo_id and episode_id")
        physical_episodes = [(record.repo_path, record.episode_index) for record in records]
        if len(physical_episodes) != len(set(physical_episodes)):
            raise _manifest_error("duplicate repo_id and episode_index")
        family_splits: dict[str, str] = {}
        for record in records:
            prior_split = family_splits.setdefault(record.family, record.split)
            if prior_split != record.split:
                raise _manifest_error(
                    f"family {record.family} appears in multiple splits"
                )
        return cls(records=records, registry_sha256=registry_sha256)

    def resolve_demo(
        self,
        demo_id: str,
        *,
        expected_task: str | None = None,
        expected_goal: str | None = None,
        expected_embodiment: str | None = None,
        expected_split: str | None = None,
    ) -> DemonstrationRecord:
        matches = [record for record in self.records if record.demo_id == demo_id]
        if not matches:
            raise _manifest_error(f"unknown demo_id: {demo_id}")
        record = matches[0]
        expectations = (
            ("task", expected_task, record.task),
            ("goal", expected_goal, record.goal),
            ("embodiment", expected_embodiment, record.embodiment),
            ("split", expected_split, record.split),
        )
        for field, expected, actual in expectations:
            if expected is not None and actual != expected:
                raise _manifest_error(f"{field} mismatch for demo_id {demo_id}")
        return record

    def select_reference(
        self, target: DemonstrationRecord, generator: torch.Generator | None = None
    ) -> DemonstrationRecord:
        candidates = self.compatible_references(target)
        if not candidates:
            raise _manifest_error(f"no compatible reference for {target.demo_id}")
        choice = torch.randint(len(candidates), (), generator=generator).item()
        return candidates[choice]

    def compatible_references(
        self, target: DemonstrationRecord
    ) -> tuple[DemonstrationRecord, ...]:
        distinct = sorted(
            (
                record
                for record in self.records
                if record.demo_id != target.demo_id
                and (
                    record.repo_path != target.repo_path
                    or record.episode_id != target.episode_id
                )
            ),
            key=lambda record: record.demo_id,
        )
        return tuple(
            record
            for record in distinct
            if record.success
            and record.split == target.split
            and record.task == target.task
            and record.family == target.family
            and record.goal == target.goal
            and record.embodiment == target.embodiment
        )


def pairable_episode_indices(
    registry: DemonstrationRegistry, repo_id: str | Path, split: str
) -> dict[int, DemonstrationRecord]:
    targets: dict[int, DemonstrationRecord] = {}
    for record in registry.records:
        if (
            record.repo_path != Path(repo_id).resolve()
            or record.split != split
            or not record.success
        ):
            continue
        if not registry.compatible_references(record):
            raise _manifest_error(f"no compatible reference for {record.demo_id}")
        targets[record.episode_index] = record
    return targets


def retain_nonempty_datasets(datasets_in):
    datasets_out = [dataset for dataset in datasets_in if len(dataset) > 0]
    if not datasets_out:
        raise RuntimeError("no pairable dataset episodes found across repositories")
    return datasets_out


__all__ = [
    "DemoTensors",
    "DemonstrationProvenance",
    "DemonstrationManifestError",
    "DemonstrationRecord",
    "DemonstrationRegistry",
    "EPISODE_FIELDS",
    "MAX_DEMO_FRAMES",
    "SCHEMA_VERSION",
    "build_demonstration_provenance",
    "load_prepared_demonstration",
    "load_prepared_demonstration_with_digest",
    "pairable_episode_indices",
    "prepare_demo_tensors",
    "retain_nonempty_datasets",
]
