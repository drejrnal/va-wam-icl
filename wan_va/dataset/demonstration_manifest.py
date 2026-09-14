from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping


SCHEMA_VERSION: Final = 1
EPISODE_FIELDS: Final = frozenset(
    {
        "demo_id",
        "repo_id",
        "episode_id",
        "episode_index",
        "task",
        "family",
        "goal",
        "embodiment",
        "success",
        "split",
        "cameras",
        "prepared_tensor_path",
        "source_seed",
    }
)


class DemonstrationManifestError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DemonstrationRecord:
    demo_id: str
    repo_id: str
    repo_path: Path
    episode_id: str
    episode_index: int
    task: str
    family: str
    goal: str
    embodiment: str
    success: bool
    split: str
    cameras: Mapping[str, str]
    prepared_tensor_path: Path
    source_seed: int


def manifest_error(message: str) -> DemonstrationManifestError:
    return DemonstrationManifestError(f"invalid demonstration manifest: {message}")


def _required_string(raw: Mapping[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise manifest_error(f"{field} must be a non-empty string")
    return value


def _resolve_local_path(manifest_dir: Path, value: str, field: str) -> Path:
    relative_path = Path(value)
    if relative_path.is_absolute():
        raise manifest_error(f"{field} must be manifest-relative")
    resolved = (manifest_dir / relative_path).resolve()
    if not resolved.is_relative_to(manifest_dir.resolve()):
        raise manifest_error(f"{field} escapes the manifest directory")
    return resolved


def parse_record(
    raw: object, manifest_dir: Path, required_cameras: frozenset[str]
) -> DemonstrationRecord:
    if not isinstance(raw, dict):
        raise manifest_error("each episode must be an object")
    unknown = set(raw) - EPISODE_FIELDS
    missing = EPISODE_FIELDS - set(raw)
    if unknown:
        raise manifest_error(f"unknown fields: {sorted(unknown)}")
    if missing:
        raise manifest_error(f"missing fields: {sorted(missing)}")

    episode_index = raw["episode_index"]
    success = raw["success"]
    source_seed = raw["source_seed"]
    cameras = raw["cameras"]
    if (
        not isinstance(episode_index, int)
        or isinstance(episode_index, bool)
        or episode_index < 0
    ):
        raise manifest_error("episode_index must be a non-negative integer")
    if not isinstance(success, bool):
        raise manifest_error("success must be boolean")
    if (
        not isinstance(source_seed, int)
        or isinstance(source_seed, bool)
        or source_seed < 0
    ):
        raise manifest_error("source_seed must be a non-negative integer")
    if not isinstance(cameras, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not value.strip()
        for key, value in cameras.items()
    ):
        raise manifest_error("cameras must map camera names to source strings")
    camera_names = frozenset(cameras)
    if camera_names != required_cameras:
        raise manifest_error(
            f"camera set must equal {sorted(required_cameras)}, got {sorted(camera_names)}"
        )

    prepared_tensor_path = _resolve_local_path(
        manifest_dir,
        _required_string(raw, "prepared_tensor_path"),
        "prepared_tensor_path",
    )
    repo_id = _required_string(raw, "repo_id")
    raw_repo_path = Path(repo_id)
    repo_path = (
        raw_repo_path.resolve()
        if raw_repo_path.is_absolute()
        else (manifest_dir / raw_repo_path).resolve()
    )
    return DemonstrationRecord(
        demo_id=_required_string(raw, "demo_id"),
        repo_id=repo_id,
        repo_path=repo_path,
        episode_id=_required_string(raw, "episode_id"),
        episode_index=episode_index,
        task=_required_string(raw, "task"),
        family=_required_string(raw, "family"),
        goal=_required_string(raw, "goal"),
        embodiment=_required_string(raw, "embodiment"),
        success=success,
        split=_required_string(raw, "split"),
        cameras={
            key: str(_resolve_local_path(manifest_dir, value, f"camera {key}"))
            for key, value in cameras.items()
        },
        prepared_tensor_path=prepared_tensor_path,
        source_seed=source_seed,
    )
