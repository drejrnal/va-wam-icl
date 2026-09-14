"""Typed registry parsing, experiment construction, and result aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Final, Iterable, Sequence

from wan_va.dataset.demonstration_provenance import (
    DemonstrationProvenance,
    build_demonstration_provenance,
)


SCHEMA_VERSION: Final = 1
EPISODE_FIELDS: Final = frozenset(
    {
        "demo_id", "repo_id", "episode_id", "episode_index", "task", "family",
        "goal", "embodiment", "success", "split", "cameras", "prepared_tensor_path", "source_seed",
    }
)
CONDITIONS: Final = (
    "a_original", "b_none", "c_correct", "wrong_task", "temporal_shuffled",
)


class ExperimentInputError(ValueError):
    """Raised when a registry, plan, or result crosses the CLI boundary malformed."""


@dataclass(frozen=True, slots=True)
class Episode:
    demo_id: str
    repo_id: str
    episode_id: str
    episode_index: int
    task: str
    family: str
    goal: str
    embodiment: str
    success: bool
    split: str
    cameras: dict[str, str]
    prepared_tensor_path: str
    source_seed: int
    provenance: DemonstrationProvenance


@dataclass(frozen=True, slots=True)
class _ProvenanceRegistry:
    registry_sha256: str


@dataclass(frozen=True, slots=True)
class _ProvenanceEpisode:
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
    prepared_tensor_path: Path


def load_registry(path: Path, *, require_provenance: bool = False) -> list[Episode]:
    """Parse the versioned, strict episode registry at the external boundary."""
    raw = _load_json(path)
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "episodes"}:
        raise ExperimentInputError("registry must contain only schema_version and episodes")
    if raw["schema_version"] != SCHEMA_VERSION or not isinstance(raw["episodes"], list):
        raise ExperimentInputError("registry has an unsupported schema_version or episodes value")
    episodes = [_parse_episode(entry) for entry in raw["episodes"]]
    demo_ids = [episode.demo_id for episode in episodes]
    if len(demo_ids) != len(set(demo_ids)):
        raise ExperimentInputError("registry demo_id values must be globally unique")
    if not require_provenance:
        return episodes
    registry = _ProvenanceRegistry(_canonical_sha256(raw))
    return [replace(episode, provenance=_provenance_for_episode(path, registry, episode)) for episode in episodes]


def episode_json(episode: Episode) -> dict[str, object]:
    """Serialize only canonical manifest fields after split reassignment."""
    return {field: getattr(episode, field) for field in EPISODE_FIELDS}


def split_episodes(episodes: Sequence[Episode], split_seed: str) -> list[Episode]:
    """Assign whole sorted families to deterministic 70/10/20 train/val/test buckets."""
    families = sorted({episode.family for episode in episodes}, key=lambda item: _rank(split_seed, item))
    if len(families) < 10:
        raise ExperimentInputError("family split requires at least 10 explicit families for nonempty 70/10/20 buckets")
    family_splits = {family: _split_for_index(index, len(families)) for index, family in enumerate(families)}
    return [replace(episode, split=family_splits[episode.family]) for episode in episodes]


def split_summary(episodes: Sequence[Episode]) -> dict[str, object]:
    """Summarize achieved family and episode counts without changing registry schema."""
    families: dict[str, set[str]] = defaultdict(set)
    episode_counts: dict[str, int] = defaultdict(int)
    for episode in episodes:
        families[episode.split].add(episode.family)
        episode_counts[episode.split] += 1
    return {
        "family_count": len({episode.family for episode in episodes}),
        "families_by_split": {split: len(families[split]) for split in ("train", "val", "test")},
        "episodes_by_split": {split: episode_counts[split] for split in ("train", "val", "test")},
    }


def build_plan(
    episodes: Sequence[Episode],
    a_checkpoint: str,
    modified_checkpoint: str,
    rollout_seeds: Sequence[int],
    evaluation_split: str,
    training_seed: int,
    task_configs: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Build trials only from held-out successful episodes and explicit controls."""
    _validate_rollout_seeds(rollout_seeds)
    _validate_family_splits(episodes)
    if set(task_configs) != {"Clean", "Random"}:
        raise ExperimentInputError("plan requires immutable Clean and Random task config profiles")
    if a_checkpoint == modified_checkpoint:
        raise ExperimentInputError("A checkpoint and modified checkpoint must differ")
    targets = [episode for episode in episodes if episode.success and episode.split == evaluation_split]
    target_groups: dict[tuple[str, str, str, str], list[Episode]] = defaultdict(list)
    for target in targets:
        target_groups[_identity(target)].append(target)
    trials: list[dict[str, object]] = []
    excluded_task_goals: list[dict[str, object]] = []
    for identity, group in sorted(target_groups.items()):
        target = min(group, key=lambda item: _rank(training_seed, item.demo_id))
        correct_pool = list(group)
        wrong_pool = [
            episode for episode in targets
            if episode.embodiment == target.embodiment and episode.task != target.task
        ]
        try:
            correct_pool = _source_seed_safe(correct_pool, rollout_seeds)
            wrong_pool = _source_seed_safe(wrong_pool, rollout_seeds)
        except ExperimentInputError as error:
            excluded_task_goals.append(_exclusion(target, str(error)))
            continue
        if len(correct_pool) < 3 or len(wrong_pool) < 3:
            excluded_task_goals.append(_exclusion(target, "fewer than three valid correct or wrong-task supports"))
            continue
        correct_supports = _select_supports(correct_pool, training_seed)
        wrong_supports = _select_supports(wrong_pool, training_seed)
        for rollout_seed in rollout_seeds:
            trials.extend(_trials_for_target(target, correct_supports, wrong_supports, rollout_seed, training_seed))
    if not trials:
        reasons = "; ".join(sorted({str(item["reason"]) for item in excluded_task_goals}))
        raise ExperimentInputError(
            "no held-out task/goal has three exact-match and three wrong-task source-seed-safe supports"
            f": {reasons}"
        )
    for trial in trials:
        trial["expected_checkpoint"] = a_checkpoint if trial["condition"] == "a_original" else modified_checkpoint
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_split": evaluation_split,
        "rollout_seeds": list(rollout_seeds),
        "rollout_count": len(rollout_seeds),
        "training_seed": training_seed,
        "task_configs": task_configs,
        "coverage": {
            "held_out_task_goal_count": len(target_groups),
            "held_out_task_goals": [_exclusion(target, None) for target in (min(group, key=lambda item: _rank(training_seed, item.demo_id)) for group in target_groups.values())],
            "eligible_task_goal_count": len(target_groups) - len(excluded_task_goals),
            "excluded_task_goals": excluded_task_goals,
            "primary_estimand": "equal-weight task success and C(correct)-B(none) among support-eligible held-out task/goals only",
        },
        "arms": [
            {"condition": "a_original", "checkpoint": a_checkpoint, "demo_mode": "none"},
            {"condition": "b_none", "checkpoint": modified_checkpoint, "demo_mode": "none"},
            {"condition": "c_correct", "checkpoint": modified_checkpoint, "demo_mode": "correct"},
            {"condition": "wrong_task", "checkpoint": modified_checkpoint, "demo_mode": "wrong_task"},
            {"condition": "temporal_shuffled", "checkpoint": modified_checkpoint, "demo_mode": "shuffled"},
        ],
        "trials": trials,
    }


def aggregate_results(plan_path: Path, results_path: Path, bootstrap_samples: int) -> dict[str, object]:
    """Aggregate observed runs without supplying any missing simulator result."""
    from evaluation.robotwin.demo_result_aggregation import aggregate_setting, parse_results

    plan = _load_json(plan_path)
    if not isinstance(plan, dict) or not isinstance(plan.get("trials"), list) or not isinstance(plan.get("coverage"), dict) or not isinstance(plan.get("task_configs"), dict):
        raise ExperimentInputError("plan must contain trials, held-out coverage, and task configs")
    trials = {str(item["trial_id"]): item for item in plan["trials"] if isinstance(item, dict)}
    result_rows = _load_json_lines(results_path)
    observations = parse_results(result_rows, trials, plan["task_configs"])
    settings = {
        "Clean": aggregate_setting(observations, "Clean", bootstrap_samples, trials),
        "Random": aggregate_setting(observations, "Random", bootstrap_samples, trials),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "primary_estimand": plan["coverage"].get("primary_estimand"),
        "coverage": plan["coverage"],
        "task_configs": plan["task_configs"],
        "settings": settings,
        "observed_result_count": len(observations),
    }


def write_json(path: Path, data: object) -> None:
    """Write a reproducible JSON artifact after all boundary validation succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentInputError(f"cannot read JSON {path}: {error}") from error


def _load_json_lines(path: Path) -> list[object]:
    try:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentInputError(f"cannot read JSONL {path}: {error}") from error


def _parse_episode(raw: object) -> Episode:
    if not isinstance(raw, dict) or set(raw) != EPISODE_FIELDS:
        raise ExperimentInputError("each episode must contain exactly the canonical manifest fields")
    string_fields = EPISODE_FIELDS - {"episode_index", "success", "cameras", "source_seed"}
    if any(not isinstance(raw[field], str) or not raw[field] for field in string_fields):
        raise ExperimentInputError("episode string fields must be nonempty strings")
    cameras = raw["cameras"]
    if not isinstance(cameras, dict) or not cameras or any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in cameras.items()):
        raise ExperimentInputError("episode cameras must be a nonempty mapping of paths")
    index = raw["episode_index"]
    source_seed = raw["source_seed"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or not isinstance(raw["success"], bool) or isinstance(source_seed, bool) or not isinstance(source_seed, int):
        raise ExperimentInputError("episode_index/source_seed must be integers and success must be boolean")
    return Episode(**raw, provenance={})


def _canonical_sha256(raw: object) -> str:
    return sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _provenance_for_episode(manifest_path: Path, registry: _ProvenanceRegistry, episode: Episode) -> DemonstrationProvenance:
    relative_path = Path(episode.prepared_tensor_path)
    if relative_path.is_absolute():
        raise ExperimentInputError("prepared_tensor_path must be manifest-relative")
    resolved_path = (manifest_path.resolve().parent / relative_path).resolve()
    if not resolved_path.is_relative_to(manifest_path.resolve().parent):
        raise ExperimentInputError("prepared_tensor_path escapes manifest directory")
    record = _ProvenanceEpisode(
        demo_id=episode.demo_id, repo_id=episode.repo_id, episode_id=episode.episode_id,
        episode_index=episode.episode_index, task=episode.task, family=episode.family,
        goal=episode.goal, embodiment=episode.embodiment, split=episode.split,
        source_seed=episode.source_seed, prepared_tensor_path=resolved_path,
    )
    try:
        digest = sha256(resolved_path.read_bytes()).hexdigest()
        return build_demonstration_provenance(registry, record, digest)
    except OSError as error:
        raise ExperimentInputError(f"cannot read prepared demonstration {resolved_path}: {error}") from error


def _split_for_index(index: int, total: int) -> str:
    if index * 10 < total * 7:
        return "train"
    if index * 10 < total * 8:
        return "val"
    return "test"


def _validate_family_splits(episodes: Iterable[Episode]) -> None:
    splits: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        splits[episode.family].add(episode.split)
    leaked = sorted(family for family, values in splits.items() if len(values) != 1)
    if leaked:
        raise ExperimentInputError(f"families cross splits: {', '.join(leaked)}")


def _validate_rollout_seeds(rollout_seeds: Sequence[int]) -> None:
    if len(rollout_seeds) != 100 or len(set(rollout_seeds)) != 100:
        raise ExperimentInputError("exactly 100 distinct rollout seeds are required")


def _identity(episode: Episode) -> tuple[str, str, str, str]:
    return episode.task, episode.family, episode.goal, episode.embodiment


def _rank(seed: str | int, value: str) -> str:
    return sha256(f"{seed}:{value}".encode()).hexdigest()


def _source_seed_safe(pool: Sequence[Episode], rollout_seeds: Sequence[int]) -> list[Episode]:
    overlaps = sorted({episode.source_seed for episode in pool} & set(rollout_seeds))
    if overlaps:
        raise ExperimentInputError(f"support source_seed overlaps rollout seed: {overlaps}")
    return list(pool)


def _exclusion(target: Episode, reason: str | None) -> dict[str, object]:
    output: dict[str, object] = {"task": target.task, "family": target.family, "goal": target.goal, "embodiment": target.embodiment}
    if reason is not None:
        output["reason"] = reason
    return output


def _select_supports(pool: Sequence[Episode], training_seed: int) -> list[Episode]:
    return sorted(pool, key=lambda episode: _rank(training_seed, episode.demo_id))[:3]


def _trials_for_target(target: Episode, correct: Sequence[Episode], wrong: Sequence[Episode], rollout_seed: int, training_seed: int) -> list[dict[str, object]]:
    values = []
    for condition in CONDITIONS:
        pool = correct if condition in {"c_correct", "temporal_shuffled"} else wrong
        selected = pool[rollout_seed % 3] if condition in {"c_correct", "wrong_task", "temporal_shuffled"} else None
        trial = {
            "trial_id": f"{target.demo_id}:{rollout_seed}:{condition}", "condition": condition,
            "target_demo_id": target.demo_id, "task": target.task, "family": target.family,
            "goal": target.goal, "embodiment": target.embodiment, "fixed_prompt": target.goal,
            "target_episode_id": target.episode_id, "rollout_seed": rollout_seed, "training_seed": training_seed,
            "checkpoint_arm": "a_original" if condition == "a_original" else "modified",
            "support_pool_demo_ids": [episode.demo_id for episode in pool] if selected is not None else [],
            "support_pool_source_seeds": [episode.source_seed for episode in pool] if selected is not None else [],
        }
        if selected is not None:
            trial["demo_id"] = selected.demo_id
            trial["expected_demo_provenance"] = selected.provenance
        if condition == "temporal_shuffled":
            trial["demo_shuffle_seed"] = selected.source_seed
        values.append(trial)
    return values
