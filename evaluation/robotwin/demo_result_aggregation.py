"""Observed-result parsing and task-weighted paired reporting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from statistics import fmean
from typing import Mapping, Sequence

from evaluation.robotwin.demo_experiment_core import ExperimentInputError


@dataclass(frozen=True, slots=True)
class Observation:
    trial_id: str
    condition: str
    setting: str
    task: str
    target_demo_id: str
    rollout_seed: int
    training_seed: int
    success: float
    latency_ms: float | None
    peak_memory_mb: float | None


def parse_results(rows: Sequence[object], trials: Mapping[str, object], task_configs: Mapping[str, object]) -> list[Observation]:
    """Join result JSONL to exact plan rows and reject altered runtime metadata."""
    observations = [_parse_row(row, trials, task_configs) for row in rows]
    identifiers = [f"{item.trial_id}:{item.setting}" for item in observations]
    if len(identifiers) != len(set(identifiers)):
        raise ExperimentInputError("result file contains duplicate trial_id and setting rows")
    return observations


def aggregate_setting(observations: Sequence[Observation], setting: str, bootstrap_samples: int, trials: Mapping[str, object]) -> dict[str, object]:
    """Return observed coverage, task rates, and primary C-minus-B paired delta."""
    rows = [row for row in observations if row.setting == setting]
    per_task = _per_task(rows)
    return {
        "planned_result_count": len(trials), "observed_result_count": len(rows),
        "missing_result_count": len(trials) - len(rows), "per_task_success": per_task,
        "task_mean_success": _task_means(per_task),
        "primary_comparison": _comparison("c_correct", "b_none", rows, trials, bootstrap_samples),
        "diagnostic_comparisons_vs_a": [
            _comparison(condition, "a_original", rows, trials, bootstrap_samples)
            for condition in ("b_none", "c_correct", "wrong_task", "temporal_shuffled")
        ],
    }


def _parse_row(raw: object, trials: Mapping[str, object], task_configs: Mapping[str, object]) -> Observation:
    allowed = {"trial_id", "condition", "setting", "success", "actual_rollout_seed", "actual_demo_id", "actual_demo_provenance", "actual_checkpoint", "actual_task_config", "fixed_prompt", "latency_ms", "peak_memory_mb"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ExperimentInputError("result rows contain unsupported fields")
    trial_id, setting, success = raw.get("trial_id"), raw.get("setting"), raw.get("success")
    if not isinstance(trial_id, str) or not isinstance(setting, str) or not isinstance(success, bool):
        raise ExperimentInputError("result requires string trial_id/setting and boolean success")
    if setting not in {"Clean", "Random"}:
        raise ExperimentInputError("setting must be Clean or Random")
    trial = trials.get(trial_id)
    if not isinstance(trial, dict):
        raise ExperimentInputError(f"result references unknown trial_id: {trial_id}")
    if raw.get("condition") is not None and raw.get("condition") != trial.get("condition"):
        raise ExperimentInputError("result condition does not match planned condition")
    _verify_runtime_metadata(raw, trial, task_configs, setting)
    return Observation(
        trial_id=trial_id, condition=_string(trial, "condition"), setting=setting,
        task=_string(trial, "task"), target_demo_id=_string(trial, "target_demo_id"),
        rollout_seed=_integer(trial, "rollout_seed"), training_seed=_integer(trial, "training_seed"),
        success=float(success), latency_ms=_number(raw, "latency_ms"), peak_memory_mb=_number(raw, "peak_memory_mb"),
    )


def _verify_runtime_metadata(raw: Mapping[str, object], trial: Mapping[str, object], task_configs: Mapping[str, object], setting: str) -> None:
    for result_field, plan_field in (("actual_rollout_seed", "rollout_seed"), ("fixed_prompt", "fixed_prompt"), ("actual_checkpoint", "expected_checkpoint")):
        if raw.get(result_field) != trial.get(plan_field):
            raise ExperimentInputError(f"result {result_field} does not match planned {plan_field}")
    if raw.get("actual_demo_id") != trial.get("demo_id"):
        raise ExperimentInputError("result actual_demo_id does not match planned demo_id")
    if raw.get("actual_demo_provenance") != trial.get("expected_demo_provenance"):
        raise ExperimentInputError("result actual_demo_provenance does not match planned support")
    if raw.get("actual_task_config") != task_configs.get(setting):
        raise ExperimentInputError("result actual_task_config does not match planned setting")


def _per_task(rows: Sequence[Observation]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for row in rows:
        grouped[(row.condition, row.task)].append(row)
    return [_summary(condition, task, values) for (condition, task), values in sorted(grouped.items())]


def _summary(condition: str, task: str, rows: Sequence[Observation]) -> dict[str, object]:
    summary: dict[str, object] = {"condition": condition, "task": task, "success_rate": fmean(row.success for row in rows), "count": len(rows)}
    _optional_mean(summary, "latency_ms", [row.latency_ms for row in rows])
    _optional_mean(summary, "peak_memory_mb", [row.peak_memory_mb for row in rows])
    return summary


def _task_means(per_task: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in per_task:
        rate = row["success_rate"]
        if isinstance(rate, float):
            grouped[_string(row, "condition")].append(rate)
    return [{"condition": condition, "mean_task_success": fmean(rates), "task_count": len(rates), "task_weighting": "equal"} for condition, rates in sorted(grouped.items())]


def _comparison(left: str, right: str, rows: Sequence[Observation], trials: Mapping[str, object], bootstrap_samples: int) -> dict[str, object]:
    expected = _expected(left, right, trials)
    observed = {(row.condition, _key(row)): row for row in rows}
    paired = [(key[0], observed[(left, key)].success - observed[(right, key)].success) for key in expected if (left, key) in observed and (right, key) in observed]
    summary: dict[str, object] = {
        "label": f"{left} minus {right}", "left_condition": left, "right_condition": right,
        "expected_pair_count": len(expected), "observed_pair_count": len(paired),
        "missing_pair_count": len(expected) - len(paired), "population": "fixed_evaluated_tasks_equal_weight",
    }
    if paired:
        task_means = _task_delta_means(paired)
        lower, upper = _bootstrap(task_means, bootstrap_samples, summary["label"])
        summary.update({"mean_task_delta": fmean(task_means.values()), "task_count": len(task_means), "bootstrap_95_ci": [lower, upper]})
    return summary


def _expected(left: str, right: str, trials: Mapping[str, object]) -> set[tuple[str, str, int, int]]:
    values: dict[str, set[tuple[str, str, int, int]]] = defaultdict(set)
    for trial in trials.values():
        if isinstance(trial, dict):
            values[_string(trial, "condition")].add((_string(trial, "task"), _string(trial, "target_demo_id"), _integer(trial, "rollout_seed"), _integer(trial, "training_seed")))
    return values[left] & values[right]


def _key(row: Observation) -> tuple[str, str, int, int]:
    return row.task, row.target_demo_id, row.rollout_seed, row.training_seed


def _task_delta_means(paired: Sequence[tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for task, delta in paired:
        grouped[task].append(delta)
    return {task: fmean(values) for task, values in grouped.items()}


def _bootstrap(task_means: Mapping[str, float], sample_count: int, seed: object) -> tuple[float, float]:
    if sample_count < 1:
        raise ExperimentInputError("bootstrap_samples must be positive")
    generator = random.Random(str(seed))
    values = list(task_means.values())
    means = sorted(fmean(generator.choice(values) for _ in values) for _ in range(sample_count))
    return means[int(0.025 * (sample_count - 1))], means[int(0.975 * (sample_count - 1))]


def _string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ExperimentInputError(f"plan {field} must be a string")
    return value


def _integer(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentInputError(f"plan {field} must be an integer")
    return value


def _number(row: Mapping[str, object], field: str) -> float | None:
    value = row.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentInputError(f"result {field} must be numeric when supplied")
    return float(value)


def _optional_mean(summary: dict[str, object], field: str, values: Sequence[float | None]) -> None:
    observed = [value for value in values if value is not None]
    if observed:
        summary[field] = fmean(observed)
