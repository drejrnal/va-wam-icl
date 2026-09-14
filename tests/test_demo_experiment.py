"""End-to-end coverage for the demonstration evaluation CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "script" / "demo_experiment.py"


def _episode(*, task: str, episode_index: int, family: str = "pick") -> dict[str, object]:
    return {
        "demo_id": f"{task}-{episode_index}",
        "repo_id": "robotwin",
        "episode_id": f"episode-{task}-{episode_index}",
        "episode_index": episode_index,
        "task": task,
        "family": family,
        "goal": "place cube",
        "embodiment": "bimanual",
        "success": True,
        "split": "test",
        "cameras": {"head": f"videos/{task}-{episode_index}.mp4"},
        "prepared_tensor_path": f"tensors/{task}-{episode_index}.pt",
        "source_seed": 1000 + episode_index,
    }


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the public CLI exactly as a researcher would."""
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _write_registry(path: Path, records: list[dict[str, object]]) -> None:
    for record in records:
        tensor_path = path.parent / str(record["prepared_tensor_path"])
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_path.write_bytes(str(record["demo_id"]).encode("utf-8"))
    path.write_text(json.dumps({"schema_version": 1, "episodes": records}), encoding="utf-8")


def _write_task_config(path: Path, *, random: bool) -> None:
    path.write_text(json.dumps({"domain_randomization": {
        "cluttered_table": random,
        "random_background": random,
        "random_light": random,
        "random_table_height": random,
        "random_head_camera_dis": random,
    }}), encoding="utf-8")


def _plan_configs(tmp_path: Path) -> tuple[Path, Path]:
    clean, random = tmp_path / "clean.yml", tmp_path / "random.yml"
    _write_task_config(clean, random=False)
    _write_task_config(random, random=True)
    return clean, random


def test_plan_when_registry_has_compatible_demos_writes_five_conditions(tmp_path: Path) -> None:
    # Given: five same-task demos and one separately named wrong-task source.
    registry = tmp_path / "registry.json"
    plan = tmp_path / "plan.json"
    records = [_episode(task="pick-cube", episode_index=index) for index in range(5)]
    records.extend(_episode(task="push-cube", episode_index=10 + index, family="push") for index in range(3))
    records.extend(_episode(task=f"task-{index}", episode_index=100 + index, family=f"family-{index}") for index in range(8))
    _write_registry(registry, records)
    clean, random = _plan_configs(tmp_path)

    # When: the public planner receives distinct support and rollout seed spaces.
    completed = _run(
        "plan",
        "--registry",
        str(registry),
        "--output",
        str(plan),
        "--a-checkpoint",
        "checkpoints/original",
        "--modified-checkpoint",
        "checkpoints/demo",
        "--training-seed",
        "0",
        "--clean-task-config", str(clean),
        "--random-task-config", str(random),
    )

    # Then: it writes reproducible A/B/C and negative-control assignments.
    assert completed.returncode == 0, completed.stderr
    output = json.loads(plan.read_text(encoding="utf-8"))
    assert output["rollout_count"] == 100
    assert {arm["condition"] for arm in output["arms"]} == {
        "a_original",
        "b_none",
        "c_correct",
        "wrong_task",
        "temporal_shuffled",
    }
    correct = [trial for trial in output["trials"] if trial["condition"] == "c_correct"]
    assert correct
    assert all(len(trial["support_pool_demo_ids"]) == 3 for trial in correct)
    assert all(trial["demo_id"] in trial["support_pool_demo_ids"] for trial in correct)
    assert set(correct[0]["expected_demo_provenance"]) == {
        "registry_sha256", "demo_id", "repo_id", "episode_id", "episode_index",
        "task", "family", "goal", "embodiment", "split", "source_seed",
        "prepared_tensor_sha256",
    }
    assert output["coverage"]["held_out_task_goal_count"] >= output["coverage"]["eligible_task_goal_count"]
    by_condition = {trial["condition"]: trial for trial in output["trials"] if trial["target_demo_id"] == correct[0]["target_demo_id"] and trial["rollout_seed"] == correct[0]["rollout_seed"]}
    assert by_condition["a_original"]["expected_checkpoint"] == "checkpoints/original"
    assert "demo_id" not in by_condition["a_original"]
    assert by_condition["b_none"]["expected_checkpoint"] == "checkpoints/demo"
    assert "demo_id" not in by_condition["b_none"]
    assert by_condition["c_correct"]["expected_checkpoint"] == "checkpoints/demo"
    assert by_condition["c_correct"]["expected_demo_provenance"]["task"] == by_condition["c_correct"]["task"]
    assert by_condition["wrong_task"]["expected_demo_provenance"]["task"] != by_condition["wrong_task"]["task"]
    assert by_condition["temporal_shuffled"]["demo_id"] == by_condition["c_correct"]["demo_id"]
    assert by_condition["temporal_shuffled"]["demo_shuffle_seed"] == by_condition["c_correct"]["expected_demo_provenance"]["source_seed"]


def test_plan_when_seed_spaces_overlap_reports_a_boundary_error(tmp_path: Path) -> None:
    # Given: a syntactically valid registry.
    registry = tmp_path / "registry.json"
    _write_registry(registry, [_episode(task="pick-cube", episode_index=index) for index in range(5)])
    clean, random = _plan_configs(tmp_path)

    # When: a rollout seed overlaps a support-selection seed.
    completed = _run(
        "plan",
        "--registry",
        str(registry),
        "--output",
        str(tmp_path / "plan.json"),
        "--a-checkpoint",
        "a",
        "--modified-checkpoint",
        "b",
        "--rollout-seeds",
        ",".join(str(value) for value in range(1000, 1100)),
        "--training-seed",
        "0",
        "--clean-task-config", str(clean),
        "--random-task-config", str(random),
    )

    # Then: no experiment plan is produced from contaminated randomization.
    assert completed.returncode != 0
    assert "source_seed overlaps rollout seed" in completed.stderr.lower()


def test_help_when_requested_describes_the_three_public_workflows() -> None:
    # Given: the CLI entry point.
    # When: a researcher asks for help.
    completed = _run("--help")

    # Then: split, planning, and aggregation are discoverable.
    assert completed.returncode == 0
    assert "split" in completed.stdout
    assert "plan" in completed.stdout
    assert "aggregate" in completed.stdout


def test_split_and_aggregate_when_observed_rollouts_are_supplied_keep_settings_separate(tmp_path: Path) -> None:
    # Given: a plan with paired A/C trials and two families that need a family-level split.
    registry = tmp_path / "registry.json"
    split = tmp_path / "split.json"
    plan = tmp_path / "plan.json"
    results = tmp_path / "results.jsonl"
    report = tmp_path / "report.json"
    records = [_episode(task="pick-cube", episode_index=index) for index in range(5)]
    records.extend(_episode(task="push-cube", episode_index=10 + index, family="push") for index in range(3))
    records.extend(
        _episode(task=f"task-{index}", episode_index=100 + index, family=f"family-{index}")
        for index in range(8)
    )
    _write_registry(registry, records)
    clean, random = _plan_configs(tmp_path)

    # When: fixed recorded outcomes are aggregated through the public command.
    split_run = _run("split", "--registry", str(registry), "--output", str(split), "--split-seed", "test")
    plan_run = _run("plan", "--registry", str(registry), "--output", str(plan), "--a-checkpoint", "a", "--modified-checkpoint", "b", "--evaluation-split", "test", "--training-seed", "0", "--clean-task-config", str(clean), "--random-task-config", str(random))
    assert split_run.returncode == 0, split_run.stderr
    assert plan_run.returncode == 0, plan_run.stderr
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    planned = plan_data["trials"]
    paired = [row for row in planned if row["condition"] in {"b_none", "c_correct"}][:2]
    results.write_text("\n".join(json.dumps({"trial_id": row["trial_id"], "setting": "Clean", "success": row["condition"] == "c_correct", "actual_rollout_seed": row["rollout_seed"], "actual_demo_id": row.get("demo_id"), "actual_demo_provenance": row.get("expected_demo_provenance"), "actual_checkpoint": row["expected_checkpoint"], "actual_task_config": plan_data["task_configs"]["Clean"], "fixed_prompt": row["fixed_prompt"], "latency_ms": 10.0}) for row in paired) + "\n", encoding="utf-8")
    completed = _run("aggregate", "--plan", str(plan), "--results", str(results), "--output", str(report), "--bootstrap-samples", "20")

    # Then: the report retains Clean and Random sections, task rates, paired deltas, and latency.
    assert completed.returncode == 0, completed.stderr
    output = json.loads(report.read_text(encoding="utf-8"))
    assert output["settings"]["Clean"]["per_task_success"][0]["latency_ms"] == 10.0
    assert output["settings"]["Clean"]["primary_comparison"]["mean_task_delta"] == 1.0
    assert output["settings"]["Random"]["observed_result_count"] == 0
    assert output["coverage"]["primary_estimand"].endswith("support-eligible held-out task/goals only")


def test_split_when_fewer_than_ten_families_reports_nonempty_bucket_requirement(tmp_path: Path) -> None:
    # Given: an explicit registry too small to populate train, validation, and test families.
    registry = tmp_path / "registry.json"
    _write_registry(registry, [_episode(task="pick-cube", episode_index=0)])

    # When: the public split command is requested.
    completed = _run("split", "--registry", str(registry), "--output", str(tmp_path / "split.json"))

    # Then: it rejects the invalid family denominator instead of emitting an empty evaluation bucket.
    assert completed.returncode != 0
    assert "at least 10 explicit families" in completed.stderr


def test_split_when_payloads_are_not_prepared_remains_metadata_only(tmp_path: Path) -> None:
    # Given: ten manifest records whose future prepared tensor paths do not exist yet.
    registry = tmp_path / "registry.json"
    records = [_episode(task=f"task-{index}", episode_index=index, family=f"family-{index}") for index in range(10)]
    registry.write_text(json.dumps({"schema_version": 1, "episodes": records}), encoding="utf-8")

    # When: metadata is split before the tensor-preparation workflow.
    completed = _run("split", "--registry", str(registry), "--output", str(tmp_path / "split.json"))

    # Then: it emits a split and a nonempty family-count summary without hashing absent payloads.
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "split.split-summary.json").read_text(encoding="utf-8"))
    assert summary["families_by_split"] == {"test": 2, "train": 7, "val": 1}
