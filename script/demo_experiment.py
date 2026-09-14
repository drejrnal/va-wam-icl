#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML"]
# ///
# ─── How to run ───
# uv run script/demo_experiment.py --help
"""Create held-out robot demonstration trials and aggregate recorded outcomes."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys
from typing import Sequence

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.robotwin.demo_experiment_core import (
    ExperimentInputError,
    aggregate_results,
    build_plan,
    episode_json,
    load_registry,
    split_summary,
    split_episodes,
    write_json,
)


DEFAULT_ROLLOUT_SEEDS = ",".join(str(value) for value in range(100, 200))
RANDOMIZATION_KEYS = (
    "cluttered_table",
    "random_background",
    "random_light",
    "random_table_height",
    "random_head_camera_dis",
)


def _integer_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or any(not item.strip() for item in value.split(",")):
        raise argparse.ArgumentTypeError("seeds must be nonempty comma-separated integers")
    return seeds


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _add_registry_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", required=True, type=_path, help="schema_version=1 episode registry JSON")


def _split_command(arguments: argparse.Namespace) -> None:
    if arguments.output.resolve().parent != arguments.registry.resolve().parent:
        raise ExperimentInputError("split output must remain beside the input manifest so relative paths stay valid")
    episodes = load_registry(arguments.registry)
    split_episodes_result = split_episodes(episodes, arguments.split_seed)
    write_json(
        arguments.output,
        {"schema_version": 1, "episodes": [episode_json(item) for item in split_episodes_result]},
    )
    summary_path = arguments.summary_output or arguments.output.with_name(f"{arguments.output.stem}.split-summary.json")
    write_json(summary_path, split_summary(split_episodes_result))


def _plan_command(arguments: argparse.Namespace) -> None:
    episodes = load_registry(arguments.registry, require_provenance=True)
    write_json(
        arguments.output,
        build_plan(
            episodes=episodes,
            a_checkpoint=arguments.a_checkpoint,
            modified_checkpoint=arguments.modified_checkpoint,
            rollout_seeds=arguments.rollout_seeds,
            evaluation_split=arguments.evaluation_split,
            training_seed=arguments.training_seed,
            task_configs={
                "Clean": _task_config_profile(arguments.clean_task_config, "Clean"),
                "Random": _task_config_profile(arguments.random_task_config, "Random"),
            },
        ),
    )


def _task_config_profile(path: Path, setting: str) -> dict[str, object]:
    try:
        contents = path.read_bytes()
        config = yaml.safe_load(contents)
    except (OSError, yaml.YAMLError) as error:
        raise ExperimentInputError(f"cannot read {setting} task config {path}: {error}") from error
    if not isinstance(config, dict) or not isinstance(config.get("domain_randomization"), dict):
        raise ExperimentInputError(f"{setting} task config requires domain_randomization mapping")
    randomization = config["domain_randomization"]
    if any(key not in randomization or not isinstance(randomization[key], bool) for key in RANDOMIZATION_KEYS):
        raise ExperimentInputError(f"{setting} task config must define boolean randomization sentinels")
    is_random = any(randomization[key] for key in RANDOMIZATION_KEYS)
    if (setting == "Random") != is_random:
        raise ExperimentInputError(f"{setting} task config randomization sentinels disagree with setting")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(contents).hexdigest(),
        "domain_randomization": randomization,
    }


def _aggregate_command(arguments: argparse.Namespace) -> None:
    write_json(arguments.output, aggregate_results(arguments.plan, arguments.results, arguments.bootstrap_samples))


def build_parser() -> argparse.ArgumentParser:
    """Construct the public split, plan, and aggregate command surface."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="assign a deterministic family-level 70/10/20 split")
    _add_registry_argument(split)
    split.add_argument("--output", required=True, type=_path)
    split.add_argument("--summary-output", type=_path, help="achieved family/episode split counts")
    split.add_argument("--split-seed", default="robot-demo-v1")
    split.set_defaults(handler=_split_command)

    plan = commands.add_parser("plan", help="assign 3 supports and 100 paired rollouts per eligible target")
    _add_registry_argument(plan)
    plan.add_argument("--output", required=True, type=_path)
    plan.add_argument("--a-checkpoint", required=True)
    plan.add_argument("--modified-checkpoint", required=True)
    plan.add_argument("--evaluation-split", default="test")
    plan.add_argument("--rollout-seeds", type=_integer_seeds, default=_integer_seeds(DEFAULT_ROLLOUT_SEEDS))
    plan.add_argument("--training-seed", type=int, required=True, help="paired A/modified training run seed")
    plan.add_argument("--clean-task-config", required=True, type=_path)
    plan.add_argument("--random-task-config", required=True, type=_path)
    plan.set_defaults(handler=_plan_command)

    aggregate = commands.add_parser("aggregate", help="report recorded Clean/Random success and paired uncertainty")
    aggregate.add_argument("--plan", required=True, type=_path)
    aggregate.add_argument("--results", required=True, type=_path, help="JSONL observed rollout records")
    aggregate.add_argument("--output", required=True, type=_path)
    aggregate.add_argument("--bootstrap-samples", type=int, default=2000)
    aggregate.set_defaults(handler=_aggregate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and present registry errors as ordinary CLI failures."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        arguments.handler(arguments)
    except ExperimentInputError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
