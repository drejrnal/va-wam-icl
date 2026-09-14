import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from wan_va.dataset.demonstration import (
    DemonstrationManifestError,
    DemonstrationRegistry,
    build_demonstration_provenance,
    load_prepared_demonstration,
    load_prepared_demonstration_with_digest,
    pairable_episode_indices,
    prepare_demo_tensors,
    retain_nonempty_datasets,
)


CAMERAS = {
    "observation.images.cam_high": "videos/high.mp4",
    "observation.images.cam_left_wrist": "videos/left.mp4",
    "observation.images.cam_right_wrist": "videos/right.mp4",
}


def _episode(demo_id: str, episode_index: int, split: str = "train") -> dict:
    return {
        "demo_id": demo_id,
        "repo_id": "repo",
        "episode_id": f"episode-{episode_index}",
        "episode_index": episode_index,
        "task": "pick-cup",
        "family": "pick",
        "goal": "cup-on-tray",
        "embodiment": "robotwin-dual-arm",
        "success": True,
        "split": split,
        "cameras": CAMERAS,
        "prepared_tensor_path": f"prepared/{demo_id}.pt",
        "source_seed": episode_index + 100,
    }


def _write_manifest(path: Path, episodes: list[dict]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "episodes": episodes}),
        encoding="utf-8",
    )


def test_registry_selects_distinct_compatible_reference_without_split_leakage(
    tmp_path: Path,
) -> None:
    # Given a target, compatible train reference, and compatible validation episode.
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [_episode("target", 0), _episode("reference", 1)],
    )
    registry = DemonstrationRegistry.from_file(manifest_path, required_cameras=CAMERAS)

    # When selecting a reference for the target.
    reference = registry.select_reference(registry.resolve_demo("target"))

    # Then it is distinct, compatible, successful, and in the target split.
    assert reference.demo_id == "reference"
    assert reference.split == "train"


@pytest.mark.parametrize(
    ("episodes", "message"),
    [
        ([_episode("same", 0), _episode("same", 1)], "duplicate demo_id"),
        ([_episode("only", 0)], "compatible reference"),
    ],
)
def test_registry_rejects_duplicate_ids_and_self_only_pairing(
    tmp_path: Path, episodes: list[dict], message: str
) -> None:
    # Given malformed or unpairable manifest input.
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, episodes)

    # When parsing or selecting from the registry, then the invalid state is rejected.
    with pytest.raises(DemonstrationManifestError, match=message):
        registry = DemonstrationRegistry.from_file(
            manifest_path, required_cameras=CAMERAS
        )
        registry.select_reference(registry.resolve_demo(episodes[0]["demo_id"]))


def test_registry_rejects_unknown_fields_missing_cameras_and_cross_split_resolution(
    tmp_path: Path,
) -> None:
    # Given three independently invalid boundary conditions.
    unknown = _episode("unknown", 0)
    unknown["invented"] = "field"
    missing_camera = _episode("missing", 1)
    missing_camera["cameras"] = {"observation.images.cam_high": "high.mp4"}

    for name, episode, message in [
        ("unknown", unknown, "unknown fields"),
        ("camera", missing_camera, "camera"),
    ]:
        manifest_path = tmp_path / f"{name}.json"
        _write_manifest(manifest_path, [episode])
        with pytest.raises(DemonstrationManifestError, match=message):
            DemonstrationRegistry.from_file(manifest_path, required_cameras=CAMERAS)

    manifest_path = tmp_path / "valid.json"
    _write_manifest(manifest_path, [_episode("valid", 0)])
    registry = DemonstrationRegistry.from_file(manifest_path, required_cameras=CAMERAS)
    with pytest.raises(DemonstrationManifestError, match="split"):
        registry.resolve_demo("valid", expected_split="test")


def test_registry_rejects_family_split_leakage(tmp_path: Path) -> None:
    # Given one family assigned to two splits.
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path, [_episode("train", 0), _episode("test", 1, "test")]
    )

    # When parsing the manifest, then global family leakage is rejected.
    with pytest.raises(DemonstrationManifestError, match="multiple splits"):
        DemonstrationRegistry.from_file(manifest_path, required_cameras=CAMERAS)


def test_registry_rejects_duplicate_physical_episode_alias(tmp_path: Path) -> None:
    # Given two selectors that point to one physical repository episode.
    duplicate = _episode("duplicate", 0)
    duplicate["episode_id"] = "different-label"
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_episode("original", 0), duplicate])

    # When parsing the aliases, then self-pairing by alternate ID is rejected.
    with pytest.raises(DemonstrationManifestError, match="episode_index"):
        DemonstrationRegistry.from_file(manifest_path, required_cameras=CAMERAS)


def test_provenance_binds_registry_record_and_prepared_payload(tmp_path: Path) -> None:
    # Given a registry and its exact prepared tensor bytes.
    manifest_path = tmp_path / "manifest.json"
    episodes = [_episode("selected", 0), _episode("reference", 1)]
    _write_manifest(manifest_path, episodes)
    prepared = tmp_path / "prepared" / "selected.pt"
    prepared.parent.mkdir()
    first_payload = prepare_demo_tensors(torch.ones(2, 3, 2, 2))
    torch.save(first_payload.as_payload(), prepared)
    registry = DemonstrationRegistry.from_file(manifest_path, required_cameras=CAMERAS)

    # When bytes are loaded and hashed once, before the path changes.
    first_tensors, first_digest = load_prepared_demonstration_with_digest(prepared)
    second_payload = prepare_demo_tensors(torch.zeros(2, 3, 2, 2))
    torch.save(second_payload.as_payload(), prepared)
    _, second_digest = load_prepared_demonstration_with_digest(prepared)
    first = build_demonstration_provenance(
        registry, registry.resolve_demo("selected"), first_digest
    )

    # Then provenance stays bound to the installed snapshot, not the changed path.
    assert first["prepared_tensor_sha256"] == first_digest
    assert first_digest != second_digest
    assert torch.equal(first_tensors.demo_latents, first_payload.demo_latents)
    assert first["repo_id"] == "repo"
    assert first["episode_id"] == "episode-0"


def test_pairable_indices_support_matched_baseline_filtering(tmp_path: Path) -> None:
    # Given a registry containing two pairable train episodes.
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_episode("one", 0), _episode("two", 1)])
    registry = DemonstrationRegistry.from_file(manifest_path, required_cameras=CAMERAS)

    # When selecting the dataset split independently of model conditioning.
    targets = pairable_episode_indices(registry, tmp_path / "repo", "train")

    # Then both matched target episodes remain available.
    assert set(targets) == {0, 1}


def test_external_manifest_repo_path_matches_discovered_dataset_root(
    tmp_path: Path,
) -> None:
    # Given a manifest outside the dataset tree with an explicit relative repo path.
    manifest_dir = tmp_path / "manifests"
    dataset_root = tmp_path / "datasets" / "robotwin" / "task"
    manifest_dir.mkdir()
    dataset_root.mkdir(parents=True)
    episodes = [_episode("one", 0), _episode("two", 1)]
    for episode in episodes:
        episode["repo_id"] = "../datasets/robotwin/task"
    manifest_path = manifest_dir / "episodes.json"
    _write_manifest(manifest_path, episodes)
    registry = DemonstrationRegistry.from_file(manifest_path, required_cameras=CAMERAS)

    # When target episodes are matched to the discovered absolute repository root.
    targets = pairable_episode_indices(registry, dataset_root, "train")

    # Then both manifest episodes select that repository without an opaque ID map.
    assert set(targets) == {0, 1}


def test_pairable_indices_exclude_failed_targets(tmp_path: Path) -> None:
    # Given a failed target and two successful compatible episodes.
    failed = _episode("failed", 0)
    failed["success"] = False
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path, [failed, _episode("one", 1), _episode("two", 2)]
    )
    registry = DemonstrationRegistry.from_file(manifest_path, required_cameras=CAMERAS)

    # When selecting pairable targets, then the failed episode is excluded.
    targets = pairable_episode_indices(registry, tmp_path / "repo", "train")
    assert set(targets) == {1, 2}


@pytest.mark.parametrize("field", ["episode_index", "source_seed"])
def test_registry_rejects_negative_source_identity(
    tmp_path: Path, field: str
) -> None:
    # Given a manifest with a negative physical identity component.
    episode = _episode("negative", 0)
    episode[field] = -1
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [episode])

    # When parsing the registry, then the invalid identity is rejected.
    with pytest.raises(DemonstrationManifestError, match="non-negative"):
        DemonstrationRegistry.from_file(manifest_path, required_cameras=CAMERAS)


def test_multi_repo_split_keeps_train_repo_and_skips_heldout_repo() -> None:
    # Given one selected repository and one held-out repository with no samples.
    train_repo = ["sample"]
    heldout_repo = []

    # When empty per-repository views are removed after split filtering.
    selected = retain_nonempty_datasets([train_repo, heldout_repo])

    # Then training continues with the selected repository only.
    assert selected == [train_repo]


def test_prepare_and_load_uniformly_samples_endpoints_and_pads(tmp_path: Path) -> None:
    # Given a full normalized latent video longer than the maximum.
    full = torch.arange(2 * 20 * 2 * 3, dtype=torch.float32).reshape(2, 20, 2, 3)

    # When preparing it for demonstration conditioning.
    prepared = prepare_demo_tensors(full, max_frames=17)

    # Then endpoints are retained with normalized original-time positions.
    assert prepared.demo_latents.shape == (2, 17, 2, 3)
    assert prepared.demo_mask.tolist() == [True] * 17
    assert prepared.demo_positions[0].item() == 0.0
    assert prepared.demo_positions[-1].item() == 1.0
    assert torch.equal(prepared.demo_latents[:, 0], full[:, 0])
    assert torch.equal(prepared.demo_latents[:, -1], full[:, -1])

    short = prepare_demo_tensors(full[:, :3], max_frames=17)
    payload_path = tmp_path / "short.pt"
    torch.save(short.as_payload(), payload_path)
    loaded = load_prepared_demonstration(payload_path, max_frames=17)
    assert loaded.demo_mask.tolist() == [True, True, True] + [False] * 14
    assert not loaded.demo_latents[:, 3:].any()


def test_preparation_cli_combines_three_encoded_cameras_without_actions(
    tmp_path: Path,
) -> None:
    # Given synchronized normalized latent tensors for three RobotWin cameras.
    high = torch.ones(2, 5, 4, 6)
    left = torch.full((2, 5, 2, 3), 2.0)
    right = torch.full((2, 5, 2, 3), 3.0)
    paths = {}
    for name, tensor in (("high", high), ("left", left), ("right", right)):
        path = tmp_path / f"{name}.pt"
        torch.save(tensor, path)
        paths[name] = path
    output = tmp_path / "prepared.pt"

    # When the real CLI prepares the payload.
    result = subprocess.run(
        [
            sys.executable,
            "script/prepare_robot_demonstrations.py",
            "--camera-latent",
            f"observation.images.cam_high={paths['high']}",
            "--camera-latent",
            f"observation.images.cam_left_wrist={paths['left']}",
            "--camera-latent",
            f"observation.images.cam_right_wrist={paths['right']}",
            "--layout",
            "robotwin_tshape",
            "--camera-latents-normalized",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then it emits only the tensor payload and preserves the camera layout.
    assert result.returncode == 0, result.stderr
    payload = torch.load(output, weights_only=True)
    assert set(payload) == {"demo_latents", "demo_positions", "demo_mask"}
    assert payload["demo_latents"].shape == (2, 17, 6, 6)
    assert payload["demo_mask"].sum().item() == 5


def test_preparation_cli_rejects_unconfirmed_latent_normalization(
    tmp_path: Path,
) -> None:
    # Given three valid encoded camera tensors without normalization confirmation.
    paths = {}
    for name in ("high", "left", "right"):
        path = tmp_path / f"{name}.pt"
        torch.save(torch.ones(2, 5, 2, 2), path)
        paths[name] = path

    # When the CLI is invoked without --camera-latents-normalized.
    result = subprocess.run(
        [
            sys.executable,
            "script/prepare_robot_demonstrations.py",
            "--camera-latent",
            f"observation.images.cam_high={paths['high']}",
            "--camera-latent",
            f"observation.images.cam_left_wrist={paths['left']}",
            "--camera-latent",
            f"observation.images.cam_right_wrist={paths['right']}",
            "--layout",
            "robotwin_tshape",
            "--output",
            str(tmp_path / "prepared.pt"),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then the boundary rejects ambiguous pre-encoded latent normalization.
    assert result.returncode != 0
    assert "--camera-latents-normalized is required" in result.stderr
