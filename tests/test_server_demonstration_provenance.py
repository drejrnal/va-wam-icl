from pathlib import Path
from types import SimpleNamespace

import torch

import wan_va.wan_va_server as server_module
from wan_va.dataset.demonstration import DemoTensors


def test_prepare_demonstration_attests_the_loaded_tensor_snapshot(monkeypatch):
    prepared_path = Path("prepared/demo.pt")
    record = SimpleNamespace(success=True, prepared_tensor_path=prepared_path)
    registry = SimpleNamespace(
        resolve_demo=lambda demo_id, **kwargs: record,
    )
    tensors = DemoTensors(
        demo_latents=torch.zeros((48, 17, 2, 3)),
        demo_positions=torch.arange(17),
        demo_mask=torch.ones(17, dtype=torch.bool),
    )
    snapshot_digest = "a" * 64
    provenance = {
        "demo_id": "support-1",
        "prepared_tensor_sha256": snapshot_digest,
    }
    observed = {}

    monkeypatch.setattr(
        server_module.DemonstrationRegistry,
        "from_file",
        lambda path, required_cameras: registry,
    )

    def load_snapshot(path, max_frames):
        observed["load"] = (path, max_frames)
        return tensors, snapshot_digest

    def build_provenance(actual_registry, actual_record, prepared_tensor_sha256):
        observed["provenance"] = (
            actual_registry,
            actual_record,
            prepared_tensor_sha256,
        )
        return provenance

    def prepare_cache(transformer, payload, **kwargs):
        observed["cache"] = (transformer, payload, kwargs)

    monkeypatch.setattr(
        server_module,
        "load_prepared_demonstration_with_digest",
        load_snapshot,
    )
    monkeypatch.setattr(
        server_module,
        "build_demonstration_provenance",
        build_provenance,
    )
    monkeypatch.setattr(
        server_module,
        "prepare_demonstration_cache",
        prepare_cache,
    )

    server = server_module.VA_Server.__new__(server_module.VA_Server)
    server.transformer = SimpleNamespace(
        config=SimpleNamespace(enable_demo_conditioning=True)
    )
    server.job_config = SimpleNamespace(
        demonstration_manifest_path="manifest.json",
        obs_cam_keys=("front",),
        demonstration_split="test",
        demonstration_embodiment="robot",
    )
    server.latent_height = 2
    server.latent_width = 3
    server.cache_name = "episode"
    server.device = torch.device("cpu")
    server.dtype = torch.float32

    actual = server._prepare_demonstration(
        "support-1",
        expected_demo_provenance=provenance,
    )

    assert actual is provenance
    assert observed["load"] == (prepared_path, 17)
    assert observed["provenance"] == (registry, record, snapshot_digest)
    _, cached_payload, cache_kwargs = observed["cache"]
    assert cached_payload.demo_latents is tensors.demo_latents
    assert cached_payload.demo_positions is tensors.demo_positions
    assert cached_payload.demo_mask is tensors.demo_mask
    assert cache_kwargs["cache_name"] == "episode"
