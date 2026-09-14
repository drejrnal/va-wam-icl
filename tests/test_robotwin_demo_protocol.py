from __future__ import annotations

import pytest

import evaluation.robotwin.websocket_client_policy as websocket_client
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy


class RecordingPacker:
    def __init__(self) -> None:
        self.payload: dict | None = None

    def pack(self, payload: dict) -> bytes:
        self.payload = payload
        return b"request"


class RecordingSocket:
    def __init__(self, response: bytes = b"\x80") -> None:
        self.sent: bytes | None = None
        self.response = response

    def send(self, data: bytes) -> None:
        self.sent = data

    def recv(self) -> bytes:
        return self.response


def _client(response: bytes = b"\x80") -> tuple[WebsocketClientPolicy, RecordingPacker]:
    client = WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    packer = RecordingPacker()
    client._packer = packer
    client._ws = RecordingSocket(response)
    return client, packer


def test_reset_when_demo_is_selected_sends_optional_fields() -> None:
    # Given
    client, packer = _client(b"\x81\xaacheckpoint\xb0checkpoints/demo")

    # When
    client.reset(
        prompt="pick up cup",
        demo_id="support-17",
        demo_shuffle_seed=9,
        expected_checkpoint="checkpoints/demo",
    )

    # Then
    assert packer.payload == {
        "reset": True,
        "prompt": "pick up cup",
        "demo_id": "support-17",
        "demo_shuffle_seed": 9,
        "expected_checkpoint": "checkpoints/demo",
    }


def test_reset_when_demo_is_absent_preserves_legacy_payload() -> None:
    # Given
    client, packer = _client()

    # When
    client.reset()

    # Then
    assert packer.payload == {"reset": True}


def test_reset_when_server_checkpoint_differs_rejects_rollout() -> None:
    # Given
    client, _ = _client()

    # When / Then
    with pytest.raises(RuntimeError, match="unexpected checkpoint"):
        client.reset(expected_checkpoint="checkpoints/demo")


def test_reset_when_provenance_is_planned_returns_server_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    expected = {
        "registry_sha256": "registry",
        "demo_id": "support-17",
        "repo_id": "robotwin",
        "episode_id": "episode-17",
        "episode_index": 17,
        "task": "pick-cup",
        "family": "pick",
        "goal": "pick up cup",
        "embodiment": "bimanual",
        "split": "test",
        "source_seed": 117,
        "prepared_tensor_sha256": "payload",
    }
    client, packer = _client(b"attested")
    monkeypatch.setattr(
        websocket_client,
        "unpackb",
        lambda _: {
            "checkpoint": "checkpoints/demo",
            "demo_provenance": expected,
        },
    )

    # When
    response = client.reset(
        demo_id="support-17",
        expected_checkpoint="checkpoints/demo",
        expected_demo_provenance=expected,
    )

    # Then
    assert packer.payload["expected_demo_provenance"] == expected
    assert response["demo_provenance"] == expected


def test_reset_when_no_demo_is_planned_rejects_server_demo_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    client, _ = _client(b"unexpected-demo")
    monkeypatch.setattr(
        websocket_client,
        "unpackb",
        lambda _: {"checkpoint": "checkpoints/demo", "demo_provenance": {}},
    )

    # When / Then
    with pytest.raises(RuntimeError, match="unexpected demo provenance"):
        client.reset()
