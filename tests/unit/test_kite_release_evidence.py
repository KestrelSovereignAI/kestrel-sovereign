"""Safety and content-boundary tests for the isolated Kite evidence harness."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from kestrel_sovereign.knowledge.kite_release_evidence import (
    ErasureSurfaceProbe,
    KiteAggregateObservation,
    KiteEvidenceError,
    KiteGate,
    KiteHttpHarness,
    KiteIsolationConfig,
    SurfaceErasureObservation,
    _KITE_EVIDENCE_CONTRACT,
    _KITE_EXPERIMENTAL_CAPABILITIES,
)
from kestrel_sovereign.knowledge import kite_evidence_signing
from kestrel_sovereign.knowledge.kite_evidence_signing import (
    KiteEvidenceNonceReplay,
    KiteEvidenceSigningError,
    consume_kite_evidence_nonce,
    kite_evidence_public_key,
)
from kestrel_sovereign.knowledge.capabilities import (
    SemanticRuntimeCapabilities,
    semantic_capabilities_from_config,
)
from kestrel_sovereign.knowledge.inference import inference_profile_from_config
from kestrel_sovereign.knowledge.release_evidence_models import _canonical_json


_TEST_SIGNING_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
_TEST_SIGNING_PUBLIC_KEY = _TEST_SIGNING_PRIVATE_KEY.public_key().public_bytes(
    serialization.Encoding.Raw,
    serialization.PublicFormat.Raw,
).hex()
from kestrel_sovereign.knowledge.release_evidence import release_gate_specs
from kestrel_sovereign.knowledge.release_evidence_models import ErasureStage


class _Reply:
    def __init__(self, body: dict[str, object]) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


class _Process:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            raise subprocess.TimeoutExpired("kite", timeout)
        return self._returncode


def _worktree(tmp_path: Path) -> Path:
    worktree = tmp_path / "source"
    (worktree / ".git").mkdir(parents=True)
    binary_dir = worktree / ".venv" / "bin"
    binary_dir.mkdir(parents=True)
    for name in ("python", "kestrel"):
        path = binary_dir / name
        path.write_text("#!/bin/sh\n", encoding="utf-8")
    return worktree


def _config(tmp_path: Path, gate: KiteGate = KiteGate.STABLE_ONLY) -> KiteIsolationConfig:
    return KiteIsolationConfig(
        worktree=_worktree(tmp_path),
        home=tmp_path / "isolated-home",
        port=48991,
        gate=gate,
    )


def _fake_create(cmd, *, cwd, env, check, text, capture_output):
    home = Path(env["KESTREL_HOME"])
    (home / "agent_data" / "kite").mkdir(parents=True)
    (home / "multi_agent.toml").write_text(
        "[host]\nport = 8888\nbind = '0.0.0.0'\n\n[agents.kite]\ndata_dir = 'agent_data/kite'\nport = 48991\nautostart = true\n",
        encoding="utf-8",
    )
    return subprocess.CompletedProcess(cmd, 0, "", "")


def test_prepare_creates_fresh_disjoint_test_home_and_experimental_profile(tmp_path: Path) -> None:
    config = _config(tmp_path, KiteGate.EXPERIMENTAL_ENABLED)
    harness = KiteHttpHarness(config, run=_fake_create)

    harness._assert_port_unused = lambda: None  # type: ignore[method-assign]
    harness.prepare()

    text = config.config_path.read_text(encoding="utf-8")
    assert "mode = \"experimental\"" in text
    assert config.marker_path.is_file()
    with pytest.raises(KiteEvidenceError, match="fresh"):
        harness.prepare()


def test_child_environment_is_minimal_and_overrides_production_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    """A Kite child must never inherit DB, provider, credential, or trust pins."""
    production_values = {
        "KESTREL_DB_BACKEND": "postgres",
        "KESTREL_DATABASE_URL": "postgresql://prod.example/private",
        "DATABASE_URL": "postgresql://legacy.example/private",
        "KESTREL_DB_PATH": "/production/kestrel.db",
        "OPENAI_API_KEY": "production-openai-key",
        "ANTHROPIC_API_KEY": "production-anthropic-key",
        "KESTREL_LLM_MODEL": "production/model",
        "KESTREL_API_KEY": "production-api-key",
        "KESTREL_TRUST_POLICY": "/production/trust-policy.json",
        "SSL_CERT_FILE": "/production/trust.pem",
    }
    for key, value in production_values.items():
        monkeypatch.setenv(key, value)
    config = _config(tmp_path)

    child = config.environment()

    for key, value in production_values.items():
        assert child.get(key) != value
    assert child["KESTREL_DB_BACKEND"] == "sqlite"
    assert child["KESTREL_DB_PATH"] == str(config.home / "kite-evidence.sqlite3")
    assert child["KESTREL_DATABASE_URL"] == (
        f"sqlite:///{config.home / 'kite-evidence.sqlite3'}"
    )
    assert child["DATABASE_URL"] == child["KESTREL_DATABASE_URL"]
    assert child["KESTREL_KITE_RELEASE_EVIDENCE_ROOT"] == str(config.home)
    assert "KESTREL_API_KEY" not in child
    assert "KESTREL_TRUST_POLICY" not in child
    assert "SSL_CERT_FILE" not in child


def test_isolation_rejects_worktree_home_overlap_and_non_loopback(tmp_path: Path) -> None:
    worktree = _worktree(tmp_path)
    with pytest.raises(KiteEvidenceError, match="disjoint"):
        KiteIsolationConfig(worktree=worktree, home=worktree, port=48991, gate=KiteGate.STABLE_ONLY)
    with pytest.raises(KiteEvidenceError, match="127.0.0.1"):
        KiteIsolationConfig(worktree=worktree, home=tmp_path / "home", port=48991, gate=KiteGate.STABLE_ONLY, host="0.0.0.0")


def test_start_invokes_only_worktree_python_and_documented_http_endpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    spawned: list[_Process] = []
    requests: list[object] = []

    def popen(command, **kwargs):
        assert command[:3] == [str(config.python_path), "-m", "uvicorn"]
        assert command[-4:] == ["--host", "127.0.0.1", "--port", str(config.port)]
        assert kwargs["env"]["KESTREL_HOME"] == str(config.home)
        process = _Process(101 + len(spawned))
        spawned.append(process)
        return process

    def opener(request, **_kwargs):
        requests.append(request)
        if isinstance(request, str):
            return _Reply(
                {
                    "key": "test-key",
                    "kite_evidence_public_key": _TEST_SIGNING_PUBLIC_KEY,
                }
            )
        assert request.full_url == f"http://127.0.0.1:{config.port}/api/agents/kite/api/agent/invoke"
        assert request.get_header("X-api-key") == "test-key"
        return _Reply({"response": "ok"})

    harness = KiteHttpHarness(config, run=_fake_create, popen=popen, opener=opener)
    harness._assert_port_unused = lambda: None  # type: ignore[method-assign]
    harness.prepare()
    harness.start()
    assert harness.invoke("harmless") == {"response": "ok"}
    assert harness._evidence_public_key == _TEST_SIGNING_PUBLIC_KEY
    harness.stop()

    assert spawned[0].terminated is True
    assert len(requests) == 2  # one bootstrap fetch, one invoke; never refetch the key


def test_stop_cannot_target_an_unowned_listener(tmp_path: Path) -> None:
    harness = KiteHttpHarness(_config(tmp_path), run=_fake_create)
    harness.stop()
    with pytest.raises(KiteEvidenceError, match="cannot restart"):
        harness.restart()


def test_occupied_port_is_rejected_without_pid_discovery(tmp_path: Path) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        config = KiteIsolationConfig(
            worktree=_worktree(tmp_path),
            home=tmp_path / "home",
            port=sock.getsockname()[1],
            gate=KiteGate.STABLE_ONLY,
        )
        with pytest.raises(KiteEvidenceError, match="occupied"):
            KiteHttpHarness(config)._assert_port_unused()
    finally:
        sock.close()


@pytest.mark.parametrize("gate", tuple(KiteGate))
def test_each_kite_aggregate_matches_its_exact_catalog_schema(gate: KiteGate) -> None:
    """A future registered runner can hand every aggregate straight to its spec."""
    observation = KiteAggregateObservation(
        gate_id=gate.value,
        invoke_count=5,
        scenario_count=5,
        provenance_check_count=1,
        experimental_selection_count=1,
        persisted_assertion_count=1,
        canonical_migration_count=0,
    )
    spec = next(item for item in release_gate_specs() if item.gate_id == gate.value)

    spec.observation_schema.validate(observation.to_mapping())


def _diagnostics(
    gate: KiteGate,
    *,
    provenance: int = 1,
    migrations: int = 0,
) -> dict[str, object]:
    capabilities = (
        semantic_capabilities_from_config(_KITE_EXPERIMENTAL_CAPABILITIES)
        if gate is KiteGate.EXPERIMENTAL_ENABLED
        else SemanticRuntimeCapabilities.stable()
    )
    return {
        "capability_mode": capabilities.mode,
        "active_capability_pins": [
            f"{key}={value}"
            for key, value in sorted(capabilities.capability_versions().items())
        ],
        "canonical_migration_count": migrations,
        "explicit_fact_provenance_present_count": provenance,
    }


def _typed_response(
    request: dict[str, object],
    observation: dict[str, object],
    *,
    signing_key: Ed25519PrivateKey = _TEST_SIGNING_PRIVATE_KEY,
    assistant_response: str = "untrusted assistant narration",
) -> dict[str, object]:
    signed = {
        "contract": _KITE_EVIDENCE_CONTRACT,
        "nonce": request["nonce"],
        "operation": request["operation"],
        "observation": observation,
    }
    signature = signing_key.sign(_canonical_json(signed).encode("utf-8")).hex()
    return {
        "response": assistant_response,
        "kite_evidence": {**signed, "signature": signature},
    }


def test_stable_live_gate_requires_content_free_http_diagnostics(tmp_path: Path, monkeypatch) -> None:
    harness = KiteHttpHarness(_config(tmp_path), run=_fake_create)
    harness._process = _Process(100)  # type: ignore[assignment]
    harness._api_key = "test-key"
    harness._evidence_public_key = _TEST_SIGNING_PUBLIC_KEY
    markers = iter(("kite-evidence-one", "kite-evidence-two"))
    monkeypatch.setattr(KiteHttpHarness, "_marker", staticmethod(lambda: next(markers)))
    operations: list[str] = []

    def invoke(_self, prompt: str, **_kwargs: Any):
        assert prompt == ""
        request = _kwargs["kite_evidence"]
        assert isinstance(request, dict)
        operation = request["operation"]
        operations.append(operation)
        if operation == "diagnostics":
            observation = _diagnostics(KiteGate.STABLE_ONLY)
        elif operation == "paraphrase_recall":
            observation = {"retrieval_count": 1, "provenance_check_count": 1}
        elif operation == "quarantine":
            observation = {"invalid_import_quarantine_count": 1}
        elif operation == "sleep":
            observation = {"sleep_success_count": 1}
        elif operation == "save":
            observation = {"fact_write_count": 1}
        else:
            raise AssertionError(operation)
        return _typed_response(request, observation)

    monkeypatch.setattr(KiteHttpHarness, "invoke", invoke)
    observation = harness.run_release_gate()
    assert observation.provenance_check_count == 1
    assert observation.canonical_migration_count == 0
    assert operations == ["save", "diagnostics", "save", "paraphrase_recall", "quarantine", "sleep"]


def test_experimental_live_gate_requires_exact_draft_pins(tmp_path: Path, monkeypatch) -> None:
    harness = KiteHttpHarness(_config(tmp_path, KiteGate.EXPERIMENTAL_ENABLED), run=_fake_create)
    harness._process = _Process(100)  # type: ignore[assignment]
    harness._api_key = "test-key"
    harness._evidence_public_key = _TEST_SIGNING_PUBLIC_KEY
    markers = iter(("kite-evidence-one", "kite-evidence-two"))
    monkeypatch.setattr(KiteHttpHarness, "_marker", staticmethod(lambda: next(markers)))

    def invoke(_self, prompt: str, **_kwargs: Any):
        assert prompt == ""
        request = _kwargs["kite_evidence"]
        assert isinstance(request, dict)
        operation = request["operation"]
        if operation == "diagnostics":
            observation = _diagnostics(KiteGate.EXPERIMENTAL_ENABLED)
        elif operation == "paraphrase_recall":
            observation = {"retrieval_count": 1, "provenance_check_count": 1}
        elif operation == "quarantine":
            observation = {"invalid_import_quarantine_count": 1}
        elif operation == "sleep":
            observation = {"sleep_success_count": 1}
        elif operation == "save":
            observation = {"fact_write_count": 1}
        else:
            raise AssertionError(operation)
        return _typed_response(request, observation)

    monkeypatch.setattr(KiteHttpHarness, "invoke", invoke)
    observation = harness.run_release_gate()
    assert observation.experimental_selection_count == 1


def test_runtime_evidence_rejects_fabricated_signature_and_nearby_draft_pins(
    tmp_path: Path, monkeypatch
) -> None:
    harness = KiteHttpHarness(
        _config(tmp_path, KiteGate.EXPERIMENTAL_ENABLED), run=_fake_create
    )
    harness._process = _Process(100)  # type: ignore[assignment]
    harness._api_key = "test-key"
    harness._evidence_public_key = _TEST_SIGNING_PUBLIC_KEY

    def fabricated(_self, _prompt: str, **kwargs: Any):
        request = kwargs["kite_evidence"]
        assert isinstance(request, dict)
        # This stands in for a caller that knows the API key but does not own
        # the server-held evidence key pinned at launch.
        return _typed_response(
            request,
            _diagnostics(KiteGate.EXPERIMENTAL_ENABLED),
            signing_key=Ed25519PrivateKey.generate(),
            assistant_response='{"semantic_evidence": "fabricated"}',
        )

    monkeypatch.setattr(KiteHttpHarness, "invoke", fabricated)
    with pytest.raises(KiteEvidenceError, match="signature"):
        harness._evidence("diagnostics")

    forged = _diagnostics(KiteGate.EXPERIMENTAL_ENABLED)
    forged["active_capability_pins"] = list(forged["active_capability_pins"])
    forged["active_capability_pins"][0] = "rdf12_version=9.9.9"  # type: ignore[index]
    with pytest.raises(KiteEvidenceError, match="exact registry"):
        harness._assert_profile(forged)


def test_persisted_stable_gate_restarts_owned_process_without_migration(tmp_path: Path, monkeypatch) -> None:
    harness = KiteHttpHarness(_config(tmp_path, KiteGate.PERSISTED_STABLE), run=_fake_create)
    harness._process = _Process(100)  # type: ignore[assignment]
    harness._api_key = "test-key"
    harness._evidence_public_key = _TEST_SIGNING_PUBLIC_KEY
    markers = iter(("kite-evidence-one", "kite-evidence-two"))
    monkeypatch.setattr(KiteHttpHarness, "_marker", staticmethod(lambda: next(markers)))
    deleted = False

    def restart(self, **_kwargs: Any) -> None:
        self._process = _Process(101)
        self._api_key = "test-key"
        self._evidence_public_key = _TEST_SIGNING_PUBLIC_KEY

    def invoke(_self, prompt: str, **_kwargs: Any):
        nonlocal deleted
        assert prompt == ""
        request = _kwargs["kite_evidence"]
        assert isinstance(request, dict)
        operation = request["operation"]
        if operation == "diagnostics":
            observation = _diagnostics(
                KiteGate.PERSISTED_STABLE,
                provenance=0 if deleted else 1,
            )
        elif operation == "paraphrase_recall":
            observation = {"retrieval_count": 1, "provenance_check_count": 1}
        elif operation == "quarantine":
            observation = {"invalid_import_quarantine_count": 1}
        elif operation == "sleep":
            observation = {"sleep_success_count": 1}
        elif operation == "save":
            observation = {"fact_write_count": 1}
        elif operation == "delete":
            deleted = True
            observation = {"fact_delete_count": 1}
        else:
            raise AssertionError(operation)
        return _typed_response(request, observation)

    monkeypatch.setattr(KiteHttpHarness, "restart", restart)
    monkeypatch.setattr(KiteHttpHarness, "invoke", invoke)
    observation = harness.run_release_gate()
    assert observation.persisted_assertion_count == 1
    assert observation.canonical_migration_count == 0


def test_content_free_artifact_excludes_transcript_marker_and_response(tmp_path: Path) -> None:
    config = _config(tmp_path)
    harness = KiteHttpHarness(config, run=_fake_create)
    artifact = config.home / "artifacts" / "result.json"
    digest = harness.write_content_free_artifact(
        KiteAggregateObservation(
            gate_id=KiteGate.STABLE_ONLY,
            invoke_count=7,
            scenario_count=6,
            provenance_check_count=1,
        ),
        artifact,
    )
    text = artifact.read_text(encoding="utf-8")
    assert digest in text
    assert "kite-evidence-" not in text
    assert "response" not in text
    assert "prompt" not in text
    with pytest.raises(KiteEvidenceError, match="inside"):
        harness.write_content_free_artifact(
            KiteAggregateObservation(
                gate_id=KiteGate.STABLE_ONLY,
                invoke_count=1,
                scenario_count=1,
                provenance_check_count=1,
            ),
            tmp_path / "outside.json",
        )


class _AllSurfaces(ErasureSurfaceProbe):
    def __init__(self, drill) -> None:
        self.drill = drill

    def observe(self, drill):
        assert drill == self.drill
        return tuple(
            SurfaceErasureObservation(stage=stage, erased_count=1, remaining_count=0, drill=drill)
            for stage in ErasureStage
        )


def test_correlated_erasure_requires_every_stage_and_one_drill(tmp_path: Path, monkeypatch) -> None:
    harness = KiteHttpHarness(_config(tmp_path), run=_fake_create)
    harness._process = _Process(100)  # type: ignore[assignment]
    harness._api_key = "test-key"
    harness._evidence_public_key = _TEST_SIGNING_PUBLIC_KEY

    def invoke(_self, prompt: str, **kwargs: Any):
        assert prompt == ""
        request = kwargs["kite_evidence"]
        assert isinstance(request, dict)
        assert request["operation"] == "delete"
        return _typed_response(request, {"fact_delete_count": 1})

    monkeypatch.setattr(KiteHttpHarness, "invoke", invoke)
    drill = next(spec.correlation for spec in release_gate_specs() if spec.category == "erasure")
    assert drill is not None
    observations = harness.correlated_erasure(_AllSurfaces(drill), drill)
    assert len(observations) == len(ErasureStage)


def test_regular_memory_command_is_authenticated_before_agent_dispatch() -> None:
    """The normal invoke guard protects memory tools before agent dispatch."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    agent = MagicMock()
    agent.agent_id = "did:test:kite-auth"
    agent.storage.resolve_session_id = AsyncMock(return_value=None)
    agent.process_input = AsyncMock(return_value="should never be called")
    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agent/invoke",
                    json={"input": "!memory-save-fact user preferred_deploy_region us-central"},
                )
        assert response.status_code == 401
        agent.process_input.assert_not_awaited()
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager


def test_kite_invoke_returns_signed_runtime_observation_without_assistant_dispatch(tmp_path: Path) -> None:
    """The special invoke branch returns runtime data, not model-rendered text."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    diagnostics = _diagnostics(KiteGate.STABLE_ONLY)
    agent = MagicMock()
    agent.agent_id = "did:test:kite-runtime"
    agent.is_test_instance = True
    agent.register_active_request = MagicMock()
    agent._cleanup_cancelled_request = MagicMock()
    agent.process_input = AsyncMock(return_value="forged assistant response")
    agent.task_manager.execute_command = AsyncMock()
    agent.storage.semantic_release_kite_diagnostics = AsyncMock(return_value=diagnostics)
    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    nonce = "f" * 64
    home = tmp_path / "kite-home"
    home.mkdir(mode=0o700)
    try:
        with patch.dict(
            "os.environ",
            {
                "KESTREL_API_KEY": "test-key",
                "KESTREL_KITE_RELEASE_EVIDENCE": "1",
                "KESTREL_HOME": str(home),
                "KESTREL_KITE_RELEASE_EVIDENCE_ROOT": str(home),
                "KESTREL_BOOTSTRAP_ALLOWED_HOSTS": "testclient",
            },
        ):
            with TestClient(app) as client:
                bootstrap = client.get("/api/auth/key")
                forged_api_key = client.post(
                    "/api/agent/invoke",
                    headers={"X-API-Key": "forged-api-key"},
                    json={
                        "kite_evidence": {
                            "operation": "diagnostics",
                            "nonce": nonce,
                        }
                    },
                )
                response = client.post(
                    "/api/agent/invoke",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "kite_evidence": {
                            "operation": "diagnostics",
                            "nonce": nonce,
                        }
                    },
                )
                replay = client.post(
                    "/api/agent/invoke",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "kite_evidence": {
                            "operation": "diagnostics",
                            "nonce": nonce,
                        }
                    },
                )
            # A new application client models restart/reload: only the
            # isolated-home ledger, not process memory, can reject this call.
            with TestClient(app) as restarted_client:
                replay_after_restart = restarted_client.post(
                    "/api/agent/invoke",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "kite_evidence": {
                            "operation": "diagnostics",
                            "nonce": nonce,
                        }
                    },
                )
            agent.is_test_instance = False
            with TestClient(app) as non_test_client:
                non_test_bootstrap = non_test_client.get("/api/auth/key")
        assert response.status_code == 200, response.text
        assert bootstrap.status_code == 200, bootstrap.text
        assert forged_api_key.status_code == 401, forged_api_key.text
        assert replay.status_code == 409, replay.text
        assert replay_after_restart.status_code == 409, replay_after_restart.text
        assert "kite_evidence_public_key" not in non_test_bootstrap.json()
        body = response.json()
        assert body["response"] == "Kite runtime evidence operation complete."
        assert body["kite_evidence"]["observation"] == diagnostics
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(bootstrap.json()["kite_evidence_public_key"])
        ).verify(
            bytes.fromhex(body["kite_evidence"]["signature"]),
            _canonical_json(
                {
                    "contract": _KITE_EVIDENCE_CONTRACT,
                    "nonce": nonce,
                    "operation": "diagnostics",
                    "observation": diagnostics,
                }
            ).encode("utf-8"),
        )
        agent.process_input.assert_not_awaited()
        agent.storage.semantic_release_kite_diagnostics.assert_awaited_once_with(
            operation_id=f"kite-diagnostics-{nonce}"
        )
        agent.task_manager.execute_command.assert_not_awaited()
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager


@pytest.mark.asyncio
async def test_kite_invoke_suppresses_evidence_when_stop_arrives_during_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI, Response
    from starlette.requests import Request

    from kestrel_sovereign.endpoints import agent as agent_endpoint

    app = FastAPI()
    agent = MagicMock()
    agent.agent_id = "did:test:kite-stop"
    agent.register_active_request = MagicMock()
    agent.is_request_cancelled = MagicMock(side_effect=[False, True])
    agent._cleanup_cancelled_request = MagicMock()
    app.state.agent = agent
    evidence = AsyncMock(return_value=("diagnostics", {"count": 1}))
    signature = MagicMock(return_value="must-not-sign")
    monkeypatch.setattr(agent_endpoint, "_kite_runtime_observation", evidence)
    monkeypatch.setattr(agent_endpoint, "_kite_evidence_signature", signature)
    body = json.dumps(
        {
            "request_id": "kite-stop-race",
            "kite_evidence": {
                "operation": "diagnostics",
                "nonce": "f" * 64,
            },
        }
    ).encode()
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/agent/invoke",
            "headers": [],
            "query_string": b"",
            "client": ("test", 1),
            "server": ("test", 80),
            "scheme": "http",
            "app": app,
        },
        receive,
    )
    endpoint = getattr(
        agent_endpoint.invoke_agent,
        "__wrapped__",
        agent_endpoint.invoke_agent,
    )

    result = await endpoint(request, Response())

    assert result["response"] == "Request stopped during execution."
    assert "kite_evidence" not in result
    signature.assert_not_called()
    agent._cleanup_cancelled_request.assert_called_once_with("kite-stop-race")


@pytest.mark.asyncio
async def test_kite_paraphrase_recall_uses_the_verified_agent_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server must not reconstruct a stable profile for the draft gate."""
    from kestrel_sovereign.endpoints.agent import _kite_runtime_observation
    from kestrel_sovereign.agent.invocation import request_provenance
    from kestrel_sovereign.knowledge.kite_release_evidence import _kite_inference_config

    profile = inference_profile_from_config(_kite_inference_config())
    assert profile is not None
    capabilities = semantic_capabilities_from_config(_KITE_EXPERIMENTAL_CAPABILITIES)
    storage = MagicMock()
    storage.semantic_rdf_capability_report.return_value = (
        capabilities.create_rdf_codec().capability_report
    )
    storage.run_semantic_maintenance = AsyncMock(
        return_value=SimpleNamespace(status=SimpleNamespace(value="complete"))
    )
    recall_candidate = SimpleNamespace(
        assertion=SimpleNamespace(assertion_id="server-owned-assertion"),
    )
    storage.semantic_recall_candidates = AsyncMock(
        return_value=SimpleNamespace(candidates=(recall_candidate,), checkpoint_generation=7)
    )
    hydrated_candidate = SimpleNamespace(source_occurrences=("server-owned-source",))
    storage.hydrate_semantic_recall_candidates = AsyncMock(
        return_value=(hydrated_candidate,)
    )
    inference_limits = SimpleNamespace()
    maintenance_limits = SimpleNamespace()
    agent = SimpleNamespace(
        is_test_instance=True,
        storage=storage,
        semantic_inference_profile=profile,
        semantic_capabilities=capabilities,
        semantic_inference_limits=inference_limits,
        semantic_maintenance_limits=maintenance_limits,
    )
    home = tmp_path / "kite-home"
    home.mkdir(mode=0o700)
    _kite_evidence_env(monkeypatch, home)

    operation, observation = await _kite_runtime_observation(
        agent,
        request_id="kite-runtime-contract",
        provenance=request_provenance(
            source_kind="test", source_locator="test:kite-runtime-contract"
        ),
        request={"operation": "paraphrase_recall", "nonce": "d" * 64},
    )

    assert operation == "paraphrase_recall"
    assert observation == {"retrieval_count": 1, "provenance_check_count": 1}
    storage.run_semantic_maintenance.assert_awaited_once_with(
        profile,
        inference_limits=inference_limits,
        maintenance_limits=maintenance_limits,
        semantic_capabilities=capabilities,
    )
    storage.semantic_recall_candidates.assert_awaited_once_with(
        query="Which region should the deployment use?",
        candidate_scan_limit=10,
        inference_profile=profile,
        inference_limits=inference_limits,
        maintenance_limits=maintenance_limits,
    )
    storage.hydrate_semantic_recall_candidates.assert_awaited_once_with(
        ("server-owned-assertion",),
        expected_checkpoint_generation=7,
        inference_profile=profile,
        inference_limits=inference_limits,
        maintenance_limits=maintenance_limits,
    )


@pytest.mark.asyncio
async def test_kite_paraphrase_recall_fails_closed_without_a_verified_runtime_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel_sovereign.endpoints.agent import _kite_runtime_observation
    from kestrel_sovereign.agent.invocation import request_provenance

    storage = MagicMock()
    agent = SimpleNamespace(
        is_test_instance=True,
        storage=storage,
        semantic_inference_profile=None,
        semantic_capabilities=SemanticRuntimeCapabilities.stable(),
    )
    home = tmp_path / "kite-home"
    home.mkdir(mode=0o700)
    _kite_evidence_env(monkeypatch, home)

    with pytest.raises(RuntimeError, match="locally verified runtime semantic selection"):
        await _kite_runtime_observation(
            agent,
            request_id="kite-runtime-contract-reject",
            provenance=request_provenance(
                source_kind="test", source_locator="test:kite-runtime-contract-reject"
            ),
            request={"operation": "paraphrase_recall", "nonce": "e" * 64},
        )
    storage.run_semantic_maintenance.assert_not_called()


def _kite_evidence_env(
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    *,
    trusted_root: Path | None = None,
) -> None:
    monkeypatch.setenv("KESTREL_KITE_RELEASE_EVIDENCE", "1")
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.setenv("KESTREL_KITE_RELEASE_EVIDENCE_ROOT", str(trusted_root or home))


def test_kite_signing_accepts_a_private_home_below_a_writable_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe_ancestor = tmp_path / "unsafe-ancestor"
    unsafe_ancestor.mkdir(mode=0o700)
    unsafe_ancestor.chmod(0o777)
    home = unsafe_ancestor / "kite-home"
    home.mkdir(mode=0o700)
    _kite_evidence_env(monkeypatch, home)

    first_public_key = kite_evidence_public_key()
    assert kite_evidence_public_key() == first_public_key
    consume_kite_evidence_nonce("a" * 64)
    with pytest.raises(KiteEvidenceNonceReplay, match="already consumed"):
        consume_kite_evidence_nonce("a" * 64)


@pytest.mark.parametrize(
    ("home_suffix", "root_suffix"),
    (
        (("..", "outside"), ()),
        (("nested", "..", "..", "outside"), ()),
        ((), ("..", "outside")),
    ),
)
def test_kite_signing_rejects_lexical_traversal_before_path_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    home_suffix: tuple[str, ...],
    root_suffix: tuple[str, ...],
) -> None:
    root = tmp_path / "trusted-root"
    root.mkdir(mode=0o700)
    (root / "nested").mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    home = root.joinpath(*home_suffix) if home_suffix else outside
    trusted_root = root.joinpath(*root_suffix) if root_suffix else root
    _kite_evidence_env(monkeypatch, home, trusted_root=trusted_root)

    with pytest.raises(KiteEvidenceSigningError, match="parent traversal"):
        kite_evidence_public_key()


def test_kite_signing_rejects_symlink_escape_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "trusted-root"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (outside / "kite-home").mkdir(mode=0o700)
    escape = root / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    _kite_evidence_env(monkeypatch, escape / "kite-home", trusted_root=root)

    with pytest.raises(KiteEvidenceSigningError, match="symlink"):
        kite_evidence_public_key()


def test_kite_signing_rejects_trusted_home_replacement_while_opening_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe_ancestor = tmp_path / "unsafe-ancestor"
    unsafe_ancestor.mkdir(mode=0o700)
    unsafe_ancestor.chmod(0o777)
    home = unsafe_ancestor / "kite-home"
    home.mkdir(mode=0o700)
    _kite_evidence_env(monkeypatch, home)
    kite_evidence_public_key()

    replacement = unsafe_ancestor / "replacement-home"
    replacement.mkdir(mode=0o700)
    replacement_key = replacement / ".kite-evidence-ed25519.key"
    replacement_key.write_bytes(bytes(reversed(range(32))))
    replacement_key.chmod(0o600)
    displaced = unsafe_ancestor / "displaced-home"
    original_open = kite_evidence_signing.os.open
    replaced = False

    def replace_before_open(path, flags, mode=0o777):
        nonlocal replaced
        if not replaced and Path(path) == home / ".kite-evidence-ed25519.key":
            replaced = True
            home.rename(displaced)
            replacement.rename(home)
        return original_open(path, flags, mode)

    monkeypatch.setattr(kite_evidence_signing.os, "open", replace_before_open)
    with pytest.raises(KiteEvidenceSigningError, match="changed while opening"):
        kite_evidence_public_key()


def test_kite_nonce_ledger_rejects_replacement_while_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe_ancestor = tmp_path / "unsafe-ancestor"
    unsafe_ancestor.mkdir(mode=0o700)
    unsafe_ancestor.chmod(0o777)
    home = unsafe_ancestor / "kite-home"
    home.mkdir(mode=0o700)
    _kite_evidence_env(monkeypatch, home)
    consume_kite_evidence_nonce("b" * 64)

    ledger = home / ".kite-evidence-nonces.sqlite3"
    replacement = home / "replacement.sqlite3"
    replacement.touch(mode=0o600)
    displaced = home / "displaced.sqlite3"
    original_connect = kite_evidence_signing.sqlite3.connect
    replaced = False

    def replace_before_connect(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            ledger.rename(displaced)
            replacement.rename(ledger)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(kite_evidence_signing.sqlite3, "connect", replace_before_connect)
    with pytest.raises(KiteEvidenceSigningError, match="nonce ledger changed"):
        consume_kite_evidence_nonce("c" * 64)


def test_kite_sleep_measurement_times_only_typed_http_sleep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel_sovereign.knowledge import kite_release_evidence

    harness = KiteHttpHarness(_config(tmp_path), run=_fake_create)
    harness._process = _Process(100)  # type: ignore[assignment]
    harness._api_key = "test-key"
    harness._evidence_public_key = _TEST_SIGNING_PUBLIC_KEY
    counters = iter((10.0, 10.005, 20.0, 20.005, 30.0, 30.005))
    monkeypatch.setattr(kite_release_evidence.time, "perf_counter", lambda: next(counters))
    monkeypatch.setattr(KiteHttpHarness, "_marker", staticmethod(lambda: "kite-evidence-sleep-marker"))
    operations: list[str] = []

    def invoke(_self, prompt: str, **kwargs: Any):
        assert prompt == ""
        request = kwargs["kite_evidence"]
        assert isinstance(request, dict)
        operation = request["operation"]
        operations.append(operation)
        observation = (
            {"fact_write_count": 1}
            if operation == "save"
            else {"sleep_success_count": 1}
        )
        return _typed_response(request, observation)

    monkeypatch.setattr(KiteHttpHarness, "invoke", invoke)
    assert harness.measure_sleep(changed=True) == pytest.approx((5.0, 5.0, 5.0))
    assert operations == ["save", "sleep_changed", "save", "sleep_changed", "save", "sleep_changed"]


def test_kite_workload_registration_has_only_core_owned_erasure_stages() -> None:
    from kestrel_sovereign.knowledge.kite_release_evidence_workloads import kite_http_workloads

    workloads = kite_http_workloads(lambda _gate, _backend: None)  # type: ignore[arg-type]
    commands = {command_id for _runner_id, command_id in workloads}
    assert "erasure_active_assertions_v1" in commands
    assert "erasure_vector_index_v1" in commands
    assert "external_corpus_consumed_v1" not in commands
    assert "erasure_served_adapter_eligibility_v1" not in commands
    assert "benchmark_changed_work_sleep_sqlite_kite_http_v1" in commands


def test_core_erasure_stage_rejects_missing_or_mismatched_aggregate_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = KiteHttpHarness(_config(tmp_path), run=_fake_create)
    harness._process = _Process(100)  # type: ignore[assignment]
    harness._api_key = "test-key"
    harness._evidence_public_key = _TEST_SIGNING_PUBLIC_KEY
    monkeypatch.setattr(
        KiteHttpHarness,
        "_evidence",
        lambda _self, operation: {"active_assertions": {"erased_count": 1, "remaining_count": 0}}
        if operation == "erasure_core_snapshot" else {},
    )
    with pytest.raises(KiteEvidenceError, match="unexpected stage set"):
        harness.core_erasure_stage(ErasureStage.ACTIVE_ASSERTIONS)
