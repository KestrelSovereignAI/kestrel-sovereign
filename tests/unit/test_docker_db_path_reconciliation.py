"""Drift guards for Docker ``KESTREL_DB_PATH`` directory semantics."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_AGENT_DATA_DIR = "/app/agent_data"


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_single_agent_dockerfiles_use_agent_data_dir_for_db_path():
    dockerfiles = [
        "Dockerfile",
        "Dockerfile.agent.remote",
        "docker/Dockerfile.remote",
        "docker/Dockerfile.standalone",
        "docker/Dockerfile.gpu",
        "docker/Dockerfile.cloudrun",
    ]

    for dockerfile in dockerfiles:
        text = _read(dockerfile)
        assert f"ENV KESTREL_DB_PATH={CANONICAL_AGENT_DATA_DIR}" in text
        assert "ENV KESTREL_DB_PATH=/app/kestrel.db" not in text
        assert "ENV KESTREL_DB_PATH=/app/kestrel_prime.db" not in text


def test_compose_mount_and_env_point_to_same_agent_data_dir():
    text = _read("docker-compose.yml")

    assert f"KESTREL_DB_PATH={CANONICAL_AGENT_DATA_DIR}" in text
    assert "./agent_data:/app/agent_data" in text
    assert "/usr/src/app/kestrel.db" not in text


def test_container_entrypoint_initializes_db_inside_agent_data_dir():
    text = _read("docker_entrypoint.sh")

    assert 'export KESTREL_DB_PATH="${KESTREL_DB_PATH:-/app/agent_data}"' in text
    assert 'mkdir -p "$KESTREL_DB_PATH"' in text
    assert '[ ! -f "$KESTREL_DB_PATH/kestrel_prime.db" ]' in text
    assert "/app/kestrel.db" not in text


def test_init_agent_identity_uses_db_path_as_target_directory():
    text = _read("scripts/init_agent_identity.py")

    assert 'os.environ.get("KESTREL_DB_PATH")' in text
    assert 'Path.cwd() / "agent_data"' in text
    assert 'create_kestrel_identity(str(target_dir))' in text
    assert "target_dir = '/app'" not in text


def test_init_agent_identity_falls_back_to_cwd_when_db_path_unset(
    monkeypatch,
    tmp_path,
):
    cwd = tmp_path / "cwd"
    calls: list[str] = []

    fake_inception = types.ModuleType("kestrel_sovereign.inception_service")

    def fake_create_kestrel_identity(target_dir: str):
        calls.append(target_dir)
        return types.SimpleNamespace(
            agent_did="did:example:test",
            db_path=str(Path(target_dir) / "kestrel_prime.db"),
        )

    fake_inception.create_kestrel_identity = fake_create_kestrel_identity

    cwd.mkdir()
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
    monkeypatch.chdir(cwd)
    monkeypatch.setitem(
        sys.modules,
        "kestrel_sovereign.inception_service",
        fake_inception,
    )

    runpy.run_path(str(REPO_ROOT / "scripts/init_agent_identity.py"))

    # Unset → writable cwd-relative dir (proves it did NOT use the
    # container-only /app/agent_data default).
    assert calls == [str(cwd / "agent_data")]
    assert (cwd / "agent_data").is_dir()


def test_init_agent_identity_honors_absolute_db_path_with_missing_parent(
    monkeypatch,
    tmp_path,
):
    custom_target = tmp_path / "missing-parent" / "custom-agent-data"
    cwd = tmp_path / "cwd"
    calls: list[str] = []

    fake_inception = types.ModuleType("kestrel_sovereign.inception_service")

    def fake_create_kestrel_identity(target_dir: str):
        calls.append(target_dir)
        return types.SimpleNamespace(
            agent_did="did:example:test",
            db_path=str(Path(target_dir) / "kestrel_prime.db"),
        )

    fake_inception.create_kestrel_identity = fake_create_kestrel_identity

    cwd.mkdir()
    monkeypatch.setenv("KESTREL_DB_PATH", str(custom_target))
    monkeypatch.chdir(cwd)
    monkeypatch.setitem(
        sys.modules,
        "kestrel_sovereign.inception_service",
        fake_inception,
    )

    runpy.run_path(str(REPO_ROOT / "scripts/init_agent_identity.py"))

    assert calls == [str(custom_target)]
    assert custom_target.is_dir()
    assert not (cwd / "agent_data").exists()
