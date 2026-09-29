"""Drift guards for Docker ``KESTREL_DB_PATH`` directory semantics."""

from __future__ import annotations

import runpy
import subprocess
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_AGENT_DATA_DIR = "/app/agent_data"
UV_RUNTIME_DOCKERFILES = (
    "Dockerfile",
    "Dockerfile.agent.remote",
    "docker/Dockerfile.cloudrun",
    "docker/Dockerfile.gpu",
    "docker/Dockerfile.multi_agent",
    "docker/Dockerfile.remote",
    "docker/Dockerfile.sovereign",
    "docker/Dockerfile.standalone",
)


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _entrypoint_python_body(function_name: str) -> str:
    """Extract one single-quoted Python heredoc from the entrypoint."""

    entrypoint = _read("docker/multi_agent_entrypoint.sh")
    function = entrypoint.split(f"{function_name}() {{", 1)[1]
    heredoc = function.split("<<'PY'", 1)[1]
    return heredoc.split("\nPY\n", 1)[0].lstrip("\n")


def test_runtime_images_install_uv_hold_filesystem_sandbox():
    """Every Linux runtime that exposes UV compute carries bubblewrap."""

    for dockerfile in UV_RUNTIME_DOCKERFILES:
        lines = {line.strip() for line in _read(dockerfile).splitlines()}
        assert "bubblewrap \\" in lines, dockerfile


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
        assert "ENV KESTREL_HOST_DB_PATH=" not in text
        assert "ENV KESTREL_DB_PATH=/app/kestrel.db" not in text
        assert "ENV KESTREL_DB_PATH=/app/kestrel_prime.db" not in text


def test_multi_agent_image_persists_host_control_database_on_agent_volume():
    text = _read("docker/Dockerfile.multi_agent")
    entrypoint = _read("docker/multi_agent_entrypoint.sh")

    assert "ENV KESTREL_HOST_DB_PATH=" not in text
    assert 'if [ -n "${KESTREL_HOST_DB_PATH:-}" ]; then' in entrypoint
    assert "unset KESTREL_DERIVED_HOST_DB_PATH" in entrypoint
    assert (
        'export KESTREL_HOST_DB_PATH="$AGENT_DATA_DIR/host-data/host-features.db"'
        in entrypoint
    )
    assert (
        'export KESTREL_DERIVED_HOST_DB_PATH="$KESTREL_HOST_DB_PATH"'
        in entrypoint
    )
    assert "paths_overlap_by_filesystem_identity" in entrypoint
    assert 'local first="${1%/}/"' not in entrypoint


def test_multi_agent_entrypoint_executes_filesystem_identity_overlap(
    tmp_path: Path,
):
    """The shell's actual heredoc rejects an unresolved case-fold alias."""

    body = _entrypoint_python_body("paths_overlap")
    parent = tmp_path / "agent-data"
    parent.mkdir()

    overlap = subprocess.run(
        [
            sys.executable,
            "-",
            str(parent / "host-data"),
            str(parent / "HOST-DATA" / "agent"),
        ],
        input=body,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    separate = subprocess.run(
        [
            sys.executable,
            "-",
            str(parent / "host-data"),
            str(parent / "ordinary-agent"),
        ],
        input=body,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert overlap.returncode == 0
    assert separate.returncode == 1


def test_compose_mount_and_env_point_to_same_agent_data_dir():
    text = _read("docker-compose.yml")

    assert f"KESTREL_DB_PATH={CANONICAL_AGENT_DATA_DIR}" in text
    assert "KESTREL_HOST_DB_PATH=" not in text
    assert "./agent_data:/app/agent_data" in text
    assert "/usr/src/app/kestrel.db" not in text


def test_sovereign_image_keeps_host_control_state_on_its_data_volume():
    text = _read("docker/Dockerfile.sovereign")

    assert "ENV KESTREL_DB_PATH=/data" in text
    assert "ENV KESTREL_HOST_DB_PATH=" not in text


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
