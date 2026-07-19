"""Direct-server command-line contract tests (issue #2612)."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import uvicorn

import kestrel_sovereign.server as server
from kestrel_sovereign.multi_agent.config import MultiAgentConfig


_SERVER_ENV_NAMES = ("KESTREL_SERVER_HOST", "PORT")


@pytest.fixture(autouse=True)
def _clear_server_environment(monkeypatch):
    """Keep host-machine bind settings from influencing CLI contract tests."""
    for name in _SERVER_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("environment", "arguments", "expected_host", "expected_port"),
    [
        ({}, [], "0.0.0.0", 8888),
        (
            {"KESTREL_SERVER_HOST": "192.0.2.10", "PORT": "9123"},
            [],
            "192.0.2.10",
            9123,
        ),
        (
            {"KESTREL_SERVER_HOST": "192.0.2.10", "PORT": "9123"},
            ["--host", "127.0.0.1", "--port", "9999"],
            "127.0.0.1",
            9999,
        ),
        (
            {"KESTREL_SERVER_HOST": "192.0.2.10", "PORT": "9123"},
            ["--host", "::1"],
            "::1",
            9123,
        ),
    ],
)
def test_main_honors_cli_environment_and_default_precedence(
    monkeypatch,
    environment,
    arguments,
    expected_host,
    expected_port,
):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    server.main(arguments)

    assert calls == [((server.app,), {"host": expected_host, "port": expected_port})]
    assert os.environ["KESTREL_SERVER_HOST"] == expected_host
    assert os.environ["PORT"] == str(expected_port)


@pytest.mark.parametrize("port", ["not-a-port", "0", "65536"])
def test_main_rejects_invalid_cli_ports_before_starting_uvicorn(
    monkeypatch, capsys, port
):
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("uvicorn must not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        server.main(["--port", port])

    assert exc_info.value.code == 2
    assert "usage: python -m kestrel_sovereign.server" in capsys.readouterr().err


def test_main_rejects_invalid_port_environment_before_starting_uvicorn(
    monkeypatch, capsys
):
    monkeypatch.setenv("PORT", "not-a-port")
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("uvicorn must not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        server.main([])

    assert exc_info.value.code == 2
    assert "argument --port: port must be an integer" in capsys.readouterr().err


def test_main_rejects_empty_host_before_starting_uvicorn(monkeypatch, capsys):
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("uvicorn must not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        server.main(["--host", " "])

    assert exc_info.value.code == 2
    assert "host must not be empty" in capsys.readouterr().err


def test_main_rejects_unknown_arguments_with_usage(monkeypatch, capsys):
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("uvicorn must not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        server.main(["--workers", "2"])

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "usage: python -m kestrel_sovereign.server" in error
    assert "unrecognized arguments: --workers 2" in error


def test_module_entry_point_rejects_unknown_arguments_in_a_subprocess():
    project_root = Path(server.__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(project_root), env.get("PYTHONPATH")))
    )

    result = subprocess.run(
        [sys.executable, "-m", "kestrel_sovereign.server", "--workers", "2"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "usage: python -m kestrel_sovereign.server" in result.stderr
    assert "unrecognized arguments: --workers 2" in result.stderr


def test_effective_module_address_updates_host_feature_context():
    config = MultiAgentConfig()

    server._apply_platform_host_port(
        config,
        {"KESTREL_SERVER_HOST": "127.0.0.1", "PORT": "9999"},
    )

    assert config.host.bind == "127.0.0.1"
    assert config.host.port == 9999


@pytest.mark.parametrize(
    "entrypoint",
    ["docker/cloudrun_entrypoint.sh", "docker/multi_agent_entrypoint.sh"],
)
def test_managed_container_entrypoints_keep_platform_bind_contract(entrypoint):
    project_root = Path(server.__file__).resolve().parents[1]
    script = (project_root / entrypoint).read_text()

    assert 'PORT="${PORT:-8080}"' in script
    assert (
        'uvicorn kestrel_sovereign.server:app --host 0.0.0.0 --port "$PORT"'
        in script
    )


def test_repo_root_compatibility_shim_delegates_to_packaged_main(monkeypatch):
    project_root = Path(server.__file__).resolve().parents[1]
    calls = []
    monkeypatch.setattr(server, "main", lambda: calls.append(True))

    runpy.run_path(str(project_root / "server.py"), run_name="__main__")

    assert calls == [True]
