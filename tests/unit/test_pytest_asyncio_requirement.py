"""A run without pytest-asyncio stops before its first test (#3522).

With ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` and no ``-p asyncio``, the suite used
to pass every sync test and then error at the first async fixture, so a
targeted gate read "N passed" over files it never reached.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import tests.conftest as root
from tests.shared.child_pytest import run_child_pytest


def _config_without_pytest_asyncio():
    return SimpleNamespace(
        pluginmanager=SimpleNamespace(is_registered=lambda plugin: False)
    )


def test_this_run_has_pytest_asyncio(request):
    root._require_pytest_asyncio(request.config)


def test_a_run_without_pytest_asyncio_is_refused_with_the_plugins_to_name(
    monkeypatch,
):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    with pytest.raises(pytest.UsageError, match="requires the pytest-asyncio") as exc:
        root._require_pytest_asyncio(_config_without_pytest_asyncio())

    assert "-p asyncio -p anyio" in str(exc.value)


def test_the_refusal_names_no_autoload_flag_the_run_did_not_set(monkeypatch):
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)

    with pytest.raises(pytest.UsageError) as exc:
        root._require_pytest_asyncio(_config_without_pytest_asyncio())

    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in str(exc.value)


def test_the_root_conftest_refuses_before_configuring_anything():
    """Mutating the wiring away must fail here."""
    with pytest.raises(pytest.UsageError, match="requires the pytest-asyncio"):
        root.pytest_configure(_config_without_pytest_asyncio())


def test_declared_and_locked_asyncio_plugin_retires_its_original_loop():
    """Both installation surfaces must exclude the known leaking release (#3963)."""
    project = Path(__file__).resolve().parents[2]
    config = tomllib.loads((project / "pyproject.toml").read_text())
    declarations = (
        config["project"]["optional-dependencies"]["test"],
        config["dependency-groups"]["test"],
    )
    for dependencies in declarations:
        plugin = [
            Requirement(value)
            for value in dependencies
            if isinstance(value, str) and Requirement(value).name == "pytest-asyncio"
        ]
        assert len(plugin) == 1
        assert Version("1.3.0") not in plugin[0].specifier
        assert Version("1.4.0") in plugin[0].specifier
    lock = tomllib.loads((project / "uv.lock").read_text())
    resolved = [p for p in lock["package"] if p["name"] == "pytest-asyncio"]
    assert resolved
    assert all(Version(p["version"]) >= Version("1.4.0") for p in resolved)


@pytest.mark.timeout(135)
def test_async_test_then_sync_asyncio_run_leaves_no_original_loop(tmp_path):
    """Exercise the real plugin, then collect its original loop and sockets."""
    completed = run_child_pytest(
        tmp_path,
        conftest="",
        test_file="test_loop_retirement.py",
        source=(
            "import asyncio\nimport gc\nimport pytest\n\n"
            "@pytest.mark.asyncio\n"
            "async def test_async_scope():\n"
            "    await asyncio.sleep(0)\n\n"
            "def test_sync_scope():\n"
            "    asyncio.run(asyncio.sleep(0))\n"
            "    gc.collect()\n"
        ),
        args=[
            "test_loop_retirement.py", "-q", "-p", "pytest_asyncio.plugin",
            "-o", "asyncio_default_fixture_loop_scope=function",
            "-W", "error::ResourceWarning",
            "-W", "error::pytest.PytestUnraisableExceptionWarning",
        ],
        timeout=120,
    )
    assert completed.returncode == pytest.ExitCode.OK, completed.stdout
    assert "2 passed" in completed.stdout
