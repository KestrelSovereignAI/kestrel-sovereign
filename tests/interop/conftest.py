"""Build public Go dependencies and own isolated native interoperability agents."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio

from tests.utils.native_go_peer import boot_native_go_environment
from tests.utils.native_tck_source import prepared_tck_source


@pytest.fixture(scope="session")
def go_peer(tmp_path_factory):
    """Compile and test the pinned public SDK, without private SDK patches."""
    go = shutil.which("go")
    assert go is not None, "Go is required for the native A2A interoperability gate"
    source = Path(__file__).with_name("a2a_go_peer")
    binary = tmp_path_factory.mktemp("native-go-peer") / "peer"
    for command in ([go, "test", "./..."], [go, "build", "-o", str(binary), "."]):
        result = subprocess.run(
            command, cwd=source, capture_output=True, text=True, timeout=180, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest_asyncio.fixture
async def native_go_environment(go_peer, tmp_path, monkeypatch):
    """Share one canonical real Kestrel boot between interop and conformance gates."""
    async with boot_native_go_environment(go_peer, tmp_path, monkeypatch) as environment:
        yield environment


@pytest.fixture(scope="session")
def a2a_tck_source(tmp_path_factory, go_peer):
    """Own the exact patched runner, its locked environment and unit verification."""
    env = {key: value for key, value in os.environ.items() if key not in {
        "VIRTUAL_ENV", "PYTHONPATH", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_ADDOPTS",
    }}
    with prepared_tck_source(tmp_path_factory.getbasetemp()) as source:
        env.update(
            PYTHONPATH=str(source),
            KESTREL_TCK_GO_PEER=str(go_peer),
        )
        log_path = source.parent / "tck-source-verification.log"
        with log_path.open("w") as log:
            for command in (
                ["uv", "sync", "--locked"],
                ["uv", "run", "--no-sync", "pytest", "tests/unit", "-q"],
                ["uv", "run", "--no-sync", "python", str(Path(__file__).with_name("tck_auth_unit.py"))],
            ):
                result = subprocess.run(
                    command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False,
                )
                assert result.returncode == 0, log_path.read_text()
        yield source
