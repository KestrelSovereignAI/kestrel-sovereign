"""Build public Go dependencies and own isolated native interoperability agents."""

import shutil
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio

from tests.utils.native_go_peer import boot_native_go_environment


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
