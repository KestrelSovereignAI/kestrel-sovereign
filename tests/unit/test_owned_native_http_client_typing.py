"""Real clients and pinned static checks for Core's one HTTP close owner."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import TracebackType

import httpx
import httpx2
import pytest

from kestrel_sovereign.execution_custody import owned_native_http_client


@pytest.mark.parametrize("negative", [False, True], ids=["native-positive", "generic-negative"])
def test_pinned_public_context_type_contract(tmp_path, negative):
    root = Path(__file__).resolve().parents[2]
    name = "generic_suppression" if negative else "contracts"
    fixture = root / "tests" / "static" / f"owned_native_http_client_{name}.py"
    config = tmp_path / "pyrightconfig.json"
    config.write_text(json.dumps({
        "extraPaths": [str(root)],
        "typeCheckingMode": "basic",
        "reportPossiblyUnboundVariable": "error",
        "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
    }))
    completed = subprocess.run(
        [sys.executable, "-m", "pyright", "--outputjson", "--project", str(config),
         "--pythonpath", sys.executable, str(fixture)],
        cwd=root, capture_output=True, text=True, timeout=120,
    )
    assert completed.stderr == "", completed.stderr
    report = json.loads(completed.stdout)
    assert report["version"] == "1.1.403"
    diagnostics = report["generalDiagnostics"]
    if negative:
        assert completed.returncode == 1, completed.stdout
        assert len(diagnostics) == 1, completed.stdout
        diagnostic = diagnostics[0]
        assert diagnostic["rule"] == "reportPossiblyUnboundVariable"
        assert diagnostic["file"] == str(fixture)
        assert '"result" is possibly unbound' == diagnostic["message"]
        lines = fixture.read_text().splitlines()
        assert lines[diagnostic["range"]["start"]["line"]].strip() == "return result"
    else:
        assert completed.returncode == 0, completed.stdout
        assert diagnostics == [], completed.stdout
    assert report["summary"]["errorCount"] == int(negative)


@pytest.fixture(params=[httpx, httpx2], ids=["httpx", "httpx2"])
def native_client(request):
    module = request.param
    retired = []

    class Transport(module.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise AssertionError("these client-lifetime controls must not issue requests")

        async def aclose(self):
            retired.append(self)

    return module.AsyncClient, Transport(), retired


async def test_native_body_failure_is_original_and_client_physically_closes(native_client):
    client_type, transport, retired = native_client
    client = client_type(transport=transport)
    original = RuntimeError("native body failure")
    try:
        with pytest.raises(RuntimeError) as raised:
            async with owned_native_http_client(client) as entered:
                assert entered is client
                raise original
        assert raised.value is original
        assert client.is_closed
        assert retired == [transport]
    finally:
        if not client.is_closed:
            await client.aclose()


async def test_native_subclass_foreign_enter_does_not_replace_original(native_client):
    client_type, transport, retired = native_client
    foreign = object()

    class ForeignEnteringClient(client_type):
        async def __aenter__(self):
            await super().__aenter__()
            return foreign

    client = ForeignEnteringClient(transport=transport)
    try:
        async with owned_native_http_client(client) as entered:
            assert entered is client
            assert entered is not foreign
        assert client.is_closed
        assert retired == [transport]
    finally:
        if not client.is_closed:
            await client.aclose()


async def test_generic_suppression_and_entered_value_are_preserved():
    original = RuntimeError("generic body failure")
    observed = []
    entered_value = object()

    class SuppressingClient:
        async def __aenter__(self) -> object:
            return entered_value

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            observed.append((exc_type, exc_value, traceback))
            return True

    async with owned_native_http_client(SuppressingClient()) as entered:
        assert entered is entered_value
        raise original
    assert len(observed) == 1
    assert observed[0][0] is RuntimeError
    assert observed[0][1] is original
    assert observed[0][2] is not None


@pytest.mark.parametrize("delegated_exit", [True, False, None])
async def test_generic_exit_is_normalized_to_actual_bool(delegated_exit):
    original = RuntimeError("generic body failure")
    observed = []

    class GenericClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            observed.append(exc_value)
            return delegated_exit

    client = GenericClient()
    context = owned_native_http_client(client)
    assert await context.__aenter__() is client
    result = await context.__aexit__(RuntimeError, original, None)
    assert type(result) is bool
    assert result is (delegated_exit is True)
    assert observed == [original]
