"""Native HTTP/process retirement and original boot receipts, entirely offline."""

import asyncio
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.lifecycle_checks import (
    verify_llm_providers_reachable,
    warn_stale_bootstrap_pending,
)
from kestrel_sovereign.llm.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerConnectionClosed,
)
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_execution_custody_review43_native import carries


async def native_process(*, close_stdout=False):
    script = "import sys; sys.stdin.buffer.read()"
    if close_stdout:
        script = "import os,sys; os.close(1); sys.stdin.buffer.read()"
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "unknown", "cancel_unknown", "group"])
async def test_default_refresh_is_physically_joined_and_original_control_retained(
    tmp_path, monkeypatch, kind,
):
    original = receipt(kind) if kind else None
    auth = tmp_path / "auth.json"
    contents = json.dumps({"tokens": {"access_token": "offline-old",
        "refresh_token": "offline-refresh", "account_id": "offline-account"}})
    auth.write_text(contents)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    entered, release = asyncio.Event(), asyncio.Event()
    retired, requests = [], []

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            requests.append(request)
            return httpx.Response(200, json={"access_token": "offline-rotated"})

        async def aclose(self):
            entered.set()
            await release.wait()
            retired.append(True)
            if original is not None:
                raise original

    http = httpx.AsyncClient(transport=Transport())
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: http)
    client = CodexAppServerClient(binary="offline-no-codex")
    proc = await native_process()
    client._proc, client._initialized = proc, True
    closer = None
    client._dispatch({"id": 1, "method": "account/chatgptAuthTokens/refresh"})
    refresh_tasks = tuple(t for t in asyncio.all_tasks()
        if t.get_coro().__qualname__.endswith("_handle_server_request"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert len(refresh_tasks) == 1
        closer = asyncio.create_task(client.aclose())
        await asyncio.sleep(0.03)
        assert not closer.done(), "shutdown reported retirement with native refresh close live"
        closer.cancel()
        closer.cancel()
        client._dispatch({"id": 2, "method": "account/chatgptAuthTokens/refresh"})
        release.set()
        outcome = await await_owned_task(closer)
        assert retired == [True]
        assert all(task.done() for task in refresh_tasks)
        assert len(requests) == 1, "retirement admitted a new default refresh"
        assert proc.returncode is not None
        assert auth.read_text() == contents, "retired refresh persisted rotated credentials"
        if original is not None:
            assert carries(outcome.error, original), "shutdown erased original transport receipt"
        else:
            assert isinstance(outcome.error, asyncio.CancelledError)
        with pytest.raises(BaseException) as repeated:
            if original is not None:
                await client.aclose()
            else:
                await client.ensure_started()
        if original is not None:
            assert carries(repeated.value, original)
        else:
            assert isinstance(repeated.value, CodexAppServerConnectionClosed)
    finally:
        release.set()
        for task in refresh_tasks:
            await await_owned_task(task)
        if closer is not None:
            await await_owned_task(closer)
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


@pytest.mark.asyncio
async def test_cancel_resistant_default_refresh_cannot_publish_after_close(
    tmp_path, monkeypatch,
):
    auth = tmp_path / "auth.json"
    contents = json.dumps({"tokens": {"refresh_token": "offline-refresh",
        "account_id": "offline-account"}})
    auth.write_text(contents)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    entered, release = asyncio.Event(), asyncio.Event()

    async def respond(request):
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        return httpx.Response(200, json={"access_token": "offline-rotated"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: http)
    client = CodexAppServerClient(binary="offline-no-codex")
    proc = await native_process()
    client._proc = proc
    client._dispatch({"id": 1, "method": "account/chatgptAuthTokens/refresh"})
    refresh_tasks = tuple(t for t in asyncio.all_tasks()
        if t.get_coro().__qualname__.endswith("_handle_server_request"))
    closer = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        closer = asyncio.create_task(client.aclose())
        await asyncio.sleep(0.03)
        assert not closer.done(), "cancel-resistant native HTTP work escaped retirement"
        release.set()
        outcome = await await_owned_task(closer)
        assert outcome.error is None
        assert auth.read_text() == contents
        assert proc.returncode is not None
    finally:
        release.set()
        for task in refresh_tasks:
            await await_owned_task(task)
        if closer is not None:
            await await_owned_task(closer)
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


@pytest.mark.asyncio
async def test_native_stdout_eof_retires_requests_before_restart():
    client = CodexAppServerClient(binary="offline-no-codex")
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(params):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await release.wait()

    client._server_request_handlers[("offline/default", None)] = handler
    proc = await native_process(close_stdout=True)
    client._proc, client._initialized = proc, True
    client._dispatch({"id": 1, "method": "offline/default"})
    requests = tuple(t for t in asyncio.all_tasks()
        if t.get_coro().__qualname__.endswith("_handle_server_request"))
    reader = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        reader = asyncio.create_task(client._read_loop())
        await asyncio.sleep(0.05)
        assert not reader.done(), "EOF escaped still-live request retirement"
        assert client._proc is proc and client._initialized
        with pytest.raises(CodexAppServerConnectionClosed):
            await client.ensure_started()
        release.set()
        outcome = await await_owned_task(reader)
        assert outcome.error is None
        assert proc.returncode is not None, "stdout EOF left native process unjoined"
        assert client._proc is None and not client._initialized
    finally:
        release.set()
        for task in requests:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        if reader is not None:
            await await_owned_task(reader)
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["authority", "unknown", "cancel_unknown", "group", "cause"])
@pytest.mark.parametrize("cloud_fallback", [False, True])
async def test_actual_native_boot_probe_never_erases_original_control(
    monkeypatch, kind, cloud_fallback,
):
    original = receipt(kind)
    if kind == "cause":
        original = OSError("native boot carrier")
        original.__cause__ = receipt("unknown")

    def respond(request):
        raise original

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: http)
    providers = [{"name": "offline-local", "vendor": "openai", "is_local": True,
        "adapter": OpenAIAdapter(), "client": None, "base_url": "http://offline.invalid/v1"}]
    if cloud_fallback:
        providers.append({"name": "offline-cloud", "is_cloud": True})
    task = asyncio.create_task(verify_llm_providers_reachable(
        SimpleNamespace(providers=providers), env={},
    ))
    outcome = await await_owned_task(task)
    assert carries(outcome.error, original), "boot fallback erased original native control"
    assert http.is_closed


@pytest.mark.asyncio
async def test_stale_bootstrap_lookup_preserves_original_control():
    original = OSError("native storage carrier")
    original.__cause__ = receipt("unknown")
    calls = []

    async def get_node(agent_id):
        raise original

    async def check_pending_timeout(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(is_stale=False)

    agent = SimpleNamespace(agent_id="offline", storage=SimpleNamespace(get_node=get_node),
        bootstrap_service=SimpleNamespace(check_pending_timeout=check_pending_timeout))
    with pytest.raises(OSError) as failure:
        await warn_stale_bootstrap_pending(agent)
    assert failure.value is original
    assert calls == [], "fallback began bootstrap work after original authority loss"
