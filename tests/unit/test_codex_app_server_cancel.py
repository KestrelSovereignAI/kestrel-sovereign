"""Cancellation cleanup for the codex app-server client (#1421).

When the user hits Ctrl-C (or ``asyncio.timeout`` fires on the agent
turn) while a codex app-server RPC is pending, the ``mid`` for that
request must be removed from ``_pending`` so a late-arriving response
doesn't try to resolve a dead future and so the dict doesn't grow for
the life of the process.

The original ``CancelledError`` is re-raised — converting to a typed
error here would break ``asyncio.timeout``'s ability to rewrite the
cancellation into ``TimeoutError``. Codex review of an earlier draft
flagged this, leading to the narrower scope landed here.
"""
import asyncio
import json

import pytest

from kestrel_sovereign.llm.cancellation import AuthCancellationToken
from kestrel_sovereign.llm.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerCancelled,
    CodexAppServerError,
)


def _stub_client() -> CodexAppServerClient:
    """Bare ``CodexAppServerClient`` without spawning a subprocess.
    Mirrors the pattern in the main ``TestDispatchLogic`` suite.
    """
    c = CodexAppServerClient.__new__(CodexAppServerClient)
    c._pending = {}
    c._turn_sinks = {}
    c._server_request_handlers = {}
    c._inflight_server_requests = {}
    c._closed_error = None
    c._client_version = "test"
    c._next_id = 1
    c._sent = []
    c._send = lambda obj: c._sent.append(obj)
    c.notify = lambda method, params=None: c._sent.append(
        {"method": method, "params": params or {}}
    )
    return c


@pytest.mark.asyncio
async def test_request_unguarded_drops_pending_on_cancel():
    """Cancel must pop ``mid`` from ``_pending`` so a late response from
    the app-server doesn't try to resolve a dead future. Without the
    cleanup, ``_pending`` grew for the life of the process every time
    the user bailed on a hung RPC.
    """
    c = _stub_client()
    task = asyncio.create_task(
        c._request_unguarded("any/method", {}, timeout=60)
    )
    # Yield so the coroutine reaches ``await asyncio.wait_for(...)``.
    await asyncio.sleep(0)
    assert len(c._pending) == 1
    mid = next(iter(c._pending))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert mid not in c._pending
    assert c._pending == {}


@pytest.mark.asyncio
async def test_request_unguarded_preserves_cancellation_for_asyncio_timeout():
    """``asyncio.timeout(...)`` cancels the inner task on expiry and
    catches ``CancelledError`` in ``__aexit__`` to rewrite it as
    ``TimeoutError``. The cancellation handler in ``_request_unguarded``
    must re-raise ``CancelledError`` (not convert to a domain error)
    so this rewrite still fires. Verified by wrapping the call in
    ``asyncio.timeout`` and asserting the outer ``TimeoutError``.
    """
    c = _stub_client()

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await c._request_unguarded("slow/method", {}, timeout=60)

    # Pending dict still cleaned up despite the outer timeout pathway.
    assert c._pending == {}


@pytest.mark.asyncio
async def test_request_unguarded_timeout_still_raises_typed_error():
    """The inner ``asyncio.wait_for`` timeout branch is unchanged —
    a wait_for timeout is *not* a cancel and continues to raise the
    typed ``CodexAppServerError`` so callers can distinguish.
    """
    c = _stub_client()
    with pytest.raises(CodexAppServerError, match="timed out"):
        await c._request_unguarded("slow/method", {}, timeout=0.01)
    assert c._pending == {}


@pytest.mark.asyncio
async def test_request_unguarded_normal_response_unaffected():
    """Happy path: dispatch resolves the fut, no cancellation involved,
    no behavior change. Guards against accidental regression in the
    cancellation cleanup path.
    """
    c = _stub_client()
    task = asyncio.create_task(
        c._request_unguarded("ping", {}, timeout=60)
    )
    await asyncio.sleep(0)
    mid = next(iter(c._pending))
    c._dispatch({"id": mid, "result": {"pong": True}})
    assert await task == {"pong": True}


@pytest.mark.asyncio
async def test_auth_token_firing_during_handshake_raises_typed_cancel():
    """Interactive auth cancellation is translated only when the
    explicit auth token fires. The pending initialize RPC is cleaned up
    and the caller gets a domain error suitable for one-line CLI output.
    """
    c = _stub_client()
    token = AuthCancellationToken()

    task = asyncio.create_task(c._handshake(cancellation_token=token))
    await asyncio.sleep(0)
    assert len(c._pending) == 1

    token.cancel()

    with pytest.raises(CodexAppServerCancelled, match="login cancelled"):
        await task
    assert c._pending == {}


@pytest.mark.asyncio
async def test_auth_token_firing_during_token_refresh_raises_typed_cancel(
    tmp_path, monkeypatch
):
    """The OAuth refresh path also honors the explicit login abort
    token while preserving the raw-cancellation behavior for non-auth
    callers.
    """
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "old-access",
                    "refresh_token": "refresh-me",
                    "account_id": "acct_123",
                },
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    import httpx

    class SlowAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            await asyncio.sleep(60)

    monkeypatch.setattr(httpx, "AsyncClient", SlowAsyncClient)

    c = _stub_client()
    token = AuthCancellationToken()
    task = asyncio.create_task(
        c._refresh_chatgpt_tokens(cancellation_token=token)
    )
    await asyncio.sleep(0)

    token.cancel()

    with pytest.raises(CodexAppServerCancelled, match="login cancelled"):
        await task
