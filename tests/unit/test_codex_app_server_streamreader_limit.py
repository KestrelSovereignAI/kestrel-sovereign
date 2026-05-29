"""Codex app-server spawn must use a buffer ``limit`` large enough for
typical JSON-RPC frames (#1438).

asyncio's default ``StreamReader`` line limit is 64 KiB. Codex routinely
emits single-line JSON-RPC frames that exceed this — e.g. an
Item-finished event echoing the full assistant text on a ~20K-input-
token turn, or a snapshot containing the cumulative reasoning trace.
When the limit is exceeded, ``StreamReader.__anext__`` raises
``ValueError('Separator is found, but chunk is longer than limit')``;
``_read_loop`` dies before any frame is parsed; the call site reports
``codex app-server exited (rc=None)`` with no clue that a buffer cap
was the cause.

Pin the spawn kwarg so a future refactor can't quietly drop it.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.llm.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
)


class _AsyncIterableMock:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


@pytest.mark.asyncio
async def test_spawn_passes_large_streamreader_limit(monkeypatch):
    """``_spawn`` must pass a ``limit=`` of at least 1 MiB to
    ``asyncio.create_subprocess_exec``. Default (64 KiB) crashes on
    realistic codex turn frames — see #1438."""

    c = CodexAppServerClient.__new__(CodexAppServerClient)
    c._binary = "/usr/bin/true"
    c._closed_error = None
    c._pending = {}
    c._turn_sinks = {}
    c._stderr_tail = []

    captured = {}

    fake_proc = MagicMock()
    fake_proc.stdout = _AsyncIterableMock([])
    fake_proc.stderr = _AsyncIterableMock([])
    fake_proc.stdin = MagicMock()

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", fake_create_subprocess_exec,
    )

    await c._spawn()

    assert "limit" in captured["kwargs"], (
        "_spawn must pass an explicit `limit=` to create_subprocess_exec — "
        "the default 64 KiB crashes the read loop on realistic codex "
        "frames (#1438). Did a refactor drop the kwarg?"
    )
    limit = captured["kwargs"]["limit"]
    assert limit >= 1024 * 1024, (
        f"`limit={limit}` is too low. Codex frames routinely exceed 1 MiB "
        f"on typical turn payloads; spawning with this limit would crash "
        f"the read loop with `ValueError('Separator is found, but chunk "
        f"is longer than limit')` and report a spurious `app-server exited "
        f"(rc=None)`. See #1438."
    )

    # Cleanup
    if c._reader_task:
        c._reader_task.cancel()
    if c._stderr_task:
        c._stderr_task.cancel()


@pytest.mark.asyncio
async def test_read_loop_logs_streamreader_limit_explicitly(monkeypatch, caplog):
    """If ``_read_loop`` ever hits the StreamReader limit (regression
    of the spawn arg, or a frame even bigger than 16 MiB), the server
    log must say "asyncio StreamReader limit" rather than letting the
    error masquerade as a clean exit."""

    c = CodexAppServerClient.__new__(CodexAppServerClient)
    c._pending = {}
    c._turn_sinks = {}
    c._stderr_tail = []
    c._closed_error = None
    c._initialized = True
    c._reader_task = None
    c._stderr_task = None

    # Mock proc whose stdout raises the exact asyncio limit error.
    class _OverflowStdout:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ValueError("Separator is found, but chunk is longer than limit")

    fake_proc = MagicMock()
    fake_proc.stdout = _OverflowStdout()
    fake_proc.stderr = _AsyncIterableMock([])
    fake_proc.returncode = None

    async def fake_wait():
        return 0

    fake_proc.wait = fake_wait
    c._proc = fake_proc

    with caplog.at_level(logging.ERROR, logger="kestrel_sovereign.llm.codex_app_server"):
        await c._read_loop()

    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "asyncio StreamReader limit" in messages, (
        f"Expected a log line naming the StreamReader limit so operators "
        f"can grep for it. Got log messages: {messages!r}"
    )
