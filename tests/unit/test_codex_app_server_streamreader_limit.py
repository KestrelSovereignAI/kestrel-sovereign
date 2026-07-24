"""Codex app-server stdout framing stays bounded without line-reader failure.

Codex can emit one large newline-delimited JSON-RPC event for a reasoning
snapshot.  ``StreamReader.readline`` treats its ``limit`` as a per-line cap,
so a single valid frame used to kill the shared bridge reader.  These tests
pin incremental framing: valid frames beyond the legacy 16 MiB cap parse,
and a genuinely oversized frame fails only active bridge work while the
reader resynchronizes for later output (#2711).
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.llm.codex_app_server import (
    CODEX_APP_SERVER_MAX_FRAME_BYTES,
    CodexAppServerClient,
    CodexAppServerFrameTooLarge,
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


class _ChunkReader:
    """In-memory ``StreamReader.read`` stand-in with optional held EOF."""

    def __init__(self, data: bytes, *, hold_eof: bool = False):
        self._data = data
        self._release_eof = asyncio.Event()
        if not hold_eof:
            self._release_eof.set()

    async def read(self, size: int = -1) -> bytes:
        if self._data:
            if size < 0:
                out, self._data = self._data, b""
            else:
                out, self._data = self._data[:size], self._data[size:]
            return out
        await self._release_eof.wait()
        return b""

    def release_eof(self) -> None:
        self._release_eof.set()


@pytest.mark.asyncio
async def test_spawn_passes_large_streamreader_limit(monkeypatch, tmp_path):
    """Spawn retains an explicit, frame-ceiling-aligned reader limit."""
    monkeypatch.setenv("HOME", str(tmp_path))

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
    assert limit == CODEX_APP_SERVER_MAX_FRAME_BYTES, (
        "the subprocess reader limit must stay aligned with the bounded "
        "JSON-RPC frame ceiling; a lower limit reintroduces the #2711 "
        "large-frame failure mode"
    )

    # Cleanup
    if c._reader_task:
        c._reader_task.cancel()
    if c._stderr_task:
        c._stderr_task.cancel()


@pytest.mark.asyncio
async def test_read_loop_accepts_frame_larger_than_legacy_limit():
    """A >16 MiB valid item frame is parsed instead of killing the loop."""

    c = CodexAppServerClient.__new__(CodexAppServerClient)
    c._pending = {}
    sink: asyncio.Queue = asyncio.Queue()
    c._turn_sinks = {"thread-1": sink}
    c._stderr_tail = []
    c._closed_error = None
    c._initialized = True
    c._reader_task = None
    c._stderr_task = None

    payload = "x" * (16 * 1024 * 1024 + 1)
    raw = (json.dumps({
        "method": "turn/completed",
        "params": {"threadId": "thread-1", "payload": payload},
    }) + "\n").encode()

    fake_proc = MagicMock()
    fake_proc.stdout = _ChunkReader(raw)
    fake_proc.stderr = _AsyncIterableMock([])
    fake_proc.returncode = 0

    async def fake_wait():
        return 0

    fake_proc.wait = fake_wait
    c._proc = fake_proc

    await c._read_loop()

    message = sink.get_nowait()
    assert message["method"] == "turn/completed"
    assert len(message["params"]["payload"]) == len(payload)


@pytest.mark.asyncio
async def test_oversized_frame_fails_active_work_but_reader_resynchronizes(
    monkeypatch, caplog,
):
    """An over-ceiling frame is discarded through its newline, not fatal."""
    import kestrel_sovereign.llm.codex_app_server as codex_app_server

    max_frame_bytes = 1024 * 1024
    monkeypatch.setattr(
        codex_app_server, "CODEX_APP_SERVER_MAX_FRAME_BYTES", max_frame_bytes,
    )
    c = CodexAppServerClient.__new__(CodexAppServerClient)
    pending = asyncio.get_running_loop().create_future()
    c._pending = {7: pending}
    sink: asyncio.Queue = asyncio.Queue()
    c._turn_sinks = {"thread-1": sink}
    c._stderr_tail = []
    c._closed_error = None
    c._initialized = True
    c._reader_task = None
    c._stderr_task = None

    oversized = (json.dumps({
        "method": "item/completed",
        "params": {"threadId": "thread-1", "payload": "x" * max_frame_bytes},
    }) + "\n").encode()
    next_frame = (json.dumps({
        "method": "turn/completed",
        "params": {"threadId": "thread-1"},
    }) + "\n").encode()

    fake_proc = MagicMock()
    fake_proc.stdout = _ChunkReader(oversized + next_frame, hold_eof=True)
    fake_proc.stderr = _AsyncIterableMock([])
    fake_proc.returncode = 0

    async def fake_wait():
        return 0

    fake_proc.wait = fake_wait
    c._proc = fake_proc

    with caplog.at_level(
        logging.ERROR, logger="kestrel_sovereign.llm.codex_app_server",
    ):
        task = asyncio.create_task(c._read_loop())
        bridge_error = await asyncio.wait_for(sink.get(), timeout=2)
        next_message = await asyncio.wait_for(sink.get(), timeout=2)

    assert isinstance(bridge_error["__bridge_error__"], CodexAppServerFrameTooLarge)
    with pytest.raises(CodexAppServerFrameTooLarge, match="frame exceeded"):
        pending.result()
    assert next_message["method"] == "turn/completed"
    assert not task.done(), "the reader must remain alive after discarding one frame"
    assert c._proc is fake_proc
    assert "discarded an oversized JSON-RPC frame" in caplog.text

    fake_proc.stdout.release_eof()
    await task
