"""Tests for the codex app-server JSON-RPC client.

Pure-logic tests run everywhere; the live test spawns the real
``codex app-server`` binary and is skipped when it (or auth) is absent.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.llm.codex_app_server import (
    MIN_CODEX_APP_SERVER_VERSION,
    CodexAppServerClient,
    CodexAppServerConnectionClosed,
    CodexAppServerError,
    _parse_user_agent_version,
    _version_tuple,
    resolve_codex_binary,
)


class TestVersionGate:
    def test_user_agent_parse(self):
        assert _parse_user_agent_version(
            "kestrel/0.131.0-alpha.9 (Mac OS 26.1.0; arm64) unknown"
        ) == "0.131.0-alpha.9"
        assert _parse_user_agent_version("codex/0.130.0") == "0.130.0"
        assert _parse_user_agent_version("garbage") is None

    def test_numeric_dominates(self):
        assert _version_tuple("0.131.0") > _version_tuple("0.125.0")
        assert _version_tuple("0.124.9") < _version_tuple("0.125.0")

    def test_prerelease_ranks_below_same_numeric_stable(self):
        assert _version_tuple("0.131.0-alpha.9") < _version_tuple("0.131.0")

    def test_alpha_still_clears_lower_floor(self):
        assert _version_tuple("0.131.0-alpha.9") >= _version_tuple(
            MIN_CODEX_APP_SERVER_VERSION
        )


class TestBinaryResolution:
    def test_env_override_wins(self, tmp_path):
        fake = tmp_path / "codex"
        fake.write_text("#!/bin/sh\n")
        with patch.dict(os.environ, {"KESTREL_CODEX_APP_SERVER_BIN": str(fake)}):
            assert resolve_codex_binary() == str(fake)

    def test_env_override_missing_path_raises(self):
        with patch.dict(
            os.environ, {"KESTREL_CODEX_APP_SERVER_BIN": "/no/such/codex"}
        ):
            with pytest.raises(CodexAppServerError, match="does not exist"):
                resolve_codex_binary()

    def test_raises_with_hint_when_unfound(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("kestrel_sovereign.llm.codex_app_server.Path.exists",
                   return_value=False), \
             patch("shutil.which", return_value=None):
            with pytest.raises(CodexAppServerError, match="off PATH"):
                resolve_codex_binary()


class TestDispatchLogic:
    """_dispatch routing without a real subprocess."""

    def _client(self):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._pending = {}
        c._turn_sinks = {}
        c._server_request_handlers = {}
        c._inflight_server_requests = {}
        c._closed_error = None
        c._sent = []
        c._send = lambda obj: c._sent.append(obj)
        return c

    @pytest.mark.asyncio
    async def test_response_resolves_pending_future(self):
        import asyncio

        c = self._client()
        fut = asyncio.get_running_loop().create_future()
        c._pending[5] = fut
        c._dispatch({"id": 5, "result": {"ok": True}})
        assert await fut == {"ok": True}

    @pytest.mark.asyncio
    async def test_error_response_raises_on_future(self):
        import asyncio

        c = self._client()
        fut = asyncio.get_running_loop().create_future()
        c._pending[6] = fut
        c._dispatch({"id": 6, "error": {"message": "nope", "code": 42}})
        with pytest.raises(CodexAppServerError, match="nope"):
            await fut

    @pytest.mark.asyncio
    async def test_threaded_notification_routes_only_to_owning_sink(self):
        """Concurrent turns must not cross-contaminate."""
        import asyncio

        c = self._client()
        qa: asyncio.Queue = asyncio.Queue()
        qb: asyncio.Queue = asyncio.Queue()
        c._turn_sinks["thrA"] = qa
        c._turn_sinks["thrB"] = qb
        c._dispatch({"method": "item/agentMessage/delta",
                     "params": {"threadId": "thrA", "delta": "for-A"}})
        c._dispatch({"method": "turn/completed",
                     "params": {"threadId": "thrB"}})
        assert qa.get_nowait()["params"]["delta"] == "for-A"
        assert qa.empty(), "turn B's completion leaked into turn A"
        assert qb.get_nowait()["method"] == "turn/completed"
        assert qb.empty(), "turn A's delta leaked into turn B"

    @pytest.mark.asyncio
    async def test_threadless_global_notification_broadcasts(self):
        import asyncio

        c = self._client()
        qa: asyncio.Queue = asyncio.Queue()
        c._turn_sinks["thrA"] = qa
        c._dispatch({"method": "remoteControl/status/changed",
                     "params": {"status": "disabled"}})
        assert qa.get_nowait()["method"] == "remoteControl/status/changed"


class TestServerRequestHandlerRegistration:
    """Per-method async handlers + default fallbacks."""

    def _client(self):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._pending = {}
        c._turn_sinks = {}
        c._server_request_handlers = {}
        c._inflight_server_requests = {}
        c._closed_error = None
        c._sent = []
        c._send = lambda obj: c._sent.append(obj)
        return c

    @pytest.mark.asyncio
    async def test_unscoped_handler_invoked_for_any_thread(self):
        c = self._client()
        seen = []

        async def handler(params):
            seen.append(params)
            return {"ok": True}

        c.register_server_request_handler("item/tool/call", handler)
        await c._handle_server_request(7, "item/tool/call",
                                       {"threadId": "anything", "tool": "t"})
        assert seen and seen[0]["tool"] == "t"
        assert c._sent == [{"id": 7, "result": {"ok": True}}]

    @pytest.mark.asyncio
    async def test_thread_scoped_handler_only_fires_for_matching_thread(self):
        """Concurrent turns must each get their own handler."""
        c = self._client()
        seen_A, seen_B = [], []

        async def hA(params):
            seen_A.append(params)
            return {"thread": "A"}

        async def hB(params):
            seen_B.append(params)
            return {"thread": "B"}

        c.register_server_request_handler("item/tool/call", hA, thread_id="thrA")
        c.register_server_request_handler("item/tool/call", hB, thread_id="thrB")

        await c._handle_server_request(11, "item/tool/call",
                                       {"threadId": "thrA"})
        await c._handle_server_request(12, "item/tool/call",
                                       {"threadId": "thrB"})

        assert seen_A and seen_A[0]["threadId"] == "thrA"
        assert seen_B and seen_B[0]["threadId"] == "thrB"
        assert c._sent[0]["result"] == {"thread": "A"}
        assert c._sent[1]["result"] == {"thread": "B"}

    @pytest.mark.asyncio
    async def test_thread_scoped_unregister_does_not_remove_others(self):
        c = self._client()

        async def hA(p):
            return {"thread": "A"}

        async def hB(p):
            return {"thread": "B"}

        unA = c.register_server_request_handler(
            "item/tool/call", hA, thread_id="thrA",
        )
        c.register_server_request_handler(
            "item/tool/call", hB, thread_id="thrB",
        )
        unA()
        await c._handle_server_request(20, "item/tool/call",
                                       {"threadId": "thrA"})
        await c._handle_server_request(21, "item/tool/call",
                                       {"threadId": "thrB"})
        # thrA fell back to the explicit-failure default; thrB still handled.
        assert c._sent[0]["result"]["success"] is False
        assert c._sent[1]["result"] == {"thread": "B"}

    @pytest.mark.asyncio
    async def test_unregister_removes_handler(self):
        c = self._client()

        async def handler(params):
            return {"ok": True}

        unreg = c.register_server_request_handler("item/tool/call", handler)
        unreg()
        await c._handle_server_request(8, "item/tool/call", {})
        # Falls back to the explicit-failure for item/tool/call.
        assert c._sent[0]["result"]["success"] is False

    @pytest.mark.asyncio
    async def test_approval_request_auto_declines(self):
        c = self._client()
        await c._handle_server_request(
            9, "item/commandExecution/requestApproval", {}
        )
        assert c._sent == [{"id": 9, "result": {"decision": "decline"}}]

    @pytest.mark.asyncio
    async def test_unregistered_tool_call_explicit_failure(self):
        c = self._client()
        await c._handle_server_request(10, "item/tool/call", {})
        assert c._sent[0]["result"]["success"] is False
        assert "did not register" in c._sent[0]["result"]["contentItems"][0]["text"]

    @pytest.mark.asyncio
    async def test_handler_exception_becomes_error_response(self):
        c = self._client()

        async def handler(params):
            raise RuntimeError("kaboom")

        c.register_server_request_handler("item/tool/call", handler)
        await c._handle_server_request(13, "item/tool/call", {})
        assert "error" in c._sent[0]
        assert "kaboom" in c._sent[0]["error"]["message"]


@pytest.mark.asyncio
class TestTurnIteration:
    async def test_iter_stops_on_turn_completed(self):
        import asyncio

        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._closed_error = None
        q: asyncio.Queue = asyncio.Queue()
        for ev in [
            {"method": "item/agentMessage/delta", "params": {"delta": "a"}},
            {"method": "turn/completed", "params": {}},
            {"method": "item/agentMessage/delta", "params": {"delta": "AFTER"}},
        ]:
            q.put_nowait(ev)
        got = [ev async for ev in c.iter_turn_events(q, idle_timeout=2)]
        assert [e["method"] for e in got] == [
            "item/agentMessage/delta", "turn/completed"
        ]

    async def test_iter_raises_on_closed(self):
        import asyncio

        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._closed_error = CodexAppServerError("gone")
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait({"__closed__": True})
        with pytest.raises(CodexAppServerError, match="gone"):
            async for _ in c.iter_turn_events(q, idle_timeout=2):
                pass


@pytest.mark.asyncio
class TestIdleWatchdogInflightServerRequest:
    """A long-running server→client request (a Talon tool callback that
    runs for minutes, an approval parked on a human) must not trip the
    idle watchdog: the app-server is correctly silent while it awaits
    OUR reply, not stalled upstream. See ``_inflight_server_requests``.
    """

    def _client(self):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._server_request_handlers = {}
        c._inflight_server_requests = {}
        c._turn_sinks = {}
        c._closed_error = None
        # Idle-raise diagnostics tails — empty/absent on this bare client.
        c._stderr_tail = []
        c._codex_home = None
        c._sent = []
        c._send = lambda obj: c._sent.append(obj)
        return c

    async def test_handler_tracks_inflight_count(self):
        """Counter is positive while the handler runs and released after,
        including when the handler raises."""
        c = self._client()
        observed = {}

        async def slow(params):
            observed["mid_run"] = c._inflight_server_requests.get("thr", 0)
            return {"ok": True}

        c.register_server_request_handler(
            "item/tool/call", slow, thread_id="thr"
        )
        await c._handle_server_request(1, "item/tool/call",
                                       {"threadId": "thr"})
        assert observed["mid_run"] == 1
        # Released (key dropped) once the handler completes.
        assert c._inflight_server_requests.get("thr", 0) == 0

        async def boom(params):
            raise RuntimeError("tool blew up")

        c.register_server_request_handler(
            "item/tool/call", boom, thread_id="thr"
        )
        await c._handle_server_request(2, "item/tool/call",
                                       {"threadId": "thr"})
        assert c._inflight_server_requests.get("thr", 0) == 0

    async def test_idle_rearms_while_request_in_flight_then_completes(self):
        import asyncio

        c = self._client()
        q: asyncio.Queue = asyncio.Queue()
        # App-server is blocked awaiting our reply on this thread.
        c._inflight_server_requests = {"thr": 1}

        async def drive():
            return [
                ev async for ev in c.iter_turn_events(
                    q, idle_timeout=0.05, thread_id="thr"
                )
            ]

        task = asyncio.create_task(drive())
        # Let the watchdog time out and re-arm several times while the
        # in-flight request is unresolved.
        await asyncio.sleep(0.2)
        assert not task.done()
        # Tool callback returns → app-server resumes → completion arrives.
        # Deliver the terminating event WITHOUT first clearing the
        # in-flight count: a received event ends the loop regardless of
        # in-flight state, and popping first would open a race where a
        # 0.05s timeout fires in the gap (count now 0) and raises before
        # the event lands — the spurious CI failure this guards against.
        q.put_nowait({"method": "turn/completed", "params": {}})
        got = await asyncio.wait_for(task, timeout=2)
        assert [e["method"] for e in got] == ["turn/completed"]

    async def test_completion_pushes_wake_marker_to_thread_sink(self):
        import asyncio

        c = self._client()
        q: asyncio.Queue = asyncio.Queue()
        c._turn_sinks = {"thr": q}

        async def ok(params):
            return {"ok": True}

        c.register_server_request_handler("item/tool/call", ok,
                                          thread_id="thr")
        await c._handle_server_request(1, "item/tool/call",
                                       {"threadId": "thr"})
        assert q.get_nowait() == {"__inflight_done__": True}

    async def test_completion_broadcasts_wake_marker_when_threadless(self):
        import asyncio

        c = self._client()
        qa: asyncio.Queue = asyncio.Queue()
        qb: asyncio.Queue = asyncio.Queue()
        c._turn_sinks = {"a": qa, "b": qb}
        # Thread-less approval-style request: no handler, default reply.
        await c._handle_server_request(1, "mcpServer/elicitation/request", {})
        assert qa.get_nowait() == {"__inflight_done__": True}
        assert qb.get_nowait() == {"__inflight_done__": True}

    async def test_budget_restarts_on_completion_marker(self):
        """A callback that finishes late in an idle window must not raise
        at the next boundary: the wake marker restarts a full budget."""
        import asyncio

        c = self._client()
        q: asyncio.Queue = asyncio.Queue()
        c._inflight_server_requests = {"thr": 1}

        async def drive():
            return [
                ev async for ev in c.iter_turn_events(
                    q, idle_timeout=0.05, thread_id="thr"
                )
            ]

        task = asyncio.create_task(drive())
        # Several intervals elapse while the request runs — re-armed.
        await asyncio.sleep(0.2)
        assert not task.done()
        # Request completes late in a window: the marker wakes the
        # watchdog and the real next event follows. We inject both into
        # the sink directly (this unit-tests iter_turn_events' marker
        # handling, independent of in-flight bookkeeping), so no pop —
        # which would otherwise race the 0.05s timeout.
        q.put_nowait({"__inflight_done__": True})
        q.put_nowait({"method": "turn/completed", "params": {}})
        got = await asyncio.wait_for(task, timeout=2)
        # Marker skipped, not yielded; completion delivered.
        assert [e["method"] for e in got] == ["turn/completed"]

    async def test_idle_rearms_for_threadless_request(self):
        """Sandbox-approval RPCs arrive without ``threadId`` and are
        tracked under the ``None`` key. A turn watcher must still treat
        them as in-flight, or a long human approval trips the watchdog.
        """
        import asyncio

        c = self._client()
        q: asyncio.Queue = asyncio.Queue()
        # Approval in flight, recorded thread-less.
        c._inflight_server_requests = {None: 1}

        async def drive():
            return [
                ev async for ev in c.iter_turn_events(
                    q, idle_timeout=0.05, thread_id="thr"
                )
            ]

        task = asyncio.create_task(drive())
        await asyncio.sleep(0.2)
        assert not task.done()
        # Terminating event ends the loop regardless of in-flight state;
        # no pop first (avoids racing the 0.05s timeout — see
        # test_idle_rearms_while_request_in_flight_then_completes).
        q.put_nowait({"method": "turn/completed", "params": {}})
        got = await asyncio.wait_for(task, timeout=2)
        assert [e["method"] for e in got] == ["turn/completed"]

    async def test_idle_still_raises_when_nothing_in_flight(self):
        import asyncio

        c = self._client()
        q: asyncio.Queue = asyncio.Queue()
        # No in-flight request → a silent app-server IS an upstream stall.
        with pytest.raises(CodexAppServerError, match="idle for"):
            async for _ in c.iter_turn_events(
                q, idle_timeout=0.05, thread_id="thr"
            ):
                pass


class TestInvoluntaryExitRecovery:
    """When the codex app-server process exits unexpectedly (panic on
    duplicate-handler registration, segfault, OOM-kill, ENV-toxic
    binary), the client must reset its state so a SUBSEQUENT request
    can spawn a fresh process. Without the reset, ``_initialized``
    stuck at True + dead ``_proc`` meant every request after the
    crash kept failing with the OLD instance's CONNECTION_CLOSED
    error until kestrel itself was restarted — observed live when
    codex panicked on ``spawn_agent`` namespace collision (#1334
    follow-up; see commit history)."""

    @pytest.mark.asyncio
    async def test_read_loop_exit_does_not_leak_stderr_to_exception(self, caplog):
        """#1412: when codex-rs exits, the stderr ring buffer is logged
        server-side at ERROR level but kept OUT of the
        ``CodexAppServerConnectionClosed`` text. Same leak boundary as
        the idle-timeout path established in #1410 — the exception
        propagates to chat callers (via ``_fail_all`` -> pending
        futures and ``endpoints/agent.py``), so cross-session stderr
        content must not surface in the user-visible error.
        """
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._proc = MagicMock()
        c._proc.returncode = 137  # OOM-kill style code, distinctive
        c._proc.stdout = _AsyncIterableMock([])

        async def _fake_wait():
            return 137
        c._proc.wait = _fake_wait

        c._initialized = True
        # Pending future will receive the exception via _fail_all.
        import asyncio as _asyncio
        loop = _asyncio.get_event_loop()
        pending_fut = loop.create_future()
        c._pending = {1: pending_fut}
        c._turn_sinks = {}
        c._stderr_tail = [
            "TRACE codex_protocol: prior session token=secret_xyz",
            "INFO  codex_login: auth refreshed for user_42",
        ]
        c._closed_error = None

        import logging
        caplog.set_level(logging.ERROR, logger="kestrel_sovereign.llm.codex_app_server")
        await c._read_loop()

        # Exception text is framework-owned: rc + nothing else.
        with pytest.raises(CodexAppServerConnectionClosed) as ei:
            pending_fut.result()
        msg = str(ei.value)
        assert msg == "codex app-server exited (rc=137)"
        # The stderr lines must NOT have leaked into the chat-facing text.
        assert "secret_xyz" not in msg
        assert "user_42" not in msg
        assert "auth refreshed" not in msg

        # The same lines DO appear in server logs for operator diagnosis.
        log_text = " ".join(rec.message for rec in caplog.records)
        assert "secret_xyz" in log_text
        assert "auth refreshed" in log_text
        assert "rc=137" in log_text

    @pytest.mark.asyncio
    async def test_read_loop_exit_with_no_stderr_emits_no_diagnostic_log(self, caplog):
        """Negative case: when the ring buffer is empty, no stderr-tail
        log line fires — keeps the happy-path log surface clean."""
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._proc = MagicMock()
        c._proc.returncode = 0
        c._proc.stdout = _AsyncIterableMock([])

        async def _fake_wait():
            return 0
        c._proc.wait = _fake_wait

        c._initialized = True
        import asyncio as _asyncio
        loop = _asyncio.get_event_loop()
        pending_fut = loop.create_future()
        c._pending = {1: pending_fut}
        c._turn_sinks = {}
        c._stderr_tail = []
        c._closed_error = None

        import logging
        caplog.set_level(logging.ERROR, logger="kestrel_sovereign.llm.codex_app_server")
        await c._read_loop()

        # Exception still raises cleanly with rc.
        with pytest.raises(CodexAppServerConnectionClosed) as ei:
            pending_fut.result()
        assert str(ei.value) == "codex app-server exited (rc=0)"
        # No "exit stderr tail" log line when there's nothing to report.
        diag_records = [
            r for r in caplog.records if "exit stderr tail" in r.message
        ]
        assert diag_records == []

    @pytest.mark.asyncio
    async def test_read_loop_exit_resets_initialized_and_proc(self):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._proc = MagicMock()
        c._proc.returncode = 0
        c._proc.stdout = _AsyncIterableMock([])  # ends immediately

        # ``_read_loop`` now awaits ``_proc.wait()`` (with a 1s timeout)
        # to capture the real return code before reporting the exit —
        # the prior version reported ``rc=None`` because the process
        # hadn't been reaped, masking whether codex was signaled vs
        # exited normally (#1399). Mock as a coroutine so the await
        # resolves.
        async def _fake_wait():
            return 0
        c._proc.wait = _fake_wait

        c._initialized = True
        c._pending = {}
        c._turn_sinks = {}
        c._stderr_tail = []
        c._closed_error = None

        await c._read_loop()

        assert c._initialized is False, (
            "after the read loop exits, ``_initialized`` must reset so "
            "the next ``ensure_started`` spawns a fresh process"
        )
        assert c._proc is None, (
            "_proc must clear after involuntary exit so the next spawn "
            "writes a fresh handle"
        )

    @pytest.mark.asyncio
    async def test_spawn_clears_prior_instance_closed_error(self, monkeypatch):
        import asyncio

        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._binary = "/usr/bin/true"  # cheap, exits immediately
        c._closed_error = CodexAppServerError("prior-instance gone")
        c._pending = {}
        c._turn_sinks = {}
        c._stderr_tail = ["old stderr line"]

        # Replace create_subprocess_exec with a stub so we don't shell
        # out — we only care that ``_spawn`` clears the prior error.
        fake_proc = MagicMock()
        fake_proc.stdout = _AsyncIterableMock([])
        fake_proc.stderr = _AsyncIterableMock([])
        fake_proc.stdin = MagicMock()

        async def fake_create_subprocess_exec(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", fake_create_subprocess_exec,
        )

        await c._spawn()

        assert c._closed_error is None, (
            "_spawn must clear the prior instance's closed_error so the "
            "next request doesn't see the OLD process's exit reported"
        )
        # cleanup tasks created by _spawn
        if c._reader_task:
            c._reader_task.cancel()
        if c._stderr_task:
            c._stderr_task.cancel()


class _AsyncIterableMock:
    """Minimal async-iterable for tests — yields bytes-lines and ends."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


# Live handshake against the real binary lives in
# ``tests/integration/test_codex_real.py``; this unit module is pure
# in-memory dispatch logic, no subprocesses.
