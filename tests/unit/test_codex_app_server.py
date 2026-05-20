"""Tests for the codex app-server JSON-RPC client.

Pure-logic tests run everywhere; the live test spawns the real
``codex app-server`` binary and is skipped when it (or auth) is absent.
"""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.codex_app_server import (
    MIN_CODEX_APP_SERVER_VERSION,
    CodexAppServerClient,
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


_BIN = "/Applications/Codex.app/Contents/Resources/codex"
_HAVE_BIN = Path(os.environ.get("KESTREL_CODEX_APP_SERVER_BIN", _BIN)).exists()
_HAVE_AUTH = (Path.home() / ".codex" / "auth.json").exists()


@pytest.mark.skipif(
    not (_HAVE_BIN and _HAVE_AUTH),
    reason="codex binary and ~/.codex/auth.json required for live test",
)
@pytest.mark.asyncio
async def test_live_handshake_and_model_list():
    """Real binary: handshake, version-gate, model/list returns a catalog."""
    c = CodexAppServerClient()
    try:
        await c.ensure_started()
        result = await c.request(
            "model/list", {"limit": 3, "cursor": None, "includeHidden": None},
            timeout=30,
        )
        ids = [m.get("id") for m in (result or {}).get("data", [])]
        assert ids, f"expected models, got {result!r}"
    finally:
        await c.aclose()
