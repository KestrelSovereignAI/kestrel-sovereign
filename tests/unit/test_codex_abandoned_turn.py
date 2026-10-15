"""#3552: a Codex turn the caller abandons does not poison the session.

The orchestrator's inactivity watchdog (#3552) fires at 180s, before the codex
transport's own 300s idle timeout. Its trip cancels the call, which unwinds
``CodexAdapter._run_turn`` with ``CancelledError`` and so skipped the idle-
timeout handler that drops the session's cached thread. The app-server was
still running the abandoned turn on that thread, so the session's next
request reused it and had its ``turn/start`` rejected.

Any abandonment of a started turn now invalidates the thread before its lock
is released: the watchdog (streamed or not), a Stop cancelling the task, the
request's cancel token, or the consumer closing the stream. A turn the
app-server ended keeps its thread, and so does a call cancelled before its
turn started.

The adapter, its retry wrapper and its thread cache are real; only the
app-server process is replaced, by one that rejects ``turn/start`` on a thread
whose turn is still running, as the real one does.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from kestrel_sovereign.agent import orchestrator_engine as oe
from kestrel_sovereign.agent.orchestrator_engine import (
    LLMCallInactivityTimeout,
    LLMCallWatchdog,
)
from kestrel_sovereign.llm.call_progress import AttemptInactivityTimeout
from kestrel_sovereign.llm.cancellation import raise_if_cancelled
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.codex_app_server import CodexAppServerError
from kestrel_sovereign.llm.error_handling import LLMError
from kestrel_sovereign.llm.output_ceiling import incomplete_generation
from tests.shared.booted_agent import booted_agent

#: A watchdog bound short enough to trip in a unit test.
BOUND = 0.2
SESSION = "s1"
MESSAGES = [{"role": "user", "content": "Where do I work?"}]


class FakeCodexAppServer:
    """The codex app-server, in-process.

    Each ``thread/start`` opens a new thread. Each ``turn/start`` takes the
    next script: ``"answer"`` streams one delta and ends the turn,
    ``"stall"`` never produces an event, and ``"partial"`` streams one delta
    and then never produces another. A turn that has not ended keeps its
    thread busy even after Kestrel stops listening, so a second ``turn/start``
    on that thread is rejected, as the real app-server rejects it.
    """

    def __init__(self, *scripts: str, answer: str = "Noted.") -> None:
        self._scripts = list(scripts)
        self.answer = answer
        self.threads: List[str] = []
        self.turns: List[str] = []
        self.rejected: List[str] = []
        self.turn_started = asyncio.Event()
        self._running: Dict[str, str] = {}

    async def ensure_started(self) -> None:
        return None

    async def aclose(self) -> None:
        self._running.clear()

    async def request(
        self, method: str, params: Optional[dict] = None, *, timeout: float = 120,
    ) -> Any:
        params = params or {}
        if method == "config/read":
            return {"config": {}}
        if method == "thread/start":
            thread_id = f"thread-{len(self.threads) + 1}"
            self.threads.append(thread_id)
            return {"thread": {"id": thread_id}}
        if method == "turn/start":
            thread_id = params["threadId"]
            if thread_id in self._running:
                self.rejected.append(thread_id)
                raise CodexAppServerError(
                    f"thread {thread_id} already has an active turn"
                )
            self._running[thread_id] = self._scripts.pop(0)
            self.turns.append(thread_id)
            self.turn_started.set()
            return {}
        raise AssertionError(f"unexpected app-server request {method!r}")

    def open_turn_sink(self, thread_id: str) -> str:
        return thread_id

    def close_turn_sink(self, thread_id: str) -> None:
        return None

    def register_server_request_handler(self, *_args: Any, **_kwargs: Any):
        return lambda: None

    async def _silence(self, cancel_token: Any) -> None:
        while True:
            await asyncio.sleep(0.01)
            raise_if_cancelled(cancel_token)

    async def iter_turn_events(
        self, sink: Any, *, thread_id: Optional[str] = None, cancel_token: Any = None,
    ):
        script = self._running[thread_id]
        if script == "stall":
            await self._silence(cancel_token)
        yield {
            "method": "item/agentMessage/delta",
            "params": {"delta": self.answer},
        }
        if script == "partial":
            await self._silence(cancel_token)
        del self._running[thread_id]
        yield {
            "method": "turn/completed",
            "params": {"turn": {"status": "completed"}},
        }


def _adapter(app: FakeCodexAppServer, monkeypatch, tmp_path) -> CodexAdapter:
    monkeypatch.setenv("KESTREL_CODEX_CWD", str(tmp_path))
    adapter = CodexAdapter()
    adapter._client = app
    return adapter


async def _ask(adapter: CodexAdapter, **kwargs: Any):
    return await adapter.get_response(
        None, "auto", MESSAGES, session_id=SESSION, **kwargs,
    )


async def _stream(adapter: CodexAdapter, **kwargs: Any) -> List[Any]:
    return [
        item async for item in adapter.get_streaming_response_with_tools(
            None, "auto", MESSAGES, session_id=SESSION, **kwargs,
        )
    ]


async def _watchdog_non_streaming(adapter: CodexAdapter, app) -> None:
    with pytest.raises(LLMCallInactivityTimeout):
        await LLMCallWatchdog(BOUND).call(_ask(adapter))


async def _watchdog_streaming(adapter: CodexAdapter, app) -> None:
    async def watched() -> None:
        async for _ in LLMCallWatchdog(BOUND).watch(
            adapter.get_streaming_response_with_tools(
                None, "auto", MESSAGES, session_id=SESSION,
            )
        ):
            pass

    with pytest.raises(LLMCallInactivityTimeout):
        await watched()


async def _stop(adapter: CodexAdapter, app) -> None:
    task = asyncio.create_task(_ask(adapter))
    await asyncio.wait_for(app.turn_started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _cancel_token(adapter: CodexAdapter, app) -> None:
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(
            _ask(adapter, cancel_token=app.turn_started.is_set), 5,
        )


async def _consumer_stops_reading(adapter: CodexAdapter, app) -> None:
    turn = adapter._run_turn("auto", MESSAGES, None, SESSION, None)
    assert await turn.__anext__() == {"text": app.answer}
    await turn.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("abandon, script", [
    pytest.param(_watchdog_non_streaming, "stall", id="watchdog-non-streaming"),
    pytest.param(_watchdog_streaming, "stall", id="watchdog-streaming"),
    pytest.param(_watchdog_streaming, "partial", id="watchdog-after-partial-text"),
    pytest.param(_stop, "stall", id="stop-cancels-the-task"),
    pytest.param(_cancel_token, "stall", id="cancel-token"),
    pytest.param(_consumer_stops_reading, "partial", id="consumer-stops-reading"),
])
async def test_an_abandoned_turn_sends_the_next_request_to_a_new_thread(
    abandon, script, monkeypatch, tmp_path,
):
    app = FakeCodexAppServer(script, "answer")
    adapter = _adapter(app, monkeypatch, tmp_path)

    await abandon(adapter, app)

    # The session's thread was dropped and its lock released.
    assert SESSION not in adapter._session_threads
    assert not adapter._thread_locks["thread-1"].locked()

    response = await asyncio.wait_for(_ask(adapter), 5)

    assert response.content == "Noted."
    assert app.threads == ["thread-1", "thread-2"]
    assert app.turns == ["thread-1", "thread-2"]
    assert app.rejected == []


@pytest.mark.asyncio
async def test_a_queued_same_session_request_does_not_get_the_abandoned_thread(
    monkeypatch, tmp_path,
):
    """A same-session request already waiting on the thread lock when the
    watchdog trips must see the invalidation once it holds the lock."""
    app = FakeCodexAppServer("stall", "answer")
    adapter = _adapter(app, monkeypatch, tmp_path)

    stalled = asyncio.create_task(LLMCallWatchdog(BOUND).call(_ask(adapter)))
    await asyncio.wait_for(app.turn_started.wait(), 5)
    queued = asyncio.create_task(_ask(adapter))
    await asyncio.sleep(0.05)
    assert app.threads == ["thread-1"]  # it resolved the cached thread

    with pytest.raises(LLMCallInactivityTimeout):
        await stalled
    response = await asyncio.wait_for(queued, 5)

    assert response.content == "Noted."
    assert app.turns == ["thread-1", "thread-2"]
    assert app.rejected == []


@pytest.mark.asyncio
@pytest.mark.parametrize("consume", [_ask, _stream], ids=["response", "stream"])
async def test_a_turn_the_app_server_ended_keeps_its_thread(
    consume, monkeypatch, tmp_path,
):
    """The control: without an abandonment a session's requests share one
    thread, so a new thread in the tests above is the invalidation's doing."""
    app = FakeCodexAppServer("answer", "answer")
    adapter = _adapter(app, monkeypatch, tmp_path)

    await consume(adapter)
    await consume(adapter)

    assert app.threads == ["thread-1"]
    assert app.turns == ["thread-1", "thread-1"]


@pytest.mark.asyncio
async def test_closing_a_turn_after_it_ended_keeps_its_thread(monkeypatch, tmp_path):
    app = FakeCodexAppServer("answer")
    adapter = _adapter(app, monkeypatch, tmp_path)

    turn = adapter._run_turn("auto", MESSAGES, None, SESSION, None)
    events = [await turn.__anext__(), await turn.__anext__()]
    await turn.aclose()

    assert "final" in events[-1]
    assert adapter._session_threads[SESSION][0] == "thread-1"


@pytest.mark.asyncio
async def test_a_call_cancelled_before_its_turn_started_keeps_the_thread(
    monkeypatch, tmp_path,
):
    """A Stop that reaches a request still queued behind its session's running
    turn abandons nothing on the app-server, so the thread and its history
    stay."""
    app = FakeCodexAppServer("partial", "answer")
    adapter = _adapter(app, monkeypatch, tmp_path)

    running = adapter._run_turn("auto", MESSAGES, None, SESSION, None)
    assert await running.__anext__() == {"text": app.answer}
    queued = asyncio.create_task(_ask(adapter))
    await asyncio.sleep(0.05)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    assert adapter._session_threads[SESSION][0] == "thread-1"
    assert app.turns == ["thread-1"]
    await running.aclose()


# --------------------------------------------------------------------------
# The turn: a stalled openai:plan turn trips the watchdog, the next takes a
# new thread
# --------------------------------------------------------------------------


def _plan_route(agent, app: FakeCodexAppServer, monkeypatch, tmp_path) -> dict:
    """The agent's only route: ``openai:plan`` built as the registry builds
    it, its app-server ``app``."""
    from kestrel_sovereign.llm import codex_app_server
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    monkeypatch.setenv("KESTREL_CODEX_CWD", str(tmp_path))
    monkeypatch.setattr(
        codex_app_server, "resolve_codex_binary", lambda: "/usr/bin/false",
    )
    info = ProviderRegistry({})._build_route(
        "openai", "plan", {"is_cloud": True}, {"adapter": "CodexAdapter"},
    )
    [provider] = agent.llm_service._convert_providers_format([info])
    provider["adapter"]._client = app
    service = agent.llm_service
    monkeypatch.setattr(service, "providers", [provider])
    monkeypatch.setattr(
        service, "config",
        {**(service.config or {}), "route_priority": ["openai:plan"]},
    )
    monkeypatch.setattr(service, "_mandate_preference", {})
    return provider


async def _streamed_turns(agent, monkeypatch) -> List[str]:
    """Answer, stall, then the same session's next turn; the stall must fail.

    The plan route is the call's only candidate: its abandoned attempt fails
    the call as its last route (#1968)."""
    from tests.unit.test_llm_call_watchdog import _conversation_holder, _turn

    answers = [await asyncio.wait_for(_turn(agent, "Hello."), 30)]
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", BOUND)
    with pytest.raises(LLMError) as stalled:
        await asyncio.wait_for(_turn(agent, "Where do I work?"), 30)
    assert isinstance(
        incomplete_generation(stalled.value), AttemptInactivityTimeout,
    )
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", 30)
    assert _conversation_holder(agent) is None
    answers.append(await asyncio.wait_for(_turn(agent, "And now?"), 30))
    return answers


async def _invoked_turns(agent, monkeypatch) -> List[str]:
    """The same three turns through ``/api/agent/invoke``."""
    from tests.unit.test_llm_call_watchdog import (
        _conversation_holder,
        _invoke,
        _invoke_client,
    )

    async with _invoke_client(agent) as client:
        first = await _invoke(client, "Hello.")
        monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", BOUND)
        stalled = await _invoke(client, "Where do I work?")
        assert stalled.status_code == 502, stalled.text
        assert stalled.json()["error"]["code"] == "generation_incomplete"
        assert _conversation_holder(agent) is None
        monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", 30)
        last = await _invoke(client, "And now?")
    for response in (first, last):
        assert response.status_code == 200, response.text
    return [first.json()["response"], last.json()["response"]]


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("turns", [_streamed_turns, _invoked_turns],
                         ids=["streamed", "invoke"])
async def test_a_stalled_plan_turn_times_out_and_the_next_starts_a_new_thread(
    turns, tmp_path, monkeypatch,
):
    """The review's reproduction, on a real agent. The session's first turn
    carries a one-time briefing, so the second starts its own thread; the
    third has the second's settings and would reuse its thread, where the
    abandoned turn is still running."""
    # A short deadline belongs to the deliberately silent turn only. A normal
    # first call may include lazy local setup and is not a 200ms performance
    # contract. The stall remains bounded and must invalidate its exact thread.
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", 30)
    app = FakeCodexAppServer("answer", "stall", "answer")
    async with booted_agent(
        tmp_path, name="Codex", provenance="test:codex_abandoned_turn",
        features=frozenset(),
    ) as agent:
        _plan_route(agent, app, monkeypatch, tmp_path)

        assert await turns(agent, monkeypatch) == ["Noted.", "Noted."]

        assert app.turns == ["thread-1", "thread-2", "thread-3"]
        assert app.rejected == []
