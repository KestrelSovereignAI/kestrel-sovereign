"""#3552: every orchestrator LLM call a turn makes has one inactivity bound.

The bound (#3300) used to cover only the streamed tool loop's follow-up calls.
A turn's FIRST call had none, and neither did any non-streaming call (the
``/api/agent/invoke`` path), so a local model that never finished held the
conversation lock until the host was killed. ``LLMCallWatchdog`` is now the one
code path for all of them, streamed or not: it re-arms on every sign of
progress (a streamed item, or progress a non-streaming adapter reports), counts
only time spent waiting on the provider, and on a trip cancels the pending
wait, which closes the provider request. The turn sees a failed attempt, never
an answer: a follow-up that stops part-way records no answer, only the
checkpoint of the tool batch it followed.

Watchdog semantics run on an event loop whose clock only the test advances, so
they cover minutes of provider silence without sleeping. The turn-level tests
drive a real agent on real storage with the bound shrunk to a fraction of a
second; only the provider calls are scripted.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.agent import orchestrator_engine as oe
from kestrel_sovereign.agent.orchestrator_engine import (
    LLMCallInactivityTimeout,
    LLMCallWatchdog,
    OrchestratorEngineMixin,
)
from kestrel_sovereign.agent.streaming import (
    INCOMPLETE_GENERATION_TOOL_BATCH_CHECKPOINT,
)
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.llm.call_progress import (
    call_progress_listener,
    report_call_progress,
)
from kestrel_sovereign.llm.output_ceiling import (
    IncompleteGenerationError,
    OutputCapReachedError,
)
from kestrel_sovereign.llm.streaming_errors import (
    agent_stream_error_block,
    safe_streaming_error_message,
)

BOUND = 180


class _VirtualClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose ``time()`` only moves when a test advances it."""

    def __init__(self) -> None:
        super().__init__()
        self._now = 0.0

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _run(loop: _VirtualClockLoop, coro):
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _silence(loop: _VirtualClockLoop, seconds: float = 10_000) -> None:
    for _ in range(int(seconds // 10)):
        loop.advance(10)
        await asyncio.sleep(0)


# --------------------------------------------------------------------------
# The watchdog
# --------------------------------------------------------------------------


def test_a_call_that_never_answers_is_abandoned_and_its_read_cancelled():
    loop = _VirtualClockLoop()
    cancelled_at: List[float] = []

    async def hung():
        try:
            await _silence(loop)
        except asyncio.CancelledError:
            cancelled_at.append(loop.time())
            raise
        yield "never"

    async def drive():
        return [item async for item in LLMCallWatchdog(BOUND).watch(hung())]

    with pytest.raises(LLMCallInactivityTimeout) as raised:
        _run(loop, drive())

    assert isinstance(raised.value, IncompleteGenerationError)
    assert raised.value.timeout == BOUND
    # The provider read itself was cancelled (that is what closes the HTTP
    # request), at the bound and not later.
    assert cancelled_at and BOUND <= cancelled_at[0] <= BOUND + 20


def test_time_the_consumer_spends_on_an_item_is_not_provider_silence():
    """A slow consumer (a client not reading its stream) is not the provider
    hanging: only the wait for the next item counts."""
    loop = _VirtualClockLoop()

    async def two_items():
        yield "a"
        yield "b"

    async def drive():
        seen = []
        async for item in LLMCallWatchdog(BOUND).watch(two_items()):
            seen.append(item)
            await _silence(loop, 3 * BOUND)
        return seen

    assert _run(loop, drive()) == ["a", "b"]


def test_a_timeout_raised_by_the_provider_itself_is_not_relabelled():
    async def provider_timeout():
        raise TimeoutError("the provider's own timeout")
        yield  # pragma: no cover

    async def drive():
        return [item async for item in LLMCallWatchdog(BOUND).watch(provider_timeout())]

    with pytest.raises(TimeoutError) as raised:
        asyncio.run(drive())
    assert not isinstance(raised.value, LLMCallInactivityTimeout)


class _InlineHost(OrchestratorEngineMixin):
    """Minimal host for the real ``_make_inline_tool_executor``: its one tool
    runs for ten virtual minutes."""

    def __init__(self, loop: _VirtualClockLoop) -> None:
        self._loop = loop

    async def execute_named_tool(self, name, args, *, session_id, source, _capture):
        await _silence(self._loop, 600)
        return {"ok": True}


def test_an_inline_tool_running_inside_the_call_is_not_provider_silence():
    """The codex app-server runs tools inside the call, on its own reader task,
    while the call waits for the model's next item. A ten-minute tool must not
    trip a three-minute bound; the model going silent afterwards still does."""
    loop = _VirtualClockLoop()
    host = _InlineHost(loop)
    watchdog = LLMCallWatchdog(BOUND)
    executor = host._make_inline_tool_executor("session-1", watchdog=watchdog)

    async def codex_like_call():
        reader_task = asyncio.get_running_loop().create_task(
            executor("long_review", {})
        )
        _effective_args, result = await reader_task
        yield result
        await _silence(loop)
        yield "never"

    async def drive():
        seen = []
        with pytest.raises(LLMCallInactivityTimeout):
            async for item in watchdog.watch(codex_like_call()):
                seen.append((item, loop.time()))
        return seen, loop.time()

    seen, ended = _run(loop, drive())
    assert seen == [({"ok": True}, 600)]
    assert BOUND <= ended - 600 <= BOUND + 20


def test_the_bound_is_the_turn_timeout_unless_a_local_route_allows_longer():
    service = MagicMock()
    service.resolve_provider_routing.return_value = (["route"], None)

    service.effective_request_timeout.return_value = None
    assert LLMCallWatchdog.for_call(
        service, model_override=None, force_local_only=True,
    ).timeout == oe.ORCHESTRATOR_TURN_TIMEOUT_SECS

    service.effective_request_timeout.return_value = 1800.0
    assert LLMCallWatchdog.for_call(
        service, model_override=None, force_local_only=True,
    ).timeout == 1800.0


def test_a_non_streaming_bound_counts_every_candidate_clients_timeout():
    """A non-streaming call is silent until it answers, so its bound is never
    shorter than its client's own timeout, cloud routes included."""
    service = MagicMock()
    service.resolve_provider_routing.return_value = (["route"], None)
    service.effective_request_timeout.return_value = 600.0

    assert LLMCallWatchdog.for_call(
        service, model_override="m", force_local_only=False, streaming=False,
    ).timeout == 600.0
    service.effective_request_timeout.assert_called_with(
        ["route"], completion_only=True,
    )


def test_a_non_streaming_call_that_never_answers_is_abandoned():
    loop = _VirtualClockLoop()
    cancelled_at: List[float] = []

    async def hung_completion():
        try:
            await _silence(loop)
        except asyncio.CancelledError:
            cancelled_at.append(loop.time())
            raise
        return "never"

    with pytest.raises(LLMCallInactivityTimeout):
        _run(loop, LLMCallWatchdog(BOUND).call(hung_completion()))
    assert cancelled_at and BOUND <= cancelled_at[0] <= BOUND + 20


def test_progress_a_non_streaming_call_reports_keeps_it_alive():
    """An answer that keeps arriving is never cut, however long it takes; one
    that stops arriving is cut a bound after its last progress."""
    loop = _VirtualClockLoop()

    async def long_completion(parts: int, then_silent: bool):
        for _ in range(parts):
            loop.advance(10)
            await asyncio.sleep(0)
            report_call_progress()
        if then_silent:
            await _silence(loop)
        return "answer"

    assert _run(loop, LLMCallWatchdog(BOUND).call(long_completion(40, False))) == "answer"
    assert loop.time() == 400

    loop = _VirtualClockLoop()
    with pytest.raises(LLMCallInactivityTimeout):
        _run(loop, LLMCallWatchdog(BOUND).call(long_completion(40, True)))
    assert 400 + BOUND <= loop.time() <= 400 + BOUND + 20


def test_a_progress_report_outside_a_bounded_call_does_nothing():
    report_call_progress()
    seen: List[int] = []
    with call_progress_listener(lambda: seen.append(1)):
        report_call_progress()
    report_call_progress()
    assert seen == [1]


class _EventStream:
    """``client.messages.stream(...)``: one event every ten virtual seconds,
    optionally followed by silence, then the final message."""

    def __init__(self, loop: _VirtualClockLoop, events: int, then_silent: bool):
        self._loop = loop
        self._events = events
        self._then_silent = then_silent

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._events == 0:
            if self._then_silent:
                await _silence(self._loop)
            raise StopAsyncIteration
        self._events -= 1
        self._loop.advance(10)
        await asyncio.sleep(0)
        return "event"

    async def get_final_message(self):
        return "final message"


def test_a_long_non_streaming_anthropic_answer_is_not_cut_while_it_arrives():
    """#3300 on the invoke path: an Anthropic answer read off its stream can
    run far past the bound. Each event is progress, so only silence cuts it."""
    from kestrel_sovereign.llm.anthropic_adapter import _anthropic_final_message

    def client(loop, then_silent):
        fake = MagicMock()
        fake.messages.stream = MagicMock(
            side_effect=lambda **_: _EventStream(loop, 40, then_silent)
        )
        return fake

    loop = _VirtualClockLoop()
    answer = _run(loop, LLMCallWatchdog(BOUND).call(
        _anthropic_final_message(client(loop, False), {})
    ))
    assert answer == "final message"
    assert loop.time() == 400

    loop = _VirtualClockLoop()
    with pytest.raises(LLMCallInactivityTimeout):
        _run(loop, LLMCallWatchdog(BOUND).call(
            _anthropic_final_message(client(loop, True), {})
        ))


def test_every_codex_turn_event_is_progress():
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    adapter = CodexAdapter()

    async def turn(*_args, **_kwargs):
        for _ in range(3):
            yield {"delta": "x"}
        yield {"final": ("done", None, {})}

    adapter._run_turn_with_retry = turn
    seen: List[int] = []

    async def drive():
        with call_progress_listener(lambda: seen.append(1)):
            return await adapter.get_response(None, "gpt-5", [])

    assert asyncio.run(drive()).content == "done"
    assert len(seen) == 4


# --------------------------------------------------------------------------
# The turn: first call and follow-up share the bound; the lock is released
# --------------------------------------------------------------------------

TURN_BOUND = 0.3


@asynccontextmanager
async def _booted_agent(tmp_path):
    """A real agent on real storage; only the provider calls are scripted."""
    from kestrel_sovereign.bootstrap import BootstrapState
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from tests.shared.genesis_audit import complete_deterministic_genesis_audit

    credentials = await create_kestrel_identity_async(
        output_dir=str(tmp_path), is_test_instance=True, agent_name="Watchdog"
    )
    llm_service = LLMService()
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=os.path.join(str(tmp_path), "kestrel_prime.db"),
        llm_service=llm_service,
    )
    try:
        await agent.initialize()
        await complete_deterministic_genesis_audit(
            agent, provenance="test:llm_call_watchdog"
        )
        await agent.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
        yield agent
    finally:
        await agent.shutdown()
        await llm_service.close()


class _ScriptedProvider:
    """The provider's calls, one script per call, streamed or not.

    A ``None`` script, or a ``None`` item in a streamed one, never answers and
    records that its wait was cancelled. A streamed script yields its items; a
    non-streaming one is returned. Calls past the last script repeat it.
    """

    def __init__(self, *scripts: Any) -> None:
        self._scripts = list(scripts)
        self.calls = 0
        self.cancelled = 0

    def _next_script(self) -> Any:
        script = self._scripts[min(self.calls, len(self._scripts) - 1)]
        self.calls += 1
        return script

    async def _never_answer(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

    async def stream_with_tool_detection(self, **_kwargs):
        script = self._next_script()
        if script is None:
            await self._never_answer()
        for item in script:
            if item is None:
                await self._never_answer()
            yield item

    async def generate_with_messages(self, **_kwargs):
        script = self._next_script()
        if script is None:
            await self._never_answer()
        return script


@pytest.fixture
def recorded_watchdogs(monkeypatch):
    """Every watchdog the turn builds, so a test can see each call's bound."""
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", TURN_BOUND)
    built: List[LLMCallWatchdog] = []
    original = LLMCallWatchdog.for_call.__func__

    def recording(cls, *args, **kwargs):
        watchdog = original(cls, *args, **kwargs)
        built.append(watchdog)
        return watchdog

    monkeypatch.setattr(LLMCallWatchdog, "for_call", classmethod(recording))
    return built


def _conversation_holder(agent):
    from kestrel_sdk.signals import ResourceLock

    return agent._get_lock_manager().holder(ResourceLock.CONVERSATION)


async def _turn(agent, text: str) -> str:
    chunks = [chunk async for chunk in agent.process_input_streaming(text, session_id="s1")]
    return "".join(c for c in chunks if isinstance(c, str))


async def _assistant_rows(agent, session_id: str) -> List[str]:
    """What the session durably recorded as the assistant's turns."""
    history = await agent.privacy_agent.get_conversation_history(
        limit=50, session_id=session_id,
    )
    return [row["content"] for row in history if row.get("role") == "assistant"]


#: A real, read-only tool, so the follow-up runs after a completed batch.
_TOOL_CALL = LLMResponse(
    content="",
    tool_calls=[ToolCall(id="t1", name="state_of_mind", arguments={})],
)


@pytest.mark.asyncio
async def test_a_first_call_that_never_answers_fails_and_releases_the_turn(
    tmp_path, monkeypatch, recorded_watchdogs,
):
    """The incident: the turn's first call never finished. It now trips the
    same bound as a follow-up, fails visibly, cancels the provider request,
    and leaves the agent able to take the next turn."""
    async with _booted_agent(tmp_path) as agent:
        provider = _ScriptedProvider(None, ["Noted."])
        monkeypatch.setattr(
            agent.llm_service, "stream_with_tool_detection",
            provider.stream_with_tool_detection,
        )

        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(LLMCallInactivityTimeout) as raised:
            await asyncio.wait_for(_turn(agent, "Where do I work?"), 30)
        elapsed = loop.time() - started

        assert [w.timeout for w in recorded_watchdogs] == [TURN_BOUND]
        assert TURN_BOUND <= elapsed < TURN_BOUND + 10
        assert provider.cancelled == 1, "the provider request was left running"
        assert _conversation_holder(agent) is None
        assert "did not finish its response" in agent_stream_error_block(raised.value)
        assert await _assistant_rows(agent, "s1") == []

        # The next turn is not queued behind the failed one.
        assert await asyncio.wait_for(_turn(agent, "And now?"), 30) == "Noted."


async def _failed_streamed_turn(agent, text: str) -> str:
    """The text a streamed turn delivered before it failed on its bound."""
    chunks: List[Any] = []
    with pytest.raises(LLMCallInactivityTimeout):
        async for chunk in agent.process_input_streaming(text, session_id="s1"):
            chunks.append(chunk)
    return "".join(c for c in chunks if isinstance(c, str))


@pytest.mark.asyncio
async def test_a_follow_up_that_never_answers_hits_the_same_bound(
    tmp_path, monkeypatch, recorded_watchdogs,
):
    async with _booted_agent(tmp_path) as agent:
        provider = _ScriptedProvider([_TOOL_CALL], None, ["Noted."])
        monkeypatch.setattr(
            agent.llm_service, "stream_with_tool_detection",
            provider.stream_with_tool_detection,
        )

        text = await asyncio.wait_for(_failed_streamed_turn(agent, "Look it up."), 30)

        from kestrel_sovereign.agent.streaming import _parse_stream_sentinels

        _clean, parts, _ = _parse_stream_sentinels(text)
        assert any(
            p["phase"] == "error" and p.get("name") == "llm"
            and "timeout" in (p.get("detail") or "")
            for p in parts
        ), text
        assert provider.calls == 2 and provider.cancelled == 1
        # One code path, one bound: the first call and the follow-up.
        assert [w.timeout for w in recorded_watchdogs] == [TURN_BOUND, TURN_BOUND]
        assert _conversation_holder(agent) is None
        # No answer was recorded; the completed tool batch was.
        assert await _assistant_rows(agent, "s1") == [
            INCOMPLETE_GENERATION_TOOL_BATCH_CHECKPOINT,
        ]
        assert await asyncio.wait_for(_turn(agent, "And now?"), 30) == "Noted."


@pytest.mark.asyncio
async def test_partial_follow_up_prose_is_never_recorded_as_the_answer(
    tmp_path, monkeypatch, recorded_watchdogs,
):
    """A follow-up that streams part of its synthesis and then goes silent
    has not answered: the turn fails, and the prose the client already saw
    is not persisted as the assistant's turn."""
    partial = "Your employer, according to my notes, is"
    async with _booted_agent(tmp_path) as agent:
        provider = _ScriptedProvider([_TOOL_CALL], [partial, None])
        monkeypatch.setattr(
            agent.llm_service, "stream_with_tool_detection",
            provider.stream_with_tool_detection,
        )

        text = await asyncio.wait_for(
            _failed_streamed_turn(agent, "Where do I work?"), 30,
        )

        assert partial in text  # it streamed live...
        rows = await _assistant_rows(agent, "s1")
        assert rows == [INCOMPLETE_GENERATION_TOOL_BATCH_CHECKPOINT]
        assert not any(partial in row for row in rows)  # ...and is not an answer
        assert provider.cancelled == 1
        assert _conversation_holder(agent) is None


# --------------------------------------------------------------------------
# The non-streaming turn: /api/agent/invoke
# --------------------------------------------------------------------------


@asynccontextmanager
async def _invoke_client(agent):
    """``/api/agent/invoke`` served for ``agent`` on the test's own loop."""
    import httpx
    from fastapi import FastAPI

    from kestrel_sovereign.api_errors import register_api_error_handlers
    from kestrel_sovereign.endpoints.agent import router
    from kestrel_sovereign.rate_limit import limiter

    app = FastAPI()
    app.state.limiter = limiter
    app.state.agent = agent
    app.include_router(router)
    register_api_error_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        yield client


def _on_an_ollama_route(agent, monkeypatch) -> None:
    """Lift the turn's bounds as the incident's route would.

    A non-streaming call's bound is never shorter than its route's own client
    timeout. The test agent's default route is a cloud SDK with a 600s
    timeout; the incident's was local Ollama, whose client sets none, so
    nothing lifts the bound there.
    """
    import ollama

    from kestrel_sovereign.llm.service import LLMService

    route = [{"is_local": True, "client": ollama.AsyncClient(host="http://127.0.0.1:9")}]
    monkeypatch.setattr(
        agent.llm_service, "effective_request_timeout",
        lambda _candidates, **kwargs: LLMService.effective_request_timeout(
            agent.llm_service, route, **kwargs,
        ),
    )


async def _invoke(client, text: str):
    return await asyncio.wait_for(
        client.post(
            "/api/agent/invoke", json={"input": text, "session_id": "inv-1"},
        ),
        30,
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
async def test_a_stalled_invocation_fails_502_and_the_next_is_admitted(
    tmp_path, monkeypatch, recorded_watchdogs,
):
    """The incident's own door. The non-streaming first call gets the same
    bound: the stalled provider request is cancelled, the caller gets
    ``502 generation_incomplete``, and the next invocation is not queued
    behind the failed one."""
    async with _booted_agent(tmp_path) as agent:
        provider = _ScriptedProvider(None, LLMResponse(content="Noted."))
        monkeypatch.setattr(
            agent.llm_service, "generate_with_messages",
            provider.generate_with_messages,
        )
        _on_an_ollama_route(agent, monkeypatch)
        async with _invoke_client(agent) as client:
            failed = await _invoke(client, "Where do I work?")

            assert failed.status_code == 502
            assert failed.json()["error"]["code"] == "generation_incomplete"
            assert provider.cancelled == 1, "the provider request was left running"
            assert [w.timeout for w in recorded_watchdogs] == [TURN_BOUND]
            assert _conversation_holder(agent) is None
            assert await _assistant_rows(agent, "inv-1") == []

            answered = await _invoke(client, "And now?")
            assert answered.status_code == 200
            assert answered.json()["response"] == "Noted."


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
async def test_a_stalled_non_streaming_follow_up_records_only_its_checkpoint(
    tmp_path, monkeypatch, recorded_watchdogs,
):
    async with _booted_agent(tmp_path) as agent:
        provider = _ScriptedProvider(_TOOL_CALL, None, LLMResponse(content="Noted."))
        monkeypatch.setattr(
            agent.llm_service, "generate_with_messages",
            provider.generate_with_messages,
        )
        _on_an_ollama_route(agent, monkeypatch)
        async with _invoke_client(agent) as client:
            failed = await _invoke(client, "Look it up.")

            assert failed.status_code == 502
            assert failed.json()["error"]["code"] == "generation_incomplete"
            assert provider.calls == 2 and provider.cancelled == 1
            assert [w.timeout for w in recorded_watchdogs] == [TURN_BOUND, TURN_BOUND]
            assert _conversation_holder(agent) is None
            assert await _assistant_rows(agent, "inv-1") == [
                INCOMPLETE_GENERATION_TOOL_BATCH_CHECKPOINT,
            ]
            assert (await _invoke(client, "And now?")).status_code == 200


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
async def test_a_stalled_premature_yield_repair_hits_the_same_bound(
    tmp_path, monkeypatch, recorded_watchdogs,
):
    """Tool-call markup written as text gets one repair call. It is an
    orchestrator call like any other, under the same bound."""
    markup = LLMResponse(content='<invoke name="state_of_mind"></invoke>')
    async with _booted_agent(tmp_path) as agent:
        provider = _ScriptedProvider(markup, None, LLMResponse(content="Noted."))
        monkeypatch.setattr(
            agent.llm_service, "generate_with_messages",
            provider.generate_with_messages,
        )
        _on_an_ollama_route(agent, monkeypatch)
        async with _invoke_client(agent) as client:
            failed = await _invoke(client, "How are you?")

            assert failed.status_code == 502
            assert provider.calls == 2 and provider.cancelled == 1
            assert [w.timeout for w in recorded_watchdogs] == [TURN_BOUND, TURN_BOUND]
            assert _conversation_holder(agent) is None
            assert await _assistant_rows(agent, "inv-1") == []


# --------------------------------------------------------------------------
# What the caller sees
# --------------------------------------------------------------------------


def test_the_stream_error_says_the_model_did_not_finish_and_nothing_else():
    from kestrel_sovereign.llm.streaming import LLMStreamingError

    cap = OutputCapReachedError(provider="ollama", model="llama3.2:1b", cap=4096)
    wrapped = LLMStreamingError(
        "Selected route ollama:local failed: WITHHELD-RESPONSE-TEXT",
        provider="ollama:local",
        underlying=cap,
    )
    for exc in (wrapped, LLMCallInactivityTimeout(180)):
        message = safe_streaming_error_message(exc)
        assert message.startswith("The model did not finish its response.")
        assert "WITHHELD" not in message and "ollama" not in message


@pytest.fixture
def invoke_app():
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = (
        app.router.lifespan_context,
        getattr(app.state, "agent", None),
        getattr(app.state, "agent_manager", None),
    )

    def boot(process_input_error: Exception):
        agent = MagicMock()
        agent.agent_id = "did:pkh:eip155:1:0xabc"
        agent.privacy_mode.value = "NORMAL"
        agent.features = {}
        agent.process_input = AsyncMock(side_effect=process_input_error)
        agent.is_request_cancelled = MagicMock(return_value=False)
        agent.storage.resolve_session_id = AsyncMock(return_value="sess-1")
        app.router.lifespan_context = noop_lifespan
        app.state.agent = agent
        app.state.agent_manager = None
        return app

    yield boot
    app.router.lifespan_context, app.state.agent, app.state.agent_manager = original


@pytest.mark.usefixtures("isolated_process_rate_limiter")
def test_invoke_answers_502_when_the_model_did_not_finish(invoke_app):
    """The incident's own path: ``/api/agent/invoke`` runs the non-streaming
    turn, where a capped generation surfaces as the selected route failing."""
    from fastapi.testclient import TestClient

    from kestrel_sovereign.llm.service import LLMServiceError

    cap = OutputCapReachedError(provider="ollama", model="llama3.2:1b", cap=4096)
    try:
        raise LLMServiceError("Selected route ollama:local failed: WITHHELD") from cap
    except LLMServiceError as exc:
        failure = exc
    app = invoke_app(failure)

    with patch.dict(os.environ, {"KESTREL_API_KEY": "test-key"}), TestClient(app) as client:
        response = client.post(
            "/api/agent/invoke",
            json={"input": "Where do I work?"},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "generation_incomplete"
    assert "WITHHELD" not in response.text
