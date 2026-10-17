"""#2658: one silent provider call must not wedge an agent's turn pipeline.

The incident: on ``openrouter:api`` / ``moonshotai/kimi-k3`` a cognition call
never answered, so every later turn on the agent, ``!status`` included,
queued behind it until the agent was restarted.

These tests drive a real agent on real storage over the incident's route: the
real OpenRouter adapter and OpenAI SDK, with only the HTTP endpoint scripted.
A request the endpoint never answers is the stall. The bound is shrunk to a
fraction of a second, or the loop clock is pinned, so nothing waits on the
wall clock for the outcome under test.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from contextlib import suppress
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock

import httpx2
import openai
import pytest

from kestrel_sovereign.agent import orchestrator_engine as oe
from kestrel_sovereign.agent.orchestrator_engine import (
    LLMCallInactivityTimeout,
    LLMCallWatchdog,
)
from kestrel_sovereign.command_handler import BUILTIN_COMMAND_SPECS, CommandHandler
from kestrel_sovereign.command_policy import (
    RECOVERY_COMMANDS,
    TURN_QUEUE_BYPASS_COMMANDS,
)
from kestrel_sovereign.llm.call_progress import (
    AttemptInactivityTimeout,
    report_call_attempt,
)
from kestrel_sovereign.llm.error_handling import LLMError
from kestrel_sovereign.llm.output_ceiling import incomplete_generation
from tests.shared.booted_agent import booted_agent
from tests.unit.test_llm_call_watchdog import _conversation_holder, _invoke_client
from tests.utils.virtual_clock import (
    VirtualClockLoop,
    await_within_hang_guard,
    pinned_loop_clock,
)

ROUTE = "openrouter:api"
MODEL = "moonshotai/kimi-k3"
TURN_BOUND = 1.0
HANG_GUARD = 30

#: A request the endpoint never answers.
HANG = object()

Reply = Callable[[Dict[str, Any]], httpx2.Response]


def _sse_event(event: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


def _sse(events: List[Dict[str, Any]]) -> bytes:
    return b"".join(_sse_event(event) for event in events) + b"data: [DONE]\n\n"


def _chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": "gen-1", "object": "chat.completion.chunk", "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


_USAGE_CHUNK = {
    "id": "gen-1", "object": "chat.completion.chunk", "created": 0,
    "model": MODEL, "choices": [],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


def _answer(content: str) -> Reply:
    """A finished reply, streamed or not as the request asked."""

    def reply(body: Dict[str, Any]) -> httpx2.Response:
        if body.get("stream"):
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse([
                    _chunk({"role": "assistant", "content": content}),
                    _chunk({}, "stop"),
                    _USAGE_CHUNK,
                ]),
            )
        return httpx2.Response(200, json={
            "id": "gen-1", "object": "chat.completion", "created": 0,
            "model": MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": _USAGE_CHUNK["usage"],
        })

    return reply


_AUDIT_PASS = _answer(json.dumps({"risk_level": 1, "reasoning": "Clear and safe."}))


class _OpenRouter:
    """The OpenRouter endpoint behind the route. Each request takes the next
    reply; the last one repeats. ``HANG`` never answers and records that its
    wait was cancelled, which is what closes the provider request."""

    def __init__(self, *replies: Any) -> None:
        self._replies = list(replies)
        self.requests: List[Dict[str, Any]] = []
        self.cancelled = 0
        self.stalled = asyncio.Event()

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        reply = self._replies[min(len(self.requests), len(self._replies)) - 1]
        if reply is HANG:
            self.stalled.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return reply(body)


def _on_the_incident_route(
    agent, monkeypatch, endpoint: _OpenRouter, *, client_timeout: float = 600.0,
) -> None:
    """Route every model call to ``openrouter:api`` / ``moonshotai/kimi-k3``,
    built as the provider registry builds it, over ``endpoint``.

    A non-streaming call's bound is never shorter than its route's client
    timeout, so a test of one sets the client timeout to the turn bound.
    """
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    info = ProviderRegistry({})._build_route(
        "openrouter", "api", {"is_cloud": True},
        {"adapter": "OpenRouterAdapter", "api_key": "sk-or-test", "model": MODEL},
    )
    service = agent.llm_service
    [provider] = service._convert_providers_format([info])
    provider["client"] = openai.AsyncOpenAI(
        api_key="sk-or-test", base_url="https://openrouter.test/api/v1",
        max_retries=0, timeout=client_timeout,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(endpoint.handle)),
    )
    assert not provider["is_local"]
    monkeypatch.setattr(service, "providers", [provider])
    monkeypatch.setattr(service, "config", {
        **(service.config or {}), "route_priority": [ROUTE],
    })
    monkeypatch.setattr(service, "_mandate_preference", {})
    monkeypatch.setattr(service, "_ensure_models_discovered", AsyncMock())


@pytest.fixture
def recorded_bounds(monkeypatch) -> List[Tuple[float, Optional[str]]]:
    """The bound of every watchdog built and the model override it was
    resolved for, with the turn bound shrunk."""
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", TURN_BOUND)
    bounds: List[Tuple[float, Optional[str]]] = []
    original = LLMCallWatchdog.for_call.__func__

    def recording(cls, *args, **kwargs):
        watchdog = original(cls, *args, **kwargs)
        bounds.append((watchdog.timeout, kwargs.get("model_override")))
        return watchdog

    monkeypatch.setattr(LLMCallWatchdog, "for_call", classmethod(recording))
    return bounds


def _restore_the_bound(monkeypatch) -> None:
    """Give the turns after a stall a bound no loaded test worker reaches, so
    only the stalled call is decided by the shrunk one."""
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", HANG_GUARD)


def _agent(tmp_path, *, features=frozenset(), **kwargs):
    return booted_agent(
        tmp_path, name="Nellie", provenance="test:turn_pipeline_stall",
        features=set(features), **kwargs,
    )


def _abandoned_warnings(caplog) -> List[str]:
    """The watchdog's own records of an abandoned call."""
    return [
        record.getMessage() for record in caplog.records
        if record.levelno >= logging.WARNING
        and record.getMessage().startswith("[ORCHESTRATOR] LLM call produced nothing")
    ]


def _assert_logged_the_stuck_route(caplog) -> None:
    warnings = _abandoned_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert ROUTE in warnings[0] and MODEL in warnings[0], warnings[0]


async def _streamed_turn(agent, text: str, session_id: str) -> str:
    chunks = [
        chunk async for chunk in agent.process_input_streaming(text, session_id=session_id)
    ]
    return "".join(chunk for chunk in chunks if isinstance(chunk, str))


async def _turn(agent, door: str, text: str, session_id: str) -> str:
    if door == "stream":
        return await _streamed_turn(agent, text, session_id)
    return await agent.process_input(text, session_id=session_id)


# --------------------------------------------------------------------------
# The watchdog names the attempt it abandoned
# --------------------------------------------------------------------------


def _abandon_on_virtual_clock(call: Callable[[], Any]) -> LLMCallInactivityTimeout:
    """Run ``call`` under a 180s watchdog on a clock only the test moves."""
    loop = VirtualClockLoop()
    watchdog = LLMCallWatchdog(180)

    async def silence() -> None:
        for _ in range(100):
            loop.advance(10)
            await asyncio.sleep(0)

    async def drive() -> None:
        await watchdog.call(call(watchdog, silence))

    try:
        with pytest.raises(LLMCallInactivityTimeout) as raised:
            loop.run_until_complete(drive())
    finally:
        loop.close()
    return raised.value


def test_the_abandoned_attempt_is_named_not_one_an_inline_tool_made(caplog):
    """An inline tool's own model call reports its attempt while the outer
    call waits on the tool; the outer call's stall still names its route."""
    caplog.set_level(logging.WARNING)

    async def call(watchdog, silence):
        report_call_attempt("openai:plan", "gpt-5.5")
        with watchdog.tool_running():
            report_call_attempt(ROUTE, MODEL)
            await asyncio.sleep(0)
        await silence()

    abandoned = _abandon_on_virtual_clock(call)

    assert (abandoned.provider, abandoned.model) == ("openai:plan", "gpt-5.5")
    [warning] = _abandoned_warnings(caplog)
    assert "openai:plan" in warning and "gpt-5.5" in warning
    assert ROUTE not in warning


def test_a_call_abandoned_before_any_attempt_says_so(caplog):
    caplog.set_level(logging.WARNING)

    async def call(_watchdog, silence):
        await silence()

    abandoned = _abandon_on_virtual_clock(call)

    assert (abandoned.provider, abandoned.model) == (None, None)
    [warning] = _abandoned_warnings(caplog)
    assert "no provider attempt had started" in warning


# --------------------------------------------------------------------------
# A stalled cognition call fails its turn on its bound and frees the queue
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["stream", "invoke"])
async def test_a_stalled_cloud_call_fails_on_its_bound_and_frees_the_queue(
    tmp_path, monkeypatch, caplog, recorded_bounds, door,
):
    """The incident's route, its first call never answering. The turn fails
    on the watchdog with an error naming the route and model, the request is
    cancelled, the WARNING names them too, and the next turn is answered.

    The route is the call's only candidate, so its attempt's bound is the
    call's (#1968): abandoned, it fails the way any failure of the call's
    last route does, with that attempt as the call's verdict."""
    caplog.set_level(logging.WARNING)
    async with _agent(tmp_path) as agent:
        endpoint = _OpenRouter(HANG, _answer("Noted."))
        _on_the_incident_route(
            agent, monkeypatch, endpoint,
            # A streamed cloud call keeps the turn bound; a non-streaming one
            # waits as long as its client would.
            client_timeout=600.0 if door == "stream" else TURN_BOUND,
        )

        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(LLMError) as raised:
            await asyncio.wait_for(
                _turn(agent, door, "What is on my calendar?", "s1"), HANG_GUARD,
            )
        elapsed = loop.time() - started

        assert recorded_bounds == [(TURN_BOUND, None)]
        assert TURN_BOUND <= elapsed < TURN_BOUND + 10
        abandoned = incomplete_generation(raised.value)
        assert isinstance(abandoned, AttemptInactivityTimeout)
        assert (abandoned.provider, abandoned.model) == (ROUTE, MODEL)
        assert abandoned.timeout == TURN_BOUND
        assert endpoint.cancelled == 1, "the provider request was left running"
        assert _conversation_holder(agent) is None
        _assert_logged_the_stuck_route(caplog)

        _restore_the_bound(monkeypatch)
        assert await asyncio.wait_for(
            _turn(agent, door, "And now?", "s1"), HANG_GUARD,
        ) == "Noted."
        assert len(endpoint.requests) == 2


# --------------------------------------------------------------------------
# !status answers beside a stalled turn
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
async def test_status_answers_beside_a_stalled_turn_without_touching_it(
    tmp_path, monkeypatch,
):
    """While a turn waits on a provider that will not answer for an hour,
    ``!status`` (and ``!help``) answer at once on every door, make no model
    call, and leave the stalled turn as they found it: its lock, its session,
    the tool statistics it has gathered, the constitutional interaction count
    and the provider credentials it is calling with."""
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", 3600)
    async with _agent(tmp_path) as agent:
        endpoint = _OpenRouter(HANG)
        _on_the_incident_route(agent, monkeypatch, endpoint)
        stalled = asyncio.create_task(
            agent.process_input("What is on my calendar?", session_id="stalled"),
        )
        try:
            await asyncio.wait_for(endpoint.stalled.wait(), HANG_GUARD)
            holder = _conversation_holder(agent)
            assert holder is not None

            # The stalled turn's tool calls so far, which a session change
            # would discard.
            agent.context_stats.record("web_search", {"query": "calendar"}, 120)
            store = agent._constitution_state_store
            assert store is not None

            async def shared_state():
                persisted = await store.load(agent.agent_id)
                return (
                    agent.context_stats._last_session_id,
                    agent.context_stats.get_analysis(),
                    agent._interaction_count,
                    persisted.interaction_count,
                    persisted.revision,
                )

            before = await shared_state()
            assert before[0] == "stalled"
            assert before[1]["total_tool_calls"] == 1

            refreshes: List[Optional[str]] = []
            refresh = agent._maybe_refresh_user_byok_resolver

            async def recording_refresh(passphrase):
                refreshes.append(passphrase)
                await refresh(passphrase)

            monkeypatch.setattr(
                agent, "_maybe_refresh_user_byok_resolver", recording_refresh,
            )

            async def assert_stall_untouched() -> None:
                assert not stalled.done()
                assert _conversation_holder(agent) == holder
                assert agent._active_session_id == "stalled"
                assert len(endpoint.requests) == 1
                assert await shared_state() == before
                assert refreshes == []

            status = f"Agent ID: {agent.agent_id}"
            assert status in await asyncio.wait_for(
                agent.process_input("!status", session_id="ops"), HANG_GUARD,
            )
            await assert_stall_untouched()

            assert status in await asyncio.wait_for(
                _streamed_turn(agent, "!status", "ops"), HANG_GUARD,
            )
            await assert_stall_untouched()

            async with _invoke_client(agent) as client:
                answered = await asyncio.wait_for(
                    client.post(
                        "/api/agent/invoke",
                        json={"input": "!status", "session_id": "ops"},
                    ),
                    HANG_GUARD,
                )
            assert answered.status_code == 200
            assert status in answered.json()["response"]
            await assert_stall_untouched()

            assert "Kestrel Agent Commands" in await asyncio.wait_for(
                agent.process_input("!help", session_id="ops"), HANG_GUARD,
            )
            await assert_stall_untouched()
        finally:
            stalled.cancel()
            with suppress(asyncio.CancelledError):
                await stalled
        assert endpoint.cancelled == 1
        assert _conversation_holder(agent) is None


@pytest.mark.asyncio
async def test_status_answers_beside_a_stalled_genesis_audit(tmp_path, monkeypatch):
    """The genesis audit runs before the turn lock and holds every first turn.
    ``!status`` bypasses it and answers while the auditor is silent."""
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", 3600)
    async with _agent(tmp_path, genesis=False) as agent:
        endpoint = _OpenRouter(HANG)
        _on_the_incident_route(agent, monkeypatch, endpoint)
        stalled = asyncio.create_task(agent.process_input("Hello", session_id="s1"))
        try:
            await asyncio.wait_for(endpoint.stalled.wait(), HANG_GUARD)

            assert f"Agent ID: {agent.agent_id}" in await asyncio.wait_for(
                agent.process_input("!status"), HANG_GUARD,
            )
            assert not stalled.done()
            assert len(endpoint.requests) == 1
        finally:
            stalled.cancel()
            with suppress(asyncio.CancelledError):
                await stalled


def test_only_pure_reads_skip_the_turn_queue():
    """The commands answered beside a turn are the read-only diagnostics, and
    each handler is synchronous: it cannot await a lock, storage or a model,
    so it cannot interleave with the turn it answers beside."""
    assert TURN_QUEUE_BYPASS_COMMANDS == frozenset({"!status", "!help"})
    assert TURN_QUEUE_BYPASS_COMMANDS <= RECOVERY_COMMANDS
    handlers = {spec["cmd"]: spec["handler"] for spec in BUILTIN_COMMAND_SPECS}
    for command in TURN_QUEUE_BYPASS_COMMANDS:
        handler = getattr(CommandHandler, handlers[command])
        assert not inspect.iscoroutinefunction(handler), command


# --------------------------------------------------------------------------
# A long but progressing stream is never cut
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stream_progressing_past_the_bound_is_not_cut(tmp_path, monkeypatch):
    """The incident's route answering slowly: each chunk arrives within the
    bound, the whole answer takes several bounds. The turn keeps it all."""
    bound, gap, step = 180.0, 100.0, 10.0
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", bound)
    pieces = ["Your ", "calendar ", "has ", "two ", "meetings ", "today."]

    async with _agent(tmp_path) as agent:
        with pinned_loop_clock() as clock:
            async def slow_body() -> AsyncIterator[bytes]:
                for piece in pieces:
                    # The gap passes in steps, each a loop iteration, so a
                    # deadline that came due within it fires.
                    for _ in range(int(gap // step)):
                        clock.advance(step)
                        await asyncio.sleep(0)
                    yield _sse_event(_chunk({"content": piece}))
                yield _sse_event(_chunk({}, "stop"))
                yield _sse_event(_USAGE_CHUNK)
                yield b"data: [DONE]\n\n"

            def slow(_body: Dict[str, Any]) -> httpx2.Response:
                return httpx2.Response(
                    200, headers={"content-type": "text/event-stream"},
                    content=slow_body(),
                )

            endpoint = _OpenRouter(slow)
            _on_the_incident_route(agent, monkeypatch, endpoint)
            started = clock.time()
            text = await await_within_hang_guard(
                _streamed_turn(agent, "What is on my calendar?", "s1"),
                step="the slow streamed turn",
            )
            elapsed = clock.time() - started

    assert text == "".join(pieces)
    assert elapsed > 3 * bound
    assert endpoint.cancelled == 0
    assert len(endpoint.requests) == 1


# --------------------------------------------------------------------------
# The calls a turn makes outside its model loop carry the same bound
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stalled_genesis_audit_leaves_genesis_pending_and_frees_the_turn(
    tmp_path, monkeypatch, caplog, recorded_bounds,
):
    """The genesis audit runs before cognition, and every first turn waits on
    it. A silent auditor is abandoned on the bound: the turn answers that
    genesis is pending, and the next turn audits and answers."""
    caplog.set_level(logging.WARNING)
    async with _agent(tmp_path, genesis=False) as agent:
        endpoint = _OpenRouter(HANG, _AUDIT_PASS, _answer("Noted."))
        _on_the_incident_route(agent, monkeypatch, endpoint, client_timeout=TURN_BOUND)

        loop = asyncio.get_running_loop()
        started = loop.time()
        pending = await asyncio.wait_for(
            agent.process_input("Hello", session_id="s1"), HANG_GUARD,
        )
        elapsed = loop.time() - started

        assert pending.startswith("⏳ GENESIS AUDIT PENDING")
        assert recorded_bounds == [(TURN_BOUND, None)]
        assert TURN_BOUND <= elapsed < TURN_BOUND + 10
        assert endpoint.cancelled == 1
        _assert_logged_the_stuck_route(caplog)

        _restore_the_bound(monkeypatch)
        assert await asyncio.wait_for(
            agent.process_input("Hello again", session_id="s1"), HANG_GUARD,
        ) == "Noted."
        assert len(endpoint.requests) == 3


@pytest.mark.asyncio
async def test_a_stalled_soul_generation_falls_back_to_the_default_soul(
    tmp_path, monkeypatch, caplog, recorded_bounds,
):
    """Generating SOUL.md runs inside the turn that completes discovery. A
    silent provider is abandoned on the bound and the default SOUL is used."""
    caplog.set_level(logging.WARNING)
    async with _agent(tmp_path) as agent:
        bootstrap = agent.bootstrap_service
        await bootstrap._save_discovery_history([
            {"role": "assistant", "content": "What should I call you?"},
            {"role": "user", "content": "I'm Ada. I keep bees."},
        ])
        endpoint = _OpenRouter(HANG)
        _on_the_incident_route(agent, monkeypatch, endpoint, client_timeout=TURN_BOUND)

        loop = asyncio.get_running_loop()
        started = loop.time()
        soul = await asyncio.wait_for(bootstrap.generate_soul_md(), HANG_GUARD)
        elapsed = loop.time() - started

        assert soul == bootstrap._get_default_soul_template()
        assert recorded_bounds == [(TURN_BOUND, None)]
        assert TURN_BOUND <= elapsed < TURN_BOUND + 10
        assert endpoint.cancelled == 1
        _assert_logged_the_stuck_route(caplog)


@pytest.mark.asyncio
async def test_a_stalled_compaction_summary_fails_the_compaction_and_frees_the_turn(
    tmp_path, monkeypatch, caplog, recorded_bounds,
):
    """Compaction's summary call runs inside a turn (``!compact``, and the
    openai:plan pre-turn compaction). A silent provider is abandoned on the
    bound: nothing is compacted, and the next turn is answered."""
    caplog.set_level(logging.WARNING)
    async with _agent(tmp_path) as agent:
        for index in range(4):
            await agent.privacy_agent.add_conversation(
                "user", f"Question {index}", session_id="s1",
            )
            await agent.privacy_agent.add_conversation(
                "assistant", f"Answer {index}", session_id="s1",
            )
        before = await agent.privacy_agent.get_conversation_history(
            limit=50, session_id="s1",
        )
        endpoint = _OpenRouter(HANG, _answer("Noted."))
        _on_the_incident_route(agent, monkeypatch, endpoint, client_timeout=TURN_BOUND)

        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await asyncio.wait_for(
            agent.process_input("!compact --force --keep 2", session_id="s1"),
            HANG_GUARD,
        )
        elapsed = loop.time() - started

        assert "Session compacted" not in result
        assert recorded_bounds == [(TURN_BOUND, None)]
        assert TURN_BOUND <= elapsed < TURN_BOUND + 10
        assert endpoint.cancelled == 1
        assert _conversation_holder(agent) is None
        _assert_logged_the_stuck_route(caplog)
        after = await agent.privacy_agent.get_conversation_history(
            limit=50, session_id="s1",
        )
        assert [row["content"] for row in after] == [row["content"] for row in before]

        _restore_the_bound(monkeypatch)
        assert await asyncio.wait_for(
            agent.process_input("And now?", session_id="s1"), HANG_GUARD,
        ) == "Noted."


async def _stored_rows(agent) -> List[Dict[str, Any]]:
    """Every stored conversation row, excluded and summarized ones included."""
    store = agent.context_manager.conversation_manager._get_conversation_store()
    return await store.get_full_history_with_ids()


def _context_command(command: str) -> Callable[[Any], Any]:
    async def run(agent) -> str:
        return await agent.process_input(command, session_id="s1")

    return run


async def _hierarchical_compact_tool(agent) -> str:
    """``hierarchical_compact`` as a feature subagent or tool call runs it.

    Its ``!context compact hierarchical`` command never reaches it: command
    routing takes the first matching prefix, which is ``!context compact``.
    """
    [feature] = [
        feature for feature in agent.features.values()
        if type(feature).__name__ == "ContextFeature"
    ]
    # Small chunks, so the history splits into several and compaction runs.
    result = await feature.hierarchical_compact(chunk_size=50)
    return result.confirmation or result.error or ""


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry_stage", ["accepted", "before-discovery", "before-endpoint"])
@pytest.mark.parametrize(
    ("invoke", "cheap_model", "succeeded"),
    [
        (_context_command("!context summarize last_n 4"), None, "Summarized "),
        (_hierarchical_compact_tool, None, "Hierarchical compact complete"),
        # The cheap-model sub-query routes by its own selector, so its bound
        # is resolved for that selector, not the agent's default.
        (
            _context_command("!context query last_4 calendar"),
            f"{ROUTE}/{MODEL}",
            "Answered query",
        ),
    ],
    ids=["summarize-command", "hierarchical-compact-tool", "query-command"],
)
async def test_a_stalled_context_summary_call_fails_and_frees_the_turn(
    tmp_path, monkeypatch, caplog, recorded_bounds, invoke, cheap_model, succeeded, expiry_stage,
):
    """Context deadlines free the turn before or after provider admission.

    An accepted silent request is actually cancelled; an expiry in discovery
    never invents a request. Both preserve rows and let the next turn answer.
    """
    caplog.set_level(logging.WARNING)
    async with _agent(tmp_path, features={"ContextFeature"}) as agent:
        for index in range(6):
            await agent.privacy_agent.add_conversation(
                "user", f"Question {index} about my calendar", session_id="s1",
            )
            await agent.privacy_agent.add_conversation(
                "assistant", f"Answer {index}", session_id="s1",
            )
        before = await _stored_rows(agent)
        endpoint = (_OpenRouter(HANG, _answer("Noted.")) if expiry_stage == "accepted"
                    else _OpenRouter(_answer("Noted.")))
        preparation_entered, preparation_release = asyncio.Event(), asyncio.Event()
        preparation_withdrawals = []
        if expiry_stage == "before-endpoint":
            actual_endpoint = endpoint.handle

            async def delayed_endpoint(request):
                preparation_entered.set()
                try:
                    await preparation_release.wait()
                except asyncio.CancelledError as withdrawal:
                    preparation_withdrawals.append(withdrawal)
                    raise
                return await actual_endpoint(request)

            monkeypatch.setattr(endpoint, "handle", delayed_endpoint)
        _on_the_incident_route(agent, monkeypatch, endpoint, client_timeout=TURN_BOUND)
        monkeypatch.setattr(
            agent.llm_service, "get_cheap_model_selector", lambda: cheap_model,
        )
        if expiry_stage == "before-discovery":
            actual_discovery = agent.llm_service._ensure_models_discovered

            async def delayed_discovery(*args, **kwargs):
                preparation_entered.set()
                try:
                    await preparation_release.wait()
                except asyncio.CancelledError as withdrawal:
                    preparation_withdrawals.append(withdrawal)
                    raise
                return await actual_discovery(*args, **kwargs)

            monkeypatch.setattr(agent.llm_service, "_ensure_models_discovered", delayed_discovery)

        loop = asyncio.get_running_loop()
        with pinned_loop_clock() as clock:
            started = loop.time()
            task = asyncio.create_task(invoke(agent))
            try:
                admission = endpoint.stalled if expiry_stage == "accepted" else preparation_entered
                await await_within_hang_guard(admission.wait(), step=f"context summary {expiry_stage} admission")
                clock.advance(TURN_BOUND + 0.1)
                result = await await_within_hang_guard(task, step="context summary deadline settlement")
                elapsed = loop.time() - started
            finally:
                preparation_release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        assert succeeded not in result
        assert "produced nothing" in result, result
        assert recorded_bounds == [(TURN_BOUND, cheap_model)]
        assert TURN_BOUND <= elapsed < TURN_BOUND + 10
        assert endpoint.cancelled == (1 if expiry_stage == "accepted" else 0)
        assert bool(endpoint.requests) is (expiry_stage == "accepted")
        assert _conversation_holder(agent) is None
        if expiry_stage != "before-discovery":
            _assert_logged_the_stuck_route(caplog)
        else:
            [warning] = _abandoned_warnings(caplog)
            assert "no provider attempt had started" in warning
            assert ROUTE not in warning and MODEL not in warning
        assert len(preparation_withdrawals) == (0 if expiry_stage == "accepted" else 1)
        assert await _stored_rows(agent) == before

        _restore_the_bound(monkeypatch)
        assert await asyncio.wait_for(
            agent.process_input("And now?", session_id="s1"), HANG_GUARD,
        ) == "Noted."
