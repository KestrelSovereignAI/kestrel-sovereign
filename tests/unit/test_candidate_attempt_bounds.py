"""#1968: each candidate of a fallback chain is timed by its own route.

#1967 lifted the orchestrator's inactivity bound to a local route's
``timeout`` only when EVERY candidate of a call is local. One bound around a
whole cloud-primary, local-fallback chain could not do both jobs: catch a hung
cloud primary at the short default, and let a slow local fallback run to its
own route's timeout.

The bound now follows the candidate being awaited. The service hands each
candidate attempt to the call's ``LLMCallWatchdog``, which bounds a streamed
attempt by its wait for its first item and then by each wait between items,
and a non-streaming attempt by its silence until it answers. A candidate that
goes its bound without progress fails inside the dispatch loop, which moves on
to its next admitted candidate exactly as after any other failure.

These tests drive the real ``LLMService`` dispatch loops (built with the #2652
fixtures) and the real watchdog over fake routes, on an event loop whose clock
only the routes move, so minutes of provider silence take no wall time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple
from unittest.mock import AsyncMock

import httpx
import pytest

from kestrel_sovereign.agent import orchestrator_engine as oe
from kestrel_sovereign.agent.orchestrator_engine import (
    LLMCallInactivityTimeout,
    LLMCallWatchdog,
)
from kestrel_sdk.llm import ProviderCapabilities

from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.call_progress import (
    FIRST_TOKEN_PHASE,
    INTER_CHUNK_PHASE,
    NON_STREAMING_PHASE,
    AttemptInactivityTimeout,
    bound_attempt_call,
    bound_attempt_stream,
)
from kestrel_sovereign.llm.error_handling import LLMError, LLMProviderError
from kestrel_sovereign.llm.model_cache import get_shared_model_cache
from kestrel_sovereign.llm.output_ceiling import attach_stop_reason, incomplete_generation
from kestrel_sovereign.llm.service import LLMServiceError
from kestrel_sovereign.llm.streaming import LLMStreamingError
from tests.unit.test_provider_policy_2652 import _service
from tests.utils.virtual_clock import VirtualClockLoop

#: The default inactivity bound (``KESTREL_ORCHESTRATOR_TURN_TIMEOUT_SECS``).
DEFAULT = 180.0
#: A local route's configured client ``timeout`` (#1957).
LOCAL_TIMEOUT = 1800.0
#: The granularity at which a fake route's silence passes: a deadline lands
#: at most this late.
STEP = 5.0
MESSAGES = [{"role": "user", "content": "Where do I work?"}]

CLOUD = "anthropic:api"
LOCAL = "llama_cpp:local"
SPARE = "openai:api"
OUTSIDER = "openrouter:api"
PLAN = "openai:plan"


@pytest.fixture(autouse=True)
def _default_bound(monkeypatch):
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", DEFAULT)
    cache = get_shared_model_cache()
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def loop():
    loop = VirtualClockLoop()
    yield loop
    loop.close()


async def _elapse(loop: VirtualClockLoop, seconds: float) -> None:
    """Let ``seconds`` of virtual time pass, a step per loop iteration, so a
    deadline that comes due on the way fires."""
    for _ in range(int(seconds // STEP)):
        loop.advance(STEP)
        await asyncio.sleep(0)


class _Route:
    """A route whose adapter plays ``script``: ``("wait", s)`` lets ``s``
    seconds pass, ``("text", t)`` streams ``t`` (or adds it to the answer),
    ``("fail",)`` raises a provider error, and ``("hang",)`` never ends. A
    script that ends answers with a natural stop."""

    def __init__(
        self,
        loop: VirtualClockLoop,
        name: str,
        *script: Tuple[Any, ...],
        local: bool = False,
        client_timeout: Any = None,
        adapter: str = "tools_stream",
    ) -> None:
        self.loop = loop
        self.name = name
        self.vendor, _, self.route = name.partition(":")
        self.script = script
        self.local = local
        self.client_timeout = client_timeout
        self.calls = 0
        self.cancelled_at: List[float] = []
        self.adapter = _ADAPTERS[adapter](self)

    def as_provider(self) -> dict:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "route": self.route,
            "client": SimpleNamespace(timeout=self.client_timeout),
            "adapter": self.adapter,
            "model": "m",
            "is_cloud": not self.local,
            "is_local": self.local,
            "base_url": None,
            "capabilities": {},
        }

    async def play(self):
        """Each step of the script, as text it produced or ``None``."""
        self.calls += 1
        try:
            for step in self.script:
                if step[0] == "wait":
                    await _elapse(self.loop, step[1])
                elif step[0] == "hang":
                    await _elapse(self.loop, 1_000_000)
                elif step[0] == "fail":
                    raise LLMProviderError(self.vendor, f"{self.name} is down")
                else:
                    yield step[1]
        except asyncio.CancelledError:
            self.cancelled_at.append(self.loop.time())
            raise


class _Adapter:
    def __init__(self, route: _Route) -> None:
        self._route = route

    async def get_streaming_response_with_tools(self, *, client, model, messages, **_):
        text = []
        async for piece in self._route.play():
            text.append(piece)
            yield piece
        yield attach_stop_reason(
            LLMResponse(content="".join(text), input_tokens=1, output_tokens=1),
            "stop",
        )

    async def get_response(self, *, client, model, messages, **_):
        text = [piece async for piece in self._route.play()]
        return attach_stop_reason(
            LLMResponse(content="".join(text), input_tokens=1, output_tokens=1),
            "stop",
        )

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(supports_structured_output=True)


class _CompletionAdapter:
    """An adapter that cannot stream: every streamed call reaches it through
    the service's non-streaming fallback."""

    def __init__(self, route: _Route) -> None:
        self._route = route

    get_response = _Adapter.get_response


class _PlainAdapter(_CompletionAdapter):
    """An adapter with a text stream but no tool-aware one: a call with tools
    reaches it through the non-streaming fallback, one without through its
    text stream."""

    get_streaming_response = _Adapter.get_streaming_response_with_tools


class _CodexAdapter(CodexAdapter):
    """The Codex app-server route's adapter, whose harness owns its transport,
    playing its route's script."""

    def __init__(self, route: _Route) -> None:
        super().__init__()
        self._route = route

    get_streaming_response_with_tools = _Adapter.get_streaming_response_with_tools
    get_response = _Adapter.get_response


_ADAPTERS = {
    "tools_stream": _Adapter,
    "text_stream": _PlainAdapter,
    "completion": _CompletionAdapter,
    "codex": _CodexAdapter,
}


def _chain(*routes: _Route, route_priority=None, **config):
    """The real service over ``routes``, tried in their order."""
    service = _service(list(routes), config={
        "route_priority": route_priority or [route.name for route in routes],
        **config,
    })
    service._ensure_models_discovered = AsyncMock()
    return service


def _streamed(loop, service, *, consumer_pause: float = 0.0, stream=None):
    """Drive ``stream_with_tool_detection`` (or what ``stream`` starts) as a
    turn does, under the turn's watchdog: the items it yielded, each with when
    it arrived, and the error it ended with."""
    items: List[Tuple[Any, float]] = []

    async def drive() -> None:
        watchdog = LLMCallWatchdog.for_call(
            service, model_override=None, force_local_only=False,
        )
        started = (
            stream(service) if stream is not None
            else service.stream_with_tool_detection(messages=MESSAGES)
        )
        async for item in watchdog.watch(started):
            items.append((item, loop.time()))
            if consumer_pause:
                await _elapse(loop, consumer_pause)

    try:
        loop.run_until_complete(drive())
    except Exception as exc:  # noqa: BLE001 - the outcome under test
        return items, exc
    return items, None


def _completed(loop, service) -> Tuple[Optional[Any], Optional[BaseException]]:
    """Drive ``generate_with_messages`` as ``/api/agent/invoke`` does."""

    async def drive():
        watchdog = LLMCallWatchdog.for_call(
            service, model_override=None, force_local_only=False, streaming=False,
        )
        return await watchdog.call(service.generate_with_messages(messages=MESSAGES))

    try:
        return loop.run_until_complete(drive()), None
    except Exception as exc:  # noqa: BLE001 - the outcome under test
        return None, exc


def _text(items) -> str:
    return "".join(item for item, _at in items if isinstance(item, str))


def _content(response: Any) -> Optional[str]:
    """A non-streaming call's answer; without tools the service returns its text."""
    return getattr(response, "content", response)


def _abandoned(caplog) -> List[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING
        and "abandoning this candidate" in record.getMessage()
    ]


# --------------------------------------------------------------------------
# Cloud primary hangs, slow local fallback answers
# --------------------------------------------------------------------------


def test_a_hung_cloud_primary_fails_fast_and_a_slow_local_fallback_streams(
    loop, caplog,
):
    """The ticket's case, streamed. The local fallback is silent for 600s
    before its first token, past the default, and still answers."""
    caplog.set_level(logging.WARNING)
    cloud = _Route(loop, CLOUD, ("hang",), client_timeout=httpx.Timeout(600.0))
    local = _Route(
        loop, LOCAL, ("wait", 600), ("text", "From the local model."),
        local=True, client_timeout=LOCAL_TIMEOUT,
    )

    items, error = _streamed(loop, _chain(cloud, local))

    assert error is None, error
    assert _text(items) == "From the local model."
    # The primary was cancelled at the default, not at the local route's
    # timeout, and asked once: a timeout is no retry.
    assert cloud.calls == 1
    assert len(cloud.cancelled_at) == 1
    assert DEFAULT <= cloud.cancelled_at[0] <= DEFAULT + STEP
    assert local.calls == 1 and local.cancelled_at == []
    assert items[0][1] >= DEFAULT + 600
    [warning] = _abandoned(caplog)
    assert CLOUD in warning and FIRST_TOKEN_PHASE in warning
    assert f"{DEFAULT:g}s" in warning


def test_a_hung_cloud_primary_fails_fast_and_a_slow_local_fallback_answers(
    loop, caplog,
):
    """The same, non-streaming. The cloud client declares no timeout of its
    own, so nothing lifts its bound above the default."""
    caplog.set_level(logging.WARNING)
    cloud = _Route(loop, CLOUD, ("hang",))
    local = _Route(
        loop, LOCAL, ("wait", 1200), ("text", "From the local model."),
        local=True, client_timeout=LOCAL_TIMEOUT,
    )

    response, error = _completed(loop, _chain(cloud, local))

    assert error is None, error
    assert _content(response) == "From the local model."
    assert cloud.calls == 1
    assert DEFAULT <= cloud.cancelled_at[0] <= DEFAULT + STEP
    assert local.cancelled_at == []
    assert loop.time() >= DEFAULT + 1200
    [warning] = _abandoned(caplog)
    assert CLOUD in warning and NON_STREAMING_PHASE in warning
    assert f"{DEFAULT:g}s" in warning


def test_a_non_streaming_candidates_bound_is_its_own_clients_timeout(loop):
    """#3552's rule for a call that is silent until it answers, applied per
    candidate: a cloud candidate's own client timeout lifts its bound, and a
    longer one on another candidate does not."""
    cloud = _Route(loop, CLOUD, ("hang",), client_timeout=httpx.Timeout(600.0))
    local = _Route(
        loop, LOCAL, ("text", "From the local model."),
        local=True, client_timeout=LOCAL_TIMEOUT,
    )

    response, error = _completed(loop, _chain(cloud, local))

    assert error is None, error
    assert _content(response) == "From the local model."
    assert 600 <= cloud.cancelled_at[0] <= 600 + STEP


def test_a_local_fallback_silent_past_its_own_timeout_is_abandoned(loop, caplog):
    """The local fallback's bound is its route's timeout, not unbounded."""
    caplog.set_level(logging.WARNING)
    cloud = _Route(loop, CLOUD, ("hang",))
    local = _Route(loop, LOCAL, ("hang",), local=True, client_timeout=LOCAL_TIMEOUT)

    items, error = _streamed(loop, _chain(cloud, local))

    assert items == []
    assert isinstance(error, LLMError)
    verdict = incomplete_generation(error)
    assert isinstance(verdict, AttemptInactivityTimeout)
    local_started = cloud.cancelled_at[0]
    assert (
        local_started + LOCAL_TIMEOUT
        <= local.cancelled_at[0]
        <= local_started + LOCAL_TIMEOUT + STEP
    )
    warnings = _abandoned(caplog)
    assert len(warnings) == 2
    assert LOCAL in warnings[1] and f"{LOCAL_TIMEOUT:g}s" in warnings[1]


_TOOL = {
    "type": "function",
    "function": {"name": "state_of_mind", "description": "", "parameters": {}},
}
_VERDICT = json.dumps({"risk_level": 1, "reasoning": "Clear and safe."})


#: Each route walk outside the two above, and the adapter that sends the
#: walk down the branch under test.
_DOORS = {
    "stream_with_messages": "tools_stream",
    "stream_with_messages_fallback": "completion",
    "generate_stream": "tools_stream",
    "generate_stream_fallback": "completion",
    "stream_with_tool_detection_text_stream": "text_stream",
    "stream_with_tool_detection_fallback": "text_stream",
    "generate": "tools_stream",
    "get_audit_response": "tools_stream",
}


@pytest.mark.parametrize("door", list(_DOORS))
def test_every_dispatch_loop_times_each_candidate_by_its_route(loop, door):
    """Every route walk the service has hands each candidate attempt to the
    call's bound: the streamed loops and their non-streaming fallbacks,
    ``get_response`` (``generate``: the compaction and context summaries) and
    the audit loop (the genesis auditor)."""
    answer = _VERDICT if door == "get_audit_response" else "From the local model."
    cloud = _Route(loop, CLOUD, ("hang",), adapter=_DOORS[door])
    local = _Route(
        loop, LOCAL, ("wait", 600), ("text", answer),
        local=True, client_timeout=LOCAL_TIMEOUT, adapter=_DOORS[door],
    )
    service = _chain(cloud, local)

    if door in ("generate", "get_audit_response"):
        if door == "generate":
            pending = service.generate(system_prompt="s", user_prompt="u")
        else:
            pending = service.get_audit_response("The response to audit.")

        async def drive():
            return await LLMCallWatchdog.bound_completion(service, pending)

        result = loop.run_until_complete(drive())
        got = result if door == "get_audit_response" else _content(result)
        expected = json.loads(_VERDICT) if door == "get_audit_response" else answer
        assert got == expected
    else:
        def start(svc):
            if door.startswith("stream_with_messages"):
                return svc.stream_with_messages(messages=MESSAGES)
            if door.startswith("generate_stream"):
                return svc.generate_stream(system_prompt="s", user_prompt="u")
            tools = [_TOOL] if door.endswith("fallback") else None
            return svc.stream_with_tool_detection(messages=MESSAGES, tools=tools)

        items, error = _streamed(loop, service, stream=start)
        assert error is None, error
        assert _text(items) == answer

    assert DEFAULT <= cloud.cancelled_at[0] <= DEFAULT + STEP
    assert local.calls == 1 and local.cancelled_at == []
    assert loop.time() >= DEFAULT + 600


# --------------------------------------------------------------------------
# Streaming: an idle watchdog, not a wall clock
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gap", [100.0, 300.0])
def test_a_local_candidate_streaming_slowly_but_steadily_is_not_cut(loop, gap):
    """Every gap is under the local route's bound; the whole answer runs far
    past the default. A gap over the default (300s) is still not a hang for
    the local candidate, though one bound for the whole cloud-primary chain
    cut it there."""
    cloud = _Route(loop, CLOUD, ("fail",))
    pieces = [f"piece{i} " for i in range(10)]
    script = [step for piece in pieces for step in (("wait", gap), ("text", piece))]
    local = _Route(loop, LOCAL, *script, local=True, client_timeout=LOCAL_TIMEOUT)

    items, error = _streamed(loop, _chain(cloud, local))

    assert error is None, error
    assert _text(items) == "".join(pieces)
    assert local.cancelled_at == []
    assert loop.time() >= 10 * gap > DEFAULT


def test_a_consumer_pausing_between_chunks_does_not_trip_the_bound(loop):
    """Only time spent waiting on the route counts. A consumer that takes far
    longer than every bound over each chunk trips none of them."""
    cloud = _Route(loop, CLOUD, ("fail",))
    local = _Route(
        loop, LOCAL, ("text", "one "), ("text", "two "), ("text", "three"),
        local=True, client_timeout=LOCAL_TIMEOUT,
    )

    items, error = _streamed(
        loop, _chain(cloud, local), consumer_pause=2 * LOCAL_TIMEOUT,
    )

    assert error is None, error
    assert _text(items) == "one two three"
    assert local.cancelled_at == []
    assert loop.time() >= 3 * 2 * LOCAL_TIMEOUT


def test_a_local_candidate_that_stops_mid_stream_trips_the_inter_chunk_bound(
    loop, caplog,
):
    """A candidate silent after its first chunk is abandoned one bound after
    it, and nothing falls back: the next route's answer would be appended to
    the text already sent."""
    caplog.set_level(logging.WARNING)
    local = _Route(
        loop, LOCAL, ("wait", 30), ("text", "Your employer is"), ("hang",),
        local=True, client_timeout=LOCAL_TIMEOUT,
    )
    spare = _Route(loop, SPARE, ("text", "A whole other answer."))

    items, error = _streamed(loop, _chain(local, spare))

    assert _text(items) == "Your employer is"
    assert isinstance(error, LLMError)
    verdict = incomplete_generation(error)
    assert isinstance(verdict, AttemptInactivityTimeout)
    assert (verdict.provider, verdict.phase) == (LOCAL, INTER_CHUNK_PHASE)
    assert verdict.timeout == LOCAL_TIMEOUT
    chunk_at = items[0][1]
    assert chunk_at + LOCAL_TIMEOUT <= local.cancelled_at[0] <= chunk_at + LOCAL_TIMEOUT + STEP
    assert spare.calls == 0
    [warning] = _abandoned(caplog)
    assert LOCAL in warning and INTER_CHUNK_PHASE in warning
    assert f"{LOCAL_TIMEOUT:g}s" in warning


def test_a_cloud_candidate_silent_mid_stream_is_cut_at_the_default(loop):
    cloud = _Route(loop, CLOUD, ("text", "Your employer is"), ("hang",))

    items, error = _streamed(loop, _chain(cloud))

    assert _text(items) == "Your employer is"
    verdict = incomplete_generation(error)
    assert isinstance(verdict, AttemptInactivityTimeout)
    assert (verdict.phase, verdict.timeout) == (INTER_CHUNK_PHASE, DEFAULT)
    assert DEFAULT <= cloud.cancelled_at[0] <= DEFAULT + STEP


# --------------------------------------------------------------------------
# A timeout moves only to a candidate the chain already admitted (#2652)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("door", ["stream", "invoke"])
def test_a_timeout_moves_on_to_the_next_approved_route_only(loop, door):
    cloud = _Route(loop, CLOUD, ("hang",))
    spare = _Route(loop, SPARE, ("text", "From the spare route."))
    outsider = _Route(loop, OUTSIDER, ("text", "From outside the chain."))
    service = _chain(cloud, spare, outsider, route_priority=[CLOUD, SPARE])

    if door == "stream":
        items, error = _streamed(loop, service)
        answer = _text(items)
    else:
        response, error = _completed(loop, service)
        answer = _content(response)

    assert error is None, error
    assert answer == "From the spare route."
    assert cloud.cancelled_at and outsider.calls == 0


@pytest.mark.parametrize("door", ["stream", "invoke"])
def test_a_timeout_on_the_last_approved_route_never_reaches_an_outsider(loop, door):
    """The 2026-07-20 lineup with the approved route hung rather than down:
    its timeout spends the approved chain, and the healthy outsider is never
    asked."""
    cloud = _Route(loop, CLOUD, ("hang",))
    outsider = _Route(loop, OUTSIDER, ("text", "From outside the chain."))
    service = _chain(cloud, outsider, route_priority=[CLOUD])

    if door == "stream":
        items, error = _streamed(loop, service)
        assert items == []
    else:
        _response, error = _completed(loop, service)

    assert isinstance(error, LLMError)
    assert isinstance(incomplete_generation(error), AttemptInactivityTimeout)
    assert cloud.cancelled_at and outsider.calls == 0


@pytest.mark.parametrize("door", ["stream", "invoke"])
def test_a_timeout_never_reaches_a_denied_provider(loop, door):
    cloud = _Route(loop, CLOUD, ("hang",))
    denied = _Route(loop, OUTSIDER, ("text", "From a denied provider."))
    service = _chain(cloud, denied, denied_providers=["openrouter"])

    if door == "stream":
        _items, error = _streamed(loop, service)
    else:
        _response, error = _completed(loop, service)

    assert isinstance(error, LLMError)
    assert cloud.cancelled_at and denied.calls == 0


def test_a_timeout_on_an_explicitly_selected_route_does_not_fall_through(loop):
    """No silent fallback from a route the caller selected."""
    cloud = _Route(loop, CLOUD, ("hang",))
    local = _Route(loop, LOCAL, ("text", "From the local model."), local=True)
    service = _chain(cloud, local)
    service._mandate_preference = {"vendor": "anthropic", "model": "m", "route": None}

    items, error = _streamed(loop, service)

    assert items == []
    assert isinstance(error, LLMError)
    assert isinstance(incomplete_generation(error), AttemptInactivityTimeout)
    assert local.calls == 0


def _generated(loop, service) -> Tuple[Optional[Any], Optional[BaseException]]:
    """Drive ``generate`` (``get_response``'s route walk) as the compaction
    and context summaries do."""

    async def drive():
        return await LLMCallWatchdog.bound_completion(
            service, service.generate(system_prompt="s", user_prompt="u"),
        )

    try:
        return loop.run_until_complete(drive()), None
    except Exception as exc:  # noqa: BLE001 - the outcome under test
        return None, exc


@pytest.mark.parametrize("door", ["stream", "invoke", "generate"])
def test_a_stalled_codex_candidate_does_not_rotate_to_another_model(
    loop, caplog, door,
):
    """The harness owns the Codex route's transport (#1429): a stall there is
    not evidence the route is broken, and answering from another model would
    give the user a wrong-model answer. Abandoned on its own bound, it fails
    the call rather than rotating, as on the harness's own idle stall. Before
    #1968 the call's one bound ended it outright; the non-streaming loops
    (``generate_with_messages`` behind invoke, ``get_response`` behind
    ``generate``) must not now fall through where the streamed ones refuse."""
    caplog.set_level(logging.INFO)
    plan = _Route(loop, PLAN, ("hang",), adapter="codex")
    local = _Route(loop, LOCAL, ("text", "From the local model."), local=True)
    service = _chain(plan, local)

    if door == "stream":
        items, error = _streamed(loop, service)
        assert items == []
        assert isinstance(error, LLMStreamingError)
    else:
        drive = _completed if door == "invoke" else _generated
        response, error = drive(loop, service)
        assert response is None
        assert isinstance(error, LLMServiceError)

    assert "Harness-owned route openai:plan" in str(error)
    verdict = incomplete_generation(error)
    assert isinstance(verdict, AttemptInactivityTimeout)
    assert verdict.provider == PLAN
    assert DEFAULT <= plan.cancelled_at[0] <= DEFAULT + STEP
    # The fallback is never asked, nor even attempted.
    assert local.calls == 0
    assert not [
        record for record in caplog.records
        if LOCAL in record.getMessage()
    ]


# --------------------------------------------------------------------------
# The watchdog: the call's bound and its candidates' bounds
# --------------------------------------------------------------------------


def _local_route(timeout: float = LOCAL_TIMEOUT) -> dict:
    return {"name": LOCAL, "is_local": True, "client": SimpleNamespace(timeout=timeout)}


def test_a_candidates_bound_follows_its_own_route():
    """The default, lifted to a local candidate's client timeout; on a call
    that does not stream, to any candidate's (#1966, #3552)."""
    cloud = {"name": CLOUD, "is_local": False, "client": SimpleNamespace(
        timeout=httpx.Timeout(600.0),
    )}
    service = _chain()
    streamed = LLMCallWatchdog.for_call(
        service, model_override=None, force_local_only=False,
    )
    completed = LLMCallWatchdog.for_call(
        service, model_override=None, force_local_only=False, streaming=False,
    )

    assert streamed.attempt_timeout(cloud) == DEFAULT
    assert streamed.attempt_timeout(_local_route()) == LOCAL_TIMEOUT
    assert streamed.attempt_timeout(_local_route(60.0)) == DEFAULT
    assert completed.attempt_timeout(cloud) == 600.0
    assert completed.attempt_timeout(_local_route()) == LOCAL_TIMEOUT
    # A watchdog with no route policy bounds every candidate by its own bound.
    assert LLMCallWatchdog(42.0).attempt_timeout(cloud) == 42.0


def test_the_calls_bound_covers_the_wait_outside_its_candidates(loop):
    """While a candidate is awaited its bound replaces the call's; once the
    candidate answers, the call's bound runs again, over its whole window."""
    watchdog = LLMCallWatchdog(
        DEFAULT, attempt_timeout=lambda _route: LOCAL_TIMEOUT,
    )

    async def call():
        await bound_attempt_call(
            _local_route(), LOCAL, "m", _elapse(loop, 2 * DEFAULT),
        )
        answered = loop.time()
        await _elapse(loop, 1_000_000)
        return answered

    with pytest.raises(LLMCallInactivityTimeout):
        loop.run_until_complete(watchdog.call(call()))
    assert 3 * DEFAULT <= loop.time() <= 3 * DEFAULT + STEP


def test_an_attempt_on_another_task_is_not_the_calls_candidate(loop):
    """An inline tool's own model call runs on its transport's task with a
    copy of the call's context; its attempts are not this call's candidates."""
    watchdog = LLMCallWatchdog(DEFAULT)
    stream = object()

    async def call():
        here = bound_attempt_stream(_local_route(), LOCAL, "m", stream)
        elsewhere = await asyncio.get_running_loop().create_task(
            _bound_elsewhere(stream)
        )
        return here, elsewhere

    async def _bound_elsewhere(stream):
        return bound_attempt_stream(_local_route(), LOCAL, "m", stream)

    here, elsewhere = loop.run_until_complete(watchdog.call(call()))
    assert here is not stream
    assert elsewhere is stream
    loop.run_until_complete(here.aclose())


def test_an_attempt_outside_a_bounded_call_runs_as_it_is():
    stream = object()
    assert bound_attempt_stream(_local_route(), LOCAL, "m", stream) is stream
    assert bound_attempt_stream(None, LOCAL, "m", stream) is stream

    async def answer():
        return "answer"

    assert asyncio.run(bound_attempt_call(_local_route(), LOCAL, "m", answer())) == "answer"
