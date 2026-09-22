"""Every turn span ends, and says HOW the turn ended (#3159 R4).

Before this, a turn span answered only *that* it ended — and on several paths
not even that. The streaming path's ``except Exception`` / ``else`` pair missed
``CancelledError`` (what a hard Stop raises), ``GeneratorExit`` (what a
consumer that walked away raises) and the safe-mode early ``return`` (a
``return`` inside ``try`` skips ``else``), so those spans were never ended and
never exported: a stopped turn and a disconnected one both read as "nothing
arrived". A cooperative checkpoint Stop returned normally and ended UNSET,
identical to a completed turn; a pre-admission Stop raised and ended ERROR,
identical to a failure.

These drive the REAL turn methods with a REAL in-memory OTel exporter — not a
mock of ``end_span``, which would assert the call rather than the export — and
one test per exit path asserts the span ENDED and carries the right
``kestrel.turn.outcome``.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from kestrel_sovereign import telemetry
from kestrel_sovereign.agent.invocation import (
    InvocationCancelledError,
    InvocationSelfFencedError,
)
from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
from kestrel_sovereign.agent.streaming import StreamingMixin
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.agent.turn_outcome import (
    TurnOutcome,
    resolve_turn_outcome,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent

# Spelled literally: a rename of the constant must not silently satisfy a test
# whose consumer is still reading the old string.
TURN_OUTCOME = "kestrel.turn.outcome"
SESSION = "2f2a0b1c-0000-4000-8000-0000000000aa"
REQUEST_ID = "req-stop-me"
STREAM_SPAN = "agent.process_input_streaming"
INVOKE_SPAN = "agent.process_input"


@pytest.fixture
def span_exporter(monkeypatch):
    """Install an in-memory-exporter-backed tracer as the lifecycle tracer."""

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        telemetry, "_tracer", provider.get_tracer("test-turn-outcome")
    )
    return exporter


def _turn_host(turn_id="turn_1", *, lifecycle_error=None, **overrides):
    """A minimal host running the REAL turn methods, stubbed below the span."""

    host = MagicMock()
    host.agent_name = "Emma"
    host.did = "did:pkh:eip155:1:0xabc"
    host._safe_mode = False
    host._constitution_audit_pending = False
    host._maybe_refresh_user_byok_resolver = AsyncMock()
    host._genesis_audit_cognition_block = AsyncMock(return_value=None)
    host._maybe_audit = AsyncMock()
    host.bootstrap_service = None
    host._turn_trace_identities = {}
    host._turn_outcome_listeners = []
    host._get_privacy_transition_lock = lambda: asyncio.Lock()

    # No Stop was recorded unless a test says so. These are real sets, which is
    # also what stops a MagicMock's truthy attribute from reading as evidence.
    host._current_request_id = REQUEST_ID
    host._active_request_generations = {REQUEST_ID: 1}
    host._cancelled_request_generations = set()
    host._cancelled_requests = set()
    host._self_fenced_request_generations = set()

    for name, source in (
        ("_turn_trace_index", TurnLifecycleMixin._turn_trace_index),
        ("get_current_turn_id", TurnLifecycleMixin.get_current_turn_id),
        (
            "bind_current_turn_trace_identity",
            TurnLifecycleMixin.bind_current_turn_trace_identity,
        ),
        ("bind_current_turn_span", TurnLifecycleMixin.bind_current_turn_span),
        (
            "_turn_outcome_listener_registry",
            TurnLifecycleMixin._turn_outcome_listener_registry,
        ),
        (
            "add_turn_outcome_listener",
            TurnLifecycleMixin.add_turn_outcome_listener,
        ),
        (
            "_request_generation_for_current_task",
            RequestLifecycleMixin._request_generation_for_current_task,
        ),
        ("is_request_cancelled", RequestLifecycleMixin.is_request_cancelled),
        (
            "is_request_self_fenced",
            RequestLifecycleMixin.is_request_self_fenced,
        ),
    ):
        setattr(host, name, source.__get__(host))

    @asynccontextmanager
    async def lifecycle():
        # The real lifecycle publishes the canonical turn address BEFORE the
        # durable admission check that a pre-registration Stop raises from, so
        # the address exists even on that path.
        with telemetry.turn_span_scope(turn_id):
            host._live_turn_id = turn_id
            try:
                if lifecycle_error is not None:
                    raise lifecycle_error
                yield turn_id
            finally:
                host._turn_trace_identities.pop(turn_id, None)
                host._live_turn_id = None

    host._turn_lifecycle = lifecycle
    for key, value in overrides.items():
        setattr(host, key, value)
    return host


def _record_stop(host):
    """Record what the request lifecycle records when a Stop is acknowledged."""

    host._cancelled_request_generations.add((REQUEST_ID, 1))
    host._cancelled_requests.add(REQUEST_ID)


def _streaming(host, body):
    host._process_input_streaming_traced_locked = body
    host.process_input_streaming = (
        StreamingMixin.process_input_streaming.__get__(host)
    )
    # Both entry points bind their invocation id AS the request id
    # (``bind_async_generator_invocation("request_id")``), and that is the id
    # the request lifecycle registers and a Stop names. Supplying it here is
    # what lets a test record the Stop against the same delivery the turn ran
    # under, rather than a generated one it could never match.
    return host.process_input_streaming(
        "what is 6*7?", session_id=SESSION, request_id=REQUEST_ID
    )


async def _drain(stream):
    return [chunk async for chunk in stream]


async def _run_invoke(host, traced_locked):
    host._process_input_traced_locked = traced_locked
    host.process_input = KestrelAgent.process_input.__get__(host)
    return await host.process_input(
        "what is 6*7?", session_id=SESSION, invocation_id=REQUEST_ID
    )


def _only(exporter, name):
    spans = [s for s in exporter.get_finished_spans() if s.name == name]
    assert len(spans) == 1, f"expected one {name} span, exported {len(spans)}"
    return spans[0]


async def _answer(*_a, **_k):
    yield "42"


class TestStreamingTurnEndsOnEveryExit:
    @pytest.mark.asyncio
    async def test_normal_completion_is_completed(self, span_exporter):
        host = _turn_host()
        assert await _drain(_streaming(host, _answer)) == ["42"]

        span = _only(span_exporter, STREAM_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "completed"
        assert span.status.status_code is not StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_exception_is_failed_and_is_the_only_error_status(
        self, span_exporter
    ):
        async def boom(*_a, **_k):
            raise ValueError("provider exploded")
            yield  # pragma: no cover - generator marker

        host = _turn_host()
        with pytest.raises(ValueError):
            await _drain(_streaming(host, boom))

        span = _only(span_exporter, STREAM_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "failed"
        assert span.status.status_code is StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_cooperative_checkpoint_stop_returns_normally_but_reads_stopped(
        self, span_exporter
    ):
        """The case that used to be indistinguishable from `completed`.

        A checkpoint Stop is observed by the turn, which then returns normally.
        Nothing about the exception (there is none) says it was stopped; the
        request lifecycle's own record does.
        """

        async def stopped_at_checkpoint(*_a, **_k):
            _record_stop(host)
            yield "partial"

        host = _turn_host()
        assert await _drain(_streaming(host, stopped_at_checkpoint)) == [
            "partial"
        ]

        span = _only(span_exporter, STREAM_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "stopped"
        # Stopped is an operator act, not a defect.
        assert span.status.status_code is not StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_hard_cancellation_from_a_stop_reads_stopped(
        self, span_exporter
    ):
        async def cancelled(*_a, **_k):
            _record_stop(host)
            raise asyncio.CancelledError()
            yield  # pragma: no cover - generator marker

        host = _turn_host()
        with pytest.raises(asyncio.CancelledError):
            await _drain(_streaming(host, cancelled))

        span = _only(span_exporter, STREAM_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "stopped"
        assert span.status.status_code is not StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_pre_admission_stop_is_stopped_not_failed(self, span_exporter):
        """The durable admission refusal raises; it is still a Stop.

        This is the path that used to end ERROR and read as a failure.
        """

        host = _turn_host(
            lifecycle_error=InvocationCancelledError("stopped before admission")
        )
        _record_stop(host)
        with pytest.raises(InvocationCancelledError):
            await _drain(_streaming(host, _answer))

        span = _only(span_exporter, STREAM_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "stopped"
        assert span.status.status_code is not StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_consumer_that_walks_away_is_disconnected(self, span_exporter):
        async def forever(*_a, **_k):
            while True:
                yield "tick"

        host = _turn_host()
        stream = _streaming(host, forever)
        assert await stream.__anext__() == "tick"
        # GeneratorExit: what a closed response body raises into the generator.
        await stream.aclose()

        span = _only(span_exporter, STREAM_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "disconnected"
        assert span.status.status_code is not StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_host_shutdown_cancel_is_interrupted_not_stopped(
        self, span_exporter
    ):
        """Cancelled, but by neither a Stop nor the consumer."""

        async def cancelled(*_a, **_k):
            raise asyncio.CancelledError()
            yield  # pragma: no cover - generator marker

        host = _turn_host()
        with pytest.raises(asyncio.CancelledError):
            await _drain(_streaming(host, cancelled))

        span = _only(span_exporter, STREAM_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "interrupted"
        assert span.status.status_code is not StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_safe_mode_early_return_still_ends_the_span(
        self, span_exporter, monkeypatch
    ):
        """A ``return`` inside ``try`` skips ``else`` — this span used to leak.

        Safe Mode is rechecked INSIDE the lifecycle because a transition can
        latch it while the turn is queued for that boundary. That recheck is
        the branch after the span exists, so the double answers "clear" once
        and "latched" on the recheck, exactly as a transition would.
        """

        answers = iter([None, "[safe mode]"])
        monkeypatch.setattr(
            "kestrel_sovereign.agent.constitution.safe_mode_cognition_block",
            lambda _agent, _text: next(answers),
        )
        host = _turn_host()
        assert await _drain(_streaming(host, _answer)) == ["[safe mode]"]

        span = _only(span_exporter, STREAM_SPAN)
        # The turn ran to its own end and delivered its refusal.
        assert dict(span.attributes)[TURN_OUTCOME] == "completed"


class TestInvokeTurnEndsOnEveryExit:
    @pytest.mark.asyncio
    async def test_normal_completion_is_completed(self, span_exporter):
        host = _turn_host()
        assert await _run_invoke(host, AsyncMock(return_value="42")) == "42"

        span = _only(span_exporter, INVOKE_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "completed"
        assert span.status.status_code is not StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_exception_is_failed(self, span_exporter):
        host = _turn_host()
        with pytest.raises(ValueError):
            await _run_invoke(host, AsyncMock(side_effect=ValueError("boom")))

        span = _only(span_exporter, INVOKE_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "failed"
        assert span.status.status_code is StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_cancellation_after_a_stop_is_not_an_error(self, span_exporter):
        host = _turn_host()

        async def cancelled(*_a, **_k):
            _record_stop(host)
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await _run_invoke(host, cancelled)

        span = _only(span_exporter, INVOKE_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "stopped"
        # OpenTelemetry's default would have marked this ERROR.
        assert span.status.status_code is not StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_cancellation_without_a_stop_is_interrupted(self, span_exporter):
        host = _turn_host()

        async def cancelled(*_a, **_k):
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await _run_invoke(host, cancelled)

        span = _only(span_exporter, INVOKE_SPAN)
        assert dict(span.attributes)[TURN_OUTCOME] == "interrupted"


class TestOutcomeIsNotDerivedFromTheExceptionType:
    """The rule the classifier exists to keep: a type is not an authority."""

    def test_identical_exception_classifies_two_ways(self):
        stopped_host = _turn_host()
        _record_stop(stopped_host)
        interrupted_host = _turn_host()
        one_exception = asyncio.CancelledError()

        assert (
            resolve_turn_outcome(stopped_host, one_exception)
            is TurnOutcome.STOPPED
        )
        assert (
            resolve_turn_outcome(interrupted_host, one_exception)
            is TurnOutcome.INTERRUPTED
        )

    def test_a_self_fenced_lease_is_not_an_operator_stop(self):
        """Infrastructure losing a lease is not evidence anybody asked."""

        host = _turn_host()
        _record_stop(host)
        host._self_fenced_request_generations.add((REQUEST_ID, 1))

        assert (
            resolve_turn_outcome(host, InvocationSelfFencedError("lease lost"))
            is TurnOutcome.INTERRUPTED
        )

    def test_a_host_without_lifecycle_state_records_no_stop(self):
        """A duck-typed host has no Stop record; a mock's truthiness is not one."""

        bare = MagicMock()
        bare._current_request_id = REQUEST_ID
        assert resolve_turn_outcome(bare, None) is TurnOutcome.COMPLETED


class TestReceiptEvidenceNeverLandsOnASpan:
    """R1: the receipt is the only authority; the span carries no part of it."""

    @pytest.mark.asyncio
    async def test_no_attribute_carries_a_receipt_reason_or_actor(
        self, span_exporter
    ):
        receipt_reason = "runaway loop burning the wallet"
        receipt_actor = "did:pkh:eip155:1:0xSOVEREIGN"

        async def stopped(*_a, **_k):
            _record_stop(host)
            yield "partial"

        host = _turn_host()
        await _drain(_streaming(host, stopped))

        span = _only(span_exporter, STREAM_SPAN)
        attributes = dict(span.attributes)
        assert attributes[TURN_OUTCOME] == "stopped"
        # By structure, not by wording: no KEY is about a reason or an actor...
        assert not [
            key
            for key in attributes
            if "reason" in key.lower() or "actor" in key.lower()
        ]
        # ...and no VALUE is the receipt's own content.
        for value in attributes.values():
            assert receipt_reason not in str(value)
            assert receipt_actor not in str(value)


class TestFeatureTurnRootReceivesTheSameOutcome:
    """R4: one outcome, computed once in core, delivered to both spans.

    The feature's turn root ends on the SDK ``Stop`` hook, which the strict
    cancel paths skip — so it cannot derive this for itself and must be told.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "make_body,expected,raises",
        [
            (lambda host: _answer, "completed", None),
            (
                lambda host: _stop_then(host, asyncio.CancelledError()),
                "stopped",
                asyncio.CancelledError,
            ),
            (lambda host: _raise(ValueError("boom")), "failed", ValueError),
        ],
    )
    async def test_listener_is_called_on_every_exit(
        self, span_exporter, make_body, expected, raises
    ):
        seen = []
        host = _turn_host()
        host.add_turn_outcome_listener(
            lambda turn_id, outcome: seen.append((turn_id, outcome))
        )

        stream = _streaming(host, make_body(host))
        if raises is None:
            await _drain(stream)
        else:
            with pytest.raises(raises):
                await _drain(stream)

        assert seen == [("turn_1", TurnOutcome(expected))]
        # A str enum, so a feature compares it without importing core.
        assert seen[0][1] == expected

    @pytest.mark.asyncio
    async def test_a_raising_listener_cannot_fail_the_turn(self, span_exporter):
        host = _turn_host()

        def explode(_turn_id, _outcome):
            raise RuntimeError("the observability feature is broken")

        host.add_turn_outcome_listener(explode)

        assert await _drain(_streaming(host, _answer)) == ["42"]
        assert dict(_only(span_exporter, STREAM_SPAN).attributes)[
            TURN_OUTCOME
        ] == "completed"

    @pytest.mark.asyncio
    async def test_pre_admission_stop_still_names_its_turn(self, span_exporter):
        """The address exists before the refusal; the listener must get it."""

        seen = []
        host = _turn_host(
            lifecycle_error=InvocationCancelledError("stopped before admission")
        )
        _record_stop(host)
        host.add_turn_outcome_listener(
            lambda turn_id, outcome: seen.append((turn_id, outcome))
        )

        with pytest.raises(InvocationCancelledError):
            await _drain(_streaming(host, _answer))

        assert seen == [("turn_1", TurnOutcome.STOPPED)]


def _stop_then(host, error):
    async def body(*_a, **_k):
        _record_stop(host)
        raise error
        yield  # pragma: no cover - generator marker

    return body


def _raise(error):
    async def body(*_a, **_k):
        raise error
        yield  # pragma: no cover - generator marker

    return body
