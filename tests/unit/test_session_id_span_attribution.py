"""Session-id span attribution on the agent runtime's turn spans (issue #2916).

The fleet Timeline groups spans into per-session bands and reads the session id
from ``kestrel.session_id`` / ``session.id`` / ``kestrel.run_id`` only. The agent
runtime stamped its turn spans with ``agent.session_id`` — none of those — so
``sessionKeyOf`` returned null and the Timeline fell back to grouping by trace
id. Every turn is its own trace, so every turn became its own band and the bands
stacked: a native-runtime lane rendered as a diagonal staircase instead of one
compact, packed band.

These tests drive real turns through ``process_input`` and
``process_input_streaming`` with an in-memory OTel exporter installed (the
issue #2602 precedent in ``test_agent_name_span_attribution.py``) and assert the
root span carries ``kestrel.session_id`` — and that two turns in one session
carry the SAME value, which is the property that collapses the staircase into a
single band. The ``agent.feature_dispatch`` spans, which carried no session id
at all, are covered in the same file.
"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from kestrel_sdk.hooks.base import HookOutput

from kestrel_sovereign import telemetry
from kestrel_sovereign.agent.orchestrator_engine import (
    OrchestratorEngineMixin,
    real_session_id,
)
from kestrel_sovereign.agent.streaming import StreamingMixin
from kestrel_sovereign.kestrel_agent import KestrelAgent


# The key the Timeline's ``sessionKeyOf`` actually reads. Spelled literally so a
# rename of the constant cannot silently satisfy these tests while the consumer
# keeps looking for the old string.
KESTREL_SESSION_ID = "kestrel.session_id"
LEGACY_SESSION_ID = "agent.session_id"

SESSION = "e9286c5f-7a20-485f-887a-2eaf250d3664"


@pytest.fixture
def span_exporter(monkeypatch):
    """Install an in-memory-exporter-backed tracer as the lifecycle tracer.

    ``optional_span`` / ``start_span`` read ``telemetry.get_tracer()``, which is
    ``None`` under the autouse tracing fixture in ``conftest`` — so a test must
    opt in explicitly, and only ever with an in-memory exporter (#2704).
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("test-session-id"))
    return exporter


@asynccontextmanager
async def _noop_lifecycle():
    yield


def _turn_host(**overrides):
    """A minimal host for the real turn methods, stubbed below the span."""
    host = MagicMock()
    host.agent_name = "Emma"
    host.did = "did:pkh:eip155:1:0xabc"
    host._safe_mode = False
    host._constitution_audit_pending = False
    host._maybe_refresh_user_byok_resolver = AsyncMock()
    host._genesis_audit_cognition_block = AsyncMock(return_value=None)
    host._maybe_audit = AsyncMock()
    host._turn_lifecycle = _noop_lifecycle
    host.bootstrap_service = None
    for key, value in overrides.items():
        setattr(host, key, value)
    return host


async def _run_process_input(host, session_id):
    host.process_input = KestrelAgent.process_input.__get__(host)
    return await host.process_input("what is 6*7?", session_id=session_id)


async def _run_process_input_streaming(host, session_id):
    async def _traced_locked(*_a, **_k):
        yield "42"

    host._process_input_streaming_traced_locked = _traced_locked
    host._get_privacy_transition_lock = lambda: asyncio.Lock()
    host.process_input_streaming = (
        StreamingMixin.process_input_streaming.__get__(host)
    )
    return [
        chunk
        async for chunk in host.process_input_streaming(
            "what is 6*7?", session_id=session_id
        )
    ]


def _roots(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


class TestTurnSpansCarryTimelineSessionKey:
    """Both turn entry points stamp the key the Timeline groups on."""

    @pytest.mark.asyncio
    async def test_process_input_stamps_kestrel_session_id(self, span_exporter):
        host = _turn_host(
            _process_input_traced_locked=AsyncMock(return_value="42"),
        )
        assert await _run_process_input(host, SESSION) == "42"

        spans = _roots(span_exporter, "agent.process_input")
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs.get(KESTREL_SESSION_ID) == SESSION
        # Stamped ALONGSIDE the pre-existing key, not instead of it — anything
        # still reading ``agent.session_id`` keeps working.
        assert attrs.get(LEGACY_SESSION_ID) == SESSION

    @pytest.mark.asyncio
    async def test_process_input_streaming_stamps_kestrel_session_id(
        self, span_exporter
    ):
        host = _turn_host()
        assert await _run_process_input_streaming(host, SESSION) == ["42"]

        spans = _roots(span_exporter, "agent.process_input_streaming")
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs.get(KESTREL_SESSION_ID) == SESSION
        assert attrs.get(LEGACY_SESSION_ID) == SESSION

    @pytest.mark.asyncio
    async def test_two_turns_in_one_session_share_the_value(self, span_exporter):
        """The property that collapses the staircase into a single band.

        Each turn is its own trace, so the Timeline can only pack them onto one
        lane track if they agree on a session key. Both transports must produce
        the same value for the same session.
        """
        non_streaming = _turn_host(
            _process_input_traced_locked=AsyncMock(return_value="42"),
        )
        await _run_process_input(non_streaming, SESSION)
        await _run_process_input(_turn_host(
            _process_input_traced_locked=AsyncMock(return_value="43"),
        ), SESSION)
        await _run_process_input_streaming(_turn_host(), SESSION)

        roots = [
            s
            for s in span_exporter.get_finished_spans()
            if s.name in ("agent.process_input", "agent.process_input_streaming")
        ]
        assert len(roots) == 3
        # Distinct traces...
        assert len({s.context.trace_id for s in roots}) == 3
        # ...but one session key, so they land in ONE band.
        assert {dict(s.attributes).get(KESTREL_SESSION_ID) for s in roots} == {
            SESSION
        }

    @pytest.mark.asyncio
    async def test_sessionless_turn_omits_the_attribute_entirely(
        self, span_exporter
    ):
        """Never stamp the empty string.

        ``optional_span``/``start_span`` drop ``None`` but export ``""``. An
        empty attribute reads the same as an absent one to ``sessionKeyOf``, so
        exporting it is pure noise.
        """
        host = _turn_host(
            _process_input_traced_locked=AsyncMock(return_value="42"),
        )
        await _run_process_input(host, None)
        await _run_process_input_streaming(_turn_host(), None)

        roots = [
            s
            for s in span_exporter.get_finished_spans()
            if s.name in ("agent.process_input", "agent.process_input_streaming")
        ]
        assert len(roots) == 2
        for span in roots:
            assert KESTREL_SESSION_ID not in dict(span.attributes)


class _AllowAllHooks:
    async def execute_hooks(self, _event, _hook_input):
        return HookOutput.allow()

    async def execute_hooks_parallel(self, _event, _hook_input):
        return None


class _Agent(OrchestratorEngineMixin):
    pass


def _dispatch_agent():
    agent = _Agent()
    agent.agent_name = "Emma"
    agent.hooks_manager = _AllowAllHooks()
    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_response = AsyncMock()
    agent._get_denied_tools = AsyncMock(return_value=set())
    agent._register_explored_feature_tools = MagicMock()
    agent._reemit_envelope_parts = MagicMock()
    agent._emit_tool_update_event = AsyncMock()
    agent._fire_post_subagent_hook = AsyncMock()
    return agent


def _subagent_feature():
    feature = MagicMock()
    feature.tool_name = "memory_feature"
    feature.execute_as_subagent = AsyncMock(return_value={"success": True})
    return feature


class TestFeatureDispatchSpansCarrySession:
    """``agent.feature_dispatch`` joins the band of the turn that drove it."""

    @pytest.mark.asyncio
    async def test_named_subagent_dispatch_stamps_session(self, span_exporter):
        agent = _dispatch_agent()
        result = await agent._execute_named_subagent(
            _subagent_feature(),
            tool_name="memory_feature",
            args={"task": "recall"},
            session_id=SESSION,
            source="inline-executor",
        )
        assert result == {"success": True}

        spans = _roots(span_exporter, "agent.feature_dispatch")
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs.get(KESTREL_SESSION_ID) == SESSION
        assert attrs.get(LEGACY_SESSION_ID) == SESSION

    @pytest.mark.asyncio
    async def test_chat_path_feature_dispatch_stamps_session(self, span_exporter):
        agent = _dispatch_agent()
        result = await agent._dispatch_feature_tool(
            tool_call=SimpleNamespace(name="memory_feature"),
            feature=_subagent_feature(),
            args={"task": "recall"},
            dispatch_start=0.0,
            dispatch_event_id="evt-feature",
            user_message="remember this",
            session_id=SESSION,
        )
        assert result == {"success": True}

        spans = _roots(span_exporter, "agent.feature_dispatch")
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs.get(KESTREL_SESSION_ID) == SESSION
        assert attrs.get(LEGACY_SESSION_ID) == SESSION

    @pytest.mark.asyncio
    async def test_sentinel_session_is_not_stamped(self, span_exporter):
        """``session_id`` doubles as a source tag; sentinels are not sessions.

        ``"orchestrator"`` is the parameter's own default. Stamping it would
        invent a fleet-wide session that every sessionless dispatch on every
        agent shares — worse than the staircase this fix removes.
        """
        agent = _dispatch_agent()
        await agent._dispatch_feature_tool(
            tool_call=SimpleNamespace(name="memory_feature"),
            feature=_subagent_feature(),
            args={"task": "recall"},
            dispatch_start=0.0,
            dispatch_event_id="evt-feature",
            user_message="remember this",
        )

        spans = _roots(span_exporter, "agent.feature_dispatch")
        assert len(spans) == 1
        assert KESTREL_SESSION_ID not in dict(spans[0].attributes)


class TestRealSessionId:
    """The one definition of "is this an actual session?"."""

    def test_sentinels_and_empty_are_not_sessions(self):
        for value in ("", "original", "orchestrator", None):
            assert real_session_id(value) is None

    def test_a_real_session_passes_through(self):
        assert real_session_id(SESSION) == SESSION
