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

Issue #2940 finishes the job. #2916 covered 3 of the 6 root span types an agent
emits, so the lane still rendered as a staircase: ``llm.generate`` could not name
a session AT ALL (``generate()`` had no ``session_id`` parameter, unlike
``generate_with_messages``), and ``signal.dispatch.cognition`` never stamped the
session the cognition was filed from. One real session sprayed across five bands.
The classes below the turn-span ones assert both entry points now agree on BOTH
session keys — ``kestrel.session_id`` (what the Timeline groups on) and
``session.id`` (what Phoenix's own session view reads) — and that a dispatched
cognition joins the band of the session that filed it.
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
from kestrel_sdk.tracing import KestrelTracer

from kestrel_sdk.features.base import tool
from kestrel_sdk.hooks.base import HookOutput
from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
)

from kestrel_sovereign import telemetry
from kestrel_sovereign.agent.orchestrator_engine import (
    OrchestratorEngineMixin,
    real_session_id,
)
from kestrel_sovereign.agent.context_manager import ContextManager
from kestrel_sovereign.agent.memory_manager import MemoryManager
from kestrel_sovereign.agent.streaming import StreamingMixin
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.memory_answerability import LLMAnswerabilityGate
from kestrel_sovereign.storage.memory_retriever import MemoryRetriever


# The key the Timeline's ``sessionKeyOf`` actually reads. Spelled literally so a
# rename of the constant cannot silently satisfy these tests while the consumer
# keeps looking for the old string.
KESTREL_SESSION_ID = "kestrel.session_id"
LEGACY_SESSION_ID = "agent.session_id"
# OpenInference's own session key, likewise spelled literally.
OI_SESSION_ID = "session.id"

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


# ---------------------------------------------------------------------------
# #2940 — llm.generate could not name a session at all
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_span_exporter():
    """In-memory exporter behind the ``KestrelTracer`` the LLM spans use.

    ``llm.*`` spans go through ``telemetry.llm_span`` → ``get_kestrel_tracer()``,
    a different lane from the ``optional_span`` tracer the turn spans above use,
    so it needs its own installation (the #2674 precedent in
    ``test_llm_content_redaction_audit.py``).
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry.set_kestrel_tracer(
        KestrelTracer(tracer=provider.get_tracer("test-2940-llm"))
    )
    try:
        yield exporter
    finally:
        telemetry.reset_kestrel_tracer()


#: "the caller passed nothing", distinct from an explicit ``None``.
_NOT_PASSED = object()


async def _run_generate(session_id=_NOT_PASSED):
    """Drive the real ``LLMService.generate``, stubbed below the span."""
    service = LLMService()
    try:
        service._get_response_frozen = AsyncMock(return_value="42")
        service.get_last_response_identity = lambda: {"model": "m"}
        kwargs = {} if session_id is _NOT_PASSED else {"session_id": session_id}
        return await service.generate(
            system_prompt="You are helpful.",
            user_prompt="what is 6*7?",
            **kwargs,
        )
    finally:
        await service.close()


async def _run_generate_with_messages(session_id):
    """Drive the real ``generate_with_messages``, stubbed below the span."""
    service = LLMService()
    try:
        service._generate_with_messages_inner = AsyncMock(return_value="42")
        service.get_last_response_identity = lambda: {"model": "m"}
        return await service.generate_with_messages(
            messages=[{"role": "user", "content": "what is 6*7?"}],
            session_id=session_id,
        )
    finally:
        await service.close()


class TestLLMGenerateSpansCarrySession:
    """``llm.generate`` was 0/74 stamped in the fleet: it had no way to say."""

    @pytest.mark.asyncio
    async def test_generate_stamps_both_session_keys(self, llm_span_exporter):
        assert await _run_generate(SESSION) == "42"

        spans = _roots(llm_span_exporter, "llm.generate")
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        # The Timeline's first-priority key...
        assert attrs.get(KESTREL_SESSION_ID) == SESSION
        # ...and OpenInference's, which Phoenix's session view reads.
        assert attrs.get(OI_SESSION_ID) == SESSION

    @pytest.mark.asyncio
    async def test_sessionless_generate_omits_the_attributes_entirely(
        self, llm_span_exporter
    ):
        """Absent, not empty — per #2916.

        Background callers (the sleep-cycle memory attestation, the salvage
        janitor) genuinely have no chat window and pass nothing. An empty
        attribute reads the same as an absent one to ``sessionKeyOf``, so
        exporting it would be pure noise.
        """
        assert await _run_generate() == "42"

        spans = _roots(llm_span_exporter, "llm.generate")
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert KESTREL_SESSION_ID not in attrs
        assert OI_SESSION_ID not in attrs

    @pytest.mark.asyncio
    async def test_sentinel_session_is_not_stamped(self, llm_span_exporter):
        """A sentinel is a source tag, not a session.

        Stamping ``"orchestrator"`` would invent a fleet-wide session that every
        sessionless call on every agent shares — worse than the staircase.
        """
        assert await _run_generate("orchestrator") == "42"

        attrs = dict(_roots(llm_span_exporter, "llm.generate")[0].attributes)
        assert KESTREL_SESSION_ID not in attrs
        assert OI_SESSION_ID not in attrs

    @pytest.mark.asyncio
    async def test_both_entry_points_agree_on_both_keys(self, llm_span_exporter):
        """The property that puts one turn's LLM calls in ONE band.

        ``generate_with_messages`` was already 44/44 stamped while ``generate``
        was 0/74; a session whose turn used both split across bands. They must
        now produce identical session attributes for the same session.
        """
        await _run_generate(SESSION)
        await _run_generate_with_messages(SESSION)

        generate = _roots(llm_span_exporter, "llm.generate")
        with_messages = _roots(llm_span_exporter, "llm.generate_with_messages")
        assert len(generate) == 1 and len(with_messages) == 1

        def _session_keys(span):
            attrs = dict(span.attributes)
            return {
                key: attrs.get(key)
                for key in (KESTREL_SESSION_ID, OI_SESSION_ID)
            }

        assert _session_keys(generate[0]) == _session_keys(with_messages[0])
        assert _session_keys(generate[0]) == {
            KESTREL_SESSION_ID: SESSION,
            OI_SESSION_ID: SESSION,
        }


# ---------------------------------------------------------------------------
# #2940 — signal.dispatch.cognition carried no session
# ---------------------------------------------------------------------------


class _DispatchAgent:
    """DispatcherAgent stand-in; the wake itself is out of scope here."""

    def __init__(self, did: str = "agent-test"):
        self._did = did
        self.agent_name = "Emma"
        self.background_tasks: list = []

    @property
    def did(self) -> str:
        return self._did

    async def process_input(self, prompt: str):
        return "ok"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


async def _dispatch_cognition(tmp_path, session_id):
    """Run a real COGNITION dispatch carrying ``session_id``.

    Mirrors the #2699 harness in ``test_telemetry_layer_span_attribution.py`` —
    real registry, real store, real dispatcher — so the span asserted on is the
    one production emits.
    """
    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _DispatchAgent()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )

    template = tmp_path / "tpl.md"
    template.write_text("wake: {source}")
    registry.register(
        SourceRegistration(
            name="cog_src",
            schema=dict,
            default_mode=SignalMode.COGNITION,
            allowed_modes=frozenset({SignalMode.COGNITION}),
            prompt_template=template,
            log_redaction=RedactionPolicy(summarize=lambda p: "<redacted>"),
        )
    )

    signal = Signal(
        source="cog_src",
        kind="tick",
        mode=SignalMode.COGNITION,
        payload={},
        target_agent="agent-test",
        session_id=session_id,
    )
    try:
        result = await dispatcher.dispatch_signal(signal)
        assert result.status == Status.OK
    finally:
        for task in agent.background_tasks:
            task.cancel()
        await backend.close()


class TestCognitionDispatchSpanCarriesSession:
    """A wake filed from a chat window belongs in that window's band."""

    @pytest.mark.asyncio
    async def test_dispatch_stamps_the_signals_session(
        self, span_exporter, tmp_path
    ):
        await _dispatch_cognition(tmp_path, SESSION)

        spans = _roots(span_exporter, "signal.dispatch.cognition")
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs.get(KESTREL_SESSION_ID) == SESSION
        assert attrs.get(OI_SESSION_ID) == SESSION

    @pytest.mark.asyncio
    async def test_system_initiated_dispatch_omits_the_attributes(
        self, span_exporter, tmp_path
    ):
        """``Signal.session_id`` is None for cron ticks and webhooks."""
        await _dispatch_cognition(tmp_path, None)

        spans = _roots(span_exporter, "signal.dispatch.cognition")
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert KESTREL_SESSION_ID not in attrs
        assert OI_SESSION_ID not in attrs


class TestSessionSpanAttributes:
    """The shared resolver both #2940 span types call."""

    def test_sentinels_blank_and_none_yield_no_attributes(self):
        for value in ("", "   ", "original", "orchestrator", None):
            assert telemetry.session_span_attributes(value) == {}

    def test_a_real_session_yields_both_keys(self):
        assert telemetry.session_span_attributes(SESSION) == {
            KESTREL_SESSION_ID: SESSION,
            OI_SESSION_ID: SESSION,
        }

    def test_surrounding_whitespace_is_stripped(self):
        assert telemetry.session_span_attributes(f"  {SESSION}  ") == {
            KESTREL_SESSION_ID: SESSION,
            OI_SESSION_ID: SESSION,
        }


# ---------------------------------------------------------------------------
# #2940 — the answerability judge, the busiest sessionless ``llm.generate``
# ---------------------------------------------------------------------------


class _EmbeddingStub:
    """An embedding service that demands second-stage evidence checking.

    ``_requires_answerability_gate`` returns ``bool(model)`` — on for every
    embedding model unless a provider opts out — so the judge below runs on
    roughly every non-trivial turn, which is why its span not naming a session
    put a sessionless root beside each of them.
    """

    def requires_answerability_gate(self):
        return True

    def current_profile_id(self):
        return "profile"

    def retrieval_similarity_floor(self):
        return 0.0


async def _retrieve_through_the_judge(session_id, *, verdict='{"answerable_ids":["c0"]}'):
    """Real retriever → real gate → real ``LLMService.generate``, stubbed below
    the span, so the assertion is on the span production actually exports."""
    store = AsyncMock()
    store.embedding_service = _EmbeddingStub()
    store.get_conversation_history.return_value = [
        {"id": 1, "role": "user", "content": "my favorite planet is Saturn", "metadata": {}}
    ]
    store.get_salient_memory_candidates.return_value = []
    store.get_lexical_memory_candidates.return_value = []

    service = LLMService()
    try:
        service._get_response_frozen = AsyncMock(return_value=verdict)
        service.get_last_response_identity = lambda: {"model": "m"}
        retriever = MemoryRetriever(
            store,
            answerability_gate=LLMAnswerabilityGate(service),
        )
        retriever._embed_query = AsyncMock(return_value=[1.0, 0.0])
        retriever._semantic_similarities_via_vector_backend = AsyncMock(
            return_value={"1": 0.9}
        )
        kwargs = {"session_id": session_id} if session_id is not None else {}
        results = await retriever.retrieve(
            "favorite planet", "test", min_score=0.0, read_only=True, **kwargs
        )
        assert [row["id"] for row in results] == [1]
        assert retriever.answerability_stats["calls"] == 1
    finally:
        await service.close()


class TestAnswerabilityJudgeSpanCarriesSession:
    """Memory retrieval's own LLM call belongs to the turn that asked."""

    @pytest.mark.asyncio
    async def test_retrieval_for_a_turn_stamps_that_session(
        self, llm_span_exporter
    ):
        await _retrieve_through_the_judge(SESSION)

        spans = _roots(llm_span_exporter, "llm.generate")
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs.get(KESTREL_SESSION_ID) == SESSION
        assert attrs.get(OI_SESSION_ID) == SESSION

    @pytest.mark.asyncio
    async def test_retrieval_outside_a_turn_stays_unstamped(
        self, llm_span_exporter
    ):
        """Offline recall has no chat window, so the attribute stays absent."""
        await _retrieve_through_the_judge(None)

        attrs = dict(_roots(llm_span_exporter, "llm.generate")[0].attributes)
        assert KESTREL_SESSION_ID not in attrs
        assert OI_SESSION_ID not in attrs


class TestRetrieveMemoriesForwardsTheSession:
    """The context-assembly seam between the turn and the judge."""

    @staticmethod
    def _manager_with_retriever():
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(return_value=[])
        return MemoryManager(MagicMock(), memory_retriever=retriever), retriever

    @pytest.mark.asyncio
    async def test_session_reaches_the_retriever(self):
        manager, retriever = self._manager_with_retriever()

        await manager.retrieve_memories(
            "favorite planet", 1000, MagicMock(), session_id=SESSION
        )

        assert retriever.retrieve.await_args.kwargs["session_id"] == SESSION

    @pytest.mark.asyncio
    async def test_no_session_forwards_no_keyword_at_all(self):
        """Matches ``min_score``: an unset value is not passed down.

        A retriever double predating the parameter keeps working on the
        sessionless path, and the retriever's own default stays authoritative.
        """
        manager, retriever = self._manager_with_retriever()

        await manager.retrieve_memories("favorite planet", 1000, MagicMock())

        assert "session_id" not in retriever.retrieve.await_args.kwargs


class TestRagRetrievalForwardsTheSession:
    """Semantic recall reaches the same judge down the RAG branch."""

    @staticmethod
    def _manager_with_builder():
        builder = MagicMock()
        builder.retrieve_context = AsyncMock(return_value="")
        builder.last_semantic_recall_metadata = {"status": "disabled"}
        manager = ContextManager(MagicMock(), context_builder=builder)
        manager.counter = MagicMock(count=len)
        return manager, builder

    async def _produce_rag(self, manager, **kwargs):
        return await manager._produce_rag(
            MagicMock(rag=37),
            "question",
            {"rag_min_score": 0.2},
            include_rag=True,
            trivial_turn=False,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_session_reaches_the_context_builder(self):
        manager, builder = self._manager_with_builder()

        await self._produce_rag(manager, session_id=SESSION)

        assert builder.retrieve_context.await_args.kwargs["session_id"] == SESSION

    @pytest.mark.asyncio
    async def test_no_session_leaves_the_builder_call_untouched(self):
        """The ContextBuilder is an injected seam (the agent supplies its own).

        A sessionless build must call it exactly as it did before #2940.
        """
        manager, builder = self._manager_with_builder()

        await self._produce_rag(manager)

        builder.retrieve_context.assert_awaited_once_with(
            "question", min_score=0.2, max_tokens=37
        )


# ---------------------------------------------------------------------------
# #2940 — a multi-step subagent split its turn across a band per step
# ---------------------------------------------------------------------------


class _SubagentFeature(Feature):
    """A real feature with one real tool, so the real tool loop runs."""

    tool_description = "probe feature"

    async def initialize(self):
        return None

    @tool(name="probe", description="Probe something.")
    async def probe(self) -> dict:
        return {"status": "ok"}


def _subagent_agent(session_id, llm_service):
    """Agent double exposing only what the subagent path reads off the agent."""
    agent = MagicMock()
    agent.hooks_manager = None
    agent.llm_service = llm_service
    agent._get_turn_bound_session_id = lambda: session_id
    return agent


async def _run_subagent(session_id):
    """Drive the real ``execute_as_subagent`` through initial → continuation →
    repair, stubbed below the spans.

    The initial call answers with a tool call, so the loop runs; the
    continuation answers with continuation-intent prose and NO tool call, which
    is what triggers the one repair turn. Three provider calls, three spans.
    """
    service = LLMService()
    try:
        service.get_last_response_identity = lambda: {"model": "m"}
        service._get_response_frozen = AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="probe", arguments={})],
            )
        )
        service._generate_with_messages_inner = AsyncMock(
            side_effect=[
                # Continuation: narrates more work, emits no tool call.
                LLMResponse(content="Let me check the repo next.", tool_calls=None),
                # Repair: settles the task.
                LLMResponse(content="Probe reported ok.", tool_calls=None),
            ]
        )
        feature = _SubagentFeature(_subagent_agent(session_id, service))
        envelope = await feature.execute_as_subagent(task="probe the thing")
        assert envelope["success"] is True, envelope
        assert envelope["result"] == "Probe reported ok."
        assert service._generate_with_messages_inner.await_count == 2
        return service
    finally:
        await service.close()


class TestSubagentContinuationSpansCarrySession:
    """Every step of one subagent run belongs to the turn that dispatched it.

    ``execute_as_subagent`` stamped only its FIRST call, so a subagent that
    used a tool — the normal case, that being the point of a subagent — emitted
    a sessionless ``llm.generate_with_messages`` root per continuation and per
    repair. One turn, a band per step: the same staircase #2940 exists to
    collapse, one level down.
    """

    @pytest.mark.asyncio
    async def test_every_step_of_the_run_carries_both_keys(self, llm_span_exporter):
        await _run_subagent(SESSION)

        spans = _roots(llm_span_exporter, "llm.generate") + _roots(
            llm_span_exporter, "llm.generate_with_messages"
        )
        # Initial + continuation + repair.
        assert len(spans) == 3
        for span in spans:
            attrs = dict(span.attributes)
            assert attrs.get(KESTREL_SESSION_ID) == SESSION, span.name
            assert attrs.get(OI_SESSION_ID) == SESSION, span.name

    @pytest.mark.asyncio
    async def test_a_sessionless_subagent_stamps_nothing(self, llm_span_exporter):
        """Unattended work (a cron tick, a detached task) has no chat window."""
        await _run_subagent(None)

        spans = _roots(llm_span_exporter, "llm.generate") + _roots(
            llm_span_exporter, "llm.generate_with_messages"
        )
        assert len(spans) == 3
        for span in spans:
            attrs = dict(span.attributes)
            assert KESTREL_SESSION_ID not in attrs, span.name
            assert OI_SESSION_ID not in attrs, span.name

    @pytest.mark.asyncio
    async def test_the_turn_session_never_becomes_the_provider_thread_key(
        self, llm_span_exporter
    ):
        """The reason the session rides ``invocation_context``, not the wire.

        ``generate_with_messages(session_id=...)`` is a provider CONTINUATION
        key: ``CodexAdapter._ensure_thread`` caches a thread per session id and
        fingerprints ``(model, instructions, tools)``. A subagent runs its own
        system prompt over its own tool palette, so handing it the user turn's
        session would evict that conversation's thread, cache the subagent's
        under the user's id, and get evicted back on the next user turn —
        losing server-side history each way (the #2841/#2845 class, and exactly
        what ``per-turn-reflection::<session>`` namespacing exists to prevent).

        Attribution therefore travels as invocation identity while the wire
        session stays unset. Mutating either half of that split fails here.
        """
        service = await _run_subagent(SESSION)

        for call in service._generate_with_messages_inner.await_args_list:
            assert call.kwargs.get("session_id") is None
            assert call.kwargs["invocation_context"].session_id == SESSION
