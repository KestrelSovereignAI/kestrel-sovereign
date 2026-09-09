"""Telemetry-layer spans carry ``kestrel.agent_name`` + OpenInference kind (#2699).

The engine/dispatch ``optional_span`` / ``start_span`` sites —
``agent.process_input`` / ``agent.process_input_streaming`` (kestrel_agent /
streaming), ``agent.feature_dispatch`` (orchestrator_engine), and
``signal.dispatch.cognition`` (signals/dispatcher) — previously carried neither
``kestrel.agent_name`` nor ``openinference.span.kind``, so in Phoenix they
grouped under agent "(none)" / kind "unknown", polluting the per-agent lanes.

These tests assert the attributes now land on the REAL spans those helpers emit,
captured with an in-memory OTel exporter (no network, no live LLM). The module
name carries "telemetry" and "span" so the acceptance gate
``pytest -k "telemetry or observability or span"`` selects it.
"""
from __future__ import annotations

import asyncio

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from kestrel_sdk.signals import (
    SignalMode,
    SourceRegistration,
    Status,
    RedactionPolicy,
)
from kestrel_sovereign import telemetry
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.storage.db import SQLiteBackend

# Literal OpenInference / Kestrel attribute keys, as an exporter observer sees
# them — deliberately NOT the module constants, so a regression that flips a key
# is caught rather than masked.
OI_KIND = "openinference.span.kind"
KESTREL_AGENT_NAME = "kestrel.agent_name"


@pytest.fixture
def otel_exporter():
    """Install an in-memory-exporter-backed tracer as ``telemetry._tracer``.

    ``optional_span`` / ``start_span`` read the process tracer through
    ``get_tracer()``; pointing that at an in-memory exporter lets the tests
    inspect the real spans those helpers emit. Restored afterward so tracing
    state never leaks between tests.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    original = telemetry._tracer
    telemetry._tracer = provider.get_tracer("test-2699")
    try:
        yield exporter
    finally:
        telemetry._tracer = original


def _one(exporter, name):
    spans = [s for s in exporter.get_finished_spans() if s.name == name]
    assert len(spans) == 1, f"expected exactly one {name!r} span, got {len(spans)}"
    return spans[0]


class TestSpanAttributeConstants:
    """The shared constants resolve to the OpenInference / Kestrel conventions."""

    def test_span_kind_key_and_values(self):
        assert telemetry.OI_SPAN_KIND == OI_KIND
        assert telemetry.OI_SPAN_KIND_CHAIN == "CHAIN"
        assert telemetry.OI_SPAN_KIND_TOOL == "TOOL"

    def test_agent_name_key(self):
        assert telemetry.KESTREL_AGENT_NAME == KESTREL_AGENT_NAME


class TestHelpersStampKindAndAgentName:
    """``optional_span`` / ``start_span`` actually record both attributes.

    This is the exact mechanism every one of the five instrumented sites relies
    on — passing the two keys in the attribute dict — so it is verified once here
    against a real span rather than at every heavy call site.
    """

    def test_optional_span_records_kind_and_agent_name(self, otel_exporter):
        with telemetry.optional_span(
            "agent.process_input",
            {
                telemetry.OI_SPAN_KIND: telemetry.OI_SPAN_KIND_CHAIN,
                telemetry.KESTREL_AGENT_NAME: "Nellie",
            },
        ):
            pass
        attrs = dict(_one(otel_exporter, "agent.process_input").attributes)
        assert attrs[OI_KIND] == "CHAIN"
        assert attrs[KESTREL_AGENT_NAME] == "Nellie"

    def test_start_span_records_kind_and_agent_name(self, otel_exporter):
        span = telemetry.start_span(
            "agent.process_input_streaming",
            {
                telemetry.OI_SPAN_KIND: telemetry.OI_SPAN_KIND_CHAIN,
                telemetry.KESTREL_AGENT_NAME: "Emma",
            },
        )
        telemetry.end_span(span)
        attrs = dict(_one(otel_exporter, "agent.process_input_streaming").attributes)
        assert attrs[OI_KIND] == "CHAIN"
        assert attrs[KESTREL_AGENT_NAME] == "Emma"

    def test_none_agent_name_is_dropped_not_stamped(self, otel_exporter):
        """A duck-typed agent with no name leaves the attr absent, never None.

        ``signal.dispatch.cognition`` passes ``getattr(agent, "agent_name",
        None)``; the helper must drop the None rather than write a literal.
        """
        with telemetry.optional_span(
            "agent.feature_dispatch",
            {
                telemetry.OI_SPAN_KIND: telemetry.OI_SPAN_KIND_CHAIN,
                telemetry.KESTREL_AGENT_NAME: None,
            },
        ):
            pass
        attrs = dict(_one(otel_exporter, "agent.feature_dispatch").attributes)
        assert attrs[OI_KIND] == "CHAIN"
        assert KESTREL_AGENT_NAME not in attrs


# ---------------------------------------------------------------------------
# End-to-end: the real dispatcher path stamps signal.dispatch.cognition
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Stand-in for KestrelAgent satisfying DispatcherAgent + carrying a name."""

    def __init__(self, did: str = "agent-test", agent_name: str = "Nellie"):
        self._did = did
        self.agent_name = agent_name
        self.background_tasks: list[asyncio.Task] = []

    @property
    def did(self) -> str:
        return self._did

    async def process_input(self, prompt: str):
        from kestrel_sovereign.telemetry import turn_span_scope

        with turn_span_scope("turn_signal_cognition"):
            await asyncio.sleep(0)
            return "ok"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.mark.asyncio
async def test_signal_dispatch_cognition_span_carries_kind_and_agent_name(
    otel_exporter, tmp_path
):
    """A real COGNITION dispatch emits ``signal.dispatch.cognition`` as CHAIN
    stamped with the dispatcher's agent name — not "(none)"."""
    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _FakeAgent(agent_name="Nellie")
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

    from kestrel_sdk.signals import Signal

    signal = Signal(
        source="cog_src",
        kind="tick",
        mode=SignalMode.COGNITION,
        payload={},
        target_agent="agent-test",
    )
    try:
        result = await dispatcher.dispatch_signal(signal)
        assert result.status == Status.OK
        assert result.turn_id == "turn_signal_cognition"

        span = _one(otel_exporter, "signal.dispatch.cognition")
        attrs = dict(span.attributes)
        assert attrs[OI_KIND] == "CHAIN"
        assert attrs[KESTREL_AGENT_NAME] == "Nellie"
        assert attrs["kestrel.turn_id"] == "turn_signal_cognition"
        # Sanity: the existing constitution-injection attrs still ride along.
        assert attrs["kestrel.signal.source"] == "cog_src"
    finally:
        pending = [t for t in agent.background_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await backend.close()
