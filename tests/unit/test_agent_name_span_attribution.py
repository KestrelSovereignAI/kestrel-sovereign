"""Agent-name span attribution across every construction path (issue #2602).

#2461 stamped ``agent.agent_name`` at ``AgentManager._register_agent``, but a
second agent object on the host lacked the attribute — some construction path
(scheduler-context / spawn-inception / single-agent host / CLI-REPL) bypasses
the registrar, so its LLM-call spans fell back to the observability emitter's
``"unknown"``.

The fix stamps the display name at the most upstream single point —
``KestrelAgent.__init__`` derives a construction-time floor from config /
data-dir — and mirrors it onto the owning ``LLMService`` so LLM spans carry
``kestrel.agent_name`` from the very first call. ``initialize()`` upgrades the
floor to the authoritative graph-node name, and ``_register_agent`` remains the
authoritative override for fleet members.

These tests assert the name is stamped and propagated on the paths that bypass
the registrar, and — via the in-memory OTel exporter — that a real LLM span
carries the agent name instead of falling back to ``"unknown"``.
"""
import json

import pytest
from unittest.mock import AsyncMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from kestrel_sdk.tracing import KestrelTracer

from kestrel_sovereign import telemetry
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.multi_agent.agent_manager import AgentManager


KESTREL_AGENT_NAME = "kestrel.agent_name"
OI_INPUT = "input.value"


def _make_agent(tmp_path, dir_name, *, did="did:pkh:eip155:1:0xabc"):
    """Construct a real KestrelAgent in ``tmp_path/dir_name`` (no async init).

    No identity artifacts are written, so ``__init__`` skips identity loading —
    exercising only the synchronous construction path the fix stamps.
    """
    agent_dir = tmp_path / dir_name
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "kestrel_prime.db"
    return KestrelAgent(did=did, storage_path=str(db_path))


class TestConstructionDisplayNameFloor:
    """``__init__`` stamps a non-"unknown" floor on every construction path."""

    def test_derives_name_from_data_directory(self):
        """Data-directory name is the floor when there is no config name."""
        name = KestrelAgent._derive_construction_display_name(
            "did:pkh:eip155:1:0xabc", "/srv/agent_data/Nellie/kestrel_prime.db"
        )
        assert name == "Nellie"

    def test_config_agent_name_wins_over_directory(self, tmp_path):
        """``[agent] name`` in the agent's kestrel.toml wins over the dir name."""
        agent_dir = tmp_path / "datadir"
        agent_dir.mkdir()
        (agent_dir / "kestrel.toml").write_text('[agent]\nname = "Nellie Prime"\n')
        name = KestrelAgent._derive_construction_display_name(
            "did:pkh:eip155:1:0xabc", str(agent_dir / "kestrel_prime.db")
        )
        assert name == "Nellie Prime"

    def test_falls_back_to_did_without_storage_path(self):
        """A pathless (in-memory/test) agent still gets a non-empty name."""
        name = KestrelAgent._derive_construction_display_name(
            "did:pkh:eip155:1:0xabc", None
        )
        assert name == "did:pkh:eip155:1:0xabc"

    def test_never_returns_unknown_or_empty(self):
        """The floor is never empty — the emitter's "unknown" can't be reached."""
        for storage_path in (None, "", "/x/Emma/kestrel_prime.db"):
            name = KestrelAgent._derive_construction_display_name(
                "did:pkh:eip155:1:0xabc", storage_path
            )
            assert name
            assert name != "unknown"


class TestAgentStampsAndPropagates:
    """A constructed agent carries the name AND mirrors it onto its LLMService."""

    @pytest.mark.asyncio
    async def test_init_stamps_agent_name_and_llm_display_name(self, tmp_path):
        """``__init__`` sets ``agent_name`` and the LLMService display name."""
        agent = _make_agent(tmp_path, "Nellie")
        try:
            assert agent.agent_name == "Nellie"
            # The value the LLM span reads (llm.service._llm_request_span).
            assert agent.llm_service._agent_display_name == "Nellie"
        finally:
            await agent.llm_service.close()

    @pytest.mark.asyncio
    async def test_set_display_name_ignores_falsy(self, tmp_path):
        """A later, less-specific resolution never blanks out a good name."""
        agent = _make_agent(tmp_path, "Emma")
        try:
            agent._set_display_name(None)
            agent._set_display_name("")
            assert agent.agent_name == "Emma"
            assert agent.llm_service._agent_display_name == "Emma"
        finally:
            await agent.llm_service.close()

    @pytest.mark.asyncio
    async def test_register_agent_overrides_and_propagates(self, tmp_path):
        """``_register_agent`` overrides with the routing key + propagates it."""
        agent = _make_agent(tmp_path, "datadir")
        try:
            manager = AgentManager(base_data_dir=tmp_path)
            manager._register_agent("Nellie", agent)
            assert agent.agent_name == "Nellie"
            assert agent.llm_service._agent_display_name == "Nellie"
        finally:
            await agent.llm_service.close()


@pytest.fixture
def otel_llm_exporter():
    """Install an in-memory-exporter-backed ``KestrelTracer`` as the LLM tracer."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = KestrelTracer(tracer=provider.get_tracer("test-agent-name"))
    telemetry.set_kestrel_tracer(tracer)
    try:
        yield exporter
    finally:
        telemetry.reset_kestrel_tracer()


class TestLlmSpanCarriesRealAgentName:
    """The emitted LLM span carries the real name — never falls back to unknown."""

    @pytest.mark.asyncio
    async def test_bypass_path_llm_span_is_attributed_not_unknown(
        self, otel_llm_exporter, tmp_path
    ):
        """An agent that never reaches ``_register_agent`` still names its spans.

        This mirrors the single-agent host / CLI / scheduler-context path: the
        agent is constructed and used directly. Without the fix the span carried
        no ``kestrel.agent_name`` (the emitter then renders "unknown"); with it
        the construction-time floor lands on the span.
        """
        agent = _make_agent(tmp_path, "Nellie")
        service = agent.llm_service
        try:
            service._get_response_frozen = AsyncMock(
                return_value=LLMResponse(content="42")
            )
            service.get_last_response_identity = lambda: {"model": "ollama:local/x"}
            result = await service.get_response(
                system_prompt="s", user_prompt="What is 6*7?"
            )
            assert result.content == "42"
        finally:
            await service.close()

        spans = otel_llm_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        # The attribute is present and is the real name — not absent (which the
        # observability layer renders as "unknown") and not the literal.
        assert attrs.get(KESTREL_AGENT_NAME) == "Nellie"
        assert attrs.get(KESTREL_AGENT_NAME) != "unknown"
        # Sanity: it really is the LLM request span for this call.
        assert {"role": "user", "content": "What is 6*7?"} in json.loads(
            attrs[OI_INPUT]
        )
