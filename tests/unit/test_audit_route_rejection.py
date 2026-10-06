"""#3492: the integrity audit on Claude Opus 5.5, and on a rejected route.

Two defects locked every agent pinned to ``anthropic:plan/claude-opus-5-5``
out of cognition after a constitution reanchor, because the genesis self-audit
and the response audit share ``LLMService.get_audit_response``:

1. the Anthropic adapter asked for structured output with a forced
   ``tool_choice``, which the model refuses with a 400; and
2. that 400 (``anthropic.BadRequestError``) was outside the audit loop's
   caught set, so it escaped to the outer handler and failed the whole audit
   at risk 3 without trying another route.

Everything below ``get_audit_response`` is real here except the network: the
real ``ClaudeMaxAdapter`` runs against a client that behaves like Opus 5.5.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookInput, PermissionDecision
from kestrel_sdk.llm import ProviderCapabilities, StructuredOutputMode
from kestrel_sovereign.constitution.genesis_audit import (
    GENESIS_AUDIT_PASSED,
    GenesisAuditPendingError,
    evaluate_genesis_constitution,
)
from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter
from kestrel_sovereign.llm.output_ceiling import attach_stop_reason
from kestrel_sovereign.llm.service import LLMService
from tests.utils.anthropic_client import (
    FORCED_TOOL_CHOICE_REFUSAL,
    anthropic_bad_request,
    forced_tool_refusing_client,
)

AUDITED_TEXT = "This is a sufficiently long agent response to be audited end to end."
CONSTITUTION = b"# Constitution\n\nArticle I. Be honest.\n"
OPUS_55_MANDATE = {"vendor": "anthropic", "model": "claude-opus-5-5", "route": "plan"}


class _StubRoute:
    """An adapter standing in for one route's network boundary."""

    def __init__(
        self,
        *,
        payload: dict | None = None,
        raise_exc: Exception | None = None,
        refuse: bool = False,
        structured: bool = True,
    ):
        self._payload = payload or {"risk_level": 1, "reasoning": "ok"}
        self._raise_exc = raise_exc
        self._refuse = refuse
        self._structured = structured
        self.calls = 0

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_structured_output=self._structured,
            structured_output_mode=(
                StructuredOutputMode.JSON_SCHEMA
                if self._structured
                else StructuredOutputMode.NONE
            ),
        )

    async def get_response(self, *, client, model, messages, response_format=None, **_):
        self.calls += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        if self._refuse:
            # What Opus 5.5 returns for a harmful sample: a refusal stop with
            # no complete JSON.
            return attach_stop_reason(LLMResponse(content='{"risk_level'), "refusal")
        return LLMResponse(content=json.dumps(self._payload))


def _route(name: str, adapter: Any, *, model: str = "claude-opus-5-5", client: Any = None) -> dict:
    vendor, route = name.split(":", 1)
    return {
        "name": name,
        "vendor": vendor,
        "route": route,
        "model": model,
        "adapter": adapter,
        "client": client if client is not None else object(),
    }


def _service(providers, *, mandate=None, config=None) -> LLMService:
    """A real LLMService holding only what ``get_audit_response`` reads."""
    svc = LLMService.__new__(LLMService)
    svc.disabled = False
    svc._disabled_routes = set()
    svc.mandate_config = {}
    svc._mandate_preference = mandate or {"vendor": None, "model": None, "route": None}
    svc.providers = providers
    if config is not None:
        svc.config = config
    # Routing is under test, not the process-wide discovery cache.
    svc._model_available_for_route = lambda provider, model_id: True
    return svc


def _opus_55_service(payload: dict) -> tuple[LLMService, Any]:
    client = forced_tool_refusing_client(payload)
    svc = _service(
        [_route("anthropic:plan", ClaudeMaxAdapter(), client=client)],
        mandate=OPUS_55_MANDATE,
    )
    return svc, client


# ---------------------------------------------------------------------------
# Structured output on an Opus-5.5-style model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_audit_succeeds_on_a_model_that_refuses_forced_tool_choice():
    svc, client = _opus_55_service({"risk_level": 1, "reasoning": "Normal response"})

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict == {"risk_level": 1, "reasoning": "Normal response"}
    [request] = client.messages.requests
    assert request["model"] == "claude-opus-5-5"
    assert "tool_choice" not in request
    assert request["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_strict_response_audit_allows_a_benign_response_on_opus_55():
    """Strict mode denied every response while the audit could not run."""
    svc, _client = _opus_55_service({"risk_level": 1, "reasoning": "Normal response"})
    agent = MagicMock()
    agent.llm_service = svc
    agent.features = {}
    hook = ResponseAuditHook(agent=agent, mode="strict", risk_threshold=3)

    output = await hook.execute(HookInput(
        session_id="audit-3492",
        hook_event_name=HookEvent.POST_RESPONSE.value,
        response_text=AUDITED_TEXT,
    ))

    assert output.permission_decision == PermissionDecision.ALLOW


@pytest.mark.asyncio
async def test_genesis_audit_passes_on_a_model_that_refuses_forced_tool_choice():
    """The prod lockout: genesis stayed pending on every opus-5-5 agent."""
    svc, client = _opus_55_service({"risk_level": 1, "reasoning": "Sound constitution"})

    record = await evaluate_genesis_constitution(
        CONSTITUTION,
        constitution_hash=hashlib.sha256(CONSTITUTION).hexdigest(),
        auditor=svc.get_audit_response,
        provenance="test-3492",
    )

    assert record["status"] == GENESIS_AUDIT_PASSED
    assert record["risk_level"] == 1
    assert record["audited"] is True
    assert len(client.messages.requests) == 1


# ---------------------------------------------------------------------------
# A route that rejects the request is a per-route failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_route_falls_through_to_the_next_eligible_route():
    rejected = _StubRoute(raise_exc=anthropic_bad_request(FORCED_TOOL_CHOICE_REFUSAL))
    healthy = _StubRoute(payload={"risk_level": 1, "reasoning": "from openai:plan"})
    svc = _service([
        _route("anthropic:plan", rejected),
        _route("openai:plan", healthy, model="gpt-x"),
    ])

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict == {"risk_level": 1, "reasoning": "from openai:plan"}
    assert rejected.calls == 1
    assert healthy.calls == 1


@pytest.mark.asyncio
async def test_rejected_only_route_fails_closed_naming_the_route():
    rejected = _StubRoute(raise_exc=anthropic_bad_request(FORCED_TOOL_CHOICE_REFUSAL))
    svc = _service([_route("anthropic:plan", rejected)])

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict["risk_level"] == 3
    assert verdict["audited"] is False
    # Folded by the per-route handler, which names the route that failed —
    # not the outer "Audit failed: <exc>" catch-all.
    assert verdict["reasoning"].startswith("Audit provider failed: anthropic:plan:")
    assert "tool_choice" in verdict["reasoning"]


@pytest.mark.asyncio
async def test_genesis_audit_stays_pending_when_every_route_rejects():
    """A rejection is never converted into a pass."""
    rejected = _StubRoute(raise_exc=anthropic_bad_request(FORCED_TOOL_CHOICE_REFUSAL))
    svc = _service([_route("anthropic:plan", rejected)])

    with pytest.raises(GenesisAuditPendingError) as pending:
        await evaluate_genesis_constitution(
            CONSTITUTION,
            constitution_hash=hashlib.sha256(CONSTITUTION).hexdigest(),
            auditor=svc.get_audit_response,
            provenance="test-3492",
        )
    assert pending.value.code == "auditor_unavailable"


@pytest.mark.asyncio
async def test_rejected_plan_route_does_not_silently_fall_to_a_paid_route():
    """Falling through must still honor ``allow_paid_fallback = false``."""
    rejected = _StubRoute(raise_exc=anthropic_bad_request(FORCED_TOOL_CHOICE_REFUSAL))
    paid = _StubRoute(payload={"risk_level": 1, "reasoning": "metered"})
    svc = _service(
        [_route("anthropic:plan", rejected), _route("anthropic:api", paid)],
        config={"allow_paid_fallback": False},
    )

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict["risk_level"] == 3
    assert verdict["audited"] is False
    assert "plan->paid" in verdict["reasoning"]
    assert paid.calls == 0


@pytest.mark.asyncio
async def test_paid_route_is_used_when_the_plan_route_was_never_attempted():
    """A plan route the audit cannot use (the codex ``openai:plan`` route has
    no structured output) never failed, so the paid route after it is not a
    silent downgrade and must still audit."""
    plan = _StubRoute(structured=False)
    paid = _StubRoute(payload={"risk_level": 1, "reasoning": "metered"})
    svc = _service(
        [_route("openai:plan", plan, model="gpt-x"), _route("openai:api", paid, model="gpt-x")],
        config={"allow_paid_fallback": False},
    )

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict == {"risk_level": 1, "reasoning": "metered"}
    assert plan.calls == 0
    assert paid.calls == 1


@pytest.mark.asyncio
async def test_rejected_plan_route_falls_to_a_paid_route_when_allowed():
    rejected = _StubRoute(raise_exc=anthropic_bad_request(FORCED_TOOL_CHOICE_REFUSAL))
    paid = _StubRoute(payload={"risk_level": 1, "reasoning": "metered"})
    svc = _service(
        [_route("anthropic:plan", rejected), _route("anthropic:api", paid)],
        config={"allow_paid_fallback": True},
    )

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict == {"risk_level": 1, "reasoning": "metered"}
    assert paid.calls == 1


# ---------------------------------------------------------------------------
# An audit model that refuses is not a verdict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refusing_route_is_named_and_the_next_route_is_tried():
    refusing = _StubRoute(refuse=True)
    healthy = _StubRoute(payload={"risk_level": 3, "reasoning": "harmful"})
    svc = _service([
        _route("anthropic:plan", refusing),
        _route("openai:plan", healthy, model="gpt-x"),
    ])

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict == {"risk_level": 3, "reasoning": "harmful"}
    assert refusing.calls == 1


@pytest.mark.asyncio
async def test_refusing_only_route_fails_closed_as_a_refusal():
    svc = _service([_route("anthropic:plan", _StubRoute(refuse=True))])

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict["risk_level"] == 3
    assert verdict["audited"] is False
    assert "refused" in verdict["reasoning"]
    assert "malformed" not in verdict["reasoning"]
