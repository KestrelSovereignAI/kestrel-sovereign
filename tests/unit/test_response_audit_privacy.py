"""#3491: the response audit follows the privacy rule of the turn it audits.

``get_audit_response`` sends the text it audits to an LLM route. In a
local-only privacy mode (EPHEMERAL, ISOLATED, ANONYMOUS, DEIDENTIFIED) the
turn's own generation is confined to local routes, so the audit must be too:
only ``is_local`` routes are candidates, a mandate narrows within them and
cannot add a cloud route back, and with no local route the audit does not run.

Everything below ``get_audit_response`` is real except the routes' network
boundary.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookInput, PermissionDecision
from kestrel_sdk.llm import ProviderCapabilities, StructuredOutputMode
from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.service import NO_LOCAL_AUDIT_ROUTE, LLMService
from kestrel_sovereign.privacy import PrivacyMode, privacy_mode_to_config

AUDITED_TEXT = "A private reply that names the patient's diagnosis and must stay local."
LOCAL_ONLY_MODES = [
    PrivacyMode.EPHEMERAL,
    PrivacyMode.ISOLATED,
    PrivacyMode.ANONYMOUS,
    PrivacyMode.DEIDENTIFIED,
]


class _Route:
    """One route's network boundary: records every text it was sent."""

    def __init__(self, reasoning: str):
        self._reasoning = reasoning
        self.sent: list[str] = []

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_structured_output=True,
            structured_output_mode=StructuredOutputMode.JSON_SCHEMA,
        )

    async def get_response(self, *, client, model, messages, response_format=None, **_):
        self.sent.append(json.dumps(messages))
        return LLMResponse(content=json.dumps(
            {"risk_level": 1, "reasoning": f"{self._reasoning} via {model}"}
        ))


def _route(name: str, adapter: _Route, *, model: str, is_local: bool) -> dict[str, Any]:
    vendor, route = name.split(":", 1)
    return {
        "name": name,
        "vendor": vendor,
        "route": route,
        "model": model,
        "adapter": adapter,
        "client": object(),
        "is_local": is_local,
    }


def _service(providers, *, mandate=None) -> LLMService:
    """A real LLMService holding only what ``get_audit_response`` reads."""
    svc = LLMService.__new__(LLMService)
    svc.disabled = False
    svc._disabled_routes = set()
    svc.mandate_config = {}
    svc._mandate_preference = mandate or {"vendor": None, "model": None, "route": None}
    svc.providers = providers
    svc._force_local_only_provider = None
    # Routing is under test, not the process-wide discovery cache.
    svc._model_available_for_route = lambda provider, model_id: model_id == provider["model"]
    return svc


def _cloud_and_local():
    """A cloud route first in priority, a local route after it."""
    cloud = _Route("cloud")
    local = _Route("local")
    svc = _service([
        _route("anthropic:api", cloud, model="claude-x", is_local=False),
        _route("ollama:local", local, model="llama-x", is_local=True),
    ])
    return svc, cloud, local


def _bind_live_mode(svc: LLMService, mode: PrivacyMode) -> None:
    """Bind the live restriction the way ``KestrelAgent.initialize`` does."""
    config = privacy_mode_to_config(mode)
    svc.set_force_local_only_provider(lambda: not config.allows_cloud_llm())


def _agent(svc: LLMService, mode: PrivacyMode | None) -> MagicMock:
    agent = MagicMock()
    agent.llm_service = svc
    agent.features = {}
    agent.privacy_agent = (
        None if mode is None
        else SimpleNamespace(privacy_config=privacy_mode_to_config(mode))
    )
    return agent


async def _execute(hook: ResponseAuditHook):
    return await hook.execute(HookInput(
        session_id="audit-3491",
        hook_event_name=HookEvent.POST_RESPONSE.value,
        response_text=AUDITED_TEXT,
    ))


# ---------------------------------------------------------------------------
# Local-only: the audit uses local routes only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", LOCAL_ONLY_MODES, ids=lambda m: m.name)
async def test_live_local_only_mode_audits_on_the_local_route_only(mode):
    svc, cloud, local = _cloud_and_local()
    _bind_live_mode(svc, mode)

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict == {"risk_level": 1, "reasoning": "local via llama-x"}
    assert cloud.sent == []
    assert len(local.sent) == 1


@pytest.mark.asyncio
async def test_isolated_hook_passes_the_turn_privacy_state_explicitly():
    """No live restriction is bound on the service: the hook alone must keep
    the response off the cloud route."""
    svc, cloud, local = _cloud_and_local()
    hook = ResponseAuditHook(agent=_agent(svc, PrivacyMode.ISOLATED), mode="strict")

    output = await _execute(hook)

    assert output.permission_decision == PermissionDecision.ALLOW
    assert cloud.sent == []
    assert len(local.sent) == 1
    assert AUDITED_TEXT in local.sent[0]


@pytest.mark.asyncio
async def test_explicit_false_cannot_loosen_the_live_restriction():
    svc, cloud, local = _cloud_and_local()
    _bind_live_mode(svc, PrivacyMode.ISOLATED)

    verdict = await svc.get_audit_response(AUDITED_TEXT, force_local_only=False)

    assert verdict["reasoning"] == "local via llama-x"
    assert cloud.sent == []


@pytest.mark.asyncio
async def test_unreadable_live_restriction_fails_closed_to_local():
    svc, cloud, local = _cloud_and_local()

    def broken() -> bool:
        raise RuntimeError("privacy state unavailable")

    svc.set_force_local_only_provider(broken)

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict["reasoning"] == "local via llama-x"
    assert cloud.sent == []


# ---------------------------------------------------------------------------
# Local-only with no local route: the audit does not run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_local_route_returns_typed_unaudited_result():
    cloud = _Route("cloud")
    svc = _service([_route("anthropic:api", cloud, model="claude-x", is_local=False)])

    verdict = await svc.get_audit_response(AUDITED_TEXT, force_local_only=True)

    assert verdict["audited"] is False
    assert verdict["reason"] == NO_LOCAL_AUDIT_ROUTE
    assert cloud.sent == []


@pytest.mark.asyncio
async def test_strict_hook_blocks_the_turn_when_no_local_route_can_audit():
    cloud = _Route("cloud")
    svc = _service([_route("anthropic:api", cloud, model="claude-x", is_local=False)])
    hook = ResponseAuditHook(agent=_agent(svc, PrivacyMode.ISOLATED), mode="strict")

    output = await _execute(hook)

    assert output.permission_decision == PermissionDecision.DENY
    assert "local-only privacy mode" in output.permission_reason
    assert cloud.sent == []


@pytest.mark.asyncio
async def test_warn_hook_does_not_send_or_block_when_no_local_route_can_audit():
    cloud = _Route("cloud")
    svc = _service([_route("anthropic:api", cloud, model="claude-x", is_local=False)])
    hook = ResponseAuditHook(agent=_agent(svc, PrivacyMode.ISOLATED), mode="warn")

    output = await _execute(hook)

    assert output.permission_decision == PermissionDecision.ALLOW
    assert output.updated_input is None
    assert cloud.sent == []


@pytest.mark.asyncio
async def test_hook_that_cannot_read_the_turn_privacy_state_makes_no_audit_call():
    svc, cloud, local = _cloud_and_local()
    hook = ResponseAuditHook(agent=_agent(svc, None), mode="strict")

    output = await _execute(hook)

    assert output.permission_decision == PermissionDecision.DENY
    assert cloud.sent == []
    assert local.sent == []


# ---------------------------------------------------------------------------
# The mandate narrows within the local routes; it never adds a cloud one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cloud_mandate_cannot_add_a_cloud_route_back():
    cloud = _Route("cloud")
    local = _Route("local")
    svc = _service(
        [
            _route("anthropic:api", cloud, model="claude-x", is_local=False),
            _route("ollama:local", local, model="llama-x", is_local=True),
        ],
        mandate={"vendor": "anthropic", "route": "api", "model": "claude-x"},
    )

    verdict = await svc.get_audit_response(AUDITED_TEXT, force_local_only=True)

    # Generation under local-only ignores the non-local mandated model; the
    # audit judges on the local route's own configured model.
    assert verdict == {"risk_level": 1, "reasoning": "local via llama-x"}
    assert cloud.sent == []


@pytest.mark.asyncio
async def test_local_mandate_narrows_within_the_local_routes():
    cloud = _Route("cloud")
    first = _Route("ollama")
    mandated = _Route("llama_cpp")
    svc = _service(
        [
            _route("anthropic:api", cloud, model="claude-x", is_local=False),
            _route("ollama:local", first, model="llama-x", is_local=True),
            _route("llama_cpp:local", mandated, model="qwen-x", is_local=True),
        ],
        mandate={"vendor": "llama_cpp", "route": "local", "model": "qwen-x"},
    )

    verdict = await svc.get_audit_response(AUDITED_TEXT, force_local_only=True)

    assert verdict == {"risk_level": 1, "reasoning": "llama_cpp via qwen-x"}
    assert cloud.sent == [] and first.sent == []


# ---------------------------------------------------------------------------
# Non-private modes: route selection is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.PUBLIC], ids=lambda m: m.name)
async def test_cloud_mode_keeps_the_existing_route_order(mode):
    svc, cloud, local = _cloud_and_local()
    _bind_live_mode(svc, mode)

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict == {"risk_level": 1, "reasoning": "cloud via claude-x"}
    assert len(cloud.sent) == 1
    assert local.sent == []


@pytest.mark.asyncio
async def test_normal_hook_keeps_the_existing_route_order():
    svc, cloud, local = _cloud_and_local()
    hook = ResponseAuditHook(agent=_agent(svc, PrivacyMode.NORMAL), mode="strict")

    output = await _execute(hook)

    assert output.permission_decision == PermissionDecision.ALLOW
    assert len(cloud.sent) == 1
    assert local.sent == []


@pytest.mark.asyncio
async def test_cloud_mode_keeps_mandate_narrowing():
    cloud = _Route("cloud")
    local = _Route("local")
    svc = _service(
        [
            _route("ollama:local", local, model="llama-x", is_local=True),
            _route("anthropic:api", cloud, model="claude-x", is_local=False),
        ],
        mandate={"vendor": "anthropic", "route": "api", "model": "claude-x"},
    )
    _bind_live_mode(svc, PrivacyMode.NORMAL)

    verdict = await svc.get_audit_response(AUDITED_TEXT)

    assert verdict == {"risk_level": 1, "reasoning": "cloud via claude-x"}
    assert local.sent == []
