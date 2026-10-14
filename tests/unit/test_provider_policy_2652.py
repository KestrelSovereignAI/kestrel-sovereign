"""Agent-written model overrides cannot widen the approved provider chain (#2652).

The incident: with Anthropic down and OpenAI over quota, the coordinator's LLM
work ran on OpenRouter's Gemini because an override the agent had written for
itself pinned it there. These tests drive the real ``LLMService`` and the real
``set_model`` tool over fake routes, with no network.

The approved chain is ``[llm] route_priority``; ``[llm] denied_providers`` names
providers that are never used.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.agent.model_preference import (
    ModelPreferenceMixin,
    apply_persisted_model_preference,
)
from kestrel_sovereign.auth import (
    AuthMethod,
    CallerContext,
    CallerRole,
    caller_context_scope,
)
from kestrel_sovereign.features.model.feature import ModelAgent
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.error_handling import (
    LLMAllProvidersFailedError,
    LLMProviderError,
)
from kestrel_sovereign.llm.model_cache import get_shared_model_cache
from kestrel_sovereign.llm.output_ceiling import attach_stop_reason
from kestrel_sovereign.llm.provider_policy import (
    OverrideProvenance,
    OverrideSource,
    ProviderPolicyConfigError,
    ProviderPolicyRefusal,
    agent_provenance,
    command_authored_scope,
    override_writer,
    provenance_for_caller,
    tool_write_provenance,
)
from kestrel_sovereign.llm.service import LLMService, LLMServiceError

APPROVED_CHAIN = ["anthropic:api", "openai:api"]
OUTSIDE_MODEL = "google/gemini-3.1-pro-preview"

SOVEREIGN = CallerContext(
    role=CallerRole.SOVEREIGN, auth_method=AuthMethod.API_KEY, identity="api_key"
)
OPERATOR = CallerContext.authenticated("operator@example.test")


class _Route:
    """A fake route whose adapter records each call and fails or answers."""

    def __init__(self, name, *, model="m", healthy=True):
        self.name = name
        self.vendor, _, self.route = name.partition(":")
        self.model = model
        self.healthy = healthy
        self.calls = 0
        adapter = Mock()
        adapter.create_messages = Mock(
            return_value=[{"role": "user", "content": "hi"}]
        )
        adapter.get_response = self._get_response
        self.adapter = adapter

    async def _get_response(self, *args, **kwargs):
        self.calls += 1
        if not self.healthy:
            raise LLMProviderError(self.vendor, f"{self.name} is down")
        return attach_stop_reason(
            LLMResponse(content=f"from {self.name}", input_tokens=1, output_tokens=1),
            "stop",
        )

    def as_provider(self):
        return {
            "name": self.name,
            "vendor": self.vendor,
            "route": self.route,
            "client": AsyncMock(),
            "adapter": self.adapter,
            "model": self.model,
            "is_cloud": True,
            "is_local": False,
            "base_url": None,
            "capabilities": {"supports_decisions": True},
        }


def _service(routes, *, config):
    """A real LLMService over ``routes`` with ``config`` as its ``[llm]``."""
    registry = Mock()
    registry.initialize_providers = Mock(return_value=[])
    registry.get_providers_with_pattern = Mock(return_value=[])
    registry.get_local_providers = Mock(return_value=[])
    with patch(
        "kestrel_sovereign.llm.service.load_section", return_value=dict(config)
    ), patch(
        "kestrel_sovereign.llm.service.load_config",
        return_value={"defaults": {}, "mandates": {}},
    ), patch("kestrel_sovereign.llm.service.ProviderRegistry", return_value=registry):
        service = LLMService()
    service._usage_db = None
    service._db_initialized = False
    service._usage_database_url = None
    service._db_backend = "sqlite"
    service.providers = [route.as_provider() for route in routes]
    service._agent_display_name = "Emma"
    return service


@pytest.fixture(autouse=True)
def _cold_model_cache():
    cache = get_shared_model_cache()
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def incident():
    """The 2026-07-20 lineup: the approved chain down, two outsiders healthy."""
    routes = {
        "anthropic:api": _Route("anthropic:api", healthy=False),
        "openai:api": _Route("openai:api", healthy=False),
        "openrouter:api": _Route("openrouter:api", model=OUTSIDE_MODEL),
        "google:api": _Route("google:api", model="gemini-3.1-pro-preview"),
    }
    service = _service(
        list(routes.values()), config={"route_priority": APPROVED_CHAIN}
    )
    return SimpleNamespace(service=service, routes=routes)


async def _model_agent(service):
    feature = ModelAgent(SimpleNamespace(llm_service=service, features={}))
    await feature.initialize()
    return feature


def _outsiders_called(incident):
    return {
        name: route.calls
        for name, route in incident.routes.items()
        if name not in APPROVED_CHAIN and route.calls
    }


# --- 1. The hard gate on agent-written overrides ----------------------------


@pytest.mark.asyncio
async def test_agent_override_outside_the_chain_is_refused_and_not_stored(
    incident, caplog
):
    """The model's own ``set_model`` call is the agent's, even inside a
    Sovereign's turn: being asked is not the Sovereign choosing the provider."""
    service = incident.service
    persisted = []

    async def _persist(model, vendor, route, provenance=None):
        persisted.append((model, vendor, route))

    service.set_preference_persistence_callback(_persist)
    feature = await _model_agent(service)

    with caplog.at_level(
        logging.WARNING, logger="kestrel_sovereign.llm.service"
    ), caller_context_scope(SOVEREIGN):
        result = await feature.set_model("openrouter", OUTSIDE_MODEL)

    assert result.status is ToolResultStatus.ERROR
    assert result.data["error"] == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert result.data["provider"] == "openrouter"
    assert result.data["model"] == OUTSIDE_MODEL
    assert result.data["source"] == "agent"
    assert result.data["approved_vendors"] == ["anthropic", "openai"]
    assert "approved chain" in result.data["reason"]
    # Nothing stored, in memory or on disk.
    assert service.get_model_preference()["model"] is None
    assert service._mandate_provenance is None
    assert persisted == []
    # One WARNING naming the agent, the task, the provider and the model.
    refusals = [r for r in caplog.records if "Model override refused" in r.getMessage()]
    assert len(refusals) == 1
    message = refusals[0].getMessage()
    assert refusals[0].levelno == logging.WARNING
    assert "agent=Emma" in message
    assert "task=" in message
    assert "provider=openrouter" in message
    assert f"model={OUTSIDE_MODEL}" in message


@pytest.mark.asyncio
async def test_agent_override_inside_the_chain_is_accepted_as_today(incident):
    service = incident.service
    feature = await _model_agent(service)

    result = await feature.set_model("openai", "gpt-5")

    assert result.status is ToolResultStatus.OK
    assert service.get_model_preference() == {
        "vendor": "openai", "model": "gpt-5", "route": None,
    }
    assert service._mandate_provenance.source is OverrideSource.AGENT
    assert service._mandate_provenance.via == "set_model"


@pytest.mark.asyncio
async def test_refused_override_records_no_consent_to_it(incident):
    """The policy runs before consent, so a refused switch is never consented to."""
    consent = SimpleNamespace(request_consent=AsyncMock())
    feature = ModelAgent(
        SimpleNamespace(
            llm_service=incident.service, features={"ConsentFeature": consent}
        )
    )
    await feature.initialize()

    result = await feature.set_model("openrouter", OUTSIDE_MODEL)

    assert result.status is ToolResultStatus.ERROR
    consent.request_consent.assert_not_awaited()


# --- Sovereign and operator overrides keep today's behaviour ----------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller", "source"),
    [(SOVEREIGN, OverrideSource.SOVEREIGN), (OPERATOR, OverrideSource.OPERATOR)],
)
async def test_typed_model_set_outside_the_chain_behaves_as_today(
    incident, caller, source
):
    """A ``!model-set`` the principal typed is theirs, not the agent's."""
    service = incident.service
    feature = await _model_agent(service)

    with caller_context_scope(caller), command_authored_scope("set_model"):
        result = await feature.set_model("openrouter", OUTSIDE_MODEL)

    assert result.status is ToolResultStatus.OK
    assert service.get_model_preference() == {
        "vendor": "openrouter", "model": OUTSIDE_MODEL, "route": None,
    }
    assert service._mandate_provenance.source is source
    assert service._mandate_provenance.via == "command:set_model"


@pytest.mark.asyncio
async def test_api_model_set_by_the_sovereign_outside_the_chain_behaves_as_today(
    incident,
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kestrel_sovereign.endpoints import models as models_endpoint

    service = incident.service
    app = FastAPI()
    app.include_router(models_endpoint.router)
    agent = SimpleNamespace(llm_service=service)

    with patch.object(models_endpoint, "get_agent", return_value=agent), patch.object(
        models_endpoint, "get_caller", return_value=SOVEREIGN
    ):
        response = TestClient(app).post(
            "/api/model/set",
            json={"vendor": "openrouter", "model": OUTSIDE_MODEL},
        )

    assert response.status_code == 200, response.text
    assert service.get_model_preference()["vendor"] == "openrouter"
    assert service._mandate_provenance.source is OverrideSource.SOVEREIGN
    assert service._mandate_provenance.via == "/api/model/set"


def test_a_write_naming_no_writer_is_the_agents(incident, caplog):
    """Fail closed: a path that forgets to name its writer is held to the
    chain, not exempted from it."""
    service = incident.service

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        with pytest.raises(ProviderPolicyRefusal) as refused:
            service.set_model_preference(OUTSIDE_MODEL, "openrouter")

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert refused.value.source is OverrideSource.AGENT
    assert service.get_model_preference()["model"] is None
    assert service._mandate_provenance is None
    assert any("Model override refused" in r.getMessage() for r in caplog.records)

    service.set_model_preference("gpt-5", "openai")
    assert service._mandate_provenance.source is OverrideSource.AGENT


class _SetModelAgent(ModelPreferenceMixin):
    """The agent's ``set_model()`` method over a real ``ModelAgent``."""

    def __init__(self, model_agent):
        self.model_agent = model_agent


@pytest.mark.asyncio
async def test_the_agents_set_model_method_is_held_to_the_chain(incident, caplog):
    """``KestrelAgent.set_model()`` reaches the service through the deprecated
    ``ModelAgent.set_model_preference`` wrapper; nothing on that path attests a
    principal, so the write is the agent's even inside a Sovereign's turn."""
    service = incident.service
    persisted = []

    async def _persist(model, vendor, route, provenance=None):
        persisted.append((model, vendor, route))

    service.set_preference_persistence_callback(_persist)
    agent = _SetModelAgent(await _model_agent(service))

    with caplog.at_level(
        logging.WARNING, logger="kestrel_sovereign.llm.service"
    ), caller_context_scope(SOVEREIGN), pytest.warns(DeprecationWarning):
        with pytest.raises(ProviderPolicyRefusal) as refused:
            agent.set_model(f"openrouter/{OUTSIDE_MODEL}")

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert refused.value.provider == "openrouter"
    assert refused.value.source is OverrideSource.AGENT
    assert service.get_model_preference()["model"] is None
    assert service._mandate_provenance is None
    assert persisted == []
    refusal = service.describe_provider_policy()["override_history"][-1]
    assert (refusal["source"], refusal["via"], refusal["accepted"]) == (
        "agent", "ModelAgent.set_model_preference", False,
    )
    assert any("Model override refused" in r.getMessage() for r in caplog.records)

    with pytest.warns(DeprecationWarning):
        agent.set_model("openai/gpt-5")

    assert service.get_model_preference() == {
        "vendor": "openai", "model": "gpt-5", "route": None,
    }
    assert service._mandate_provenance.source is OverrideSource.AGENT
    assert service._mandate_provenance.via == "ModelAgent.set_model_preference"


# --- Who the writer is ------------------------------------------------------


@pytest.mark.asyncio
async def test_typed_command_marker_is_set_by_execute_command(monkeypatch):
    """Wiring: the real ``execute_command`` marks the typed tool for its call."""
    seen = []

    async def _record(self, **kwargs):
        seen.append(tool_write_provenance(kwargs["skill_id"]))
        return SimpleNamespace(
            status=SimpleNamespace(state=None, message=None), artifacts=[]
        )

    manager = TaskManager.__new__(TaskManager)
    manager._agents = {}
    monkeypatch.setattr(
        TaskManager,
        "get_agent_for_command",
        lambda self, user_input: ("model_agent", "set_model"),
        raising=False,
    )
    monkeypatch.setattr(TaskManager, "execute_skill", _record, raising=False)

    with caller_context_scope(SOVEREIGN):
        await manager.execute_command(f"!model-set openrouter {OUTSIDE_MODEL}")

    assert [p.source for p in seen] == [OverrideSource.SOVEREIGN]
    # Outside the command, the same call is the agent's.
    with caller_context_scope(SOVEREIGN):
        assert tool_write_provenance("set_model").source is OverrideSource.AGENT


def test_the_marker_attributes_only_the_tool_that_was_typed():
    with caller_context_scope(SOVEREIGN), command_authored_scope("workflow_run"):
        assert tool_write_provenance("set_model").source is OverrideSource.AGENT


@pytest.mark.parametrize(
    "caller",
    [
        None,
        CallerContext.a2a_transport(),
        CallerContext.anonymous(),
    ],
)
def test_a_command_with_no_principal_is_the_agents(caller):
    """No caller (an unattended turn, a spawned child's prompt), an A2A
    transport caller or an anonymous one is not a principal."""
    with caller_context_scope(caller), command_authored_scope("set_model"):
        provenance = tool_write_provenance("set_model")
    assert provenance.source is OverrideSource.AGENT
    assert provenance_for_caller(caller, via="x").source is OverrideSource.AGENT


# --- 2. Fallback cannot widen the chain (the incident reproduction) ---------


@pytest.mark.asyncio
async def test_incident_every_approved_provider_down_fails_and_calls_no_outsider(
    incident,
):
    """Anthropic down, OpenAI over quota, OpenRouter and Google healthy: the
    agent's attempt to pin OpenRouter is refused, and the task fails visibly
    rather than landing on a provider outside the chain."""
    service = incident.service
    feature = await _model_agent(service)

    refused = await feature.set_model("openrouter", OUTSIDE_MODEL)
    assert refused.status is ToolResultStatus.ERROR

    # Fails visibly, naming the approved route that failed last.
    with pytest.raises(LLMServiceError, match="openai:api is down"):
        await service.get_response(system_prompt="s", user_prompt="hi")
    with pytest.raises(LLMServiceError, match="openai:api is down"):
        await service.generate_with_messages(
            messages=[{"role": "user", "content": "hi"}]
        )

    assert incident.routes["anthropic:api"].calls >= 1
    assert incident.routes["openai:api"].calls >= 1
    assert _outsiders_called(incident) == {}


# --- 3. The deny-list --------------------------------------------------------


def _denied_lineup(*, route_priority=None):
    routes = {
        "anthropic:api": _Route("anthropic:api", healthy=False),
        "openrouter:api": _Route("openrouter:api", model=OUTSIDE_MODEL),
    }
    config = {"denied_providers": ["openrouter"]}
    if route_priority is not None:
        config["route_priority"] = route_priority
    return routes, _service(list(routes.values()), config=config)


@pytest.mark.asyncio
async def test_a_denied_provider_is_never_called_even_when_it_is_the_only_healthy_one():
    # No chain declared: without the deny-list, the default order would land
    # on the healthy OpenRouter route.
    routes, service = _denied_lineup()

    with pytest.raises(LLMAllProvidersFailedError, match="anthropic:api is down"):
        await service.get_response(system_prompt="s", user_prompt="hi")
    with pytest.raises(LLMServiceError, match="anthropic:api is down"):
        await service.generate_with_messages(
            messages=[{"role": "user", "content": "hi"}]
        )

    assert routes["anthropic:api"].calls >= 1
    assert routes["openrouter:api"].calls == 0


@pytest.mark.asyncio
async def test_a_denied_provider_is_never_called_through_a_per_call_override():
    routes, service = _denied_lineup()

    with pytest.raises(ProviderPolicyRefusal) as refused:
        await service.get_response(
            system_prompt="s",
            user_prompt="hi",
            model_override=f"openrouter/{OUTSIDE_MODEL}",
        )
    assert refused.value.code == ProviderPolicyRefusal.PROVIDER_DENIED
    with pytest.raises(ProviderPolicyRefusal):
        await service.generate_with_messages(
            messages=[{"role": "user", "content": "hi"}],
            model_override=f"openrouter/{OUTSIDE_MODEL}",
        )
    with pytest.raises(ProviderPolicyRefusal):
        await service.get_response_with_model(
            OUTSIDE_MODEL, system_prompt="s", user_prompt="hi"
        )
    assert routes["openrouter:api"].calls == 0


@pytest.mark.parametrize(
    "writer",
    [
        OverrideProvenance(source=OverrideSource.SOVEREIGN, role="sovereign"),
        OverrideProvenance(source=OverrideSource.OPERATOR, role="authenticated"),
        agent_provenance(via="set_model"),
    ],
)
def test_a_denied_provider_is_refused_whoever_writes_the_override(writer):
    _, service = _denied_lineup()

    with override_writer(writer), pytest.raises(ProviderPolicyRefusal) as refused:
        service.set_model_preference(OUTSIDE_MODEL, "openrouter")

    assert refused.value.code == ProviderPolicyRefusal.PROVIDER_DENIED
    assert refused.value.source is writer.source
    assert service.get_model_preference()["model"] is None


@pytest.mark.asyncio
async def test_a_denied_provider_is_never_a_mandate_fallback():
    """An operator-declared mandate fallback cannot reach a denied route."""
    routes, service = _denied_lineup()
    service.set_model_preference("m", "anthropic")
    service._disabled_routes = {"anthropic:api": "401 after a key was revoked"}
    service._mandate_fallbacks = [{"vendor": "openrouter", "model": OUTSIDE_MODEL}]

    with pytest.raises(LLMServiceError, match="No usable LLM routes"):
        await service.get_response(system_prompt="s", user_prompt="hi")

    assert routes["openrouter:api"].calls == 0


def test_a_denied_route_is_dropped_from_decisions_and_the_available_chain():
    _, service = _denied_lineup()

    assert [p["name"] for p in service._available_providers()] == ["anthropic:api"]
    assert [p["name"] for p in service._decision_routes()] == ["anthropic:api"]


def test_a_denied_route_entry_denies_only_that_route():
    routes = [_Route("google:api"), _Route("google:vertex")]
    service = _service(routes, config={"denied_providers": ["google:vertex"]})

    assert [p["name"] for p in service._available_providers()] == ["google:api"]
    service.set_model_preference("gemini", "google")  # google:api still serves it
    with pytest.raises(ProviderPolicyRefusal):
        service.set_model_preference("gemini", "google", "vertex")


@pytest.mark.asyncio
async def test_denying_the_whole_approved_chain_admits_no_agent_override(caplog):
    """An empty approved set is a chain that admits nothing, not a missing
    chain that admits everything."""
    routes = {
        "anthropic:api": _Route("anthropic:api"),
        "openrouter:api": _Route("openrouter:api", model=OUTSIDE_MODEL),
    }
    service = _service(
        list(routes.values()),
        config={
            "route_priority": ["anthropic:api"],
            "denied_providers": ["anthropic"],
        },
    )
    feature = await _model_agent(service)

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        result = await feature.set_model("openrouter", OUTSIDE_MODEL)

    assert result.status is ToolResultStatus.ERROR
    assert result.data["error"] == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert result.data["approved_vendors"] == []
    assert "approved chain (empty)" in result.data["reason"]
    assert service.get_model_preference()["model"] is None
    assert service._mandate_provenance is None
    assert any("Model override refused" in r.getMessage() for r in caplog.records)
    # The same write through the agent's own method and with no writer named.
    with pytest.raises(ProviderPolicyRefusal), pytest.warns(DeprecationWarning):
        _SetModelAgent(feature).set_model(f"openrouter/{OUTSIDE_MODEL}")
    with pytest.raises(ProviderPolicyRefusal):
        service.set_model_preference(OUTSIDE_MODEL, "openrouter")
    assert service.get_model_preference()["model"] is None
    assert service.describe_provider_policy()["approved_vendors"] == []

    # The Sovereign is not narrowed by the chain; only the deny-list binds it.
    with caller_context_scope(SOVEREIGN), command_authored_scope("set_model"):
        result = await feature.set_model("openrouter", OUTSIDE_MODEL)
    assert result.status is ToolResultStatus.OK
    assert service._mandate_provenance.source is OverrideSource.SOVEREIGN


def test_no_declared_chain_and_no_initialized_route_admits_no_agent_override():
    service = _service([], config={})

    with pytest.raises(ProviderPolicyRefusal) as refused:
        service.set_model_preference("gpt-5", "openai")

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert service.get_model_preference()["model"] is None


@pytest.mark.parametrize("value", ["openrouter", [""], [3], {"openrouter": True}])
def test_a_malformed_deny_list_fails_at_startup(value):
    with pytest.raises(ProviderPolicyConfigError):
        _service([_Route("anthropic:api")], config={"denied_providers": value})


def test_the_deny_list_is_empty_by_default():
    routes = [_Route("anthropic:api"), _Route("openrouter:api")]
    service = _service(routes, config={})

    assert service.describe_provider_policy()["denied_providers"] == []
    assert len(service._available_providers()) == 2


# --- Stored overrides keep their writer across a restart ---------------------


class _MetadataDB:
    """agent_metadata as a dict, through the two statements the agent uses."""

    def __init__(self):
        self.rows = {}

    async def execute(self, sql, params):
        assert sql.lstrip().startswith("INSERT OR REPLACE")
        for i in range(0, len(params), 4):
            agent_id, key, value, _ = params[i:i + 4]
            self.rows[(agent_id, key)] = value

    async def fetchall(self, sql, params):
        value = self.rows.get(tuple(params))
        return [(value,)] if value is not None else []


class _Agent(ModelPreferenceMixin):
    agent_id = "did:test:emma"

    def __init__(self, db):
        self._raw_storage = SimpleNamespace(db=db)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("writer", "replayed"),
    [
        (OverrideProvenance(source=OverrideSource.SOVEREIGN, role="sovereign"), True),
        (agent_provenance(via="set_model"), False),
        (None, False),
    ],
)
async def test_boot_replays_a_stored_override_for_the_writer_that_wrote_it(
    incident, writer, replayed
):
    """A Sovereign's override outside the chain survives a restart; an agent's,
    or one with no recorded writer (a row stored before #2652), does not."""
    db = _MetadataDB()
    await _Agent(db)._write_model_preference(
        OUTSIDE_MODEL, "openrouter", None, provenance=writer
    )
    assert json.loads(db.rows[(_Agent.agent_id, "model_preference")]) == {
        "vendor": "openrouter", "model": OUTSIDE_MODEL, "route": None,
    }

    error = await apply_persisted_model_preference(
        incident.service, db, _Agent.agent_id
    )

    if replayed:
        assert error is None
        assert incident.service.get_model_preference()["vendor"] == "openrouter"
        assert incident.service._mandate_provenance == writer
    else:
        assert ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN in error
        assert incident.service.get_model_preference()["model"] is None


@pytest.mark.asyncio
async def test_a_writer_recorded_for_another_mandate_is_not_trusted(incident):
    db = _MetadataDB()
    agent = _Agent(db)
    await agent._write_model_preference(
        "gpt-5", "openai", None,
        provenance=OverrideProvenance(source=OverrideSource.SOVEREIGN, role="sovereign"),
    )
    # A writer that does not record provenance replaces only the mandate row.
    db.rows[(_Agent.agent_id, "model_preference")] = json.dumps(
        {"vendor": "openrouter", "model": OUTSIDE_MODEL, "route": None}
    )

    error = await apply_persisted_model_preference(
        incident.service, db, _Agent.agent_id
    )

    assert ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN in error


def test_a_privacy_transition_rewrites_the_mandate_in_force_under_its_own_writer(
    incident,
):
    """A transition re-persists the mandate in force with the writer it already
    has, so a Sovereign's override survives the next boot; a mandate the mode
    forces or restores is Kestrel's own."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    service = incident.service
    sovereign = OverrideProvenance(source=OverrideSource.SOVEREIGN, role="sovereign")
    with override_writer(sovereign):
        service.set_model_preference(OUTSIDE_MODEL, "openrouter")
    agent = SimpleNamespace(llm_service=service)

    in_force = {"vendor": "openrouter", "model": OUTSIDE_MODEL, "route": None}
    forced = {"vendor": "ollama", "model": "llama3.2:3b", "route": "local"}

    assert KestrelAgent._privacy_mandate_provenance(agent, in_force) == sovereign
    assert (
        KestrelAgent._privacy_mandate_provenance(agent, forced).source
        is OverrideSource.SYSTEM
    )


# --- 4. The read-only audit surface -----------------------------------------


@pytest.mark.asyncio
async def test_the_audit_surface_reports_the_chain_and_every_overrides_provenance(
    incident,
):
    service = incident.service
    service.config["denied_providers"] = ["google"]
    feature = await _model_agent(service)

    await feature.set_model("openrouter", OUTSIDE_MODEL)  # agent: refused
    with caller_context_scope(SOVEREIGN), command_authored_scope("set_model"):
        await feature.set_model("openrouter", OUTSIDE_MODEL)  # sovereign: accepted

    result = await feature.get_current_model()
    policy = result.data["provider_policy"]

    assert policy["route_priority"] == APPROVED_CHAIN
    assert policy["chain_declared"] is True
    assert policy["approved_vendors"] == ["anthropic", "openai"]
    assert policy["denied_providers"] == ["google"]
    states = {route["name"]: (route["approved"], route["state"]) for route in policy["chain"]}
    assert states == {
        "anthropic:api": (True, "available"),
        "openai:api": (True, "available"),
        "openrouter:api": (False, "available"),
        "google:api": (False, "denied"),
    }
    override = policy["override"]
    assert (override["vendor"], override["model"]) == ("openrouter", OUTSIDE_MODEL)
    assert override["agent"] == "Emma"
    assert override["provenance"]["source"] == "sovereign"
    assert override["provenance"]["role"] == "sovereign"
    assert override["provenance"]["at"]

    refused, accepted = policy["override_history"][-2:]
    assert (refused["source"], refused["role"], refused["accepted"]) == (
        "agent", "none", False,
    )
    assert refused["refusal"]["error"] == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert (refused["vendor"], refused["model"], refused["agent"]) == (
        "openrouter", OUTSIDE_MODEL, "Emma",
    )
    assert refused["decided_at"]
    assert (accepted["source"], accepted["accepted"]) == ("sovereign", True)

    text = result.confirmation
    assert "Approved chain ([llm] route_priority)" in text
    assert "Denied providers: google" in text
    assert f"Override in force: openrouter/{OUTSIDE_MODEL} by sovereign" in text
    assert "REFUSED (provider_outside_approved_chain)" in text
