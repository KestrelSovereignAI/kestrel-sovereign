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
    LLMError,
    LLMProviderError,
    LLMProviderUnavailableError,
)
from kestrel_sovereign.llm.model_cache import get_shared_model_cache
from kestrel_sovereign.llm.output_ceiling import attach_stop_reason
from kestrel_sovereign.llm.provider_policy import (
    FALLBACK_WRITER_KEY,
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
    manager._providers = {}
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
    """Not even an operator can add a denied fallback, and an operator's
    fallback already stored is refused by name when the mandate falls back."""
    routes, service = _denied_lineup()
    service.set_model_preference("m", "anthropic")
    operator = OverrideProvenance(source=OverrideSource.OPERATOR, role="authenticated")

    with override_writer(operator), pytest.raises(ProviderPolicyRefusal) as refused:
        service.add_fallback_model(OUTSIDE_MODEL, "openrouter")
    assert refused.value.code == ProviderPolicyRefusal.PROVIDER_DENIED
    assert service.get_current_mandate()["fallbacks"] == []

    service._disabled_routes = {"anthropic:api": "401 after a key was revoked"}
    service._mandate_fallbacks = [{
        "vendor": "openrouter",
        "model": OUTSIDE_MODEL,
        FALLBACK_WRITER_KEY: operator,
    }]
    with pytest.raises(ProviderPolicyRefusal) as refused:
        await service.get_response(system_prompt="s", user_prompt="hi")

    assert refused.value.code == ProviderPolicyRefusal.PROVIDER_DENIED
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

    backend_type = "sqlite"

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
    has, so a Sovereign's override survives the next boot. Any other mandate a
    transition writes names its writer in its plan; one that names none is the
    agent's, never Kestrel's unrestricted own."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    service = incident.service
    sovereign = OverrideProvenance(source=OverrideSource.SOVEREIGN, role="sovereign")
    with override_writer(sovereign):
        service.set_model_preference(OUTSIDE_MODEL, "openrouter")
    agent = SimpleNamespace(llm_service=service)

    in_force = {"vendor": "openrouter", "model": OUTSIDE_MODEL, "route": None}
    other = {"vendor": "ollama", "model": "llama3.2:3b", "route": "local"}

    assert KestrelAgent._privacy_mandate_provenance(agent, in_force) == sovereign
    assert (
        KestrelAgent._privacy_mandate_provenance(agent, other).source
        is OverrideSource.AGENT
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
    assert "REFUSED (provider_outside_approved_chain) (mandate)" in text

    # A per-call override is reported beside the mandates, marked as one.
    with pytest.raises(ProviderPolicyRefusal):
        service.resolve_provider_routing(model_override="google/gemini-3.1-pro-preview")
    result = await feature.get_current_model()
    per_call = result.data["provider_policy"]["override_history"][-1]
    assert (per_call["kind"], per_call["vendor"], per_call["accepted"]) == (
        "call", "google", False,
    )
    assert per_call["refusal"]["error"] == ProviderPolicyRefusal.PROVIDER_DENIED
    assert "REFUSED (provider_denied) (per-call): google/gemini" in result.confirmation


# --- Per-call overrides go through the same gate -----------------------------

OVERRIDE = f"openrouter/{OUTSIDE_MODEL}"
MESSAGES = [{"role": "user", "content": "hi"}]


def _call_decisions(service):
    return [
        entry
        for entry in service.describe_provider_policy()["override_history"]
        if entry["kind"] == "call"
    ]


def _refusal_warnings(caplog):
    return [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "Model override refused" in r.getMessage()
    ]


def test_routing_refuses_an_unattributed_per_call_override_and_records_it(
    incident, caplog
):
    """The round-2 reproduction: with an anthropic+openai chain, routing
    returned ``["openrouter:api"]`` for an unattributed OpenRouter override and
    the audit stayed empty."""
    service = incident.service

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        with pytest.raises(ProviderPolicyRefusal) as refused:
            service.resolve_provider_routing(model_override=OVERRIDE)

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert refused.value.provider == "openrouter"
    assert refused.value.source is OverrideSource.AGENT
    [decision] = _call_decisions(service)
    assert (decision["vendor"], decision["model"], decision["source"]) == (
        "openrouter", OUTSIDE_MODEL, "agent",
    )
    assert decision["accepted"] is False
    assert decision["refusal"]["error"] == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    [warning] = _refusal_warnings(caplog)
    assert "kind=call" in warning.getMessage()
    assert "provider=openrouter" in warning.getMessage()
    # Nothing the override names was routed to, and no mandate was touched.
    assert service.get_model_preference()["model"] is None


async def _drain(stream):
    return [chunk async for chunk in stream]


_ENTRY_POINTS = {
    "get_response": lambda s: s.get_response(
        system_prompt="s", user_prompt="hi", model_override=OVERRIDE
    ),
    "generate": lambda s: s.generate(
        system_prompt="s", user_prompt="hi", model_override=OVERRIDE
    ),
    "generate_with_messages": lambda s: s.generate_with_messages(
        messages=list(MESSAGES), model_override=OVERRIDE
    ),
    "get_streaming_response": lambda s: _drain(s.get_streaming_response(
        system_prompt="s", user_prompt="hi", model_override=OVERRIDE
    )),
    "generate_stream": lambda s: _drain(s.generate_stream(
        system_prompt="s", user_prompt="hi", model_override=OVERRIDE
    )),
    "stream_with_messages": lambda s: _drain(s.stream_with_messages(
        messages=list(MESSAGES), model_override=OVERRIDE
    )),
    "stream_with_tool_detection": lambda s: _drain(s.stream_with_tool_detection(
        messages=list(MESSAGES), model_override=OVERRIDE
    )),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", sorted(_ENTRY_POINTS))
async def test_an_unattributed_per_call_override_outside_the_chain_is_never_served(
    incident, caplog, entry
):
    """Every sync and streaming entry point refuses it before any route runs."""
    service = incident.service

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        with pytest.raises(ProviderPolicyRefusal) as refused:
            await _ENTRY_POINTS[entry](service)

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert _outsiders_called(incident) == {}
    for name in ("openrouter:api", "google:api"):
        assert incident.routes[name].adapter.method_calls == [], name
    assert [d["accepted"] for d in _call_decisions(service)] == [False]
    assert len(_refusal_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_a_per_call_override_inside_the_chain_is_accepted_as_today(incident):
    service = incident.service
    incident.routes["openai:api"].healthy = True

    response = await service.generate_with_messages(
        messages=list(MESSAGES), model_override="openai/gpt-5"
    )

    assert response == "from openai:api"
    [decision] = _call_decisions(service)
    assert (decision["vendor"], decision["source"], decision["accepted"]) == (
        "openai", "agent", True,
    )


# Routing ignores surrounding whitespace (``_resolve_model_selector`` strips the
# override), so admission must too: unstripped, `` openrouter `` named no
# provider, and the call failed with no structured refusal and no audit entry.
_PADDED_ENTRY_POINTS = {
    "resolve_provider_routing": lambda s, o: s.resolve_provider_routing(
        model_override=o
    ),
    "generate_with_messages": lambda s, o: s.generate_with_messages(
        messages=list(MESSAGES), model_override=o
    ),
    "stream_with_messages": lambda s, o: _drain(s.stream_with_messages(
        messages=list(MESSAGES), model_override=o
    )),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", sorted(_PADDED_ENTRY_POINTS))
@pytest.mark.parametrize("override", [" openrouter ", "openrouter:api\n"])
async def test_a_padded_per_call_override_is_admitted_as_routing_reads_it(
    incident, caplog, entry, override
):
    service = incident.service

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        with pytest.raises(ProviderPolicyRefusal) as refused:
            result = _PADDED_ENTRY_POINTS[entry](service, override)
            if hasattr(result, "__await__"):
                await result

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert refused.value.provider == override.strip()
    assert _outsiders_called(incident) == {}
    [decision] = _call_decisions(service)
    assert (decision["vendor"], decision["source"], decision["accepted"]) == (
        "openrouter", "agent", False,
    )
    assert len(_refusal_warnings(caplog)) == 1


def test_a_bare_model_id_selects_no_provider_and_is_not_gated(incident):
    """``llama3.2:3b`` or ``gpt-5`` names a model, not a provider: routing tries
    it on the chain's routes, so it is not an override of the chain."""
    service = incident.service

    for bare in ("gpt-5", "llama3.2:3b"):
        providers, target = service.resolve_provider_routing(model_override=bare)
        assert target == bare
        assert {p["name"] for p in providers} >= set(APPROVED_CHAIN)
    assert _call_decisions(service) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller", "source"),
    [(SOVEREIGN, OverrideSource.SOVEREIGN), (OPERATOR, OverrideSource.OPERATOR)],
)
async def test_a_principals_per_call_override_behaves_as_today(incident, caller, source):
    """A model a principal passes with a request is theirs for that turn: the
    turn's calls with it are served, and are continuations, recorded once.
    Any other override the turn makes is the agent's, and the chain binds it."""
    from kestrel_sovereign.llm.provider_policy import caller_call_override_scope
    from kestrel_sovereign.telemetry import bind_current_turn_id

    service = incident.service
    with caller_call_override_scope(
        OVERRIDE, caller, via="model_override"
    ), bind_current_turn_id("turn-1"):
        for _ in range(3):
            response = await service.generate_with_messages(
                messages=list(MESSAGES), model_override=OVERRIDE
            )
            assert response == "from openrouter:api"
        # The orchestrator's watchdog probes routing with the same override.
        assert [p["name"] for p in service.resolve_provider_routing(
            model_override=OVERRIDE
        ).providers] == ["openrouter:api"]

        with pytest.raises(ProviderPolicyRefusal) as refused:
            await service.generate_with_messages(
                messages=list(MESSAGES),
                model_override="google/gemini-3.1-pro-preview",
            )
    assert refused.value.source is OverrideSource.AGENT

    assert incident.routes["openrouter:api"].calls == 3
    assert incident.routes["google:api"].calls == 0
    accepted, refusal = _call_decisions(service)
    assert (accepted["source"], accepted["accepted"], accepted["task"]) == (
        source.value, True, "turn-1",
    )
    assert accepted["via"] == "model_override"
    assert (refusal["vendor"], refusal["source"], refusal["accepted"]) == (
        "google", "agent", False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller", [None, CallerContext.a2a_transport(), CallerContext.anonymous()]
)
async def test_a_per_call_override_with_no_principal_is_the_agents(incident, caller):
    from kestrel_sovereign.llm.provider_policy import caller_call_override_scope

    service = incident.service
    with caller_call_override_scope(OVERRIDE, caller, via="model_override"):
        with pytest.raises(ProviderPolicyRefusal) as refused:
            await service.generate_with_messages(
                messages=list(MESSAGES), model_override=OVERRIDE
            )
    assert refused.value.source is OverrideSource.AGENT
    assert _outsiders_called(incident) == {}


def test_an_enclosing_turns_attested_override_does_not_leak_into_a_nested_turn():
    """A turn with no override of its own clears the one an enclosing turn's
    principal chose, so a nested turn cannot borrow that authority."""
    from kestrel_sovereign.llm.provider_policy import (
        attested_call_override_writer,
        caller_call_override_scope,
    )

    with caller_call_override_scope(OVERRIDE, SOVEREIGN, via="model_override"):
        assert attested_call_override_writer(OVERRIDE).source is OverrideSource.SOVEREIGN
        assert attested_call_override_writer("openai/gpt-5") is None
        with caller_call_override_scope(None, SOVEREIGN, via="model_override"):
            assert attested_call_override_writer(OVERRIDE) is None
        with caller_call_override_scope(OVERRIDE, None, via="model_override"):
            assert attested_call_override_writer(OVERRIDE) is None
        assert attested_call_override_writer(OVERRIDE).source is OverrideSource.SOVEREIGN


@pytest.mark.asyncio
async def test_a_turns_attested_override_reaches_a_task_created_before_the_turn():
    """A tool the turn runs on a reader task created before it (the codex
    app-server topology) passes the turn's override on, so the attestation is
    a declared turn-scoped value carried by ``capture_turn_scope``."""
    import asyncio

    from kestrel_sovereign.llm.provider_policy import (
        attested_call_override_writer,
        caller_call_override_scope,
    )
    from kestrel_sovereign.turn_scope import capture_turn_scope

    started = asyncio.Event()
    work: asyncio.Queue = asyncio.Queue()

    async def reader():
        started.set()
        snapshot = await work.get()
        before = attested_call_override_writer(OVERRIDE)
        with snapshot.bind():
            return before, attested_call_override_writer(OVERRIDE)

    task = asyncio.create_task(reader())
    await started.wait()
    with caller_call_override_scope(OVERRIDE, SOVEREIGN, via="model_override"):
        await work.put(capture_turn_scope(None))
        before, carried = await task

    assert before is None
    assert carried.source is OverrideSource.SOVEREIGN


def test_a_selector_the_operators_model_mandate_names_is_the_operators(incident):
    """``model_mandate.toml`` is the operator's: an explicit ``cheap_model`` or
    a ``[mandates]`` selector outside the chain routes as it does today."""
    service = incident.service
    service.mandate_config = {
        "defaults": {"cheap_model": "openrouter/deepseek/deepseek-chat"},
        "mandates": {"truth": "google"},
    }

    cheap = service.resolve_provider_routing(
        model_override=service.get_cheap_model_selector()
    )
    keyword = service.resolve_provider_routing(model_override="google")

    assert [p["name"] for p in cheap.providers] == ["openrouter:api"]
    assert keyword.target_model == "google"
    configured = _call_decisions(service)
    assert [(d["vendor"], d["source"], d["role"], d["via"], d["accepted"]) for d in configured] == [
        ("openrouter", "operator", "config", "model_mandate.toml", True),
        ("google", "operator", "config", "model_mandate.toml", True),
    ]
    # The same vendor under a model the operator did not name is the agent's.
    with pytest.raises(ProviderPolicyRefusal):
        service.resolve_provider_routing(model_override=OVERRIDE)


def test_a_banned_mandate_selector_is_never_the_operators(incident):
    service = incident.service
    service.mandate_config = {
        "defaults": {"banned": ["google"]},
        "mandates": {"truth": "google"},
    }

    with pytest.raises(ProviderPolicyRefusal):
        service.resolve_provider_routing(model_override="google")


def test_a_cheap_model_hint_never_picks_a_route_outside_the_chain(incident):
    service = incident.service
    service.mandate_config = {"defaults": {"cheap_model_hints": ["flash", "mini"]}}
    service.provider_registry.get_providers_with_pattern = Mock(return_value=[
        SimpleNamespace(
            name="openrouter:api", vendor="openrouter", route="api",
            model="google/gemini-2.5-flash",
        ),
        SimpleNamespace(
            name="openai:api", vendor="openai", route="api", model="gpt-5-mini",
        ),
    ])

    assert service.get_cheap_model_selector() == "openai/gpt-5-mini"

    service.config["denied_providers"] = ["openai"]
    assert service.get_cheap_model_selector() is None


@pytest.mark.asyncio
async def test_get_response_with_model_admits_the_route_that_serves_the_model(incident):
    service = incident.service

    with pytest.raises(ProviderPolicyRefusal) as refused:
        await service.get_response_with_model(
            OUTSIDE_MODEL, system_prompt="s", user_prompt="hi"
        )

    assert refused.value.provider == "openrouter:api"
    assert incident.routes["openrouter:api"].calls == 0
    [decision] = _call_decisions(service)
    assert (decision["via"], decision["accepted"]) == ("get_response_with_model", False)


# --- A turn attests the model its caller passed ------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller", "source"),
    [(SOVEREIGN, OverrideSource.SOVEREIGN), (None, None)],
)
async def test_process_input_attests_the_callers_model_for_the_turns_calls(
    tmp_path, monkeypatch, caller, source
):
    """Wiring: the real non-streaming turn names its caller as the writer of
    the model it was given, around the turn's own model calls."""
    from kestrel_sovereign.llm.provider_policy import attested_call_override_writer
    from tests.shared.booted_agent import booted_agent

    seen = []

    async def generate(**kwargs):
        writer = attested_call_override_writer(kwargs.get("model_override"))
        seen.append((kwargs.get("model_override"), writer.source if writer else None))
        return attach_stop_reason(LLMResponse(content="ack"), "stop")

    async with booted_agent(
        tmp_path, name="Emma", provenance="test:provider_policy_2652", features=set()
    ) as agent:
        monkeypatch.setattr(agent.llm_service, "generate_with_messages", generate)
        await agent.process_input("hello", model_override=OVERRIDE, caller=caller)

    assert seen and all(entry == (OVERRIDE, source) for entry in seen), seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller", "source"),
    [(SOVEREIGN, OverrideSource.SOVEREIGN), (None, None)],
)
async def test_process_input_streaming_attests_the_callers_model_for_the_turns_calls(
    tmp_path, monkeypatch, caller, source
):
    """Wiring: the real streaming turn does the same."""
    from kestrel_sovereign.llm.provider_policy import attested_call_override_writer
    from tests.shared.booted_agent import booted_agent

    seen = []

    async def stream(**kwargs):
        writer = attested_call_override_writer(kwargs.get("model_override"))
        seen.append((kwargs.get("model_override"), writer.source if writer else None))
        yield "ack"

    async with booted_agent(
        tmp_path, name="Emma", provenance="test:provider_policy_2652", features=set()
    ) as agent:
        monkeypatch.setattr(agent.llm_service, "stream_with_tool_detection", stream)
        async for _chunk in agent.process_input_streaming(
            "hello", model_override=OVERRIDE, caller=caller
        ):
            pass

    assert seen and all(entry == (OVERRIDE, source) for entry in seen), seen


# --- Leaving a local-only mode restores a mandate under its own writer --------


def _saved_outside(provenance):
    from kestrel_sovereign.agent.privacy_state import SavedModelPreference

    return SavedModelPreference(
        vendor="openrouter", model=OUTSIDE_MODEL, route=None, provenance=provenance
    )


@pytest.mark.parametrize(
    ("writer", "via"),
    [
        (
            OverrideProvenance(
                source=OverrideSource.AGENT,
                role="none",
                via="set_model",
                task="turn-0720",
                at="2026-07-20T17:25:29+00:00",
            ),
            "set_model",
        ),
        # Saved before its writer was recorded: the agent's.
        (None, "privacy restore"),
    ],
)
def test_an_agent_mandate_a_local_only_mode_displaced_is_not_restored_outside_the_chain(
    incident, caplog, writer, via
):
    """Round-2 P2: restoring it must not promote it to Kestrel's own,
    unrestricted authority, and its writer must survive in the audit."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    service = incident.service
    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        plan = KestrelAgent._plan_privacy_restored_mandate(
            service, _saved_outside(writer)
        )

    assert plan.install is False
    assert "could not be restored" in plan.warning
    assert service.get_model_preference()["model"] is None
    refusal = service.describe_provider_policy()["override_history"][-1]
    assert (refusal["vendor"], refusal["model"], refusal["accepted"]) == (
        "openrouter", OUTSIDE_MODEL, False,
    )
    assert (refusal["source"], refusal["via"]) == ("agent", via)
    if writer is not None:
        assert (refusal["task"], refusal["at"]) == (writer.task, writer.at)
    assert len(_refusal_warnings(caplog)) == 1


def _privacy_agent(service, written=None):
    """A ``KestrelAgent`` holding only ``service``, recording mandate writes."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    class _PrivacyAgent(KestrelAgent):
        def __init__(self):
            self.llm_service = service

        def _privacy_record_db(self):
            return object()

        async def _write_model_preference(self, model, vendor, route, *, provenance=None):
            written.append(((vendor, model, route), provenance))

    return _PrivacyAgent()


@pytest.mark.asyncio
async def test_a_sovereign_mandate_a_local_only_mode_displaced_is_restored_under_the_sovereign(
    incident,
):
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    service = incident.service
    sovereign = OverrideProvenance(
        source=OverrideSource.SOVEREIGN,
        role="sovereign",
        identity="api_key",
        via="/api/model/set",
        at="2026-07-20T17:00:00+00:00",
    )
    written = []
    agent = _privacy_agent(service, written)
    plan = KestrelAgent._plan_privacy_restored_mandate(service, _saved_outside(sovereign))
    assert plan.install is True
    assert plan.provenance == sovereign

    assert await agent._write_privacy_mandate(plan.preference, plan.provenance)
    agent._install_privacy_mandate(plan)

    assert service.get_model_preference()["vendor"] == "openrouter"
    assert service.get_model_preference_provenance() == sovereign
    assert written == [(("openrouter", OUTSIDE_MODEL, None), sovereign)]


def test_entering_a_local_only_mode_saves_the_mandate_with_its_writer(incident):
    from kestrel_sovereign.privacy import PrivacyMode, privacy_mode_to_config

    service = incident.service
    service.providers.append({
        **_Route("ollama:local", model="llama3.2:3b").as_provider(),
        "is_cloud": False,
        "is_local": True,
    })
    sovereign = OverrideProvenance(source=OverrideSource.SOVEREIGN, role="sovereign")
    with override_writer(sovereign):
        service.set_model_preference(OUTSIDE_MODEL, "openrouter")

    plan = _privacy_agent(service)._plan_privacy_model_state(
        privacy_mode_to_config(PrivacyMode.ISOLATED),
        privacy_mode_to_config(PrivacyMode.NORMAL),
        None,
    )

    assert plan.saved.provenance == sovereign
    # The local route the mode forces is Kestrel's own.
    assert plan.install is True
    assert plan.provenance.source is OverrideSource.SYSTEM
    assert plan.preference["vendor"] == "ollama"


@pytest.mark.asyncio
async def test_the_saved_mandates_writer_survives_a_restart():
    from kestrel_sovereign.agent.privacy_state import (
        PRIVACY_SAVED_PROVENANCE_KEY,
        PrivacyState,
        load_privacy_state,
        write_privacy_state,
    )
    from kestrel_sovereign.privacy import PrivacyMode

    db = _MetadataDB()
    sovereign = OverrideProvenance(
        source=OverrideSource.SOVEREIGN,
        role="sovereign",
        via="/api/model/set",
        at="2026-07-20T17:00:00+00:00",
    )
    state = PrivacyState(
        mode=PrivacyMode.ISOLATED, saved_preference=_saved_outside(sovereign)
    )
    await write_privacy_state(db, _Agent.agent_id, state)

    loaded = await load_privacy_state(db, _Agent.agent_id)
    assert loaded == state
    assert loaded.saved_preference.provenance == sovereign

    # A provenance row that describes another mandate is not trusted.
    db.rows[(_Agent.agent_id, PRIVACY_SAVED_PROVENANCE_KEY)] = json.dumps({
        "saved_preference": {"vendor": "openai", "route": None, "model": "gpt-5"},
        "provenance": sovereign.to_dict(),
    })
    assert (await load_privacy_state(db, _Agent.agent_id)).saved_preference.provenance is None

    # A record written before the provenance row existed: the agent's.
    del db.rows[(_Agent.agent_id, PRIVACY_SAVED_PROVENANCE_KEY)]
    loaded = await load_privacy_state(db, _Agent.agent_id)
    assert loaded == state
    assert loaded.saved_preference.provenance is None


# --- Mandate fallbacks are overrides too ---------------------------------------


def _fallback_lineup():
    """An Anthropic-only chain with OpenRouter healthy outside it."""
    routes = {
        "anthropic:api": _Route("anthropic:api"),
        "openrouter:api": _Route("openrouter:api", model=OUTSIDE_MODEL),
    }
    service = _service(
        list(routes.values()), config={"route_priority": ["anthropic:api"]}
    )
    return routes, service


def _fallback_decisions(service):
    return [
        entry
        for entry in service.describe_provider_policy()["override_history"]
        if entry["kind"] == "fallback"
    ]


@pytest.mark.asyncio
async def test_an_unattributed_fallback_outside_the_chain_is_refused_and_never_served(
    caplog,
):
    """The round-4 reproduction: an agent-written Anthropic mandate, an
    unattributed OpenRouter fallback, Anthropic disabled after an auth failure.
    ``get_response`` used to answer from OpenRouter with no decision recorded."""
    routes, service = _fallback_lineup()
    service.set_model_preference("m", "anthropic")

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        with pytest.raises(ProviderPolicyRefusal) as refused:
            service.add_fallback_model(OUTSIDE_MODEL, "openrouter")

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert refused.value.provider == "openrouter"
    assert refused.value.source is OverrideSource.AGENT
    assert service.get_current_mandate()["fallbacks"] == []
    [decision] = _fallback_decisions(service)
    assert (decision["vendor"], decision["model"], decision["source"]) == (
        "openrouter", OUTSIDE_MODEL, "agent",
    )
    assert decision["via"] == "add_fallback_model"
    assert decision["accepted"] is False
    [warning] = _refusal_warnings(caplog)
    assert "kind=fallback" in warning.getMessage()
    assert "provider=openrouter" in warning.getMessage()

    # Nothing to fall back to: the mandate's task fails, naming its route.
    service._disabled_routes = {"anthropic:api": "401 after a key was revoked"}
    with pytest.raises(LLMProviderUnavailableError, match="anthropic"):
        await service.get_response(system_prompt="s", user_prompt="hi")
    assert routes["openrouter:api"].calls == 0


@pytest.mark.asyncio
async def test_an_unattributed_fallback_inside_the_chain_is_accepted_and_kept_with_its_writer():
    routes = {
        "anthropic:api": _Route("anthropic:api"),
        "openai:api": _Route("openai:api"),
    }
    service = _service(list(routes.values()), config={"route_priority": APPROVED_CHAIN})
    service.set_model_preference("m", "anthropic")

    service.add_fallback_model("gpt-5", "openai")

    # The fallback's public shape is unchanged; its writer is kept beside it.
    assert service.get_current_mandate()["fallbacks"] == [
        {"model": "gpt-5", "provider": "openai"}
    ]
    [decision] = _fallback_decisions(service)
    assert (decision["vendor"], decision["source"], decision["accepted"]) == (
        "openai", "agent", True,
    )
    [listed] = service.describe_provider_policy()["fallbacks"]
    assert (listed["vendor"], listed["model"]) == ("openai", "gpt-5")
    assert listed["provenance"]["source"] == "agent"

    service._disabled_routes = {"anthropic:api": "401 after a key was revoked"}
    assert await service.get_response(system_prompt="s", user_prompt="hi") == (
        "from openai:api"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "writer",
    [
        OverrideProvenance(source=OverrideSource.SOVEREIGN, role="sovereign"),
        OverrideProvenance(source=OverrideSource.OPERATOR, role="authenticated"),
    ],
)
async def test_a_principals_fallback_outside_the_chain_behaves_as_today(writer):
    routes, service = _fallback_lineup()
    service.set_model_preference("m", "anthropic")

    with override_writer(writer):
        service.add_fallback_model(OUTSIDE_MODEL, "openrouter")

    [decision] = _fallback_decisions(service)
    assert (decision["source"], decision["accepted"]) == (writer.source.value, True)
    service._disabled_routes = {"anthropic:api": "401 after a key was revoked"}
    assert await service.get_response(system_prompt="s", user_prompt="hi") == (
        "from openrouter:api"
    )
    assert routes["openrouter:api"].calls == 1


_FALLBACK_ENTRY_POINTS = {
    "get_response": lambda s: s.get_response(system_prompt="s", user_prompt="hi"),
    "generate_with_messages": lambda s: s.generate_with_messages(
        messages=list(MESSAGES)
    ),
    "get_streaming_response": lambda s: _drain(
        s.get_streaming_response(system_prompt="s", user_prompt="hi")
    ),
    "stream_with_messages": lambda s: _drain(
        s.stream_with_messages(messages=list(MESSAGES))
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", sorted(_FALLBACK_ENTRY_POINTS))
async def test_a_stored_fallback_with_no_recorded_writer_cannot_widen_the_chain(
    caplog, entry
):
    """A fallback written straight into the list, past ``add_fallback_model``,
    is the agent's like any stored override with no recorded writer: routing
    refuses it when it would select it, and never calls its provider."""
    routes, service = _fallback_lineup()
    service.set_model_preference("m", "anthropic")
    service._mandate_fallbacks.append({"vendor": "openrouter", "model": OUTSIDE_MODEL})
    service._disabled_routes = {"anthropic:api": "401 after a key was revoked"}

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        with pytest.raises(LLMError):
            await _FALLBACK_ENTRY_POINTS[entry](service)

    assert routes["openrouter:api"].adapter.method_calls == []
    assert routes["openrouter:api"].calls == 0
    decisions = _fallback_decisions(service)
    assert decisions and all(d["accepted"] is False for d in decisions)
    assert decisions[0]["vendor"] == "openrouter"
    assert decisions[0]["source"] == "agent"
    # No writer was recorded, so neither is when it was written.
    assert (decisions[0]["via"], decisions[0]["at"]) == ("mandate_fallback", None)
    assert _refusal_warnings(caplog)


def test_routing_raises_the_refusal_when_every_fallback_is_refused():
    _, service = _fallback_lineup()
    service.set_model_preference("m", "anthropic")
    service._mandate_fallbacks.append({"vendor": "openrouter", "model": OUTSIDE_MODEL})
    service._disabled_routes = {"anthropic:api": "401 after a key was revoked"}

    with pytest.raises(ProviderPolicyRefusal) as refused:
        service.resolve_provider_routing()

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert refused.value.provider == "openrouter"
    # The shared authorization helper never authorizes the refused vendor.
    _, authorized = service._compute_route_authorization()
    assert "openrouter" not in authorized


def test_a_refused_stored_fallback_is_recorded_once_per_task():
    """Routing and the authorization helper both consult the fallbacks; one
    refusal per task (and one outside any task) is the decision, not each
    consultation."""
    from kestrel_sovereign.telemetry import bind_current_turn_id

    _, service = _fallback_lineup()
    service.set_model_preference("m", "anthropic")
    # An agent's fallback stored in an earlier turn, outside the chain the
    # service now enforces.
    service._mandate_fallbacks.append({
        "vendor": "openrouter",
        "model": OUTSIDE_MODEL,
        FALLBACK_WRITER_KEY: OverrideProvenance(
            source=OverrideSource.AGENT, role="none", via="add_fallback_model",
            task="turn-0",
        ),
    })
    service._disabled_routes = {"anthropic:api": "401 after a key was revoked"}

    def _route_three_times():
        for _ in range(3):
            with pytest.raises(ProviderPolicyRefusal):
                service.resolve_provider_routing()
            service._compute_route_authorization()

    _route_three_times()
    with bind_current_turn_id("turn-1"):
        _route_three_times()
    with bind_current_turn_id("turn-2"):
        _route_three_times()

    assert [d["task"] for d in _fallback_decisions(service)] == [
        None, "turn-1", "turn-2",
    ]


@pytest.mark.asyncio
async def test_a_refused_fallback_is_skipped_and_the_admitted_ones_are_followed(
    incident,
):
    """Mixed fallbacks: the refused one is left out of the chain routing
    returns, and an admitted one still serves the task."""
    service = incident.service
    service.set_model_preference("m", "anthropic")
    service._mandate_fallbacks.append({"vendor": "openrouter", "model": OUTSIDE_MODEL})
    service.add_fallback_model("gpt-5", "openai")
    service._disabled_routes = {"anthropic:api": "401 after a key was revoked"}
    incident.routes["openai:api"].healthy = True

    providers, _ = service.resolve_provider_routing()

    assert [p["name"] for p in providers] == ["openai:api"]
    assert await service.get_response(system_prompt="s", user_prompt="hi") == (
        "from openai:api"
    )
    assert _outsiders_called(incident) == {}
    refused = [d for d in _fallback_decisions(service) if not d["accepted"]]
    assert [d["vendor"] for d in refused] == ["openrouter"]


def test_a_route_qualified_fallback_authorizes_its_vendor(incident):
    """``openai:api`` authorizes the vendor ``openai``, not a vendor named
    ``openai:api`` that no route has (which would skip the very route)."""
    service = incident.service
    with override_writer(
        OverrideProvenance(source=OverrideSource.OPERATOR, role="authenticated")
    ):
        service.set_model_preference("m", "openrouter", validate=False)
        service.add_fallback_model("gemini-3.1-pro-preview", "google:api")
    service._disabled_routes = {"openrouter:api": "401 after a key was revoked"}

    _, authorized = service._compute_route_authorization()

    assert "google" in authorized
    assert "google:api" not in authorized


# --- Decisions go through the same gate (round-5 finding) -------------------
#
# ``LLMService.decide(model_override=...)`` parsed and dispatched the requested
# provider without admission, and automatic decision routing used any
# decision-capable route. Reproduced: with an anthropic+openai chain, an
# unnamed ``openrouter/gemini`` override called OpenRouter and recorded nothing.

from kestrel_sdk.llm.decisions import (  # noqa: E402
    DecisionModelInfo,
    DecisionRequest,
    DecisionUnavailable,
    NoulQuestion,
    UnavailableReason,
)

from kestrel_sovereign.llm import (
    decision_service as decision_service_module,  # noqa: E402
)
from kestrel_sovereign.llm.decision_service import DecisionOverrideRefused  # noqa: E402
from kestrel_sovereign.llm.decisions.config import DecisionRouteConfig  # noqa: E402
from kestrel_sovereign.llm.decisions.resolve import RouteDecisionState  # noqa: E402
from kestrel_sovereign.llm.decisions.thresholds import ThresholdBook  # noqa: E402
from kestrel_sovereign.llm.provider_policy import (  # noqa: E402
    caller_call_override_scope,
    config_provenance,
    named_call_override_scope,
)

DECISION_OVERRIDE = "openrouter/gemini"


class _DecisionAdapter:
    """A decision route that records discovery and every decision it is sent."""

    def __init__(self, model):
        self.model = model
        self.discovery_calls = 0
        self.decide_calls = []

    async def list_decision_models(self, client):
        self.discovery_calls += 1
        return [
            DecisionModelInfo(id=self.model, vendor="", route="", context_limit=8192)
        ]

    async def adecide(self, client, model, request, *, timeout):
        self.decide_calls.append(model)
        return {
            "answers": {
                question: {"type": "noul", "noul": 0.1}
                for question in request.questions
            }
        }


def _decision_route(name, model, *, local=False):
    vendor, _, route = name.partition(":")
    adapter = _DecisionAdapter(model)
    return adapter, {
        "name": name, "vendor": vendor, "route": route,
        "is_local": local, "is_cloud": not local,
        "client": object(), "adapter": adapter, "model": "auto",
        "capabilities": {"supports_decisions": True},
        "decision_state": RouteDecisionState(config=DecisionRouteConfig()),
    }


def _decision_lineup(*names, config=None):
    """A real LLMService whose decision routes are ``names``, in that order."""
    models = {
        "openrouter:api": "gemini",
        "openai:api": "d1",
        "ollama:local": "tev1",
    }
    service = _service(
        [], config=config if config is not None else {"route_priority": APPROVED_CHAIN}
    )
    service._decision_thresholds = ThresholdBook.from_config({})
    service._track_model_usage = AsyncMock()
    adapters = {}
    providers = []
    for name in names:
        adapter, provider = _decision_route(
            name, models[name], local=name.endswith(":local")
        )
        adapters[name] = adapter
        providers.append(provider)
    service.providers = providers
    return SimpleNamespace(service=service, adapters=adapters)


def _decision_request():
    return DecisionRequest(
        state={"text": "The sky is blue."},
        questions={"q": NoulQuestion(instructions="Does `text` name a colour?")},
    )


async def _decide(service, model_override=None, **kwargs):
    return await service.decide(
        _decision_request(),
        caller="policy-test",
        timeout_seconds=5,
        model_override=model_override,
        **kwargs,
    )


def _untouched(adapter):
    return adapter.discovery_calls == 0 and adapter.decide_calls == []


@pytest.mark.asyncio
async def test_decide_refuses_an_unnamed_override_outside_the_chain(caplog):
    lineup = _decision_lineup("openrouter:api", "openai:api")
    service = lineup.service

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        with pytest.raises(DecisionOverrideRefused) as refused:
            await _decide(service, DECISION_OVERRIDE)

    # A DecisionError, so every caller handles it as a failed decision.
    assert isinstance(refused.value, DecisionUnavailable)
    assert refused.value.reason is UnavailableReason.NO_ROUTE
    assert refused.value.refusal.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert refused.value.to_dict()["provider"] == "openrouter"
    assert refused.value.refusal.source is OverrideSource.AGENT
    # Refused before discovery or dispatch: OpenRouter was never contacted.
    assert _untouched(lineup.adapters["openrouter:api"])
    [decision] = _call_decisions(service)
    assert (
        decision["vendor"], decision["model"], decision["source"],
        decision["via"], decision["accepted"],
    ) == ("openrouter", "gemini", "agent", "decide", False)
    [warning] = _refusal_warnings(caplog)
    assert "provider=openrouter" in warning.getMessage()
    assert "model=gemini" in warning.getMessage()


@pytest.mark.asyncio
async def test_decide_accepts_an_unnamed_override_inside_the_chain():
    lineup = _decision_lineup("openrouter:api", "openai:api")

    result = await _decide(lineup.service, "openai/d1")

    assert (result.route, result.model) == ("openai:api", "d1")
    assert _untouched(lineup.adapters["openrouter:api"])
    [decision] = _call_decisions(lineup.service)
    assert (decision["vendor"], decision["source"], decision["accepted"]) == (
        "openai", "agent", True,
    )


_ATTESTED_OPERATOR = {
    "configured": lambda: named_call_override_scope(
        DECISION_OVERRIDE,
        config_provenance(via="[retrieval] memory_answerability_decision_model"),
    ),
    "caller": lambda: caller_call_override_scope(
        DECISION_OVERRIDE, OPERATOR, via="model_override"
    ),
    "writer": lambda: override_writer(
        OverrideProvenance(source=OverrideSource.OPERATOR, role="authenticated")
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("attestation", sorted(_ATTESTED_OPERATOR))
async def test_decide_accepts_an_attested_operator_override_outside_the_chain(
    attestation,
):
    lineup = _decision_lineup("openrouter:api", "openai:api")

    with _ATTESTED_OPERATOR[attestation]():
        result = await _decide(lineup.service, DECISION_OVERRIDE)

    assert (result.route, result.model) == ("openrouter:api", "gemini")
    assert lineup.adapters["openrouter:api"].decide_calls == ["gemini"]
    [decision] = _call_decisions(lineup.service)
    assert (decision["vendor"], decision["source"], decision["accepted"]) == (
        "openrouter", "operator", True,
    )


@pytest.mark.asyncio
async def test_a_configured_decision_model_names_only_its_own_override():
    """A component's configured model is the operator's; any other override
    in the same scope is still the agent's."""
    lineup = _decision_lineup("openrouter:api", "openai:api")
    writer = config_provenance(via="KESTREL_RESPONSE_AUDIT_DECISION_MODEL")

    with named_call_override_scope("openrouter:api/gemini", writer):
        with pytest.raises(DecisionOverrideRefused):
            await _decide(lineup.service, DECISION_OVERRIDE)

    assert _untouched(lineup.adapters["openrouter:api"])


@pytest.mark.asyncio
async def test_repeated_accepted_decisions_are_recorded_once_and_refusals_each_time(
    caplog,
):
    """A sleep cycle asks one decision per memory, outside any turn. Its
    accepted calls are one decision, so they cannot evict refusals from the
    bounded history; every refusal is still recorded and logged."""
    lineup = _decision_lineup("openrouter:api", "openai:api")
    service = lineup.service

    with named_call_override_scope(
        DECISION_OVERRIDE,
        config_provenance(via="[retrieval] memory_attestation_decision_model"),
    ):
        for _ in range(5):
            await _decide(service, DECISION_OVERRIDE)
    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        for _ in range(2):
            with pytest.raises(DecisionOverrideRefused):
                await _decide(service, DECISION_OVERRIDE)

    assert lineup.adapters["openrouter:api"].decide_calls == ["gemini"] * 5
    assert [d["accepted"] for d in _call_decisions(service)] == [True, False, False]
    assert len(_refusal_warnings(caplog)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ["groq", "groq:api", "llama3.2:3b", "auto"])
async def test_a_decision_selector_naming_no_initialized_vendor_reaches_no_route(
    selector,
):
    """Admission recognizes every selector that can match a route: one naming
    no initialized vendor is not admitted, and matches no route either."""
    lineup = _decision_lineup("openrouter:api", "openai:api")

    with pytest.raises(DecisionUnavailable) as unavailable:
        await _decide(lineup.service, selector)

    assert not isinstance(unavailable.value, DecisionOverrideRefused)
    assert all(_untouched(adapter) for adapter in lineup.adapters.values())
    assert _call_decisions(lineup.service) == []


# The round-3 reproduction: ``parse_decision_selector`` strips the override, so
# `` openrouter `` routed to OpenRouter, while admission read it unstripped,
# recognized no provider and admitted nothing. OpenRouter answered and the
# audit stayed empty.
PADDED_DECISION_OVERRIDES = {
    " openrouter ": None,
    "openrouter\n": None,
    "\topenrouter:api ": None,
    " openrouter/gemini ": "gemini",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", sorted(PADDED_DECISION_OVERRIDES))
async def test_decide_admits_a_padded_override_as_it_routes_it(selector, caplog):
    lineup = _decision_lineup("openrouter:api", "openai:api")
    service = lineup.service

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        with pytest.raises(DecisionOverrideRefused) as refused:
            await _decide(service, selector)

    assert refused.value.refusal.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert refused.value.to_dict()["vendor"] == "openrouter"
    # Refused before discovery or dispatch: no route was contacted.
    assert all(_untouched(adapter) for adapter in lineup.adapters.values())
    [decision] = _call_decisions(service)
    assert (
        decision["vendor"], decision["model"], decision["source"],
        decision["via"], decision["accepted"],
    ) == ("openrouter", PADDED_DECISION_OVERRIDES[selector], "agent", "decide", False)
    [warning] = _refusal_warnings(caplog)
    assert "provider=openrouter" in warning.getMessage()


@pytest.mark.asyncio
async def test_decide_admits_the_provider_its_own_parse_selects(monkeypatch):
    """Admission reads the selector ``decide`` routes on, not the raw string:
    a parse that reads an override some way the policy's reader does not
    (here, ignoring case) still cannot route around admission."""
    lineup = _decision_lineup("openrouter:api", "openai:api")
    real_parse = decision_service_module.parse_decision_selector
    monkeypatch.setattr(
        decision_service_module,
        "parse_decision_selector",
        lambda raw: real_parse(raw.lower()),
    )

    with pytest.raises(DecisionOverrideRefused) as refused:
        await _decide(lineup.service, "OpenRouter/gemini")

    assert refused.value.to_dict()["provider"] == "openrouter"
    assert all(_untouched(adapter) for adapter in lineup.adapters.values())


@pytest.mark.asyncio
async def test_decide_accepts_a_padded_override_the_operator_named():
    """The writer is still looked up by the override exactly as passed, so an
    operator's configured decision model with stray whitespace stays theirs."""
    lineup = _decision_lineup("openrouter:api", "openai:api")
    padded = " openrouter/gemini "

    with named_call_override_scope(
        padded, config_provenance(via="[retrieval] memory_attestation_decision_model")
    ):
        result = await _decide(lineup.service, padded)

    assert (result.route, result.model) == ("openrouter:api", "gemini")
    [decision] = _call_decisions(lineup.service)
    assert (decision["vendor"], decision["source"], decision["accepted"]) == (
        "openrouter", "operator", True,
    )


@pytest.mark.asyncio
async def test_automatic_decision_routing_keeps_to_the_approved_chain():
    """The incident shape for decisions: the only decision routes are outside
    the chain, so automatic routing fails rather than pick one."""
    lineup = _decision_lineup("openrouter:api")

    with pytest.raises(DecisionUnavailable) as unavailable:
        await _decide(lineup.service)

    assert unavailable.value.reason is UnavailableReason.NO_ROUTE
    assert "approved chain" in str(unavailable.value)
    assert _untouched(lineup.adapters["openrouter:api"])

    # With a chain route present it is used, though the outsider is listed first.
    lineup = _decision_lineup("openrouter:api", "openai:api")
    result = await _decide(lineup.service)
    assert result.route == "openai:api"
    assert _untouched(lineup.adapters["openrouter:api"])


@pytest.mark.asyncio
async def test_automatic_decision_routing_without_a_declared_chain_is_unchanged():
    lineup = _decision_lineup("openrouter:api", "openai:api", config={})

    result = await _decide(lineup.service)

    assert result.route == "openrouter:api"


@pytest.mark.asyncio
async def test_an_explicit_decision_route_is_the_operators_choice():
    lineup = _decision_lineup(
        "openrouter:api", "openai:api",
        config={"route_priority": APPROVED_CHAIN, "decision_route": "openrouter:api"},
    )

    result = await _decide(lineup.service)

    assert result.route == "openrouter:api"


@pytest.mark.asyncio
async def test_local_only_decisions_use_the_local_route_privacy_forces():
    """As in chat routing, a local-only mode's local routes are not narrowed
    by the chain: they are Kestrel's choice, not an override."""
    lineup = _decision_lineup("openrouter:api", "ollama:local")

    result = await _decide(lineup.service, local_only=True)

    assert result.route == "ollama:local"
    assert _untouched(lineup.adapters["openrouter:api"])


@pytest.mark.asyncio
async def test_a_denied_provider_is_never_a_decision_route():
    config = {
        "route_priority": APPROVED_CHAIN,
        "denied_providers": ["openrouter"],
        "decision_route": "openrouter:api",
    }
    lineup = _decision_lineup("openrouter:api", "openai:api", config=config)

    # Not by the operator's explicit decision route...
    with pytest.raises(DecisionUnavailable):
        await _decide(lineup.service)
    # ...and not by an operator's override.
    with override_writer(
        OverrideProvenance(source=OverrideSource.OPERATOR, role="authenticated")
    ):
        with pytest.raises(DecisionOverrideRefused) as refused:
            await _decide(lineup.service, DECISION_OVERRIDE)

    assert refused.value.refusal.code == ProviderPolicyRefusal.PROVIDER_DENIED
    assert _untouched(lineup.adapters["openrouter:api"])


# The components that pass on a decision model the operator configured name
# the operator as its writer, so a configured model outside the chain is used
# as it is today. An unnamed one (the same model given by anything else) is
# the agent's and is refused.


@pytest.mark.asyncio
async def test_the_answerability_gate_names_its_configured_decision_model():
    from kestrel_sovereign.storage import memory_system as memory_system_module
    from kestrel_sovereign.storage.memory_answerability import (
        AnswerabilityCandidate,
        DecisionAnswerabilityGate,
    )

    lineup = _decision_lineup("openrouter:api", "openai:api")
    service = lineup.service
    service._current_force_local_only = lambda: False
    settings = memory_system_module.AnswerabilitySettings(
        enabled=True, timeout_seconds=5.0, model=None,
        backend="decision", decision_model=DECISION_OVERRIDE,
    )
    candidates = [AnswerabilityCandidate("m1", "The sky is blue.")]

    configured = memory_system_module._build_answerability_gate(service, settings)
    decision = await configured.filter("What colour is the sky?", candidates)
    unnamed = DecisionAnswerabilityGate(service, model_override=DECISION_OVERRIDE)
    refused = await unnamed.filter("What colour is the sky?", candidates)

    assert decision.completed is True
    assert refused.completed is False
    assert refused.reason == "decision_error:DecisionOverrideRefused"
    assert lineup.adapters["openrouter:api"].decide_calls == ["gemini"]
    assert [(d["source"], d["via"], d["accepted"]) for d in _call_decisions(service)] == [
        ("operator", "[retrieval] memory_answerability_decision_model", True),
        ("agent", "decide", False),
    ]


@pytest.mark.asyncio
async def test_the_chat_answerability_gate_names_its_configured_model(incident):
    from kestrel_sovereign.storage import memory_system as memory_system_module
    from kestrel_sovereign.storage.memory_answerability import AnswerabilityCandidate

    service = incident.service
    settings = memory_system_module.AnswerabilitySettings(
        enabled=True, timeout_seconds=5.0, model=OVERRIDE,
    )
    gate = memory_system_module._build_answerability_gate(service, settings)

    await gate.filter(
        "What colour is the sky?", [AnswerabilityCandidate("m1", "Blue.")]
    )

    assert incident.routes["openrouter:api"].calls == 1
    [decision] = _call_decisions(service)
    assert (decision["source"], decision["via"], decision["accepted"]) == (
        "operator", "[retrieval] memory_answerability_model", True,
    )


@pytest.mark.asyncio
async def test_the_memory_attestation_hook_names_its_configured_decision_model():
    from kestrel_sovereign.features.memory.feature import MemoryFeature
    from kestrel_sovereign.features.memory.reflection_hook import (
        RetrievedMemoryCandidate,
    )

    lineup = _decision_lineup("openrouter:api", "openai:api")
    service = lineup.service
    agent = SimpleNamespace(sleep_hooks=[], llm_service=service)
    with patch(
        "kestrel_sovereign.storage.memory_system.load_section",
        return_value={
            "memory_attestation_backend": "decision",
            "memory_attestation_decision_model": DECISION_OVERRIDE,
        },
    ):
        await MemoryFeature.__new__(MemoryFeature).post_all_features_loaded(agent)
    [hook] = agent.sleep_hooks
    memory = SimpleNamespace(mark_applied=AsyncMock())
    marks = SimpleNamespace(record=AsyncMock(return_value=True))
    candidate = RetrievedMemoryCandidate(
        message_id=1, content="The sky is blue.", retrieved_at="", created_at=None,
        role="assistant", session_context="the turn that retrieved it",
    )

    result = await hook._attest_with_decisions(agent, memory, [candidate], marks)

    assert result["success"] is True
    assert lineup.adapters["openrouter:api"].decide_calls == ["gemini"]
    [decision] = _call_decisions(service)
    assert (decision["source"], decision["via"], decision["accepted"]) == (
        "operator", "[retrieval] memory_attestation_decision_model", True,
    )


@pytest.mark.asyncio
async def test_the_response_audit_names_its_configured_decision_model(monkeypatch):
    from kestrel_sovereign.features.response_audit.feature import ResponseAuditFeature

    lineup = _decision_lineup("openrouter:api", "openai:api")
    agent = Mock()
    agent.llm_service = lineup.service
    agent.features = {}
    monkeypatch.setenv("KESTREL_RESPONSE_AUDIT_BACKEND", "decision")
    monkeypatch.setenv("KESTREL_RESPONSE_AUDIT_DECISION_MODEL", DECISION_OVERRIDE)

    audit = ResponseAuditFeature(agent)._decision_auditor()
    verdict = await audit("An ordinary helpful answer.", False)

    assert verdict["audited"] is True
    assert lineup.adapters["openrouter:api"].decide_calls == ["gemini"]
    [decision] = _call_decisions(lineup.service)
    assert (decision["source"], decision["via"]) == (
        "operator", "KESTREL_RESPONSE_AUDIT_DECISION_MODEL",
    )


@pytest.mark.asyncio
async def test_the_continuation_check_names_its_configured_decision_model(monkeypatch):
    from kestrel_sovereign.turn_completion import (
        CONTINUATION_CHECK_ENV,
        CONTINUATION_CHECK_MODEL_ENV,
        confirm_unfinished,
    )

    lineup = _decision_lineup("openrouter:api", "openai:api")
    monkeypatch.setenv(CONTINUATION_CHECK_ENV, "decision")
    monkeypatch.setenv(CONTINUATION_CHECK_MODEL_ENV, DECISION_OVERRIDE)

    # p(unfinished) = 0.1: a decision that ran says no repair; a refused one
    # would repair.
    assert await confirm_unfinished(lineup.service, "Done: the file is written.") is False
    assert lineup.adapters["openrouter:api"].decide_calls == ["gemini"]
    [decision] = _call_decisions(lineup.service)
    assert (decision["source"], decision["via"]) == (
        "operator", CONTINUATION_CHECK_MODEL_ENV,
    )


@pytest.mark.asyncio
async def test_the_decisions_eval_cli_names_the_operator_for_each_model(capsys):
    """``kestrel decisions eval`` evaluates every model the operator selected,
    including one outside the agent's chain."""
    import argparse

    from kestrel_sovereign import cli_decisions

    lineup = _decision_lineup("openrouter:api", "openai:api")
    args = argparse.Namespace(
        samples=None, caller="memory_answerability", local_only=False,
        allow_cloud=True, route=["openrouter:api"], model=None, baseline=None,
        timeout=5.0, concurrency=1, target_accuracy=0.9, json=None,
    )

    await cli_decisions._eval(lineup.service, args)

    assert lineup.adapters["openrouter:api"].decide_calls
    decisions = _call_decisions(lineup.service)
    assert decisions
    assert {(d["vendor"], d["source"], d["role"], d["via"], d["accepted"]) for d in decisions} == {
        ("openrouter", "operator", "cli", "kestrel decisions eval", True),
    }


# --- The response audit and the embedding route keep to the chain -----------


@pytest.mark.asyncio
async def test_the_response_audit_never_sends_a_response_outside_the_chain(incident):
    """The chat audit tried every available route, so with the approved chain
    down it sent the assistant's response to OpenRouter and Google."""
    verdict = await incident.service.get_audit_response("An assistant response.")

    assert verdict["audited"] is False
    assert verdict["risk_level"] == 3
    assert "outside the approved chain" in verdict["reasoning"]
    assert _outsiders_called(incident) == {}
    assert incident.routes["anthropic:api"].calls == 1
    assert incident.routes["openai:api"].calls == 1


@pytest.mark.asyncio
async def test_the_response_audit_follows_the_operators_preferred_model(incident):
    """``model_mandate.toml``'s ``[defaults] preferred`` is the operator's, so
    the audit still runs on it outside the chain, as today."""
    incident.service.mandate_config = {"defaults": {"preferred": OVERRIDE}}

    await incident.service.get_audit_response("An assistant response.")

    assert incident.routes["openrouter:api"].calls == 1
    assert incident.routes["google:api"].calls == 0


@pytest.mark.asyncio
async def test_a_bare_model_audit_target_authorizes_no_vendor(incident):
    """A model-only mandate names no provider, so the route a bare model
    happens to resolve to is not authorized for the audit."""
    incident.service._mandate_preference = {
        "vendor": None, "route": None, "model": "gemini-3.1-pro-preview",
    }

    verdict = await incident.service.get_audit_response("An assistant response.")

    assert verdict["audited"] is False
    assert _outsiders_called(incident) == {}


@pytest.mark.asyncio
async def test_the_response_audit_without_a_declared_chain_is_unchanged(incident):
    incident.service.config.pop("route_priority")

    await incident.service.get_audit_response("An assistant response.")

    assert incident.routes["openrouter:api"].calls == 1


def test_the_embedding_route_never_follows_chat_outside_the_chain(incident):
    service = incident.service
    for provider in service.providers:
        provider["capabilities"] = {"supports_embeddings": True}
    service._disabled_routes = {
        "anthropic:api": "401 after a key was revoked",
        "openai:api": "401 after a key was revoked",
    }

    assert service.resolve_embedding_provider() is None

    service.config.pop("route_priority")
    assert service.resolve_embedding_provider()["name"] == "openrouter:api"


# --- The lease route admits first; every dispatch is guarded (round-7 finding)
#
# ``_generate_with_messages_inner`` took the private inference lease route
# before admission, and ``_remote_first_allowed`` read a bare vendor such as
# ``openrouter`` as a model id. Reproduced: with an Anthropic-only chain and an
# active lease, ``generate_with_messages(model_override="openrouter")`` called
# the lease adapter with model ``openrouter`` and recorded nothing. All five
# entry points share ``_remote_route_attempt``, so it admits there; and every
# adapter dispatch now passes ``_guard_dispatch``, so a path that skips
# admission fails closed.

from datetime import UTC, datetime, timedelta  # noqa: E402
from decimal import Decimal  # noqa: E402

from kestrel_sdk.llm import (  # noqa: E402
    InferenceLease,
    InferenceLeaseState,
    InferencePrivacy,
    InferenceRoute,
)
from pydantic import SecretStr  # noqa: E402

from kestrel_sovereign.llm.provider_policy import (  # noqa: E402
    known_vendors,
)
from kestrel_sovereign.llm.remote_backend import REMOTE_ROUTE_NAME  # noqa: E402

LEASE_MODEL = "llama-3"


class _LeaseAdapter:
    """The lease route's adapter: records every call it is sent."""

    def __init__(self):
        self.calls = []

    def create_messages(self, *, user_prompt=None, system_prompt=None, **_):
        self.calls.append(("create_messages", None))
        return [{"role": "user", "content": user_prompt or ""}]

    async def get_response(self, **kwargs):
        self.calls.append(("get_response", kwargs.get("model")))
        return attach_stop_reason(
            LLMResponse(content="from lease", input_tokens=1, output_tokens=1),
            "stop",
        )

    async def get_streaming_response(self, **kwargs):
        self.calls.append(("get_streaming_response", kwargs.get("model")))
        yield "from lease"

    async def get_streaming_response_with_tools(self, **kwargs):
        self.calls.append(("get_streaming_response_with_tools", kwargs.get("model")))
        yield attach_stop_reason(LLMResponse(content="from lease"), "stop")


class _LeaseClient:
    async def close(self):
        pass


def _inference_lease(model=LEASE_MODEL):
    now = datetime.now(UTC)
    return InferenceLease(
        lease_id="lease-1",
        quote_id="quote-1",
        request_id="request-1",
        owner_id="agent-1",
        provider_name="runpod",
        state=InferenceLeaseState.READY,
        model=model,
        runtime="ollama",
        privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(minutes=10),
        hourly_cost_usd=Decimal("0.50"),
        estimated_total_cost_usd=Decimal("0.20"),
        route=InferenceRoute(
            endpoint=SecretStr("https://private.example.test/v1"),
            model=model,
            api_key=SecretStr("route-secret"),
        ),
    )


async def _with_active_lease(service, monkeypatch, lease_model=LEASE_MODEL):
    """Activate a ready lease on ``service`` whose adapter records each call."""
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **_kwargs: _LeaseClient(),
    )
    touched = []

    async def touch(lease_id):
        touched.append(lease_id)
        return service._remote_lease

    await service.activate_inference_lease(
        _inference_lease(lease_model),
        capabilities=("chat", "streaming", "tools", "structured_output", "vision"),
        touch_lease=touch,
    )
    adapter = _LeaseAdapter()
    service._remote_adapter = adapter
    return SimpleNamespace(adapter=adapter, touched=touched)


def _lease_lineup(config):
    routes = {
        "anthropic:api": _Route("anthropic:api"),
        "openai:api": _Route("openai:api"),
        "openrouter:api": _Route("openrouter:api", model=OUTSIDE_MODEL),
    }
    return SimpleNamespace(
        service=_service(list(routes.values()), config=config), routes=routes
    )


_REMOTE_ENTRY_POINTS = {
    "generate": lambda s, o: s.generate(
        system_prompt="s", user_prompt="hi", model_override=o
    ),
    "generate_with_messages": lambda s, o: s.generate_with_messages(
        messages=list(MESSAGES), model_override=o
    ),
    "generate_stream": lambda s, o: _drain(s.generate_stream(
        system_prompt="s", user_prompt="hi", model_override=o
    )),
    "stream_with_messages": lambda s, o: _drain(s.stream_with_messages(
        messages=list(MESSAGES), model_override=o
    )),
    "stream_with_tool_detection": lambda s, o: _drain(s.stream_with_tool_detection(
        messages=list(MESSAGES), model_override=o
    )),
}

_LEASE_REFUSALS = {
    "outside_chain": (
        {"route_priority": ["anthropic:api"]},
        OVERRIDE,
        ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN,
    ),
    "outside_chain_bare_vendor": (
        {"route_priority": ["anthropic:api"]},
        "openrouter",
        ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN,
    ),
    "denied": (
        {"denied_providers": ["openrouter"]},
        OVERRIDE,
        ProviderPolicyRefusal.PROVIDER_DENIED,
    ),
    "denied_bare_vendor": (
        {"denied_providers": ["openrouter"]},
        "openrouter",
        ProviderPolicyRefusal.PROVIDER_DENIED,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(_LEASE_REFUSALS))
@pytest.mark.parametrize("entry", sorted(_REMOTE_ENTRY_POINTS))
async def test_an_active_lease_never_serves_a_refused_override(
    monkeypatch, caplog, entry, case
):
    config, override, code = _LEASE_REFUSALS[case]
    lineup = _lease_lineup(config)
    service = lineup.service
    lease = await _with_active_lease(service, monkeypatch)
    try:
        with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
            with pytest.raises(ProviderPolicyRefusal) as refused:
                await _REMOTE_ENTRY_POINTS[entry](service, override)

        assert refused.value.code == code
        assert refused.value.provider == "openrouter"
        # Refused before the lease was consulted: no renewal, no adapter call,
        # on the lease route or any other.
        assert lease.touched == []
        assert lease.adapter.calls == []
        assert {name: r.calls for name, r in lineup.routes.items()} == {
            name: 0 for name in lineup.routes
        }
        [decision] = service.describe_provider_policy()["override_history"]
        assert (
            decision["kind"], decision["vendor"], decision["source"],
            decision["accepted"],
        ) == ("call", "openrouter", "agent", False)
        assert decision["refusal"]["error"] == code
        assert len(_refusal_warnings(caplog)) == 1
        assert service.get_backend_status()["remote_active"] is True
    finally:
        await service.deactivate_inference_lease("lease-1")


@pytest.mark.asyncio
async def test_a_bare_model_id_still_takes_the_lease_route(monkeypatch):
    lineup = _lease_lineup({"route_priority": ["anthropic:api"]})
    service = lineup.service
    lease = await _with_active_lease(service, monkeypatch)
    try:
        response = await service.generate_with_messages(
            messages=list(MESSAGES), model_override=LEASE_MODEL
        )
    finally:
        await service.deactivate_inference_lease("lease-1")

    assert response == "from lease"
    assert lease.adapter.calls == [("get_response", LEASE_MODEL)]
    assert service.describe_provider_policy()["override_history"] == []


@pytest.mark.asyncio
async def test_an_override_admitted_at_the_lease_entry_is_recorded_once(monkeypatch):
    """The lease entry admits it, routing admits it again: one decision."""
    lineup = _lease_lineup({"route_priority": ["anthropic:api", "openai:api"]})
    service = lineup.service
    lease = await _with_active_lease(service, monkeypatch)
    try:
        response = await service.generate_with_messages(
            messages=list(MESSAGES), model_override="openai/gpt-5"
        )
    finally:
        await service.deactivate_inference_lease("lease-1")

    assert response == "from openai:api"
    assert lease.adapter.calls == []
    history = service.describe_provider_policy()["override_history"]
    assert [(d["kind"], d["vendor"], d["accepted"]) for d in history] == [
        ("call", "openai", True),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [OVERRIDE, "openrouter"])
@pytest.mark.parametrize("entry", sorted(_REMOTE_ENTRY_POINTS))
async def test_the_lease_entry_admits_even_when_the_vendor_check_lets_it_through(
    monkeypatch, entry, override
):
    """Admission at the lease entry does not lean on ``_remote_first_allowed``:
    were that check to let a provider selector through, the override is still
    refused before the lease is consulted."""
    lineup = _lease_lineup({"route_priority": ["anthropic:api"]})
    service = lineup.service
    lease = await _with_active_lease(service, monkeypatch)
    monkeypatch.setattr(service, "_remote_first_allowed", lambda _override: True)
    try:
        with pytest.raises(ProviderPolicyRefusal) as refused:
            await _REMOTE_ENTRY_POINTS[entry](service, override)
    finally:
        await service.deactivate_inference_lease("lease-1")

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert lease.touched == []
    assert lease.adapter.calls == []


@pytest.mark.asyncio
async def test_an_in_chain_bare_vendor_goes_to_its_vendor_not_to_the_lease(
    monkeypatch,
):
    """``anthropic`` names a provider, so it is not sent to the lease as a
    model id: the call is routed to Anthropic, and recorded once."""
    lineup = _lease_lineup({"route_priority": ["anthropic:api", "openai:api"]})
    service = lineup.service
    lease = await _with_active_lease(service, monkeypatch)
    try:
        response = await service.generate_with_messages(
            messages=list(MESSAGES), model_override="anthropic"
        )
    finally:
        await service.deactivate_inference_lease("lease-1")

    assert response == "from anthropic:api"
    assert lease.adapter.calls == []
    history = service.describe_provider_policy()["override_history"]
    assert [(d["kind"], d["vendor"], d["accepted"]) for d in history] == [
        ("call", "anthropic", True),
    ]


_BARE_VENDORS = {
    "an initialized route's vendor": ({}, "openrouter"),
    "padded": ({}, " openrouter "),
    "a configured vendor whose route did not come up": (
        {"vendors": {"mistral": {"routes": {}}}}, "mistral",
    ),
    "a deny-listed vendor": ({"denied_providers": ["deepseek"]}, "deepseek"),
    "a vendor the chain names": ({"route_priority": ["xai:api"]}, "xai"),
}


@pytest.mark.parametrize("case", sorted(_BARE_VENDORS))
def test_a_bare_vendor_selector_is_never_a_lease_model_id(case):
    config, selector = _BARE_VENDORS[case]
    service = _lease_lineup(config).service

    assert service._remote_first_allowed(selector) is False
    assert selector.strip() in known_vendors(service.config, service.providers)


@pytest.mark.parametrize("model", [LEASE_MODEL, "gpt-5", None])
def test_a_bare_model_id_may_still_take_the_lease_route(model):
    service = _lease_lineup({"route_priority": ["anthropic:api"]}).service

    assert service._remote_first_allowed(model) is True


def _dispatch_decisions(service):
    return [
        entry
        for entry in service.describe_provider_policy()["override_history"]
        if entry["kind"] == "dispatch"
    ]


def _unlisted_attempt(service, route, *, provider_name=None, model=OUTSIDE_MODEL, **kw):
    """Dispatch to ``route`` the way an unlisted path would: no admission."""
    attempt = route.adapter.get_response(model=model, messages=list(MESSAGES))
    return attempt, service._run_provider_attempt(
        attempt,
        provider_name or route.name,
        model,
        path="unlisted",
        invocation_context=service._resolve_invocation_context(),
        **kw,
    )


@pytest.mark.asyncio
async def test_the_dispatch_guard_refuses_an_unadmitted_path(incident, caplog):
    """No entry point admitted this call, so nothing narrowed it: the guard at
    the adapter refuses it before the adapter runs, and records who did it."""
    service = incident.service
    route = incident.routes["openrouter:api"]
    attempt, dispatch = _unlisted_attempt(service, route)

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        with pytest.raises(ProviderPolicyRefusal) as refused:
            await dispatch

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert refused.value.provider == "openrouter:api"
    assert route.calls == 0
    assert attempt.cr_frame is None  # closed, never run
    [decision] = _dispatch_decisions(service)
    assert (
        decision["vendor"], decision["route"], decision["model"],
        decision["source"], decision["via"], decision["accepted"],
    ) == ("openrouter", "api", OUTSIDE_MODEL, "agent", "dispatch:unlisted", False)
    [warning] = _refusal_warnings(caplog)
    assert "kind=dispatch" in warning.getMessage()
    assert "provider=openrouter:api" in warning.getMessage()


@pytest.mark.asyncio
async def test_the_dispatch_guard_refuses_an_unadmitted_streaming_path(incident):
    service = incident.service
    route = incident.routes["openrouter:api"]
    stream = service._stream_adapter_with_usage(
        adapter=route.adapter,
        client=AsyncMock(),
        model=OUTSIDE_MODEL,
        messages=list(MESSAGES),
        provider_name="openrouter:api",
        path="unlisted",
        invocation_context=service._resolve_invocation_context(),
        expose_protocol_events=True,
    )

    with pytest.raises(ProviderPolicyRefusal) as refused:
        await _drain(stream)

    assert refused.value.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert route.adapter.method_calls == []
    assert [d["via"] for d in _dispatch_decisions(service)] == ["dispatch:unlisted"]


@pytest.mark.asyncio
async def test_the_streaming_fallback_runner_guards_its_own_dispatch(incident):
    """With no invocation context it awaits the adapter itself, not through
    ``_run_provider_attempt``; the guard runs there too."""
    service = incident.service
    route = incident.routes["openrouter:api"]
    attempt = route.adapter.get_response(model=OUTSIDE_MODEL)

    with pytest.raises(ProviderPolicyRefusal):
        await service._run_stream_fallback_attempt(
            attempt,
            provider_name="openrouter:api",
            model=OUTSIDE_MODEL,
            path="unlisted",
            invocation_context=None,
        )

    assert route.calls == 0
    assert attempt.cr_frame is None
    assert [d["via"] for d in _dispatch_decisions(service)] == ["dispatch:unlisted"]


@pytest.mark.asyncio
async def test_the_dispatch_guard_refuses_a_denied_route_whoever_chose_it(incident):
    service = incident.service
    service.config = {"denied_providers": ["openrouter"]}
    route = incident.routes["openrouter:api"]

    with override_writer(
        OverrideProvenance(source=OverrideSource.SOVEREIGN, role="sovereign")
    ):
        attempt, dispatch = _unlisted_attempt(service, route)
        with pytest.raises(ProviderPolicyRefusal) as refused:
            await dispatch

    assert refused.value.code == ProviderPolicyRefusal.PROVIDER_DENIED
    assert route.calls == 0


@pytest.mark.asyncio
async def test_the_dispatch_guard_holds_the_lease_route_to_an_active_lease(
    incident, monkeypatch
):
    """The lease route serves only an active lease, and only a model that
    names no provider."""
    service = incident.service
    lease_adapter = _LeaseAdapter()

    def lease_dispatch(model):
        return service._run_provider_attempt(
            lease_adapter.get_response(model=model),
            REMOTE_ROUTE_NAME,
            model,
            path="unlisted",
            invocation_context=service._resolve_invocation_context(),
        )

    with pytest.raises(ProviderPolicyRefusal):
        await lease_dispatch(LEASE_MODEL)  # no lease is active

    await _with_active_lease(service, monkeypatch)
    try:
        with pytest.raises(ProviderPolicyRefusal) as refused:
            await lease_dispatch("openrouter")
        assert refused.value.provider == "openrouter"
        response = await lease_dispatch(LEASE_MODEL)
    finally:
        await service.deactivate_inference_lease("lease-1")

    assert response.content == "from lease"
    assert lease_adapter.calls == [("get_response", LEASE_MODEL)]
    assert [d["vendor"] for d in _dispatch_decisions(service)] == [
        REMOTE_ROUTE_NAME, "openrouter",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("lease_model", ["Qwen/Qwen3-8B", "openai/gpt-oss-20b"])
async def test_a_lease_whose_model_id_looks_like_a_selector_still_serves(
    monkeypatch, lease_model
):
    """A lease's own model id is the lease's, even when it reads like
    ``vendor/model`` and names a vendor this host knows."""
    lineup = _lease_lineup({"route_priority": ["anthropic:api", "openai:api"]})
    service = lineup.service
    lease = await _with_active_lease(service, monkeypatch, lease_model)
    try:
        response = await service.generate_with_messages(messages=list(MESSAGES))
    finally:
        await service.deactivate_inference_lease("lease-1")

    assert response == "from lease"
    assert lease.adapter.calls == [("get_response", lease_model)]
    assert service.describe_provider_policy()["override_history"] == []


@pytest.mark.asyncio
async def test_the_dispatch_guard_admits_what_routing_may_reach(incident):
    """A chain route, a local route on a local-only call, and a route the
    Sovereign's mandate names pass the guard, and it records nothing."""
    service = incident.service
    incident.routes["anthropic:api"].healthy = True
    local = _Route("ollama:local")
    local_provider = dict(local.as_provider(), is_local=True, is_cloud=False)
    service.providers.append(local_provider)

    _, dispatch = _unlisted_attempt(
        service, incident.routes["anthropic:api"], model="m"
    )
    assert (await dispatch).content == "from anthropic:api"

    _, dispatch = _unlisted_attempt(service, local, model="m", force_local_only=True)
    assert (await dispatch).content == "from ollama:local"
    _, dispatch = _unlisted_attempt(service, local, model="m")
    with pytest.raises(ProviderPolicyRefusal):
        await dispatch
    assert local.calls == 1

    with override_writer(
        OverrideProvenance(source=OverrideSource.SOVEREIGN, role="sovereign")
    ):
        service.set_model_preference(OUTSIDE_MODEL, "openrouter")
    _, dispatch = _unlisted_attempt(service, incident.routes["openrouter:api"])
    assert (await dispatch).content == "from openrouter:api"

    assert [d["vendor"] for d in _dispatch_decisions(service)] == ["ollama"]


def _local_streaming_route(service):
    """A local route outside the chain whose adapter streams and records."""
    adapter = _LeaseAdapter()
    service.providers.append({
        "name": "ollama:local", "vendor": "ollama", "route": "local",
        "client": AsyncMock(), "adapter": adapter, "model": "m",
        "is_cloud": False, "is_local": True, "base_url": None,
        "capabilities": {},
    })
    return adapter


@pytest.mark.asyncio
async def test_a_local_only_stream_reaches_its_local_route_through_the_guard(
    incident,
):
    """The streaming leaf is told the call is local-only, so the guard admits
    the local route routing chose, though its vendor is outside the chain."""
    service = incident.service
    local = _local_streaming_route(service)

    await _drain(service.stream_with_messages(
        messages=list(MESSAGES), force_local_only=True
    ))

    assert local.calls == [("get_streaming_response_with_tools", "m")]
    assert _dispatch_decisions(service) == []


@pytest.mark.asyncio
async def test_a_local_only_audit_reaches_its_local_route_through_the_guard(
    incident,
):
    """The audit passes its local-only flag to the dispatch, as generation does."""
    service = incident.service
    local = _Route("ollama:local")
    service.providers.append(dict(local.as_provider(), is_local=True, is_cloud=False))

    await service.get_audit_response("An assistant response.", force_local_only=True)

    assert local.calls == 1
    assert _outsiders_called(incident) == {}
    assert _dispatch_decisions(service) == []


@pytest.mark.asyncio
async def test_a_principals_override_passes_the_guard_without_a_second_record(
    incident,
):
    """The entry point admitted and recorded it; the guard adds nothing."""
    from kestrel_sovereign.llm.provider_policy import caller_call_override_scope
    from kestrel_sovereign.telemetry import bind_current_turn_id

    service = incident.service
    with caller_call_override_scope(
        OVERRIDE, SOVEREIGN, via="model_override"
    ), bind_current_turn_id("turn-1"):
        for _ in range(2):
            assert await service.generate_with_messages(
                messages=list(MESSAGES), model_override=OVERRIDE
            ) == "from openrouter:api"
            assert await service.get_response(
                system_prompt="s", user_prompt="hi", model_override=OVERRIDE
            ) == "from openrouter:api"

    history = service.describe_provider_policy()["override_history"]
    assert [(d["kind"], d["source"], d["accepted"]) for d in history] == [
        ("call", "sovereign", True),
    ]


@pytest.mark.asyncio
async def test_the_dispatch_guard_refuses_a_decision_route_nothing_narrowed(
    monkeypatch, caplog
):
    """An automatic decision that skipped the chain narrowing, as an unlisted
    decision path would, is refused at dispatch: nothing is sent."""
    lineup = _decision_lineup("openrouter:api")
    service = lineup.service
    monkeypatch.setattr(
        service, "_chain_decision_routes", lambda routes, *, local_only: routes
    )

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.llm.service"):
        with pytest.raises(DecisionOverrideRefused) as refused:
            await _decide(service)

    assert refused.value.refusal.code == ProviderPolicyRefusal.OUTSIDE_APPROVED_CHAIN
    assert lineup.adapters["openrouter:api"].decide_calls == []
    [decision] = _dispatch_decisions(service)
    assert (decision["vendor"], decision["via"], decision["accepted"]) == (
        "openrouter", "dispatch:decide", False,
    )
    assert len(_refusal_warnings(caplog)) == 1
