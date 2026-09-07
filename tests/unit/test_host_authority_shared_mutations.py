"""Shared host resources need the sovereign, not an agent's consent (#3221, #3223).

The Ollama daemon and the Cloud Run fleet are one per host. `pull_model`
downloaded onto the shared daemon and `cleanup_models(dry_run=False)`
deleted from it under any agent's consent, protecting only the models the
calling agent's own service named; `deploy_agent` could deploy or tear
down a multi-agent profile — the whole configured fleet — after a generic
ASK that AUTO could promote. Operational consent is not host authority.

One predicate now answers all of them, the one whole-host restart already
enforced: the turn must carry an endpoint-bound sovereign-key caller whose
credential still matches the host's stable key.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.auth import CallerContext, caller_context_scope
from kestrel_sovereign.features.deploy.feature import DeployFeature
from kestrel_sovereign.features.deploy.models import DeploymentProfile, DeployProviderType
from kestrel_sovereign.features.model.feature import ModelAgent
from kestrel_sovereign.llm.usage_tracking import UsageTrackingMixin
from kestrel_sovereign.security.host_authority import (
    HostAuthorityError,
    require_sovereign_caller,
)

STABLE_KEY = "test-stable-sovereign-key-3221"


@pytest.fixture
def stable_key(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", STABLE_KEY)
    return STABLE_KEY


def sovereign(credential=STABLE_KEY):
    return CallerContext.sovereign(identity="operator", credential=credential)


NON_SOVEREIGN = [
    pytest.param(None, id="no-caller-scheduler-wake"),
    pytest.param(CallerContext.authenticated("user@example.com"), id="oauth-user"),
    pytest.param(CallerContext.a2a_transport(), id="a2a-transport"),
    pytest.param(CallerContext.anonymous(), id="anonymous"),
    pytest.param(sovereign(credential="a-key-since-rotated"), id="sovereign-under-rotated-key"),
]


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("caller", NON_SOVEREIGN)
def test_predicate_refuses_every_non_sovereign_turn(stable_key, caller):
    with caller_context_scope(caller):
        with pytest.raises(HostAuthorityError) as excinfo:
            require_sovereign_caller("the operation")
    assert "the operation" in str(excinfo.value) or "actor" in str(excinfo.value)


def test_predicate_admits_the_sovereign_under_the_current_key(stable_key):
    with caller_context_scope(sovereign()):
        assert require_sovereign_caller("the operation") == "operator"


def test_predicate_refuses_without_a_stable_key(monkeypatch):
    monkeypatch.delenv("KESTREL_API_KEY", raising=False)
    with caller_context_scope(sovereign()):
        with pytest.raises(HostAuthorityError, match="no stable sovereign key"):
            require_sovereign_caller("the operation")


def test_a_sovereign_turn_does_not_leak_into_a_dispatched_wake(stable_key):
    """Signal dispatch binds None on purpose; the predicate sees that."""
    with caller_context_scope(sovereign()):
        with caller_context_scope(None):
            with pytest.raises(HostAuthorityError):
                require_sovereign_caller("the operation")


# ---------------------------------------------------------------------------
# The model service: one door for the tool and the silent auto-pull
# ---------------------------------------------------------------------------


class _Service(UsageTrackingMixin):
    """The mixin over the attributes its two methods read."""

    def __init__(self, *, providers, preference=None, mandate_config=None):
        self.providers = providers
        self.mandate_config = mandate_config
        self._preference = preference
        self._storage_cache = None
        self._usage_db = None
        self.get_storage_info = AsyncMock(return_value={
            "available_gb": 5.0, "total_gb": 100.0,
            "models": [
                {"id": "peer-pinned:8b", "last_used": "2020-01-01T00:00:00+00:00"},
                {"id": "mine:1b", "last_used": "2020-01-01T00:00:00+00:00"},
                {"id": "junk:7b", "last_used": "2020-01-01T00:00:00+00:00"},
            ],
        })

    def get_model_preference(self):
        return {"model": self._preference, "provider": None}

    async def _ensure_db_initialized(self):
        return None


def _ollama_provider(model="ollama-default:3b"):
    client = SimpleNamespace(pull=AsyncMock(), delete=AsyncMock())
    return {"vendor": "ollama", "model": model, "client": client}


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", NON_SOVEREIGN)
async def test_pull_refuses_before_touching_the_daemon(stable_key, caller):
    provider = _ollama_provider()
    service = _Service(providers=[provider])
    with caller_context_scope(caller):
        with pytest.raises(HostAuthorityError, match="shared local model installation"):
            await service.pull_model("llama3:8b")
    provider["client"].pull.assert_not_awaited()
    service.get_storage_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_pull_proceeds_for_the_sovereign(stable_key):
    provider = _ollama_provider()
    service = _Service(providers=[provider])
    service.get_storage_info.return_value = {"available_gb": 500.0, "total_gb": 1000.0, "models": []}
    with caller_context_scope(sovereign()):
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache") as cache:
            cache.return_value = MagicMock()
            assert await service.pull_model("llama3:8b") is True
    provider["client"].pull.assert_awaited_once_with(model="llama3:8b")


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", NON_SOVEREIGN)
async def test_real_deletion_refuses_before_inspecting_storage(stable_key, caller):
    provider = _ollama_provider()
    service = _Service(providers=[provider])
    with caller_context_scope(caller):
        with pytest.raises(HostAuthorityError, match="shared local model deletion"):
            await service.cleanup_unused_models(dry_run=False)
    provider["client"].delete.assert_not_awaited()
    service.get_storage_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_dry_run_is_a_report_and_needs_no_authority(stable_key):
    provider = _ollama_provider()
    service = _Service(providers=[provider], preference="mine:1b")
    with caller_context_scope(None):
        plan = await service.cleanup_unused_models(dry_run=True, protected_models={"peer-pinned:8b"})
    # The peer's pinned model and this agent's own preference are kept;
    # only the model nobody names is planned for deletion.
    assert plan == ["junk:7b"]
    provider["client"].delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_deletion_keeps_every_co_hosted_agents_models(stable_key):
    provider = _ollama_provider()
    service = _Service(providers=[provider], preference="mine:1b")
    with caller_context_scope(sovereign()):
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache") as cache:
            cache.return_value = MagicMock()
            deleted = await service.cleanup_unused_models(
                dry_run=False, protected_models={"peer-pinned:8b"}
            )
    assert deleted == ["junk:7b"]
    provider["client"].delete.assert_awaited_once_with(model="junk:7b")


def test_locally_protected_models_include_the_active_preference():
    service = _Service(
        providers=[_ollama_provider("ollama-default:3b")],
        preference="pinned:70b",
        mandate_config={"defaults": {"preferred": "pref:1b"}, "mandates": {"code": "coder:7b", "chat": "gpt-5"}},
    )
    assert service.locally_protected_models() == {
        "ollama-default:3b", "pref:1b", "coder:7b", "pinned:70b"
    }


@pytest.mark.asyncio
async def test_auto_pull_in_model_selection_is_the_same_door(stable_key):
    """`get_response_with_model` pulls a missing model on its own; without
    a sovereign turn that is refused and surfaced, not installed."""
    from kestrel_sovereign.llm.service import LLMService

    registry = MagicMock()
    registry.initialize_providers = Mock(return_value=[])
    registry.get_providers_with_pattern = Mock(return_value=[])
    registry.get_local_providers = Mock(return_value=[])
    with patch("kestrel_sovereign.llm.service.load_config", return_value={}), patch(
        "kestrel_sovereign.llm.service.ProviderRegistry", return_value=registry
    ):
        service = LLMService()
    service.providers = [_ollama_provider()]
    service._check_policy = lambda: None

    with caller_context_scope(CallerContext.authenticated("user@example.com")):
        with pytest.raises(ValueError, match="auto-pull failed.*sovereign"):
            await service.get_response_with_model("missing:7b", "sys", "hi")
    service.providers[0]["client"].pull.assert_not_awaited()


# ---------------------------------------------------------------------------
# The model tool: fleet-wide protection and an honest refusal
# ---------------------------------------------------------------------------


def _peer(protected):
    return SimpleNamespace(llm_service=SimpleNamespace(locally_protected_models=lambda: set(protected)))


async def _model_feature(peers=None):
    llm_service = MagicMock()
    llm_service.locally_protected_models = lambda: {"mine:1b"}
    llm_service.cleanup_unused_models = AsyncMock(return_value={"would_delete": []})
    llm_service.pull_model = AsyncMock(side_effect=HostAuthorityError("shared local model installation requires an authenticated sovereign-key caller"))
    manager = SimpleNamespace(list_agents=lambda: dict(peers or {}))
    agent = SimpleNamespace(llm_service=llm_service, features={}, _agent_manager=manager)
    feature = ModelAgent(agent)
    await feature.initialize()
    return feature, llm_service


@pytest.mark.asyncio
async def test_cleanup_tool_protects_every_co_hosted_agents_models():
    feature, llm_service = await _model_feature(
        peers={"Emma": _peer({"emma:70b"}), "Nellie": _peer({"nellie:8b"})}
    )
    await feature.cleanup_models(dry_run=True)
    _, kwargs = llm_service.cleanup_unused_models.call_args
    assert set(kwargs["protected_models"]) == {"mine:1b", "emma:70b", "nellie:8b"}


@pytest.mark.asyncio
async def test_cleanup_tool_reports_the_refusal_as_authority():
    feature, llm_service = await _model_feature()
    llm_service.cleanup_unused_models = AsyncMock(
        side_effect=HostAuthorityError("shared local model deletion requires an authenticated sovereign-key caller")
    )
    result = await feature.cleanup_models(dry_run=False)
    assert result.status is ToolResultStatus.ERROR
    assert "sovereign" in result.error
    assert result.data["authority"] == "sovereign"


@pytest.mark.asyncio
async def test_pull_tool_reports_the_refusal_as_authority():
    feature, _ = await _model_feature()
    result = await feature.pull_model("llama3:8b")
    assert result.status is ToolResultStatus.ERROR
    assert "sovereign" in result.error
    assert result.data == {"model_name": "llama3:8b", "pulled": False, "authority": "sovereign"}


# ---------------------------------------------------------------------------
# Fleet deployment
# ---------------------------------------------------------------------------


def _profile(mode):
    return DeploymentProfile(
        provider=DeployProviderType.CLOUD_RUN,
        service_name=f"kestrel-{mode}",
        region="us-central1",
        deployment_mode=mode,
    )


def _deploy_feature():
    feature = DeployFeature(SimpleNamespace(features={}))
    feature.disabled = False
    feature.manager = SimpleNamespace(
        profiles={"fleet": _profile("multi_agent"), "solo": _profile("agent")},
        deploy_profile=AsyncMock(return_value={"success": True, "message": "deployed"}),
        teardown_profile=AsyncMock(return_value={"success": True, "message": "torn down"}),
    )
    return feature


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", NON_SOVEREIGN)
@pytest.mark.parametrize("action", ["deploy", "start", "teardown", "stop", "delete"])
async def test_fleet_profile_mutation_refuses_every_non_sovereign_turn(stable_key, caller, action):
    feature = _deploy_feature()
    with caller_context_scope(caller):
        result = await feature.deploy_agent(action=action, profile="fleet", tag="v1.0.0")
    assert result.status is ToolResultStatus.ERROR, result
    # Two refusal shapes: no sovereign caller, or a sovereign caller whose
    # credential no longer matches the host key.
    assert "sovereign-key caller" in result.error or "no longer matches" in result.error
    assert result.data["authority"] == "sovereign"
    assert result.data["profile"] == "fleet"
    feature.manager.deploy_profile.assert_not_awaited()
    feature.manager.teardown_profile.assert_not_awaited()


@pytest.mark.asyncio
async def test_fleet_profile_mutation_proceeds_for_the_sovereign(stable_key):
    feature = _deploy_feature()
    with caller_context_scope(sovereign()):
        deploy = await feature.deploy_agent(action="deploy", profile="fleet", tag="v1.0.0")
        teardown = await feature.deploy_agent(action="teardown", profile="fleet")
    assert deploy.status is ToolResultStatus.OK
    assert teardown.status is ToolResultStatus.OK
    feature.manager.deploy_profile.assert_awaited_once_with("fleet", "v1.0.0")
    feature.manager.teardown_profile.assert_awaited_once_with("fleet")


@pytest.mark.asyncio
async def test_single_agent_profile_stays_under_ordinary_consent(stable_key):
    """The agent deploying itself is its own change; no host authority."""
    feature = _deploy_feature()
    with caller_context_scope(None):
        result = await feature.deploy_agent(action="deploy", profile="solo", tag="v1.0.0")
    assert result.status is ToolResultStatus.OK
    feature.manager.deploy_profile.assert_awaited_once_with("solo", "v1.0.0")


@pytest.mark.asyncio
async def test_reads_and_missing_profile_are_not_gated(stable_key):
    feature = _deploy_feature()
    feature.manager.list_sessions = AsyncMock(return_value={})
    with caller_context_scope(None):
        status = await feature.deploy_agent(action="status")
        missing = await feature.deploy_agent(action="deploy", profile="")
    assert status.status is ToolResultStatus.OK
    assert missing.status is ToolResultStatus.ERROR
    assert missing.error == "Profile required"
