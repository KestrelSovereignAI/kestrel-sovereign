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


async def _model_feature(peers=None, *, manager=True, configured=None):
    """A ModelAgent named "Me" with loaded peers and a configured roster.

    ``configured`` stands in for the host's multi_agent.toml; the real
    reader is tested on its own below. ``manager=False`` is the
    per-agent-process topology (``kestrel start <name>``): no manager at all.
    """
    llm_service = MagicMock()
    llm_service.locally_protected_models = lambda: {"mine:1b"}
    llm_service.cleanup_unused_models = AsyncMock(return_value={"would_delete": []})
    llm_service.pull_model = AsyncMock(side_effect=HostAuthorityError("shared local model installation requires an authenticated sovereign-key caller"))
    agent = SimpleNamespace(llm_service=llm_service, features={}, agent_name="Me")
    if manager:
        agent._agent_manager = SimpleNamespace(list_agents=lambda: dict(peers or {}))
    feature = ModelAgent(agent)
    await feature.initialize()
    roster = ["Me", *(peers or {})] if configured is None else configured
    feature._configured_agent_names = lambda: list(roster)
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


# ---------------------------------------------------------------------------
# A partial roster refuses; the plan says who was asked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_reports_who_it_consulted_to_the_sovereign(stable_key):
    feature, llm_service = await _model_feature(peers={"Emma": _peer({"emma:70b"})})
    with caller_context_scope(sovereign()):
        plan = await feature.cleanup_models(dry_run=True)
    assert plan.data["consulted_agents"] == ["Emma", "Me"]
    assert plan.data["unconsulted_agents"] == []
    assert "could not account" not in plan.error


@pytest.mark.asyncio
async def test_a_non_sovereign_dry_run_learns_no_agent_names(stable_key):
    """The roster is host information. A dry run needs no authority, so a
    non-sovereign caller gets the plan and a count-free caveat — never the
    names, on either the data or the error side."""
    feature, llm_service = await _model_feature(
        peers={"Emma": _peer({"emma:70b"})}, configured=["Emma", "Me", "Cold-Tenant", "Secret-Peer"]
    )
    with caller_context_scope(CallerContext.authenticated("u")):
        plan = await feature.cleanup_models(dry_run=True)
    assert plan.status is ToolResultStatus.PARTIAL
    assert "consulted_agents" not in plan.data and "unconsulted_agents" not in plan.data
    assert "could not account for every configured agent" in plan.error
    for name in ("Cold-Tenant", "Secret-Peer", "Emma"):
        assert name not in plan.error and name not in str(plan.data)


@pytest.mark.asyncio
async def test_a_cold_configured_agent_blocks_a_real_deletion(stable_key):
    """Nellie is configured (autostart = false) and not loaded: her pinned
    model is exactly what this process cannot see. The plan says so; the
    deletion refuses rather than deleting under a partial set."""
    feature, llm_service = await _model_feature(
        peers={"Emma": _peer({"emma:70b"})}, configured=["Emma", "Me", "Nellie"]
    )
    with caller_context_scope(sovereign()):
        plan = await feature.cleanup_models(dry_run=True)
        deletion = await feature.cleanup_models(dry_run=False)
    assert plan.status is ToolResultStatus.PARTIAL
    assert plan.data["unconsulted_agents"] == ["Nellie"]
    assert "Nellie" in plan.error
    assert deletion.status is ToolResultStatus.ERROR, deletion
    assert "Nellie" in deletion.error and "cannot account" in deletion.error
    assert deletion.data["unconsulted_agents"] == ["Nellie"]
    # Only the dry run reached the service; the deletion never did.
    assert llm_service.cleanup_unused_models.await_count == 1
    assert llm_service.cleanup_unused_models.await_args.kwargs["dry_run"] is True


@pytest.mark.asyncio
async def test_per_agent_process_cannot_delete_when_the_host_has_peers(stable_key):
    """`kestrel start <name>` runs the agent alone in its own process with no
    manager; the shared daemon still serves every other configured agent."""
    feature, llm_service = await _model_feature(manager=False, configured=["Emma", "Me"])
    with caller_context_scope(sovereign()):
        deletion = await feature.cleanup_models(dry_run=False)
    assert deletion.status is ToolResultStatus.ERROR
    assert "Emma" in deletion.error
    llm_service.cleanup_unused_models.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_non_sovereign_caller_learns_nothing_about_the_roster():
    """Authority is asked before the roster is disclosed."""
    feature, llm_service = await _model_feature(configured=["Me", "Secret-Peer"])
    with caller_context_scope(CallerContext.authenticated("u")):
        result = await feature.cleanup_models(dry_run=False)
    assert result.status is ToolResultStatus.ERROR
    assert "Secret-Peer" not in result.error
    assert result.data == {"dry_run": False, "authority": "sovereign"}
    llm_service.cleanup_unused_models.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_roster_is_an_honest_refusal(stable_key):
    """A bad multi_agent.toml must not delete under a roster that could not
    be read, and must not escape the tool as a bare exception either."""
    feature, llm_service = await _model_feature()

    def broken():
        raise ValueError("Invalid TOML in multi_agent.toml")

    feature._configured_agent_names = broken
    with caller_context_scope(sovereign()):
        result = await feature.cleanup_models(dry_run=False)
    assert result.status is ToolResultStatus.ERROR
    assert "Invalid TOML" in result.error
    llm_service.cleanup_unused_models.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_an_unreadable_rosters_detail_reaches_only_the_sovereign(stable_key, dry_run):
    """The loader's own message names an agent or the host path (round 3,
    P2). A non-sovereign caller learns that the read failed, nothing more."""
    feature, llm_service = await _model_feature()

    def broken():
        raise ValueError("Agent 'Secret-Peer' must have either 'url' (remote) or 'data_dir' + 'port' (local)")

    feature._configured_agent_names = broken
    with caller_context_scope(CallerContext.authenticated("u")):
        result = await feature.cleanup_models(dry_run=dry_run)
    assert result.status is ToolResultStatus.ERROR
    assert "Secret-Peer" not in result.error and "Secret-Peer" not in str(result.data)
    assert "roster could not be read" in result.error
    llm_service.cleanup_unused_models.assert_not_awaited()

    with caller_context_scope(sovereign()):
        result = await feature.cleanup_models(dry_run=dry_run)
    assert "Secret-Peer" in result.error


def test_configured_agent_names_reads_the_hosts_roster(tmp_path, monkeypatch):
    from kestrel_sovereign.features.model.feature import _configured_agent_names

    monkeypatch.setattr("kestrel_sovereign.paths.project_dir", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    assert _configured_agent_names() == []
    # No file, but agent directories: `kestrel start` auto-discovers and
    # launches them, so the roster must name them too (review round 2, P1).
    for name in ("claw", "emma"):
        (tmp_path / "agent_data" / name).mkdir(parents=True)
        (tmp_path / "agent_data" / name / "kestrel_prime.db").write_bytes(b"")
    assert _configured_agent_names() == ["claw", "emma"]
    (tmp_path / "multi_agent.toml").write_text(
        "[agents.Emma]\ndata_dir = \"agent_data/emma\"\nport = 8801\n\n"
        "[agents.Nellie]\ndata_dir = \"agent_data/nellie\"\nport = 8802\nautostart = false\n"
    )
    assert _configured_agent_names() == ["Emma", "Nellie"]


# ---------------------------------------------------------------------------
# The restart authority's messages are byte-identical after the refactor
# ---------------------------------------------------------------------------


def test_restart_authority_messages_are_unchanged(stable_key, monkeypatch):
    from kestrel_sovereign.features.restart_coordinator.authority import (
        RestartAuthorityError,
        _sovereign_secret,
        require_restart_request_authority,
    )

    with caller_context_scope(None):
        with pytest.raises(RestartAuthorityError) as excinfo:
            require_restart_request_authority()
    assert str(excinfo.value) == "whole-host restart requires an authenticated sovereign-key caller"

    with caller_context_scope(sovereign(credential="a-key-since-rotated")):
        with pytest.raises(RestartAuthorityError) as excinfo:
            require_restart_request_authority()
    assert str(excinfo.value) == (
        "whole-host restart authority no longer matches the authenticated "
        "credential at request entry"
    )

    with caller_context_scope(sovereign()):
        assert require_restart_request_authority() == "operator"
    assert _sovereign_secret() == STABLE_KEY.encode()

    monkeypatch.delenv("KESTREL_API_KEY")
    with pytest.raises(RestartAuthorityError) as excinfo:
        _sovereign_secret()
    assert str(excinfo.value) == "whole-host restart authority is unavailable: no stable sovereign key"
