"""F277: a SpawnMandate must only RESTRICT the child relative to the parent.

The manager must (a) refuse a mandate that grants features the parent lacks or
adds capability-granting constraints, and (b) actually forward the mandate to
inception so the delegation edge records it (previously dropped)."""

from types import SimpleNamespace

import pytest

from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.spawn.mandate import SpawnMandate


def _parent(features):
    # Parent agent stand-in: only `.features` (name -> feature) is consulted by
    # the subset check.
    return SimpleNamespace(
        agent_id="did:key:parent",
        features={name: object() for name in features},
    )


def _mandate(**kw):
    kw.setdefault("parent_did", "did:key:parent")
    return SpawnMandate(**kw)


def test_subset_ok_when_features_are_a_subset():
    mgr = AgentManager()
    parent = _parent({"WebSearchFeature", "MemoryFeature", "SpawnFeature"})
    # subset + a real restriction → valid
    mgr._validate_mandate_subset(
        parent,
        _mandate(features_allowed=["WebSearchFeature"],
                 additional_constraints={"restricted_tools": ["run_script"]}),
    )  # must not raise


def test_refuses_features_not_available_to_parent():
    mgr = AgentManager()
    parent = _parent({"WebSearchFeature"})
    with pytest.raises(ValueError, match="Spawn refused"):
        mgr._validate_mandate_subset(
            parent, _mandate(features_allowed=["WalletFeature"])
        )


def test_refuses_capability_granting_constraint():
    mgr = AgentManager()
    parent = _parent({"WebSearchFeature"})
    with pytest.raises(ValueError, match="grant"):
        mgr._validate_mandate_subset(
            parent,
            _mandate(features_allowed=["WebSearchFeature"],
                     additional_constraints={"grant_features": ["WalletFeature"]}),
        )


@pytest.mark.asyncio
async def test_create_agent_forwards_mandate_to_inception(monkeypatch, tmp_path):
    mgr = AgentManager(base_data_dir=tmp_path)
    captured = {}

    async def fake_inception(**kwargs):
        captured.update(kwargs)

    async def fake_load(name, config):
        return SimpleNamespace(agent_id="did:key:child", features={})

    monkeypatch.setattr(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        fake_inception,
    )
    monkeypatch.setattr(mgr, "load_agent", fake_load)

    mandate = _mandate(features_allowed=["WebSearchFeature"], purpose="research")
    await mgr.create_agent("child", parent_did="did:key:parent",
                           features=["WebSearchFeature"], mandate=mandate)

    # Previously the mandate never reached inception (F277).
    assert captured.get("spawn_mandate") is mandate
    assert captured.get("parent_did") == "did:key:parent"


@pytest.mark.asyncio
async def test_omitted_allowlist_inherits_parent_ceiling_not_all(monkeypatch, tmp_path):
    """codex P1: a restricted parent that omits features_allowed must NOT get a
    child loaded with all features — the child inherits the parent's ceiling."""
    mgr = AgentManager(base_data_dir=tmp_path)
    parent = SimpleNamespace(
        agent_id="did:key:parent",
        features={"WebSearchFeature": object(), "MemoryFeature": object()},
        _private_key=None,
        identity=None,
    )
    captured = {}

    async def fake_create_agent(name, parent_did=None, features=None, mandate=None):
        captured["features"] = features
        return SimpleNamespace(agent_id="did:key:child", features={})

    monkeypatch.setattr(mgr, "create_agent", fake_create_agent)

    # Mandate omits features_allowed (default empty list).
    await mgr._do_spawn("child", parent, _mandate())

    # Must be the parent's ceiling, NOT None (= load all).
    assert captured["features"] == ["MemoryFeature", "WebSearchFeature"]
