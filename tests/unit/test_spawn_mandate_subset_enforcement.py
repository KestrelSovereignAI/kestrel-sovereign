"""F277: a SpawnMandate must only RESTRICT the child relative to the parent.

The manager must refuse a mandate that grants features the parent lacks or adds
capability-granting constraints. A mandate-bearing creation must also remain
inside ``spawn_agent`` so its final child-DID receipt is signed before publish."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

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


def test_spawn_tool_max_tokens_constraint_validates():
    """#2138 regression: the spawn tool parses `max_tokens=1000` and the mandate
    must pass _validate_mandate_subset. Before the coercion fix the value was the
    string "1000", which validate_constraints (type-checks int/float) refused —
    so every documented spawn with a max_tokens constraint failed."""
    from kestrel_sovereign.features.spawn.feature import _coerce_constraint_value

    # The tool coerces numeric constraint values to their natural type.
    assert _coerce_constraint_value("1000") == 1000
    assert isinstance(_coerce_constraint_value("1000"), int)
    assert _coerce_constraint_value("true") == "true"  # flags stay strings
    assert _coerce_constraint_value("nan") == "nan"     # non-finite kept as str

    mgr = AgentManager()
    parent = _parent({"MemoryFeature", "SpawnFeature"})
    # Exactly what SpawnFeature.spawn_agent builds from
    # constraints="max_tokens=1000,no_web" — must not raise.
    mgr._validate_mandate_subset(
        parent,
        _mandate(
            features_allowed=["MemoryFeature"],
            additional_constraints={
                "max_tokens": _coerce_constraint_value("1000"),
                "no_web": "true",
            },
        ),
    )


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
async def test_direct_create_rejects_mandate_before_inception(monkeypatch, tmp_path):
    mgr = AgentManager(base_data_dir=tmp_path)
    inception = AsyncMock()

    monkeypatch.setattr(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        inception,
    )

    mandate = _mandate(features_allowed=["WebSearchFeature"], purpose="research")
    with pytest.raises(ValueError, match="spawn_agent"):
        await mgr.create_agent(
            "child",
            parent_did="did:key:parent",
            features=["WebSearchFeature"],
            mandate=mandate,
        )

    inception.assert_not_awaited()
    assert not (tmp_path / "agent_data" / "child").exists()
    assert mgr._spawn_authority_registry.pending() == ()


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
