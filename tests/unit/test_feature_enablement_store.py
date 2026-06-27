"""Tests for the per-agent feature/MCP-server enablement delta store + the
startup reconcile-union (bootstrap config ∪ DB deltas)."""

from __future__ import annotations

import pytest

from kestrel_sovereign.a2a.stores.unified.feature_enablement_store import (
    FeatureEnablementStore,
    KIND_FEATURE,
    KIND_MCP_SERVER,
    STATE_DISABLED,
    STATE_ENABLED,
)
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


# ---------------------------------------------------------------------------
# Store CRUD
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_store_upsert_get_clear(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "enablement.db"))
    await backend.connect()
    store = FeatureEnablementStore(backend)
    await store.initialize()
    try:
        did = "did:test:agent"
        await store.set_state(agent_did=did, kind=KIND_FEATURE, name="VoiceFeature",
                              state=STATE_ENABLED, actor="agent",
                              metadata={"pre_explore": True})
        await store.set_state(agent_did=did, kind=KIND_MCP_SERVER, name="fetch",
                              state=STATE_ENABLED, actor="agent")

        feat = await store.get_deltas(did, KIND_FEATURE)
        assert len(feat) == 1
        assert feat[0]["name"] == "VoiceFeature" and feat[0]["state"] == "enabled"
        assert feat[0]["metadata"] == {"pre_explore": True}

        mcp = await store.get_deltas(did, KIND_MCP_SERVER)
        assert [d["name"] for d in mcp] == ["fetch"]

        # both kinds returned when unfiltered
        assert len(await store.get_deltas(did)) == 2

        # upsert overwrites state in place (still one row for the key)
        await store.set_state(agent_did=did, kind=KIND_FEATURE, name="VoiceFeature",
                              state=STATE_DISABLED)
        feat = await store.get_deltas(did, KIND_FEATURE)
        assert len(feat) == 1 and feat[0]["state"] == "disabled"

        # isolation by agent_did
        assert await store.get_deltas("did:other") == []

        # clear reverts (row gone)
        await store.clear(did, KIND_FEATURE, "VoiceFeature")
        assert await store.get_deltas(did, KIND_FEATURE) == []
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Reconcile-union (_effective_allowed_features)
# ---------------------------------------------------------------------------
class _UnionAgent:
    """Minimal stand-in exposing only what the union helpers need."""

    from kestrel_sovereign.kestrel_agent import KestrelAgent as _KA
    _effective_allowed_features = _KA._effective_allowed_features
    _disabled_feature_names = _KA._disabled_feature_names

    def __init__(self, allowed, deltas):
        self._allowed_features = allowed
        self._deltas = deltas

    async def get_enablement_deltas(self, kind=None):
        return self._deltas


@pytest.mark.asyncio
async def test_union_no_deltas_equals_bootstrap():
    agent = _UnionAgent({"VoiceFeature", "MemoryFeature"}, [])
    eff = await agent._effective_allowed_features()
    assert eff == {"VoiceFeature", "MemoryFeature"}


@pytest.mark.asyncio
async def test_union_enabled_adds_disabled_removes():
    agent = _UnionAgent(
        {"VoiceFeature", "MemoryFeature"},
        [
            {"name": "WebSearchFeature", "state": "enabled"},
            {"name": "VoiceFeature", "state": "disabled"},
        ],
    )
    eff = await agent._effective_allowed_features()
    assert eff == {"MemoryFeature", "WebSearchFeature"}


@pytest.mark.asyncio
async def test_union_mandatory_cannot_be_disabled():
    from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES
    mandatory = next(iter(MANDATORY_FEATURES))
    agent = _UnionAgent(
        {mandatory, "MemoryFeature"},
        [{"name": mandatory, "state": "disabled"}],
    )
    eff = await agent._effective_allowed_features()
    assert mandatory in eff  # delta cannot drop a mandatory feature


@pytest.mark.asyncio
async def test_union_none_bootstrap_passthrough():
    agent = _UnionAgent(None, [{"name": "X", "state": "enabled"}])
    assert await agent._effective_allowed_features() is None


@pytest.mark.asyncio
async def test_disabled_skip_applies_without_allowlist():
    # Bootstrap-less agent (None) must still honor a persisted disabled delta —
    # discover_features loads all, so the load loop skips these names.
    agent = _UnionAgent(None, [
        {"name": "WebSearchFeature", "state": "disabled"},
        {"name": "VoiceFeature", "state": "enabled"},
    ])
    disabled = await agent._disabled_feature_names()
    assert disabled == {"WebSearchFeature"}


@pytest.mark.asyncio
async def test_disabled_skip_never_includes_mandatory():
    from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES
    mandatory = next(iter(MANDATORY_FEATURES))
    agent = _UnionAgent(None, [{"name": mandatory, "state": "disabled"}])
    assert mandatory not in await agent._disabled_feature_names()


# ---------------------------------------------------------------------------
# End-to-end primitive: host API writes a delta -> startup union reads it back
# (proves persist_feature_enablement -> get_enablement_deltas ->
# _effective_allowed_features round-trips through a real store, i.e. a change
# survives a "restart". The production callers — feature_add/remove + MCP — wire
# into persist_feature_enablement in the follow-up PRs.)
# ---------------------------------------------------------------------------
class _StoreBackedAgent:
    from kestrel_sovereign.kestrel_agent import KestrelAgent as _KA
    persist_feature_enablement = _KA.persist_feature_enablement
    get_enablement_deltas = _KA.get_enablement_deltas
    clear_feature_enablement = _KA.clear_feature_enablement
    _effective_allowed_features = _KA._effective_allowed_features

    def __init__(self, did, store, allowed):
        self.did = did
        self._feature_enablement_store = store
        self._allowed_features = allowed


@pytest.mark.asyncio
async def test_persist_then_union_reflects_change_across_restart(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "e2e.db"))
    await backend.connect()
    store = FeatureEnablementStore(backend)
    await store.initialize()
    try:
        # "Session 1": agent enables a non-bootstrap feature, disables a bootstrap one.
        a1 = _StoreBackedAgent("did:test:e2e", store, {"VoiceFeature"})
        await a1.persist_feature_enablement("feature", "WebSearchFeature", "enabled", actor="agent")
        await a1.persist_feature_enablement("feature", "VoiceFeature", "disabled", actor="agent")

        # "Session 2": a fresh agent object over the SAME db reflects the deltas.
        a2 = _StoreBackedAgent("did:test:e2e", store, {"VoiceFeature"})
        eff = await a2._effective_allowed_features()
        assert eff == {"WebSearchFeature"}  # voice disabled, websearch enabled — persisted

        # clearing reverts to the bootstrap default
        await a2.clear_feature_enablement("feature", "VoiceFeature")
        await a2.clear_feature_enablement("feature", "WebSearchFeature")
        assert await a2._effective_allowed_features() == {"VoiceFeature"}
    finally:
        await backend.close()
