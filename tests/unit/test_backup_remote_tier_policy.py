from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from kestrel_sovereign.storage.sync import service as sync_service
from kestrel_sovereign.storage.sync.service import (
    RemoteTierPolicyContext,
    SyncService,
)
from kestrel_sovereign.storage.sync.targets import SyncResult, SyncTarget, TrustTier


class ConstructorSpyTarget(SyncTarget):
    def __init__(self, name: str = "remote://target"):
        self._name = name
        self.snapshot_calls = 0

    @property
    def name(self):
        return self._name

    @property
    def trust_tier(self):
        return TrustTier.EXPEDIENT

    async def sync_snapshot(self, db_path):
        self.snapshot_calls += 1
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=1,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
        )

    async def sync_wal(self, wal_path, position):
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=0,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_latest_position(self):
        return None


class LocalTrustTier:
    name = "LOCAL"
    value = 0


class LocalSpyTarget(SyncTarget):
    def __init__(self, name: str = "local://target"):
        self._name = name
        self.snapshot_calls = 0

    @property
    def name(self):
        return self._name

    @property
    def trust_tier(self):
        return LocalTrustTier

    async def sync_snapshot(self, db_path):
        self.snapshot_calls += 1
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=1,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
        )

    async def sync_wal(self, wal_path, position):
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=0,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_latest_position(self):
        return None


def _service(context):
    return SyncService(
        db_path=context.db_path or "/var/lib/kestrel/kestrel_prime.db",
        policy_context=context,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context", "reason"),
    [
        (
            RemoteTierPolicyContext(
                identity="did:pkh:eip155:1:0xabc",
                db_path="/var/lib/kestrel/kestrel_prime.db",
                is_test_instance=True,
                has_constitution_anchor=True,
                privacy_mode="normal",
            ),
            "test_instance",
        ),
        (
            RemoteTierPolicyContext(
                identity="did:test:agent",
                db_path="/var/lib/kestrel/kestrel_prime.db",
                has_constitution_anchor=True,
                privacy_mode="normal",
            ),
            "test_did",
        ),
        (
            RemoteTierPolicyContext(
                identity="did:pkh:eip155:1:0xabc",
                db_path="__tmp__/test.db",
                has_constitution_anchor=True,
                privacy_mode="normal",
            ),
            "fixture_or_temp_db_path",
        ),
        (
            RemoteTierPolicyContext(
                identity="did:pkh:eip155:1:0xabc",
                db_path="/var/lib/kestrel/test_storage_agent.db",
                has_constitution_anchor=True,
                privacy_mode="normal",
            ),
            "fixture_or_temp_db_path",
        ),
        (
            RemoteTierPolicyContext(
                identity="did:pkh:eip155:1:0xabc",
                db_path="/var/lib/kestrel/gcs-live-test/kestrel_prime.db",
                has_constitution_anchor=True,
                privacy_mode="normal",
            ),
            "fixture_or_temp_db_path",
        ),
        (
            RemoteTierPolicyContext(
                identity="did:pkh:eip155:1:no_constitution_agent",
                db_path="/var/lib/kestrel/kestrel_prime.db",
                has_constitution_anchor=True,
                privacy_mode="normal",
            ),
            "denied_fixture_identity",
        ),
        (
            RemoteTierPolicyContext(
                identity="did:pkh:eip155:1:0xabc",
                db_path="/var/lib/kestrel/kestrel_prime.db",
                has_constitution_anchor=False,
                privacy_mode="normal",
            ),
            "missing_constitution_anchor",
        ),
        (
            RemoteTierPolicyContext(
                identity="did:pkh:eip155:1:malicious_agent",
                db_path="/var/lib/kestrel/kestrel_prime.db",
                has_constitution_anchor=True,
                privacy_mode="normal",
            ),
            "denied_fixture_identity",
        ),
        (
            RemoteTierPolicyContext(
                identity="did:pkh:eip155:1:0xabc",
                db_path="/var/lib/kestrel/kestrel_prime.db",
                has_constitution_anchor=True,
                is_sovereign_identity=False,
                privacy_mode="normal",
            ),
            "non_sovereign_identity",
        ),
        (
            RemoteTierPolicyContext(
                identity="did:pkh:eip155:1:0xabc",
                db_path="/var/lib/kestrel/kestrel_prime.db",
                has_constitution_anchor=True,
                privacy_mode="isolated",
            ),
            "privacy_mode_local_only",
        ),
    ],
)
async def test_denied_remote_identities_skip_before_target_constructor(
    context, reason, tmp_path
):
    if context.db_path and context.db_path.startswith("__tmp__"):
        context = RemoteTierPolicyContext(
            identity=context.identity,
            db_path=str(tmp_path / "test.db"),
            has_constitution_anchor=True,
            privacy_mode="normal",
        )
    sync = _service(context)
    constructor = Mock(side_effect=lambda: ConstructorSpyTarget())

    added = sync.add_remote_target(
        "remote://target",
        TrustTier.EXPEDIENT,
        constructor,
    )
    results = await sync.force_snapshot()

    assert added is False
    constructor.assert_not_called()
    assert results["remote://target"].success is True
    assert results["remote://target"].metadata == {
        "skipped": True,
        "policy_denied": True,
        "reason": reason,
    }


def test_policy_decision_runs_before_constructor(monkeypatch):
    events = []
    real_policy = sync_service._remote_tiers_allowed

    def policy_spy(context):
        events.append("policy")
        return real_policy(context)

    monkeypatch.setattr(sync_service, "_remote_tiers_allowed", policy_spy)
    context = RemoteTierPolicyContext(
        identity="did:pkh:eip155:1:0xabc",
        db_path="/var/lib/kestrel/kestrel_prime.db",
        has_constitution_anchor=True,
        privacy_mode="normal",
    )
    sync = _service(context)

    def constructor():
        events.append("constructor")
        return ConstructorSpyTarget()

    assert sync.add_remote_target(
        "remote://target",
        TrustTier.EXPEDIENT,
        constructor,
    )
    assert events == ["policy", "constructor"]


def test_normal_sovereign_agent_allows_remote_target_constructor():
    context = RemoteTierPolicyContext(
        identity="did:pkh:eip155:1:0xabc",
        db_path="/var/lib/kestrel/kestrel_prime.db",
        has_constitution_anchor=True,
        privacy_mode="normal",
    )
    sync = _service(context)
    constructor = Mock(side_effect=lambda: ConstructorSpyTarget())

    added = sync.add_remote_target(
        "remote://target",
        TrustTier.EXPEDIENT,
        constructor,
    )

    assert added is True
    constructor.assert_called_once_with()
    assert sync.targets == ["remote://target"]


@pytest.mark.asyncio
async def test_live_privacy_mode_denies_remote_snapshot_after_target_construction():
    live_privacy = {"mode": "normal"}

    def context_provider():
        return RemoteTierPolicyContext(
            identity="did:pkh:eip155:1:0xabc",
            db_path="/var/lib/kestrel/kestrel_prime.db",
            has_constitution_anchor=True,
            privacy_mode=live_privacy["mode"],
        )

    sync = SyncService(
        db_path="/var/lib/kestrel/kestrel_prime.db",
        policy_context_provider=context_provider,
    )
    remote_target = ConstructorSpyTarget()
    local_target = LocalSpyTarget()

    added = sync.add_remote_target(
        remote_target.name,
        TrustTier.EXPEDIENT,
        lambda: remote_target,
    )
    sync.add_target(local_target)
    live_privacy["mode"] = "isolated"

    results = await sync.force_snapshot()

    assert added is True
    assert remote_target.snapshot_calls == 0
    assert results[remote_target.name].success is True
    assert results[remote_target.name].metadata == {
        "skipped": True,
        "policy_denied": True,
        "reason": "privacy_mode_local_only",
    }
    assert local_target.snapshot_calls == 1
    assert results[local_target.name].success is True
