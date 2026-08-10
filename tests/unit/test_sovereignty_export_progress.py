import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.storage.providers.base import StorageResult, StorageTier
from kestrel_sovereign.features.sovereignty.feature import (
    REFUSAL_ENCRYPTION_KEY_UNAVAILABLE,
    SovereigntyFeature,
)


class _FakeStorage:
    async def create_backup_blob(self, include_db=True):
        return b"x" * 1024

    async def record_backup_artifact(self, agent_id, result):
        return "backup-node-1"

    async def add_node(self, node):
        self.receipt = node


class _FakeAgent:
    agent_id = "agent-progress"

    def __init__(self):
        self.storage = _FakeStorage()
        self.features = {}
        self.events = []

    async def emit_event(self, event_type, data):
        self.events.append((event_type, data))


class _BackupCallRecorder:
    """Underlying store proving an export did not reach blob creation."""

    def __init__(self):
        self.backup_calls = 0

    async def create_backup_blob(self, include_db=True):
        self.backup_calls += 1
        raise AssertionError("encryption preflight must run before blob creation")


@pytest.mark.asyncio
async def test_export_sovereignty_reports_upload_progress():
    agent = _FakeAgent()
    feature = SovereigntyFeature(agent)
    progress = []

    def fake_store_content(*args, **kwargs):
        on_progress = kwargs["on_progress"]
        on_progress(256, 1024)
        on_progress(768, 1024)
        return StorageResult(
            content_hash="abc123",
            cid="QmProgress",
            tier=StorageTier.IPFS,
            provider="ipfs",
            encrypted=False,
            size_bytes=1024,
        )

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter.store_content",
        side_effect=fake_store_content,
    ):
        result = await feature.export_sovereignty(
            storage_tier="local",
            encrypt=False,
            on_progress=lambda sent, total: progress.append((sent, total)),
        )

    for _ in range(5):
        await asyncio.sleep(0)

    assert result.data["cid"] == "QmProgress"
    assert (0, 1024) in progress
    assert (256, 1024) in progress
    assert (768, 1024) in progress
    assert progress[-1] == (1024, 1024)

    progress_events = [
        data
        for event_type, data in agent.events
        if event_type == "sovereignty_export_progress"
    ]
    assert progress_events
    assert progress_events[-1]["percent"] == 100


from kestrel_sdk.tools.result import ToolResultStatus


@pytest.mark.asyncio
async def test_export_sovereignty_unknown_tier_is_failed():
    """An unknown/typo'd storage_tier must FAIL loudly, not silently
    fall through to IPFS (#1946). Pre-fix, ``tier_map.get(..., IPFS)``
    meant a bogus tier attempted a network publish without the agent
    knowing, while its identity twin defaulted silently to local-only."""
    agent = _FakeAgent()
    feature = SovereigntyFeature(agent)

    result = await feature.export_sovereignty(storage_tier="ipsf", encrypt=False)

    assert result.status is ToolResultStatus.ERROR
    assert "local" in result.error and "ipfs" in result.error and "filecoin" in result.error
    assert "ipsf" in result.error
    # Validation fired BEFORE any backup blob / network attempt.
    assert not hasattr(agent.storage, "receipt")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_tier", [False, 0, [], {}])
async def test_export_sovereignty_falsy_nonstring_tier_is_failed(bad_tier):
    """A falsy NON-STRING tier (false/0/[]/{}) is a wrong type, not an
    omission — it must be rejected, not coerced to the 'ipfs' default."""
    agent = _FakeAgent()
    feature = SovereigntyFeature(agent)
    result = await feature.export_sovereignty(storage_tier=bad_tier, encrypt=False)
    assert result.status is ToolResultStatus.ERROR
    assert "local" in result.error and "ipfs" in result.error
    assert not hasattr(agent.storage, "receipt")


@pytest.mark.asyncio
async def test_export_sovereignty_valid_local_tier_resolves():
    """A valid tier ('local') passes validation and completes."""
    agent = _FakeAgent()
    feature = SovereigntyFeature(agent)

    def fake_store_content(*args, **kwargs):
        return StorageResult(
            content_hash="hashlocal",
            cid=None,
            tier=StorageTier.LOCAL_ONLY,
            provider="local",
            encrypted=False,
            size_bytes=1024,
        )

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter.store_content",
        side_effect=fake_store_content,
    ):
        result = await feature.export_sovereignty(storage_tier="local", encrypt=False)

    assert result.status is ToolResultStatus.OK
    assert result.data["tier"] == "local_only"
    assert result.data["tier_requested"] == "local_only"


@pytest.mark.asyncio
@pytest.mark.parametrize("cloud_tier", ["cloud_hot", "cloud_cold"])
async def test_export_sovereignty_cloud_tier_is_rejected(cloud_tier):
    """cloud_hot/cloud_cold have StorageTier enum members but no
    FilecoinAdapter export path, so they must be REJECTED (not charged +
    receipted for an unstored blob, and not silently mapped to IPFS as the
    old ``.get(..., IPFS)`` did) (#1946). The endpoint allowlist excludes
    them in lockstep — see test_endpoint_allowlist_matches_feature."""
    agent = _FakeAgent()
    feature = SovereigntyFeature(agent)
    result = await feature.export_sovereignty(storage_tier=cloud_tier, encrypt=False)
    assert result.status is ToolResultStatus.ERROR
    assert cloud_tier in result.error
    # No store / receipt / wallet charge happened.
    assert not hasattr(agent.storage, "receipt")


@pytest.mark.asyncio
async def test_encrypted_local_export_is_honoured_not_downgraded():
    """``encrypt=True`` on the local tier must reach the adapter as True.

    The old code coerced it to False and wrote a plaintext copy of the whole
    database to disk (#2872). ``FilecoinAdapter`` encrypts before it branches
    on tier and re-derives the portable per-content key on retrieval, so the
    local tier has nothing special about it.
    """
    agent = _FakeAgent()
    feature = SovereigntyFeature(agent)
    captured = {}

    def fake_store_content(*args, **kwargs):
        captured["encrypt"] = kwargs.get("encrypt")
        return StorageResult(
            content_hash="hashlocal",
            cid=None,
            tier=StorageTier.LOCAL_ONLY,
            provider="local",
            encrypted=True,
            encryption_key_hash="hashlocal",
            size_bytes=1024,
        )

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter.store_content",
        side_effect=fake_store_content,
    ):
        result = await feature.export_sovereignty(storage_tier="local", encrypt=True)

    assert result.status is ToolResultStatus.OK
    assert captured["encrypt"] is True
    assert result.data["encrypted"] is True
    assert result.data["tier"] == "local_only"
    assert agent.storage.receipt.properties["encrypted"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", ["local", "ipfs"])
async def test_encrypted_export_without_master_key_is_refused_before_backup(tier):
    """No master key means ``encrypt=True`` cannot be honoured on ANY tier.

    The refusal fires before blob creation, adapter construction, and any
    wallet debit, and carries a typed refusal code so the HTTP layer can
    answer 4xx instead of a catch-all 500 (#2918).
    """
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    underlying = _BackupCallRecorder()
    agent = _FakeAgent()
    agent.storage = PrivacyEnforcingStorage(underlying, PrivacyMode.NORMAL)

    with (
        patch(
            "kestrel_sovereign.security.encryption.get_master_key_bytes",
            return_value=None,
        ),
        patch(
            "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter"
        ) as adapter,
    ):
        result = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier=tier,
            encrypt=True,
        )

    assert result.status is ToolResultStatus.ERROR
    assert result.data["refusal"] == REFUSAL_ENCRYPTION_KEY_UNAVAILABLE
    assert "encrypt=True requires KESTREL_DATA_KEY" in result.error
    assert "no sovereignty backup was created" in result.error
    assert underlying.backup_calls == 0
    adapter.assert_not_called()


@pytest.mark.asyncio
async def test_unencrypted_export_needs_no_master_key():
    """``encrypt=False`` is a deliberate plaintext backup — never refused."""
    agent = _FakeAgent()
    feature = SovereigntyFeature(agent)

    def fake_store_content(*args, **kwargs):
        return StorageResult(
            content_hash="hashlocal",
            cid=None,
            tier=StorageTier.LOCAL_ONLY,
            provider="local",
            encrypted=False,
            size_bytes=1024,
        )

    with (
        patch(
            "kestrel_sovereign.security.encryption.get_master_key_bytes",
            return_value=None,
        ),
        patch(
            "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter.store_content",
            side_effect=fake_store_content,
        ),
    ):
        result = await feature.export_sovereignty(storage_tier="local", encrypt=False)

    assert result.status is ToolResultStatus.OK
    assert result.data["encrypted"] is False


def test_endpoint_allowlist_matches_feature_tier_map():
    """The endpoint ALLOWED_TIERS must be a subset of the feature's
    supported tiers, or the endpoint accepts tiers the feature 500s on."""
    from kestrel_sovereign.endpoints.sovereignty import ALLOWED_TIERS

    supported = {"local", "ipfs", "filecoin"}
    assert ALLOWED_TIERS == supported


@pytest.mark.asyncio
async def test_export_sovereignty_omitted_tier_keeps_ipfs_default():
    """Omitting storage_tier keeps the documented 'ipfs' default."""
    agent = _FakeAgent()
    # The 'ipfs' default is a paid tier, so a wallet is required to reach
    # the store path; without it the export refuses before resolving tier.
    wallet = MagicMock()
    wallet.can_afford.return_value = True
    wallet.transfer = AsyncMock()
    agent.wallet = wallet
    feature = SovereigntyFeature(agent)

    captured = {}

    def fake_store_content(*args, **kwargs):
        captured["tier"] = kwargs.get("storage_tier")
        return StorageResult(
            content_hash="abc",
            cid="QmDefault",
            tier=StorageTier.IPFS,
            provider="ipfs",
            encrypted=False,
            size_bytes=1024,
        )

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter.store_content",
        side_effect=fake_store_content,
    ):
        result = await feature.export_sovereignty(encrypt=False)

    assert captured["tier"] == StorageTier.IPFS
    assert result.data["tier_requested"] == "ipfs"
