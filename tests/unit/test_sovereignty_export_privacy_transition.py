"""#2875 — a sovereignty export and a privacy transition do not interleave.

The export reads the agent's privacy policy before it builds the backup blob,
and reads it again immediately before the store, holding the durable
persistence gate shared from that second read until the store and its receipts
are written. ``privacy_transition()`` takes the same gate exclusive, so a
transition either completes before the second read, which then refuses, or
waits for the store.

These tests drive a real ``KestrelAgent``'s ``privacy_transition()`` (the
CONVERSATION -> privacy-transition lock -> persistence gate span) against a
real ``PrivacyEnforcingStorage`` over SQLite. Inside the span each transition
applies the same storage flip ``_apply_privacy_mode_locked`` makes. Only the
provider stack behind ``FilecoinAdapter`` is replaced, by one that records
every store and can hold a store in flight on its executor thread.
Interleavings are driven by events, never by sleeps.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.sovereignty.feature import (
    REFUSAL_PRIVACY_MODE_FORBIDS_BACKUP,
    SovereigntyFeature,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.storage.providers.base import StorageResult, StorageTier

# Bounds a deadlock, not the work: a passing run returns as soon as the awaited
# event fires. Kept well inside CI's 60s per-test timeout (see #3454).
_HANG_GUARD_SECONDS = 30

AGENT_DID = "did:pkh:eip155:1:0x2875export"


class _Provider:
    """The provider stack behind ``FilecoinAdapter``: records every store.

    A store runs on the export's executor thread. While ``release`` is clear
    it blocks there after setting ``entered``, so a test can act while a
    store is in flight. Each store records whether the agent's privacy policy
    permitted a backup when it began and when it finished.
    """

    def __init__(self, agent: KestrelAgent) -> None:
        self._agent = agent
        self._lock = threading.Lock()
        self.stores: list[dict] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.barrier: threading.Barrier | None = None

    def store_content(self, content, *, storage_tier, encrypt, **_kwargs):
        record = {
            "tier": storage_tier,
            "permitted_at_start": self._agent.storage.allows_backups(),
        }
        with self._lock:
            index = len(self.stores)
            self.stores.append(record)
        self.entered.set()
        if self.barrier is not None:
            self.barrier.wait()
        if not self.release.wait(_HANG_GUARD_SECONDS):
            raise AssertionError("the store was never released")
        record["permitted_at_end"] = self._agent.storage.allows_backups()
        return StorageResult(
            content_hash=f"{index + 1:064x}",
            cid=f"bafy2875{index}",
            tier=storage_tier,
            provider="ipfs",
            encrypted=encrypt,
            size_bytes=len(content),
        )


@pytest.fixture
async def rig(tmp_path):
    agent = KestrelAgent(did=AGENT_DID, storage_path=":memory:")
    raw = Storage(str(tmp_path / "kestrel_prime.db"), agent_id=agent.did)
    await raw.initialize()
    agent.storage = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
    wallet = MagicMock()
    wallet.can_afford.return_value = True
    wallet.transfer = AsyncMock()
    agent.wallet = wallet
    provider = _Provider(agent)
    try:
        with patch(
            "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter",
            return_value=provider,
        ):
            yield SimpleNamespace(
                agent=agent,
                raw=raw,
                provider=provider,
                wallet=wallet,
                feature=SovereigntyFeature(agent),
            )
    finally:
        provider.release.set()
        await raw.close()


async def _transition(
    agent: KestrelAgent, mode: PrivacyMode, entered: asyncio.Event | None = None
) -> None:
    """Change the privacy mode inside the agent's real transition span."""
    async with agent.privacy_transition():
        if entered is not None:
            entered.set()
        agent.storage.set_privacy_mode(mode)


async def _wait_for_store(provider: _Provider) -> None:
    entered = await asyncio.to_thread(provider.entered.wait, _HANG_GUARD_SECONDS)
    assert entered, "the export never reached its store"


async def _wait_until_queued_on_the_gate(agent: KestrelAgent) -> None:
    """Return once a transition waits for the persistence gate exclusive."""
    gate = agent._get_durable_persistence_gate()
    for _ in range(200):
        if gate._lock._waiting_writers:
            return
        await asyncio.sleep(0)
    raise AssertionError("the transition never queued on the persistence gate")


async def _receipts(raw: Storage) -> list:
    return await raw.get_nodes_by_type("backup_artifact")


async def _export_receipts(raw: Storage) -> list:
    return await raw.get_nodes_by_type("sovereignty_receipt")


@pytest.mark.asyncio
async def test_a_transition_between_the_policy_read_and_the_store_prevents_the_remote_store(
    rig,
):
    """Required test 1: no remote store under a mode that forbids it."""
    agent = rig.agent
    building = asyncio.Event()
    release_blob = asyncio.Event()
    build_backup_blob = rig.raw.create_backup_blob

    async def paused_backup_blob(include_db: bool = True) -> bytes:
        # The export has read the policy, and the storage wrapper has
        # admitted the blob, both under NORMAL.
        building.set()
        await release_blob.wait()
        return await build_backup_blob(include_db)

    rig.raw.create_backup_blob = paused_backup_blob
    export = asyncio.create_task(
        rig.feature.export_sovereignty(storage_tier="ipfs", encrypt=True)
    )
    await asyncio.wait_for(building.wait(), timeout=_HANG_GUARD_SECONDS)

    # Nothing holds the gate yet, so the transition completes here.
    await asyncio.wait_for(
        _transition(agent, PrivacyMode.EPHEMERAL), timeout=_HANG_GUARD_SECONDS
    )
    assert agent.storage.allows_backups() is False
    release_blob.set()
    result = await asyncio.wait_for(export, timeout=_HANG_GUARD_SECONDS)

    assert result.status is ToolResultStatus.ERROR
    assert result.data == {"refusal": REFUSAL_PRIVACY_MODE_FORBIDS_BACKUP}
    assert "ephemeral" in result.error
    assert rig.provider.stores == [], "the export stored after the transition"
    rig.wallet.transfer.assert_not_awaited()
    assert await _receipts(rig.raw) == []


@pytest.mark.asyncio
async def test_a_transition_waits_for_an_in_flight_store(rig):
    """Required test 2: the transition does not complete inside the store."""
    agent = rig.agent
    rig.provider.release.clear()
    export = asyncio.create_task(
        rig.feature.export_sovereignty(storage_tier="ipfs", encrypt=True)
    )
    await _wait_for_store(rig.provider)

    entered = asyncio.Event()
    transition = asyncio.create_task(
        _transition(agent, PrivacyMode.EPHEMERAL, entered)
    )
    await _wait_until_queued_on_the_gate(agent)
    assert not entered.is_set(), "a transition completed inside the export's store"
    assert agent.storage.allows_backups() is True

    rig.provider.release.set()
    result = await asyncio.wait_for(export, timeout=_HANG_GUARD_SECONDS)
    await asyncio.wait_for(transition, timeout=_HANG_GUARD_SECONDS)

    assert result.status is ToolResultStatus.OK, result
    assert result.data["tier"] == "ipfs"
    assert rig.provider.stores == [
        {
            "tier": StorageTier.IPFS,
            "permitted_at_start": True,
            "permitted_at_end": True,
        }
    ]
    # Its receipts were written before the transition, which would refuse them.
    assert [node.node_id for node in await _receipts(rig.raw)] == [f"{1:064x}"]
    assert entered.is_set()
    assert agent.storage.allows_backups() is False


@pytest.mark.asyncio
async def test_a_cancelled_export_holds_the_gate_until_its_store_has_finished(rig):
    """Cancelling the export does not release the gate under a running store.

    The store's executor thread cannot be interrupted, so the export finishes
    it before the cancellation unwinds through the gate. The backup it landed
    keeps its receipts and its fee: they settle with the store, and the
    cancellation is raised after them.
    """
    agent = rig.agent
    rig.provider.release.clear()
    export = asyncio.create_task(
        rig.feature.export_sovereignty(storage_tier="ipfs", encrypt=True)
    )
    await _wait_for_store(rig.provider)

    entered = asyncio.Event()
    transition = asyncio.create_task(
        _transition(agent, PrivacyMode.EPHEMERAL, entered)
    )
    await _wait_until_queued_on_the_gate(agent)
    export.cancel()
    for _ in range(50):
        await asyncio.sleep(0)
    assert not export.done(), "the export unwound while its store was in flight"
    assert not entered.is_set(), "the transition completed inside a cancelled store"

    rig.provider.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(export, timeout=_HANG_GUARD_SECONDS)
    await asyncio.wait_for(transition, timeout=_HANG_GUARD_SECONDS)
    assert entered.is_set()
    assert rig.provider.stores[0]["permitted_at_end"] is True

    artifacts = await _receipts(rig.raw)
    assert [node.node_id for node in artifacts] == [f"{1:064x}"]
    receipts = await _export_receipts(rig.raw)
    assert [node.properties["cid"] for node in receipts] == ["bafy28750"]
    assert receipts[0].properties["node_id"] == artifacts[0].node_id
    rig.wallet.transfer.assert_awaited_once()
    assert rig.wallet.transfer.await_args.kwargs["memo"] == (
        f"backup:ipfs:{artifacts[0].node_id}"
    )


@pytest.mark.asyncio
async def test_an_export_cancelled_during_its_fee_still_pays_it(rig):
    """A cancellation that arrives while the fee is charged waits for it."""
    charging = asyncio.Event()
    release_charge = asyncio.Event()
    charged = []

    async def transfer(amount, *, memo):
        charging.set()
        await release_charge.wait()
        charged.append((amount, memo))

    rig.wallet.transfer = transfer
    export = asyncio.create_task(
        rig.feature.export_sovereignty(storage_tier="ipfs", encrypt=True)
    )
    await asyncio.wait_for(charging.wait(), timeout=_HANG_GUARD_SECONDS)
    export.cancel()
    for _ in range(50):
        await asyncio.sleep(0)
    assert not export.done(), "the export unwound while its fee was being charged"

    release_charge.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(export, timeout=_HANG_GUARD_SECONDS)
    (artifact,) = await _receipts(rig.raw)
    assert len(await _export_receipts(rig.raw)) == 1
    assert [memo for _, memo in charged] == [f"backup:ipfs:{artifact.node_id}"]


@pytest.mark.asyncio
async def test_an_export_behind_a_queued_transition_reads_the_policy_it_installs(rig):
    """The second policy read happens under the gate, after a queued transition."""
    agent = rig.agent
    rig.provider.release.clear()
    first = asyncio.create_task(
        rig.feature.export_sovereignty(storage_tier="ipfs", encrypt=True)
    )
    await _wait_for_store(rig.provider)
    transition = asyncio.create_task(_transition(agent, PrivacyMode.EPHEMERAL))
    await _wait_until_queued_on_the_gate(agent)

    building = asyncio.Event()
    build_backup_blob = rig.raw.create_backup_blob

    async def observed_backup_blob(include_db: bool = True) -> bytes:
        building.set()
        return await build_backup_blob(include_db)

    rig.raw.create_backup_blob = observed_backup_blob
    # Read under NORMAL before the blob; then it waits behind the transition.
    second = asyncio.create_task(
        rig.feature.export_sovereignty(storage_tier="ipfs", encrypt=True)
    )
    await asyncio.wait_for(building.wait(), timeout=_HANG_GUARD_SECONDS)
    for _ in range(50):
        await asyncio.sleep(0)
    assert not second.done()

    rig.provider.release.set()
    first_result = await asyncio.wait_for(first, timeout=_HANG_GUARD_SECONDS)
    await asyncio.wait_for(transition, timeout=_HANG_GUARD_SECONDS)
    second_result = await asyncio.wait_for(second, timeout=_HANG_GUARD_SECONDS)

    assert first_result.status is ToolResultStatus.OK, first_result
    assert second_result.status is ToolResultStatus.ERROR
    assert second_result.data == {"refusal": REFUSAL_PRIVACY_MODE_FORBIDS_BACKUP}
    assert len(rig.provider.stores) == 1, "the second export stored after the transition"


@pytest.mark.asyncio
async def test_concurrent_exports_with_no_transition_both_succeed(rig):
    """Required test 3: exports share the gate rather than serialize.

    Each store waits at a two-party barrier, which only opens while both
    stores are in flight at once. Exports that serialized on the gate would
    break it.
    """
    rig.provider.barrier = threading.Barrier(2, timeout=_HANG_GUARD_SECONDS)

    results = await asyncio.wait_for(
        asyncio.gather(
            rig.feature.export_sovereignty(storage_tier="ipfs", encrypt=True),
            rig.feature.export_sovereignty(storage_tier="ipfs", encrypt=True),
        ),
        timeout=_HANG_GUARD_SECONDS,
    )

    assert [result.status for result in results] == [ToolResultStatus.OK] * 2, results
    assert {result.data["cid"] for result in results} == {"bafy28750", "bafy28751"}
    assert len(rig.provider.stores) == 2
    assert len(await _receipts(rig.raw)) == 2
    assert rig.wallet.transfer.await_count == 2


class _BackupRecorder:
    """Underlying storage that records whether a backup blob was built."""

    def __init__(self) -> None:
        self.backup_calls = 0

    async def create_backup_blob(self, include_db: bool = True) -> bytes:
        self.backup_calls += 1
        return b"backup-bytes"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "getter",
    [None, lambda: MagicMock()],
    ids=["no-accessor", "accessor-returns-no-gate"],
)
async def test_an_export_without_a_persistence_gate_is_refused_and_stores_nothing(
    getter,
):
    """Required test 4: no gate to hold, no export."""
    underlying = _BackupRecorder()
    agent = SimpleNamespace(
        agent_id=AGENT_DID,
        storage=PrivacyEnforcingStorage(underlying, PrivacyMode.NORMAL),
        features={},
    )
    if getter is not None:
        agent._get_durable_persistence_gate = getter

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter"
    ) as adapter:
        result = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="local", encrypt=False
        )

    assert result.status is ToolResultStatus.ERROR
    assert "no durable persistence gate" in result.error
    assert "no sovereignty backup was created" in result.error
    assert underlying.backup_calls == 0
    adapter.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED],
)
async def test_an_export_in_a_mode_that_forbids_backups_is_refused_before_the_blob(
    mode,
):
    from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate

    underlying = _BackupRecorder()
    gate = DurablePersistenceGate()
    agent = SimpleNamespace(
        agent_id=AGENT_DID,
        storage=PrivacyEnforcingStorage(underlying, mode),
        features={},
        _get_durable_persistence_gate=lambda: gate,
    )

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter"
    ) as adapter:
        result = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="local", encrypt=False
        )

    assert result.status is ToolResultStatus.ERROR
    assert result.data == {"refusal": REFUSAL_PRIVACY_MODE_FORBIDS_BACKUP}
    assert mode.value in result.error
    assert underlying.backup_calls == 0
    adapter.assert_not_called()


# --- The sleep cycle's export, which the scheduler runs ------------------
#
# ``SleepMixin._export_sovereignty`` reads the database beneath the privacy
# wrapper and stores through its own ``SovereignStorageAdapter``, so nothing
# in the wrapper refuses it. It is driven here through the scheduler's
# ``sleep`` handler with export enabled, as a schedule's args can request.

_SLEEP_EXPORT_ARGS = {
    "skip_export": False,
    "skip_consolidation": True,
    "skip_reflection": True,
}


@pytest.fixture
async def sleep_rig(tmp_path):
    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature

    agent = KestrelAgent(did=AGENT_DID, storage_path=":memory:")
    raw = Storage(str(tmp_path / "kestrel_prime.db"), agent_id=agent.did)
    await raw.initialize()
    agent._raw_storage = raw
    agent.storage = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
    provider = _Provider(agent)
    scheduler = SchedulerFeature(agent)
    scheduler._agent_id = agent.did
    try:
        with patch(
            "kestrel_sovereign.filecoin_adapter.FilecoinAdapter",
            return_value=provider,
        ):
            yield SimpleNamespace(
                agent=agent, raw=raw, provider=provider, scheduler=scheduler
            )
    finally:
        provider.release.set()
        await raw.close()


def _assert_sleep_export_succeeded(outcome) -> None:
    assert isinstance(outcome, str), outcome
    assert '"success": true' in outcome, outcome
    assert '"cid": "bafy28750"' in outcome, outcome


def _assert_sleep_export_refused(outcome) -> None:
    assert getattr(outcome, "status", None) == "failed", outcome
    assert outcome.reason_code == "export_failed", outcome


@pytest.fixture
def paused_archive(monkeypatch):
    """Pause the sleep export after its first policy read, before its store."""
    from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter

    paused = SimpleNamespace(
        building=asyncio.Event(), release=asyncio.Event(), calls=0
    )
    shard_conversations = SovereignStorageAdapter._shard_conversations

    async def paused_shards(self):
        paused.calls += 1
        paused.building.set()
        await paused.release.wait()
        return await shard_conversations(self)

    monkeypatch.setattr(SovereignStorageAdapter, "_shard_conversations", paused_shards)
    return paused


@pytest.mark.asyncio
async def test_a_scheduled_sleep_export_stores_with_no_transition(sleep_rig):
    outcome = await asyncio.wait_for(
        sleep_rig.scheduler._handle_sleep(dict(_SLEEP_EXPORT_ARGS)),
        timeout=_HANG_GUARD_SECONDS,
    )

    _assert_sleep_export_succeeded(outcome)
    assert sleep_rig.provider.stores == [
        {
            "tier": StorageTier.IPFS,
            "permitted_at_start": True,
            "permitted_at_end": True,
        }
    ]


@pytest.mark.asyncio
async def test_a_transition_before_a_scheduled_sleep_store_prevents_it(
    sleep_rig, paused_archive
):
    """Required test 1, for the scheduled sleep export."""
    sleep = asyncio.create_task(
        sleep_rig.scheduler._handle_sleep(dict(_SLEEP_EXPORT_ARGS))
    )
    await asyncio.wait_for(paused_archive.building.wait(), timeout=_HANG_GUARD_SECONDS)

    # Nothing holds the gate while the archive is built, so this completes.
    await asyncio.wait_for(
        _transition(sleep_rig.agent, PrivacyMode.EPHEMERAL),
        timeout=_HANG_GUARD_SECONDS,
    )
    paused_archive.release.set()
    outcome = await asyncio.wait_for(sleep, timeout=_HANG_GUARD_SECONDS)

    _assert_sleep_export_refused(outcome)
    assert sleep_rig.provider.stores == [], "the sleep export stored after the transition"


@pytest.mark.asyncio
async def test_a_transition_waits_for_an_in_flight_scheduled_sleep_store(sleep_rig):
    """Required test 2, for the scheduled sleep export."""
    agent = sleep_rig.agent
    sleep_rig.provider.release.clear()
    sleep = asyncio.create_task(
        sleep_rig.scheduler._handle_sleep(dict(_SLEEP_EXPORT_ARGS))
    )
    await _wait_for_store(sleep_rig.provider)

    entered = asyncio.Event()
    transition = asyncio.create_task(
        _transition(agent, PrivacyMode.EPHEMERAL, entered)
    )
    await _wait_until_queued_on_the_gate(agent)
    assert not entered.is_set(), "a transition completed inside the sleep export's store"

    sleep_rig.provider.release.set()
    outcome = await asyncio.wait_for(sleep, timeout=_HANG_GUARD_SECONDS)
    await asyncio.wait_for(transition, timeout=_HANG_GUARD_SECONDS)

    _assert_sleep_export_succeeded(outcome)
    assert sleep_rig.provider.stores[0]["permitted_at_end"] is True
    assert entered.is_set()


@pytest.mark.asyncio
async def test_a_cancelled_sleep_export_holds_the_gate_until_its_store_has_finished(
    sleep_rig,
):
    agent = sleep_rig.agent
    sleep_rig.provider.release.clear()
    sleep = asyncio.create_task(
        sleep_rig.scheduler._handle_sleep(dict(_SLEEP_EXPORT_ARGS))
    )
    await _wait_for_store(sleep_rig.provider)

    entered = asyncio.Event()
    transition = asyncio.create_task(
        _transition(agent, PrivacyMode.EPHEMERAL, entered)
    )
    await _wait_until_queued_on_the_gate(agent)
    sleep.cancel()
    for _ in range(50):
        await asyncio.sleep(0)
    assert not sleep.done(), "the sleep export unwound while its store was in flight"
    assert not entered.is_set(), "the transition completed inside a cancelled store"

    sleep_rig.provider.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(sleep, timeout=_HANG_GUARD_SECONDS)
    await asyncio.wait_for(transition, timeout=_HANG_GUARD_SECONDS)
    assert sleep_rig.provider.stores[0]["permitted_at_end"] is True


@pytest.mark.asyncio
async def test_a_scheduled_sleep_export_without_a_persistence_gate_is_refused(
    sleep_rig, paused_archive
):
    """Required test 4, for the scheduled sleep export."""
    sleep_rig.agent._get_durable_persistence_gate = lambda: None
    paused_archive.release.set()

    outcome = await asyncio.wait_for(
        sleep_rig.scheduler._handle_sleep(dict(_SLEEP_EXPORT_ARGS)),
        timeout=_HANG_GUARD_SECONDS,
    )

    _assert_sleep_export_refused(outcome)
    assert paused_archive.calls == 0, "the archive was built with no gate to hold"
    assert sleep_rig.provider.stores == []


@pytest.mark.asyncio
async def test_a_scheduled_sleep_export_in_a_mode_that_forbids_backups_is_refused_before_the_archive(
    sleep_rig, paused_archive
):
    paused_archive.release.set()
    await _transition(sleep_rig.agent, PrivacyMode.ISOLATED)

    outcome = await asyncio.wait_for(
        sleep_rig.scheduler._handle_sleep(dict(_SLEEP_EXPORT_ARGS)),
        timeout=_HANG_GUARD_SECONDS,
    )

    _assert_sleep_export_refused(outcome)
    assert paused_archive.calls == 0
    assert sleep_rig.provider.stores == []
