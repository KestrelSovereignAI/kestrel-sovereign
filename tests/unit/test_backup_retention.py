from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kestrel_sovereign.storage.sync.retention import (
    DataClass,
    RetentionItem,
    RetentionPolicy,
    classify,
    parse_timestamp,
)
from kestrel_sovereign.storage.sync.service import SyncService
from kestrel_sovereign.storage.sync.targets import GCSTarget, SyncResult, SyncTarget


NOW = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)


def _item(
    key: str,
    days_old: int,
    data_class: DataClass = DataClass.WORKING_MEMORY,
) -> RetentionItem:
    return RetentionItem(
        key=key,
        name=key,
        timestamp=NOW - timedelta(days=days_old),
        data_class=data_class,
    )


def _deleted_keys(policy: RetentionPolicy, items: list[RetentionItem]) -> set[str]:
    return {item.key for item in policy.deletions(items, now=NOW)}


def test_working_memory_keeps_all_under_14_days_then_one_per_iso_week():
    policy = RetentionPolicy()
    items = [
        _item("newest.db", 1),
        _item("recent-a.db", 7),
        _item("recent-b.db", 13),
        _item("week-a-old.db", 18),
        _item("week-a-new.db", 17),
        _item("week-b-old.db", 33),
        _item("week-b-new.db", 32),
    ]

    deleted = _deleted_keys(policy, items)

    assert "recent-a.db" not in deleted
    assert "recent-b.db" not in deleted
    assert "week-a-old.db" in deleted
    assert "week-a-new.db" not in deleted
    assert "week-b-old.db" in deleted
    assert "week-b-new.db" not in deleted


def test_identity_keeps_all_under_30_days_and_survives_longer_than_working():
    policy = RetentionPolicy()
    working = [
        _item("working-newest.db", 1),
        _item("working-18-old.db", 18),
        _item("working-17-new.db", 17),
    ]
    identity = [
        _item("identity-newest.json", 1, DataClass.IDENTITY),
        _item("identity-18-old.json", 18, DataClass.IDENTITY),
        _item("identity-17-new.json", 17, DataClass.IDENTITY),
    ]

    deleted = _deleted_keys(policy, working + identity)

    assert "working-18-old.db" in deleted
    assert "identity-18-old.json" not in deleted


def test_identity_uses_weekly_until_12_months_then_monthly_forever():
    policy = RetentionPolicy()
    items = [
        _item("identity-newest.json", 1, DataClass.IDENTITY),
        _item("weekly-old.json", 120, DataClass.IDENTITY),
        _item("weekly-new.json", 118, DataClass.IDENTITY),
        _item("monthly-old.json", 740, DataClass.IDENTITY),
        _item("monthly-new.json", 730, DataClass.IDENTITY),
    ]

    deleted = _deleted_keys(policy, items)

    assert "weekly-old.json" in deleted
    assert "weekly-new.json" not in deleted
    assert "monthly-old.json" in deleted
    assert "monthly-new.json" not in deleted


def test_newest_item_per_class_is_never_pruned():
    policy = RetentionPolicy.from_config(
        {
            "backup": {
                "retention": {
                    "working_memory": {
                        "keep_all_days": 0,
                        "weekly_forever": False,
                        "monthly_forever": False,
                    },
                    "identity": {
                        "keep_all_days": 0,
                        "weekly_until_months": 0,
                        "monthly_forever": False,
                    },
                }
            }
        }
    )
    items = [
        _item("only-working.db", 900, DataClass.WORKING_MEMORY),
        _item("only-identity.json", 900, DataClass.IDENTITY),
    ]

    assert _deleted_keys(policy, items) == set()


def test_config_override_changes_working_memory_keep_all_window():
    default_policy = RetentionPolicy()
    override_policy = RetentionPolicy.from_config(
        {
            "backup": {
                "retention": {
                    "working_memory": {
                        "keep_all_days": 40,
                        "weekly_forever": True,
                    }
                }
            }
        }
    )
    items = [
        _item("newest.db", 1),
        _item("same-week-old.db", 33),
        _item("same-week-new.db", 32),
    ]

    assert "same-week-old.db" in _deleted_keys(default_policy, items)
    assert "same-week-old.db" not in _deleted_keys(override_policy, items)


def test_classify_known_identity_and_working_names():
    assert classify("kestrel_prime.db") == DataClass.WORKING_MEMORY
    assert classify("did-document.json") == DataClass.IDENTITY
    assert classify({"tag": "kestrel-manifest-agent"}) == DataClass.IDENTITY
    assert (
        classify({"role": "snapshot", "key": "snapshots/20260620_120000.db"})
        == DataClass.WORKING_MEMORY
    )


class PruneFailingTarget(SyncTarget):
    @property
    def name(self) -> str:
        return "prune-failing-target"

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=db_path.stat().st_size,
            frames_synced=0,
            timestamp=NOW,
            metadata={"uploaded": True},
        )

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        return SyncResult(True, self.name, 0, 0, NOW)

    async def get_latest_position(self):
        return 2**63

    async def prune(self, policy: RetentionPolicy):
        raise RuntimeError("prune boom")


@pytest.mark.asyncio
async def test_prune_failure_does_not_fail_backup_cycle(tmp_path):
    db_path = tmp_path / "kestrel_prime.db"
    db_path.write_bytes(b"sqlite bytes")
    sync = SyncService(
        db_path=str(db_path),
        state_file=str(tmp_path / "sync.state"),
        retention_policy=RetentionPolicy(),
    )
    sync.add_target(PruneFailingTarget())
    await sync.start()

    result = (await sync.force_snapshot())["prune-failing-target"]

    assert result.success is True
    assert result.metadata["uploaded"] is True
    assert result.metadata["prune_error"] == "prune boom"


class FakeBlob:
    def __init__(self, name: str):
        self.name = name
        self.deleted = False

    def delete(self):
        self.deleted = True


class FakeBucket:
    def __init__(self, blobs: list[FakeBlob]):
        self.blobs = blobs

    def list_blobs(self, prefix: str):
        return [blob for blob in self.blobs if blob.name.startswith(prefix)]


@pytest.mark.asyncio
async def test_gcs_prune_scans_snapshots_and_leaves_latest_pointer():
    blobs = [
        FakeBlob("kestrel/agent-1/snapshots/20240102_120000.db"),
        FakeBlob("kestrel/agent-1/snapshots/20240103_120000.db"),
        FakeBlob("kestrel/agent-1/latest.db"),
    ]
    target = GCSTarget(bucket="bucket", agent_id="agent-1")
    target._bucket = FakeBucket(blobs)

    result = await target.prune(RetentionPolicy())

    assert result["deleted"] == 1
    assert blobs[0].deleted is True
    assert blobs[1].deleted is False
    assert blobs[2].deleted is False


def test_parse_timestamp_handles_lighthouse_epoch_ms():
    # Lighthouse `createdAt` is Unix milliseconds; must parse, not skip.
    ms = 1781956879692  # 2026-06-...
    parsed = parse_timestamp(ms)
    assert parsed is not None
    assert parsed.tzinfo is not None
    # Same instant whether passed as int or numeric string.
    assert parse_timestamp(str(ms)) == parsed
    # Sanity: milliseconds, not seconds (year must be sane, not 1970/56xxx).
    assert 2020 <= parsed.year <= 2100


def test_parse_timestamp_handles_epoch_seconds_and_calendar_codes():
    secs = 1781956879  # ~2026 in seconds
    assert parse_timestamp(secs) is not None
    assert 2020 <= parse_timestamp(secs).year <= 2100
    # An 8-digit calendar code must NOT be misread as a ~1970 epoch; the >=1e9
    # guard lets it fall through to ISO-basic parsing (2026-01-01).
    cal = parse_timestamp("20260101")
    assert cal is not None and cal.year == 2026
