import inspect
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from kestrel_sovereign.storage.sync.retention import DataClass, RetentionPolicy
from scripts import backup_cleanup


NOW = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)


def _record(
    key: str,
    agent_id: str | None,
    days_old: int,
    *,
    store: str = "gcs",
    size: int = 100,
    data_class: DataClass = DataClass.WORKING_MEMORY,
    name: str | None = None,
    attributed: bool = True,
):
    name = name or key.rsplit("/", 1)[-1]
    return backup_cleanup.BackupRecord(
        store=store,
        key=key,
        agent_id=agent_id,
        name=name,
        size=size,
        timestamp=NOW - timedelta(days=days_old),
        data_class=data_class,
        metadata={},
        attributed=attributed,
    )


def _delete_keys(plan):
    return {row.record.key for row in plan.deletions}


def test_dry_run_reports_correct_keep_delete_sets_against_synthetic_fixture():
    records = [
        _record("gs://bucket/kestrel/agent-a/snapshots/20260619_120000.db", "agent-a", 1),
        _record("gs://bucket/kestrel/agent-a/snapshots/20260603_120000.db", "agent-a", 17),
        _record("gs://bucket/kestrel/agent-a/snapshots/20260602_120000.db", "agent-a", 18),
        _record("gs://bucket/kestrel/agent-b/snapshots/20240101_120000.db", "agent-b", 900),
        _record(
            "cid-manifest-new",
            "did:key:agent-a",
            1,
            store="lighthouse",
            data_class=DataClass.IDENTITY,
            name="manifest_did:key:agent-a.json",
        ),
        _record(
            "cid-manifest-old",
            "did:key:agent-a",
            40,
            store="lighthouse",
            data_class=DataClass.IDENTITY,
            name="manifest_did:key:agent-a.json",
        ),
    ]

    plan = backup_cleanup.build_plan(records, RetentionPolicy(), now=NOW)
    report = backup_cleanup.render_report(plan)

    assert _delete_keys(plan) == {
        "gs://bucket/kestrel/agent-a/snapshots/20260602_120000.db"
    }
    assert "gcs agent-a: keep 2" in report
    assert "delete 1" in report
    assert "gcs agent-b: keep 1" in report  # newest per agent is preserved.
    assert "lighthouse did:key:agent-a: keep 2" in report


@pytest.mark.asyncio
async def test_apply_without_confirmation_refuses_to_delete(tmp_path):
    plan = backup_cleanup.build_plan(
        [
            _record("gs://bucket/kestrel/agent-a/snapshots/new.db", "agent-a", 1),
            _record("gs://bucket/kestrel/agent-a/snapshots/old.db", "agent-a", 900),
        ],
        RetentionPolicy.from_config(
            {
                "backup": {
                    "retention": {
                        "working_memory": {
                            "keep_all_days": 0,
                            "weekly_forever": False,
                            "monthly_forever": False,
                        }
                    }
                }
            }
        ),
        now=NOW,
    )

    with patch("scripts.backup_cleanup.input", side_effect=EOFError), patch(
        "scripts.backup_cleanup.subprocess.run"
    ) as run:
        rc = await backup_cleanup.apply_plan(
            plan,
            lighthouse_client=None,
            confirmation=None,
            audit_log=tmp_path / "audit.jsonl",
        )

    assert rc == 2
    run.assert_not_called()


@pytest.mark.asyncio
async def test_apply_with_confirmation_deletes_and_audits_gcs_and_lighthouse(tmp_path):
    records = [
        _record("gs://bucket/kestrel/agent-a/snapshots/new.db", "agent-a", 1),
        _record("gs://bucket/kestrel/agent-a/snapshots/old.db", "agent-a", 900),
        _record(
            "cid-new",
            "agent-a",
            1,
            store="lighthouse",
            name="new.db",
        ),
        _record(
            "cid-old",
            "agent-a",
            900,
            store="lighthouse",
            name="old.db",
        ),
    ]
    plan = backup_cleanup.build_plan(
        records,
        RetentionPolicy.from_config(
            {
                "backup": {
                    "retention": {
                        "working_memory": {
                            "keep_all_days": 0,
                            "weekly_forever": False,
                            "monthly_forever": False,
                        }
                    }
                }
            }
        ),
        now=NOW,
    )
    lighthouse_client = AsyncDeleteClient()
    audit_log = tmp_path / "audit.jsonl"

    with patch("scripts.backup_cleanup.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        rc = await backup_cleanup.apply_plan(
            plan,
            lighthouse_client=lighthouse_client,
            confirmation=backup_cleanup.CONFIRMATION_PHRASE,
            audit_log=audit_log,
        )

    assert rc == 0
    run.assert_called_once()
    assert run.call_args.args[0][-1] == "gs://bucket/kestrel/agent-a/snapshots/old.db"
    assert lighthouse_client.deleted == ["cid-old"]
    audit_entries = [json.loads(line) for line in audit_log.read_text().splitlines()]
    assert {entry["key"] for entry in audit_entries} == {
        "gs://bucket/kestrel/agent-a/snapshots/old.db",
        "cid-old",
    }


class AsyncDeleteClient:
    def __init__(self):
        self.deleted = []

    async def get_uploads(self, last_key=None):
        return {"fileList": [], "totalFiles": 0}

    async def download(self, cid, timeout=None):
        return b"{}"

    async def delete_file(self, cid):
        self.deleted.append(cid)
        return {"deleted": True}

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_unattributed_lighthouse_files_are_reported_and_never_deleted():
    class FakeClient:
        async def get_uploads(self, last_key=None):
            return {
                "fileList": [
                    {
                        "cid": "cid-known",
                        "fileName": "snapshot.db",
                        "fileSizeInBytes": "100",
                        "createdAt": int((NOW - timedelta(days=900)).timestamp() * 1000),
                    },
                    {
                        "cid": "cid-manifest",
                        "fileName": "manifest_did:key:agent-a.json",
                        "fileSizeInBytes": "20",
                        "createdAt": int((NOW - timedelta(days=1)).timestamp() * 1000),
                    },
                    {
                        "cid": "cid-unknown",
                        "fileName": "orphan.db",
                        "fileSizeInBytes": "100",
                        "createdAt": int((NOW - timedelta(days=900)).timestamp() * 1000),
                    },
                ],
                "totalFiles": 3,
            }

        async def download(self, cid, timeout=None):
            assert cid == "cid-manifest"
            return json.dumps(
                {"agent_id": "did:key:agent-a", "snapshot_cid": "cid-known"}
            ).encode()

        async def delete_file(self, cid):
            raise AssertionError("delete_file should not be called in this test")

        async def close(self):
            pass

    records = await backup_cleanup.lighthouse_records(FakeClient())
    plan = backup_cleanup.build_plan(
        records,
        RetentionPolicy.from_config(
            {
                "backup": {
                    "retention": {
                        "working_memory": {
                            "keep_all_days": 0,
                            "weekly_forever": False,
                            "monthly_forever": False,
                        }
                    }
                }
            }
        ),
        now=NOW,
    )
    report = backup_cleanup.render_report(plan)

    assert "cid-unknown" not in _delete_keys(plan)
    assert "cid-known" not in _delete_keys(plan)  # newest/only working item for agent.
    assert "(unattributed)" in report
    assert "cid-unknown" in report


def test_latest_db_is_never_deleted_even_when_it_matches_test_artifact():
    record = backup_cleanup.BackupRecord(
        store="gcs",
        key="gs://bucket/kestrel/did:test:agent/latest.db",
        agent_id="did:test:agent",
        name="latest.db",
        size=100,
        timestamp=NOW - timedelta(days=900),
        data_class=DataClass.WORKING_MEMORY,
        metadata={},
        attributed=True,
        test_artifact=True,
        protected=True,
        reason="latest.db",
    )

    plan = backup_cleanup.build_plan([record], RetentionPolicy(), now=NOW)

    assert plan.deletions == ()
    assert plan.records[0].reason == "latest.db"


def test_test_artifacts_are_deleted_outside_gfs_policy():
    records = [
        _record("gs://bucket/kestrel/did:test:agent/test-old.db", "did:test:agent", 1),
        _record("gs://bucket/kestrel/agent-a/gcs-live-test.db", "agent-a", 1),
    ]
    records = [backup_cleanup._with_test_flag(record) for record in records]

    plan = backup_cleanup.build_plan(records, RetentionPolicy(), now=NOW)

    assert _delete_keys(plan) == {record.key for record in records}
    assert {row.reason for row in plan.deletions} == {"test_artifact"}


def test_reuses_shared_retention_policy_and_classify_without_local_gfs_copy():
    source = inspect.getsource(backup_cleanup)

    assert backup_cleanup.RetentionPolicy is RetentionPolicy
    assert "from kestrel_sovereign.storage.sync.retention import" in source
    assert "RetentionPolicy" in source
    assert "classify" in source
    assert "weekly_until_months" not in source
    assert "monthly_forever" not in source


def test_parse_gsutil_ls_handles_real_single_token_iso_timestamp():
    # Real `gsutil ls -l` output: size, single ISO8601 token, URI, then a
    # trailing TOTAL line. Regression for the two-token regex that silently
    # dropped every GCS object.
    output = (
        "    299177514  2026-06-19T12:00:00Z  "
        "gs://bucket/kestrel/agent-a/snapshots/20260619_120000.db\n"
        "    179382805  2024-01-01T12:00:00Z  "
        "gs://bucket/kestrel/agent-a/snapshots/20240101_120000.db\n"
        "          529  2026-06-19T12:00:01Z  "
        "gs://bucket/kestrel/agent-a/latest.db\n"
        "TOTAL: 3 objects, 479000000 bytes\n"
    )
    records = backup_cleanup.parse_gsutil_ls(
        output, bucket="bucket", prefix="kestrel/"
    )
    keys = {r.key for r in records}
    assert (
        "gs://bucket/kestrel/agent-a/snapshots/20260619_120000.db" in keys
    )
    assert (
        "gs://bucket/kestrel/agent-a/snapshots/20240101_120000.db" in keys
    )
    # All three lines parsed (latest.db is parsed but marked protected).
    assert len(records) == 3
    latest = next(r for r in records if r.name == "latest.db")
    assert latest.protected is True
    # Timestamps resolved from the ISO token, not None.
    assert all(r.timestamp is not None for r in records)
