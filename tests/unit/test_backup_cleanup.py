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
    for record in records:
        record.metadata["manifest_cid"] = f"manifest-{record.agent_id}"
        record.metadata["manifest_cid_kind"] = "snapshot"
    plan = backup_cleanup.build_delete_plan(
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
        quarantine_state={"objects": {}},
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

    async def get_deal_status(self, cid):
        return {"deals": [{"dealExpiry": "2030-01-01T00:00:00Z"}]}

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_lighthouse_pagination_consumes_empty_and_multi_page_responses(caplog):
    class FakeClient:
        def __init__(self):
            self.calls = []
            self.pages = {
                None: {
                    "fileList": [{"cid": "cid-1", "fileName": "one.db"}],
                    "nextLastKey": "cursor-1",
                    "totalFiles": 3,
                },
                "cursor-1": {
                    "fileList": [],
                    "nextLastKey": "cursor-empty",
                    "totalFiles": 3,
                },
                "cursor-empty": {
                    "fileList": [{"cid": "cid-2", "fileName": "two.db"}],
                    "lastKey": "cursor-2",
                    "totalFiles": 3,
                },
                "cursor-2": {
                    "fileList": [{"cid": "cid-3", "fileName": "three.db"}],
                    "totalFiles": 3,
                },
            }

        async def get_uploads(self, last_key=None):
            self.calls.append(last_key)
            return self.pages[last_key]

        async def download(self, cid, timeout=None):
            return b"{}"

        async def delete_file(self, cid):
            raise AssertionError("delete_file should not be called")

        async def close(self):
            pass

    client = FakeClient()

    with caplog.at_level("INFO"):
        uploads = await backup_cleanup._all_lighthouse_uploads(client)

    assert client.calls == [None, "cursor-1", "cursor-empty", "cursor-2"]
    assert [upload["cid"] for upload in uploads] == ["cid-1", "cid-2", "cid-3"]
    assert "Lighthouse total files seen: 3" in caplog.text


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


@pytest.mark.asyncio
async def test_manifest_index_attributes_export_car_and_rejects_malformed_manifest():
    class FakeClient:
        async def get_uploads(self, last_key=None):
            return {
                "fileList": [
                    {
                        "cid": "cid-export",
                        "fileName": "export.car",
                        "fileSizeInBytes": "300",
                    },
                    {
                        "cid": "cid-payload",
                        "fileName": "payload.bin",
                        "fileSizeInBytes": "200",
                    },
                    {
                        "cid": "cid-bad-export",
                        "fileName": "export.car",
                        "fileSizeInBytes": "300",
                    },
                    {
                        "cid": "cid-manifest-good",
                        "fileName": "kestrel_manifest__agent-a__20260620_120000.json",
                    },
                    {
                        "cid": "cid-manifest-bad",
                        "fileName": "manifest_agent-b.json",
                    },
                ],
                "totalFiles": 5,
            }

        async def download(self, cid, timeout=None):
            if cid == "cid-manifest-good":
                return json.dumps(
                    {
                        "agent_id": "agent-a",
                        "snapshot_cid": "cid-export",
                        "snapshot_payload_cid": "cid-payload",
                        "snapshot_format": "car-v1/raw-sqlite",
                        "uploaded_at": "2026-06-20T12:00:00Z",
                        "raw_snapshot_size": 123,
                        "source_file": "kestrel_prime.db",
                        "content_hash": "abc",
                    }
                ).encode()
            if cid == "cid-manifest-bad":
                return json.dumps(
                    {"agent_id": "agent-x", "snapshot_cid": "cid-bad-export"}
                ).encode()
            raise AssertionError(f"unexpected download {cid}")

        async def delete_file(self, cid):
            raise AssertionError("delete_file should not be called")

        async def close(self):
            pass

    records = await backup_cleanup.lighthouse_records(FakeClient())
    by_key = {record.key: record for record in records}

    assert by_key["cid-export"].agent_id == "agent-a"
    assert by_key["cid-export"].metadata["manifest_cid"] == "cid-manifest-good"
    assert by_key["cid-export"].metadata["manifest_cid_kind"] == "snapshot"
    assert by_key["cid-payload"].agent_id == "agent-a"
    assert by_key["cid-payload"].metadata["manifest_cid_kind"] == "snapshot_payload"
    assert by_key["cid-bad-export"].agent_id is None
    assert "manifest_cid_kind" not in by_key["cid-bad-export"].metadata


def test_inventory_classifier_assigns_expected_class_confidence_and_reason():
    records = [
        _record(
            "cid-export",
            "agent-a",
            1,
            store="lighthouse",
            name="export.car",
        ),
        _record(
            "cid-test",
            "did:test:agent",
            1,
            store="lighthouse",
            name="test-alpha.db",
        ),
        _record(
            "cid-db",
            None,
            1,
            store="lighthouse",
            name="kestrel_prime.db",
            attributed=False,
        ),
        _record(
            "cid-wal",
            None,
            1,
            store="lighthouse",
            name="kestrel_prime.db-wal",
            attributed=False,
        ),
        _record(
            "cid-bin",
            None,
            1,
            store="lighthouse",
            name="random.bin",
            attributed=False,
        ),
        _record(
            "cid-private",
            None,
            1,
            store="lighthouse",
            name="unknown.car",
            attributed=False,
        ),
    ]
    records[0].metadata["manifest_cid_kind"] = "snapshot"
    rows = backup_cleanup.classify_records(
        [backup_cleanup._with_test_flag(record) for record in records]
    )
    by_key = {row.record.key: row for row in rows}

    assert by_key["cid-export"].inventory_class == "attributed_snapshot"
    assert by_key["cid-export"].confidence == "high"
    assert "manifest index" in by_key["cid-export"].reason
    assert by_key["cid-test"].inventory_class == "test_proven_orphan"
    assert "fixture marker" in by_key["cid-test"].reason
    assert by_key["cid-db"].inventory_class == "legacy_private_candidate"
    assert by_key["cid-wal"].inventory_class == "legacy_private_candidate"
    assert by_key["cid-bin"].inventory_class == "unattributed_bin"
    assert by_key["cid-private"].inventory_class == "unattributed_private_candidate"


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


def test_manifest_agent_falls_back_to_filename_when_body_missing_agent_id():
    # Legacy manifest_<agent>.json with no agent_id in body -> attribute via filename.
    agent = backup_cleanup._valid_manifest_agent(
        {"snapshot_cid": "cid-1"},
        filename="manifest_did:pkh:eip155:1:0xABC.json",
    )
    assert agent == "did:pkh:eip155:1:0xABC"
    # Body present + matching filename still works.
    assert backup_cleanup._valid_manifest_agent(
        {"agent_id": "agent-a", "snapshot_cid": "cid-1"},
        filename="manifest_agent-a.json",
    ) == "agent-a"
    # Body vs filename mismatch is still rejected (provenance guard).
    assert backup_cleanup._valid_manifest_agent(
        {"agent_id": "agent-x", "snapshot_cid": "cid-1"},
        filename="manifest_agent-b.json",
    ) is None


def test_manifest_cid_entries_parses_legacy_and_collection_fields():
    # Historical scalar fields.
    entries = backup_cleanup._manifest_cid_entries(
        {"cid": "cid-legacy", "state_cid": "cid-state", "backup_cid": "cid-bak"}
    )
    assert {"cid-legacy", "cid-state", "cid-bak"} <= set(entries)
    # Collection fields: list of strings and list of {cid: ...} dicts.
    entries = backup_cleanup._manifest_cid_entries(
        {"snapshots": ["cid-a", {"cid": "cid-b"}], "snapshot_cids": ["cid-c"]}
    )
    assert {"cid-a", "cid-b", "cid-c"} <= set(entries)
    # A manifest using only legacy fields is still schema-valid (has a CID).
    assert backup_cleanup._manifest_schema_valid({"cid": "cid-legacy"}) is True
    # A manifest with no CID at all is invalid.
    assert backup_cleanup._manifest_schema_valid({"agent_id": "agent-a"}) is False


def _attributed_snapshot(key: str, agent_id: str, days_old: int):
    record = _record(key, agent_id, days_old, store="lighthouse", name=f"{key}.car")
    record.metadata["manifest_cid"] = f"manifest-{agent_id}"
    record.metadata["manifest_cid_kind"] = "snapshot"
    return record


def _expire_everything_policy():
    return RetentionPolicy.from_config(
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
    )


def test_delete_plan_refuses_raw_bin_and_unattributed_without_promotion():
    safe_old = _attributed_snapshot("cid-safe-old", "agent-a", 900)
    safe_new = _attributed_snapshot("cid-safe-new", "agent-a", 1)
    records = [
        safe_new,
        safe_old,
        _record(
            "cid-test",
            None,
            900,
            store="lighthouse",
            name="test-orphan.db",
            attributed=False,
        ),
        _record(
            "cid-raw",
            None,
            900,
            store="lighthouse",
            name="kestrel_prime.db",
            attributed=False,
        ),
        _record(
            "cid-wal",
            None,
            900,
            store="lighthouse",
            name="kestrel_prime.db-wal",
            attributed=False,
        ),
        _record(
            "cid-bin",
            None,
            900,
            store="lighthouse",
            name="payload.bin",
            attributed=False,
        ),
        _record(
            "cid-private",
            None,
            900,
            store="lighthouse",
            name="unknown.car",
            attributed=False,
        ),
    ]

    plan = backup_cleanup.build_delete_plan(
        [backup_cleanup._with_test_flag(record) for record in records],
        _expire_everything_policy(),
        quarantine_state={"objects": {}},
        now=NOW,
    )

    assert _delete_keys(plan) == {"cid-safe-old", "cid-test"}
    reasons = {row.record.key: row.reason for row in plan.records}
    assert reasons["cid-raw"].startswith("quarantine_required:")
    assert reasons["cid-wal"].startswith("quarantine_required:")
    assert reasons["cid-bin"].startswith("quarantine_required:")
    assert reasons["cid-private"].startswith("quarantine_required:")


def test_quarantine_writes_state_without_provider_delete(tmp_path):
    state_path = tmp_path / "quarantine.json"
    records = [
        _record(
            "cid-raw",
            None,
            1,
            store="lighthouse",
            name="kestrel_prime.db",
            attributed=False,
        ),
        _attributed_snapshot("cid-safe", "agent-a", 1),
    ]

    result = backup_cleanup.quarantine_records(
        backup_cleanup.classify_records(records),
        state_path=state_path,
    )

    state = json.loads(state_path.read_text())
    assert result["added"] == 1
    assert "lighthouse:cid-raw" in state["objects"]
    assert state["objects"]["lighthouse:cid-raw"]["status"] == "quarantined"
    assert "cid-safe" not in json.dumps(state)
    assert "provider_deletes: 0" in backup_cleanup.render_quarantine_report(result)


def test_promoted_quarantine_object_becomes_delete_eligible(tmp_path):
    state_path = tmp_path / "quarantine.json"
    raw = _record(
        "cid-raw",
        None,
        900,
        store="lighthouse",
        name="kestrel_prime.db",
        attributed=False,
    )
    backup_cleanup.quarantine_records(
        backup_cleanup.classify_records([raw]),
        state_path=state_path,
    )

    unpromoted = backup_cleanup.build_delete_plan(
        [raw],
        _expire_everything_policy(),
        quarantine_state=backup_cleanup.load_quarantine_state(state_path),
        now=NOW,
    )
    backup_cleanup.promote_quarantine_object(state_path, "lighthouse:cid-raw")
    promoted = backup_cleanup.build_delete_plan(
        [raw],
        _expire_everything_policy(),
        quarantine_state=backup_cleanup.load_quarantine_state(state_path),
        now=NOW,
    )

    assert _delete_keys(unpromoted) == set()
    assert _delete_keys(promoted) == {"cid-raw"}
    assert promoted.deletions[0].reason == "promoted_legacy_private_candidate"


@pytest.mark.asyncio
async def test_lighthouse_delete_audit_records_filecoin_caveat_and_pending_expiry(tmp_path):
    old = _attributed_snapshot("cid-old", "agent-a", 900)
    new = _attributed_snapshot("cid-new", "agent-a", 1)
    plan = backup_cleanup.build_delete_plan(
        [old, new],
        _expire_everything_policy(),
        quarantine_state={"objects": {}},
        now=NOW,
    )
    audit_log = tmp_path / "audit.jsonl"
    client = AsyncDeleteClient()

    rc = await backup_cleanup.apply_plan(
        plan,
        lighthouse_client=client,
        confirmation=backup_cleanup.CONFIRMATION_PHRASE,
        audit_log=audit_log,
        manifest_index_hash="abc123",
        policy_version="policy-test",
    )

    assert rc == 0
    assert client.deleted == ["cid-old"]
    entry = json.loads(audit_log.read_text().strip())
    assert entry["key"] == "cid-old"
    assert entry["delete_call_result"] == {"deleted": True}
    assert backup_cleanup.DEAL_IMMUTABILITY_CAVEAT in entry["deal_immutability_caveat"]
    assert entry["filecoin_status"] == (
        "deleted_from_account_but_deal_may_persist_until_expiry"
    )
    assert entry["deal_expiry"] == "2030-01-01T00:00:00Z"
    assert entry["manifest_index_hash"] == "abc123"
    assert entry["policy_version"] == "policy-test"


def test_delete_plan_preserves_newest_and_live_manifest_referenced_objects():
    live_manifest = _record(
        "manifest-live",
        "agent-a",
        1,
        store="lighthouse",
        data_class=DataClass.IDENTITY,
        name="manifest_agent-a.json",
    )
    live_manifest.metadata["manifest_cid"] = "manifest-live"
    live_snapshot = _attributed_snapshot("cid-live-snapshot", "agent-a", 900)
    live_snapshot.metadata["manifest_cid"] = "manifest-live"
    old_snapshot = _attributed_snapshot("cid-old-snapshot", "agent-a", 900)
    old_snapshot.metadata["manifest_cid"] = "manifest-old"
    newest_snapshot = _attributed_snapshot("cid-newest-snapshot", "agent-a", 1)
    newest_snapshot.metadata["manifest_cid"] = "manifest-old"

    plan = backup_cleanup.build_delete_plan(
        [live_manifest, live_snapshot, old_snapshot, newest_snapshot],
        _expire_everything_policy(),
        quarantine_state={"objects": {}},
        now=NOW,
    )
    reasons = {row.record.key: row.reason for row in plan.records}

    assert _delete_keys(plan) == {"cid-old-snapshot"}
    assert reasons["cid-live-snapshot"] == "live_manifest_referenced"
    assert reasons["cid-newest-snapshot"] == "newest"


@pytest.mark.asyncio
async def test_dry_run_preview_matches_apply_plan_for_promoted_quarantine(tmp_path):
    # Regression: the default dry-run preview must use the SAME quarantine-aware
    # build_delete_plan that --apply executes, so a promoted quarantine object
    # shown as a planned delete in preview is exactly what apply would remove.
    raw = _record(
        "gs://bucket/kestrel/orphan/kestrel_prime.db",
        None,
        900,
        store="gcs",
        name="kestrel_prime.db",
    )
    classified = backup_cleanup.classify_inventory_record(raw)
    assert classified.inventory_class in backup_cleanup.QUARANTINE_CLASSES

    state_path = tmp_path / "q.json"
    backup_cleanup.quarantine_records([classified], state_path=state_path)
    backup_cleanup.promote_quarantine_object(state_path, raw.key)

    args = backup_cleanup.build_parser().parse_args(
        [
            "--skip-lighthouse",
            "--gcs-bucket",
            "bucket",
            "--quarantine-state",
            str(state_path),
        ]
    )

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with patch("scripts.backup_cleanup.list_gcs_records", return_value=[raw]):
        with redirect_stdout(buf):
            rc = await backup_cleanup._main_async(args)

    assert rc == 0
    out = buf.getvalue()
    # Preview is the delete preflight (quarantine-aware), not the legacy report,
    # and the promoted quarantine class is shown as an eligible delete — exactly
    # what --apply would remove.
    assert "manifest-index hash" in out
    assert "legacy_private_candidate: 1 objects" in out

    # Negative control: with a FRESH (un-promoted) quarantine state, the same
    # object is NOT eligible — proving the preview reflects promotion state.
    fresh_state = tmp_path / "q_fresh.json"
    backup_cleanup.quarantine_records([classified], state_path=fresh_state)
    args2 = backup_cleanup.build_parser().parse_args(
        ["--skip-lighthouse", "--gcs-bucket", "bucket",
         "--quarantine-state", str(fresh_state)]
    )
    buf2 = io.StringIO()
    with patch("scripts.backup_cleanup.list_gcs_records", return_value=[raw]):
        with redirect_stdout(buf2):
            await backup_cleanup._main_async(args2)
    assert "legacy_private_candidate: 1 objects" not in buf2.getvalue()
