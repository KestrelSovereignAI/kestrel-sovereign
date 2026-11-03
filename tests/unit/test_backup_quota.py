"""Quota-aware Lighthouse backups (#3894).

The host's Lighthouse account filled its 500 GiB plan (538.4 GB used against
536.9 GB) and every ``backup_snapshot`` upload failed as a bare ``ReadError``
for days. These tests cover the precheck that prunes before an upload would
cross the quota threshold, the failure that names the quota, the reading the
sync service keeps, and the health check that warns about it.

No test reaches Lighthouse: the account is ``FakeLighthouseAPI`` behind an
``httpx.MockTransport``, and any request a real transport would send fails.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from kestrel_sovereign.features.health.checks import (
    check_backup_quota,
    derive_overall_status,
    run_standard_checks,
)
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.storage.providers import lighthouse_rest
from kestrel_sovereign.storage.sync.lighthouse_target import LighthouseTarget
from kestrel_sovereign.storage.sync.quota import (
    DEFAULT_QUOTA_THRESHOLD,
    StorageQuota,
    load_quota_threshold,
)
from kestrel_sovereign.storage.sync.retention import RetentionPolicy
from kestrel_sovereign.storage.sync.service import SyncService

AGENT = "did:key:z6MkQuotaAgent"
OTHER_AGENT = "did:key:z6MkOtherAgent"
SNAPSHOT_TAG = f"kestrel-state-{AGENT}"
MANIFEST_TAG = f"kestrel-manifest-{AGENT}"
LIMIT_500_GIB = 536_870_912_000


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Any request a real httpx transport would send fails the test."""

    async def refuse(self, request):
        raise AssertionError(f"real network request attempted: {request.url}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _record(file_id: str, name: str, size: int, created: datetime) -> dict:
    return {
        "id": file_id,
        "cid": f"Qm{file_id}",
        "fileName": name,
        "fileSizeInBytes": str(size),
        "createdAt": _ms(created),
    }


def _snapshot_name(agent: str, moment: datetime) -> str:
    return f"kestrel_state__{agent}__{moment:%Y%m%d_%H%M%S}.car"


class FakeLighthouseAPI:
    """One Lighthouse account's HTTP API, in memory: ``user_data_usage``,
    ``files_uploaded``, ``delete_file`` and ``/api/v0/add``.

    ``dataUsed`` is the sum of the account's records unless ``used`` pins it.
    """

    def __init__(self, *, limit: int, files: list[dict], used: int | None = None):
        self.limit = limit
        self.files = list(files)
        self.pinned_used = used
        self.calls: list[str] = []
        self.snapshot_upload_error: Exception | None = None
        self.usage_error: Exception | None = None

    @property
    def used(self) -> int:
        if self.pinned_used is not None:
            return self.pinned_used
        return sum(int(record["fileSizeInBytes"]) for record in self.files)

    def names(self) -> set[str]:
        return {record["fileName"] for record in self.files}

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/api/user/user_data_usage":
            self.calls.append("usage")
            if self.usage_error is not None:
                raise self.usage_error
            return httpx.Response(
                200, json={"dataLimit": self.limit, "dataUsed": self.used}
            )
        if request.method == "GET" and path == "/api/user/files_uploaded":
            self.calls.append("list")
            first_page = request.url.params.get("lastKey") in (None, "null")
            return httpx.Response(
                200,
                json={
                    "fileList": list(self.files) if first_page else [],
                    "totalFiles": len(self.files),
                },
            )
        if request.method == "DELETE" and path == "/api/user/delete_file":
            file_id = request.url.params["id"]
            self.calls.append(f"delete:{file_id}")
            self.files = [r for r in self.files if r["id"] != file_id]
            return httpx.Response(200, json={"data": "Deleted"})
        if request.method == "POST" and path == "/api/v0/add":
            tag = request.url.params["tag"]
            self.calls.append(f"upload:{tag}")
            body = await request.aread()
            if tag == SNAPSHOT_TAG and self.snapshot_upload_error is not None:
                raise self.snapshot_upload_error
            file_id = f"new-{len(self.calls)}"
            self.files.append(
                _record(
                    file_id,
                    f"upload-{file_id}",
                    len(body),
                    datetime.now(timezone.utc),
                )
            )
            return httpx.Response(
                200, json={"Hash": f"Qm{file_id}", "Name": file_id, "Size": str(len(body))}
            )
        raise AssertionError(f"unexpected Lighthouse request {request.method} {request.url}")


def _client_class(api: FakeLighthouseAPI):
    class _FakeLighthouseClient(lighthouse_rest.LighthouseRestClient):
        # A failed patch must not reach the real service.
        UPLOAD_URL = "https://upload.lighthouse.invalid"
        API_URL = "https://api.lighthouse.invalid"

        async def _get_client(self) -> httpx.AsyncClient:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    transport=httpx.MockTransport(api.handle), timeout=self.timeout
                )
            return self._client

    return _FakeLighthouseClient


@pytest.fixture
def lighthouse(monkeypatch):
    """Install a fake account; returns a factory taking its limit and files."""

    def install(*, limit: int, files: list[dict], used: int | None = None):
        api = FakeLighthouseAPI(limit=limit, files=files, used=used)
        monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", _client_class(api))
        return api

    return install


def _agent_db(path: Path, payload_bytes: int = 4096) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x BLOB)")
    conn.execute("INSERT INTO t VALUES (?)", (b"\x5a" * payload_bytes,))
    conn.commit()
    conn.close()
    return path


def _target(tmp_path: Path, **kwargs) -> LighthouseTarget:
    return LighthouseTarget(
        api_key="test-key-not-real",
        agent_id=AGENT,
        state_dir=tmp_path,
        retention_policy=kwargs.pop("retention_policy", RetentionPolicy()),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# StorageQuota
# ---------------------------------------------------------------------------


def test_storage_quota_near_over_and_fit():
    at = datetime(2026, 11, 2, tzinfo=timezone.utc)
    quota = StorageQuota(used_bytes=89, limit_bytes=100, observed_at=at)

    assert quota.threshold == DEFAULT_QUOTA_THRESHOLD
    assert not quota.near_quota and not quota.over_quota
    assert not quota.would_cross_threshold(1)
    assert quota.would_cross_threshold(2)
    assert quota.fits(11) and not quota.fits(12)
    assert StorageQuota(used_bytes=90, limit_bytes=100, observed_at=at).near_quota
    over = StorageQuota(used_bytes=538_400_000_000, limit_bytes=536_900_000_000, observed_at=at)
    assert over.over_quota and over.near_quota
    assert over.describe() == "538.4 GB used of 536.9 GB limit (100.3%)"


def test_storage_quota_round_trips_and_refuses_malformed_records():
    quota = StorageQuota(
        used_bytes=5, limit_bytes=10, observed_at=datetime(2026, 11, 2), threshold=0.5
    )

    assert quota.observed_at.tzinfo is timezone.utc
    assert StorageQuota.from_dict(json.loads(json.dumps(quota.to_dict()))) == quota
    for malformed in (
        None,
        [],
        {"used_bytes": 5},
        {**quota.to_dict(), "limit_bytes": 0},
        {**quota.to_dict(), "used_bytes": "5"},
        {**quota.to_dict(), "observed_at": "yesterday"},
        {**quota.to_dict(), "threshold": 2},
    ):
        assert StorageQuota.from_dict(malformed) is None


@pytest.mark.parametrize("threshold", [0, -0.1, 1.5, float("nan"), True, "0.9"])
def test_a_threshold_outside_zero_to_one_is_refused(threshold):
    with pytest.raises(ValueError):
        LighthouseTarget(api_key="k", agent_id=AGENT, quota_threshold=threshold)


def test_the_threshold_comes_from_backup_quota(monkeypatch):
    import kestrel_sovereign.config as config

    monkeypatch.setattr(config, "load_section", lambda section: {})
    assert load_quota_threshold() == DEFAULT_QUOTA_THRESHOLD == 0.9

    monkeypatch.setattr(
        config, "load_section", lambda section: {"quota": {"threshold": 0.75}}
    )
    assert load_quota_threshold() == 0.75

    monkeypatch.setattr(
        config, "load_section", lambda section: {"quota": {"threshold": 90}}
    )
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        load_quota_threshold()


# ---------------------------------------------------------------------------
# The precheck prunes before uploading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_precheck_prunes_before_an_upload_that_would_cross_the_threshold(
    tmp_path, lighthouse, caplog
):
    now = datetime.now(timezone.utc)
    expired_a = now - timedelta(days=420)
    expired_b = now - timedelta(days=400)
    recent = now - timedelta(days=1)
    api = lighthouse(
        limit=10_000_000,
        files=[
            # Past the bounded working-memory retention: pruned.
            _record("old-a", _snapshot_name(AGENT, expired_a), 3_900_000, expired_a),
            _record("old-b", _snapshot_name(AGENT, expired_b), 3_900_000, expired_b),
            # The agent's current restore point: kept.
            _record("recent", _snapshot_name(AGENT, recent), 150_000, recent),
            # Another agent's snapshot on the shared host account: never touched.
            _record(
                "other", _snapshot_name(OTHER_AGENT, expired_a), 1_000_000, expired_a
            ),
        ],
    )
    # 8.95 MB of 10 MB: below 90%, which a 100 KB snapshot would cross.
    assert api.used == 8_950_000
    target = _target(tmp_path)

    with caplog.at_level("WARNING"):
        result = await target.sync_snapshot(
            _agent_db(tmp_path / "agent.db", payload_bytes=100_000)
        )

    assert result.success is True, result.error
    # Usage, then the prune (listing pages until an empty one, as
    # list_all_uploads does, then deleting), then usage again, and only then
    # the upload.
    assert api.calls[:3] == ["usage", "list", "list"]
    assert set(api.calls[3:5]) == {"delete:old-a", "delete:old-b"}
    assert api.calls[5:] == [
        "usage",
        f"upload:{SNAPSHOT_TAG}",
        f"upload:{MANIFEST_TAG}",
    ]
    assert result.metadata["prune_before_upload"]["deleted"] == 2
    assert _snapshot_name(AGENT, recent) in api.names()
    assert _snapshot_name(OTHER_AGENT, expired_a) in api.names()
    warning = next(
        record.getMessage()
        for record in caplog.records
        if "near quota before uploading" in record.getMessage()
    )
    assert "dataUsed 8.9 MB of dataLimit 10.0 MB (89.5%)" in warning
    assert "would cross 90% of dataLimit" in warning
    # The reading after the prune is the one the target keeps.
    assert target.storage_quota.used_bytes == 1_150_000
    assert target.storage_quota.limit_bytes == 10_000_000


@pytest.mark.asyncio
async def test_an_upload_well_below_the_threshold_prunes_nothing_first(
    tmp_path, lighthouse, caplog
):
    expired = datetime.now(timezone.utc) - timedelta(days=400)
    api = lighthouse(
        limit=LIMIT_500_GIB,
        files=[_record("old", _snapshot_name(AGENT, expired), 1_000_000, expired)],
    )
    target = _target(tmp_path)

    with caplog.at_level("WARNING"):
        result = await target.sync_snapshot(_agent_db(tmp_path / "agent.db"))

    assert result.success is True, result.error
    assert api.calls == ["usage", f"upload:{SNAPSHOT_TAG}", f"upload:{MANIFEST_TAG}"]
    assert "prune_before_upload" not in result.metadata
    assert "near quota" not in caplog.text
    assert target.storage_quota == StorageQuota(
        used_bytes=1_000_000,
        limit_bytes=LIMIT_500_GIB,
        observed_at=target.storage_quota.observed_at,
    )


@pytest.mark.asyncio
async def test_an_unreadable_usage_does_not_stop_the_upload(tmp_path, lighthouse, caplog):
    api = lighthouse(limit=LIMIT_500_GIB, files=[])
    api.usage_error = httpx.ConnectError("usage endpoint down")
    target = _target(tmp_path)

    with caplog.at_level("WARNING"):
        result = await target.sync_snapshot(_agent_db(tmp_path / "agent.db"))

    assert result.success is True, result.error
    assert api.calls == ["usage", f"upload:{SNAPSHOT_TAG}", f"upload:{MANIFEST_TAG}"]
    assert "Could not read Lighthouse storage usage" in caplog.text
    assert "ConnectError: usage endpoint down" in caplog.text
    assert target.storage_quota is None


@pytest.mark.asyncio
async def test_a_failed_prune_before_upload_does_not_stop_the_upload(
    tmp_path, lighthouse, caplog
):
    lighthouse(limit=10_000_000, files=[], used=9_500_000)

    class _FailingPolicy(RetentionPolicy):
        def deletions(self, items, *, now=None):
            raise RuntimeError("policy boom")

    target = _target(tmp_path, retention_policy=_FailingPolicy())

    with caplog.at_level("WARNING"):
        result = await target.sync_snapshot(_agent_db(tmp_path / "agent.db"))

    assert result.success is True, result.error
    assert result.metadata["prune_before_upload"] == {"error": "RuntimeError: policy boom"}
    assert "Retention prune before upload failed" in caplog.text


# ---------------------------------------------------------------------------
# A failed upload names the quota
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_upload_failed_for_lack_of_room_names_data_used_against_data_limit(
    tmp_path, lighthouse
):
    """The prod failure: 538.4 GB used of 536.9 GB, and the upload dies as an
    empty ``ReadError``."""
    api = lighthouse(limit=536_900_000_000, files=[], used=538_400_000_000)
    api.snapshot_upload_error = httpx.ReadError("")
    target = _target(tmp_path)

    result = await target.sync_snapshot(_agent_db(tmp_path / "agent.db"))

    assert result.success is False
    assert result.error.startswith(
        "StorageQuotaExceeded: Lighthouse account over quota: dataUsed 538.4 GB "
        "against dataLimit 536.9 GB (100.3%); a "
    )
    assert result.error.endswith("upload does not fit (upload failed: ReadError: )")
    # The precheck (whose prune found nothing of this agent's to delete), the
    # upload, and the reading taken after the failure.
    assert api.calls == ["usage", "list", f"upload:{SNAPSHOT_TAG}", "usage"]


@pytest.mark.asyncio
async def test_an_upload_failure_with_room_left_keeps_its_own_error(tmp_path, lighthouse):
    api = lighthouse(limit=LIMIT_500_GIB, files=[], used=1_000)
    api.snapshot_upload_error = httpx.ReadError("connection reset")
    target = _target(tmp_path)

    result = await target.sync_snapshot(_agent_db(tmp_path / "agent.db"))

    assert result.success is False
    assert result.error == "ReadError: connection reset"


@pytest.mark.asyncio
async def test_the_scheduled_backup_reports_the_quota_not_a_bare_read_error(
    tmp_path, lighthouse
):
    """What ``backup_snapshot`` logs for the failed target names the quota."""
    api = lighthouse(limit=536_900_000_000, files=[], used=538_400_000_000)
    api.snapshot_upload_error = httpx.ReadError("")
    db = _agent_db(tmp_path / "agent.db")
    service = SyncService(
        db_path=str(db),
        state_file=str(tmp_path / "agent.db.sync"),
        retention_policy=RetentionPolicy(),
    )
    service.add_target(_target(tmp_path))
    await service.start()
    feature = SchedulerFeature(SimpleNamespace(did=AGENT, _sync_service=service))

    payload = json.loads(await feature._handle_backup_snapshot({}))

    assert payload["success"] is False
    assert payload["reason_code"] == "BACKUP_ALL_TARGETS_FAILED"
    assert payload["error"].startswith(
        "backup_snapshot_failed: lighthouse (StorageQuotaExceeded: Lighthouse "
        "account over quota: dataUsed 538.4 GB against dataLimit 536.9 GB (100.3%)"
    )


# ---------------------------------------------------------------------------
# The sync service keeps the reading; the health check warns
# ---------------------------------------------------------------------------


async def _service_after_one_backup(tmp_path, lighthouse, *, used: int, limit: int):
    lighthouse(limit=limit, files=[], used=used)
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = _agent_db(tmp_path / "agent.db")
    service = SyncService(
        db_path=str(db),
        state_file=str(tmp_path / "agent.db.sync"),
        retention_policy=RetentionPolicy(),
    )
    service.add_target(_target(tmp_path))
    await service.start()
    await service.force_snapshot()
    return service


@pytest.mark.asyncio
async def test_an_over_quota_account_produces_the_health_warning(tmp_path, lighthouse):
    service = await _service_after_one_backup(
        tmp_path, lighthouse, used=538_400_000_000, limit=536_900_000_000
    )

    check = await check_backup_quota(SimpleNamespace(_sync_service=service))

    assert check["name"] == "backup_quota"
    assert check["status"] == "warn"
    assert check["message"].startswith(
        f"Backup target near or over quota: lighthouse://{AGENT} is over quota: "
        "538.4 GB used of 536.9 GB limit (100.3%), threshold 90%, read "
    )
    target = check["details"]["targets"][f"lighthouse://{AGENT}"]
    assert target["over_quota"] is True and target["near_quota"] is True
    assert check["details"]["state"] == "near_or_over_quota"
    # A full backup account degrades the report; it never makes it unhealthy.
    assert derive_overall_status([check]) == "degraded"


@pytest.mark.asyncio
async def test_a_near_quota_account_warns_and_one_with_room_passes(tmp_path, lighthouse):
    near = await _service_after_one_backup(
        tmp_path / "near", lighthouse, used=92_000_000, limit=100_000_000
    )
    roomy = await _service_after_one_backup(
        tmp_path / "roomy", lighthouse, used=10_000_000, limit=100_000_000
    )

    near_check = await check_backup_quota(SimpleNamespace(_sync_service=near))
    roomy_check = await check_backup_quota(SimpleNamespace(_sync_service=roomy))

    assert near_check["status"] == "warn"
    assert f"lighthouse://{AGENT} is near quota" in near_check["message"]
    assert roomy_check["status"] == "pass"
    assert roomy_check["message"].startswith("Backup target quota OK: ")
    assert roomy_check["details"]["state"] == "within_quota"


@pytest.mark.asyncio
async def test_the_warning_survives_a_restart_before_the_next_backup(tmp_path, lighthouse):
    """The reading is kept in the sync state file, so a restarted host warns
    at once rather than after its next four-hourly backup."""
    await _service_after_one_backup(
        tmp_path, lighthouse, used=538_400_000_000, limit=536_900_000_000
    )
    restarted = SyncService(
        db_path=str(tmp_path / "agent.db"),
        state_file=str(tmp_path / "agent.db.sync"),
        retention_policy=RetentionPolicy(),
    )
    restarted.add_target(_target(tmp_path))
    await restarted.start()

    assert restarted.targets and restarted._targets[0].storage_quota is None
    check = await check_backup_quota(SimpleNamespace(_sync_service=restarted))

    assert check["status"] == "warn"
    assert "is over quota: 538.4 GB used of 536.9 GB limit" in check["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent",
    [
        SimpleNamespace(),
        SimpleNamespace(_sync_service=None),
        MagicMock(),
    ],
    ids=["no-sync-attribute", "no-sync-service", "mock-agent"],
)
async def test_without_a_quota_reading_the_check_passes(agent):
    check = await check_backup_quota(agent)

    assert check["name"] == "backup_quota"
    assert check["status"] == "pass"


@pytest.mark.asyncio
async def test_a_target_that_never_read_its_quota_is_not_reported(tmp_path):
    service = SyncService(
        db_path=str(tmp_path / "agent.db"),
        state_file=str(tmp_path / "agent.db.sync"),
        retention_policy=RetentionPolicy(),
    )
    service.add_target(_target(tmp_path))
    await service.start()

    check = await check_backup_quota(SimpleNamespace(_sync_service=service))

    assert check["status"] == "pass"
    assert check["details"]["state"] == "not_observed"


@pytest.mark.asyncio
async def test_an_unreadable_quota_report_warns():
    def boom():
        raise RuntimeError("state unreadable")

    check = await check_backup_quota(
        SimpleNamespace(_sync_service=SimpleNamespace(storage_quotas=boom))
    )

    assert check["status"] == "warn"
    assert "state unreadable" in check["message"]


@pytest.mark.asyncio
async def test_every_health_surface_runs_the_backup_quota_check(tmp_path, monkeypatch):
    at = datetime.now(timezone.utc)
    over = StorageQuota(used_bytes=110, limit_bytes=100, observed_at=at)
    agent = SimpleNamespace(
        _sync_service=SimpleNamespace(storage_quotas=lambda: {"lighthouse://x": over}),
        bootstrap_service=None,
        context_manager=None,
        dispatcher=None,
        llm_service=None,
        memory_retriever=None,
        memory_consolidator=None,
        storage=None,
    )

    checks = await run_standard_checks(agent, None)

    (backup_quota,) = [check for check in checks if check["name"] == "backup_quota"]
    assert backup_quota["status"] == "warn"
    assert "lighthouse://x is over quota" in backup_quota["message"]


def test_the_persisted_reading_ignores_malformed_state(tmp_path):
    service = SyncService(
        db_path=str(tmp_path / "agent.db"),
        state_file=str(tmp_path / "agent.db.sync"),
        retention_policy=RetentionPolicy(),
    )
    target = _target(tmp_path)
    service.add_target(target)
    from kestrel_sovereign.storage.sync.service import SyncState

    service._state = SyncState(db_path=str(tmp_path / "agent.db"), targets=["not", "a", "dict"])
    assert service.storage_quotas() == {}

    target.storage_quota = StorageQuota(
        used_bytes=1, limit_bytes=2, observed_at=datetime.now(timezone.utc)
    )
    service._record_storage_quota(target)
    assert service._state.targets == {target.name: {"storage_quota": target.storage_quota.to_dict()}}


def test_boot_builds_the_lighthouse_target_with_the_services_policy_and_threshold(
    tmp_path, monkeypatch
):
    """The prune before an upload applies the same policy as the prune after
    it, and the threshold is the host's ``[backup.quota]``."""
    import kestrel_sovereign.config as config
    from kestrel_sovereign.kestrel_agent import _add_lighthouse_snapshot_target

    monkeypatch.setenv("LIGHTHOUSE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        config,
        "load_section",
        lambda section: {"quota": {"threshold": 0.8}} if section == "backup" else {},
    )
    policy = RetentionPolicy()
    service = SyncService(db_path=str(tmp_path / "agent.db"), retention_policy=policy)

    assert _add_lighthouse_snapshot_target(
        service, storage_permitted=True, agent_id=AGENT, state_dir=tmp_path
    )

    (target,) = service._targets
    assert isinstance(target, LighthouseTarget)
    assert target._retention() is policy
    assert target._quota_threshold == 0.8
