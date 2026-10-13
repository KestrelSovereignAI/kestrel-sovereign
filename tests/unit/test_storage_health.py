from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.storage.sync.health import (
    build_storage_health_report,
    check_gcs_health,
    check_lighthouse_health,
    check_sovereign_ipfs_health,
    load_env_file,
)


NOW = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)


class FakeLighthouseClient:
    uploads = []
    deals = []
    downloads = {}

    def __init__(self, api_key):
        self.api_key = api_key
        self.closed = False

    async def get_uploads(self):
        return {"fileList": self.uploads}

    async def get_deal_status(self, cid):
        return self.deals

    async def download(self, cid):
        return self.downloads[cid]

    async def close(self):
        self.closed = True


class FakeFailingIpfsClient:
    def __init__(self, timeout):
        self.timeout = timeout
        self.closed = False

    async def post(self, url):
        raise OSError(f"cannot connect to {url}")

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_sovereign_ipfs_unset_reports_decommissioned():
    result = await check_sovereign_ipfs_health(api_url=None)

    assert result.name == "sovereign_ipfs"
    assert result.label == "sovereign-operated"
    assert result.status == "decommissioned"
    assert result.configured is False
    assert "decommissioned" in result.message


@pytest.mark.asyncio
async def test_sovereign_ipfs_unreachable_reports_decommissioned():
    result = await check_sovereign_ipfs_health(
        api_url="http://127.0.0.1:5001",
        client_factory=FakeFailingIpfsClient,
    )

    assert result.label == "sovereign-operated"
    assert result.status == "decommissioned"
    assert result.configured is True
    assert result.details["api_url"] == "http://127.0.0.1:5001"


@pytest.mark.asyncio
async def test_storage_health_report_uses_honest_tier_labels():
    report = await build_storage_health_report(
        agent_id="agent-1",
        env={},
        now=NOW,
    )

    data = report.to_dict()
    assert data["sovereign_ipfs"]["label"] == "sovereign-operated"
    assert data["sovereign_ipfs"]["status"] == "decommissioned"
    assert data["lighthouse"]["label"] == "delegated-decentralized"
    assert data["gcs"]["label"] == "expedient-cloud"


async def _lighthouse_status(uploads, deals, grace=timedelta(hours=24)):
    class Client(FakeLighthouseClient):
        pass

    Client.uploads = uploads
    Client.deals = deals
    Client.downloads = {}
    return await check_lighthouse_health(
        api_key="key",
        agent_id="agent-1",
        grace_period=grace,
        now=NOW,
        client_factory=Client,
    )


@pytest.mark.asyncio
async def test_lighthouse_active_deals_are_ok():
    result = await _lighthouse_status(
        uploads=[
            {
                "cid": "QmNew",
                "tag": "kestrel-state-agent-1",
                "createdAt": int(NOW.timestamp() * 1000),
            }
        ],
        deals=[{"DealID": 1, "Provider": 10479}],
    )

    assert result.status == "ok"
    assert result.details["cid"] == "QmNew"
    assert result.details["deal_count"] == 1


@pytest.mark.asyncio
async def test_lighthouse_health_finds_structured_snapshot_without_tag():
    result = await _lighthouse_status(
        uploads=[
            {
                "cid": "QmOther",
                "fileName": "kestrel_state__agent-2__20260510_120000.car",
                "tag": None,
                "createdAt": int(NOW.timestamp() * 1000),
            },
            {
                "cid": "QmNew",
                "fileName": "kestrel_state__agent-1__20260510_120000.car",
                "tag": None,
                "createdAt": int(NOW.timestamp() * 1000),
            },
        ],
        deals=[{"DealID": 1, "Provider": 10479}],
    )

    assert result.status == "ok"
    assert result.details["cid"] == "QmNew"


@pytest.mark.asyncio
async def test_lighthouse_fresh_no_deal_is_pending():
    result = await _lighthouse_status(
        uploads=[
            {
                "cid": "QmFresh",
                "tag": "kestrel-state-agent-1",
                "createdAt": int((NOW - timedelta(hours=2)).timestamp() * 1000),
            }
        ],
        deals=[],
    )

    assert result.status == "pending"
    assert result.details["age_seconds"] == 7200


@pytest.mark.asyncio
async def test_lighthouse_stale_no_deal_warns():
    result = await _lighthouse_status(
        uploads=[
            {
                "cid": "QmStale",
                "tag": "kestrel-state-agent-1",
                "createdAt": int((NOW - timedelta(days=2)).timestamp() * 1000),
            }
        ],
        deals=[],
    )

    assert result.status == "warning"
    assert "beyond grace" in result.message


@pytest.mark.asyncio
async def test_lighthouse_ignores_untagged_uploads_from_other_contexts():
    result = await _lighthouse_status(
        uploads=[
            {
                "cid": "QmOther",
                "fileName": "kestrel_prime.db",
                "tag": None,
                "createdAt": int(NOW.timestamp() * 1000),
            }
        ],
        deals=[{"DealID": 1}],
    )

    assert result.status == "unavailable"
    assert "No Lighthouse snapshot" in result.message


@pytest.mark.asyncio
async def test_lighthouse_resolves_snapshot_cid_from_manifest_when_tags_missing():
    class Client(FakeLighthouseClient):
        pass

    Client.uploads = [
        {
            "cid": "QmManifest",
            "fileName": "manifest_agent-1.json",
            "tag": None,
            "createdAt": int((NOW - timedelta(hours=1)).timestamp() * 1000),
        }
    ]
    Client.deals = []
    Client.downloads = {
        "QmManifest": (
            b'{"snapshot_cid":"QmSnapshot","uploaded_at":"2026-05-10T11:00:00+00:00"}'
        )
    }

    result = await check_lighthouse_health(
        api_key="key",
        agent_id="agent-1",
        now=NOW,
        client_factory=Client,
    )

    assert result.status == "pending"
    assert result.details["cid"] == "QmSnapshot"
    assert result.details["resolved_via"] == "manifest"


def _export_upload(cid, agent_id, created_at, *, tag="default"):
    """A sovereignty export's upload record (#2874)."""
    return {
        "cid": cid,
        "fileName": (
            f"kestrel_sovereignty__{agent_id}__"
            f"{created_at.strftime('%Y%m%d_%H%M%S')}.bin"
        ),
        "tag": f"kestrel-sovereignty-{agent_id}" if tag == "default" else tag,
        "createdAt": int(created_at.timestamp() * 1000),
    }


async def _lighthouse_status_with_deals(uploads, deals_by_cid):
    class Client(FakeLighthouseClient):
        async def get_deal_status(self, cid):
            return deals_by_cid.get(cid, [])

    Client.uploads = uploads
    Client.downloads = {}
    return await check_lighthouse_health(
        api_key="key",
        agent_id="agent-1",
        now=NOW,
        client_factory=Client,
    )


@pytest.mark.asyncio
async def test_lighthouse_reports_a_sovereignty_export_when_no_snapshot_exists():
    result = await _lighthouse_status_with_deals(
        uploads=[_export_upload("QmExport", "agent-1", NOW - timedelta(hours=2), tag=None)],
        deals_by_cid={},
    )

    assert result.status == "pending"
    assert result.message == (
        "Latest Lighthouse sovereignty export has no deals yet but is within grace"
    )
    assert result.details["resolved_via"] == "sovereignty_export"
    assert result.details["cid"] == "QmExport"
    export = result.details["sovereignty_export"]
    assert export["agent_id"] == "agent-1"
    assert export["cid"] == "QmExport"
    assert export["age_seconds"] == 7200


@pytest.mark.asyncio
async def test_lighthouse_reports_the_worse_of_snapshot_and_sovereignty_export():
    result = await _lighthouse_status_with_deals(
        uploads=[
            {
                "cid": "QmSnapshot",
                "fileName": "kestrel_state__agent-1__20260510_120000.car",
                "createdAt": int(NOW.timestamp() * 1000),
            },
            _export_upload("QmOldExport", "agent-1", NOW - timedelta(days=9)),
            _export_upload("QmExport", "agent-1", NOW - timedelta(days=3)),
        ],
        deals_by_cid={"QmSnapshot": [{"DealID": 1}]},
    )

    assert result.status == "warning"
    assert result.message == (
        "Latest Lighthouse snapshot has 1 Filecoin deal(s); "
        "latest sovereignty export has no Filecoin deals beyond grace"
    )
    assert result.details["cid"] == "QmSnapshot"
    assert result.details["sovereignty_export"]["cid"] == "QmExport"
    assert result.details["sovereignty_export"]["status"] == "warning"


@pytest.mark.asyncio
async def test_lighthouse_credits_no_agent_with_another_agents_sovereignty_export():
    result = await _lighthouse_status_with_deals(
        uploads=[
            _export_upload("QmOther", "agent-2", NOW),
            # Extends agent-1's id: a prefix match would credit it to agent-1.
            _export_upload("QmExtended", "agent-1__20260101_000000", NOW),
            # Named for agent-1 but tagged for agent-2: attributable to neither.
            _export_upload(
                "QmConflict", "agent-1", NOW, tag="kestrel-sovereignty-agent-2"
            ),
            {"cid": "QmLegacy", "fileName": "content.bin", "tag": "kestrel-sovereignty"},
        ],
        deals_by_cid={},
    )

    assert result.status == "unavailable"
    assert "No Lighthouse snapshot" in result.message


@pytest.mark.asyncio
async def test_lighthouse_missing_key_is_not_configured():
    result = await check_lighthouse_health(api_key=None, agent_id="agent-1")

    assert result.status == "not_configured"
    assert result.configured is False


class FakeBlob:
    def __init__(self, exists):
        self._exists = exists

    def exists(self):
        return self._exists


class FakeBucket:
    def __init__(self, latest_exists):
        self.latest_exists = latest_exists

    def blob(self, _name):
        return FakeBlob(self.latest_exists)


class FakeGCSTarget:
    healthy = True
    latest_exists = True

    def __init__(self, bucket, prefix, agent_id, project=None, credentials_path=None):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.agent_id = agent_id
        self.project = project
        self.credentials_path = credentials_path

    async def health_check(self):
        return self.healthy

    def _get_bucket(self):
        return FakeBucket(self.latest_exists)


@pytest.mark.asyncio
async def test_gcs_healthy_with_latest_snapshot_is_ok():
    class Target(FakeGCSTarget):
        healthy = True
        latest_exists = True

    result = await check_gcs_health(
        bucket="backup",
        agent_id="agent-1",
        target_factory=Target,
    )

    assert result.status == "ok"
    assert result.details["latest_exists"] is True


@pytest.mark.asyncio
async def test_gcs_healthy_without_latest_snapshot_warns():
    class Target(FakeGCSTarget):
        healthy = True
        latest_exists = False

    result = await check_gcs_health(
        bucket="backup",
        agent_id="agent-1",
        target_factory=Target,
    )

    assert result.status == "warning"
    assert result.details["latest_exists"] is False


@pytest.mark.asyncio
async def test_gcs_unavailable_when_bucket_unhealthy():
    class Target(FakeGCSTarget):
        healthy = False
        latest_exists = False

    result = await check_gcs_health(
        bucket="backup",
        agent_id="agent-1",
        target_factory=Target,
    )

    assert result.status == "unavailable"


def test_load_env_file_preserves_existing_values(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("LIGHTHOUSE_API_KEY=file-value\nGCS_BACKUP_BUCKET=backup\n")

    result = load_env_file(env_path, env={"LIGHTHOUSE_API_KEY": "existing"})

    assert result["LIGHTHOUSE_API_KEY"] == "existing"
    assert result["GCS_BACKUP_BUCKET"] == "backup"


def test_aggregate_status_counts_active_sovereign_as_healthy():
    from kestrel_sovereign.storage.sync.health import _aggregate_overall_status

    # Reachable sovereign-only config must be "ok", not "unavailable".
    assert _aggregate_overall_status(
        {"active", "not_configured", "not_configured"}
    ) == "ok"
    # Decommissioned + unconfigured everywhere is genuinely unavailable.
    assert _aggregate_overall_status(
        {"decommissioned", "not_configured", "not_configured"}
    ) == "unavailable"
    # Any error/warning dominates.
    assert _aggregate_overall_status({"active", "warning"}) == "warning"
    assert _aggregate_overall_status({"ok", "error"}) == "warning"
