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
