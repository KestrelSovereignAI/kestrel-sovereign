from datetime import datetime, timedelta, timezone

import httpx
import pytest

from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient
from kestrel_sovereign.storage.sync.health import (
    build_storage_health_report,
    check_gcs_health,
    check_lighthouse_health,
    check_sovereign_ipfs_health,
)


NOW = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)


class FakeLighthouseClient:
    uploads = []
    deals = []
    downloads = {}

    def __init__(self, api_key):
        self.api_key = api_key
        self.closed = False

    async def get_uploads(self, last_key=None):
        return {"fileList": self.uploads if last_key is None else []}

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
        export_api_key=None,
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
        snapshot_api_key="key",
        export_api_key="key",
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
        snapshot_api_key="key",
        export_api_key="key",
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
        snapshot_api_key="key",
        export_api_key="key",
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


def _lighthouse_api(records, deals):
    """Lighthouse's REST API over ``records`` (listed 1000 to a page, the
    cursor the last record's id) and ``deals`` (each CID's deal_status body,
    exactly as the endpoint returns it)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/user/files_uploaded":
            assert request.headers["Authorization"] == "Bearer key"
            last_key = request.url.params["lastKey"]
            ids = [record["id"] for record in records]
            start = 0 if last_key == "null" else ids.index(last_key) + 1
            return httpx.Response(
                200,
                json={
                    "fileList": records[start:start + 1000],
                    "totalFiles": len(records),
                },
            )
        if request.url.path == "/api/lighthouse/deal_status":
            return httpx.Response(200, json=deals[request.url.params["cid"]])
        raise AssertionError(f"unexpected Lighthouse request {request.url}")

    class Client(LighthouseRestClient):
        """The real REST client, its HTTP answered by ``handler``."""

        def __init__(self, api_key):
            super().__init__(api_key=api_key)
            self._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return Client


def _filler(index):
    return {
        "id": f"file-{index:05d}",
        "cid": f"QmFiller{index}",
        "fileName": f"kestrel_state__agent-2__20260101_{index % 60:02d}0000.car",
        "createdAt": int((NOW - timedelta(days=30)).timestamp() * 1000),
    }


def _export_record(index, cid, created_at):
    record = _export_upload(cid, "agent-1", created_at)
    record["id"] = f"file-{index:05d}"
    return record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "deal_count"),
    [
        # The SDK's documented shape: the deal list wrapped in "data".
        (
            {
                "data": [
                    {"DealID": 100046008, "Provider": 10479},
                    {"DealID": 100049432, "Provider": 97777},
                ]
            },
            2,
        ),
        # The endpoint's own list of deals.
        ([{"DealID": 100046008, "Provider": 10479}], 1),
        ([{"DealID": 100046008}, {"DealID": 100049432}], 2),
        (
            [
                {
                    "chainDealID": 22340843,
                    "dealId": 22340843,
                    "storageProvider": "f01173170",
                    "dealStatus": "Active",
                }
            ],
            1,
        ),
        # One deal record, unwrapped.
        ({"DealID": 100046008, "Provider": 10479}, 1),
    ],
)
async def test_lighthouse_counts_deals_in_every_deal_status_shape(body, deal_count):
    """``get_deal_status`` returns the endpoint's JSON as it is, a list or a
    dict. Health reads the deals in either, through the real REST client
    (#2874): a dict of deals is not zero deals."""
    created = NOW - timedelta(days=3)
    client = _lighthouse_api(
        [_export_record(0, "QmExport", created)], {"QmExport": body}
    )

    result = await check_lighthouse_health(
        snapshot_api_key="key",
        export_api_key="key",
        agent_id="agent-1",
        now=NOW,
        client_factory=client,
    )

    export = result.details["sovereignty_export"]
    assert (result.status, export["deal_count"]) == ("ok", deal_count)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [[], {"data": []}, {"dealStatus": "Queued"}])
async def test_lighthouse_counts_no_deals_where_the_response_names_none(body):
    created = NOW - timedelta(days=3)
    client = _lighthouse_api(
        [_export_record(0, "QmExport", created)], {"QmExport": body}
    )

    result = await check_lighthouse_health(
        snapshot_api_key="key",
        export_api_key="key",
        agent_id="agent-1",
        now=NOW,
        client_factory=client,
    )

    assert result.status == "warning"
    assert result.details["sovereignty_export"]["deal_count"] == 0


@pytest.mark.asyncio
async def test_lighthouse_health_reads_every_page_of_uploads():
    """The agent's latest export is on the second page of the account's
    uploads, behind an older one on the first (#2874)."""
    records = [_export_record(0, "QmOlder", NOW - timedelta(days=9))]
    records += [_filler(index) for index in range(1, 1000)]
    records.append(_export_record(1000, "QmLatest", NOW - timedelta(hours=2)))
    client = _lighthouse_api(records, {"QmLatest": {"data": []}, "QmOlder": []})

    result = await check_lighthouse_health(
        snapshot_api_key="key",
        export_api_key="key",
        agent_id="agent-1",
        now=NOW,
        client_factory=client,
    )

    assert result.status == "pending"
    assert result.details["sovereignty_export"]["cid"] == "QmLatest"


@pytest.mark.asyncio
async def test_lighthouse_missing_key_is_not_configured():
    result = await check_lighthouse_health(
        snapshot_api_key=None, export_api_key=None, agent_id="agent-1"
    )

    assert result.status == "not_configured"
    assert result.configured is False


def _snapshot_upload(cid, agent_id, created_at):
    return {
        "cid": cid,
        "fileName": (
            f"kestrel_state__{agent_id}__{created_at.strftime('%Y%m%d_%H%M%S')}.car"
        ),
        "createdAt": int(created_at.timestamp() * 1000),
    }


def _accounts(uploads_by_key, deals_by_cid, listed):
    """Lighthouse with one account per key; ``listed`` records each listing."""

    class Client(FakeLighthouseClient):
        async def get_uploads(self, last_key=None):
            listed.append(self.api_key)
            if last_key is not None:
                return {"fileList": []}
            return {"fileList": uploads_by_key.get(self.api_key, [])}

        async def get_deal_status(self, cid):
            return deals_by_cid.get(cid, [])

    return Client


@pytest.mark.asyncio
async def test_lighthouse_reads_snapshots_and_exports_on_their_own_accounts():
    """Snapshots are uploaded with the host's key and exports with the
    agent's, so each is read on its own account. An export of the agent's
    on the host account, and a snapshot on the agent's, are not the uploads
    those keys stand for (#2874)."""
    listed = []
    client = _accounts(
        {
            "host": [
                _snapshot_upload("QmSnapshot", "agent-1", NOW - timedelta(hours=1)),
                _export_upload("QmHostExport", "agent-1", NOW),
            ],
            "agent": [
                _export_upload("QmExport", "agent-1", NOW - timedelta(days=3)),
                _snapshot_upload("QmStray", "agent-1", NOW),
            ],
        },
        {"QmSnapshot": [{"DealID": 1}], "QmExport": [{"DealID": 2}]},
        listed,
    )

    result = await check_lighthouse_health(
        snapshot_api_key="host",
        export_api_key="agent",
        agent_id="agent-1",
        now=NOW,
        client_factory=client,
    )

    assert result.status == "ok"
    assert result.details["cid"] == "QmSnapshot"
    assert result.details["sovereignty_export"]["cid"] == "QmExport"
    assert sorted(listed) == ["agent", "host"]


@pytest.mark.asyncio
async def test_lighthouse_lists_an_account_both_keys_name_once():
    listed = []
    client = _accounts(
        {
            "host": [
                _snapshot_upload("QmSnapshot", "agent-1", NOW - timedelta(hours=1)),
                _export_upload("QmExport", "agent-1", NOW - timedelta(hours=2)),
            ]
        },
        {},
        listed,
    )

    result = await check_lighthouse_health(
        snapshot_api_key="host",
        export_api_key="host",
        agent_id="agent-1",
        now=NOW,
        client_factory=client,
    )

    assert result.status == "pending"
    assert result.details["cid"] == "QmSnapshot"
    assert result.details["sovereignty_export"]["cid"] == "QmExport"
    assert listed == ["host"]


@pytest.mark.asyncio
async def test_an_unresolvable_export_key_is_an_error_beside_the_snapshot():
    """The agent's export credential cannot be read. The tier is an error,
    and the snapshot on the host account is still reported (#2874)."""
    listed = []
    client = _accounts(
        {"host": [_snapshot_upload("QmSnapshot", "agent-1", NOW)]},
        {"QmSnapshot": [{"DealID": 1}]},
        listed,
    )

    result = await check_lighthouse_health(
        snapshot_api_key="host",
        export_api_key=None,
        export_key_error="the agent's Lighthouse key cannot be read",
        agent_id="agent-1",
        now=NOW,
        client_factory=client,
    )

    assert result.status == "error"
    assert result.message == (
        "the agent's Lighthouse key cannot be read; "
        "Latest Lighthouse snapshot has 1 Filecoin deal(s)"
    )
    assert result.details["cid"] == "QmSnapshot"
    assert result.details["sovereignty_export"] is None
    assert listed == ["host"]


@pytest.mark.asyncio
async def test_storage_health_report_reads_snapshots_with_the_host_key(monkeypatch):
    """The report reads snapshots with ``env``'s LIGHTHOUSE_API_KEY, the key
    the snapshot target uploads with, and exports with the agent's key."""
    listed = []
    client = _accounts(
        {
            "host": [_snapshot_upload("QmSnapshot", "agent-1", NOW)],
            "agent": [_export_upload("QmExport", "agent-1", NOW)],
        },
        {},
        listed,
    )
    monkeypatch.setattr(
        "kestrel_sovereign.storage.providers.lighthouse_rest.LighthouseRestClient",
        client,
    )

    report = await build_storage_health_report(
        agent_id="agent-1",
        export_api_key="agent",
        env={"LIGHTHOUSE_API_KEY": "host"},
        now=NOW,
    )

    assert report.lighthouse.details["cid"] == "QmSnapshot"
    assert report.lighthouse.details["sovereignty_export"]["cid"] == "QmExport"
    assert sorted(listed) == ["agent", "host"]


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
