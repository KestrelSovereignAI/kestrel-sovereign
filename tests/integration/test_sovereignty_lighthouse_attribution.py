"""Sovereignty exports on Lighthouse are attributed to their agent (#2874).

An export the health check cannot see is one nobody notices missing, and an
export a retention sweep cannot attribute is one it can delete without knowing
whose restore point it destroyed. Every test here goes through the entry
points: ``SovereigntyFeature.export_sovereignty``, ``check_sovereignty_status``
and ``import_sovereignty``, ``build_storage_health_report``,
``LighthouseTarget.prune`` and the backlog cleanup tool in
``scripts/backup_cleanup.py``.

Lighthouse keeps one account per API key, behind the REST client core uses.
Each agent's Lighthouse provider is built the way boot builds it, from its
PayerPolicy resolution: an agent-scoped key in its encrypted key storage, or
else the host ``LIGHTHOUSE_API_KEY``. Agents that resolve the host key share
its account, which is what makes attribution matter; an agent with its own key
publishes to another account, which every reader and pruner of its exports
must then use. The adapter's own IPFS node is offline unless a test brings it
up: the agent's Lighthouse provider then takes the IPFS tier.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import zlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from cryptography.fernet import Fernet

from kestrel_sdk.payer_policy import PayerPolicy, ResourceClass
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.sovereignty.feature import SovereigntyFeature
from kestrel_sovereign.filecoin_adapter import ENVELOPE_MAGIC, FilecoinAdapter, StorageTier
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
from kestrel_sovereign.services.payer_resolver import FoundationPayerResolver
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.privacy_wrapper import (
    DurablePersistenceGate,
    PrivacyEnforcingStorage,
)
from kestrel_sovereign.storage.providers.lighthouse_provider import lighthouse_provider_for
from kestrel_sovereign.storage.sync.health import build_storage_health_report
from kestrel_sovereign.storage.sync.lighthouse_target import LighthouseTarget
from kestrel_sovereign.storage.sync.retention import RetentionPolicy
from scripts import backup_cleanup

AGENT_A = "did:pkh:eip155:1:0x2874alpha"
AGENT_B = "did:pkh:eip155:1:0x2874bravo"
# A DID that extends AGENT_A's past a "__": a prefix match on AGENT_A's
# upload names would take this agent's exports for AGENT_A's.
AGENT_A_EXTENDED = AGENT_A + "__20260101_000000"
# The host's LIGHTHOUSE_API_KEY, and one agent's own Lighthouse key.
API_KEY = "shared-host-key"
AGENT_KEY = "agent-alpha-own-key"
START = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
MESSAGES = [
    ("user", "my sister's name is Ingrid"),
    ("assistant", "I'll remember that your sister is Ingrid."),
    ("user", "and the spare key is under the blue pot"),
]
# Keep only the newest of each kind, so a prune that saw another agent's
# export as its own would delete it.
NEWEST_ONLY = RetentionPolicy.from_config(
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


class _Lighthouse:
    """Lighthouse: one account per API key, and one clock."""

    def __init__(self):
        self.now = START
        self.accounts: dict[str, _LighthouseAccount] = {}
        self.ids = itertools.count(1)

    def account(self, api_key: str) -> "_LighthouseAccount":
        return self.accounts.setdefault(api_key, _LighthouseAccount(self))

    def client(self, api_key=None, gateway_url=None, timeout=60.0):
        assert api_key in self.accounts, f"no Lighthouse account for key {api_key!r}"
        return _LighthouseClient(self.accounts[api_key])


class _LighthouseAccount:
    """One Lighthouse account: what ``LighthouseRestClient`` reads and writes."""

    def __init__(self, service: _Lighthouse):
        self.service = service
        self.files: list[dict] = []
        self.blobs: dict[str, bytes] = {}
        self.deals: dict[str, list] = {}
        self.fail_deletes = False

    @property
    def now(self) -> datetime:
        return self.service.now

    @now.setter
    def now(self, value: datetime) -> None:
        self.service.now = value

    def client(self, api_key=None, gateway_url=None, timeout=60.0):
        return self.service.client(api_key)

    def add(self, content: bytes, filename: str, tag, *, created_at=None) -> dict:
        cid = "bafylh" + hashlib.sha256(content).hexdigest()[:40]
        record = {
            "id": f"file-{next(self.service.ids)}",
            "cid": cid,
            "fileName": filename,
            "tag": tag,
            "createdAt": int((created_at or self.now).timestamp() * 1000),
            "fileSizeInBytes": str(len(content)),
        }
        self.files.append(record)
        self.blobs[cid] = content
        return record

    def cids(self) -> set[str]:
        return {record["cid"] for record in self.files}

    def record(self, cid: str) -> dict:
        [record] = [record for record in self.files if record["cid"] == cid]
        return record


class _LighthouseClient:
    def __init__(self, account: _LighthouseAccount):
        self.account = account

    async def upload(self, content, filename, tag="kestrel-storage", on_progress=None):
        if on_progress is not None:
            on_progress(len(content), len(content))
        record = self.account.add(content, filename, tag)
        return {"Hash": record["cid"], "Name": filename, "Size": str(len(content))}

    async def get_uploads(self, last_key=None):
        files = self.account.files
        start = 0
        if last_key not in (None, "null"):
            start = [record["id"] for record in files].index(last_key) + 1
        return {
            "fileList": [dict(record) for record in files[start:start + 1000]],
            "totalFiles": len(files),
        }

    async def delete_file(self, file_id):
        if self.account.fail_deletes:
            raise httpx.HTTPStatusError(
                "delete_file failed",
                request=httpx.Request("DELETE", "https://api.lighthouse.storage"),
                response=httpx.Response(503),
            )
        self.account.files = [r for r in self.account.files if r["id"] != file_id]
        return {"deleted": True}

    async def get_deal_status(self, cid):
        return list(self.account.deals.get(cid, []))

    async def download(self, cid, timeout=None, *, expected_bytes=None):
        return self.account.blobs[cid]

    async def get_balance(self):
        return {"data": {"dataUsed": "0"}}

    async def close(self):
        pass


class _Response:
    def __init__(self, status_code, *, body=b"", json_body=None):
        self.status_code = status_code
        self.content = body
        self.headers = {"Content-Length": str(len(body))}
        self._json = json_body

    def json(self):
        return self._json

    def iter_content(self, chunk_size=65536):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]

    def close(self):
        pass


class _Kubo:
    """The adapter's own IPFS node: offline unless a test brings it up."""

    def __init__(self):
        self.online = False
        self.blocks: dict[str, bytes] = {}
        self.upload_names: list[str] = []

    def post(self, url, *, files=None, params=None, timeout=None, stream=False):
        if not self.online:
            raise ConnectionError("IPFS daemon unreachable")
        rpc = url.split("/api/v0/", 1)[1]
        if rpc == "version":
            return _Response(200, json_body={"Version": "0.0.0-test"})
        if rpc == "add":
            name, payload = files["file"][0], files["file"][1]
            data = payload.read() if hasattr(payload, "read") else payload
            cid = "bafykubo" + hashlib.sha256(data).hexdigest()[:40]
            self.blocks[cid] = data
            self.upload_names.append(name)
            return _Response(200, json_body={"Hash": cid})
        if rpc == "cat":
            data = self.blocks.get(params["arg"])
            if data is None:
                return _Response(500, body=b"block not found")
            return _Response(200, body=data)
        raise AssertionError(f"unexpected Kubo RPC {rpc}")


@pytest.fixture
def data_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("KESTREL_DATA_KEY", key)
    return key


@pytest.fixture
def kubo():
    node = _Kubo()
    with patch("kestrel_sovereign.filecoin_adapter.requests.post", side_effect=node.post):
        yield node


@pytest.fixture
def lighthouse(monkeypatch):
    """The host account, behind the host's LIGHTHOUSE_API_KEY. Other accounts
    are ``lighthouse.service.account(key)``."""
    service = _Lighthouse()
    host = service.account(API_KEY)
    monkeypatch.setenv("LIGHTHOUSE_API_KEY", API_KEY)
    with patch(
        "kestrel_sovereign.storage.providers.lighthouse_provider.LighthouseRestClient",
        side_effect=service.client,
    ), patch(
        "kestrel_sovereign.storage.providers.lighthouse_rest.LighthouseRestClient",
        side_effect=service.client,
    ):
        yield host


@pytest.fixture(autouse=True)
def newest_only(monkeypatch):
    """The backup retention policy an export's thinning reads."""
    monkeypatch.setattr(
        "kestrel_sovereign.features.sovereignty.feature.load_retention_policy",
        lambda: NEWEST_ONLY,
    )


@asynccontextmanager
async def _host(directory, monkeypatch, *, did, lighthouse=True, agent_key=None):
    """One agent on a host. Agents co-hosted in one directory share its
    ``storage_cache`` and the host's Lighthouse key, and keep their
    databases. ``agent_key`` is stored in the agent's own encrypted key
    storage, as ``!add-key`` or a SELF_WALLET mint stores it."""
    directory.mkdir(exist_ok=True)
    monkeypatch.chdir(directory)
    database = f"{hashlib.sha256(did.encode()).hexdigest()[:12]}.db"
    raw = Storage(str(directory / database), agent_id=did)
    await raw.initialize()
    keys = ServiceKeyStorage(raw.db, did)
    if agent_key is not None and not await keys.has_key("lighthouse"):
        await keys.store_key("lighthouse", agent_key)
    provider = None
    if lighthouse:
        # Built only when policy permits a remote tier, as boot builds it.
        resolved = await FoundationPayerResolver(
            PayerPolicy.host_env_default(), db=raw.db
        ).resolve_for(did, ResourceClass.STORAGE)
        provider = await lighthouse_provider_for(resolved)
    wallet = MagicMock()
    wallet.can_afford.return_value = True
    wallet.transfer = AsyncMock()
    gate = DurablePersistenceGate()
    agent = SimpleNamespace(
        agent_id=did,
        storage=PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL),
        features={},
        wallet=wallet,
        _get_durable_persistence_gate=lambda: gate,
        lighthouse_provider=provider,
    )
    try:
        yield agent, raw
    finally:
        await raw.close()


async def _export_result(agent, raw):
    for role, content in MESSAGES:
        await raw.add_conversation(role, content, metadata={})
    result = await SovereigntyFeature(agent).export_sovereignty(
        storage_tier="ipfs", encrypt=True
    )
    assert result.status is ToolResultStatus.OK, result
    assert result.data["tier"] == "ipfs"
    return result


async def _export(agent, raw) -> str:
    return (await _export_result(agent, raw)).data["cid"]


async def _status(agent) -> dict:
    """What the agent's own health check reports for its Lighthouse export."""
    result = await SovereigntyFeature(agent).check_sovereignty_status()
    assert result.status is ToolResultStatus.OK, result
    return result.data["lighthouse"]


async def _history(raw):
    return [(m["role"], m["content"]) for m in await raw.get_conversation_history()]


def _health(agent_id, account, *, after=timedelta(hours=1)):
    """``kestrel storage health``: the host LIGHTHOUSE_API_KEY account."""
    return build_storage_health_report(
        agent_id=agent_id,
        env={"LIGHTHOUSE_API_KEY": API_KEY},
        now=account.now + after,
    )


async def _purge_everything(account, audit_log):
    """The account-wide age purge, told to delete everything it may."""
    return await backup_cleanup.age_purge_lighthouse(
        account.client(API_KEY),
        days=0,
        keep_cids=set(),
        audit_log=audit_log,
        confirmation=backup_cleanup.CONFIRMATION_PHRASE,
        now=account.now + timedelta(days=365),
    )


@pytest.mark.asyncio
async def test_export_is_found_by_the_health_check_and_attributed_to_its_agent(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    async with _host(tmp_path / "host", monkeypatch, did=AGENT_A) as (agent, raw):
        cid = await _export(agent, raw)

        # Published through the agent's provider, named for the agent alone.
        record = lighthouse.record(cid)
        assert re.fullmatch(
            rf"kestrel_sovereignty__{re.escape(AGENT_A)}__\d{{8}}_\d{{6}}\.bin",
            record["fileName"],
        )
        assert record["tag"] == f"kestrel-sovereignty-{AGENT_A}"
        [receipt] = await agent.storage.get_nodes_by_type("sovereignty_receipt")
        assert receipt.properties["provider"] == "lighthouse"
        assert receipt.properties["storage_tier"] == "ipfs"
        # The record names the agent and a time; the payload is the sealed
        # envelope. No history reaches Lighthouse in the clear (Amendment II).
        payload = zlib.decompress(lighthouse.blobs[cid])
        assert payload.startswith(ENVELOPE_MAGIC)
        assert b"blue pot" not in payload

        # The health check finds it, for this agent, before any deal lands.
        report = await _health(AGENT_A, lighthouse)
        assert report.lighthouse.status == "pending", report.lighthouse
        assert report.lighthouse.details["resolved_via"] == "sovereignty_export"
        assert report.lighthouse.details["cid"] == cid
        export = report.lighthouse.details["sovereignty_export"]
        assert export["agent_id"] == AGENT_A
        assert export["cid"] == cid
        assert export["file_name"] == record["fileName"]
        assert "sovereignty export" in report.lighthouse.message

        lighthouse.deals[cid] = [{"DealID": 1, "Provider": 10479}]
        settled = await _health(AGENT_A, lighthouse)
        assert settled.lighthouse.status == "ok"
        assert settled.lighthouse.details["sovereignty_export"]["deal_count"] == 1

        # So does the agent's own check, on the account it published to.
        status = await _status(agent)
        assert status["status"] == "ok"
        assert status["cid"] == cid
        assert status["agent_id"] == AGENT_A
        assert status["deal_count"] == 1

        # Another agent sharing the key is not credited with it.
        other = await _health(AGENT_B, lighthouse)
        assert other.lighthouse.status == "unavailable"

        # And the export restores on its host from the receipt it recorded.
        await raw.db.execute_commit("DELETE FROM conversation_history")
        restored = await SovereigntyFeature(agent).import_sovereignty(cid)
        assert restored.status is ToolResultStatus.OK, restored
        assert await _history(raw) == MESSAGES


@pytest.mark.asyncio
async def test_cleanup_for_one_agent_never_touches_another_agents_export(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    host = tmp_path / "host"
    exports: dict[str, list[str]] = {AGENT_A: [], AGENT_B: [], AGENT_A_EXTENDED: []}
    thinned: dict[str, list[str]] = {}
    for did in (AGENT_A, AGENT_B, AGENT_A_EXTENDED, AGENT_A, AGENT_B):
        lighthouse.now += timedelta(days=1)
        async with _host(host, monkeypatch, did=did) as (agent, raw):
            result = await _export_result(agent, raw)
        exports[did].append(result.data["cid"])
        thinned.setdefault(did, []).extend(result.data["export_retention"]["cids"])

    # Each agent's export thinned that agent's older export, through the
    # provider that published it, and nothing of anyone else's: not the
    # agent whose id extends AGENT_A's, which shares the account.
    assert thinned == {
        AGENT_A: [exports[AGENT_A][0]],
        AGENT_B: [exports[AGENT_B][0]],
        AGENT_A_EXTENDED: [],
    }
    assert lighthouse.cids() == {
        exports[AGENT_A][1],
        exports[AGENT_B][1],
        *exports[AGENT_A_EXTENDED],
    }
    survivors = lighthouse.cids()

    # The snapshot target's retention, for either agent, deletes no export.
    for did in (AGENT_A, AGENT_B):
        await LighthouseTarget(api_key=API_KEY, agent_id=did).prune(NEWEST_ONLY)
        assert lighthouse.cids() == survivors

    # Each agent's health check still finds its own latest export.
    for did in (AGENT_A, AGENT_B, AGENT_A_EXTENDED):
        report = await _health(did, lighthouse)
        assert report.lighthouse.details["sovereignty_export"]["cid"] == exports[did][-1]

    # The backlog cleanup attributes each export to its one agent and deletes
    # none, even when told to clear everything it has quarantined.
    records = await backup_cleanup.lighthouse_records(lighthouse.client(API_KEY))
    classified = {
        row.record.key: row for row in backup_cleanup.classify_records(records)
    }
    for did in exports:
        for cid in exports[did]:
            if cid in survivors:
                assert classified[cid].record.agent_id == did
                assert classified[cid].inventory_class == "attributed_sovereignty_export"
    plan = backup_cleanup.build_delete_plan(
        records,
        NEWEST_ONLY,
        quarantine_state={"objects": {}},
        include_quarantined=True,
    )
    assert plan.deletions == ()

    # The account-wide age purge keeps every agent's exports too.
    assert await _purge_everything(lighthouse, tmp_path / "age-purge.jsonl") == 0
    assert lighthouse.cids() == survivors


@pytest.mark.asyncio
async def test_an_unattributable_legacy_upload_is_retained_by_cleanup(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    # An upload in the shape a sovereignty export carried before #2874: no
    # agent in its name or tag. And one whose name and tag disagree.
    legacy = lighthouse.add(b"legacy-export", "content.bin", "kestrel-sovereignty")
    conflicting = lighthouse.add(
        b"conflicting-export",
        f"kestrel_sovereignty__{AGENT_A}__20260101_000000.bin",
        f"kestrel-sovereignty-{AGENT_B}",
    )
    unattributable = {legacy["cid"], conflicting["cid"]}
    for _ in range(2):
        lighthouse.now += timedelta(days=1)
        async with _host(tmp_path / "host", monkeypatch, did=AGENT_A) as (agent, raw):
            await _export(agent, raw)
    # Agent A's own thinning, by the newest-only policy, kept both.
    assert unattributable <= lighthouse.cids()

    for did in (AGENT_A, AGENT_B):
        await LighthouseTarget(api_key=API_KEY, agent_id=did).prune(NEWEST_ONLY)
        assert unattributable <= lighthouse.cids()

    # Neither is credited to an agent by the health check.
    report = await _health(AGENT_B, lighthouse)
    assert report.lighthouse.status == "unavailable"
    report = await _health(AGENT_A, lighthouse)
    assert report.lighthouse.details["sovereignty_export"]["cid"] not in unattributable

    # The backlog cleanup attributes neither and keeps both, by default and
    # when told to clear everything it has quarantined.
    records = await backup_cleanup.lighthouse_records(lighthouse.client(API_KEY))
    by_key = {record.key: record for record in records}
    assert by_key[legacy["cid"]].agent_id is None
    assert by_key[conflicting["cid"]].agent_id is None
    for include_quarantined in (False, True):
        plan = backup_cleanup.build_delete_plan(
            records,
            NEWEST_ONLY,
            quarantine_state={"objects": {}},
            include_quarantined=include_quarantined,
        )
        assert not unattributable & {row.record.key for row in plan.deletions}

    # And the account-wide age purge, told to delete everything older than
    # zero days, keeps both.
    assert await _purge_everything(lighthouse, tmp_path / "age-purge.jsonl") == 0
    assert unattributable <= lighthouse.cids()


@pytest.mark.asyncio
async def test_an_agent_with_its_own_key_and_no_host_key_is_checked_and_thinned_on_its_account(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """An agent-scoped key (``!add-key``, a SELF_WALLET mint) with no host
    LIGHTHOUSE_API_KEY: the export, its health and its thinning all use the
    account that key names (#2874)."""
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")
    own = lighthouse.service.account(AGENT_KEY)
    host = tmp_path / "host"

    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        first = await _export(agent, raw)
        assert own.cids() == {first}
        own.deals[first] = [{"DealID": 1}]
        status = await _status(agent)
        assert (status["status"], status["cid"], status["agent_id"]) == (
            "ok",
            first,
            AGENT_A,
        )

    lighthouse.now += timedelta(days=1)
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        result = await _export_result(agent, raw)
        second = result.data["cid"]
        # The provider that published the second export thinned the first.
        assert result.data["export_retention"]["cids"] == [first]
        assert own.cids() == {second}
        status = await _status(agent)
        assert (status["cid"], status["agent_id"]) == (second, AGENT_A)

    # Nothing reached the host account, which the host has no key for.
    assert lighthouse.files == []
    report = await build_storage_health_report(agent_id=AGENT_A, env={})
    assert report.lighthouse.status == "not_configured"


@pytest.mark.asyncio
async def test_an_agent_whose_key_differs_from_the_hosts_is_checked_and_thinned_on_its_account(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The agent's key names one account, the host's another. Every reader
    and pruner of the agent's exports uses the agent's; nothing on the host
    account is credited to it or deleted for it (#2874)."""
    own = lighthouse.service.account(AGENT_KEY)
    host = tmp_path / "host"

    # Agent B resolves the host key and publishes to the host account.
    async with _host(host, monkeypatch, did=AGENT_B) as (agent, raw):
        b_export = await _export(agent, raw)
    a_exports = []
    for _ in range(2):
        lighthouse.now += timedelta(days=1)
        async with _host(
            host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY
        ) as (agent, raw):
            a_exports.append(await _export(agent, raw))
            status = await _status(agent)
            assert (status["cid"], status["agent_id"]) == (a_exports[-1], AGENT_A)

    # A's exports went to A's account, where A's second thinned its first;
    # B's export on the host account is untouched by A's thinning.
    assert own.cids() == {a_exports[1]}
    assert lighthouse.cids() == {b_export}

    # The host account's snapshot retention for A deletes nothing.
    await LighthouseTarget(api_key=API_KEY, agent_id=AGENT_A).prune(NEWEST_ONLY)
    assert lighthouse.cids() == {b_export}
    assert own.cids() == {a_exports[1]}

    # ``kestrel storage health`` reads the host account: it credits B with
    # B's export and A with none of B's. A's is reported by A's own check.
    assert (await _health(AGENT_A, lighthouse)).lighthouse.status == "unavailable"
    report_b = await _health(AGENT_B, lighthouse)
    assert report_b.lighthouse.details["sovereignty_export"]["cid"] == b_export


@pytest.mark.asyncio
async def test_a_failed_thinning_pass_keeps_the_exports_and_says_so(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        first = await _export(agent, raw)
    lighthouse.now += timedelta(days=1)
    lighthouse.fail_deletes = True
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        result = await _export_result(agent, raw)

    # The export landed; the pass that would have thinned the first failed,
    # is reported, and left it in place.
    retention = result.data["export_retention"]
    assert retention["deleted"] == 0
    assert "delete_file failed" in retention["error"]
    assert lighthouse.cids() == {first, result.data["cid"]}


@pytest.mark.asyncio
async def test_an_export_made_before_attribution_still_restores(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    kubo.online = True
    async with _host(tmp_path / "origin", monkeypatch, did=AGENT_A) as (agent, raw):
        for role, content in MESSAGES:
            await raw.add_conversation(role, content, metadata={})
        # What export_sovereignty stored before #2874: the same sealed payload
        # uploaded as content.bin, recorded under no upload name.
        blob = await agent.storage.create_backup_blob(include_db=True)
        stored = FilecoinAdapter().store_content(
            blob,
            storage_tier=StorageTier.IPFS,
            encrypt=True,
            metadata={"agent": AGENT_A},
            envelope_owner=AGENT_A,
        )
        await agent.storage.record_backup_artifact(AGENT_A, stored)
        assert kubo.upload_names == ["content.bin"]
        assert stored.filename is None
        cid = stored.ipfs_cid

        # The host loses its history and keeps the receipt.
        await raw.db.execute_commit("DELETE FROM conversation_history")
        restored = await SovereigntyFeature(agent).import_sovereignty(cid)
        assert restored.status is ToolResultStatus.OK, restored
        assert await _history(raw) == MESSAGES

    # A fresh host holds no receipt and restores it from IPFS.
    async with _host(tmp_path / "fresh", monkeypatch, did=AGENT_A) as (agent, raw):
        restored = await SovereigntyFeature(agent).import_sovereignty(cid)
        assert restored.status is ToolResultStatus.OK, restored
        assert await _history(raw) == MESSAGES


@pytest.mark.asyncio
async def test_a_reachable_ipfs_node_still_takes_the_export(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """A sovereign-operated node outranks the delegated service: the
    provider publishes only when the adapter's own node is unreachable."""
    kubo.online = True
    async with _host(tmp_path / "host", monkeypatch, did=AGENT_A) as (agent, raw):
        result = await _export_result(agent, raw)

    assert result.data["cid"] in kubo.blocks
    assert "export_retention" not in result.data
    assert lighthouse.files == []
    [name] = kubo.upload_names
    assert name.startswith(f"kestrel_sovereignty__{AGENT_A}__")


@pytest.mark.asyncio
async def test_without_a_provider_or_node_the_export_stays_local_and_says_so(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    async with _host(
        tmp_path / "host", monkeypatch, did=AGENT_A, lighthouse=False
    ) as (agent, raw):
        for role, content in MESSAGES:
            await raw.add_conversation(role, content, metadata={})
        result = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs", encrypt=True
        )
        status = await _status(agent)

    assert result.status is ToolResultStatus.PARTIAL, result
    assert result.data["tier"] == "local_only"
    assert lighthouse.files == []
    assert status["status"] == "not_configured"
