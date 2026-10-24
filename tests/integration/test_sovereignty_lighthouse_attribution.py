"""Sovereignty exports on Lighthouse are attributed to their agent (#2874).

An export the health check cannot see is one nobody notices missing, and an
export a retention sweep cannot attribute is one it can delete without knowing
whose restore point it destroyed. Every test here goes through the entry
points: ``SovereigntyFeature.export_sovereignty``, ``check_sovereignty_status``
and ``import_sovereignty``, ``kestrel storage health --agent``, the snapshot
target boot registers and its retention under ``SyncService.force_snapshot``,
a real ``KestrelAgent`` boot and cold-start restore, and the backlog cleanup
tool in ``scripts/backup_cleanup.py``.

Lighthouse keeps one account per API key, behind the REST client core uses.
Each agent's Lighthouse provider is built the way boot builds it, from its
PayerPolicy resolution: an agent-scoped key in its encrypted key storage, or
else the host ``LIGHTHOUSE_API_KEY``. Agents that resolve the host key share
its account, which is what makes attribution matter; an agent with its own key
publishes to another account, which every reader and pruner of its exports
must then use. State snapshots stay on the host key's account whatever key an
agent's exports use: it is the one account cold-start restore can read. The
gateway is public: any account's upload downloads by CID, with no key, and an
import reads it through the real gateway reader, answered in-process.
The adapter's own IPFS node is offline unless a test brings it up: the agent's
Lighthouse provider then takes the IPFS tier, and serves it back.
"""

from __future__ import annotations

from tests.shared.native_shutdown_budget import shutdown_fixture_agent

import asyncio
import base64
import hashlib
import itertools
import json
import os
import re
import tracemalloc
import zlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import toml
from cryptography.fernet import Fernet

from kestrel_sdk.payer_policy import PayerKind, PayerPolicy, PayerSpec
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign import cli
from kestrel_sovereign.features.sovereignty.feature import SovereigntyFeature
from kestrel_sovereign.filecoin_adapter import (
    ENVELOPE_MAGIC,
    FilecoinAdapter,
    StorageTier,
    _encoded_retrieval_limit,
)
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
from kestrel_sovereign.services.key_resolution import KeyUnreadableError
from kestrel_sovereign.kestrel_agent import (
    _add_lighthouse_snapshot_target,
    _build_lighthouse_provider,
)
from kestrel_sovereign.storage import GraphNode, Storage
from kestrel_sovereign.storage.privacy_wrapper import (
    DurablePersistenceGate,
    PrivacyEnforcingStorage,
)
from kestrel_sovereign.storage.providers.lighthouse_provider import (
    resolve_lighthouse_storage,
)
from kestrel_sovereign.storage.providers.lighthouse_rest import (
    UPLOAD_CHUNK_BYTES,
    LighthouseDownloadLimitError,
)
from kestrel_sovereign.paths import spawned_agent_env
from kestrel_sovereign.storage.sync.health import build_storage_health_report
from kestrel_sovereign.storage.sync.lighthouse_target import LighthouseTarget
from kestrel_sovereign.storage.sync.retention import RetentionPolicy
from kestrel_sovereign.storage.sync.service import SyncService
from kestrel_sovereign.storage.sync.sovereignty_exports import (
    sovereignty_export_upload_name,
)
from scripts import backup_cleanup
from tests.utils.lighthouse_gateway import GatewayServer

AGENT_A = "did:pkh:eip155:1:0x2874alpha"
AGENT_B = "did:pkh:eip155:1:0x2874bravo"
# A DID that extends AGENT_A's past a "__": a prefix match on AGENT_A's
# upload names would take this agent's exports for AGENT_A's.
AGENT_A_EXTENDED = AGENT_A + "__20260101_000000"
# Each agent's name in its host's multi_agent.toml.
NAMES = {AGENT_A: "alpha", AGENT_B: "bravo", AGENT_A_EXTENDED: "alpha-extended"}
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

    def public(self, cid: str):
        """What the public gateway serves for ``cid``: any account's upload,
        or None."""
        for account in self.accounts.values():
            if cid in account.blobs:
                return account.blobs[cid]
        return None

    def blob(self, cid: str) -> bytes:
        """What the public gateway serves for ``cid``, from any account."""
        content = self.public(cid)
        if content is not None:
            return content
        raise httpx.HTTPStatusError(
            "gateway 404",
            request=httpx.Request("GET", f"https://gateway.lighthouse.storage/ipfs/{cid}"),
            response=httpx.Response(404),
        )

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
        self.fail_uploads = False
        self.fail_listing = False
        # Awaited after each upload lands, for a test to change the world.
        self.after_upload = None
        self.downloads: list[str] = []
        # The byte ceiling each gateway read was made under.
        self.download_ceilings: list = []

    @property
    def gateway(self) -> GatewayServer:
        return self.service.gateway

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

    def records(self) -> list[dict]:
        """The upload records, without any malformed entry a test planted."""
        return [record for record in self.files if isinstance(record, dict)]

    def cids(self) -> set[str]:
        return {record["cid"] for record in self.records()}

    def record(self, cid: str) -> dict:
        [record] = [record for record in self.records() if record["cid"] == cid]
        return record


class _LighthouseClient:
    def __init__(self, account: _LighthouseAccount):
        self.account = account

    async def upload(self, content, filename, tag="kestrel-storage", on_progress=None):
        if self.account.fail_uploads:
            raise httpx.HTTPStatusError(
                "upload failed",
                request=httpx.Request("POST", "https://upload.lighthouse.storage"),
                response=httpx.Response(503),
            )
        if on_progress is not None:
            on_progress(len(content), len(content))
        record = self.account.add(content, filename, tag)
        if self.account.after_upload is not None:
            await self.account.after_upload()
        return {"Hash": record["cid"], "Name": filename, "Size": str(len(content))}

    async def upload_car(self, car_bytes=None, tag="kestrel-storage", filename="export.car",
                         on_progress=None, *, body=None):
        if body is not None:
            # The snapshot target streams its CAR as an UploadBody (#3662).
            body.seek(0)
            car_bytes = b"".join(iter(lambda: body.read(UPLOAD_CHUNK_BYTES), b""))
        return await self.upload(car_bytes, filename, tag, on_progress)

    async def get_uploads(self, last_key=None):
        if self.account.fail_listing:
            raise httpx.HTTPStatusError(
                "files_uploaded failed",
                request=httpx.Request("GET", "https://api.lighthouse.storage"),
                response=httpx.Response(503),
            )
        files = self.account.files
        start = 0
        if last_key not in (None, "null"):
            ids = [r.get("id") if isinstance(r, dict) else None for r in files]
            start = ids.index(last_key) + 1
        return {
            "fileList": [
                dict(record) if isinstance(record, dict) else record
                for record in files[start:start + 1000]
            ],
            "totalFiles": len(files),
        }

    async def delete_file(self, file_id):
        if self.account.fail_deletes:
            raise httpx.HTTPStatusError(
                "delete_file failed",
                request=httpx.Request("DELETE", "https://api.lighthouse.storage"),
                response=httpx.Response(503),
            )
        self.account.files = [
            r for r in self.account.files if not isinstance(r, dict) or r["id"] != file_id
        ]
        return {"deleted": True}

    async def get_deal_status(self, cid):
        return list(self.account.deals.get(cid, []))

    async def download(self, cid, timeout=None, *, expected_bytes=None, max_bytes=None):
        content = self.account.service.blob(cid)
        self.account.downloads.append(cid)
        self.account.download_ceilings.append(max_bytes)
        if max_bytes is not None and len(content) > max_bytes:
            raise LighthouseDownloadLimitError(f"{cid} exceeds {max_bytes} bytes")
        return content

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
    are ``lighthouse.service.account(key)``; the public gateway, read with no
    key, is ``lighthouse.gateway``."""
    service = _Lighthouse()
    host = service.account(API_KEY)
    monkeypatch.setenv("LIGHTHOUSE_API_KEY", API_KEY)
    service.gateway = GatewayServer(service.public).install(monkeypatch)
    with patch(
        "kestrel_sovereign.storage.providers.lighthouse_provider.LighthouseRestClient",
        side_effect=service.client,
    ), patch(
        "kestrel_sovereign.storage.providers.lighthouse_rest.LighthouseRestClient",
        side_effect=service.client,
    ):
        yield host


@pytest.fixture(autouse=True)
def no_ambient_backup_tiers(monkeypatch):
    """The other backup tiers ``kestrel storage health`` checks are not
    configured, whatever the shell running the tests exports."""
    for name in (
        "GCS_BACKUP_BUCKET",
        "GCP_PROJECT",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "SOVEREIGN_IPFS_URL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def newest_only(monkeypatch):
    """The backup retention policy an export's thinning reads."""
    monkeypatch.setattr(
        "kestrel_sovereign.features.sovereignty.feature.load_retention_policy",
        lambda: NEWEST_ONLY,
    )


def _register(directory, name):
    """Add ``name`` to the host's multi_agent.toml, as ``kestrel create`` does."""
    path = directory / "multi_agent.toml"
    config = toml.loads(path.read_text()) if path.exists() else {
        "host": {"port": 8888, "bind": "127.0.0.1"},
        "agents": {},
    }
    agents = config["agents"]
    if name not in agents:
        agents[name] = {
            "data_dir": f"agent_data/{name}",
            "port": 8801 + len(agents),
            "autostart": True,
        }
        path.write_text(toml.dumps(config))


@asynccontextmanager
async def _host(directory, monkeypatch, *, did, lighthouse=True, agent_key=None):
    """One agent on a host, laid out as ``kestrel create`` lays it out. Agents
    co-hosted in one directory share its ``storage_cache``, its registry and
    the host's Lighthouse key, and keep their databases. ``agent_key`` is
    stored in the agent's own encrypted key storage, as ``!add-key`` or a
    SELF_WALLET mint stores it."""
    directory.mkdir(exist_ok=True)
    monkeypatch.chdir(directory)
    monkeypatch.setenv("KESTREL_HOME", str(directory))
    name = NAMES[did]
    agent_dir = directory / "agent_data" / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    _register(directory, name)
    raw = Storage(str(agent_dir / "kestrel_prime.db"), agent_id=did)
    await raw.initialize()
    if await raw.get_node(did) is None:
        await raw.add_node(
            GraphNode(node_id=did, node_type="agent", label=name, properties={})
        )
    keys = ServiceKeyStorage(raw.db, did)
    if agent_key is not None and not await keys.has_key("lighthouse"):
        await keys.store_key("lighthouse", agent_key)
    provider = provider_error = None
    storage_permitted = False
    if lighthouse:
        # Built only when policy permits a remote tier, by boot's own helper.
        async def resolve_storage():
            return await resolve_lighthouse_storage(
                did, policy=PayerPolicy.host_env_default(), db=raw.db, host_db=None
            )

        provider, provider_error, storage_permitted = await _build_lighthouse_provider(
            resolve_storage
        )
    wallet = MagicMock()
    wallet.can_afford.return_value = True
    wallet.transfer = AsyncMock()
    gate = DurablePersistenceGate()
    agent = SimpleNamespace(
        agent_id=did,
        name=name,
        agent_dir=agent_dir,
        storage=PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL),
        features={},
        wallet=wallet,
        _get_durable_persistence_gate=lambda: gate,
        lighthouse_provider=provider,
        lighthouse_provider_error=provider_error,
        # Boot's gate for the snapshot target, which uses the host's key.
        lighthouse_storage_permitted=storage_permitted,
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


async def _health(directory, did, account, *, after=timedelta(hours=1)):
    """``kestrel storage health --agent <name>``: exports read on the account
    the agent's PayerPolicy resolves, the one it publishes them to, and
    snapshots on the host key's account, both in the environment the
    launcher hands the agent, as the command reads them."""
    launch_env = spawned_agent_env(directory)
    credential = await cli._storage_health_lighthouse_key(
        directory, launch_env, agent_name=NAMES[did], agent_id=None
    )
    assert (credential.agent_id, credential.with_agent_database) == (did, True)
    assert credential.error is None, credential.error
    return await build_storage_health_report(
        agent_id=credential.agent_id,
        export_api_key=credential.api_key,
        env=launch_env,
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
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_B):
        pass
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        cid = await _export(agent, raw)

        # Published through the agent's provider, named for the agent alone
        # and, by its nonce, for this publication alone (#3602).
        record = lighthouse.record(cid)
        assert re.fullmatch(
            rf"kestrel_sovereignty__{re.escape(AGENT_A)}__\d{{8}}_\d{{6}}__[0-9a-f]{{16}}\.bin",
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
        report = await _health(host, AGENT_A, lighthouse)
        assert report.lighthouse.status == "pending", report.lighthouse
        assert report.lighthouse.details["resolved_via"] == "sovereignty_export"
        assert report.lighthouse.details["cid"] == cid
        export = report.lighthouse.details["sovereignty_export"]
        assert export["agent_id"] == AGENT_A
        assert export["cid"] == cid
        assert export["file_name"] == record["fileName"]
        assert "sovereignty export" in report.lighthouse.message

        lighthouse.deals[cid] = [{"DealID": 1, "Provider": 10479}]
        settled = await _health(host, AGENT_A, lighthouse)
        assert settled.lighthouse.status == "ok"
        assert settled.lighthouse.details["sovereignty_export"]["deal_count"] == 1

        # So does the agent's own check, on the account it published to.
        status = await _status(agent)
        assert status["status"] == "ok"
        assert status["cid"] == cid
        assert status["agent_id"] == AGENT_A
        assert status["deal_count"] == 1

        # Another agent sharing the key is not credited with it.
        other = await _health(host, AGENT_B, lighthouse)
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
        report = await _health(host, did, lighthouse)
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
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_B):
        pass
    for _ in range(2):
        lighthouse.now += timedelta(days=1)
        async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
            await _export(agent, raw)
    # Agent A's own thinning, by the newest-only policy, kept both.
    assert unattributable <= lighthouse.cids()

    for did in (AGENT_A, AGENT_B):
        await LighthouseTarget(api_key=API_KEY, agent_id=did).prune(NEWEST_ONLY)
        assert unattributable <= lighthouse.cids()

    # Neither is credited to an agent by the health check.
    report = await _health(host, AGENT_B, lighthouse)
    assert report.lighthouse.status == "unavailable"
    report = await _health(host, AGENT_A, lighthouse)
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
    # ``kestrel storage health`` reads the agent's account through the key in
    # its database, and finds its latest export there, attributed to it.
    own.deals[second] = [{"DealID": 2}]
    report = await _health(host, AGENT_A, own)
    assert report.lighthouse.status == "ok", report.lighthouse
    export = report.lighthouse.details["sovereignty_export"]
    assert (export["cid"], export["agent_id"]) == (second, AGENT_A)


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

    # ``kestrel storage health`` reads each agent's own account: A's latest
    # export on A's, B's on the host account, and neither the other's.
    report_a = await _health(host, AGENT_A, own)
    assert report_a.lighthouse.details["sovereignty_export"]["cid"] == a_exports[1]
    report_b = await _health(host, AGENT_B, lighthouse)
    assert report_b.lighthouse.details["sovereignty_export"]["cid"] == b_export


@pytest.mark.asyncio
async def test_a_snapshot_account_that_fails_hides_no_export_from_health(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The agent's exports go to its own account, its snapshots to the
    host's, and the host account stops answering. ``kestrel storage health``
    reports that account's error and still the agent's latest export on its
    own account (#3602)."""
    own = lighthouse.service.account(AGENT_KEY)
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        export = await _export(agent, raw)
        await _snapshot(agent, raw, "a snapshot on the host account")
    own.deals[export] = [{"DealID": 7}]

    lighthouse.fail_listing = True
    report = await _health(host, AGENT_A, own)

    assert report.lighthouse.status == "error", report.lighthouse
    details = report.lighthouse.details
    assert "files_uploaded failed" in details["snapshot_error"]
    assert details["sovereignty_export_error"] is None
    exported = details["sovereignty_export"]
    assert (exported["cid"], exported["agent_id"], exported["deal_count"]) == (
        export,
        AGENT_A,
        1,
    )
    assert report.lighthouse.message == (
        f"{details['snapshot_error']}; "
        "Latest Lighthouse sovereignty export has 1 Filecoin deal(s)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "offset",
    [timedelta(days=-1), timedelta(0), timedelta(days=1)],
    ids=["earlier", "same-second", "later-dated"],
)
async def test_another_upload_of_the_same_cid_is_thinned_by_its_own_record(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, offset
):
    """The account already holds an identical upload of the export, from
    another attempt: two records of the CID just published, each taking
    quota. Thinning decides on each record by its own id. An earlier record
    is deleted, and so is one made in the same second, whose record id sorts
    after the export's; one dated after the export, by a clock ahead of this
    one's, is the newest and kept. The record just published is kept in
    every case: it is the one with its own name (#3602)."""
    other = {}

    async def another_identical_upload():
        lighthouse.after_upload = None
        [published] = lighthouse.records()
        other.update(
            lighthouse.add(
                lighthouse.blobs[published["cid"]],
                sovereignty_export_upload_name(AGENT_A, lighthouse.now + offset),
                published["tag"],
                created_at=lighthouse.now + offset,
            )
        )
        # An id that sorts after the export's: the record just published is
        # the smaller of two of one time (#3602).
        other["id"] = lighthouse.files[-1]["id"] = f"{published['id']}-retry"
        # Listed first, where an upload made before this one would be.
        lighthouse.files.insert(0, lighthouse.files.pop())

    lighthouse.after_upload = another_identical_upload
    async with _host(tmp_path / "host", monkeypatch, did=AGENT_A) as (agent, raw):
        result = await _export_result(agent, raw)

    cid = result.data["cid"]
    assert other["cid"] == cid
    survivors = [r for r in lighthouse.records() if r["id"] != other["id"]]
    assert len(survivors) == 1, "the record just published was deleted"
    [published] = survivors
    assert published["cid"] == cid
    assert published["fileName"] != other["fileName"]
    assert other["id"] > published["id"]
    retention = result.data["export_retention"]
    if offset <= timedelta(0):
        assert (retention["deleted"], retention["file_ids"], retention["cids"]) == (
            1,
            [other["id"]],
            [cid],
        )
        assert lighthouse.records() == [published]
    else:
        assert (retention["deleted"], retention["file_ids"]) == (0, [])
        assert lighthouse.records() == [other, published]


@pytest.mark.asyncio
async def test_an_account_listing_the_export_twice_thins_nothing_and_says_so(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The account lists the export just published as two records under its
    one name. Neither can be told apart from the record the upload made, so
    the pass deletes nothing, not even the agent's older export, and says
    why (#3602)."""
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        first = await _export(agent, raw)
    lighthouse.now += timedelta(days=1)

    async def listed_twice():
        lighthouse.after_upload = None
        published = lighthouse.records()[-1]
        lighthouse.files.append(dict(published, id=f"file-{next(lighthouse.service.ids)}"))

    lighthouse.after_upload = listed_twice
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        result = await _export_result(agent, raw)

    retention = result.data["export_retention"]
    assert (retention["deleted"], retention["file_ids"]) == (0, [])
    assert "upload records are named" in retention["error"]
    assert lighthouse.cids() == {first, result.data["cid"]}
    assert len(lighthouse.records()) == 3


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


def _lose_host_cache(directory):
    """The host's ``storage_cache``, export copies and all, is gone."""
    cache = directory / "storage_cache"
    for entry in cache.iterdir():
        entry.unlink()


@pytest.mark.asyncio
async def test_a_lighthouse_export_restores_after_the_host_cache_is_lost(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """Kubo is offline throughout: the export went to Lighthouse, and an
    import reads it back from Lighthouse once no cache holds it (#2874)."""
    origin = tmp_path / "origin"
    async with _host(origin, monkeypatch, did=AGENT_A) as (agent, raw):
        cid = await _export(agent, raw)
        assert lighthouse.record(cid)
        _lose_host_cache(origin)

        # The host that made it, holding its receipt but not its cache copy.
        await raw.db.execute_commit("DELETE FROM conversation_history")
        restored = await SovereigntyFeature(agent).import_sovereignty(cid)
        assert restored.status is ToolResultStatus.OK, restored
        assert await _history(raw) == MESSAGES
    # Read from the public gateway, not through the agent's provider.
    assert lighthouse.gateway.cids == [cid]
    assert lighthouse.downloads == []

    # A brand-new host and database: no receipt, no cache, Kubo offline.
    _lose_host_cache(origin)
    fresh = tmp_path / "fresh"
    async with _host(fresh, monkeypatch, did=AGENT_A) as (agent, raw):
        assert await agent.storage.get_nodes_by_type("backup_artifact") == []
        restored = await SovereigntyFeature(agent).import_sovereignty(cid)
        assert restored.status is ToolResultStatus.OK, restored
        assert await _history(raw) == MESSAGES
    assert lighthouse.gateway.cids == [cid, cid]
    assert kubo.blocks == {}


@pytest.mark.asyncio
async def test_a_lighthouse_export_is_restored_only_by_its_agent_and_under_the_ceiling(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """What Lighthouse serves goes through the envelope's ownership check
    and the import ceiling exactly as Kubo's answer does (#2873, #2874), on
    hosts that hold no Lighthouse key (#3601)."""
    origin = tmp_path / "origin"
    async with _host(origin, monkeypatch, did=AGENT_A) as (agent, raw):
        # Incompressible, so the export outgrows the ceiling below.
        noise = base64.b64encode(os.urandom(512 * 1024)).decode()
        await raw.add_conversation("user", noise, metadata={})
        cid = await _export(agent, raw)
    _lose_host_cache(origin)
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")

    async with _host(tmp_path / "other", monkeypatch, did=AGENT_B) as (agent, raw):
        refused = await SovereigntyFeature(agent).import_sovereignty(cid)
        assert refused.status is ToolResultStatus.ERROR
        assert refused.data["refusal"] == "export_of_another_agent"
        assert await _history(raw) == []

    monkeypatch.setattr(
        "kestrel_sovereign.features.sovereignty.feature.MAX_SOVEREIGNTY_IMPORT_BYTES",
        16,
    )
    published = len(lighthouse.blobs[cid])
    ceiling = _encoded_retrieval_limit(16)
    assert published > 3 * ceiling
    # The gateway need not say how long its answer is: only counting what
    # arrives can stop the read.
    lighthouse.gateway.declare_length = False
    async with _host(tmp_path / "fresh", monkeypatch, did=AGENT_A) as (agent, raw):
        refused = await SovereigntyFeature(agent).import_sovereignty(cid)
        assert refused.status is ToolResultStatus.ERROR
        assert refused.data["refusal"] == "export_exceeds_import_limit"
        assert await _history(raw) == []
    # The gateway was read under the ceiling Kubo is read under, not in
    # full: the read stopped at the first chunk past it.
    assert lighthouse.gateway.cids == [cid, cid]
    assert lighthouse.gateway.sent[-1] <= ceiling + 64 * 1024 + 4096 < published


@pytest.mark.asyncio
async def test_a_host_with_no_lighthouse_key_restores_an_export_from_the_gateway(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The export went to Lighthouse on the agent's own key, and the host
    configured none. The key was lost with the agent's database, and the new
    host configures none either, so boot builds no provider there. With Kubo
    offline and no cache, the export still restores into the empty database
    by CID, from Lighthouse's public gateway, which needs no key (#3601)."""
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")
    own = lighthouse.service.account(AGENT_KEY)
    origin = tmp_path / "origin"
    async with _host(origin, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        cid = await _export(agent, raw)
    assert own.cids() == {cid}
    _lose_host_cache(origin)

    fresh = tmp_path / "fresh"
    async with _host(fresh, monkeypatch, did=AGENT_A) as (agent, raw):
        # No key anywhere: boot built no provider, and not because building
        # one failed.
        assert "LIGHTHOUSE_API_KEY" not in os.environ
        assert not await ServiceKeyStorage(raw.db, AGENT_A).has_key("lighthouse")
        assert (agent.lighthouse_provider, agent.lighthouse_provider_error) == (None, None)
        assert await agent.storage.get_nodes_by_type("backup_artifact") == []
        assert await _history(raw) == []

        restored = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert restored.status is ToolResultStatus.OK, restored
        assert restored.data["messages_restored"] == len(MESSAGES)
        assert await _history(raw) == MESSAGES
    [request] = lighthouse.gateway.requests
    assert lighthouse.gateway.cids == [cid]
    assert "authorization" not in request.headers
    assert all(client.is_closed for client in lighthouse.gateway.clients)
    assert own.downloads == [] and lighthouse.downloads == []
    assert kubo.blocks == {}


@pytest.mark.asyncio
async def test_a_privacy_mode_that_keeps_storage_local_asks_no_gateway(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The keyless read still tells Lighthouse which CID this host wants, so
    it follows the agent's privacy mode as it is when the import runs: a mode
    that permits no remote tier makes no request (#3601)."""
    origin = tmp_path / "origin"
    async with _host(origin, monkeypatch, did=AGENT_A) as (agent, raw):
        cid = await _export(agent, raw)
    _lose_host_cache(origin)
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")

    async with _host(tmp_path / "fresh", monkeypatch, did=AGENT_A) as (agent, raw):
        agent.storage.set_privacy_mode(PrivacyMode.ISOLATED)
        refused = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert refused.status is ToolResultStatus.ERROR, refused
        assert "Content not found" in refused.error
        assert lighthouse.gateway.requests == []

        agent.storage.set_privacy_mode(PrivacyMode.NORMAL)
        restored = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert restored.status is ToolResultStatus.OK, restored
        assert await _history(raw) == MESSAGES
    assert lighthouse.gateway.cids == [cid]


@pytest.mark.asyncio
async def test_an_agent_whose_key_was_removed_still_restores_from_the_gateway(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The agent's provider resolves no key once its own is removed, so it
    publishes nothing more (#2874). Reading its earlier export back never
    asked it for one (#3601)."""
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")
    lighthouse.service.account(AGENT_KEY)
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        cid = await _export(agent, raw)
        assert await ServiceKeyStorage(raw.db, AGENT_A).delete_key("lighthouse")
        _lose_host_cache(host)
        await raw.db.execute_commit("DELETE FROM conversation_history")

        restored = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert restored.status is ToolResultStatus.OK, restored
        assert await _history(raw) == MESSAGES
        assert await agent.lighthouse_provider.resolve_api_key() is None
    assert lighthouse.gateway.cids == [cid]


def _tamper_ciphertext(served: bytes) -> bytes:
    """The export's sealed envelope with one ciphertext byte changed."""
    payload = zlib.decompress(served)
    flipped = b"A" if payload[-10:-9] != b"A" else b"B"
    return zlib.compress(payload[:-10] + flipped + payload[-9:])


@pytest.mark.asyncio
async def test_a_substituted_gateway_answer_is_refused_on_a_host_with_no_key(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The gateway is not trusted for being reachable without a key: the
    envelope it serves is opened and authenticated as Kubo's is (#2873)."""
    origin = tmp_path / "origin"
    async with _host(origin, monkeypatch, did=AGENT_A) as (agent, raw):
        cid = await _export(agent, raw)
    _lose_host_cache(origin)
    lighthouse.blobs[cid] = _tamper_ciphertext(lighthouse.blobs[cid])
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")

    async with _host(tmp_path / "fresh", monkeypatch, did=AGENT_A) as (agent, raw):
        assert agent.lighthouse_provider is None
        refused = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert refused.status is ToolResultStatus.ERROR, refused
        assert refused.data["refusal"] == "decryption_failed"
        assert await _history(raw) == []
    assert lighthouse.gateway.cids == [cid]


@pytest.mark.asyncio
async def test_a_compression_bomb_from_the_gateway_is_refused_while_it_is_decompressed(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """64 MiB of zeros compresses to about 64 KB, inside the read ceiling.
    Decompression stops at the ceiling instead of inflating it, whatever
    served it (#2873)."""
    limit = 1024 * 1024
    monkeypatch.setattr(
        "kestrel_sovereign.features.sovereignty.feature.MAX_SOVEREIGNTY_IMPORT_BYTES",
        limit,
    )
    inflated = b"\0" * (64 * limit)
    lighthouse.blobs["bafybomb"] = zlib.compress(inflated, 9)
    del inflated
    assert len(lighthouse.blobs["bafybomb"]) < _encoded_retrieval_limit(limit)
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")

    async with _host(tmp_path / "fresh", monkeypatch, did=AGENT_A) as (agent, raw):
        restore = AsyncMock(wraps=agent.storage.restore_from_backup_blob)
        monkeypatch.setattr(agent.storage, "restore_from_backup_blob", restore)
        tracemalloc.start()
        try:
            refused = await SovereigntyFeature(agent).import_sovereignty("bafybomb")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert refused.status is ToolResultStatus.ERROR, refused
        assert refused.data["refusal"] == "export_exceeds_import_limit"
        restore.assert_not_awaited()
        assert await _history(raw) == []
    assert lighthouse.gateway.cids == ["bafybomb"]
    # Inflating it whole would allocate 64 MiB.
    assert peak < 8 * limit


@pytest.mark.asyncio
async def test_health_finds_an_export_beyond_the_first_page_of_uploads(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The account lists 1,000 uploads to a page. The agent's latest export
    is on the second, behind an older one on the first (#2874)."""
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        older = await _export(agent, raw)
    for index in range(999):
        lighthouse.add(f"filler-{index}".encode(), f"{index:016x}.bin", "kestrel-storage")
    lighthouse.now += timedelta(days=1)
    keep_all = RetentionPolicy.from_config(
        {"backup": {"retention": {"identity": {"keep_all_days": 36500}}}}
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.sovereignty.feature.load_retention_policy",
        lambda: keep_all,
    )
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        latest = await _export(agent, raw)
    assert [r["cid"] for r in lighthouse.files].index(latest) == 1000
    assert older in lighthouse.cids()

    report = await _health(host, AGENT_A, lighthouse)

    assert report.lighthouse.details["sovereignty_export"]["cid"] == latest


def _snapshot_service(agent) -> SyncService:
    """The agent's sync service, with the Lighthouse target boot registers."""
    service = SyncService(
        db_path=str(agent.agent_dir / "kestrel_prime.db"),
        retention_policy=NEWEST_ONLY,
    )
    assert _add_lighthouse_snapshot_target(
        service,
        storage_permitted=agent.lighthouse_storage_permitted,
        agent_id=agent.agent_id,
        state_dir=agent.agent_dir,
    )
    return service


async def _snapshot(agent, raw, message):
    """The agent's sync service takes a snapshot: the upload and the
    retention prune after it."""
    await raw.add_conversation("user", message, metadata={})
    [result] = (await _snapshot_service(agent).force_snapshot()).values()
    assert result.success, result
    return result


def _snapshot_cids(account):
    return {r["cid"] for r in account.files if r["fileName"].startswith("kestrel_state__")}


def _export_records(account):
    return [
        r for r in account.records() if r["fileName"].startswith("kestrel_sovereignty__")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("host_key", [True, False], ids=["differing-host-key", "no-host-key"])
async def test_snapshots_stay_on_the_host_account_while_exports_go_to_the_agents(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, host_key
):
    """The agent's exports go to the account its own key names. Its state
    snapshots, and the retention prune after each, stay on the host's
    LIGHTHOUSE_API_KEY account, the one cold-start restore reads. With no
    host key, boot registers no snapshot target at all (#2874)."""
    if not host_key:
        monkeypatch.delenv("LIGHTHOUSE_API_KEY")
    own = lighthouse.service.account(AGENT_KEY)
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        export = await _export(agent, raw)
        if host_key:
            first = (await _snapshot(agent, raw, "first")).metadata["cid"]
            lighthouse.now += timedelta(days=1)
            second = (await _snapshot(agent, raw, "second")).metadata["cid"]
        else:
            service = SyncService(db_path=str(agent.agent_dir / "kestrel_prime.db"))
            assert not _add_lighthouse_snapshot_target(
                service,
                storage_permitted=agent.lighthouse_storage_permitted,
                agent_id=AGENT_A,
                state_dir=agent.agent_dir,
            )
            assert service.targets == []

    # The export, alone, is on the agent's account.
    assert own.cids() == {export}
    if host_key:
        # Retention kept the newest snapshot, on the host account, which
        # holds no export of the agent's.
        assert _snapshot_cids(lighthouse) == {second}
        assert first != second
        assert _export_records(lighthouse) == []
    else:
        assert lighthouse.files == []


@asynccontextmanager
async def _booted(did, db_path):
    """A real agent through every boot phase, on the database at ``db_path``,
    loading no optional feature, under the host-env PayerPolicy."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService

    llm_service = LLMService()
    agent = KestrelAgent(
        did=did,
        storage_path=str(db_path),
        llm_service=llm_service,
        allowed_features=set(),
        payer_policy=PayerPolicy.host_env_default(),
    )
    try:
        await agent.initialize()
        yield agent
    finally:
        await shutdown_fixture_agent(agent)
        await llm_service.close()


async def _incept_with_own_key(tmp_path, monkeypatch):
    """An agent incepted in ``tmp_path`` holding a Lighthouse key of its own,
    AGENT_KEY, in its encrypted key storage, ready for :func:`_booted`.
    Returns its DID and database path."""
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.storage.sync import service as sync_service

    # The remote-tier gate refuses a test instance in a temporary directory.
    # It is opened here: the account a snapshot goes to is under test, not
    # the gate.
    monkeypatch.setattr(
        sync_service,
        "_remote_tiers_allowed",
        lambda context: sync_service.RemoteTierPolicyDecision(True),
    )
    monkeypatch.delenv("LIGHTHOUSE_STATE_CID", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    credentials = await create_kestrel_identity_async(
        output_dir=str(tmp_path), is_test_instance=True, agent_name="kite"
    )
    did = credentials.agent_did
    db_path = tmp_path / "kestrel_prime.db"
    raw = Storage(str(db_path), agent_id=did)
    await raw.initialize()
    await ServiceKeyStorage(raw.db, did).store_key("lighthouse", AGENT_KEY)
    await raw.close()
    return did, db_path


@pytest.mark.asyncio
async def test_a_snapshot_from_boot_restores_on_a_cold_start_when_the_agents_key_differs(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """Through the real boot phases. An agent whose PayerPolicy resolves a key
    of its own, naming another account than the host's, boots and takes a
    snapshot with the target boot registered. Its database and the target's
    local manifest are then lost, as on an ephemeral host. A cold start,
    which can read only the host's key, finds the snapshot and restores the
    database (#2874)."""
    own = lighthouse.service.account(AGENT_KEY)
    did, db_path = await _incept_with_own_key(tmp_path, monkeypatch)

    async with _booted(did, db_path) as agent:
        assert await agent.lighthouse_provider.resolve_api_key() == AGENT_KEY
        await agent._raw_storage.add_conversation(
            "user", "the heron nests by the weir", metadata={}
        )
        results = await agent._sync_service.force_snapshot()
    snapshot = results[f"lighthouse://{did}"]
    assert snapshot.success, snapshot
    assert snapshot.metadata["cid"] in _snapshot_cids(lighthouse)
    assert own.files == []

    # The host is lost: the database, its sidecars, and the snapshot target's
    # local record of what it uploaded.
    for lost in [*tmp_path.glob("kestrel_prime.db*"), *tmp_path.glob(".lighthouse_manifest_*")]:
        lost.unlink()
    assert not db_path.exists()

    async with _booted(did, db_path) as restored:
        history = await restored._raw_storage.get_conversation_history()

    assert "the heron nests by the weir" in [m["content"] for m in history]


@pytest.mark.asyncio
async def test_a_boot_that_cannot_read_the_agents_key_still_snapshots_to_the_host(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """Through the real boot phases. The agent's own Lighthouse key cannot be
    read, so boot builds no provider and its exports stay on this host. The
    snapshot target never used that key: boot still registers it, and its
    snapshot reaches the host account cold-start restore reads (#2874)."""
    own = lighthouse.service.account(AGENT_KEY)
    did, db_path = await _incept_with_own_key(tmp_path, monkeypatch)
    raw = Storage(str(db_path), agent_id=did)
    await raw.initialize()
    await _make_agent_key_unreadable(raw, did)
    await raw.close()

    async with _booted(did, db_path) as agent:
        provider, provider_error = agent.lighthouse_provider, agent.lighthouse_provider_error
        results = await agent._sync_service.force_snapshot()
        on_host = _snapshot_cids(lighthouse)

    assert provider is None
    assert provider_error.startswith(f"{KeyUnreadableError.__name__}: ")
    snapshot = results[f"lighthouse://{did}"]
    assert snapshot.success, snapshot
    assert on_host == {snapshot.metadata["cid"]}
    assert own.files == []


@pytest.mark.asyncio
async def test_kestrel_storage_health_reports_the_agents_export(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, capsys
):
    """The command, end to end: ``--agent`` names the agent, whose own key,
    with no host key at all, reaches the account its export is on (#2874)."""
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")
    own = lighthouse.service.account(AGENT_KEY)
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        cid = await _export(agent, raw)
    own.deals[cid] = [{"DealID": 1, "Provider": 10479}]
    args = cli.build_parser().parse_args(
        ["storage", "health", "--agent", NAMES[AGENT_A], "--json"]
    )

    assert await asyncio.to_thread(cli.cmd_storage_health, args) == 0

    report = json.loads(capsys.readouterr().out)
    export = report["lighthouse"]["details"]["sovereignty_export"]
    assert report["lighthouse"]["status"] == "ok"
    assert (export["cid"], export["agent_id"]) == (cid, AGENT_A)


@pytest.mark.asyncio
async def test_kestrel_storage_health_refuses_an_agent_id_that_is_not_the_agent(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, capsys
):
    async with _host(tmp_path / "host", monkeypatch, did=AGENT_A):
        pass
    args = cli.build_parser().parse_args(
        ["storage", "health", "--agent", NAMES[AGENT_A], "--agent-id", AGENT_B]
    )

    assert await asyncio.to_thread(cli.cmd_storage_health, args) == 2
    assert AGENT_A in capsys.readouterr().err


@pytest.mark.asyncio
async def test_kestrel_storage_health_resolves_in_the_agents_environment(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, capsys
):
    """``kestrel setup`` writes the host key to the project ``.env``, which
    the launcher hands the agent and the shell need not export. The command
    resolves the agent's credential in the agent's environment, so it reads
    the account the agent published to (#2874)."""
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_B) as (agent, raw):
        cid = await _export(agent, raw)
    (host / ".env").write_text(f"LIGHTHOUSE_API_KEY={API_KEY}\n")
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")
    lighthouse.deals[cid] = [{"DealID": 1, "Provider": 10479}]
    args = cli.build_parser().parse_args(
        ["storage", "health", "--agent", NAMES[AGENT_B], "--json"]
    )

    assert await asyncio.to_thread(cli.cmd_storage_health, args) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["lighthouse"]["details"]["sovereignty_export"]["cid"] == cid
    # The shell's environment is its own again afterwards.
    assert "LIGHTHOUSE_API_KEY" not in os.environ


@pytest.mark.asyncio
async def test_kestrel_storage_health_reads_snapshots_on_the_account_the_agent_uses(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, capsys
):
    """The shell exports one LIGHTHOUSE_API_KEY and the project ``.env``
    names another. The launcher hands the agent the file's, so its snapshot
    target uploads to that account. The command reads snapshots there too,
    not on the account the shell's key names, which holds only an older
    snapshot from before the ``.env`` changed (#2874)."""
    shell_key = "shell-exported-key"
    shell = lighthouse.service.account(shell_key)
    host = tmp_path / "host"
    monkeypatch.setenv("LIGHTHOUSE_API_KEY", shell_key)
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        stale = (await _snapshot(agent, raw, "before the .env changed")).metadata["cid"]
        (host / ".env").write_text(f"LIGHTHOUSE_API_KEY={API_KEY}\n")
        lighthouse.now += timedelta(days=1)
        # The agent runs in the environment ``kestrel start`` hands it.
        with patch.dict(os.environ, spawned_agent_env(host), clear=True):
            assert os.environ["LIGHTHOUSE_API_KEY"] == API_KEY
            fresh = (await _snapshot(agent, raw, "after")).metadata["cid"]
    assert (_snapshot_cids(shell), _snapshot_cids(lighthouse)) == ({stale}, {fresh})
    lighthouse.deals[fresh] = [{"DealID": 1, "Provider": 10479}]
    args = cli.build_parser().parse_args(
        ["storage", "health", "--agent", NAMES[AGENT_A], "--json"]
    )

    assert await asyncio.to_thread(cli.cmd_storage_health, args) == 0

    report = json.loads(capsys.readouterr().out)
    details = report["lighthouse"]["details"]
    assert report["lighthouse"]["status"] == "ok", report["lighthouse"]
    assert (details["resolved_via"], details["cid"]) == ("snapshot_upload", fresh)
    assert shell.downloads == []
    # The shell's environment is its own again afterwards.
    assert os.environ["LIGHTHOUSE_API_KEY"] == shell_key


@pytest.mark.asyncio
async def test_when_lighthouse_cannot_take_the_export_it_stays_local_and_says_so(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """Kubo is offline and Lighthouse refuses the upload: the export is kept
    on this host and reported as not durable off-host (#2872, #2874)."""
    lighthouse.fail_uploads = True
    async with _host(tmp_path / "host", monkeypatch, did=AGENT_A) as (agent, raw):
        for role, content in MESSAGES:
            await raw.add_conversation(role, content, metadata={})
        result = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs", encrypt=True
        )
        [receipt] = await agent.storage.get_nodes_by_type("sovereignty_receipt")

    assert result.status is ToolResultStatus.PARTIAL, result
    assert result.data["tier"] == "local_only"
    assert "export_retention" not in result.data
    assert receipt.properties["storage_tier"] == "local_only"
    assert lighthouse.files == []


@pytest.mark.asyncio
async def test_a_malformed_account_record_does_not_fail_an_export(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The thinning pass after an export reads the whole account. An entry
    that is not an upload record is skipped, not raised (#2874)."""
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        first = await _export(agent, raw)
    lighthouse.files.append("not-a-record")
    lighthouse.now += timedelta(days=1)
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        result = await _export_result(agent, raw)

    assert result.data["export_retention"]["cids"] == [first]
    assert "error" not in result.data["export_retention"]


@pytest.mark.asyncio
async def test_kestrel_storage_health_finds_a_local_agent_by_its_did(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, capsys
):
    """``--agent-id`` alone names a local agent by its DID, whose database
    then supplies the key it holds itself (#2874)."""
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")
    own = lighthouse.service.account(AGENT_KEY)
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_B):
        pass
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        cid = await _export(agent, raw)
    own.deals[cid] = [{"DealID": 1, "Provider": 10479}]
    args = cli.build_parser().parse_args(
        ["storage", "health", "--agent-id", AGENT_A, "--json"]
    )

    assert await asyncio.to_thread(cli.cmd_storage_health, args) == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["lighthouse"]["details"]["sovereignty_export"]["cid"] == cid
    assert "note:" not in captured.err


@pytest.mark.asyncio
async def test_kestrel_storage_health_says_when_no_local_agent_holds_the_did(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, capsys
):
    async with _host(tmp_path / "host", monkeypatch, did=AGENT_A):
        pass
    args = cli.build_parser().parse_args(
        ["storage", "health", "--agent-id", AGENT_B, "--json"]
    )

    await asyncio.to_thread(cli.cmd_storage_health, args)

    captured = capsys.readouterr()
    assert json.loads(captured.out)["lighthouse"]["status"] == "unavailable"
    assert "no local agent is" in captured.err


@pytest.mark.asyncio
async def test_kestrel_storage_health_writes_nothing_to_the_host_database(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, capsys
):
    """A health check reads the host key database without running its
    startup DDL or changing its journal (#2874)."""
    import sqlite3

    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A):
        pass
    host_db = host / "agent_data" / "host.db"
    with sqlite3.connect(host_db) as connection:
        connection.execute("CREATE TABLE marker (id INTEGER)")
    connection.close()
    before = host_db.read_bytes()
    args = cli.build_parser().parse_args(
        ["storage", "health", "--agent", NAMES[AGENT_A], "--json"]
    )

    await asyncio.to_thread(cli.cmd_storage_health, args)

    assert host_db.read_bytes() == before
    assert sorted(p.name for p in host_db.parent.glob("host.db*")) == ["host.db"]


@pytest.mark.asyncio
async def test_a_rotated_agent_key_moves_exports_not_snapshots(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """After the agent's own key is rotated its exports go to the account
    the new key names; its snapshots stay on the host account (#2874)."""
    old = lighthouse.service.account(AGENT_KEY)
    new = lighthouse.service.account("agent-alpha-rotated-key")
    async with _host(tmp_path / "host", monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (
        agent,
        raw,
    ):
        service = _snapshot_service(agent)
        before = await _export(agent, raw)
        await raw.add_conversation("user", "before", metadata={})
        await service.force_snapshot()
        await ServiceKeyStorage(raw.db, AGENT_A).store_key(
            "lighthouse", "agent-alpha-rotated-key", replace=True
        )
        lighthouse.now += timedelta(days=1)
        await raw.add_conversation("user", "after", metadata={})
        [result] = (await service.force_snapshot()).values()
        after = await _export(agent, raw)

    assert result.success, result
    assert (old.cids(), new.cids()) == ({before}, {after})
    assert _snapshot_cids(lighthouse) == {result.metadata["cid"]}


async def _force_snapshot(service):
    [result] = (await service.force_snapshot()).values()
    return result


@pytest.mark.asyncio
async def test_a_removed_agent_key_is_not_used_again(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The agent's own key is deleted and no host key stands in. Its provider
    resolves none, so nothing more reaches the account the removed key named:
    an export stays local and says so, and the agent's health check reports
    no credential, as ``kestrel storage health`` does (#2874)."""
    monkeypatch.delenv("LIGHTHOUSE_API_KEY")
    own = lighthouse.service.account(AGENT_KEY)
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        export = await _export(agent, raw)
        assert own.cids() == {export}

        assert await ServiceKeyStorage(raw.db, AGENT_A).delete_key("lighthouse")

        status = await _status(agent)
        result = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs", encrypt=True
        )
        provider_key = await agent.lighthouse_provider.resolve_api_key()

    assert status["status"] == "not_configured", status
    assert result.status is ToolResultStatus.PARTIAL, result
    assert result.data["tier"] == "local_only"
    assert provider_key is None
    assert not agent.lighthouse_provider.is_available()
    assert own.cids() == {export}
    # ``kestrel storage health`` resolves no key for the agent either.
    credential = await cli._storage_health_lighthouse_key(
        host, spawned_agent_env(host), agent_name=NAMES[AGENT_A], agent_id=None
    )
    assert (credential.api_key, credential.error) == (None, None)


@pytest.mark.asyncio
async def test_kestrel_storage_health_reports_an_unresolvable_credential_and_the_other_tiers(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, capsys
):
    """The agent's own kestrel.toml asks for a SELF_WALLET key it has not
    minted, and the check has no wallet to mint one with. That is the
    Lighthouse tier's error; the other tiers are still checked (#2874)."""
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        agent_dir = agent.agent_dir
    self_wallet = PayerPolicy(
        llm=PayerSpec(vendor="openrouter", kind=PayerKind.HOST_ENV),
        storage=PayerSpec(vendor="lighthouse", kind=PayerKind.SELF_WALLET),
        compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
    )
    (agent_dir / "kestrel.toml").write_text(
        toml.dumps({"payments": self_wallet.to_toml_section()})
    )
    args = cli.build_parser().parse_args(
        ["storage", "health", "--agent", NAMES[AGENT_A], "--json"]
    )

    assert await asyncio.to_thread(cli.cmd_storage_health, args) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["lighthouse"]["status"] == "error"
    assert "cannot be resolved" in report["lighthouse"]["message"]
    assert report["gcs"]["status"] == "not_configured"
    assert report["sovereign_ipfs"]["status"] == "decommissioned"


# The ciphertext an agent's stored Lighthouse key is damaged to: one that does
# not decrypt (a damaged row, or another KESTREL_DATA_KEY), or none at all in
# a row that is still active. Its key storage raises on lookup either way.
DAMAGED_KEYS = {
    "undecryptable": base64.b64encode(b"not a ciphertext").decode(),
    "empty": "",
}


async def _make_agent_key_unreadable(raw, did, how="undecryptable"):
    """The agent's stored Lighthouse key stops being readable, as
    ``DAMAGED_KEYS[how]`` leaves it."""
    await raw.db.execute_commit(
        "UPDATE agent_service_keys SET encrypted_key = ? "
        "WHERE agent_did = ? AND provider_id = 'lighthouse'",
        (DAMAGED_KEYS[how], did),
    )


def _account_state(account):
    return sorted(json.dumps(record, sort_keys=True) for record in account.records())


@pytest.mark.asyncio
@pytest.mark.parametrize("how", sorted(DAMAGED_KEYS))
async def test_an_unreadable_agent_key_is_never_replaced_by_the_hosts(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, capsys, how
):
    """The agent's own Lighthouse key stops decrypting, or its row loses its
    ciphertext, while the host's LIGHTHOUSE_API_KEY names another account.
    Nothing reads past it to that account for the agent's exports: the export
    is refused, nothing is published or thinned on either account, and both
    health checks report the credential error. The agent's snapshots, which
    use the host key whatever the agent's own, carry on there (#2874)."""
    own = lighthouse.service.account(AGENT_KEY)
    host = tmp_path / "host"
    # Before it held a key of its own the agent published an export to the
    # host account, which an export or a thinning there would change.
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        await _export(agent, raw)
    host_exports = _export_records(lighthouse)
    assert host_exports
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        await _export(agent, raw)
        on_own = _account_state(own)
        receipts = await agent.storage.get_nodes_by_type("sovereignty_receipt")
        charged = agent.wallet.transfer.await_count
        service = _snapshot_service(agent)

        await _make_agent_key_unreadable(raw, AGENT_A, how)

        refused = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs", encrypt=True
        )
        lighthouse.now += timedelta(days=1)
        await raw.add_conversation("user", "after the key broke", metadata={})
        snapshot = await _force_snapshot(service)
        status = await _status(agent)
        receipts_after = await agent.storage.get_nodes_by_type("sovereignty_receipt")
        charged_after = agent.wallet.transfer.await_count

    assert refused.status is ToolResultStatus.ERROR, refused
    assert "cannot be read" in refused.error
    assert "host's key was not used" in refused.error
    # Nothing was stored, recorded or charged for the refused export.
    assert len(receipts_after) == len(receipts)
    assert charged_after == charged
    assert status["status"] == "error", status
    assert "cannot be read" in status["message"]
    # Neither account's exports changed: no upload and no thinning.
    assert _account_state(own) == on_own
    assert _export_records(lighthouse) == host_exports
    # The snapshot went to the host account, as every snapshot does.
    assert snapshot.success, snapshot
    assert _snapshot_cids(lighthouse) == {snapshot.metadata["cid"]}

    args = cli.build_parser().parse_args(
        ["storage", "health", "--agent", NAMES[AGENT_A], "--json"]
    )
    assert await asyncio.to_thread(cli.cmd_storage_health, args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["lighthouse"]["status"] == "error"
    assert "cannot be read" in report["lighthouse"]["message"]
    # The snapshot on the host account is still reported beside the error.
    assert report["lighthouse"]["details"]["cid"] == snapshot.metadata["cid"]


@pytest.mark.asyncio
async def test_a_key_that_breaks_after_publishing_thins_nothing(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """The agent's own key stops decrypting just after an export lands. The
    thinning pass that follows prunes on no account, the host's included,
    and says why; the export it would have thinned stays (#2874)."""
    own = lighthouse.service.account(AGENT_KEY)
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        first = await _export(agent, raw)
    # The host account holds two older exports of this agent, from before it
    # held a key of its own: thinning there would delete the older one.
    async with _host(host, monkeypatch, did=AGENT_A, lighthouse=False) as (agent, raw):
        await ServiceKeyStorage(raw.db, AGENT_A).delete_key("lighthouse")
    on_host_exports = set()
    for _ in range(2):
        lighthouse.now += timedelta(days=1)
        lighthouse.fail_deletes = True  # each export's own thinning keeps both
        async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
            on_host_exports.add(await _export(agent, raw))
        lighthouse.fail_deletes = False
    on_host = _account_state(lighthouse)
    lighthouse.now += timedelta(days=1)
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):

        async def break_the_key():
            own.after_upload = None
            await _make_agent_key_unreadable(raw, AGENT_A)

        own.after_upload = break_the_key
        result = await _export_result(agent, raw)

    # No account was pruned: the host's still holds both of its exports.
    assert on_host_exports <= lighthouse.cids()
    assert _account_state(lighthouse) == on_host
    assert {first, result.data["cid"]} <= own.cids()
    retention = result.data["export_retention"]
    assert retention["deleted"] == 0
    assert "cannot be read" in retention["error"]


@pytest.mark.asyncio
async def test_an_agent_without_a_key_of_its_own_still_uses_the_hosts(
    tmp_path, monkeypatch, kubo, lighthouse, data_key
):
    """Holding no key is not holding an unreadable one: an agent with no
    Lighthouse key of its own publishes, snapshots and is checked on the
    host's account, as before (#2874)."""
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A) as (agent, raw):
        assert agent.lighthouse_provider is not None
        cid = await _export(agent, raw)
        snapshot = await _snapshot(agent, raw, "a change")
        status = await _status(agent)

    assert {cid, snapshot.metadata["cid"]} <= lighthouse.cids()
    assert status["cid"] == cid
    report = await _health(host, AGENT_A, lighthouse)
    assert report.lighthouse.details["sovereignty_export"]["cid"] == cid


@pytest.mark.asyncio
@pytest.mark.parametrize("how", sorted(DAMAGED_KEYS))
async def test_an_agent_key_unreadable_at_boot_builds_no_provider_and_says_why(
    tmp_path, monkeypatch, kubo, lighthouse, data_key, how
):
    """Boot meets the unreadable key: it builds no provider rather than one on
    the host's key, the export stays on this host and says so, and the
    agent's health check reports why it has no provider. Its snapshots,
    which use the host's key, carry on (#2874)."""
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        await _make_agent_key_unreadable(raw, AGENT_A, how)
    async with _host(host, monkeypatch, did=AGENT_A, agent_key=AGENT_KEY) as (agent, raw):
        status = await _status(agent)
        for role, content in MESSAGES:
            await raw.add_conversation(role, content, metadata={})
        result = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs", encrypt=True
        )
        # The policy permits Lighthouse storage, so boot still registers the
        # snapshot target, which never used the agent's key.
        snapshot = await _snapshot(agent, raw, "the agent's key is unreadable")

    assert agent.lighthouse_provider is None
    assert agent.lighthouse_provider_error.startswith(
        f"{KeyUnreadableError.__name__}: "
    )
    assert status["status"] == "error", status
    assert "cannot be read" in status["message"]
    assert result.status is ToolResultStatus.PARTIAL, result
    assert result.data["tier"] == "local_only"
    assert _export_records(lighthouse) == []
    assert _snapshot_cids(lighthouse) == {snapshot.metadata["cid"]}
