"""An encrypted sovereignty export restores on a fresh host (#2873).

The disaster a backup exists for is the loss of the host or its database, and
with it the ``backup_artifact`` record that used to hold the only copy of an
encrypted export's key hash. Every test here exports through
``SovereigntyFeature.export_sovereignty`` and restores through
``SovereigntyFeature.import_sovereignty`` alone: nothing calls
``retrieve_content(key_hash=...)``, which is how an earlier attempt passed
without ever exercising the gap.

The "fresh host" is a second database, created empty, and a second working
directory, so the export host's ``storage_cache`` is out of reach. IPFS is an
in-memory Kubo daemon answering the three RPCs ``FilecoinAdapter`` makes.
"""

from __future__ import annotations

import hashlib
import os
import tracemalloc
import zlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.sovereignty import feature as sovereignty_feature
from kestrel_sovereign.features.sovereignty.feature import (
    REFUSAL_DECRYPTION_FAILED,
    REFUSAL_ENCRYPTION_KEY_MISMATCH,
    REFUSAL_EXPORT_EXCEEDS_IMPORT_LIMIT,
    REFUSAL_EXPORT_OF_ANOTHER_AGENT,
    REFUSAL_LEGACY_DECRYPTION_METADATA_UNAVAILABLE,
    REFUSAL_UNENCRYPTED_WITHOUT_RECEIPT,
    SovereigntyFeature,
)
from kestrel_sovereign.filecoin_adapter import (
    ENVELOPE_MAGIC,
    FilecoinAdapter,
    StorageResult,
    StorageTier,
)
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

AGENT_DID = "did:pkh:eip155:1:0x2873fresh"
NEIGHBOUR_DID = "did:pkh:eip155:1:0x2873neighbour"
MESSAGES = [
    ("user", "my sister's name is Ingrid"),
    ("assistant", "I'll remember that your sister is Ingrid."),
    ("user", "and the spare key is under the blue pot"),
]


class _Response:
    def __init__(self, status_code: int, *, body: bytes = b"", json_body=None,
                 declare_length: bool = True):
        self.status_code = status_code
        self.content = body
        self.headers = {"Content-Length": str(len(body))} if declare_length else {}
        self._json = json_body

    def json(self):
        return self._json

    def iter_content(self, chunk_size=65536):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]

    def close(self):
        pass


class _FakeKubo:
    """An in-memory IPFS daemon: ``version``, ``add`` and ``cat``."""

    def __init__(self):
        self.blocks: dict[str, bytes] = {}
        # A gateway need not say how long its answer is.
        self.declare_length = True

    def post(self, url, *, files=None, params=None, timeout=None, stream=False):
        rpc = url.split("/api/v0/", 1)[1]
        if rpc == "version":
            return _Response(200, json_body={"Version": "0.0.0-test"})
        if rpc == "add":
            payload = files["file"][1]
            data = payload.read() if hasattr(payload, "read") else payload
            cid = "bafy" + hashlib.sha256(data).hexdigest()
            self.blocks[cid] = data
            return _Response(200, json_body={"Hash": cid})
        if rpc == "cat":
            data = self.blocks.get(params["arg"])
            if data is None:
                return _Response(500, body=b"block not found")
            return _Response(200, body=data, declare_length=self.declare_length)
        raise AssertionError(f"unexpected Kubo RPC {rpc}")

    def payload(self, cid: str) -> bytes:
        """The decompressed payload an export put on IPFS."""
        return zlib.decompress(self.blocks[cid])

    def replace_payload(self, cid: str, payload: bytes) -> None:
        self.blocks[cid] = zlib.compress(payload)


@pytest.fixture
def kubo():
    daemon = _FakeKubo()
    with patch("kestrel_sovereign.filecoin_adapter.requests.post", side_effect=daemon.post):
        yield daemon


@pytest.fixture
def data_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("KESTREL_DATA_KEY", key)
    return key


@asynccontextmanager
async def _host(directory, monkeypatch, *, did=AGENT_DID, database="kestrel_prime.db"):
    """One host: its own working directory (so its own ``storage_cache``)
    and its own database, wrapped the way ``KestrelAgent`` wraps it. Agents
    co-hosted in one directory share its cache and keep their databases."""
    directory.mkdir(exist_ok=True)
    monkeypatch.chdir(directory)
    raw = Storage(str(directory / database), agent_id=did)
    await raw.initialize()
    wallet = MagicMock()
    wallet.can_afford.return_value = True
    wallet.transfer = AsyncMock()
    agent = SimpleNamespace(
        agent_id=did,
        storage=PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL),
        features={},
        wallet=wallet,
    )
    try:
        yield agent, raw
    finally:
        await raw.close()


async def _export(tmp_path, monkeypatch, *, encrypt=True):
    """Export a known history from the origin host; return the CID."""
    async with _host(tmp_path / "origin", monkeypatch) as (agent, raw):
        for role, content in MESSAGES:
            await raw.add_conversation(role, content, metadata={})
        result = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs", encrypt=encrypt
        )
    assert result.status is ToolResultStatus.OK, result
    assert result.data["tier"] == "ipfs"
    assert result.data["encrypted"] is encrypt
    return result.data["cid"]


async def _history(raw):
    return [(m["role"], m["content"]) for m in await raw.get_conversation_history()]


@pytest.mark.asyncio
async def test_encrypted_export_restores_into_a_fresh_database(
    tmp_path, monkeypatch, kubo, data_key
):
    cid = await _export(tmp_path, monkeypatch)

    # What travels is ciphertext in the envelope, never the history.
    payload = kubo.payload(cid)
    assert payload.startswith(ENVELOPE_MAGIC)
    assert b"blue pot" not in payload

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        # The disaster case: this database has never heard of the export.
        assert await agent.storage.get_nodes_by_type("backup_artifact") == []
        assert await _history(raw) == []

        result = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert result.status is ToolResultStatus.OK, result
        assert result.data["messages_restored"] == len(MESSAGES)
        assert await _history(raw) == MESSAGES


@pytest.mark.asyncio
async def test_restore_under_another_key_names_the_mismatch_and_restores_nothing(
    tmp_path, monkeypatch, kubo, data_key
):
    cid = await _export(tmp_path, monkeypatch)
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        result = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert result.status is ToolResultStatus.ERROR
        assert result.data["refusal"] == REFUSAL_ENCRYPTION_KEY_MISMATCH
        export_key_id = result.data["export_key_id"]
        configured_key_id = result.data["configured_key_id"]
        assert export_key_id != configured_key_id
        assert export_key_id in result.error and configured_key_id in result.error
        assert "Nothing was restored" in result.error
        assert await _history(raw) == []


@pytest.mark.asyncio
async def test_export_of_another_agent_is_refused_and_restores_nothing(
    tmp_path, monkeypatch, kubo, data_key
):
    """Every agent on a host shares the data key, so it decrypts any of
    their exports. Without a receipt, an export restores only into the agent
    it was sealed for: another agent's conversation rows are encrypted at
    rest under that agent's DID, so restoring them would report success and
    leave history this agent can never read."""
    cid = await _export(tmp_path, monkeypatch)

    async with _host(tmp_path / "neighbour", monkeypatch, did=NEIGHBOUR_DID) as (agent, raw):
        assert await agent.storage.get_nodes_by_type("backup_artifact") == []

        result = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert result.status is ToolResultStatus.ERROR
        assert result.data["refusal"] == REFUSAL_EXPORT_OF_ANOTHER_AGENT
        assert result.data["export_owner_id"] != result.data["agent_owner_id"]
        assert result.data["export_owner_id"] in result.error
        assert "Nothing was restored" in result.error
        assert await _history(raw) == []


def _tamper_key_hash(payload: bytes) -> bytes:
    """Flip one hex digit of the header's key hash, keeping it canonical."""
    start = payload.index(b'"key_hash":"') + len(b'"key_hash":"')
    digit = payload[start:start + 1]
    replacement = b"0" if digit != b"0" else b"1"
    return payload[:start] + replacement + payload[start + 1:]


def _tamper_version(payload: bytes) -> bytes:
    return payload.replace(b'"version":1', b'"version":2', 1)


def _tamper_token(payload: bytes) -> bytes:
    flipped = b"A" if payload[-10:-9] != b"A" else b"B"
    return payload[:-10] + flipped + payload[-9:]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper",
    [_tamper_key_hash, _tamper_version, _tamper_token],
    ids=["header-key-hash", "header-version", "ciphertext"],
)
async def test_tampered_envelope_is_refused_and_restores_nothing(
    tmp_path, monkeypatch, kubo, data_key, tamper
):
    cid = await _export(tmp_path, monkeypatch)
    original = kubo.payload(cid)
    tampered = tamper(original)
    assert tampered != original
    kubo.replace_payload(cid, tampered)

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        result = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert result.status is ToolResultStatus.ERROR
        assert result.data["refusal"] == REFUSAL_DECRYPTION_FAILED
        assert "Nothing was restored" in result.error
        assert await _history(raw) == []


@pytest.mark.asyncio
async def test_damaged_envelope_under_an_encrypted_receipt_without_a_key_hash_is_refused(
    tmp_path, monkeypatch, kubo, data_key
):
    """The receipt, not the payload, says the export is encrypted. An
    envelope whose magic bytes are damaged no longer looks like ciphertext,
    and a CID lookup skips the content-hash check, so a receipt holding no
    key hash must still refuse the payload rather than hand it to the
    restore as plaintext (where a non-tar blob "restores" 0 messages OK)."""
    async with _host(tmp_path / "origin", monkeypatch) as (agent, raw):
        for role, content in MESSAGES:
            await raw.add_conversation(role, content, metadata={})
        exported = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs", encrypt=True
        )
    assert exported.status is ToolResultStatus.OK, exported
    cid = exported.data["cid"]
    payload = kubo.payload(cid)
    assert payload.startswith(ENVELOPE_MAGIC)
    kubo.replace_payload(cid, b"X" + payload[1:])

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        # A receipt that records an encrypted export but lost its key hash.
        await agent.storage.record_backup_artifact(
            AGENT_DID,
            StorageResult(
                content_hash=exported.data["content_hash"],
                tier=StorageTier.IPFS,
                cid=cid,
                encrypted=True,
                encryption_key_hash=None,
            ),
        )
        [receipt] = await agent.storage.get_nodes_by_type("backup_artifact")
        assert receipt.properties["encrypted"] is True
        assert receipt.properties["encryption_key_hash"] is None
        restore = AsyncMock(wraps=agent.storage.restore_from_backup_blob)
        monkeypatch.setattr(agent.storage, "restore_from_backup_blob", restore)

        result = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert result.status is ToolResultStatus.ERROR, result
        assert result.data["refusal"] == REFUSAL_DECRYPTION_FAILED
        assert "receipt records an encrypted export" in result.error
        assert "Nothing was restored" in result.error
        restore.assert_not_awaited()
        assert await _history(raw) == []


async def _legacy_export(agent, raw):
    """An encrypted export as written before the envelope (#2873): the same
    adapter call without ``envelope_owner``, receipted the same way."""
    for role, content in MESSAGES:
        await raw.add_conversation(role, content, metadata={})
    blob = await agent.storage.create_backup_blob(include_db=True)
    result = FilecoinAdapter().store_content(
        blob, storage_tier=StorageTier.IPFS, encrypt=True, metadata={"agent": AGENT_DID}
    )
    assert result.encrypted and result.ipfs_cid
    await agent.storage.record_backup_artifact(AGENT_DID, result)
    return result.ipfs_cid


@pytest.mark.asyncio
async def test_legacy_export_still_restores_with_its_backup_receipt(
    tmp_path, monkeypatch, kubo, data_key
):
    async with _host(tmp_path / "origin", monkeypatch) as (agent, raw):
        cid = await _legacy_export(agent, raw)
        assert not kubo.payload(cid).startswith(ENVELOPE_MAGIC)
        # Data loss that keeps the graph: the receipt still holds the key hash.
        await raw.db.execute_commit("DELETE FROM conversation_history")

        result = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert result.status is ToolResultStatus.OK, result
        assert result.data["messages_restored"] == len(MESSAGES)
        assert await _history(raw) == MESSAGES


@pytest.mark.asyncio
async def test_legacy_export_without_its_receipt_is_refused_not_restored_as_ciphertext(
    tmp_path, monkeypatch, kubo, data_key
):
    """Requested by its IPFS CID, a legacy export has no key to try: the key
    derives from its content hash, which neither the payload nor the CID
    carries."""
    async with _host(tmp_path / "origin", monkeypatch) as (agent, raw):
        cid = await _legacy_export(agent, raw)

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        result = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert result.status is ToolResultStatus.ERROR
        assert result.data["refusal"] == REFUSAL_LEGACY_DECRYPTION_METADATA_UNAVAILABLE
        assert result.data["decryption_attempted"] is False
        assert "legacy encrypted export" in result.error
        assert await _history(raw) == []


async def _legacy_export_named_by_content_hash(tmp_path, monkeypatch, kubo):
    """A legacy export the IPFS node also answers for under its content hash:
    the name ``export_sovereignty`` reports as the CID of an export no IPFS
    node returned a CID for, and the only record-free source of the key hash
    a legacy payload's key derives from."""
    async with _host(tmp_path / "origin", monkeypatch) as (agent, raw):
        cid = await _legacy_export(agent, raw)
        [receipt] = await agent.storage.get_nodes_by_type("backup_artifact")
    # A backup_artifact node is keyed by its content hash.
    content_hash = receipt.node_id
    assert content_hash != cid and len(content_hash) == 64
    assert not kubo.payload(cid).startswith(ENVELOPE_MAGIC)
    kubo.blocks[content_hash] = kubo.blocks[cid]
    return content_hash


@pytest.mark.asyncio
async def test_legacy_export_without_its_receipt_decrypts_with_the_configured_key(
    tmp_path, monkeypatch, kubo, data_key
):
    content_hash = await _legacy_export_named_by_content_hash(tmp_path, monkeypatch, kubo)

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        assert await agent.storage.get_nodes_by_type("backup_artifact") == []

        result = await SovereigntyFeature(agent).import_sovereignty(content_hash)

        assert result.status is ToolResultStatus.OK, result
        assert result.data["messages_restored"] == len(MESSAGES)
        assert await _history(raw) == MESSAGES


@pytest.mark.asyncio
async def test_legacy_export_the_configured_key_cannot_decrypt_is_refused(
    tmp_path, monkeypatch, kubo, data_key
):
    content_hash = await _legacy_export_named_by_content_hash(tmp_path, monkeypatch, kubo)
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        result = await SovereigntyFeature(agent).import_sovereignty(content_hash)

        assert result.status is ToolResultStatus.ERROR
        assert result.data["refusal"] == REFUSAL_LEGACY_DECRYPTION_METADATA_UNAVAILABLE
        assert result.data["decryption_attempted"] is True
        assert "legacy encrypted export" in result.error
        assert "Nothing was restored" in result.error
        assert await _history(raw) == []


@pytest.mark.asyncio
async def test_unencrypted_export_without_its_receipt_is_still_refused(
    tmp_path, monkeypatch, kubo, data_key
):
    """Plaintext exports are unchanged: without a receipt nothing vouches
    for them, so a fresh database does not take one."""
    cid = await _export(tmp_path, monkeypatch, encrypt=False)

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        result = await SovereigntyFeature(agent).import_sovereignty(cid)

        assert result.status is ToolResultStatus.ERROR
        assert result.data["refusal"] == REFUSAL_UNENCRYPTED_WITHOUT_RECEIPT
        assert await _history(raw) == []


@pytest.mark.asyncio
async def test_envelope_substituted_under_a_receipt_is_refused(
    tmp_path, monkeypatch, kubo, data_key
):
    """A receipt names the export it vouches for. Another export sealed under
    the same host key decrypts just as readily, so the receipt's key hash
    must bind the envelope it is handed."""
    async with _host(tmp_path / "origin", monkeypatch) as (agent, raw):
        for role, content in MESSAGES:
            await raw.add_conversation(role, content, metadata={})
        feature = SovereigntyFeature(agent)
        first = await feature.export_sovereignty(storage_tier="ipfs", encrypt=True)
        await raw.add_conversation("user", "a later message", metadata={})
        second = await feature.export_sovereignty(storage_tier="ipfs", encrypt=True)
        first_cid, second_cid = first.data["cid"], second.data["cid"]
        assert first_cid != second_cid
        kubo.replace_payload(first_cid, kubo.payload(second_cid))
        await raw.db.execute_commit("DELETE FROM conversation_history")
        # The receipt read must not be satisfied by a cached copy.
        for cached in (tmp_path / "origin" / "storage_cache").glob("*.cache"):
            cached.unlink()

        result = await feature.import_sovereignty(first_cid)

        assert result.status is ToolResultStatus.ERROR
        assert result.data["refusal"] == REFUSAL_DECRYPTION_FAILED
        assert second.data["content_hash"] in result.error
        assert first.data["content_hash"] in result.error
        assert await _history(raw) == []


@pytest.mark.asyncio
async def test_envelope_substituted_under_a_receipt_without_a_key_hash_is_refused(
    tmp_path, monkeypatch, kubo, data_key
):
    """A receipt that lost its key hash still names its export: a
    ``backup_artifact`` node's id is the content hash. Another export of the
    same agent, sealed under the same key, must not restore under it."""
    async with _host(tmp_path / "origin", monkeypatch) as (agent, raw):
        for role, content in MESSAGES:
            await raw.add_conversation(role, content, metadata={})
        feature = SovereigntyFeature(agent)
        first = await feature.export_sovereignty(storage_tier="ipfs", encrypt=True)
        await raw.add_conversation("user", "a later message", metadata={})
        second = await feature.export_sovereignty(storage_tier="ipfs", encrypt=True)
    first_cid, second_cid = first.data["cid"], second.data["cid"]
    first_payload = kubo.payload(first_cid)
    kubo.replace_payload(first_cid, kubo.payload(second_cid))

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        await agent.storage.record_backup_artifact(
            AGENT_DID,
            StorageResult(
                content_hash=first.data["content_hash"],
                tier=StorageTier.IPFS,
                cid=first_cid,
                encrypted=True,
                encryption_key_hash=None,
            ),
        )
        restore = AsyncMock(wraps=agent.storage.restore_from_backup_blob)
        monkeypatch.setattr(agent.storage, "restore_from_backup_blob", restore)

        refused = await SovereigntyFeature(agent).import_sovereignty(first_cid)

        assert refused.status is ToolResultStatus.ERROR, refused
        assert refused.data["refusal"] == REFUSAL_DECRYPTION_FAILED
        assert second.data["content_hash"] in refused.error
        assert first.data["content_hash"] in refused.error
        assert "Nothing was restored" in refused.error
        restore.assert_not_awaited()
        assert await _history(raw) == []

        # The receipt's own export still restores under it.
        kubo.replace_payload(first_cid, first_payload)
        restored = await SovereigntyFeature(agent).import_sovereignty(first_cid)

        assert restored.status is ToolResultStatus.OK, restored
        assert await _history(raw) == MESSAGES


@pytest.mark.asyncio
async def test_two_agents_exporting_one_plaintext_on_one_host_each_restore_their_own(
    tmp_path, monkeypatch, kubo, data_key
):
    """Both agents' encrypted exports of one plaintext land in the host
    cache, which every co-hosted agent shares. Keyed by content hash alone,
    the second overwrote the first, whose restore was then refused as
    another agent's export (#2873)."""
    host = tmp_path / "host"
    async with _host(host, monkeypatch, did=NEIGHBOUR_DID, database="blank.db") as (blank, _):
        blob = await blank.storage.create_backup_blob(include_db=True)

    exported = {}
    for did, database in ((AGENT_DID, "mine.db"), (NEIGHBOUR_DID, "theirs.db")):
        async with _host(host, monkeypatch, did=did, database=database) as (agent, _):
            agent.storage.create_backup_blob = AsyncMock(return_value=blob)
            exported[did] = await SovereigntyFeature(agent).export_sovereignty(
                storage_tier="local", encrypt=True
            )
            assert exported[did].status is ToolResultStatus.OK, exported[did]
    content_hash = exported[AGENT_DID].data["content_hash"]
    assert exported[NEIGHBOUR_DID].data["content_hash"] == content_hash
    assert len(list((host / "storage_cache").glob(f"{content_hash}.*.cache"))) == 2

    for did, database in ((AGENT_DID, "mine.db"), (NEIGHBOUR_DID, "theirs.db")):
        async with _host(host, monkeypatch, did=did, database=database) as (agent, _):
            result = await SovereigntyFeature(agent).import_sovereignty(content_hash)

            assert result.status is ToolResultStatus.OK, (did, result)
            assert result.data["messages_restored"] == 0


IMPORT_LIMIT = 1024 * 1024


@pytest.fixture
def small_import_limit(monkeypatch):
    """The real ceiling is 1 GiB; the bounds scale with it."""
    monkeypatch.setattr(sovereignty_feature, "MAX_SOVEREIGNTY_IMPORT_BYTES", IMPORT_LIMIT)
    return IMPORT_LIMIT


async def _import_refused_as_too_large(agent, raw, monkeypatch, cid):
    restore = AsyncMock(wraps=agent.storage.restore_from_backup_blob)
    monkeypatch.setattr(agent.storage, "restore_from_backup_blob", restore)

    tracemalloc.start()
    try:
        result = await SovereigntyFeature(agent).import_sovereignty(cid)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.status is ToolResultStatus.ERROR, result
    assert result.data["refusal"] == REFUSAL_EXPORT_EXCEEDS_IMPORT_LIMIT
    assert result.data["limit_bytes"] == IMPORT_LIMIT
    assert f"{IMPORT_LIMIT}-byte limit" in result.error
    assert "Nothing was restored" in result.error
    restore.assert_not_awaited()
    assert await _history(raw) == []
    return peak


async def _receipt_for(agent, cid):
    """A receipt recording an encrypted export under this CID."""
    content_hash = hashlib.sha256(cid.encode()).hexdigest()
    await agent.storage.record_backup_artifact(
        AGENT_DID,
        StorageResult(
            content_hash=content_hash,
            tier=StorageTier.IPFS,
            cid=cid,
            encrypted=True,
            encryption_key_hash=content_hash,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("declare_length", [True, False], ids=["declared", "streamed"])
@pytest.mark.parametrize("with_receipt", [False, True], ids=["fresh-host", "receipt"])
async def test_oversized_payload_is_refused_before_it_is_read_whole(
    tmp_path, monkeypatch, kubo, data_key, small_import_limit, declare_length, with_receipt
):
    """A caller-supplied CID can name anything. The read stops at the
    ceiling, whether or not the gateway says how long its answer is."""
    cid = "bafyoversized"
    kubo.blocks[cid] = os.urandom(3 * IMPORT_LIMIT)
    kubo.declare_length = declare_length

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        if with_receipt:
            await _receipt_for(agent, cid)
        peak = await _import_refused_as_too_large(agent, raw, monkeypatch, cid)

    assert peak < 4 * IMPORT_LIMIT + 1024 * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize("with_receipt", [False, True], ids=["fresh-host", "receipt"])
async def test_compression_bomb_is_refused_while_it_is_decompressed(
    tmp_path, monkeypatch, kubo, data_key, small_import_limit, with_receipt
):
    """64 MiB of zeros compresses to about 64 KB, well inside the read
    limit. Decompression stops at the ceiling instead of inflating it."""
    inflated = b"\0" * (64 * IMPORT_LIMIT)
    cid = "bafybomb"
    kubo.blocks[cid] = zlib.compress(inflated, 9)
    del inflated
    assert len(kubo.blocks[cid]) < IMPORT_LIMIT

    async with _host(tmp_path / "fresh", monkeypatch) as (agent, raw):
        if with_receipt:
            await _receipt_for(agent, cid)
        peak = await _import_refused_as_too_large(agent, raw, monkeypatch, cid)

    # Inflating it whole would allocate 64 MiB.
    assert peak < 8 * IMPORT_LIMIT
