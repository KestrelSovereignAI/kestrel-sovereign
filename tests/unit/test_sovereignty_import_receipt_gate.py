"""Reading the host cache by a caller-supplied name needs a receipt (#3225).

Review round 1 found the browser's invariant open one door over:
`FilecoinAdapter.retrieve_content` reads ``storage_cache/{name}.cache`` by
whatever name it is given, and both `!import-sovereignty <cid>` and the
identity import reach it with a caller-supplied name. A 64-hex content
hash passes the CID pattern, so any co-hosted agent could restore another
agent's backup into its own database, decrypt another agent's never-
published identity package under F187 (where ``key_hash`` *is* the hash),
or learn by the error text which hashes exist.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.features.sovereignty.feature import (
    MAX_SOVEREIGNTY_IMPORT_BYTES,
    REFUSAL_BACKUP_RECEIPT_INVALID,
    SovereigntyFeature,
    _LighthouseGatewayIpfs,
)
from kestrel_sovereign.filecoin_adapter import EnvelopePolicy, FilecoinAdapter, StorageTier
from kestrel_sovereign.identity import package_intake
from kestrel_sovereign.privacy import PrivacyMode

HASH = "a" * 64


class _LighthouseGateway:
    """Equal to the keyless Lighthouse gateway reader an import is given,
    which needs no provider on the agent (#3601)."""

    def __eq__(self, other):
        return isinstance(other, _LighthouseGatewayIpfs)

    def __repr__(self):
        return "<the Lighthouse gateway reader>"


def _feature(receipts, privacy_mode=PrivacyMode.NORMAL):
    storage = SimpleNamespace(
        get_nodes_by_type=AsyncMock(return_value=receipts),
        restore_from_backup_blob=AsyncMock(return_value={"messages_restored": 1}),
        privacy_mode=privacy_mode,
    )
    agent = SimpleNamespace(storage=storage, features={}, agent_id="did:test:me", did="did:test:me")
    return SovereigntyFeature(agent)


@pytest.mark.asyncio
async def test_import_without_a_receipt_never_reads_the_host_cache():
    """A restore with no receipt (a fresh host, #2873) may read only IPFS,
    and only an encrypted export is accepted from it. IPFS includes the
    Lighthouse gateway, though this agent has no Lighthouse provider (#3601)."""
    feature = _feature(receipts=[])
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter") as adapter_cls:
        adapter_cls.return_value.retrieve_content = MagicMock(side_effect=ValueError(f"Content not found: {HASH}"))
        result = await feature.import_sovereignty(HASH)
    assert result.status.value == "error"
    adapter_cls.return_value.retrieve_content.assert_called_once_with(
        HASH,
        ipfs_cid=HASH,
        max_output_bytes=MAX_SOVEREIGNTY_IMPORT_BYTES,
        allow_local_cache=False,
        envelope=EnvelopePolicy.ENCRYPTED_ONLY,
        envelope_owner="did:test:me",
        remote_ipfs=_LighthouseGateway(),
    )
    feature.agent.storage.restore_from_backup_blob.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "privacy_mode",
    [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED, None],
    ids=["ephemeral", "isolated", "deidentified", "no-privacy-mode"],
)
@pytest.mark.parametrize("receipts", [[], [SimpleNamespace(node_id=HASH, properties={})]],
                         ids=["fresh-host", "receipt"])
async def test_a_privacy_mode_that_keeps_storage_local_reads_no_gateway(
    privacy_mode, receipts
):
    """The gateway read needs no key, but it tells Lighthouse which CID this
    host wants, so a privacy mode that permits no remote tier makes none
    (#3601). A storage that exposes no privacy mode is not taken to permit
    one."""
    feature = _feature(receipts=receipts, privacy_mode=privacy_mode)
    if privacy_mode is None:
        del feature.agent.storage.privacy_mode
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter") as adapter_cls:
        adapter_cls.return_value.retrieve_content = MagicMock(
            side_effect=ValueError(f"Content not found: {HASH}")
        )
        await feature.import_sovereignty(HASH)
    [call] = adapter_cls.return_value.retrieve_content.call_args_list
    assert call.kwargs["remote_ipfs"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy_mode", [PrivacyMode.ANONYMOUS, PrivacyMode.PUBLIC])
async def test_a_privacy_mode_that_permits_a_remote_tier_reads_the_gateway(privacy_mode):
    feature = _feature(receipts=[], privacy_mode=privacy_mode)
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter") as adapter_cls:
        adapter_cls.return_value.retrieve_content = MagicMock(return_value=b"blob")
        await feature.import_sovereignty(HASH)
    [call] = adapter_cls.return_value.retrieve_content.call_args_list
    assert call.kwargs["remote_ipfs"] == _LighthouseGateway()


@pytest.mark.asyncio
@pytest.mark.parametrize("seal_envelope", [True, False], ids=["envelope", "legacy"])
async def test_import_without_a_receipt_cannot_restore_a_cached_export(
    tmp_path, monkeypatch, seal_envelope
):
    """Another agent's never-published export sits in the host cache,
    encrypted under the host key every co-hosted agent shares. Naming its
    content hash with no receipt must not restore it: the cache does not
    answer. A legacy export is the sharper case, because the configured key
    does decrypt it under that name (#2873)."""
    from cryptography.fernet import Fernet

    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    monkeypatch.chdir(tmp_path)
    with patch(
        "kestrel_sovereign.filecoin_adapter.requests.post",
        side_effect=ConnectionError("no IPFS daemon"),
    ):
        stored = FilecoinAdapter().store_content(
            b"another agent's history",
            storage_tier=StorageTier.LOCAL_ONLY,
            encrypt=True,
            envelope_owner="did:test:other" if seal_envelope else None,
        )
        feature = _feature(receipts=[])
        result = await feature.import_sovereignty(stored.content_hash)
    assert result.status.value == "error"
    assert "Content not found" in result.error
    feature.agent.storage.restore_from_backup_blob.assert_not_awaited()


ENVELOPE_NAME = f"{HASH}.{'b' * 32}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("properties", "key_hash", "policy", "envelope_key_hash", "cache_name"),
    [
        # A receipt written before the cache name was recorded holds none: its
        # export is cached under its content hash, the receipt's node id,
        # never under the CID it is requested by (#2873).
        ({"encryption_key_hash": "kh", "encrypted": True}, "kh", EnvelopePolicy.ENCRYPTED_ONLY, "kh", HASH),
        # An encrypted receipt that lost its key hash still refuses a payload
        # that does not look like ciphertext (a damaged envelope magic), and
        # still binds an envelope to the content hash it is keyed by.
        ({"encryption_key_hash": None, "encrypted": True}, None, EnvelopePolicy.ENCRYPTED_ONLY, HASH, HASH),
        # A receipt that does not say it is plaintext is not taken as one.
        ({}, None, EnvelopePolicy.ENCRYPTED_ONLY, HASH, HASH),
        ({"encryption_key_hash": "kh", "encrypted": False}, "kh", EnvelopePolicy.ENCRYPTED_ONLY, "kh", HASH),
        # Only a receipt recording a plaintext export accepts plaintext.
        ({"encryption_key_hash": None, "encrypted": False}, None, EnvelopePolicy.OPEN, HASH, HASH),
        # The cache is read under the name the receipt records (#2873).
        (
            {"encryption_key_hash": HASH, "encrypted": True, "cache_name": ENVELOPE_NAME},
            HASH, EnvelopePolicy.ENCRYPTED_ONLY, HASH, ENVELOPE_NAME,
        ),
        (
            {"encryption_key_hash": None, "encrypted": False, "cache_name": HASH},
            None, EnvelopePolicy.OPEN, HASH, HASH,
        ),
    ],
    ids=[
        "encrypted", "encrypted-no-key-hash", "unrecorded", "contradictory", "plaintext",
        "envelope-cache-name", "plaintext-cache-name",
    ],
)
async def test_import_with_a_receipt_follows_what_it_records(
    properties, key_hash, policy, envelope_key_hash, cache_name
):
    receipt = SimpleNamespace(node_id=HASH, properties=properties)
    feature = _feature(receipts=[receipt])
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter") as adapter_cls:
        adapter_cls.return_value.retrieve_content = MagicMock(return_value=b"blob")
        result = await feature.import_sovereignty(HASH)
    assert result.status.value == "ok", result
    adapter_cls.return_value.retrieve_content.assert_called_once_with(
        HASH,
        ipfs_cid=HASH,
        key_hash=key_hash,
        max_output_bytes=MAX_SOVEREIGNTY_IMPORT_BYTES,
        envelope=policy,
        envelope_owner="did:test:me",
        envelope_key_hash=envelope_key_hash,
        cache_name=cache_name,
        remote_ipfs=_LighthouseGateway(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("node_id", "cache_name"),
    [(HASH, HASH), ("not-a-content-hash", None), (HASH.upper(), None)],
    ids=["content-hash", "other-node-id", "uppercase-node-id"],
)
async def test_a_receipt_without_a_cache_name_reads_its_content_hash_not_its_cid(
    node_id, cache_name
):
    """A receipt written before the cache name was recorded names its export
    by its node id, the content hash every export was then cached under.
    The CID it is requested by is never a cache name; a node id that is not
    a content hash names no cache entry."""
    receipt = SimpleNamespace(
        node_id=node_id,
        properties={"ipfs_cid": "bafy-export", "encryption_key_hash": "kh", "encrypted": True},
    )
    feature = _feature(receipts=[receipt])
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter") as adapter_cls:
        adapter_cls.return_value.retrieve_content = MagicMock(return_value=b"blob")
        result = await feature.import_sovereignty("bafy-export")
    assert result.status.value == "ok", result
    adapter_cls.return_value.retrieve_content.assert_called_once_with(
        "bafy-export",
        ipfs_cid="bafy-export",
        key_hash="kh",
        max_output_bytes=MAX_SOVEREIGNTY_IMPORT_BYTES,
        envelope=EnvelopePolicy.ENCRYPTED_ONLY,
        envelope_owner="did:test:me",
        envelope_key_hash="kh",
        cache_name=cache_name,
        remote_ipfs=_LighthouseGateway(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cache_name",
    [
        f"{'c' * 64}.{'b' * 32}",  # another export's envelope
        "c" * 64,  # another export
        "../../etc/passwd",
        f"{HASH}.{'b' * 31}",
        f"{HASH}.{'B' * 32}",
        f"{HASH}.",
        HASH.upper(),
        "",
        7,
    ],
)
async def test_import_refuses_a_receipt_whose_cache_name_is_not_its_exports(cache_name):
    """A receipt names the cache entry of the export it records, and only
    that: a name of another export, or no cache name at all, is never read."""
    receipt = SimpleNamespace(
        node_id=HASH,
        properties={"encryption_key_hash": HASH, "encrypted": True, "cache_name": cache_name},
    )
    feature = _feature(receipts=[receipt])
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter") as adapter_cls:
        result = await feature.import_sovereignty(HASH)
    assert result.status.value == "error"
    assert result.data["refusal"] == REFUSAL_BACKUP_RECEIPT_INVALID
    assert "Nothing was restored" in result.error
    adapter_cls.return_value.retrieve_content.assert_not_called()
    feature.agent.storage.restore_from_backup_blob.assert_not_awaited()


def test_identity_intake_never_answers_a_cid_from_the_local_cache():
    # The intake imports the adapter at call time; patch it at its source.
    with patch("kestrel_sovereign.filecoin_adapter.FilecoinAdapter") as adapter_cls:
        adapter_cls.return_value.retrieve_content = MagicMock(return_value=b'{"x": 1}')
        package_intake._retrieve_cid_identity_package(HASH, key_hash=HASH)
    _, kwargs = adapter_cls.return_value.retrieve_content.call_args
    assert kwargs["allow_local_cache"] is False


def test_adapter_honours_allow_local_cache(tmp_path):
    """With the flag off, a name that IS in the host cache is 'not found',
    exactly as an absent one — and nothing is read."""
    adapter = FilecoinAdapter(cache_dir=str(tmp_path / "cache"))
    stored = adapter.store_content(b"private-bytes", storage_tier=StorageTier.LOCAL_ONLY, encrypt=False)
    assert adapter.retrieve_content(stored.content_hash) == b"private-bytes"
    with pytest.raises(ValueError, match="Content not found"):
        adapter.retrieve_content(stored.content_hash, allow_local_cache=False)
    with pytest.raises(ValueError, match="Content not found"):
        adapter.retrieve_content("b" * 64, allow_local_cache=False)


def test_artifacts_module_imports_standalone():
    """Round 2 found a cycle: importing the ownership module before the
    endpoints package failed. A script or feature package reusing the
    predicate imports it first."""
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", "import kestrel_sovereign.features.sovereignty.artifacts as a; print(a.owned_artifacts.__name__)"],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "owned_artifacts"
