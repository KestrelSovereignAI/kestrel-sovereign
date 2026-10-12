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

from kestrel_sovereign.features.sovereignty.feature import SovereigntyFeature
from kestrel_sovereign.filecoin_adapter import EnvelopePolicy, FilecoinAdapter, StorageTier
from kestrel_sovereign.identity import package_intake

HASH = "a" * 64


def _feature(receipts):
    storage = SimpleNamespace(
        get_nodes_by_type=AsyncMock(return_value=receipts),
        restore_from_backup_blob=AsyncMock(return_value={"messages_restored": 1}),
    )
    agent = SimpleNamespace(storage=storage, features={}, agent_id="did:test:me", did="did:test:me")
    return SovereigntyFeature(agent)


@pytest.mark.asyncio
async def test_import_without_a_receipt_never_reads_the_host_cache():
    """A restore with no receipt (a fresh host, #2873) may read only IPFS,
    and only an encrypted export is accepted from it."""
    feature = _feature(receipts=[])
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter") as adapter_cls:
        adapter_cls.return_value.retrieve_content = MagicMock(side_effect=ValueError(f"Content not found: {HASH}"))
        result = await feature.import_sovereignty(HASH)
    assert result.status.value == "error"
    adapter_cls.return_value.retrieve_content.assert_called_once_with(
        HASH, ipfs_cid=HASH, allow_local_cache=False, envelope=EnvelopePolicy.ENCRYPTED_ONLY
    )
    feature.agent.storage.restore_from_backup_blob.assert_not_awaited()


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
            seal_envelope=seal_envelope,
        )
        feature = _feature(receipts=[])
        result = await feature.import_sovereignty(stored.content_hash)
    assert result.status.value == "error"
    assert "Content not found" in result.error
    feature.agent.storage.restore_from_backup_blob.assert_not_awaited()


@pytest.mark.asyncio
async def test_import_with_a_receipt_reads_and_uses_its_key_hash():
    receipt = SimpleNamespace(node_id=HASH, properties={"encryption_key_hash": "kh", "encrypted": True})
    feature = _feature(receipts=[receipt])
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter") as adapter_cls:
        adapter_cls.return_value.retrieve_content = MagicMock(return_value=b"blob")
        result = await feature.import_sovereignty(HASH)
    assert result.status.value == "ok", result
    adapter_cls.return_value.retrieve_content.assert_called_once_with(
        HASH, ipfs_cid=HASH, key_hash="kh", envelope=EnvelopePolicy.OPEN
    )


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
