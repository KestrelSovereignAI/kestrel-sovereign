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
from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier
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
async def test_import_without_a_receipt_is_refused_before_any_read():
    feature = _feature(receipts=[])
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter") as adapter_cls:
        result = await feature.import_sovereignty(HASH)
    assert result.status.value == "error"
    # Same message as an artifact that does not exist: not an oracle.
    assert result.error == f"Could not retrieve content for CID {HASH}"
    adapter_cls.return_value.retrieve_content.assert_not_called()
    feature.agent.storage.restore_from_backup_blob.assert_not_awaited()


@pytest.mark.asyncio
async def test_import_with_a_receipt_reads_and_uses_its_key_hash():
    receipt = SimpleNamespace(node_id=HASH, properties={"encryption_key_hash": "kh", "encrypted": True})
    feature = _feature(receipts=[receipt])
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter") as adapter_cls:
        adapter_cls.return_value.retrieve_content = MagicMock(return_value=b"blob")
        result = await feature.import_sovereignty(HASH)
    assert result.status.value == "ok", result
    adapter_cls.return_value.retrieve_content.assert_called_once_with(HASH, ipfs_cid=HASH, key_hash="kh")


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
