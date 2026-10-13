"""The IPFS tier's pinning service when the adapter's node is unreachable (#2874)."""

import asyncio
import json
import threading
import zlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.features.sovereignty.feature import _LighthousePinning
from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier

UPLOAD_NAME = "kestrel_sovereignty__did:example:a__20261012_000000.bin"


class _Pinning:
    provider_name = "lighthouse"

    def __init__(self):
        self.calls = []

    def pin(self, payload, *, filename, on_progress=None):
        self.calls.append((payload, filename))
        return "bafypinned"


class _Response:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _adapter(tmp_path, *, node_online):
    added = []

    def post(url, *, files=None, params=None, timeout=None, stream=False):
        if not node_online:
            raise ConnectionError("IPFS daemon unreachable")
        if url.endswith("/api/v0/version"):
            return _Response({"Version": "test"})
        if url.endswith("/api/v0/add"):
            added.append(files["file"][0])
            return _Response({"Hash": "bafynode"})
        raise AssertionError(url)

    patcher = patch("kestrel_sovereign.filecoin_adapter.requests.post", side_effect=post)
    patcher.start()
    adapter = FilecoinAdapter(cache_dir=str(tmp_path / "cache"))
    return adapter, added, patcher


def test_an_unreachable_node_publishes_through_the_pinning_service(tmp_path):
    adapter, added, patcher = _adapter(tmp_path, node_online=False)
    pinning = _Pinning()
    try:
        result = adapter.store_content(
            b"backup", storage_tier=StorageTier.IPFS,
            upload_name=UPLOAD_NAME, ipfs_pinning=pinning,
        )
    finally:
        patcher.stop()

    [(payload, filename)] = pinning.calls
    assert zlib.decompress(payload) == b"backup"
    assert filename == UPLOAD_NAME
    assert result.storage_tier is StorageTier.IPFS
    assert result.ipfs_cid == "bafypinned"
    assert result.provider == "lighthouse"
    assert result.filename == UPLOAD_NAME
    assert added == []
    meta = json.loads((tmp_path / "cache" / f"{result.cache_name}.meta").read_text())
    assert meta["_kestrel_tier"] == "ipfs"
    assert meta["_kestrel_cid"] == "bafypinned"


def test_a_reachable_node_takes_the_upload_under_its_name(tmp_path):
    adapter, added, patcher = _adapter(tmp_path, node_online=True)
    pinning = _Pinning()
    try:
        result = adapter.store_content(
            b"backup", storage_tier=StorageTier.IPFS,
            upload_name=UPLOAD_NAME, ipfs_pinning=pinning,
        )
        unnamed = adapter.store_content(b"other", storage_tier=StorageTier.IPFS)
    finally:
        patcher.stop()

    assert pinning.calls == []
    assert added == [UPLOAD_NAME, "content.bin"]
    assert result.ipfs_cid == "bafynode"
    assert result.filename == UPLOAD_NAME
    assert unnamed.filename is None


def test_the_pinning_service_needs_the_name_it_pins_under(tmp_path):
    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    pinning = _Pinning()
    try:
        with pytest.raises(ValueError, match="upload_name"):
            adapter.store_content(b"backup", storage_tier=StorageTier.IPFS, ipfs_pinning=pinning)
    finally:
        patcher.stop()

    assert pinning.calls == []
    assert list((tmp_path / "cache").iterdir()) == []


@pytest.mark.parametrize("tier", [StorageTier.LOCAL_ONLY, StorageTier.FILECOIN])
def test_only_the_ipfs_tier_consults_the_pinning_service(tmp_path, tier):
    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    pinning = _Pinning()
    try:
        result = adapter.store_content(
            b"backup", storage_tier=tier, upload_name=UPLOAD_NAME, ipfs_pinning=pinning,
        )
    finally:
        patcher.stop()

    assert pinning.calls == []
    assert result.storage_tier is StorageTier.LOCAL_ONLY
    assert result.ipfs_cid is None


@pytest.mark.asyncio
async def test_lighthouse_pinning_uploads_on_the_agent_loop_from_a_worker():
    provider = MagicMock(provider_name="lighthouse")
    provider.publish = AsyncMock(return_value="bafylh")
    pinning = _LighthousePinning(
        provider, tag="kestrel-sovereignty-did:example:a", loop=asyncio.get_running_loop()
    )

    cid = await asyncio.to_thread(pinning.pin, b"payload", filename=UPLOAD_NAME)

    assert cid == "bafylh"
    assert pinning.provider_name == "lighthouse"
    provider.publish.assert_awaited_once_with(
        b"payload",
        filename=UPLOAD_NAME,
        tag="kestrel-sovereignty-did:example:a",
        on_progress=None,
    )


def test_lighthouse_pinning_refuses_to_block_the_agent_loop():
    """Called on the loop it uploads on, ``pin`` would wait on itself. The
    call runs on its own daemon thread so a regression fails here, by
    timeout, rather than hanging the suite."""
    provider = MagicMock(provider_name="lighthouse")
    provider.publish = AsyncMock(return_value="bafylh")
    outcome = {}

    def call_pin_on_its_loop():
        loop = asyncio.new_event_loop()
        pinning = _LighthousePinning(provider, tag="t", loop=loop)

        async def on_the_loop():
            return pinning.pin(b"payload", filename=UPLOAD_NAME)

        try:
            outcome["cid"] = loop.run_until_complete(on_the_loop())
        except RuntimeError as exc:
            outcome["error"] = exc
        finally:
            loop.close()

    caller = threading.Thread(target=call_pin_on_its_loop, daemon=True)
    caller.start()
    caller.join(timeout=10)

    assert not caller.is_alive(), "pin blocked the loop it uploads on"
    assert "worker thread" in str(outcome["error"])
    provider.publish.assert_not_called()
