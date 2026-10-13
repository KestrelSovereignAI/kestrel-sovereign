"""The IPFS tier's remote publisher and reader when the adapter's node is
unreachable (#2874, #3601)."""

import asyncio
import json
import threading
import zlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kestrel_sovereign.features.sovereignty.feature import (
    _LighthouseGatewayIpfs,
    _LighthouseIpfs,
)
from kestrel_sovereign.filecoin_adapter import (
    ContentRetrievalLimitError,
    FilecoinAdapter,
    RemoteIpfsError,
    StorageTier,
)
from tests.utils.lighthouse_gateway import GATEWAY_URL, GatewayServer

UPLOAD_NAME = "kestrel_sovereignty__did:example:a__20261012_000000.bin"


class _Pinning:
    provider_name = "lighthouse"

    def __init__(self):
        self.calls = []
        self.served = {}
        self.fetches = []

    def pin(self, payload, *, filename, on_progress=None):
        self.calls.append((payload, filename))
        self.served["bafypinned"] = payload
        return "bafypinned"

    def fetch(self, cid, *, max_bytes):
        self.fetches.append((cid, max_bytes))
        return self.served.get(cid, b"")


class _Blocks:
    """A Kubo ``cat`` response streaming ``body``."""

    status_code = 200

    def __init__(self, body):
        self._body = body
        self.headers = {"Content-Length": str(len(body))}

    def iter_content(self, chunk_size):
        yield self._body

    def close(self):
        pass


class _Response:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _adapter(tmp_path, *, node_online, node_blocks=None):
    added = []

    def post(url, *, files=None, params=None, timeout=None, stream=False):
        if not node_online:
            raise ConnectionError("IPFS daemon unreachable")
        if url.endswith("/api/v0/version"):
            return _Response({"Version": "test"})
        if url.endswith("/api/v0/add"):
            added.append(files["file"][0])
            return _Response({"Hash": "bafynode"})
        if url.endswith("/api/v0/cat"):
            return _Blocks(node_blocks[params["arg"]])
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
            upload_name=UPLOAD_NAME, remote_ipfs=pinning,
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
            upload_name=UPLOAD_NAME, remote_ipfs=pinning,
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
            adapter.store_content(b"backup", storage_tier=StorageTier.IPFS, remote_ipfs=pinning)
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
            b"backup", storage_tier=tier, upload_name=UPLOAD_NAME, remote_ipfs=pinning,
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
    pinning = _LighthouseIpfs(
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
        pinning = _LighthouseIpfs(provider, tag="t", loop=loop)

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


def _published(tmp_path):
    """Content the remote service took while the node was unreachable, its
    host-cache copy then lost."""
    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    service = _Pinning()
    try:
        result = adapter.store_content(
            b"backup", storage_tier=StorageTier.IPFS,
            upload_name=UPLOAD_NAME, remote_ipfs=service,
        )
    finally:
        patcher.stop()
    for entry in (tmp_path / "cache").iterdir():
        entry.unlink()
    return result, service


def test_retrieval_reads_the_remote_service_when_the_node_cannot_answer(tmp_path):
    result, service = _published(tmp_path)
    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    try:
        content = adapter.retrieve_content(
            result.content_hash,
            ipfs_cid=result.ipfs_cid,
            max_output_bytes=1024,
            remote_ipfs=service,
        )
    finally:
        patcher.stop()

    assert content == b"backup"
    # Read under the encoded ceiling the node is read under.
    assert service.fetches == [("bafypinned", 1024 * 2 + 64 * 1024)]
    # And cached again under its content hash, as a node's answer is.
    assert (tmp_path / "cache" / f"{result.content_hash}.cache").exists()


def test_a_node_that_answers_is_read_before_the_remote_service(tmp_path):
    result, service = _published(tmp_path)
    adapter, _, patcher = _adapter(
        tmp_path,
        node_online=True,
        node_blocks={"bafypinned": zlib.compress(b"backup")},
    )
    try:
        content = adapter.retrieve_content(
            result.content_hash, ipfs_cid=result.ipfs_cid,
            max_output_bytes=1024, remote_ipfs=service,
        )
    finally:
        patcher.stop()

    assert content == b"backup"
    assert service.fetches == []


def test_remote_content_is_checked_like_the_nodes(tmp_path):
    """Bytes the remote service substitutes fail the content-hash check."""
    result, service = _published(tmp_path)
    service.served["bafypinned"] = zlib.compress(b"substituted")
    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    try:
        with pytest.raises(ValueError, match="integrity check failed"):
            adapter.retrieve_content(
                result.content_hash, ipfs_cid=result.ipfs_cid,
                max_output_bytes=1024, remote_ipfs=service,
            )
    finally:
        patcher.stop()

    assert not (tmp_path / "cache" / f"{result.content_hash}.cache").exists()


def test_the_remote_ceiling_is_raised_not_swallowed(tmp_path):
    result, _ = _published(tmp_path)

    class _Oversized(_Pinning):
        def fetch(self, cid, *, max_bytes):
            raise ContentRetrievalLimitError("over the ceiling")

    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    try:
        with pytest.raises(ContentRetrievalLimitError):
            adapter.retrieve_content(
                result.content_hash, ipfs_cid=result.ipfs_cid,
                max_output_bytes=1024, remote_ipfs=_Oversized(),
            )
    finally:
        patcher.stop()


def test_a_content_hash_is_never_sent_to_the_remote_service(tmp_path):
    result, service = _published(tmp_path)
    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    try:
        with pytest.raises(ValueError, match="Content not found"):
            adapter.retrieve_content(
                result.content_hash, ipfs_cid=result.content_hash,
                allow_local_cache=False, remote_ipfs=service,
            )
    finally:
        patcher.stop()

    assert service.fetches == []


@pytest.fixture
def gateway(monkeypatch):
    """Lighthouse's public gateway, serving what ``served`` holds, on a host
    with no Lighthouse key."""
    monkeypatch.delenv("LIGHTHOUSE_API_KEY", raising=False)
    served: dict = {}
    server = GatewayServer(served.get).install(monkeypatch)
    server.served = served
    return server


@pytest.mark.asyncio
async def test_the_lighthouse_gateway_reads_by_cid_with_no_credential(gateway):
    """An import reads an export back from the public gateway, which needs
    no key: a fresh host has none (#3601)."""
    gateway.served["bafylh"] = b"sealed"
    remote = _LighthouseGatewayIpfs(loop=asyncio.get_running_loop())

    content = await asyncio.to_thread(remote.fetch, "bafylh", max_bytes=99)

    assert content == b"sealed"
    assert remote.provider_name == "lighthouse"
    [request] = gateway.requests
    assert str(request.url) == f"{GATEWAY_URL}/bafylh"
    assert "authorization" not in request.headers
    # Read raw, so the ceiling counts the bytes the gateway sends.
    assert request.headers["accept-encoding"] == "identity"
    # The read's client is closed with it.
    [client] = gateway.clients
    assert client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("declare_length", [True, False], ids=["declared", "streamed"])
async def test_the_lighthouse_gateway_ceiling_is_the_adapters_ceiling(
    gateway, declare_length
):
    gateway.served["bafylh"] = b"x" * 100_000
    gateway.declare_length = declare_length
    remote = _LighthouseGatewayIpfs(loop=asyncio.get_running_loop())

    with pytest.raises(ContentRetrievalLimitError, match="exceeds 99 bytes"):
        await asyncio.to_thread(remote.fetch, "bafylh", max_bytes=99)

    # The read stopped at the ceiling rather than buffering the answer.
    [sent] = gateway.sent
    assert sent <= 99 + 64 * 1024
    [client] = gateway.clients
    assert client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [404, 503])
async def test_a_gateway_that_cannot_serve_is_the_adapters_remote_error(gateway, failure):
    gateway.failure = failure
    remote = _LighthouseGatewayIpfs(loop=asyncio.get_running_loop())

    with pytest.raises(RemoteIpfsError, match=str(failure)):
        await asyncio.to_thread(remote.fetch, "bafylh", max_bytes=99)

    [client] = gateway.clients
    assert client.is_closed


@pytest.mark.asyncio
async def test_retrieval_reads_the_lighthouse_gateway_when_the_node_cannot_answer(
    tmp_path, gateway
):
    """Through the adapter: the node is unreachable and the gateway answers
    with what an IPFS-tier store published, read under the encoded ceiling
    and checked as the node's answer is."""
    result, service = _published(tmp_path)
    gateway.served[result.ipfs_cid] = service.served[result.ipfs_cid]
    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    try:
        content = await asyncio.to_thread(
            adapter.retrieve_content,
            result.content_hash,
            ipfs_cid=result.ipfs_cid,
            max_output_bytes=1024,
            remote_ipfs=_LighthouseGatewayIpfs(loop=asyncio.get_running_loop()),
        )
    finally:
        patcher.stop()

    assert content == b"backup"
    assert gateway.cids == [result.ipfs_cid]


class _Failing(_Pinning):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def pin(self, payload, *, filename, on_progress=None):
        raise self.error

    def fetch(self, cid, *, max_bytes):
        raise self.error


def test_a_service_that_cannot_take_the_upload_leaves_it_local_and_says_so(tmp_path):
    """Kubo is unreachable and so is the service: the export is kept locally
    and its tier says so, as with no service at all (#2872, #2874)."""
    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    try:
        result = adapter.store_content(
            b"backup", storage_tier=StorageTier.IPFS,
            upload_name=UPLOAD_NAME,
            remote_ipfs=_Failing(RemoteIpfsError("Lighthouse is down")),
        )
    finally:
        patcher.stop()

    assert result.storage_tier is StorageTier.LOCAL_ONLY
    assert result.ipfs_cid is None
    assert result.provider == "local"
    assert (tmp_path / "cache" / f"{result.cache_name}.cache").exists()


def test_an_unexpected_service_error_is_not_swallowed(tmp_path):
    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    try:
        with pytest.raises(RuntimeError, match="a bug"):
            adapter.store_content(
                b"backup", storage_tier=StorageTier.IPFS,
                upload_name=UPLOAD_NAME, remote_ipfs=_Failing(RuntimeError("a bug")),
            )
    finally:
        patcher.stop()


def test_a_service_that_cannot_serve_reports_the_content_missing(tmp_path):
    result, _ = _published(tmp_path)
    adapter, _, patcher = _adapter(tmp_path, node_online=False)
    try:
        with pytest.raises(ValueError, match="Content not found"):
            adapter.retrieve_content(
                result.content_hash, ipfs_cid=result.ipfs_cid,
                max_output_bytes=1024,
                remote_ipfs=_Failing(RemoteIpfsError("gateway 404")),
            )
    finally:
        patcher.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("Lighthouse provider not available"),
        httpx.HTTPStatusError(
            "503",
            request=httpx.Request("POST", "https://upload.lighthouse.storage/api/v0/add"),
            response=httpx.Response(503),
        ),
        ValueError("Lighthouse returned no CID"),
    ],
)
async def test_lighthouse_failures_are_the_adapters_remote_error(error):
    provider = MagicMock(provider_name="lighthouse")
    provider.publish = AsyncMock(side_effect=error)
    remote = _LighthouseIpfs(provider, tag="t", loop=asyncio.get_running_loop())

    with pytest.raises(RemoteIpfsError):
        await asyncio.to_thread(remote.pin, b"payload", filename=UPLOAD_NAME)


def test_a_call_that_cannot_be_scheduled_leaves_no_coroutine_behind():
    loop = asyncio.new_event_loop()
    loop.close()
    provider = MagicMock(provider_name="lighthouse")
    calls = []

    async def publish(payload, *, filename, tag, on_progress):
        return "bafylh"

    def tracked(payload, **kwargs):
        calls.append(publish(payload, **kwargs))
        return calls[-1]

    provider.publish = tracked
    remote = _LighthouseIpfs(provider, tag="t", loop=loop)

    with pytest.raises(RuntimeError):
        remote.pin(b"payload", filename=UPLOAD_NAME)
    [coroutine] = calls
    assert coroutine.cr_frame is None  # closed, not left un-awaited


def test_the_lighthouse_gateway_refuses_to_block_the_agent_loop():
    """Called on the loop it reads on, ``fetch`` would wait on itself."""
    outcome = {}

    def call_fetch_on_its_loop():
        loop = asyncio.new_event_loop()
        remote = _LighthouseGatewayIpfs(loop=loop)

        async def on_the_loop():
            return remote.fetch("bafylh", max_bytes=1)

        try:
            outcome["content"] = loop.run_until_complete(on_the_loop())
        except RuntimeError as exc:
            outcome["error"] = exc
        finally:
            loop.close()

    caller = threading.Thread(target=call_fetch_on_its_loop, daemon=True)
    caller.start()
    caller.join(timeout=10)

    assert not caller.is_alive(), "fetch blocked the loop it reads on"
    assert "worker thread" in str(outcome["error"])
