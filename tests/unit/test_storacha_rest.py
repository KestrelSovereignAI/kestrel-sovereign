"""Tests for storacha_rest.py — Storacha w3up HTTP client."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import cbor2
import httpx
import pytest

from kestrel_sovereign.storage.providers.storacha_rest import (
    StorachaRestClient,
    StorachaError,
    _cid_str_to_bytes,
    _wrap_in_car,
)
from kestrel_sovereign.storage.providers.storacha_ucan import (
    StorachaUCAN,
    build_car,
    cid_to_string,
    cid_v1,
    parse_car,
    _pubkey_to_did,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_seed_and_key():
    import os
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    seed = os.urandom(32)
    return seed, Ed25519PrivateKey.from_private_bytes(seed)


def _seed_to_w3_key(seed: bytes) -> str:
    prefix = bytes([0x80, 0x26])
    return "M" + base64.b64encode(prefix + seed).decode().rstrip("=")


def _make_minimal_proof_car(agent_did: str) -> str:
    delegation = {
        "v": "1.0.0-rc.1",
        "iss": "did:key:z6MkTestSpace",
        "aud": agent_did,
        "att": [{"with": "did:key:z6MkTestSpace", "can": "*"}],
        "prf": [],
        "exp": 9999999999,
    }
    block_bytes = cbor2.dumps(delegation, canonical=True)
    block_cid = cid_v1(block_bytes)
    car = build_car([block_cid], [(block_cid, block_bytes)])
    return base64.b64encode(car).decode()


def _make_ucan() -> StorachaUCAN:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    seed, pk = _make_seed_and_key()
    w3_key = _seed_to_w3_key(seed)
    pub = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    agent_did = _pubkey_to_did(pub)
    proof = _make_minimal_proof_car(agent_did)
    return StorachaUCAN(
        agent_key=w3_key,
        space_did="did:key:z6MkTestSpace",
        proof=proof,
    )


def _make_receipt_car(ok: dict) -> bytes:
    """Build a fake bridge response CAR with a receipt block."""
    receipt = {"out": {"ok": ok}}
    block_bytes = cbor2.dumps(receipt, canonical=True)
    block_cid = cid_v1(block_bytes)
    return build_car([block_cid], [(block_cid, block_bytes)])


@pytest.fixture
def ucan():
    return _make_ucan()


@pytest.fixture
def client(ucan):
    return StorachaRestClient(ucan=ucan, gateway_url="https://w3s.link/ipfs")


@pytest.fixture
def mock_response():
    def _make(status_code=200, content=b"", raise_error=False):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.content = content
        resp.raise_for_status = MagicMock()
        if raise_error or status_code >= 400:
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=resp
            )
        return resp
    return _make


# ---------------------------------------------------------------------------
# _wrap_in_car
# ---------------------------------------------------------------------------

class TestWrapInCar:
    def test_produces_valid_car(self):
        data = b"hello"
        cid = cid_v1(data, codec=0x55)
        car = _wrap_in_car(data, cid)
        root_cids, blocks = parse_car(car)
        assert root_cids[0] == cid
        assert blocks[cid] == data


# ---------------------------------------------------------------------------
# _cid_str_to_bytes
# ---------------------------------------------------------------------------

class TestCIDStrToBytes:
    def test_roundtrip(self):
        original = cid_v1(b"roundtrip")
        s = cid_to_string(original)
        recovered = _cid_str_to_bytes(s)
        assert recovered == original

    def test_invalid_prefix_raises(self):
        with pytest.raises(ValueError, match="Unsupported CID"):
            _cid_str_to_bytes("Qm12345678901234567890123456789012345678")


# ---------------------------------------------------------------------------
# _parse_bridge_response
# ---------------------------------------------------------------------------

class TestParseBridgeResponse:
    def test_parses_ok_result(self):
        ok = {"status": "upload", "url": "https://s3.example.com/put"}
        car = _make_receipt_car(ok)
        result = StorachaRestClient._parse_bridge_response(car)
        assert result["status"] == "upload"
        assert result["url"] == "https://s3.example.com/put"

    def test_raises_on_error_receipt(self):
        error_receipt = {"out": {"error": {"name": "SpaceNotFound", "message": "Space not found"}}}
        block_bytes = cbor2.dumps(error_receipt, canonical=True)
        block_cid = cid_v1(block_bytes)
        car = build_car([block_cid], [(block_cid, block_bytes)])
        with pytest.raises(StorachaError, match="SpaceNotFound"):
            StorachaRestClient._parse_bridge_response(car)

    def test_raises_on_empty_car(self):
        # A car with a root that references a missing block
        cid = cid_v1(b"missing")
        car = build_car([cid], [])
        with pytest.raises(StorachaError, match="missing root block"):
            StorachaRestClient._parse_bridge_response(car)


# ---------------------------------------------------------------------------
# StorachaRestClient.upload
# ---------------------------------------------------------------------------

class TestUpload:
    @pytest.mark.asyncio
    async def test_upload_status_upload(self, client, mock_response):
        """Full upload flow: store/add → PUT → upload/add."""
        content = b"test file content"
        content_cid = client.ucan.content_cid(content)

        store_ok = {
            "status": "upload",
            "url": "https://s3.example.com/presigned",
            "headers": {"x-amz-meta": "test"},
            "link": cbor2.CBORTag(42, b"\x00" + content_cid),
        }
        upload_ok = {"root": cbor2.CBORTag(42, b"\x00" + content_cid)}

        store_car = _make_receipt_car(store_ok)
        upload_car = _make_receipt_car(upload_ok)

        call_count = 0
        put_called = False

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock(spec=httpx.Response)
            resp.raise_for_status = MagicMock()
            resp.content = store_car if call_count == 1 else upload_car
            return resp

        async def mock_put(url, **kwargs):
            nonlocal put_called
            put_called = True
            resp = MagicMock(spec=httpx.Response)
            resp.raise_for_status = MagicMock()
            resp.content = b""
            return resp

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = mock_post
            mock_http.put = mock_put
            mock_get.return_value = mock_http

            result = await client.upload(content, "test.bin")

        assert result["cid"].startswith("b")
        assert result["size"] == len(content)
        assert put_called, "PUT to presigned URL should have been called"
        assert call_count == 2, "Should make two bridge invocations (store/add + upload/add)"

    @pytest.mark.asyncio
    async def test_upload_status_done_skips_put(self, client):
        """If store/add returns 'done', skip the S3 PUT."""
        content = b"already uploaded"
        content_cid = client.ucan.content_cid(content)

        store_ok = {"status": "done", "link": cbor2.CBORTag(42, b"\x00" + content_cid)}
        upload_ok = {"root": cbor2.CBORTag(42, b"\x00" + content_cid)}

        store_car = _make_receipt_car(store_ok)
        upload_car = _make_receipt_car(upload_ok)

        call_count = 0
        put_called = False

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock(spec=httpx.Response)
            resp.raise_for_status = MagicMock()
            resp.content = store_car if call_count == 1 else upload_car
            return resp

        async def mock_put(url, **kwargs):
            nonlocal put_called
            put_called = True

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = mock_post
            mock_http.put = mock_put
            mock_get.return_value = mock_http

            result = await client.upload(content, "dup.bin")

        assert not put_called, "PUT should be skipped when status is 'done'"
        assert result["cid"].startswith("b")


# ---------------------------------------------------------------------------
# StorachaRestClient.get_by_cid
# ---------------------------------------------------------------------------

class TestGetByCID:
    @pytest.mark.asyncio
    async def test_get_returns_content(self, client, mock_response):
        resp = mock_response(content=b"retrieved bytes")
        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            data = await client.get_by_cid("bafkreitest")

        assert data == b"retrieved bytes"

    @pytest.mark.asyncio
    async def test_get_404_raises(self, client, mock_response):
        resp = mock_response(status_code=404, raise_error=True)
        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            with pytest.raises(httpx.HTTPStatusError):
                await client.get_by_cid("bafkreimissing")


# ---------------------------------------------------------------------------
# StorachaRestClient.delete
# ---------------------------------------------------------------------------

class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(self, client):
        cid_str = cid_to_string(cid_v1(b"to delete"))
        delete_ok = {}

        async def mock_post(url, **kwargs):
            resp = MagicMock(spec=httpx.Response)
            resp.raise_for_status = MagicMock()
            resp.content = _make_receipt_car(delete_ok)
            return resp

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = mock_post
            mock_get.return_value = mock_http

            result = await client.delete(cid_str)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_error(self, client):
        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_get.return_value = mock_http

            result = await client.delete("bafkrei_nonexistent")

        assert result is False


# ---------------------------------------------------------------------------
# StorachaRestClient.close
# ---------------------------------------------------------------------------

class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_close_no_client(self, client):
        # Should not raise when no client created yet
        await client.close()

    @pytest.mark.asyncio
    async def test_close_with_client(self, client):
        # Initialise the client, then close it
        await client._get_client()
        await client.close()
        assert client._client is None or client._client.is_closed

    @pytest.mark.asyncio
    async def test_get_client_reuses_instance(self, client):
        c1 = await client._get_client()
        c2 = await client._get_client()
        assert c1 is c2
