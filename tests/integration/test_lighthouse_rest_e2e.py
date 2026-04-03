"""
End-to-end integration tests for LighthouseRestClient.

These tests make REAL API calls to Lighthouse storage.
Gated by LIGHTHOUSE_API_KEY env var — skipped when not set.

Run with:
    LIGHTHOUSE_API_KEY=xxx uv run python -m pytest tests/integration/test_lighthouse_rest_e2e.py -v
"""

import importlib.util
import os
import time

import pytest
import pytest_asyncio

LIGHTHOUSE_API_KEY = os.environ.get("LIGHTHOUSE_API_KEY")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not LIGHTHOUSE_API_KEY,
        reason="LIGHTHOUSE_API_KEY not set — skipping real Lighthouse tests",
    ),
]


@pytest_asyncio.fixture
async def client():
    """Create a real LighthouseRestClient, close after test."""
    from kestrel_storage_lighthouse.lighthouse_rest import LighthouseRestClient

    c = LighthouseRestClient(api_key=LIGHTHOUSE_API_KEY)
    yield c
    await c.close()


# ---------------------------------------------------------------------------
# Upload + download round-trip
# ---------------------------------------------------------------------------
class TestUploadDownload:
    """Real upload and download via Lighthouse REST API."""

    async def test_upload_returns_cid(self, client):
        """Upload a small payload and verify we get a CID back."""
        payload = f"kestrel integration test {time.time()}".encode()
        result = await client.upload(payload, filename="test_upload.txt")

        assert "Hash" in result, f"Expected 'Hash' key in response, got {result}"
        cid = result["Hash"]
        assert len(cid) > 10, f"CID looks too short: {cid}"
        # CIDv1 starts with 'baf' (base32) or CIDv0 starts with 'Qm'
        assert cid.startswith("Qm") or cid.startswith("baf"), f"Unexpected CID format: {cid}"

    async def test_upload_and_download_roundtrip(self, client):
        """Upload bytes, download by CID, verify content matches."""
        payload = f"roundtrip test {time.time()}".encode()
        result = await client.upload(payload, filename="roundtrip.txt")
        cid = result["Hash"]

        downloaded = await client.download(cid, timeout=30.0)
        assert downloaded == payload, (
            f"Content mismatch: uploaded {len(payload)} bytes, "
            f"downloaded {len(downloaded)} bytes"
        )

    async def test_upload_binary_content(self, client):
        """Upload non-text binary content."""
        payload = os.urandom(256)
        result = await client.upload(payload, filename="binary_test.bin")
        cid = result["Hash"]
        assert cid

        downloaded = await client.download(cid, timeout=30.0)
        assert downloaded == payload

    async def test_upload_with_custom_tag(self, client):
        """Upload with a custom tag for organization."""
        payload = b"tagged upload test"
        result = await client.upload(
            payload,
            filename="tagged.txt",
            tag="kestrel-integration-test",
        )
        assert "Hash" in result


# ---------------------------------------------------------------------------
# CAR file upload
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not importlib.util.find_spec("cbor2"),
    reason="cbor2 not installed (optional wallet dependency)",
)
class TestCARUpload:
    """Upload CAR files to Lighthouse."""

    async def test_upload_car_roundtrip(self, client):
        """Build a CAR, upload it, verify CID returned."""
        from kestrel_sovereign.storage.car_builder import CARBuilder

        builder = CARBuilder()
        data = b"car upload integration test"
        cid = builder.add_raw_block(data)
        builder.set_root(cid)
        car_bytes = builder.build()

        result = await client.upload_car(car_bytes, tag="kestrel-car-test")
        assert "Hash" in result, f"CAR upload failed: {result}"

    async def test_upload_car_multi_block(self, client):
        """Upload a multi-block CAR (sovereignty export pattern)."""
        from kestrel_sovereign.storage.car_builder import CARBuilder

        builder = CARBuilder()

        # Simulate sovereignty export: shards + manifest
        shard1 = builder.add_raw_block(b"shard-1-data")
        shard2 = builder.add_raw_block(b"shard-2-data")

        manifest = {
            "type": "sovereignty-export",
            "shards": [str(shard1), str(shard2)],
            "timestamp": time.time(),
        }
        manifest_cid = builder.add_dag_cbor_block(manifest)
        builder.set_root(manifest_cid)

        car_bytes = builder.build()
        result = await client.upload_car(car_bytes, tag="kestrel-sovereignty-test")
        assert "Hash" in result


# ---------------------------------------------------------------------------
# Account queries
# ---------------------------------------------------------------------------
class TestAccountQueries:
    """Test list/balance/status endpoints against real API."""

    async def test_get_balance(self, client):
        """Get account storage balance — should return usage data."""
        balance = await client.get_balance()
        assert isinstance(balance, dict)
        assert "data" in balance or "dataUsed" in balance, (
            f"Unexpected balance response: {balance}"
        )

    async def test_get_uploads_returns_file_list(self, client):
        """List uploads — should return dict with fileList."""
        result = await client.get_uploads()
        assert isinstance(result, dict)
        assert "fileList" in result
        assert isinstance(result["fileList"], list)

    async def test_get_uploads_cursor_pagination(self, client):
        """Cursor-based pagination should work without errors."""
        result = await client.get_uploads()
        assert isinstance(result, dict)
        file_list = result.get("fileList", [])
        # If there are uploads, we can use the last one as a cursor
        if file_list:
            last_id = file_list[-1].get("id", "")
            if last_id:
                page2 = await client.get_uploads(last_key=last_id)
                assert isinstance(page2, dict)


class TestDealStatus:
    """Filecoin deal status queries."""

    async def test_deal_status_for_uploaded_cid(self, client):
        """Upload something, then query its deal status."""
        payload = f"deal status test {time.time()}".encode()
        result = await client.upload(payload, filename="deal_test.txt")
        cid = result["Hash"]

        # Deal status returns a list of deals (empty for new uploads)
        status = await client.get_deal_status(cid)
        assert isinstance(status, (list, dict))

    async def test_deal_status_unknown_cid(self, client):
        """Query deal status for a CID that doesn't exist — should not crash."""
        fake_cid = "bafkreiaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        try:
            status = await client.get_deal_status(fake_cid)
            assert isinstance(status, (list, dict))
        except Exception as e:
            # Some API versions return 404 for unknown CIDs — acceptable
            assert "404" in str(e) or "not found" in str(e).lower(), (
                f"Unexpected error for unknown CID: {e}"
            )


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------
class TestClientLifecycle:
    """Test client creation, reuse, and cleanup."""

    async def test_client_reuse(self):
        """Internal httpx client should be reused across calls."""
        from kestrel_storage_lighthouse.lighthouse_rest import LighthouseRestClient

        c = LighthouseRestClient(api_key=LIGHTHOUSE_API_KEY)
        try:
            c1 = await c._get_client()
            c2 = await c._get_client()
            assert c1 is c2, "Client should be reused"
        finally:
            await c.close()

    async def test_close_and_reopen(self):
        """After close(), next call should create a new client."""
        from kestrel_storage_lighthouse.lighthouse_rest import LighthouseRestClient

        c = LighthouseRestClient(api_key=LIGHTHOUSE_API_KEY)
        c1 = await c._get_client()
        await c.close()
        c2 = await c._get_client()
        assert c1 is not c2, "New client should be created after close"
        await c.close()


# ---------------------------------------------------------------------------
# LighthouseProvider integration (higher level)
# ---------------------------------------------------------------------------
class TestLighthouseProvider:
    """Test the provider layer that wraps LighthouseRestClient."""

    async def test_provider_store_and_retrieve(self, tmp_path):
        """Store via provider, retrieve, verify round-trip."""
        from kestrel_storage_lighthouse.lighthouse_provider import LighthouseProvider

        provider = LighthouseProvider(api_key=LIGHTHOUSE_API_KEY)
        provider.cache_dir = tmp_path / "cache"
        provider.cache_dir.mkdir()
        payload = f"provider integration test {time.time()}".encode()

        # store() returns StorageResult, not a CID string
        result = await provider.store(payload, metadata={"test": True}, encrypt=False)
        assert result.cid, "Store should return a result with a CID"

        retrieved = await provider.retrieve(result.cid)
        assert retrieved == payload, "Retrieved content should match stored"

    async def test_provider_get_balance(self):
        """Provider.get_balance should return storage usage."""
        from kestrel_storage_lighthouse.lighthouse_provider import LighthouseProvider

        provider = LighthouseProvider(api_key=LIGHTHOUSE_API_KEY)
        # Use the REST client directly for balance — provider.get_balance
        # requires wallet_address and currency (payment balance, not storage)
        balance = await provider._client.get_balance()
        assert isinstance(balance, dict)
        assert "data" in balance
