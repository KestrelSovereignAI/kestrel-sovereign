"""Tests for Lighthouse REST client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient


@pytest.fixture
def client():
    return LighthouseRestClient(api_key="test-api-key")


@pytest.fixture
def mock_response():
    """Create a mock httpx.Response."""
    def _make(status_code=200, json_data=None, content=b""):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.content = content
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=resp
            )
        return resp
    return _make


class TestLighthouseRestClient:
    """Test the REST client methods."""

    @pytest.mark.asyncio
    async def test_upload(self, client, mock_response):
        resp = mock_response(json_data={"data": {"Hash": "QmTest123", "Name": "test.bin", "Size": "1024"}})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.upload(b"test content", "test.bin", tag="test")

        assert result["Hash"] == "QmTest123"
        assert result["Size"] == "1024"

    @pytest.mark.asyncio
    async def test_upload_car(self, client, mock_response):
        resp = mock_response(json_data={"data": {"Hash": "QmCarTest", "Size": "2048"}})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.upload_car(b"car file bytes", tag="test")

        assert result["Hash"] == "QmCarTest"

    @pytest.mark.asyncio
    async def test_download(self, client, mock_response):
        resp = mock_response(content=b"file content here")

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            data = await client.download("QmTest123")

        assert data == b"file content here"

    @pytest.mark.asyncio
    async def test_get_uploads(self, client, mock_response):
        file_list = [
            {"cid": "QmTest1", "fileName": "a.bin", "fileSizeInBytes": "100"},
            {"cid": "QmTest2", "fileName": "b.bin", "fileSizeInBytes": "200"},
        ]
        resp = mock_response(json_data={"fileList": file_list, "totalFiles": 2})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.get_uploads()

        assert result["fileList"] == file_list
        assert result["totalFiles"] == 2

    @pytest.mark.asyncio
    async def test_get_deal_status(self, client, mock_response):
        resp = mock_response(json_data={"dealStatus": "active"})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.get_deal_status("QmTest123")

        assert result["dealStatus"] == "active"

    @pytest.mark.asyncio
    async def test_get_balance(self, client, mock_response):
        resp = mock_response(json_data={"data": {"dataUsed": "1000", "dataLimit": "5000000000"}})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.get_balance()

        assert result["data"]["dataUsed"] == "1000"
        assert result["data"]["dataLimit"] == "5000000000"

    @pytest.mark.asyncio
    async def test_get_auth_message(self, client, mock_response):
        resp = mock_response(json_data={"data": {"message": "Sign this"}})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.get_auth_message("0xabc")

        assert result == "Sign this"
        mock_http.get.assert_awaited_once_with(
            f"{client.API_URL}/api/auth/get_message",
            params={"publicKey": "0xabc"},
        )

    @pytest.mark.asyncio
    async def test_get_auth_message_accepts_string_response(self, client, mock_response):
        resp = mock_response(json_data="Sign this")

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.get_auth_message("0xabc")

        assert result == "Sign this"

    @pytest.mark.asyncio
    async def test_create_api_key(self, client, mock_response):
        resp = mock_response(json_data="lh-key")

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.create_api_key("0xabc", "0xsig")

        assert result == "lh-key"
        mock_http.post.assert_awaited_once_with(
            f"{client.API_URL}/api/auth/create_api_key",
            json={"publicKey": "0xabc", "signedMessage": "0xsig"},
            headers={"Accept": "application/json"},
        )

    @pytest.mark.asyncio
    async def test_create_api_key_accepts_wrapped_response(self, client, mock_response):
        resp = mock_response(json_data={"data": {"apiKey": "lh-key"}})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.create_api_key("0xabc", "0xsig")

        assert result == "lh-key"

    @pytest.mark.asyncio
    async def test_upload_error_handling(self, client, mock_response):
        resp = mock_response(status_code=401, json_data={"error": "Unauthorized"})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            with pytest.raises(httpx.HTTPStatusError):
                await client.upload(b"data", "test.bin")

    @pytest.mark.asyncio
    async def test_auth_headers(self, client):
        assert client._auth_headers == {"Authorization": "Bearer test-api-key"}

    @pytest.mark.asyncio
    async def test_close(self, client):
        # Should not raise even when no client created
        await client.close()

    @pytest.mark.asyncio
    async def test_upload_normalizes_flat_response(self, client, mock_response):
        """Test that responses without 'data' wrapper are returned as-is."""
        resp = mock_response(json_data={"Hash": "QmDirect", "Size": "512"})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.upload(b"test", "test.bin")

        assert result["Hash"] == "QmDirect"
