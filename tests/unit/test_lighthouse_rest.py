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
    async def test_upload_progress_callback_reads_file_bytes(self, client, mock_response):
        resp = mock_response(json_data={"data": {"Hash": "QmTest123", "Name": "test.bin", "Size": "131072"}})
        progress = []

        async def fake_post(*args, **kwargs):
            file_obj = kwargs["files"]["file"][1]
            while file_obj.read(16 * 1024):
                pass
            return resp

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=fake_post)
            mock_get.return_value = mock_http

            await client.upload(
                b"x" * (128 * 1024),
                "test.bin",
                tag="test",
                on_progress=lambda sent, total: progress.append((sent, total)),
            )

        assert progress[0] == (0, 128 * 1024)
        assert progress[-1] == (128 * 1024, 128 * 1024)
        assert len(progress) > 2

    @pytest.mark.asyncio
    async def test_upload_car(self, client, mock_response):
        resp = mock_response(json_data={"data": {"Hash": "QmCarTest", "Size": "2048"}})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.upload_car(
                b"car file bytes",
                tag="test",
                filename="kestrel_state__agent-1__20260620_120000.car",
            )

        assert result["Hash"] == "QmCarTest"
        files = mock_http.post.await_args.kwargs["files"]
        assert files["file"][0] == "kestrel_state__agent-1__20260620_120000.car"

    @pytest.mark.asyncio
    async def test_upload_car_progress_callback_reads_file_bytes(self, client, mock_response):
        resp = mock_response(json_data={"data": {"Hash": "QmCarTest", "Size": "131072"}})
        progress = []

        async def fake_post(*args, **kwargs):
            file_obj = kwargs["files"]["file"][1]
            while file_obj.read(32 * 1024):
                pass
            return resp

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(side_effect=fake_post)
            mock_get.return_value = mock_http

            await client.upload_car(
                b"c" * (128 * 1024),
                tag="test",
                filename="export.car",
                on_progress=lambda sent, total: progress.append((sent, total)),
            )

        assert progress[0] == (0, 128 * 1024)
        assert progress[-1] == (128 * 1024, 128 * 1024)
        assert len(progress) > 2

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
    async def test_delete_file_uses_file_id_param(self, client, mock_response):
        resp = mock_response(json_data={"data": {"deleted": True}})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.delete = AsyncMock(return_value=resp)
            mock_get.return_value = mock_http

            result = await client.delete_file("file-uuid-123")

        assert result == {"deleted": True}
        mock_http.delete.assert_awaited_once_with(
            f"{client.API_URL}/api/user/delete_file",
            headers={
                "Authorization": "Bearer test-api-key",
                "Content-Type": "application/json",
            },
            params={"id": "file-uuid-123"},
        )

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


# ---------------------------------------------------------------------------
# #3189: the upload budget grows with the payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_car_budget_is_proportional_to_the_payload(client, mock_response):
    """A 1.2 GB snapshot cannot cross a link in the flat 60 s default; for
    eighteen days every Lighthouse upload timed out with an empty message.
    The read and write budgets are sized by the payload at the floor rate;
    connect and pool stay at the default."""
    import builtins

    import httpx

    resp = mock_response(json_data={"data": {"Hash": "QmBig", "Size": "1"}})
    size = 1_209_462_784
    payload = b"x"  # never sent: post is a mock; only ITS length is faked

    def sized(obj):
        return size if obj is payload else builtins.len(obj)

    with patch.object(client, "_get_client") as mock_get:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=resp)
        mock_get.return_value = mock_http
        with patch("kestrel_sovereign.storage.providers.lighthouse_rest.len", create=True) as fake_len:
            fake_len.side_effect = sized
            await client.upload_car(payload, tag="t", filename="big.car")

    assert any(call.args == (payload,) for call in fake_len.call_args_list), (
        "the budget must be sized from the CAR payload, not another len()"
    )
    budget = mock_http.post.await_args.kwargs["timeout"]
    assert isinstance(budget, httpx.Timeout)
    expected = size / client.UPLOAD_FLOOR_BYTES_PER_SECOND
    assert budget.read == pytest.approx(expected) and budget.write == pytest.approx(expected)
    assert expected > 2000
    assert budget.connect == client.timeout and budget.pool == client.timeout


@pytest.mark.asyncio
async def test_upload_budget_is_sized_from_the_content_too(client, mock_response):
    """The sibling door: ``upload()`` carries arbitrary user content through
    the same client and had the flat default."""
    import builtins

    import httpx

    resp = mock_response(json_data={"data": {"Hash": "QmFile", "Size": "1"}})
    size = 600 * 1024 * 1024
    content = b"y"

    def sized(obj):
        return size if obj is content else builtins.len(obj)

    with patch.object(client, "_get_client") as mock_get:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=resp)
        mock_get.return_value = mock_http
        with patch("kestrel_sovereign.storage.providers.lighthouse_rest.len", create=True) as fake_len:
            fake_len.side_effect = sized
            await client.upload(content, filename="big.bin")

    assert any(call.args == (content,) for call in fake_len.call_args_list)
    budget = mock_http.post.await_args.kwargs["timeout"]
    assert isinstance(budget, httpx.Timeout)
    assert budget.write == pytest.approx(size / client.UPLOAD_FLOOR_BYTES_PER_SECOND)


def test_a_small_upload_keeps_the_default_budget(client):
    budget = client.upload_timeout(1024)
    assert budget.read == client.timeout and budget.write == client.timeout


def test_the_budget_is_the_larger_of_the_default_and_the_payload_rate(client):
    at_floor = int(client.timeout * client.UPLOAD_FLOOR_BYTES_PER_SECOND)
    assert client.upload_timeout(at_floor).write == pytest.approx(client.timeout)
    assert client.upload_timeout(at_floor * 3).write == pytest.approx(client.timeout * 3)
