"""
Async REST client for Lighthouse storage API.

Replaces the unmaintained lighthouseweb3 Python SDK (v0.1.1, May 2023)
with direct REST API calls using httpx for full async support.

Lighthouse REST API reference:
- Upload:  https://upload.lighthouse.storage/api/v0/add (POST multipart)
- List:    https://api.lighthouse.storage/api/user/files_uploaded (GET, cursor)
- Deals:   https://api.lighthouse.storage/api/lighthouse/deal_status?cid=X (no auth)
- Balance: https://api.lighthouse.storage/api/user/user_data_usage
- Gateway: https://gateway.lighthouse.storage/ipfs/{cid}
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LighthouseRestClient:
    """Async HTTP client for Lighthouse storage REST API."""

    UPLOAD_URL = "https://upload.lighthouse.storage"
    API_URL = "https://api.lighthouse.storage"

    def __init__(
        self,
        api_key: str,
        gateway_url: str = "https://gateway.lighthouse.storage/ipfs",
        timeout: float = 60.0,
    ):
        """
        Initialize Lighthouse REST client.

        Args:
            api_key: Lighthouse API key for authentication
            gateway_url: IPFS gateway URL for downloads
            timeout: Default request timeout in seconds
        """
        self.api_key = api_key
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def _auth_headers(self) -> Dict[str, str]:
        """Authorization headers for API requests."""
        return {"Authorization": f"Bearer {self.api_key}"}

    async def upload(
        self,
        content: bytes,
        filename: str,
        tag: str = "kestrel-storage",
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Upload file via multipart POST.

        Args:
            content: File content bytes
            filename: Name for the uploaded file
            tag: Tag for organizing uploads
            on_progress: Optional callback(bytes_sent, total_bytes)

        Returns:
            Dict with 'Hash' (CID), 'Name', and 'Size' keys

        Raises:
            httpx.HTTPStatusError: On API errors
        """
        client = await self._get_client()
        files = {"file": (filename, content)}

        response = await client.post(
            f"{self.UPLOAD_URL}/api/v0/add",
            headers=self._auth_headers,
            files=files,
            params={"tag": tag},
        )
        response.raise_for_status()

        data = response.json()
        # Normalize: Lighthouse wraps in {"data": {...}} sometimes
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data

    async def upload_car(
        self,
        car_bytes: bytes,
        tag: str = "kestrel-storage",
    ) -> Dict[str, Any]:
        """
        Upload CAR (Content Addressable aRchive) file.

        Lighthouse SDK v0.4.4+ supports direct CAR upload for
        faster, verifiable ingestion of content-addressed DAGs.

        Args:
            car_bytes: CAR v1 file bytes
            tag: Tag for organizing uploads

        Returns:
            Dict with 'Hash' (CID), 'Name', and 'Size' keys
        """
        client = await self._get_client()
        files = {"file": ("export.car", car_bytes, "application/vnd.ipld.car")}

        response = await client.post(
            f"{self.UPLOAD_URL}/api/v0/add",
            headers=self._auth_headers,
            files=files,
            params={"tag": tag},
        )
        response.raise_for_status()

        data = response.json()
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data

    async def download(self, cid: str, timeout: Optional[float] = None) -> bytes:
        """
        Download content from IPFS gateway.

        Args:
            cid: IPFS Content ID
            timeout: Override timeout for large downloads

        Returns:
            Content bytes
        """
        client = await self._get_client()
        response = await client.get(
            f"{self.gateway_url}/{cid}",
            timeout=timeout or 120.0,
        )
        response.raise_for_status()
        return response.content

    async def get_uploads(self, last_key: Optional[str] = None) -> Dict[str, Any]:
        """
        List uploads with cursor pagination.

        Args:
            last_key: Cursor for pagination (None for first page)

        Returns:
            Dict with 'fileList' (list of upload records) and 'totalFiles'
        """
        client = await self._get_client()
        response = await client.get(
            f"{self.API_URL}/api/user/files_uploaded",
            headers=self._auth_headers,
            params={"lastKey": last_key or "null"},
        )
        response.raise_for_status()

        data = response.json()
        # Normalize: SDK wraps as {data: {fileList: [...], totalFiles: N}}
        if isinstance(data, dict):
            if "fileList" in data:
                return data
            if "data" in data and isinstance(data["data"], dict):
                return data["data"]
        return {"fileList": [], "totalFiles": 0}

    async def get_deal_status(self, cid: str) -> Dict[str, Any]:
        """
        Get Filecoin deal status for a CID.

        Args:
            cid: Content ID to check

        Returns:
            Deal status information
        """
        client = await self._get_client()
        response = await client.get(
            f"{self.API_URL}/api/lighthouse/deal_status",
            params={"cid": cid},
        )
        response.raise_for_status()
        return response.json()

    async def get_balance(self) -> Dict[str, Any]:
        """
        Get storage quota and usage.

        Returns:
            Dict with 'dataUsed' and 'dataLimit' (bytes)
        """
        client = await self._get_client()
        response = await client.get(
            f"{self.API_URL}/api/user/user_data_usage",
            headers=self._auth_headers,
        )
        response.raise_for_status()

        data = response.json()
        # Normalize response format
        if isinstance(data, dict) and "data" in data:
            return data
        return {"data": data}
