"""
Async REST client for Lighthouse storage API.

Replaces the unmaintained lighthouseweb3 Python SDK (v0.1.1, May 2023)
with direct REST API calls using httpx for full async support.

Lighthouse REST API reference:
- POST /api/v0/add          — Upload file (multipart)
- GET  /api/user/uploads     — List uploads (paginated)
- GET  /api/lighthouse/deal_status?cid=X — Filecoin deal status
- GET  /api/user/user_data_usage — Storage quota and usage
- Gateway: https://gateway.lighthouse.storage/ipfs/{cid}
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LighthouseRestClient:
    """Async HTTP client for Lighthouse storage REST API."""

    BASE_URL = "https://api.lighthouse.storage"

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
            f"{self.BASE_URL}/api/v0/add",
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
            f"{self.BASE_URL}/api/v0/add",
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

    async def get_uploads(self, page: int = 1) -> List[Dict[str, Any]]:
        """
        List uploads with pagination.

        Args:
            page: Page number (1-indexed)

        Returns:
            List of upload records
        """
        client = await self._get_client()
        response = await client.get(
            f"{self.BASE_URL}/api/user/uploads",
            headers=self._auth_headers,
            params={"page": page},
        )
        response.raise_for_status()

        data = response.json()
        if isinstance(data, dict) and "data" in data:
            uploads = data["data"]
            return uploads if isinstance(uploads, list) else [uploads]
        return data if isinstance(data, list) else []

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
            f"{self.BASE_URL}/api/lighthouse/deal_status",
            headers=self._auth_headers,
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
            f"{self.BASE_URL}/api/user/user_data_usage",
            headers=self._auth_headers,
        )
        response.raise_for_status()

        data = response.json()
        # Normalize response format
        if isinstance(data, dict) and "data" in data:
            return data
        return {"data": data}
