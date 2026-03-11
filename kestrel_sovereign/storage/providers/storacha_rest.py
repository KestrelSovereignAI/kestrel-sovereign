"""
Async HTTP client for the Storacha (web3.storage) w3up bridge API.

Upload flow (w3up protocol):
  1. Hash content → CIDv1 (raw codec)
  2. Wrap content in a single-block CARv1 (root = content CID)
  3. Build signed store/add UCAN invocation → POST CAR to bridge
     Response contains a pre-signed S3 URL for the actual upload
  4. PUT the CAR bytes to the pre-signed S3 URL
  5. Build signed upload/add UCAN invocation → POST CAR to bridge
     Registers the content CID in the space

Retrieval:
  GET https://w3s.link/ipfs/{cid}  (or any IPFS gateway)

Storacha bridge: https://up.storacha.network
w3up protocol:   https://github.com/storacha/w3up
"""

import logging
from typing import Any, Dict, List, Optional

import cbor2
import httpx

from kestrel_sovereign.storage.providers.storacha_ucan import (
    StorachaUCAN,
    STORACHA_BRIDGE_URL,
    build_car,
    cid_to_string,
    cid_v1,
    parse_car,
)

logger = logging.getLogger(__name__)

_MC_RAW = 0x55  # raw bytes codec (same as storacha_ucan._MC_RAW)


class StorachaError(Exception):
    """Raised when the Storacha bridge returns an error."""


class StorachaRestClient:
    """
    Async HTTP client for the Storacha w3up bridge API.

    Handles the two-phase upload (store/add → PUT → upload/add) and
    content retrieval via an IPFS gateway.
    """

    def __init__(
        self,
        ucan: StorachaUCAN,
        gateway_url: str = "https://w3s.link/ipfs",
        timeout: float = 120.0,
    ):
        """
        Args:
            ucan:        Initialised StorachaUCAN instance for signing
            gateway_url: IPFS gateway base URL for retrieval
            timeout:     Default HTTP request timeout in seconds
        """
        self.ucan = ucan
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Return (or create) the shared httpx async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    async def upload(
        self,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        """
        Upload content to Storacha via the w3up bridge.

        Steps:
          1. Compute content CID (raw codec)
          2. Wrap in a single-block CAR
          3. store/add → get pre-signed PUT URL
          4. PUT the CAR to S3
          5. upload/add → register the CID in the space

        Args:
            content:      Raw bytes to store
            filename:     Hint for the filename (stored in metadata)
            content_type: MIME type hint

        Returns:
            {"cid": str, "size": int}  where cid is the base32lower CIDv1 string

        Raises:
            StorachaError: On bridge errors
            httpx.HTTPStatusError: On HTTP errors
        """
        content_cid = self.ucan.content_cid(content)
        car_bytes = _wrap_in_car(content, content_cid)
        car_size = len(car_bytes)

        logger.debug(
            "Uploading to Storacha: cid=%s size=%d filename=%s",
            cid_to_string(content_cid)[:20],
            len(content),
            filename,
        )

        # Phase 1: store/add — get a pre-signed S3 URL
        store_block, store_cid = self.ucan.build_store_add(content_cid, car_size)
        store_car = self.ucan.build_invocation_car([(store_block, store_cid)])
        store_result = await self._invoke(store_car)

        status = store_result.get("status")
        if status == "upload":
            # Phase 2: PUT the CAR to the pre-signed URL
            put_url: str = store_result["url"]
            put_headers: Dict[str, str] = store_result.get("headers", {})
            await self._put_car(put_url, put_headers, car_bytes)
        elif status == "done":
            logger.debug("Content already stored (deduplication): %s", cid_to_string(content_cid)[:20])
        else:
            logger.warning("Unexpected store/add status: %s", status)

        # Phase 3: upload/add — register the CID in the space
        upload_block, upload_cid = self.ucan.build_upload_add(
            root_cid=content_cid,
            shard_cids=[content_cid],  # CAR shard == content CID for single-block CARs
        )
        upload_car = self.ucan.build_invocation_car([(upload_block, upload_cid)])
        await self._invoke(upload_car)

        cid_str = cid_to_string(content_cid)
        logger.info("Uploaded to Storacha: cid=%s", cid_str)
        return {"cid": cid_str, "size": len(content)}

    async def _put_car(
        self,
        url: str,
        headers: Dict[str, str],
        car_bytes: bytes,
    ) -> None:
        """PUT the CAR bytes to the pre-signed S3 URL."""
        client = await self._get_client()
        put_headers = {
            "Content-Type": "application/vnd.ipld.car",
            **headers,  # include any signed headers from the bridge
        }
        response = await client.put(
            url,
            content=car_bytes,
            headers=put_headers,
            timeout=300.0,  # large uploads may be slow
        )
        response.raise_for_status()
        logger.debug("PUT to pre-signed URL succeeded (%d bytes)", len(car_bytes))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def get_by_cid(self, cid: str, timeout: Optional[float] = None) -> bytes:
        """
        Retrieve content from the IPFS gateway by CID string.

        Args:
            cid:     CID string (base32lower "b..." or base58 "Qm...")
            timeout: Override timeout for large downloads

        Returns:
            Raw content bytes
        """
        client = await self._get_client()
        url = f"{self.gateway_url}/{cid}"
        logger.debug("Fetching from gateway: %s", url)
        response = await client.get(url, timeout=timeout or self.timeout)
        response.raise_for_status()
        return response.content

    # ------------------------------------------------------------------
    # Listing (uses upload/list UCAN invocation)
    # ------------------------------------------------------------------

    async def list_uploads(
        self,
        cursor: Optional[str] = None,
        size: int = 25,
    ) -> Dict[str, Any]:
        """
        List uploads in the space via upload/list UCAN invocation.

        Args:
            cursor: Pagination cursor from a previous response
            size:   Maximum results per page (default 25)

        Returns:
            {"results": [...], "cursor": str | None, "size": int}
        """
        nb: Dict[str, Any] = {"size": size}
        if cursor:
            nb["cursor"] = cursor

        list_block, list_cid = self.ucan._sign_invocation("upload/list", nb)
        list_car = self.ucan.build_invocation_car([(list_block, list_cid)])
        result = await self._invoke(list_car)

        return {
            "results": result.get("results", []),
            "cursor": result.get("cursor"),
            "size": result.get("size", 0),
        }

    # ------------------------------------------------------------------
    # Deletion (upload/remove — removes from space index, not from IPFS)
    # ------------------------------------------------------------------

    async def delete(self, cid_str: str) -> bool:
        """
        Remove a CID from the space index via upload/remove.

        Note: IPFS content is content-addressed and cannot be deleted from
        the network. This removes it from the space's upload index only.

        Args:
            cid_str: CID string of the upload to remove

        Returns:
            True if successfully removed from index
        """
        try:
            # Parse the CID string back to bytes for the invocation
            cid_bytes = _cid_str_to_bytes(cid_str)
            nb: Dict[str, Any] = {
                "root": cbor2.CBORTag(42, b"\x00" + cid_bytes),
            }
            del_block, del_cid = self.ucan._sign_invocation("upload/remove", nb)
            del_car = self.ucan.build_invocation_car([(del_block, del_cid)])
            await self._invoke(del_car)
            return True
        except Exception as e:
            logger.warning("Failed to remove CID %s from Storacha index: %s", cid_str, e)
            return False

    # ------------------------------------------------------------------
    # Usage / stats
    # ------------------------------------------------------------------

    async def get_usage(self) -> Dict[str, Any]:
        """
        Get space usage via store/list UCAN invocation.

        Returns:
            {"size": int}  — total bytes used in the space
        """
        nb: Dict[str, Any] = {"size": 25}
        usage_block, usage_cid = self.ucan._sign_invocation("store/list", nb)
        usage_car = self.ucan.build_invocation_car([(usage_block, usage_cid)])
        result = await self._invoke(usage_car)
        return {"size": result.get("size", 0)}

    # ------------------------------------------------------------------
    # Bridge invocation
    # ------------------------------------------------------------------

    async def _invoke(self, invocation_car: bytes) -> Dict[str, Any]:
        """
        POST a UCAN invocation CAR to the Storacha bridge.

        Args:
            invocation_car: CARv1 bytes containing the signed invocation(s)

        Returns:
            The "ok" result dict from the bridge receipt

        Raises:
            StorachaError: If the bridge returns an error receipt
            httpx.HTTPStatusError: On HTTP-level errors
        """
        client = await self._get_client()
        response = await client.post(
            STORACHA_BRIDGE_URL,
            content=invocation_car,
            headers={
                "Content-Type": "application/vnd.ipld.car",
                "Accept": "application/vnd.ipld.car",
            },
        )
        response.raise_for_status()
        return self._parse_bridge_response(response.content)

    @staticmethod
    def _parse_bridge_response(car_bytes: bytes) -> Dict[str, Any]:
        """
        Parse the CARv1 response from the bridge.

        The root block of the response CAR is a CBOR-encoded receipt with:
          {"ran": CID, "out": {"ok": {...}} | {"error": {...}}, ...}

        Returns:
            The contents of out.ok on success.

        Raises:
            StorachaError: If the receipt contains an error.
        """
        root_cids, blocks = parse_car(car_bytes)
        if not root_cids:
            raise StorachaError("Bridge returned an empty CAR response")

        root_data = blocks.get(root_cids[0])
        if root_data is None:
            raise StorachaError("Bridge response CAR missing root block")

        receipt = cbor2.loads(root_data)
        out = receipt.get("out", {})

        if "error" in out:
            error = out["error"]
            name = error.get("name", "UnknownError") if isinstance(error, dict) else str(error)
            msg = error.get("message", "") if isinstance(error, dict) else ""
            raise StorachaError(f"Storacha bridge error [{name}]: {msg}")

        return out.get("ok", {})


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _wrap_in_car(content: bytes, content_cid: bytes) -> bytes:
    """
    Wrap raw content bytes in a single-block CARv1.

    The resulting CAR has one root (the content CID) and one block
    (CID + content bytes). This is the format expected by store/add.
    """
    return build_car(
        root_cids=[content_cid],
        blocks=[(content_cid, content)],
    )


def _cid_str_to_bytes(cid_str: str) -> bytes:
    """
    Convert a base32lower CIDv1 string ("b...") back to raw CID bytes.

    Only handles the "b" (base32lower) multibase prefix used by CIDv1.
    Raises ValueError for unsupported formats.
    """
    import base64 as _base64

    if cid_str.startswith("b"):
        # base32lower: pad to multiple of 8 and decode
        b32 = cid_str[1:].upper()
        missing = len(b32) % 8
        if missing:
            b32 += "=" * (8 - missing)
        return _base64.b32decode(b32)

    raise ValueError(
        f"Unsupported CID format: {cid_str!r}. "
        "Only base32lower CIDv1 strings (starting with 'b') are supported."
    )
