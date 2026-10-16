"""
Async REST client for Lighthouse storage API.

Replaces the unmaintained lighthouseweb3 Python SDK (v0.1.1, May 2023)
with direct REST API calls using httpx for full async support.

Lighthouse REST API reference:
- Upload:  https://upload.lighthouse.storage/api/v0/add (POST multipart)
- List:    https://api.lighthouse.storage/api/user/files_uploaded (GET, cursor)
- Deals:   https://api.lighthouse.storage/api/lighthouse/deal_status?cid=X (no auth)
- Balance: https://api.lighthouse.storage/api/user/user_data_usage
- Gateway: https://gateway.lighthouse.storage/ipfs/{cid} (no auth)

The gateway serves any account's upload by CID to anyone, so reading it needs
no API key: ``LighthouseGateway`` reads it alone, and ``LighthouseRestClient``
adds the account API, which does.
"""

import io
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Sequence, Union

import httpx

logger = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 64 * 1024

#: The most bytes one read of an ``UploadBody`` returns, so the most the HTTP
#: stack hands a TLS connection in one write. httpx sends a ``bytes``
#: multipart file as one chunk, and httpcore passes that chunk to a single
#: ``ssl.SSLObject.write``, which encrypts all of it into OpenSSL's memory BIO
#: before any of it reaches the socket. That BIO cannot grow past ~1.9 GB
#: (``BUF_MEM_grow_clean``'s expansion limit): on OpenSSL 3.5.4 with TLS 1.3
#: one write of 1,902,794,057 bytes raises ``OSError: [Errno 0] Error``, with
#: nothing sent, and leaves a stale error on the thread's OpenSSL error queue
#: that fails the next read of every other TLS connection on the event loop.
#: Emma's snapshot CAR crossed that size on 2026-10-08 (#3662).
UPLOAD_CHUNK_BYTES = 64 * 1024

_CAR_CONTENT_TYPE = "application/vnd.ipld.car"

UploadSegment = Union[bytes, "os.PathLike[str]"]

# Fields that name one Filecoin deal in a deal_status record: its deal id
# (chain, client or UUID) or the storage provider holding it.
_DEAL_IDENTITY_KEYS = frozenset(
    {
        "DealID",
        "dealId",
        "dealID",
        "deal_id",
        "chainDealID",
        "dealUUID",
        "Provider",
        "StorageProvider",
        "storageProvider",
        "miner",
    }
)
# Keys a deal_status response wraps its list of deals in.
_DEAL_LIST_KEYS = ("data", "deals", "dealStatus", "dealInfo")


class LighthouseDownloadLimitError(ValueError):
    """A bounded gateway download exceeded its byte ceiling."""


def deal_records(status: Any) -> List[Mapping]:
    """The Filecoin deals a ``get_deal_status`` response names.

    The endpoint answers with a JSON list of deal records; the SDK wraps that
    list as ``{"data": [...]}``, and a single deal may arrive as one record.
    A record counts only when it names a deal (a deal id or the provider
    holding it), so a response this cannot read counts no deals rather than
    reporting storage that is not shown to exist.
    """
    if isinstance(status, (list, tuple)):
        records: List[Mapping] = []
        for item in status:
            records.extend(deal_records(item))
        return records
    if not isinstance(status, Mapping):
        return []
    if _DEAL_IDENTITY_KEYS.intersection(status):
        return [status]
    for key in _DEAL_LIST_KEYS:
        if key in status:
            return deal_records(status[key])
    return []


async def list_all_uploads(client: Any) -> List[Dict[str, Any]]:
    """Every upload record in the account ``client`` authenticates as.

    ``files_uploaded`` has no nextLastKey/lastKey: the cursor is the last
    record's id (page size 1000), and listing continues until a page is
    empty. A repeated or missing cursor ends it rather than looping. An
    entry that is not a record is dropped, so every reader gets mappings.
    """
    uploads: List[Dict[str, Any]] = []
    last_key: Optional[str] = None
    seen_cursors: set[str] = set()
    while True:
        page = await client.get_uploads(last_key=last_key)
        file_list = page.get("fileList", [])
        if not isinstance(file_list, list) or not file_list:
            break
        # A record is a JSON object; anything else names no upload.
        records = [item for item in file_list if isinstance(item, dict)]
        uploads.extend(records)
        last_key = str(records[-1].get("id")) if records and records[-1].get("id") else None
        if not last_key or last_key in seen_cursors:
            break
        seen_cursors.add(last_key)
    return uploads


class UploadBody(io.RawIOBase):
    """A known-length upload body, read at most ``UPLOAD_CHUNK_BYTES`` at a time.

    The body is its segments in order: ``bytes`` already in memory, or a file
    on disk that is read chunk by chunk as the upload proceeds, so a
    multi-GB file is never held in memory whole. httpx learns the length by
    seeking (so it sends a Content-Length) and seeks back to 0 before it
    sends. Every read returns at most ``UPLOAD_CHUNK_BYTES``, and an unsized
    ``read()`` is refused rather than drawing the whole body into one buffer.
    ``tell()`` is how much of the body the HTTP stack has taken so far.

    A file segment's size is fixed when the body is made. A file that
    shrinks while the body is in use fails the read instead of sending less
    than the declared length.
    """

    def __init__(
        self,
        segments: Sequence[UploadSegment],
        *,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        super().__init__()
        self._file_index: Optional[int] = None
        self._file: Optional[BinaryIO] = None
        self._segments: List[Union[bytes, Path]] = []
        self._sizes: List[int] = []
        for segment in segments:
            if isinstance(segment, bytes):
                self._segments.append(segment)
                self._sizes.append(len(segment))
            elif isinstance(segment, os.PathLike):
                path = Path(segment)
                self._segments.append(path)
                self._sizes.append(path.stat().st_size)
            else:
                raise TypeError(
                    "an upload body segment is bytes or a path, not "
                    f"{type(segment).__name__}"
                )
        #: Total bytes across every segment: the upload's declared length.
        self.length = sum(self._sizes)
        self._position = 0
        self._on_progress = on_progress
        self._report(0)

    def _report(self, sent: int) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(sent, self.length)
        except Exception:
            logger.debug("Lighthouse upload progress callback failed", exc_info=True)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._check_open()
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._check_open()
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"invalid whence ({whence})")
        if position < 0:
            raise ValueError(f"negative seek position {position}")
        self._position = position
        return position

    def read(self, size: Optional[int] = -1) -> bytes:
        self._check_open()
        if size is None or size < 0:
            raise io.UnsupportedOperation(
                "an UploadBody is read in bounded chunks; pass a size"
            )
        wanted = min(size, UPLOAD_CHUNK_BYTES, self.length - self._position)
        if wanted <= 0:
            return b""
        index, offset = self._locate(self._position)
        wanted = min(wanted, self._sizes[index] - offset)
        segment = self._segments[index]
        if isinstance(segment, bytes):
            chunk = segment[offset : offset + wanted]
        else:
            chunk = self._read_file(index, offset, wanted)
        self._position += len(chunk)
        self._report(self._position)
        return chunk

    def readinto(self, buffer: Any) -> int:
        chunk = self.read(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)

    def close(self) -> None:
        self._close_file()
        super().close()

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on a closed UploadBody")

    def _locate(self, position: int) -> tuple[int, int]:
        start = 0
        for index, size in enumerate(self._sizes):
            if position < start + size:
                return index, position - start
            start += size
        raise ValueError(f"position {position} is past the body's end")

    def _read_file(self, index: int, offset: int, wanted: int) -> bytes:
        handle = self._file
        if handle is None or self._file_index != index:
            self._close_file()
            handle = self._file = open(self._segments[index], "rb")
            self._file_index = index
        if handle.tell() != offset:
            handle.seek(offset)
        chunk = handle.read(wanted)
        if not chunk:
            raise ValueError(
                f"{self._segments[index]} ended at {offset} bytes, short of the "
                f"{self._sizes[index]} it had when the upload body was made"
            )
        return chunk

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._file_index = None


def _exception_chain(exc: BaseException, limit: int = 8) -> Iterator[BaseException]:
    """``exc`` and what it was raised from, outermost first.

    Follows ``__cause__``, else ``__context__`` even when suppressed: httpcore
    re-raises with ``from None``, which would otherwise hide the socket or
    TLS error underneath.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen and len(seen) < limit:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def describe_upload_failure(
    exc: BaseException, *, sent: int, total: int, elapsed: float
) -> str:
    """One line naming what an upload failure was and how far it got.

    The type, the errno of the first ``OSError`` underneath it, the HTTP
    status if the server answered, the body bytes the HTTP stack had taken,
    the time the request ran, and the exception chain: enough to tell a
    local TLS failure from a remote reset or refusal. ``OSError: [Errno 0]
    Error`` alone named none of these (#3662).
    """
    chain = list(_exception_chain(exc))
    errno = next(
        (link.errno for link in chain if isinstance(link, OSError) and link.errno is not None),
        None,
    )
    status = getattr(getattr(exc, "response", None), "status_code", None)
    cause = " <- ".join(
        type(link).__name__
        if type(link).__module__ == "builtins"
        else f"{type(link).__module__.split('.')[0]}.{type(link).__name__}"
        for link in chain
    )
    return (
        f"{type(exc).__name__} errno={errno} status={status} "
        f"bytes_sent={sent}/{total} elapsed={elapsed:.1f}s cause={cause}: {exc}"
    )


class LighthouseGateway:
    """Lighthouse's public IPFS gateway: any upload, from any account, by CID.

    A read needs no API key and is not tied to the account that uploaded the
    content, so a host that holds no Lighthouse key at all (a fresh host
    restoring a sovereignty export, #3601) reads through this alone.
    """

    #: Payload per second of patience a transfer is granted, per socket
    #: operation, in either direction. The budget for a payload is
    #: ``max(timeout, size / this)``, so a 60 s default still applies to
    #: small requests and a 1.2 GB snapshot gets ~2300 s for the server's
    #: processing before it answers.
    TRANSFER_FLOOR_BYTES_PER_SECOND = 512 * 1024

    def __init__(
        self,
        gateway_url: str = "https://gateway.lighthouse.storage/ipfs",
        timeout: float = 60.0,
    ):
        """
        Initialize a Lighthouse gateway reader.

        Args:
            gateway_url: IPFS gateway URL for downloads
            timeout: Default request timeout in seconds
        """
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

    def transfer_timeout(self, payload_bytes: int) -> "httpx.Timeout":
        """The request budget for transferring ``payload_bytes`` either way.

        httpx applies ``read`` and ``write`` per socket operation, not per
        transfer: each chunk written and each read of the response headers
        gets the full budget again. What fired for eighteen days was the
        read after the body was sent: Lighthouse hashes and stores a 1.2 GB
        CAR before it answers, and that wait grows with the payload, so the
        budget is sized by the payload (``payload_bytes /
        TRANSFER_FLOOR_BYTES_PER_SECOND``, never below the default) and given
        to both operations. The floor is a rate of payload per second of
        patience, not a link speed. Connect and pool stay at the default:
        they are not proportional to the payload (#3189).
        """
        budget = max(
            float(self.timeout), payload_bytes / self.TRANSFER_FLOOR_BYTES_PER_SECOND
        )
        return httpx.Timeout(
            connect=self.timeout, read=budget, write=budget, pool=self.timeout
        )

    async def download(
        self,
        cid: str,
        timeout: Optional[float] = None,
        *,
        expected_bytes: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> bytes:
        """
        Download content from IPFS gateway.

        Args:
            cid: IPFS Content ID
            timeout: Explicit request timeout; wins over ``expected_bytes``
            expected_bytes: The object's size when known (a manifest's
                ``snapshot_size``); buys the payload-sized budget of
                :meth:`transfer_timeout` instead of the flat default
            max_bytes: A ceiling on the bytes read. The body is streamed and
                the read stops, raising ``LighthouseDownloadLimitError``, as
                soon as it would pass the ceiling, so a caller restoring
                whatever a CID returns never holds more than it allows.

        Returns:
            Content bytes
        """
        client = await self._get_client()
        url = f"{self.gateway_url}/{cid}"
        # The gateway assembles the whole object before the first byte, so
        # a known size buys the same payload-proportional patience as an
        # upload; an explicit timeout wins, then the flat default.
        request_timeout = (
            timeout
            if timeout is not None
            else self.transfer_timeout(expected_bytes)
            if expected_bytes
            else 120.0
        )
        if max_bytes is None:
            response = await client.get(url, timeout=request_timeout)
            response.raise_for_status()
            return response.content

        # Identity encoding only, read raw, so the ceiling counts the bytes
        # the gateway sends; a compressed body would be expanded, chunk by
        # chunk, before anything could count it.
        async with client.stream(
            "GET",
            url,
            headers={"Accept-Encoding": "identity"},
            timeout=request_timeout,
        ) as response:
            response.raise_for_status()
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if encoding not in ("", "identity"):
                raise ValueError(
                    f"gateway sent {cid} with Content-Encoding {encoding!r}; a "
                    "bounded download reads identity bodies only"
                )
            declared = response.headers.get("Content-Length")
            if declared is not None and declared.isdigit() and int(declared) > max_bytes:
                raise LighthouseDownloadLimitError(
                    f"gateway response for {cid} exceeds {max_bytes} bytes"
                )
            chunks: List[bytes] = []
            total = 0
            async for chunk in response.aiter_raw(_DOWNLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    raise LighthouseDownloadLimitError(
                        f"gateway response for {cid} exceeds {max_bytes} bytes"
                    )
                chunks.append(chunk)
        return b"".join(chunks)


class LighthouseRestClient(LighthouseGateway):
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
        super().__init__(gateway_url=gateway_url, timeout=timeout)
        self.api_key = api_key

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
        with UploadBody([content], on_progress=on_progress) as body:
            return await self._post_upload(body, filename=filename, tag=tag)

    async def upload_car(
        self,
        car_bytes: Optional[bytes] = None,
        tag: str = "kestrel-storage",
        filename: str = "export.car",
        on_progress: Optional[Callable[[int, int], None]] = None,
        *,
        body: Optional[UploadBody] = None,
    ) -> Dict[str, Any]:
        """
        Upload CAR (Content Addressable aRchive) file.

        Lighthouse SDK v0.4.4+ supports direct CAR upload for
        faster, verifiable ingestion of content-addressed DAGs.

        Args:
            car_bytes: CAR v1 file bytes
            tag: Tag for organizing uploads
            filename: Name for the uploaded CAR file
            on_progress: Optional callback(bytes_sent, total_bytes)
            body: The CAR as an ``UploadBody`` instead of ``car_bytes``, for
                an archive streamed from disk. It reports its own progress
                and stays open for the caller to close.

        Returns:
            Dict with 'Hash' (CID), 'Name', and 'Size' keys
        """
        if (car_bytes is None) == (body is None):
            raise TypeError("upload_car takes exactly one of car_bytes or body")
        if body is None:
            with UploadBody([car_bytes], on_progress=on_progress) as owned:
                return await self._post_upload(
                    owned, filename=filename, tag=tag, content_type=_CAR_CONTENT_TYPE
                )
        if on_progress is not None:
            raise TypeError("an UploadBody reports its own progress; pass on_progress to it")
        return await self._post_upload(
            body, filename=filename, tag=tag, content_type=_CAR_CONTENT_TYPE
        )

    async def _post_upload(
        self,
        body: UploadBody,
        *,
        filename: str,
        tag: str,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST ``body`` as the multipart ``file`` of ``/api/v0/add``.

        A failure is logged at WARNING with ``describe_upload_failure`` and
        re-raised.
        """
        client = await self._get_client()
        file_field = (filename, body) if content_type is None else (filename, body, content_type)
        started = time.monotonic()
        try:
            response = await client.post(
                f"{self.UPLOAD_URL}/api/v0/add",
                headers=self._auth_headers,
                files={"file": file_field},
                params={"tag": tag},
                timeout=self.transfer_timeout(body.length),
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning(
                "Lighthouse upload of %s failed: %s",
                filename,
                describe_upload_failure(
                    exc,
                    sent=body.tell(),
                    total=body.length,
                    elapsed=time.monotonic() - started,
                ),
            )
            raise
        # Normalize: Lighthouse wraps in {"data": {...}} sometimes
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data

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

    async def delete_file(self, file_id: str) -> Dict[str, Any]:
        """
        Delete an upload record from the Lighthouse account.

        This removes the account-visible upload and quota usage; it does not
        attempt to cancel immutable Filecoin deals for already-pinned content.
        """
        client = await self._get_client()
        response = await client.delete(
            f"{self.API_URL}/api/user/delete_file",
            headers={**self._auth_headers, "Content-Type": "application/json"},
            params={"id": file_id},
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data if isinstance(data, dict) else {"deleted": True}

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

    async def get_auth_message(self, public_key: str) -> str:
        """
        Fetch Lighthouse's wallet-auth challenge for an EVM public address.

        Args:
            public_key: EVM address (0x...) that will sign the challenge

        Returns:
            Message string to sign with Ethereum personal_sign semantics.
        """
        client = await self._get_client()
        response = await client.get(
            f"{self.API_URL}/api/auth/get_message",
            params={"publicKey": public_key},
        )
        response.raise_for_status()

        data = response.json()
        if isinstance(data, str) and data:
            return data
        if isinstance(data, dict):
            if isinstance(data.get("data"), dict) and data["data"].get("message"):
                return str(data["data"]["message"])
            if data.get("message"):
                return str(data["message"])
        if (
            isinstance(data, list)
            and data
            and isinstance(data[0], dict)
            and data[0].get("message")
        ):
            return str(data[0]["message"])
        raise ValueError("Lighthouse auth message response did not include a message")

    async def create_api_key(self, public_key: str, signed_message: str) -> str:
        """
        Create a Lighthouse API key from a signed wallet-auth challenge.

        Args:
            public_key: EVM address (0x...) that signed the challenge
            signed_message: Ethereum personal_sign signature for the challenge

        Returns:
            Lighthouse API key.
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.API_URL}/api/auth/create_api_key",
            json={"publicKey": public_key, "signedMessage": signed_message},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()

        data = response.json()
        if isinstance(data, str) and data:
            return data
        candidates = []
        if isinstance(data, dict):
            candidates.append(data)
            if isinstance(data.get("data"), dict):
                candidates.append(data["data"])
        for item in candidates:
            for key in ("apiKey", "api_key", "key"):
                value = item.get(key)
                if value:
                    return str(value)
        raise ValueError("Lighthouse create_api_key response did not include an API key")
