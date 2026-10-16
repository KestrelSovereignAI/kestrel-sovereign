"""Lighthouse snapshot uploads reach TLS in bounded writes (#3662).

Emma's Lighthouse backup leg failed every run from 2026-10-08 00:02Z with
``OSError: [Errno 0] Error``. httpx sends a ``bytes`` multipart file as one
chunk, httpcore hands each chunk to one ``anyio`` TLS ``send``, and that is
one ``ssl.SSLObject.write``: OpenSSL encrypts the whole write into a memory
BIO before any of it reaches the socket. That BIO cannot grow past ~1.9 GB,
so on OpenSSL 3.5.4 a single write of more than 1,902,794,056 bytes raises
``OSError(0, "Error")`` locally, with nothing of the body sent. Emma's last
good CAR was 1,902,750,356 bytes; the next one was ~1,906,760,000.

The backend below keeps httpx, httpcore and h11 real and replaces only the
connection, with one whose ``write`` fails the way that BIO does, at a
ceiling scaled down so a test payload can cross it. The real ceiling is
exercised, opt-in, by ``tests/load/test_lighthouse_tls_write_ceiling.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import re
import sqlite3
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import anyio
import httpcore
import httpx
import pytest

from kestrel_sovereign.storage.car_builder import (
    CARBuilder,
    single_raw_block_car_prefix,
)
from kestrel_sovereign.storage.providers import lighthouse_rest
from kestrel_sovereign.storage.providers.lighthouse_rest import (
    UPLOAD_CHUNK_BYTES,
    LighthouseRestClient,
    UploadBody,
    describe_upload_failure,
)
from kestrel_sovereign.storage.sync import lighthouse_target, targets
from kestrel_sovereign.storage.sync.lighthouse_target import LighthouseTarget

#: Stands in for the BIO's 1,902,794,056 bytes: the largest single write the
#: fake connection accepts.
SCALED_CEILING = 1024 * 1024


class _Request:
    """What one fake connection received."""

    def __init__(self) -> None:
        self.pending = b""
        self.head: Optional[bytes] = None
        self.content_length: Optional[int] = None
        self.body_bytes = 0
        self.max_body_write = 0
        self.body = bytearray()


class _MemoryBioTLSStream(httpcore.AsyncNetworkStream):
    def __init__(self, backend: "_MemoryBioTLSBackend") -> None:
        self._backend = backend
        self.request = _Request()
        backend.requests.append(self.request)
        self._response: Optional[bytes] = None

    async def write(self, buffer: bytes, timeout: Optional[float] = None) -> None:
        if len(buffer) > self._backend.ceiling:
            # What SSLObject.write raised for Emma, before sending anything.
            raise OSError(0, "Error")
        request = self.request
        if request.head is None:
            request.pending += buffer
            if b"\r\n\r\n" not in request.pending:
                return
            head, _, buffer = request.pending.partition(b"\r\n\r\n")
            request.head = head
            match = re.search(rb"(?im)^content-length:\s*(\d+)", head)
            request.content_length = int(match.group(1)) if match else None
            if not buffer:
                return
        request.max_body_write = max(request.max_body_write, len(buffer))
        fail_at = self._backend.fail_at_body_bytes
        if fail_at is not None and request.body_bytes + len(buffer) > fail_at:
            raise self._backend.failure
        request.body_bytes += len(buffer)
        if self._backend.keep_body:
            request.body += buffer

    async def read(self, max_bytes: int, timeout: Optional[float] = None) -> bytes:
        if self._response is None:
            payload = json.dumps(
                {"Hash": f"QmFake{len(self._backend.requests)}", "Size": str(self.request.body_bytes)}
            ).encode()
            reason = b"OK" if self._backend.status == 200 else b"Error"
            self._response = (
                b"HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
                b"Content-Length: %d\r\nConnection: close\r\n\r\n"
                % (self._backend.status, reason, len(payload))
            ) + payload
        chunk, self._response = self._response[:max_bytes], self._response[max_bytes:]
        return chunk

    async def aclose(self) -> None:
        return None

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self._backend.tls_handshakes += 1
        return self

    def get_extra_info(self, info: str):
        return None


class _MemoryBioTLSBackend(httpcore.AsyncNetworkBackend):
    """Connections whose writes fail like a TLS write into OpenSSL's memory BIO."""

    def __init__(
        self,
        *,
        ceiling: int = SCALED_CEILING,
        status: int = 200,
        fail_at_body_bytes: Optional[int] = None,
        failure: Optional[BaseException] = None,
        keep_body: bool = False,
    ) -> None:
        self.ceiling = ceiling
        self.status = status
        self.fail_at_body_bytes = fail_at_body_bytes
        self.failure = failure
        self.keep_body = keep_body
        self.requests: list[_Request] = []
        self.tls_handshakes = 0

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        return _MemoryBioTLSStream(self)

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise NotImplementedError

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


def _client_over(backend: _MemoryBioTLSBackend) -> type[LighthouseRestClient]:
    class _Client(LighthouseRestClient):
        # A failed patch must not reach the real service.
        UPLOAD_URL = "https://upload.lighthouse.invalid"

        async def _get_client(self) -> httpx.AsyncClient:
            if self._client is None or self._client.is_closed:
                transport = httpx.AsyncHTTPTransport()
                # httpx 0.28 takes no network backend; its pool does.
                transport._pool = httpcore.AsyncConnectionPool(network_backend=backend)
                self._client = httpx.AsyncClient(transport=transport, timeout=self.timeout)
            return self._client

    return _Client


def _sqlite_db(path: Path, blob_bytes: int) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x BLOB)")
    conn.execute("INSERT INTO t VALUES (?)", (b"\x5a" * blob_bytes,))
    conn.commit()
    conn.close()
    return path


def _multipart_file(body: bytes) -> bytes:
    """The content of the single file part of a multipart body."""
    boundary = body.split(b"\r\n", 1)[0]
    _, _, part = body.partition(b"\r\n\r\n")
    return part[: part.rindex(b"\r\n" + boundary + b"--")]


# ---------------------------------------------------------------------------
# The failure: one write the size of the payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_car_larger_than_one_tls_write_is_uploaded_in_bounded_writes():
    """Before #3662 the whole CAR was one write, so this raised
    ``OSError(0, "Error")`` exactly as Emma's 52 failed runs did."""
    backend = _MemoryBioTLSBackend()
    client = _client_over(backend)(api_key="k")
    car = bytes(range(256)) * (3 * SCALED_CEILING // 256)
    try:
        result = await client.upload_car(car, tag="t", filename="big.car")
    finally:
        await client.close()

    assert result["Hash"] == "QmFake1"
    assert backend.tls_handshakes == 1, "the fake connection must carry the upload"
    (request,) = backend.requests
    assert request.body_bytes == request.content_length > len(car)
    assert request.max_body_write <= UPLOAD_CHUNK_BYTES


@pytest.mark.asyncio
async def test_the_upload_door_sovereignty_exports_use_is_bounded_too():
    """``upload()`` carries sovereignty exports through the same POST."""
    backend = _MemoryBioTLSBackend()
    client = _client_over(backend)(api_key="k")
    try:
        await client.upload(b"e" * (2 * SCALED_CEILING + 1), filename="export.bin")
    finally:
        await client.close()

    (request,) = backend.requests
    assert request.body_bytes == request.content_length
    assert request.max_body_write <= UPLOAD_CHUNK_BYTES


@pytest.mark.asyncio
async def test_a_snapshot_whose_car_outgrows_one_tls_write_is_backed_up(tmp_path, monkeypatch):
    """Emma's symptom end to end: before #3662 this returned
    ``success=False, error="OSError: [Errno 0] Error"``. The archive that
    arrives is the same single-block CAR, and it restores."""
    backend = _MemoryBioTLSBackend(keep_body=True)
    monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", _client_over(backend))
    db = _sqlite_db(tmp_path / "agent.db", 3 * SCALED_CEILING)
    target = LighthouseTarget(api_key="k", agent_id="agent", state_dir=tmp_path)

    result = await target.sync_snapshot(db)

    assert result.success is True, result.error
    snapshot, manifest_upload = backend.requests
    assert snapshot.body_bytes == snapshot.content_length
    assert snapshot.max_body_write <= UPLOAD_CHUNK_BYTES
    uploaded_car = _multipart_file(bytes(snapshot.body))
    raw = targets._create_consistent_snapshot(db)
    assert uploaded_car == target._build_snapshot_car(raw)[0]
    assert target._extract_snapshot_content(uploaded_car) == raw

    manifest = target._load_local_manifest()
    assert manifest["snapshot_cid"] == "QmFake1"
    assert manifest["raw_snapshot_size"] == len(raw)
    assert manifest["content_hash"] == hashlib.sha256(raw).hexdigest()
    assert manifest["snapshot_payload_cid"] == target._build_snapshot_car(raw)[1]
    assert manifest_upload.body_bytes == manifest_upload.content_length


@pytest.mark.asyncio
async def test_a_snapshot_over_2_gib_is_never_held_whole_in_memory(tmp_path, monkeypatch):
    """Emma's DB is 2,184,335,360 bytes now. Before #3662 the target held the
    snapshot and its CAR whole (a 12 GiB peak for that size, measured);
    the streamed upload's Python allocations stay within a few chunks."""
    size = 2 * 1024**3 + 3 * UPLOAD_CHUNK_BYTES + 17
    sparse = tmp_path / "sparse.db"
    with open(sparse, "wb") as f:
        f.truncate(size)

    @contextmanager
    def _the_sparse_snapshot(db_path):
        yield sparse

    monkeypatch.setattr(lighthouse_target, "_consistent_snapshot_file", _the_sparse_snapshot)
    backend = _MemoryBioTLSBackend()
    monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", _client_over(backend))
    target = LighthouseTarget(api_key="k", agent_id="agent", state_dir=tmp_path)

    tracemalloc.start()
    try:
        result = await target.sync_snapshot(tmp_path / "agent.db")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.success is True, result.error
    snapshot = backend.requests[0]
    assert snapshot.body_bytes == snapshot.content_length > size
    assert snapshot.max_body_write <= UPLOAD_CHUNK_BYTES
    assert peak < 32 * 1024 * 1024, f"peak traced allocation {peak:,} bytes"
    assert target._load_local_manifest()["raw_snapshot_size"] == size


# ---------------------------------------------------------------------------
# The failure log
# ---------------------------------------------------------------------------


def _upload_warning(caplog) -> logging.LogRecord:
    (record,) = [
        r for r in caplog.records
        if r.name == lighthouse_rest.__name__ and r.getMessage().startswith("Lighthouse upload of ")
    ]
    return record


@pytest.mark.asyncio
async def test_an_upload_failure_logs_type_errno_bytes_sent_and_elapsed(tmp_path, monkeypatch, caplog):
    backend = _MemoryBioTLSBackend(
        fail_at_body_bytes=4 * UPLOAD_CHUNK_BYTES,
        failure=ConnectionResetError(54, "Connection reset by peer"),
    )
    monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", _client_over(backend))
    db = _sqlite_db(tmp_path / "agent.db", 2 * SCALED_CEILING)
    target = LighthouseTarget(api_key="k", agent_id="agent", state_dir=tmp_path)

    with caplog.at_level(logging.WARNING):
        result = await target.sync_snapshot(db)

    assert result.success is False
    assert result.error == "ConnectionResetError: [Errno 54] Connection reset by peer"
    assert (
        "Failed to sync to Lighthouse: ConnectionResetError: [Errno 54] Connection reset by peer"
        in caplog.text
    )
    record = _upload_warning(caplog)
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert re.match(r"Lighthouse upload of kestrel_state__agent__\d{8}_\d{6}\.car failed: ", message)
    assert "ConnectionResetError errno=54 status=None" in message
    sent, total = map(int, re.search(r"bytes_sent=(\d+)/(\d+)", message).groups())
    raw_size = (tmp_path / "agent.db").stat().st_size
    assert total > raw_size
    assert 3 * UPLOAD_CHUNK_BYTES < sent <= 5 * UPLOAD_CHUNK_BYTES < total
    assert re.search(r" elapsed=\d+\.\ds ", message)
    assert "cause=ConnectionResetError: [Errno 54] Connection reset by peer" in message


@pytest.mark.asyncio
async def test_a_refused_upload_logs_its_status_and_the_whole_body_sent(caplog):
    backend = _MemoryBioTLSBackend(status=500)
    client = _client_over(backend)(api_key="k")
    with caplog.at_level(logging.WARNING):
        with pytest.raises(httpx.HTTPStatusError):
            try:
                await client.upload_car(b"c" * 1000, tag="t", filename="small.car")
            finally:
                await client.close()

    message = _upload_warning(caplog).getMessage()
    assert "HTTPStatusError errno=None status=500 bytes_sent=1000/1000 " in message


def test_the_failure_line_finds_the_errno_beneath_httpx_and_anyio():
    """httpx's ReadError carries no errno itself; it sits under httpcore's
    and anyio's wrappers, and httpcore re-raises ``from None``."""
    try:
        try:
            try:
                try:
                    raise ConnectionResetError(54, "Connection reset by peer")
                except OSError as exc:
                    raise anyio.BrokenResourceError from exc
            except anyio.BrokenResourceError as exc:
                raise httpcore.ReadError("") from exc
        except httpcore.ReadError as exc:
            raise httpx.ReadError("") from exc
    except httpx.ReadError as exc:
        line = describe_upload_failure(exc, sent=10, total=20, elapsed=1.25)

    assert line.startswith("ReadError errno=54 status=None bytes_sent=10/20 elapsed=1.2s ")
    assert (
        "cause=httpx.ReadError <- httpcore.ReadError <- anyio.BrokenResourceError"
        " <- ConnectionResetError: " in line
    )


def test_the_failure_line_for_emmas_error():
    line = describe_upload_failure(OSError(0, "Error"), sent=0, total=5, elapsed=0.6)
    assert line == "OSError errno=0 status=None bytes_sent=0/5 elapsed=0.6s cause=OSError: [Errno 0] Error"


# ---------------------------------------------------------------------------
# UploadBody
# ---------------------------------------------------------------------------


def test_a_body_never_returns_more_than_one_chunk_per_read(tmp_path):
    payload = tmp_path / "payload"
    payload.write_bytes(b"p" * (3 * UPLOAD_CHUNK_BYTES))
    with UploadBody([b"head", payload, b"tail"]) as body:
        assert body.length == 3 * UPLOAD_CHUNK_BYTES + 8
        sizes = []
        data = bytearray()
        while chunk := body.read(10 * UPLOAD_CHUNK_BYTES):
            sizes.append(len(chunk))
            data += chunk
        assert max(sizes) <= UPLOAD_CHUNK_BYTES
        assert bytes(data) == b"head" + payload.read_bytes() + b"tail"
        assert body.tell() == body.length


def test_an_unsized_read_is_refused_rather_than_buffering_the_body():
    with UploadBody([b"x" * 10]) as body:
        with pytest.raises(io.UnsupportedOperation):
            body.read()
        with pytest.raises(io.UnsupportedOperation):
            body.read(-1)


def test_a_body_reports_its_length_by_seeking_and_rereads_from_zero(tmp_path):
    payload = tmp_path / "payload"
    payload.write_bytes(b"0123456789")
    with UploadBody([b"ab", payload]) as body:
        assert body.read(5) == b"ab"
        assert body.read(5) == b"01234"
        assert httpx._utils.peek_filelike_length(body) == 12
        assert body.tell() == 7
        body.seek(0)
        assert body.read(100) == b"ab"
        assert body.read(100) == b"0123456789"
        assert body.read(100) == b""


def test_a_file_that_shrinks_under_the_body_fails_the_read(tmp_path):
    payload = tmp_path / "payload"
    payload.write_bytes(b"z" * 100)
    with UploadBody([payload]) as body:
        payload.write_bytes(b"z" * 40)
        assert body.read(100) == b"z" * 40
        with pytest.raises(ValueError, match="short of the 100"):
            body.read(100)


def test_a_body_reports_progress_in_file_bytes():
    progress = []
    with UploadBody(
        [b"y" * (2 * UPLOAD_CHUNK_BYTES)],
        on_progress=lambda sent, total: progress.append((sent, total)),
    ) as body:
        while body.read(UPLOAD_CHUNK_BYTES):
            pass
    total = 2 * UPLOAD_CHUNK_BYTES
    assert progress == [(0, total), (UPLOAD_CHUNK_BYTES, total), (total, total)]


def test_a_body_segment_is_bytes_or_a_path():
    with pytest.raises(TypeError, match="bytes or a path"):
        UploadBody(["not a path"])


def test_a_closed_body_refuses_reads():
    body = UploadBody([b"x"])
    body.close()
    with pytest.raises(ValueError):
        body.read(1)
    with pytest.raises(ValueError):
        body.tell()


@pytest.mark.asyncio
async def test_upload_car_takes_exactly_one_payload():
    client = LighthouseRestClient(api_key="k")
    with pytest.raises(TypeError, match="exactly one"):
        await client.upload_car(tag="t")
    with UploadBody([b"x"]) as body:
        with pytest.raises(TypeError, match="exactly one"):
            await client.upload_car(b"x", body=body)
        with pytest.raises(TypeError, match="its own progress"):
            await client.upload_car(body=body, on_progress=lambda *_: None)


# ---------------------------------------------------------------------------
# The streamed archive and the on-disk snapshot
# ---------------------------------------------------------------------------


def test_the_streamed_car_is_byte_for_byte_the_built_one():
    content = bytes(range(256)) * 300
    prefix, cid = single_raw_block_car_prefix(hashlib.sha256(content).digest(), len(content))
    builder = CARBuilder()
    built_cid = builder.add_raw_block(content)
    builder.set_root(built_cid)
    assert prefix + content == builder.build()
    assert cid == built_cid


def test_the_snapshot_file_is_removed_on_exit_and_on_error(tmp_path):
    db = _sqlite_db(tmp_path / "agent.db", 1000)
    with targets._consistent_snapshot_file(db) as snapshot:
        assert snapshot.read_bytes() == targets._create_consistent_snapshot(db)
        assert snapshot != db
    assert not snapshot.exists()

    with pytest.raises(RuntimeError):
        with targets._consistent_snapshot_file(db) as snapshot:
            raise RuntimeError("upload failed")
    assert not snapshot.exists()


def test_a_file_that_is_not_sqlite_is_copied_raw(tmp_path):
    raw = tmp_path / "raw.db"
    raw.write_bytes(b"not a database at all")
    with targets._consistent_snapshot_file(raw) as snapshot:
        assert snapshot.read_bytes() == b"not a database at all"
    assert not snapshot.exists()
