"""A GCS snapshot is sent once, under limits a slow uplink can meet (#3944).

Each 2+ GB snapshot was uploaded twice in full, the second time for
``latest.db``, under the library defaults: 100 MiB chunks, 60 s per request
and a 120 s retry deadline. On a shared uplink two of four agents failed the
backup. ``latest.db`` is now a server-side rewrite of the uploaded blob, and
the upload states its own chunk size, timeout and retry.

The ``google.cloud.storage`` client is a local fake: no test reaches GCS or
reads credentials. Every fake call is bound against the real library's
signature, so a keyword the library does not take fails here.
"""

from __future__ import annotations

import hashlib
import inspect
import sqlite3
import tempfile
import threading
from contextlib import closing

import pytest
import requests
from google.api_core.exceptions import RetryError
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY

from kestrel_sovereign.storage.sync import gcs_transfer
from kestrel_sovereign.storage.sync.gcs_target import (
    GCS_REQUEST_TIMEOUT_SECONDS,
    GCS_RETRY_DEADLINE_SECONDS,
    GCS_UPLOAD_CHUNK_BYTES,
    SNAPSHOT_CONTENT_TYPE,
    GCSTarget,
)

_BUCKET = "unit-test-bucket"
_LATEST = "kestrel/agent/latest.db"
_TIMED_OUT = RetryError(
    "Timeout of 120.0s exceeded", TimeoutError("The write operation timed out")
)


class _FakeBlob:
    def __init__(self, bucket, name, chunk_size):
        storage.Blob(name, bucket=None, chunk_size=chunk_size)  # the real setter's check
        self.bucket = bucket
        self.name = name
        self.chunk_size = chunk_size
        self.content_type = None

    def _upload(self, method, data, kwargs):
        inspect.signature(getattr(storage.Blob, method)).bind(self, data, **kwargs)
        if self.bucket.upload_error is not None:
            raise self.bucket.upload_error
        self.bucket.uploads.append(
            {"blob": self, "chunk_size": self.chunk_size, "data": data, **kwargs}
        )

    def upload_from_string(self, data, **kwargs):
        self._upload("upload_from_string", data, kwargs)

    def upload_from_file(self, file_obj, **kwargs):
        self._upload("upload_from_file", file_obj.read(), kwargs)

    def upload_from_filename(self, filename, **kwargs):
        with open(filename, "rb") as source:
            self._upload("upload_from_filename", source.read(), kwargs)

    def rewrite(self, source, **kwargs):
        inspect.signature(storage.Blob.rewrite).bind(self, source, **kwargs)
        self.bucket.rewrites.append(
            {"blob": self, "source": source, "content_type": self.content_type, **kwargs}
        )
        if self.bucket.rewrite_error is not None:
            raise self.bucket.rewrite_error
        token = self.bucket.rewrite_tokens.pop(0)
        return token, 0, len(self.bucket.uploads[-1]["data"])


class _FakeBucket:
    def __init__(self):
        self.uploads: list[dict] = []
        self.rewrites: list[dict] = []
        self.rewrite_tokens: list[str | None] = [None]
        self.upload_error: Exception | None = None
        self.rewrite_error: Exception | None = None
        self.transports: list[requests.Session] = []

    def blob(self, blob_name, **kwargs):
        inspect.signature(storage.Bucket.blob).bind(self, blob_name, **kwargs)
        return _FakeBlob(self, blob_name, kwargs.get("chunk_size"))


@pytest.fixture
def bucket(monkeypatch):
    fake = _FakeBucket()
    clients: list[dict] = []

    class Client:
        def __init__(self, **kwargs):
            clients.append(kwargs)
            # The real client's transport is a ``requests`` session.
            self._http = requests.Session()
            fake.transports.append(self._http)

        def bucket(self, name):
            assert name == _BUCKET
            return fake

    monkeypatch.setattr(storage, "Client", Client)
    yield fake
    assert clients == [{}]  # no project, no credentials file


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return GCSTarget(_BUCKET, agent_id="agent", state_dir=tmp_path / "state")


@pytest.fixture
def agent_db(tmp_path):
    path = tmp_path / "agent.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, body TEXT)")
        conn.executemany(
            "INSERT INTO memories VALUES (?, ?)",
            [(index, f"memory {index}") for index in range(50)],
        )
        conn.commit()
    return path


def _add_row(path):
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("INSERT INTO memories (body) VALUES ('later')")
        conn.commit()


async def test_one_sync_uploads_the_snapshot_once_and_rewrites_latest(
    bucket, target, agent_db
):
    result = await target.sync_snapshot(agent_db)

    assert result.success is True, result.error
    (upload,) = bucket.uploads
    blob = upload["blob"]
    assert blob.name == result.metadata["blob"]
    assert blob.name.startswith("kestrel/agent/snapshots/")
    assert upload["content_type"] == SNAPSHOT_CONTENT_TYPE
    assert result.bytes_synced == len(upload["data"])
    manifest = target._load_local_manifest()
    assert manifest["content_hash"] == hashlib.sha256(upload["data"]).hexdigest()

    (rewrite,) = bucket.rewrites
    assert rewrite["blob"].name == _LATEST
    assert rewrite["source"] is blob
    assert rewrite["token"] is None
    assert rewrite["content_type"] == SNAPSHOT_CONTENT_TYPE


async def test_the_upload_and_rewrite_use_the_explicit_limits(
    bucket, target, agent_db
):
    result = await target.sync_snapshot(agent_db)

    assert result.success is True, result.error
    (upload,) = bucket.uploads
    assert upload["chunk_size"] == GCS_UPLOAD_CHUNK_BYTES
    assert upload["timeout"] == GCS_REQUEST_TIMEOUT_SECONDS
    for call in [upload, *bucket.rewrites]:
        assert call["timeout"] == GCS_REQUEST_TIMEOUT_SECONDS
        retry = call["retry"]
        assert retry.timeout == GCS_RETRY_DEADLINE_SECONDS
        assert retry._predicate is DEFAULT_RETRY._predicate


def test_the_limits_let_one_slow_chunk_finish():
    assert GCS_UPLOAD_CHUNK_BYTES % storage.Blob._CHUNK_SIZE_MULTIPLE == 0
    # The library defaults that failed: 100 MiB, 60 s, a 120 s deadline.
    assert GCS_UPLOAD_CHUNK_BYTES < 100 * 1024 * 1024
    assert GCS_REQUEST_TIMEOUT_SECONDS > 60
    assert GCS_RETRY_DEADLINE_SECONDS > GCS_REQUEST_TIMEOUT_SECONDS


async def test_a_rewrite_that_returns_tokens_runs_until_it_completes(
    bucket, target, agent_db
):
    bucket.rewrite_tokens = ["token-1", "token-2", None]

    result = await target.sync_snapshot(agent_db)

    assert result.success is True, result.error
    assert [call["token"] for call in bucket.rewrites] == [None, "token-1", "token-2"]
    assert {call["blob"].name for call in bucket.rewrites} == {_LATEST}
    assert len(bucket.uploads) == 1
    assert target._load_local_manifest()["blob_name"] == result.metadata["blob"]


@pytest.mark.parametrize("failing", ["upload", "rewrite"])
async def test_a_retry_error_fails_the_sync_and_keeps_the_manifest(
    failing, bucket, target, agent_db, caplog
):
    first = await target.sync_snapshot(agent_db)
    assert first.success is True, first.error
    manifest = target._manifest_path.read_bytes()
    _add_row(agent_db)
    setattr(bucket, f"{failing}_error", _TIMED_OUT)

    with caplog.at_level("ERROR"):
        result = await target.sync_snapshot(agent_db)

    assert result.success is False
    assert result.bytes_synced == 0
    expected = (
        "RetryError: Timeout of 120.0s exceeded, "
        "last exception: The write operation timed out"
    )
    assert result.error == expected
    assert f"Failed to sync snapshot to GCS: {expected}" in caplog.text
    assert target._manifest_path.read_bytes() == manifest
    # A failed upload is never copied to latest.db.
    assert len(bucket.rewrites) == (1 if failing == "upload" else 2)


async def test_the_client_transport_can_be_interrupted(bucket, target, agent_db):
    """The upload's requests go through a transport its cancellation can
    interrupt (#4028), and it runs on a thread of its own."""
    threads: list[str] = []
    original = _FakeBlob.upload_from_string

    def record_thread(self, data, **kwargs):
        threads.append(threading.current_thread().name)
        original(self, data, **kwargs)

    _FakeBlob.upload_from_string = record_thread
    try:
        result = await target.sync_snapshot(agent_db)
    finally:
        _FakeBlob.upload_from_string = original

    assert result.success is True, result.error
    (transport,) = bucket.transports
    assert {type(adapter) for adapter in transport.adapters.values()} == {
        gcs_transfer._AbortableHTTPAdapter
    }
    assert threads == [f"{gcs_transfer.GCS_TRANSFER_THREAD_PREFIX}{target.name}"]
