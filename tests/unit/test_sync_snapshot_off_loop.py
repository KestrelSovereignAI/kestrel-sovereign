"""A storage snapshot cannot stall the event loop (#3916).

``sync_snapshot`` ran ``sqlite3.backup()`` and hashed the result on the event
loop. Against a 1.6-2.4 GB agent DB that held the loop past the 2.0 s
distributed Stop lease, which fenced the host's Stop owner and cancelled every
in-flight turn on the host. The snapshot now runs in a worker thread.

Every client here is a local fake: no test reaches Lighthouse, GCS or IPFS.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from kestrel_sovereign.storage.providers import lighthouse_rest
from kestrel_sovereign.storage.sync import targets
from kestrel_sovereign.storage.sync.gcs_target import GCSTarget
from kestrel_sovereign.storage.sync.lighthouse_target import LighthouseTarget
from kestrel_sovereign.storage.sync.sovereign_ipfs_target import SovereignIPFSTarget

_SLOW_BACKUP_SECONDS = 0.5
_MAX_HEARTBEAT_GAP_SECONDS = 0.2
_HEARTBEAT_SECONDS = 0.01
_ROWS = [(index, f"memory {index}") for index in range(50)]


def _lighthouse(tmp_path, monkeypatch, upload_error=None):
    uploads: list[bytes] = []  # the raw SQLite bytes each upload received

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def get_balance(self):
            return {"data": {"dataUsed": "0", "dataLimit": str(10**12)}}

        async def upload_car(self, *, body, tag, filename):
            if upload_error is not None:
                raise upload_error
            car = b""
            while chunk := body.read(lighthouse_rest.UPLOAD_CHUNK_BYTES):
                car += chunk
            uploads.append(target._extract_snapshot_content(car))
            return {"Hash": "QmSnapshot", "Size": str(body.length)}

        async def upload(self, *, content, filename, tag):
            return {"Hash": "QmManifest"}

        async def close(self):
            pass

    monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", Client)
    target = LighthouseTarget(api_key="unit-test-key", agent_id="agent", state_dir=tmp_path)
    return target, uploads


def _gcs(tmp_path, monkeypatch, upload_error=None):
    uploads: list[bytes] = []  # the raw SQLite bytes each upload received

    class Blob:
        def __init__(self, name, chunk_size=None):
            self.name = name

        def upload_from_string(self, content, **kwargs):
            if upload_error is not None:
                raise upload_error
            uploads.append(content)

        def rewrite(self, source, **kwargs):
            return None, 0, 0

    bucket = SimpleNamespace(blob=Blob)
    target = GCSTarget("unit-test-bucket", agent_id="agent", state_dir=tmp_path)
    monkeypatch.setattr(target, "_get_bucket", lambda: bucket)
    return target, uploads


def _ipfs(tmp_path, monkeypatch, upload_error=None):
    uploads: list[bytes] = []  # the raw SQLite bytes each upload received

    class Client:
        async def post(self, url, *, params=None, files=None, timeout=None):
            if upload_error is not None:
                raise upload_error
            uploads.append(files["file"][1])
            return SimpleNamespace(
                raise_for_status=lambda: None, json=lambda: {"Hash": "QmIpfs"}
            )

    target = SovereignIPFSTarget("http://ipfs.invalid", "agent", state_dir=tmp_path)
    target._client = Client()
    return target, uploads


_TARGETS = pytest.mark.parametrize(
    "build", [_lighthouse, _gcs, _ipfs], ids=["lighthouse", "gcs", "sovereign_ipfs"]
)


@pytest.fixture
def snapshot_dir(tmp_path, monkeypatch):
    """Where the snapshot temp files go, so a test can see them all."""
    directory = tmp_path / "snapshots"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


@pytest.fixture
def agent_db(tmp_path):
    """A temp agent DB in WAL mode whose rows are still in the WAL."""
    path = tmp_path / "agent.db"
    keeper = sqlite3.connect(path)
    try:
        keeper.execute("PRAGMA journal_mode=WAL")
        keeper.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, body TEXT)")
        keeper.executemany("INSERT INTO memories VALUES (?, ?)", _ROWS)
        keeper.commit()
        yield path
    finally:
        keeper.close()


def _rows(content: bytes, tmp_path: Path) -> list[tuple]:
    restored = tmp_path / "restored.db"
    restored.write_bytes(content)
    with closing(sqlite3.connect(restored)) as conn:
        rows = conn.execute("SELECT id, body FROM memories ORDER BY id").fetchall()
    restored.unlink()
    return rows


def _connect_with(monkeypatch, connection_class) -> list[int]:
    """Make ``targets`` open its connections as ``connection_class``; return
    the ids of the threads that opened them."""
    threads: list[int] = []

    def connect(database, *args, **kwargs):
        threads.append(threading.get_ident())
        return sqlite3.connect(database, *args, factory=connection_class, **kwargs)

    monkeypatch.setattr(
        targets,
        "sqlite3",
        SimpleNamespace(connect=connect, DatabaseError=sqlite3.DatabaseError),
    )
    return threads


class _SlowBackup(sqlite3.Connection):
    def backup(self, *args, **kwargs):
        time.sleep(_SLOW_BACKUP_SECONDS)
        return super().backup(*args, **kwargs)


async def _largest_heartbeat_gap(operation):
    """Run ``operation`` beside a heartbeat; return its result, the longest
    gap between heartbeats, and how long it took."""
    loop = asyncio.get_running_loop()
    gaps: list[float] = []
    done = asyncio.Event()

    async def heartbeat():
        last = loop.time()
        while not done.is_set():
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            now = loop.time()
            gaps.append(now - last)
            last = now

    beating = asyncio.create_task(heartbeat())
    await asyncio.sleep(_HEARTBEAT_SECONDS * 3)
    started = loop.time()
    try:
        result = await operation
    finally:
        elapsed = loop.time() - started
        done.set()
        await beating
    return result, max(gaps), elapsed


@pytest.mark.asyncio
@_TARGETS
async def test_a_slow_backup_never_stops_the_event_loop(
    build, agent_db, snapshot_dir, tmp_path, monkeypatch
):
    target, uploads = build(tmp_path, monkeypatch)
    threads = _connect_with(monkeypatch, _SlowBackup)

    result, gap, elapsed = await _largest_heartbeat_gap(target.sync_snapshot(agent_db))

    assert result.success is True, result.error
    assert elapsed >= _SLOW_BACKUP_SECONDS
    assert gap < _MAX_HEARTBEAT_GAP_SECONDS, (
        f"the snapshot held the event loop for {gap:.3f} s"
    )
    # Both connections were opened in the worker, not on the loop's thread.
    assert len(threads) == 2
    assert threading.get_ident() not in threads
    (uploaded,) = uploads
    assert _rows(uploaded, tmp_path) == _ROWS
    assert list(snapshot_dir.iterdir()) == []


@pytest.mark.asyncio
@_TARGETS
async def test_an_unchanged_snapshot_is_not_uploaded_again(
    build, agent_db, snapshot_dir, tmp_path, monkeypatch
):
    target, uploads = build(tmp_path, monkeypatch)

    first = await target.sync_snapshot(agent_db)
    second = await target.sync_snapshot(agent_db)

    assert first.success is True, first.error
    assert second.success is True, second.error
    assert second.metadata["skipped"] is True
    assert second.bytes_synced == 0
    assert len(uploads) == 1
    assert list(snapshot_dir.iterdir()) == []


class _FailingBackup(sqlite3.Connection):
    def backup(self, *args, **kwargs):
        raise OSError(28, "No space left on device")


@pytest.mark.asyncio
@_TARGETS
async def test_a_backup_that_fails_in_the_worker_fails_the_sync_and_leaves_no_file(
    build, agent_db, snapshot_dir, tmp_path, monkeypatch
):
    target, uploads = build(tmp_path, monkeypatch)
    _connect_with(monkeypatch, _FailingBackup)

    result = await target.sync_snapshot(agent_db)

    assert result.success is False
    assert result.error == "OSError: [Errno 28] No space left on device"
    assert uploads == []
    assert list(snapshot_dir.iterdir()) == []


@pytest.mark.asyncio
@_TARGETS
async def test_a_failed_upload_leaves_no_snapshot_file(
    build, agent_db, snapshot_dir, tmp_path, monkeypatch
):
    target, _ = build(tmp_path, monkeypatch, upload_error=httpx.ConnectError("refused"))

    result = await target.sync_snapshot(agent_db)

    assert result.success is False
    assert result.error == "ConnectError: refused"
    assert list(snapshot_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_a_cancelled_snapshot_keeps_its_file_until_the_worker_finishes(
    agent_db, snapshot_dir, tmp_path, monkeypatch
):
    """A worker that opens the file after it was removed recreates it, and
    nothing removes that one, so the cancellation waits for the worker."""
    entered, release = threading.Event(), threading.Event()

    class HeldBackup(sqlite3.Connection):
        def backup(self, *args, **kwargs):
            entered.set()
            release.wait(10)
            return super().backup(*args, **kwargs)

    target, uploads = _lighthouse(tmp_path, monkeypatch)
    _connect_with(monkeypatch, HeldBackup)
    task = asyncio.create_task(target.sync_snapshot(agent_db))
    try:
        for _ in range(500):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set(), "the backup never started"
        (snapshot,) = snapshot_dir.iterdir()

        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "the cancellation did not wait for the worker"
        assert snapshot.exists()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert uploads == []
    assert list(snapshot_dir.iterdir()) == []
