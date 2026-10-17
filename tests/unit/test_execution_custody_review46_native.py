"""Original bootstrap controls and rejected Codex startup, entirely offline."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.bootstrap.service import BootstrapService, BootstrapState
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    bind_execution_custody,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_execution_custody import Authority
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_execution_custody_review43_native import carries
from tests.unit.test_execution_custody_review45_native import native_process


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["codex/0.1.0", "unparseable-version"])
async def test_rejected_native_version_joins_process_outside_startup_lock(monkeypatch, version):
    client = CodexAppServerClient(binary="offline-no-codex")
    proc = await native_process()

    async def spawn():
        client._proc = proc

    async def initialize(*args, **kwargs):
        return {"userAgent": version}

    monkeypatch.setattr(client, "_spawn", spawn)
    monkeypatch.setattr(client, "_request_unguarded", initialize)
    startup = asyncio.create_task(client.ensure_started())
    try:
        done, _ = await asyncio.wait({startup}, timeout=0.5)
        assert startup in done, "version rejection joined retirement under its own startup lock"
        outcome = await await_owned_task(startup)
        assert isinstance(outcome.error, CodexAppServerError)
        assert version in str(outcome.error) or "version" in str(outcome.error)
        assert proc.returncode is not None, "rejected native process escaped physical retirement"
        assert client._proc is None and not client._initialized
    finally:
        # Old candidate deadlocks: release ONLY this test's original lock so its
        # retained close task can settle; never leave the negative-proof child.
        if not startup.done() and client._start_lock.locked():
            client._start_lock.release()
        await await_owned_task(startup)
        if client._close_task is not None:
            await await_owned_task(client._close_task)
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


async def service_for(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "bootstrap.db"))
    try:
        await db.execute("CREATE TABLE IF NOT EXISTS agent_metadata(agent_id TEXT, key TEXT, value TEXT, updated_at TEXT, PRIMARY KEY(agent_id,key))")
        await db.execute("CREATE TABLE IF NOT EXISTS conversations(agent_id TEXT)")
    except BaseException:
        await db.close()
        raise
    storage = SimpleNamespace(agent_resources=None)
    service = BootstrapService(db, "did:test:bootstrap", "Offline", None, tmp_path, storage)
    return service, db


@pytest.mark.asyncio
async def test_cancelled_startup_lock_waiter_does_not_retire_original_birth(monkeypatch):
    client = CodexAppServerClient(binary="offline-no-codex")
    proc = await native_process()
    entered, release = asyncio.Event(), asyncio.Event()

    async def spawn():
        client._proc = proc

    async def handshake(**kwargs):
        entered.set()
        await release.wait()

    monkeypatch.setattr(client, "_spawn", spawn)
    monkeypatch.setattr(client, "_handshake", handshake)
    owner = asyncio.create_task(client.ensure_started())
    waiter = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        waiter = asyncio.create_task(client.ensure_started())
        await asyncio.sleep(0)
        waiter.cancel()
        done, _ = await asyncio.wait({waiter}, timeout=0.3)
        assert waiter in done, "cancelled non-owner began connection retirement"
        outcome = await await_owned_task(waiter)
        assert isinstance(outcome.error, asyncio.CancelledError)
        assert not client._closing and proc.returncode is None
        release.set()
        assert (await await_owned_task(owner)).error is None
        assert client._initialized
    finally:
        release.set()
        await await_owned_task(owner)
        if waiter is not None:
            await await_owned_task(waiter)
        await client.aclose()
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["authority", "unknown", "cancel_unknown", "group"])
async def test_actual_discovery_caller_never_enters_fallback_on_original_control(tmp_path, monkeypatch, kind):
    service, db = await service_for(tmp_path)
    await service.set_bootstrap_state(BootstrapState.DISCOVERY)
    original = receipt(kind)
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.bootstrap_service = service
    agent.privacy_agent = SimpleNamespace(can_store=lambda category: False)

    async def discovery(*args, **kwargs):
        scope.revoke("lost original physical session during discovery")
        raise original

    monkeypatch.setattr(service, "process_discovery_message", discovery)
    try:
        with bind_execution_custody(Authority()) as scope:
            # Read happened before the original live admission, keeping SQLite
            # out of the hosted PostgreSQL scope. Only the actual caller and
            # filesystem fallback run under this original revoked admission.
            async def known_state():
                return BootstrapState.DISCOVERY
            monkeypatch.setattr(service, "get_bootstrap_state", known_state)
            with pytest.raises(BaseException) as caught:
                await agent._handle_bootstrap("hello")
            assert caught.value is original, "discovery normalized original control into ordinary fallback"
            assert not (tmp_path / "SOUL.md").exists(), "revoked discovery wrote a new personality"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_actual_soul_writer_rechecks_original_custody_after_privacy_lock(tmp_path):
    service, db = await service_for(tmp_path)
    lock = asyncio.Lock()
    await lock.acquire()
    service._privacy_transition_lock = lock
    path = tmp_path / "SOUL.md"
    path.write_text("original personality")
    writer = None
    try:
        with bind_execution_custody(Authority()) as scope:
            writer = asyncio.create_task(service.save_soul_md("unauthorized replacement"))
            await asyncio.sleep(0)
            assert not writer.done()
            scope.revoke("lost original authority while waiting on privacy lock")
            lock.release()
            outcome = await await_owned_task(writer)
            assert isinstance(outcome.error, ExecutionAuthorityError)
            assert path.read_text() == "original personality", "privacy serialization granted replacement authority"
    finally:
        if lock.locked():
            lock.release()
        if writer is not None:
            await await_owned_task(writer)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["authority", "unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("route", ["started", "state", "metadata", "completion", "canonical", "history", "save_history", "user_name", "stale", "promotion"])
async def test_bootstrap_consumers_preserve_original_native_receipt(tmp_path, monkeypatch, kind, route):
    service, db = await service_for(tmp_path)
    original = receipt(kind)
    native_execute = db.execute
    committed = []

    async def fail_read(*args, **kwargs):
        raise original

    async def committed_then_lost_receipt(*args, **kwargs):
        result = await native_execute(*args, **kwargs)
        committed.append(result)
        raise original

    monkeypatch.setattr(db, "fetchall", fail_read)
    monkeypatch.setattr(db, "execute", committed_then_lost_receipt)
    service.storage.agent_resources = SimpleNamespace(get_current=fail_read)
    service.storage.promote_soul_seed = fail_read
    operations = {
        "started": service._get_started_time,
        "state": service.get_bootstrap_state,
        "metadata": lambda: service._get_bootstrap_state_from_metadata_or_node(None),
        "completion": service._has_completion_evidence,
        "canonical": service._has_canonical_soul_resource,
        "history": service.get_discovery_history,
        "save_history": lambda: service._save_discovery_history([{"role": "user", "content": "offline"}]),
        "user_name": lambda: service._save_user_name("offline"),
        "stale": service.mark_stale_bootstrap,
        "promotion": lambda: service._promote_soul_resource("offline", source="offline"),
    }
    try:
        with pytest.raises(BaseException) as caught:
            await operations[route]()
        assert carries(caught.value, original), "bootstrap erased the original receipt/cause"
        if route in {"save_history", "user_name", "stale"}:
            assert len(committed) == 1, "lost first acknowledgement admitted a second write"
    finally:
        await db.close()
