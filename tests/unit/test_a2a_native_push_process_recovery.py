"""SIGKILL of a real native HTTPS owner must have a durable recovery boundary."""

import asyncio
import json
import os
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

import kestrel_sovereign
from kestrel_sovereign.a2a.push_outbox import NativePushOutbox
from kestrel_sovereign.a2a.push_worker import PushRetirementPending
from kestrel_sovereign.a2a.stores.unified.push_config_store import (
    PushConfigConflict,
    PushConfigStore,
)
from kestrel_sovereign.multi_agent.liveness import (
    ProcessIdentityUnavailable,
    recorded_process_exited,
)
from kestrel_sovereign.signals.durable import DurableSignalStore
from tests.unit.test_a2a_native_push_config_store import config as stored_config
from tests.unit.test_a2a_native_push_outbox import callback
from tests.unit.test_a2a_native_push_outbox import (
    outbox as outbox,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_push_outbox import (
    push_store as push_store,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_push_worker import worker
from tests.unit.test_a2a_native_task_manager import (
    manager as manager,  # noqa: PLC0414 - fixture dependency
)
from tests.utils.native_push import configure_push_test_key
from tests.utils.outbound_tls import (
    outbound_tls as outbound_tls,  # noqa: PLC0414 - fixture dependency
)


@pytest.mark.parametrize("field", ["kernel_session", "birth_id"])
async def test_copied_and_foreign_process_witnesses_cannot_spend_an_original_attempt(outbox, field):
    value, task = outbox
    store = value.configurations
    configuration = await value.register(callback(task), creator_id="did:creator")
    original = store.prepare_delivery(configuration, executor_id="original-runtime")
    copied = original.owner_identity
    if field == "birth_id":
        platform, number = copied[field].split(":")
        copied[field] = f"{platform}:{int(number) + 1}"
    else:
        copied[field] = "foreign-kernel"
    assert original.owner_identity[field] != copied[field]
    foreign = replace(original, _owner_identity_bytes=json.dumps(copied).encode())
    assert not await store.admit_delivery(foreign)
    assert await value.backend.fetch_all("SELECT attempt_id FROM a2a_push_delivery_attempts") == []
    assert await store.admit_delivery(original)
    assert not await store.delivery_is_current(foreign)
    assert not await store.retire_delivery(foreign)
    assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(0,)]
    assert await store.delivery_is_current(original)
    assert await store.retire_delivery(original)


@pytest.mark.parametrize("shift_before_admission", [False, True])
async def test_original_process_ownership_survives_a_diagnostic_clock_change(
    outbox, monkeypatch, shift_before_admission
):
    import kestrel_sovereign.a2a.stores.unified.push_config_store as module

    value, task = outbox
    config = await value.register(callback(task), creator_id="did:creator")
    store = value.configurations
    attempt = store.prepare_delivery(config, executor_id="original-runtime")
    if not shift_before_admission:
        assert await store.admit_delivery(attempt)
    original_identity = module.process_identity

    def clock_adjusted(**kwargs):
        identity = original_identity(**kwargs)
        identity["started_at"] += 3600
        return identity

    monkeypatch.setattr(module, "process_identity", clock_adjusted)
    if shift_before_admission:
        assert await store.admit_delivery(attempt)
    assert await store.delivery_is_current(attempt)
    assert await store.retire_delivery(attempt)
    assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(1,)]


async def test_push_startup_requires_physical_owner_identity_before_optional_ddl(manager, monkeypatch):
    import kestrel_sovereign.a2a.stores.unified.push_config_store as module

    configure_push_test_key(monkeypatch)
    backend = manager.task_store.backend
    signals = DurableSignalStore(backend)
    await signals.initialize()
    outbox = NativePushOutbox(PushConfigStore(manager.task_store), signals, retention_days=30, lease_seconds=60)
    original = ProcessIdentityUnavailable("original physical owner identity was refused")

    def unavailable(**kwargs):
        raise original

    monkeypatch.setattr(module, "process_identity", unavailable)
    with pytest.raises(ProcessIdentityUnavailable) as caught:
        await outbox.initialize()
    assert caught.value is original
    for table in ("a2a_push_configs", "a2a_push_delivery_attempts", "a2a_push_subscriptions"):
        assert not await backend.table_exists(table)
    assert await backend.fetch_val("SELECT COUNT(*) FROM a2a_native_store_schema WHERE component = 'push'") == 0


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
async def test_invalid_process_recovery_batch_cannot_consume_original_attempts(outbox, limit):
    value, task = outbox
    config = await value.register(callback(task), creator_id="did:creator")
    store = value.configurations
    attempt = store.prepare_delivery(config, executor_id="original-runtime")
    assert await store.admit_delivery(attempt)
    with pytest.raises(ValueError, match="positive"):
        await store.recover_exited_attempts(limit=limit)
    assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(0,)]


async def test_malformed_original_process_witness_refuses_recovery(outbox):
    value, task = outbox
    config = await value.register(callback(task), creator_id="did:creator")
    store = value.configurations
    attempt = store.prepare_delivery(config, executor_id="original-runtime")
    assert await store.admit_delivery(attempt)
    await value.backend.execute("UPDATE a2a_push_delivery_attempts SET owner_identity = '{'")
    with pytest.raises(PushConfigConflict, match="witness is invalid"):
        await store.recover_exited_attempts(limit=1)
    assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(0,)]


async def test_process_inventory_uses_the_live_attempt_partial_index(outbox, monkeypatch):
    value, _ = outbox
    store = value.configurations
    if store.is_postgres:
        rows = await value.backend.fetch_all(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() "
            "AND indexname = 'a2a_push_unretired_processes'"
        )
        assert len(rows) == 1 and "WHERE (retired = 0)" in rows[0][0]
        return
    original = value.backend.fetch_all
    queries = []

    async def fetch(query, parameters=()):
        result = await original(query, parameters)
        queries.append((query, parameters))
        return result

    monkeypatch.setattr(value.backend, "fetch_all", fetch)
    assert await store.recover_exited_attempts(limit=1) == 0
    query, parameters = queries[-1]
    plan = await original("EXPLAIN QUERY PLAN " + query, parameters)
    assert any("SEARCH" in row[-1] and "_unretired_processes" in row[-1] for row in plan), plan


@asynccontextmanager
async def native_push_child(value, outbound_tls, tmp_path, **request):
    """Own one isolated native sender/observer process and its control pipe."""
    root = Path(kestrel_sovereign.__file__).resolve().parents[1]
    home = tmp_path / "child-home"
    home.mkdir()
    environment = {
        "PATH": os.environ["PATH"], "HOME": str(home), "PYTHONPATH": str(root),
        "PYTHONDONTWRITEBYTECODE": "1", "KESTREL_HOME": str(home),
        "SSL_CERT_FILE": os.environ["SSL_CERT_FILE"],
        "KESTREL_DATA_KEY": os.environ["KESTREL_DATA_KEY"],
    }
    payload = {
        "backend_type": value.backend.backend_type,
        "connection": value.backend.db_path if value.is_sqlite else value.backend._dsn,
        "recipient_id": value.configurations.scope.recipient_id,
        "tenant": value.configurations.scope.tenant,
        "hostname": outbound_tls.host, "port": outbound_tls.port,
    }
    payload.update(request)
    log = tmp_path / "child-stderr.log"
    process = None
    with log.open("wb") as stderr:
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-B", "-m", "tests.utils.native_push_process",
                cwd=tmp_path, env=environment, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=stderr,
            )
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()
            process.stdin.close()
            line = await asyncio.wait_for(process.stdout.readline(), 15)
            assert line, f"native child ended before its real response close: {log.read_text()}"
            report = json.loads(line)
            yield process, report, log
        finally:
            if process is not None:
                if process.returncode is None:
                    process.kill()
                await process.wait()


async def test_a_different_real_process_cannot_monitor_or_retire_the_original_attempt(
    outbox, outbound_tls, tmp_path
):
    value, task = outbox
    config = await value.register(callback(task), creator_id="did:creator")
    attempt = value.configurations.prepare_delivery(config, executor_id="original-runtime")
    assert await value.configurations.admit_delivery(attempt)
    fields = {**vars(attempt), "_owner_identity_bytes": attempt._owner_identity_bytes.decode("utf-8")}
    async with native_push_child(
        value, outbound_tls, tmp_path, operation="inspect-original", attempt=fields
    ) as (process, report, log):
        assert report["event"] == "foreign-process-observation"
        assert report["owner"]["pid"] == process.pid != attempt.owner_identity["pid"]
        assert report["owner"]["birth_id"] != attempt.owner_identity["birth_id"]
        assert report["current"] is False and report["retired"] is False
        assert await asyncio.wait_for(process.wait(), 5) == 0, log.read_text()
    assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(0,)]
    assert not outbound_tls.posts
    assert await value.configurations.retire_delivery(attempt)


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 4}], indirect=True)
async def test_resident_observes_an_exited_http_owner_without_replaying_or_deleting_its_grant(
    outbox, outbound_tls, tmp_path
):
    value, task = outbox
    if os.name != "posix":
        pytest.skip("Real POSIX SIGKILL owner")
    native = callback(task)
    native.url = outbound_tls.url + "/callback"
    config = await value.register(native, creator_id="did:creator")
    async with native_push_child(value, outbound_tls, tmp_path) as (process, report, log):
        assert report["event"] == "native-response-close"
        process.kill()
        assert await process.wait() == -signal.SIGKILL
        assert recorded_process_exited(log, report["owner"], require_kernel_session=True)
        runner = worker(value)
        assert await runner.run_once() == 0
        assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(1,)]
        assert await value.configurations.get(task.id, config.config.id, creator_id="did:creator") == config
        assert len(outbound_tls.posts) == 1
        await runner.delete(task.id, config.config.id, creator_id="did:creator")


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 4}], indirect=True)
async def test_bounded_recovery_rotates_past_a_live_owner_and_keeps_foreign_scopes_out(
    outbox, outbound_tls, tmp_path, monkeypatch
):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore

    value, task = outbox
    if os.name != "posix":
        pytest.skip("Real POSIX SIGKILL owner with independently opened primary storage")
    native = callback(task)
    native.url = outbound_tls.url + "/callback"
    config = await value.register(native, creator_id="did:creator")
    store = value.configurations
    live_config = await store.create(stored_config(task, "live-config"), creator_id="did:creator")
    live = replace(store.prepare_delivery(live_config, executor_id="live-runtime"), attempt_id="00-live")
    assert await store.admit_delivery(live)
    async with native_push_child(value, outbound_tls, tmp_path) as (process, report, log):
        assert report["event"] == "native-response-close"
        dead_id = await value.backend.fetch_val(
            "SELECT attempt_id FROM a2a_push_delivery_attempts WHERE generation = ?", (config.generation,)
        )
        assert dead_id > live.attempt_id
        process.kill()
        assert await process.wait() == -signal.SIGKILL
        assert recorded_process_exited(log, report["owner"], require_kernel_session=True)
        for scope in (
            NativeRecipientScope(store.scope.recipient_id, "foreign-tenant"),
            NativeRecipientScope("did:foreign-recipient", store.scope.tenant),
        ):
            foreign = PushConfigStore(TaskStore(value.backend, scope=scope, owns_backend=False))
            assert await foreign.recover_exited_attempts(limit=2) == 0
        assert await value.backend.fetch_val(
            "SELECT retired FROM a2a_push_delivery_attempts WHERE attempt_id = ?", (dead_id,)
        ) == 0
        observed = []
        original_observe = store._retire_exited_attempts

        async def observe(rows):
            observed.extend(row[0] for row in rows)
            return await original_observe(rows)

        monkeypatch.setattr(store, "_retire_exited_attempts", observe)
        assert await store.recover_exited_attempts(limit=1) == 0
        assert await value.backend.fetch_val(
            "SELECT retired FROM a2a_push_delivery_attempts WHERE attempt_id = ?", (dead_id,)
        ) == 0
        assert await store.recover_exited_attempts(limit=1) == 1
        observed.clear()
        assert await store.recover_exited_attempts(limit=1) == 0
        assert observed == [live.attempt_id], "bounded inventory stopped observing live owners after its first pass"
        assert await value.backend.fetch_val(
            "SELECT retired FROM a2a_push_delivery_attempts WHERE attempt_id = ?", (live.attempt_id,)
        ) == 0
        assert len(outbound_tls.posts) == 1
    await worker(value).delete(task.id, config.config.id, creator_id="did:creator")
    assert await store.get(task.id, "live-config", creator_id="did:creator") == live_config
    assert await store.retire_delivery(live)
    assert await store.begin_delete(task.id, "live-config", creator_id="did:creator") == (live_config,)
    assert await store.finish_delete(live_config)


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 4}], indirect=True)
@pytest.mark.parametrize("check_live_owner", [False, True])
@pytest.mark.parametrize("foreign_field", [None, "kernel_session", "host", "pid_namespace"])
async def test_killed_native_https_owner_can_finish_delete_without_a_clock_receipt(
    outbox, outbound_tls, tmp_path, check_live_owner, foreign_field
):
    value, task = outbox
    if os.name != "posix":
        pytest.skip("Real POSIX SIGKILL with independently opened primary storage")
    native = callback(task)
    native.url = outbound_tls.url + "/callback"
    config = await value.register(native, creator_id="did:creator")
    async with native_push_child(value, outbound_tls, tmp_path) as (process, report, log):
        assert report["event"] == "native-response-close"
        assert report["owner"]["pid"] == process.pid
        assert len(outbound_tls.posts) == 1
        assert process.returncode is None
        assert not recorded_process_exited(log, report["owner"])
        assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(0,)]
        assert json.loads(await value.backend.fetch_val("SELECT owner_identity FROM a2a_push_delivery_attempts")) == report["owner"]
        runner = worker(value)
        if check_live_owner:
            with pytest.raises(PushRetirementPending):
                await runner.delete(task.id, config.config.id, creator_id="did:creator")
            assert await value.backend.fetch_val("SELECT ciphertext IS NOT NULL FROM a2a_push_configs")
            assert process.returncode is None
        process.kill()
        assert await process.wait() == -signal.SIGKILL
        assert recorded_process_exited(log, report["owner"])
        if foreign_field is not None:
            original_owner = await value.backend.fetch_val("SELECT owner_identity FROM a2a_push_delivery_attempts")
            foreign_owner = {**report["owner"], foreign_field: "foreign-original-owner"}
            await value.backend.execute("UPDATE a2a_push_delivery_attempts SET owner_identity = ?", (json.dumps(foreign_owner),))
            assert await value.configurations.recover_exited_attempts(limit=1) == 0
            with pytest.raises(PushRetirementPending):
                await runner.delete(task.id, config.config.id, creator_id="did:creator")
            assert await value.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(0,)]
            # Restore the exact original fixture witness, never a new
            # receipt inferred from the foreign process's absence here.
            await value.backend.execute("UPDATE a2a_push_delivery_attempts SET owner_identity = ?", (original_owner,))
        await runner.delete(task.id, config.config.id, creator_id="did:creator")
        assert await value.backend.fetch_all("SELECT state, ciphertext FROM a2a_push_configs") == [("deleted", None)]
        assert await value.backend.fetch_all("SELECT attempt_id FROM a2a_push_delivery_attempts") == []
        assert len(outbound_tls.posts) == 1, "process recovery replayed a physical callback"
        assert await value.configurations.task_store.delete(task.id)
