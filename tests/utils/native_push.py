"""Real managed native push storage with an explicitly paused delivery producer."""

import asyncio
import base64
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import timedelta

from kestrel_sovereign._async_ownership import create_owned_task
from kestrel_sovereign.a2a.push_outbox import NativePushOutbox
from kestrel_sovereign.a2a.push_service import NativePushService
from kestrel_sovereign.a2a.push_transport import NativePushTransport
from kestrel_sovereign.a2a.push_worker import NativePushWorker
from kestrel_sovereign.a2a.stores.unified.push_config_store import PushConfigStore
from kestrel_sovereign.signals.durable import DurableSignalStore


def configure_push_test_key(monkeypatch):
    """Use the real shared key hierarchy and actual AEAD in owned test storage."""
    monkeypatch.setenv(
        "KESTREL_DATA_KEY", base64.urlsafe_b64encode(os.urandom(32)).decode()
    )
    monkeypatch.delenv("KESTREL_DATA_KEY_FILE", raising=False)


def intercept_push_commit_ack(
    outbox, monkeypatch, after_commit, *, owner=None, method="forward_pending"
):
    """Observe the real top-level native COMMIT before its ACK returns.

    Initialization and savepoint releases are excluded. The native database
    producer always runs first; the observer can then retain or fail its ACK.
    """
    forwarding = ContextVar("native-push-test-forward-commit", default=False)
    owner = outbox if owner is None else owner
    original_operation = getattr(owner, method)

    async def forwarding_scope(*args, **kwargs):
        token = forwarding.set(True)
        try:
            return await original_operation(*args, **kwargs)
        finally:
            forwarding.reset(token)

    monkeypatch.setattr(owner, method, forwarding_scope)
    if outbox.backend.backend_type == "sqlite":
        connection = outbox.backend._connection
        original_commit = connection.commit

        async def commit_ack():
            await original_commit()
            if forwarding.get():
                await after_commit()

        monkeypatch.setattr(connection, "commit", commit_ack)
    elif outbox.backend.backend_type == "postgres":
        from asyncpg.transaction import Transaction

        original_exit = Transaction.__aexit__

        async def commit_ack(transaction, exc_type, exc, tb):
            result = await original_exit(transaction, exc_type, exc, tb)
            if forwarding.get() and exc_type is None and not transaction._nested:
                await after_commit()
            return result

        monkeypatch.setattr(Transaction, "__aexit__", commit_ack)
    else:
        raise ValueError("Native push commit probes require SQLite or PostgreSQL")


@asynccontextmanager
async def managed_push_service(manager, monkeypatch, *, pause_delivery: bool):
    """Hold the original runner before claims while admission tests inspect commits.

    The real resident task, primary stores and consumer ledger stay live. This
    explicit producer barrier prevents network effects without a timing oracle;
    transport/worker/service tests separately drive actual HTTPS delivery.
    """
    configure_push_test_key(monkeypatch)
    signals = DurableSignalStore(manager.task_store.backend)
    await signals.initialize()
    outbox = NativePushOutbox(
        PushConfigStore(manager.task_store),
        signals,
        retention_days=30,
        lease_seconds=60,
    )
    worker = NativePushWorker(
        outbox,
        NativePushTransport(outbox.configurations, timeout=timedelta(seconds=5)),
        poll_interval=timedelta(seconds=1),
        renewal_interval=timedelta(seconds=20),
        retry_delay=timedelta(seconds=1),
        batch_size=10,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original_iteration = worker.run_once

    async def paused_iteration():
        entered.set()
        await release.wait()
        return await original_iteration()

    if pause_delivery:
        monkeypatch.setattr(worker, "run_once", paused_iteration)
    service = NativePushService(worker, task_factory=create_owned_task)
    try:
        await service.start()
        manager.bind_push_service(service)
        if pause_delivery:
            await entered.wait()
        yield service
    finally:
        # Keep the producer at its barrier while its actual resident owner
        # handles cancellation and physically settles.
        await service.close()
