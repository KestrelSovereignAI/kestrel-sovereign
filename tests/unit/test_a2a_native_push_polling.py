"""Bounded push polling, terminal retirement and scoped retention over real TLS."""

from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, TaskPushNotificationConfig, TaskState
from kestrel_sdk.signals import Signal, SignalMode

from kestrel_sovereign.storage.db.interface import DatabaseError
from tests.unit.test_a2a_native_push_worker import worker
from tests.unit.test_a2a_native_task_manager import (
    manager as manager,  # noqa: PLC0414
)
from tests.unit.test_a2a_native_task_manager import request as native_request
from tests.utils.native_push import managed_push_service
from tests.utils.outbound_tls import outbound_tls as outbound_tls  # noqa: PLC0414


async def _task(manager, *, complete=False):
    task = await manager.create_task(
        native_request(context_id=uuid4().hex), creator_agent_id="did:creator"
    )
    if complete:
        await manager.update_status(
            task.id,
            TaskState.TASK_STATE_COMPLETED,
            message=new_text_message("done", role=Role.ROLE_AGENT),
            recipient_agent_id=manager.did,
        )
    return task


async def _drain(outbox, runner):
    while await outbox.forward_pending(limit=1000):
        pass
    counts = []
    for _ in range(100):
        counts.append(await runner.run_once())
        if not counts[-1]:
            return counts
    raise AssertionError("bounded test history did not finish delivering")


@pytest_asyncio.fixture
async def completed_push_history(manager, monkeypatch, outbound_tls, request):
    terminal = getattr(request, "param", True)
    async with managed_push_service(
        manager, monkeypatch, pause_delivery=True
    ) as service:
        outbox = service.worker.outbox
        tasks = []
        for _ in range(8):
            task = await _task(manager)
            await outbox.register(
                TaskPushNotificationConfig(
                    tenant=manager.task_store.scope.tenant,
                    task_id=task.id,
                    id="callback",
                    url=outbound_tls.url + "/callback",
                ),
                creator_id="did:creator",
            )
            if terminal:
                await manager.update_status(
                    task.id,
                    TaskState.TASK_STATE_COMPLETED,
                    message=new_text_message("done", role=Role.ROLE_AGENT),
                    recipient_agent_id=manager.did,
                )
            tasks.append(task)
        for _ in range(24):
            await _task(manager, complete=True)
        runner = worker(outbox)
        await _drain(outbox, runner)
        assert len(outbound_tls.posts) >= len(tasks)
        attempts = await manager.task_store.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        )
        assert len(attempts) == len(outbound_tls.posts)
        assert all(row[0] for row in attempts)
        yield SimpleNamespace(
            outbox=outbox, runner=runner, manager=manager, tasks=tasks
        )


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 1000}], indirect=True)
class TestRetainedPushHistory:
    async def test_terminal_task_retires_its_consumer_after_real_https_ack(
        self, completed_push_history
    ):
        p = completed_push_history
        assert await p.outbox.subscriptions() == [], (
            "terminal push subscriptions remained on the resident poll inventory"
        )
        assert await p.outbox.backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_consumers "
            "WHERE agent_id = ? AND active",
            (p.outbox.agent_id,),
        ) == 0
        # Standard configuration reads remain valid until the creator deletes
        # the configuration; terminal consumer retirement is a separate fact.
        for task in p.tasks:
            assert await p.outbox.configurations.get(
                task.id, "callback", creator_id="did:creator"
            ) is not None

    @pytest.mark.parametrize("completed_push_history", [False, True], indirect=True)
    async def test_idle_poll_neither_claims_subscribers_nor_decodes_retained_history(
        self, completed_push_history, monkeypatch
    ):
        p = completed_push_history
        actual_fetch = p.outbox.backend.fetch_all
        actual_claim = p.outbox.signals.claim_delivery
        backfill_rows, claims = [], []

        async def fetch(query, params=()):
            rows = await actual_fetch(query, params)
            if "FROM durable_signal_events" in query and "retention_until >=" in query:
                backfill_rows.extend(rows)
            return rows

        async def claim(**kwargs):
            claims.append(kwargs["consumer_id"])
            return await actual_claim(**kwargs)

        monkeypatch.setattr(p.outbox.backend, "fetch_all", fetch)
        monkeypatch.setattr(p.outbox.signals, "claim_delivery", claim)
        assert await p.runner.run_once() == 0
        assert not claims and not backfill_rows, (
            f"idle poll made {len(claims)} claims and decoded "
            f"{len(backfill_rows)} retained event rows"
        )

    @pytest.mark.parametrize("completed_push_history", [False], indirect=True)
    async def test_later_notification_claim_never_replays_retained_source_history(
        self, completed_push_history, outbound_tls, monkeypatch
    ):
        p = completed_push_history
        actual = p.outbox.backend.fetch_all
        replayed = []

        async def fetch(query, params=()):
            rows = await actual(query, params)
            if "FROM durable_signal_events" in query and "retention_until >=" in query:
                replayed.extend(rows)
            return rows

        monkeypatch.setattr(p.outbox.backend, "fetch_all", fetch)
        before = len(outbound_tls.posts)
        await p.manager.update_status(
            p.tasks[0].id,
            TaskState.TASK_STATE_WORKING,
            recipient_agent_id=p.manager.did,
        )
        assert await p.runner.run_once() == 1
        assert len(outbound_tls.posts) == before + 1
        assert not replayed, "a later native claim decoded retained source history"

    async def test_resident_retention_purges_only_its_private_source_and_journal(
        self, completed_push_history, monkeypatch
    ):
        p = completed_push_history
        before = await p.outbox.backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_events WHERE agent_id = ?",
            (p.outbox.agent_id,),
        )
        assert before > 0
        await p.outbox.signals.persist_signal(
            Signal(
                source="other.test",
                kind="retained",
                mode=SignalMode.ACTION,
                payload={},
                target_agent="did:other",
            ),
            agent_id="did:other",
            source_event_id="other-retained-event",
            retention_days=1,
        )
        future = p.outbox.signals.now_utc() + timedelta(days=31)
        monkeypatch.setattr(p.outbox.signals, "now_utc", lambda: future)
        assert await p.runner.run_once() == 0
        remaining = await p.outbox.backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_events WHERE agent_id = ?",
            (p.outbox.agent_id,),
        )
        assert remaining == max(0, before - p.runner.batch_size), (
            "the resident did not retire exactly one bounded source batch"
        )
        while remaining:
            assert await p.runner.run_once() == 0
            following = await p.outbox.backend.fetch_val(
                "SELECT COUNT(*) FROM durable_signal_events WHERE agent_id = ?",
                (p.outbox.agent_id,),
            )
            assert following < remaining, "push source retention made no progress"
            remaining = following
        assert await p.outbox.backend.fetch_val(
            "SELECT COUNT(*) FROM a2a_task_changes"
        ) == 0, "forwarded journal history outlived its retired durable events"
        assert await p.outbox.backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_events WHERE agent_id = 'did:other'"
        ) == 1

    async def test_original_source_and_reference_deletions_roll_back_together(
        self, completed_push_history, monkeypatch
    ):
        p = completed_push_history
        backend = p.outbox.backend
        event_query = (
            "SELECT event_id FROM durable_signal_events WHERE agent_id = ? ORDER BY event_id"
        )
        events = await backend.fetch_all(event_query, (p.outbox.agent_id,))
        journal_query = "SELECT sequence, forwarded_event_id FROM a2a_task_changes ORDER BY sequence"
        references = await backend.fetch_all(journal_query)
        assert events and references
        original_retire = p.outbox.changes.retire_forwarded
        original_error = ValueError("original journal retirement acknowledgement refused")
        retired = []

        async def refuse_original_retirement(ids):
            retired.append(await original_retire(ids))
            raise original_error

        future = p.outbox.signals.now_utc() + timedelta(days=31)
        monkeypatch.setattr(p.outbox.signals, "now_utc", lambda: future)
        monkeypatch.setattr(p.outbox.changes, "retire_forwarded", refuse_original_retirement)
        with pytest.raises((DatabaseError, ValueError)) as caught:
            await p.runner.run_once()
        assert caught.value is original_error or caught.value.__cause__ is original_error
        assert retired == [min(p.runner.batch_size, len(events))]
        assert await backend.fetch_all(event_query, (p.outbox.agent_id,)) == events
        assert await backend.fetch_all(journal_query) == references


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 1000}], indirect=True)
async def test_poll_delivery_work_is_limited_by_the_configured_batch(
    manager, monkeypatch, outbound_tls
):
    async with managed_push_service(
        manager, monkeypatch, pause_delivery=True
    ) as service:
        outbox = service.worker.outbox
        for _ in range(7):
            task = await _task(manager)
            await outbox.register(
                TaskPushNotificationConfig(
                    tenant=manager.task_store.scope.tenant,
                    task_id=task.id,
                    id="callback",
                    url=outbound_tls.url + "/callback",
                ),
                creator_id="did:creator",
            )
        runner = worker(outbox)
        runner.batch_size = 3
        counts = await _drain(outbox, runner)
        assert len(outbound_tls.posts) == 7
        assert all(count <= runner.batch_size for count in counts), (
            f"one push iteration exceeded its configured batch: {counts}"
        )


@pytest.mark.parametrize("push_enabled", [False, True])
async def test_unsubscribed_task_mutations_do_not_accumulate_push_history(
    manager, monkeypatch, push_enabled
):
    from contextlib import nullcontext

    context = (
        managed_push_service(manager, monkeypatch, pause_delivery=True)
        if push_enabled
        else nullcontext()
    )
    async with context:
        for _ in range(24):
            await _task(manager, complete=True)
        assert await manager.task_store.backend.fetch_val(
            "SELECT COUNT(*) FROM a2a_task_changes"
        ) == 0, "ordinary task mutations accumulated unused push journal history"
