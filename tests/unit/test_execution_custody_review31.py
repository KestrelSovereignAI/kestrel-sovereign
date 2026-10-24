"""Real proxy/native SQLite effect beyond unmodified ACK ownership deadlines."""
import asyncio
import gc
from types import SimpleNamespace
import weakref
import threading

import pytest
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError, _explicit_control_chain, execution_commit_outcome
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from tests.utils.native_a2a import manager_for
from kestrel_sovereign.features.base import Feature
from tests.unit.test_execution_custody_proxy_ingress import owner
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review30 import carrier, service_owner
from tests.utils.original_failure import OriginalFailure

@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["ack", "retry"])
@pytest.mark.parametrize("kind", ["error", "unknown", "cancel", "cancel_unknown", "group", "ordinary"])
async def test_late_native_effect_receipt_survives_facade_retirement(monkeypatch, tmp_path, completion, kind):
    error = OSError("ordinary late RPC failure") if kind == "ordinary" else carrier(kind)
    agent, db, feature, client, event, _, release = await owner(monkeypatch, tmp_path, error)
    effect_done, cancelled, completed, late_settled = (asyncio.Event() for _ in range(4))
    task = operation = None
    async def late_call(name, payload):
        await db.execute("INSERT INTO effects VALUES (1)")
        effect_done.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        raise error
    client.call_host_ingress = late_call
    try:
        data = event["payload"]
        request = (feature._split_inbound_event_acknowledgement(data)[1]
                   if completion == "ack" else feature._inbound_event_retry_completion(data))
        feature._schedule_event_ingress_completion(client, request,
            expected_state="acknowledged" if completion == "ack" else "retrying", kind=completion)
        (task,) = tuple(feature._event_ack_tasks)
        task.add_done_callback(lambda done: completed.set())
        await asyncio.wait_for(effect_done.wait(), 5)
        await asyncio.wait_for(cancelled.wait(), 5)
        # The unchanged production3s ACK deadline plus1s cancellation grace
        # fences this exact source without pretending its RPC has settled.
        await asyncio.wait_for(completed.wait(), 5)
        (ownership,) = tuple(feature._terminal_lifecycle_tasks)
        operation = ownership.task
        operation.add_done_callback(lambda done: late_settled.set())
        assert not operation.done()
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        release.set()
        await asyncio.wait_for(late_settled.wait(), 5)
        assert not feature._terminal_lifecycle_tasks
        if kind == "ordinary":
            await feature.shutdown()
        else:
            with pytest.raises(BaseException) as caught:
                await feature.shutdown()
            assert caught.value is error
        assert client.stopped
        assert feature._client is None
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        if operation is not None:
            await await_owned_task(operation._host_task)
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["agent", "feature", "runtime", "feature-runtime", "a2a"])
@pytest.mark.parametrize("finite", [False, True])
async def test_completed_native_background_owners_retain_one_strongest_receipt(tmp_path, route, finite, caplog):
    import logging

    caplog.set_level(logging.CRITICAL, logger="kestrel_sovereign.a2a.task_manager")
    agent, db = await native_owner(tmp_path)
    from kestrel_sovereign.execution_custody import _explicit_control_chain

    # Declared finite lifecycle control alongside actual SQLite worker effects;
    # the integration matrix separately exercises actual native close errors.
    lifecycle = carrier("error") if finite else None
    if lifecycle is not None:
        agent._remember_shutdown_control(lifecycle)

    class Native(Feature):
        async def initialize(self):
            pass

        @property
        def tool_description(self):
            return "bounded native owner"

    class Payload:
        pass

    feature = Native(agent)
    manager = manager_for(db.backend, NativeRecipientScope(agent.did), owns_backend=False)
    payloads, strongest, strongest_source = [], None, None
    try:
        for index in range(40):
            error = ExecutionCommitOutcomeError("unknown") if index >= 20 else carrier("error")
            error.retained_payload = Payload()
            payloads.append(weakref.ref(error.retained_payload))
            if index == 20:
                strongest = error
                strongest_source = OriginalFailure.capture(error)

            async def effect(index=index, error=error):
                await db.execute("INSERT INTO effects VALUES (?)", (index,))
                raise error

            completed = asyncio.Event()
            if route == "agent":
                task = agent._track_background_task(effect(), name="native-bounded-receipt")
            elif route == "feature":
                task = feature._track_owned_background_task(effect(), name="native-bounded-receipt")
            elif route == "runtime":
                task = agent._track_runtime_task(effect(), name="native-bounded-receipt")
            elif route == "feature-runtime":
                task = feature._track_owned_runtime_task(effect(), name="native-bounded-receipt")
            else:
                task = manager._track_execution_task(
                    effect(), task_id=str(index), authority_agent_id=agent.did,
                )
            task.add_done_callback(lambda done: completed.set())
            await asyncio.wait_for(completed.wait(), 5)
            del task, error, effect
        # Captured LogRecords hold exception tracebacks independently of the owner.
        caplog.clear()
        gc.collect()
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 40
        assert not agent._background_tasks
        assert not getattr(feature, "_owned_background_tasks", ())
        assert not manager._execution_tasks
        assert sum(reference() is not None for reference in payloads) == 1
        if finite:
            # Re-publication of the same finite inventory is not new evidence
            # and must not grow a wrapper history around the bounded worker.
            size = len(tuple(_explicit_control_chain(agent._shutdown_control)))
            for _ in range(20):
                agent._remember_shutdown_control(agent._shutdown_control)
            assert len(tuple(_explicit_control_chain(agent._shutdown_control))) == size
        if route == "a2a":
            assert manager._execution_control is strongest
            with pytest.raises(BaseException) as caught:
                await manager.drain_execution_tasks()
        elif route in {"feature", "feature-runtime"}:
            assert feature._owned_background_control is strongest
            with pytest.raises(BaseException) as caught:
                await feature._cancel_owned_background_tasks()
        else:
            chain = tuple(_explicit_control_chain(agent._shutdown_control))
            assert strongest in chain
            if finite:
                assert lifecycle in chain
            else:
                strongest_source.assert_retained(agent._shutdown_control)
                assert execution_commit_outcome(agent._shutdown_control) == "unknown"
            with pytest.raises(BaseException) as caught:
                agent._raise_shutdown_control()
        assert strongest in tuple(_explicit_control_chain(caught.value))
    finally:
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["preference", "adapter", "client"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown"])
@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("observer_delay", [0, 0.05])
async def test_llm_bounded_handoff_retains_late_native_control_and_later_cleanup(
    tmp_path, monkeypatch, stage, kind, interrupt, observer_delay,
):
    from kestrel_sovereign.llm import service as service_module

    agent, db = await native_owner(tmp_path)
    service = service_owner()
    entered, release, settled = (asyncio.Event() for _ in range(3))
    cancel_delivered, retry_entered, retry_release = (asyncio.Event() for _ in range(3))
    failure, calls = carrier(kind), []
    first_close = retry_close = writer = None
    withdrawals = []
    actual_wait = service_module.await_owned_task

    async def observed_wait(task, *args, **kwargs):
        outcome = await actual_wait(task, *args, **kwargs)
        if task is writer and outcome.cancellation is not None:
            withdrawals.append(outcome.cancellation)
        return outcome

    monkeypatch.setattr(service_module, "await_owned_task", observed_wait)

    async def stubborn():
        calls.append(stage)
        await db.execute("INSERT INTO effects VALUES (1)")
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        raise failure

    async def later():
        calls.append("later")

    async def usage():
        calls.append("usage")

    service.close_usage_db = usage
    if stage == "preference":
        writer = asyncio.create_task(stubborn())
        writer.add_done_callback(service._handle_preference_persistence_done)
        service._preference_persistence_tasks.add(writer)
        service.providers = [{"client": SimpleNamespace(close=later)}]
        target_key = "preference-writers"
    else:
        resource = SimpleNamespace(close=stubborn, aclose=stubborn)
        service.providers = [{stage: resource}, {"client": SimpleNamespace(close=later)}]
        target_key = (stage, id(resource))

    original_wait = asyncio.wait

    async def wait_after_native_setup(tasks, *, timeout):
        if timeout == 0.02 and any(
            task is vars(service).get("_llm_close_tasks", {}).get(target_key)
            for task in tasks
        ):
            await asyncio.wait_for(entered.wait(), 5)
            current = asyncio.current_task()
            if current is first_close and interrupt and not cancel_delivered.is_set():
                # Withdraw while the original bounded observation is live,
                # not after a delayed controller finds it already completed.
                cancel_delivered.set()
                assert first_close.cancel("actual bounded-close observer withdrawal")
            if current is retry_close and not retry_entered.is_set():
                # The retry must observe the SAME owner's late receipt. Its
                # earlier pending timeout is not evidence about a later error.
                retry_entered.set()
                await retry_release.wait()
        return await original_wait(tasks, timeout=timeout)

    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.02)
    monkeypatch.setattr(asyncio, "wait", wait_after_native_setup)
    try:
        first_close = asyncio.create_task(service.close())
        if interrupt:
            await asyncio.sleep(observer_delay)
            await asyncio.wait_for(cancel_delivered.wait(), 5)
        done, _ = await original_wait({first_close}, timeout=1.5)
        assert done, "a stubborn owner exceeded bounded LLM cleanup"
        if interrupt:
            with pytest.raises(asyncio.CancelledError):
                await first_close
        else:
            with pytest.raises(RuntimeError, match="retained cleanup owners"):
                await first_close
        assert "later" in calls
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        pending = tuple(task for task in service._llm_close_tasks.values() if task is not None)
        assert pending
        if stage == "preference":
            assert "usage" not in calls, "storage retired before its native writer settled"
        retry_close = asyncio.create_task(service.close())
        await asyncio.wait_for(retry_entered.wait(), 5)
        await asyncio.sleep(observer_delay)
        assert not retry_close.done()
        assert calls.count(stage) == 1, "retry replaced the actual unfinished operation"
        for task in pending:
            task.add_done_callback(lambda done: settled.set())
        release.set()
        await asyncio.wait_for(settled.wait(), 5)
        for task in pending:
            await await_owned_task(task)
        retry_release.set()
        with pytest.raises(BaseException) as caught:
            await retry_close
        chain = tuple(_explicit_control_chain(caught.value))
        assert failure in chain and execution_commit_outcome(caught.value) == "unknown"
        assert all(error in chain for error in withdrawals)
        with pytest.raises(BaseException) as caught:
            await service.close()
        chain = tuple(_explicit_control_chain(caught.value))
        assert failure in chain and execution_commit_outcome(caught.value) == "unknown"
        assert all(error in chain for error in withdrawals)
        if stage == "preference":
            assert service._preference_persistence_control is failure
            assert all(error not in tuple(_explicit_control_chain(failure)) for error in withdrawals)
            # A writer receipt is independent of physical writer settlement.
            assert not any(task is not None for task in service._llm_close_tasks.values())
        else:
            # A failed native close is not a retirement receipt. Its original
            # owner remains sticky; another close may not repeat the effect.
            retained = tuple(task for task in service._llm_close_tasks.values() if task is not None)
            assert retained == pending
            assert all(task.done() for task in retained)
            assert service.has_pending_close_owners()
        assert calls.count(stage) == 1
        assert calls.count("later") == 1
        assert calls.count("usage") == 1
    finally:
        release.set()
        retry_release.set()
        for task in (first_close, retry_close, writer):
            if task is not None:
                await await_owned_task(task)
        for task in tuple(vars(service).get("_llm_close_tasks", {}).values()):
            if task is not None:
                await await_owned_task(task)
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind", ["task", "future"])
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "error", "ordinary"])
async def test_facade_owner_retains_first_source_receipt_without_exporting_ordinary_errors(
    source_kind, kind,
):
    from kestrel_sovereign.features import isolated_runtime as runtime

    failure = RuntimeError("private-facade-no-export") if kind == "ordinary" else carrier(kind)
    if source_kind == "task":
        async def fail():
            raise failure
        source = asyncio.create_task(fail())
    else:
        source = asyncio.get_running_loop().create_future()
        source.set_exception(failure)
    operation = runtime._create_host_owned_facade_task(source, name="canonical-source-receipt")
    settled = asyncio.Event()
    operation.add_done_callback(lambda done: settled.set())
    try:
        await asyncio.wait_for(settled.wait(), 5)
        await await_owned_task(operation._host_task)
        for _ in range(3):
            exported = runtime._consume_late_lifecycle_task_outcome(operation)
            assert exported is (None if kind == "ordinary" else failure)
            with pytest.raises(BaseException) as caught:
                operation.result()
            assert caught.value is failure
    finally:
        await await_owned_task(source)
        await await_owned_task(operation._host_task)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "error", "ordinary"])
async def test_foreign_facade_exports_only_sanitized_canonical_control(kind, caplog):
    from kestrel_sovereign.features import isolated_runtime as runtime
    from kestrel_sovereign.execution_custody import execution_commit_outcome

    foreign_loop = asyncio.new_event_loop()
    source = foreign_loop.create_future()
    failure = carrier(kind) if kind != "ordinary" else RuntimeError("private-foreign-control")
    failure.private_marker = "private-foreign-control"
    source.set_exception(failure)
    operation = runtime._create_host_owned_facade_task(source, name="foreign-control-receipt")
    settled = asyncio.Event()
    operation.add_done_callback(lambda done: settled.set())
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(foreign_loop)
        foreign_loop.call_soon(ready.set)
        foreign_loop.run_forever()
        foreign_loop.close()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert await asyncio.to_thread(ready.wait, 5)
        operation.retry_owner_loop_cancellation_and_observation()
        await asyncio.wait_for(settled.wait(), 5)
        exported = runtime._consume_late_lifecycle_task_outcome(operation)
        if kind == "ordinary":
            assert exported is None
        else:
            assert exported is not failure
            assert exported.__traceback__ is None
            assert exported.__cause__ is None
            assert exported.__context__ is None
            assert not hasattr(exported, "private_marker")
            assert execution_commit_outcome(exported) == (
                "unknown" if "unknown" in kind else None
            )
        assert operation._source is None
        assert "private-foreign-control" not in caplog.text
    finally:
        foreign_loop.call_soon_threadsafe(foreign_loop.stop)
        await asyncio.to_thread(thread.join, 5)
        assert not thread.is_alive()
