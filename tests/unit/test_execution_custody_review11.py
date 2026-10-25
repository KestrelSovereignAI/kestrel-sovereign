"""Provider-free probes of review11's remaining production consumers."""

import asyncio

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    _acknowledged_commits,
    _explicit_control_chain,
    execution_commit_outcome,
    bind_execution_custody,
    is_execution_control_error,
    require_execution_work,
)
from tests.unit.test_execution_custody import Authority
from tests.unit.test_streaming_usage_metering import _FakeService


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
async def test_isolated_invocation_preserves_cancel_carried_commit_evidence(outcome):
    from kestrel_sovereign.agent.invocation import bind_async_invocation
    from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
    dispositions = []
    error = asyncio.CancelledError("effect acknowledgement raced Stop")
    error.__cause__ = ExecutionCommitOutcomeError(outcome)

    class Owner:
        def register_active_request(self, request_id, *, nested=False):
            pass
        async def await_durable_request_admission(self, request_id):
            return True
        def bind_request_operation(self, request_id, operation):
            pass
        def _cleanup_cancelled_request(self, request_id, **kwargs):
            dispositions.append(kwargs.get("disposition"))
        @bind_async_invocation("request_id", track_request_lifecycle=True)
        async def turn(self, request_id=None):
            raise error

    with pytest.raises(asyncio.CancelledError) as caught:
        await Owner().turn(request_id="irreversible")
    assert caught.value is error
    assert dispositions == [RequestCompletionDisposition.ABANDONED]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
@pytest.mark.parametrize("partial", [False, True])
async def test_unbound_stream_control_bypasses_ordinary_accounting(outcome, carrier, partial):
    service = _FakeService()
    error = ExecutionAuthorityError("denied") if outcome == "denied" else ExecutionCommitOutcomeError(outcome)
    if carrier != "direct":
        wrapper = asyncio.CancelledError("Stop") if carrier == "cancel" else RuntimeError("wrapper")
        wrapper.__cause__ = error
        error = wrapper

    class Adapter:
        supports_partial_usage_flush = True
        async def get_streaming_response_with_tools(self, *, usage_sink, **kwargs):
            if partial:
                usage_sink["input_tokens"] = 23
                yield "partial"
            raise error
            yield

    stream = service._stream_adapter_with_usage(
        adapter=Adapter(), client=None, model="synthetic", messages=[],
        provider_name="local-test", path="stream", invocation_context=None,
        expose_protocol_events=False,
    )
    try:
        with pytest.raises(type(error)) as caught:
            async for _ in stream:
                pass
        assert caught.value is error
        service._log_llm_call.assert_not_awaited()
        service._track_model_usage.assert_not_awaited()
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_ordinary_live_stream_failure_keeps_native_accounting(partial):
    service = _FakeService()
    service._execution_custody = ExecutionCustody(Authority())
    error = ValueError("ordinary provider failure with live authority")

    class Adapter:
        supports_partial_usage_flush = True

        async def get_streaming_response_with_tools(self, *, usage_sink, **kwargs):
            if partial:
                usage_sink["input_tokens"] = 23
                yield "partial"
            raise error
            yield

    stream = service._stream_adapter_with_usage(
        adapter=Adapter(), client=None, model="synthetic", messages=[],
        provider_name="local-test", path="stream", invocation_context=None,
        expose_protocol_events=False,
    )
    try:
        with pytest.raises(ValueError) as caught:
            async for _ in stream:
                pass
        assert caught.value is error
        service._log_llm_call.assert_awaited_once()
        if partial:
            service._track_model_usage.assert_awaited_once()
        require_execution_work(service)
    finally:
        await stream.aclose()








# The native host fixture supplies real SDK bindings, signed sender proofs,
# primary sender/recipient ledgers and a durable cognition dispatcher.
from contextlib import asynccontextmanager
from types import SimpleNamespace

from kestrel_sovereign._async_ownership import await_owned_task
from tests.unit.test_a2a_native_host import (
    hosted as hosted,  # noqa: PLC0414 - pytest fixture export
)
from tests.unit.test_a2a_native_host import (
    hosted_recipient_signing as hosted_recipient_signing,  # noqa: PLC0414 - pytest fixture export
)
from tests.unit.test_a2a_native_host import (
    local_native_peer as local_native_peer,  # noqa: PLC0414 - pytest fixture export
)
from tests.unit.test_a2a_native_host import (
    native_question_peer as native_question_peer,  # noqa: PLC0414 - pytest fixture export
)
from tests.unit.test_a2a_native_host import (
    publication as publication,  # noqa: PLC0414 - pytest fixture export
)


async def _use_original_runtime(p, tmp_path):
    await p.feature.shutdown()
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    owner = KestrelAgent(
        did=p.sender.did,
        storage_path=str(tmp_path / "question-runtime.db"),
    )
    owner._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(p.sender.did))
    owner.identity = p.sender.identity
    owner.task_manager = p.sender_manager
    owner.dispatcher = p.dispatcher
    # The real dispatcher must share the same original runtime owner. Its
    # default fixture agent deliberately has no hosted resident handoff.
    previous_agent = p.dispatcher._agent
    owner.process_input = previous_agent.process_input
    owner._durable_persistence_gate = previous_agent._get_durable_persistence_gate()
    p.dispatcher._agent = owner
    owner.get_current_turn_id = lambda: "original-question-turn"
    owner.get_turn_bound_session_id = lambda: "original-question-session"
    p.feature.agent = owner
    owner.features = {"Peers": p.feature}
    return owner


def _assert_original_runtime_retirement(error):
    assert is_execution_control_error(error)
    assert any(
        isinstance(original, ExecutionAuthorityError)
        and "original runtime retired" in str(original)
        for original in _explicit_control_chain(error)
    )


async def _shutdown_question_runtime(feature, owner):
    """Harvest only a physically acknowledged, retired query cancellation."""
    try:
        await feature.shutdown()
    except asyncio.CancelledError as error:
        chain = tuple(_explicit_control_chain(error))
        assert all(isinstance(original, asyncio.CancelledError) for original in chain)
        assert execution_commit_outcome(error) == "committed"
        receipts = tuple(receipt for original in chain for receipt in _acknowledged_commits(original))
        assert receipts and all(receipt.acknowledged and receipt.retired for receipt in receipts)
        assert not feature._owned_background_tasks
        require_execution_work(owner)
        # This test has inspected the exact receipts after their original
        # native reset finished. The fixture must not report them anew.
        feature._owned_background_control = None
        return error
    return None


async def test_native_question_resident_waits_for_ready_and_outlives_boot_claim(
    native_question_peer, monkeypatch, tmp_path,
):
    p = native_question_peer
    if p.backend.backend_type != "postgres":
        pytest.skip("Transaction-bound execution custody requires PostgreSQL")
    owner = await _use_original_runtime(p, tmp_path)
    observed = asyncio.Event()
    original = p.sender_manager.question_store.unsettled
    async def observed_read():
        require_execution_work(owner)
        observed.set()
        return await original()
    monkeypatch.setattr(p.sender_manager.question_store, "unsettled", observed_read)
    with bind_execution_custody(owner._execution_custody.fence) as boot_claim:
        await p.feature.post_all_features_loaded(owner)
    resident = p.feature._question_supervisor_task
    assert resident.get_name() == "native_a2a_question_recovery"
    assert not observed.is_set()
    boot_claim.revoke("cold boot finished")
    owner._runtime_publication_ready.set()
    await asyncio.wait_for(observed.wait(), 2)
    assert not resident.done(), "resident retained the completed boot claim"
    await _shutdown_question_runtime(p.feature, owner)
    assert resident.done() and not p.feature._owned_background_tasks


@pytest.mark.parametrize("retirement,deadline_query,hold_native_commit", [
    ("none", False, False), ("generation", False, False), ("disable", False, False),
    ("generation", True, False), ("disable", False, True),
])
async def test_native_question_recovery_retains_original_runtime_after_request_turn(
    native_question_peer, retirement, deadline_query, hold_native_commit, tmp_path, monkeypatch,
):
    p = native_question_peer
    if p.backend.backend_type != "postgres":
        pytest.skip("Transaction-bound execution custody requires PostgreSQL")
    owner = await _use_original_runtime(p, tmp_path)
    owner._runtime_publication_ready.set()
    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    query_entered, query_release, query_closed = (
        asyncio.Event(), asyncio.Event(), asyncio.Event()
    )
    native_commit_entered, native_commit_release, arm_native_commit = (
        asyncio.Event(), asyncio.Event(), asyncio.Event()
    )
    native_cancel_requested = asyncio.Event()
    shutdown_task = None
    native_exit_owner = None
    if deadline_query or hold_native_commit:
        from kestrel_sovereign.a2a.question_supervisor import NativeQuestionSupervisor

        # Track the actual deadline lane. The generation case pauses SQL
        # admission; the COMMIT case pauses only its original native exit
        # after the transaction has physically committed.
        deadline_task = None
        original_deadline = NativeQuestionSupervisor._deadline_loop
        original_executor = p.backend._execution_query_executor

        async def deadline_loop(supervisor):
            nonlocal deadline_task
            deadline_task = asyncio.current_task()
            try:
                await original_deadline(supervisor)
            finally:
                query_closed.set()

        @asynccontextmanager
        async def held_executor(executor, commit_state=None):
            if asyncio.current_task() is deadline_task and not query_entered.is_set():
                query_entered.set()
                await query_release.wait()
            async with original_executor(executor, commit_state) as checked:
                yield checked

        monkeypatch.setattr(NativeQuestionSupervisor, "_deadline_loop", deadline_loop)
        if deadline_query:
            monkeypatch.setattr(p.backend, "_execution_query_executor", held_executor)
        if hold_native_commit:
            import asyncpg

            original_exit = asyncpg.transaction.Transaction.__aexit__

            async def held_native_exit(transaction, error_type, error, traceback):
                nonlocal native_exit_owner
                result = await original_exit(transaction, error_type, error, traceback)
                if (
                    error_type is None and arm_native_commit.is_set()
                    and not native_commit_entered.is_set() and deadline_task is not None
                    and asyncio.current_task().get_name()
                    == f"{deadline_task.get_name()}:postgres-original-transaction-exit"
                ):
                    assert transaction._state.name == "COMMITTED"
                    assert not transaction._connection.is_in_transaction()
                    native_exit_owner = asyncio.current_task()
                    native_commit_entered.set()
                    await native_commit_release.wait()
                return result

            monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_native_exit)
    original_open = p.feature._open_native_dispatch
    @asynccontextmanager
    async def held_read(dispatch_id):
        async with original_open(dispatch_id) as (session, dispatch):
            async def get_task(retained):
                entered.set()
                try:
                    await release.wait()
                    require_execution_work(owner)
                    return await session.get_task(retained)
                finally:
                    closed.set()
            yield SimpleNamespace(get_task=get_task), dispatch
    p.feature._open_native_dispatch = held_read
    await p.feature.post_all_features_loaded(owner)
    with bind_execution_custody(owner._execution_custody.fence) as request_claim:
        result = await p.feature.send_a2a_question("Peer", "Native custody question")
    assert result.data["sent"] is True
    request_claim.revoke("request turn finished")
    resident = p.feature._question_supervisor_task
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await p.manager.complete_task(result.data["remote_task_id"], "Native reply", recipient_agent_id=p.recipient.did)
        if retirement == "generation":
            if deadline_query:
                await asyncio.wait_for(query_entered.wait(), 3)
            owner._execution_custody.revoke("original runtime retired")
            if deadline_query:
                query_release.set()
                await asyncio.wait_for(query_closed.wait(), 3)
        elif retirement == "disable":
            if hold_native_commit:
                from tests.utils.aiosqlite_workers import wait_for_lifecycle_checkpoint

                arm_native_commit.set()
                p.feature._question_supervisor.notify()
                await wait_for_lifecycle_checkpoint(
                    native_commit_entered.wait(), resident,
                    description="the deadline lane's actual PostgreSQL COMMIT",
                )
                original_cancel = deadline_task.cancel

                def observed_cancel(*args, **kwargs):
                    accepted = original_cancel(*args, **kwargs)
                    if accepted:
                        native_cancel_requested.set()
                    return accepted

                monkeypatch.setattr(deadline_task, "cancel", observed_cancel)
                shutdown_task = asyncio.create_task(_shutdown_question_runtime(p.feature, owner))
                await wait_for_lifecycle_checkpoint(
                    native_cancel_requested.wait(), shutdown_task,
                    description="deadline cancellation before its COMMIT acknowledgement",
                )
                assert not native_commit_release.is_set()
                native_commit_release.set()
                receipt_error = await shutdown_task
                assert receipt_error is not None
                assert any(
                    receipt.commit_owner is native_exit_owner
                    for original in _explicit_control_chain(receipt_error)
                    for receipt in _acknowledged_commits(original)
                ), "shutdown lost the exact original deadline COMMIT receipt"
            else:
                await _shutdown_question_runtime(p.feature, owner)
        release.set()
        if retirement == "none":
            async with asyncio.timeout(3):
                while not (await p.sender_manager.question_store.get(result.data["question_id"])).receiver_settled or await p.sender_manager.question_store.pending_wakes():
                    await asyncio.sleep(0.01)
            assert (await p.backend.fetch_one("SELECT COUNT(*) FROM durable_signal_events WHERE source = ?", ("a2a.question_answered",)))[0] == 1
            assert not resident.done(), "request turn retired its resident recovery"
        else:
            outcome = await await_owned_task(resident)
            if retirement == "generation":
                _assert_original_runtime_retirement(outcome.error)
            else:
                assert isinstance(outcome.error, asyncio.CancelledError)
            assert not (await p.sender_manager.question_store.get(result.data["question_id"])).receiver_settled
            assert (await p.backend.fetch_one("SELECT COUNT(*) FROM durable_signal_events WHERE source = ?", ("a2a.question_answered",)))[0] == 0
        assert closed.is_set()
    finally:
        release.set()
        query_release.set()
        native_commit_release.set()
        if shutdown_task is not None:
            await await_owned_task(shutdown_task)
        try:
            if retirement == "generation":
                await p.feature.shutdown()
            else:
                await _shutdown_question_runtime(p.feature, owner)
        except BaseException as error:
            assert retirement == "generation"
            _assert_original_runtime_retirement(error)
            if deadline_query:
                from kestrel_sdk.storage.database.interface import QueryError

                assert any(
                    isinstance(original, QueryError)
                    for original in _explicit_control_chain(error)
                )
            assert not p.feature._owned_background_tasks
            # This test has harvested the expected retained receipt. The
            # shared fixture repeats shutdown and must not report it anew.
            p.feature._owned_background_control = None
