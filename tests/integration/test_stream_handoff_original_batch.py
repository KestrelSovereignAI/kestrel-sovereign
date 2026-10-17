"""Actual deferred batch producer retains evidence across an aborted handoff."""

import asyncio
import secrets
from contextvars import ContextVar
from types import SimpleNamespace

import aiosqlite
import asyncpg
import pytest

from kestrel_sovereign import _async_ownership, execution_custody
from kestrel_sovereign._async_ownership import (
    OwnedAsyncIterator,
    await_owned_task,
    create_owned_task,
)
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.streaming import _DeferredToolBatchCancellation
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    _explicit_control_chain,
    bind_execution_custody_snapshot,
    close_execution_stream,
    execution_commit_outcome,
    is_execution_control_error,
    owned_execution_stream,
)
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "reader-close",
        "scoped-reader-close",
        "closer-observer-withdrawal",
        "native-closer-withdrawal",
        "post-validation",
        "checkpoint",
    ],
)
async def test_original_batch_receipt_survives_aborted_source_handoff(
    native_pg, monkeypatch, mode
):
    backend, observer, schema = native_pg
    await observer.execute(f'CREATE TABLE "{schema}".handoff_checkpoints (id INTEGER)')
    async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
        lease.require_live()
    entered, release = asyncio.Event(), asyncio.Event()
    native_active = ContextVar("actual_handoff_native_commit", default=False)
    original_exit = asyncpg.transaction.Transaction.__aexit__
    ack_failure = OSError("actual tool COMMIT acknowledgement lost before handoff")
    runtime = ExecutionCustody(ProcessRuntimeExecutionFence("actual-stream-component"))
    (
        original_sources,
        source_chains,
        batch_receipts,
        source_owners,
        received,
        reader_errors,
    ) = [], [], [], [], [], []
    denied = []
    closer_tasks = []
    original_receipts = []
    closer_started, closer_joining = asyncio.Event(), asyncio.Event()
    actual_create = _async_ownership.create_owned_task
    actual_join = _async_ownership.await_owned_task

    def create(operation, *args, **kwargs):
        task = actual_create(operation, *args, **kwargs)
        if isinstance(kwargs.get("retirement_resource"), OwnedAsyncIterator):
            closer_tasks.append(task)
            closer_started.set()
        return task

    async def join(task, *args, **kwargs):
        if asyncio.current_task() in closer_tasks:
            closer_joining.set()
        outcome = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() in closer_tasks and is_execution_control_error(
            outcome.error
        ):
            if all(outcome.error is not prior for prior, _ in original_receipts):
                original_receipts.append(
                    (outcome.error, tuple(_explicit_control_chain(outcome.error)))
                )
        return outcome

    monkeypatch.setattr(_async_ownership, "create_owned_task", create)
    monkeypatch.setattr(_async_ownership, "await_owned_task", join)
    actual_terminal = execution_custody.execution_terminal_error

    def terminal(*errors, **kwargs):
        for error in errors:
            if (
                isinstance(error, ExecutionAuthorityError)
                and str(error) == ("actual original reader admission retired")
                and error not in denied
            ):
                denied.append(error)
            if (
                isinstance(error, asyncio.CancelledError)
                and error.args
                in {
                    ("actual outer reader withdrawal during batch join",),
                    ("actual closer observer withdrawal",),
                    ("actual native closer withdrawal",),
                }
                and error not in reader_errors
            ):
                reader_errors.append(error)
        return actual_terminal(*errors, **kwargs)

    monkeypatch.setattr(execution_custody, "execution_terminal_error", terminal)

    async def exit(transaction, typ, error, tb):
        result = await original_exit(transaction, typ, error, tb)
        if native_active.get() and error is None and mode != "post-validation":
            entered.set()
            await release.wait()
            raise ack_failure
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", exit)

    class AgentInput(OrchestratorEngineMixin):
        features = _tool_to_feature = {}
        _direct_tools = {
            "native": SimpleNamespace(schema=SimpleNamespace(is_concurrency_safe=True))
        }

        def is_request_cancelled(self, request_id=None):
            return False

        def _visible_features_by_tool_name(self):
            return {}

        def _known_tool_names(self):
            return {"native"}

        async def _dispatch_tool_call(self, *args, **kwargs):
            token = native_active.set(True)
            tool_scope = ExecutionCustody(
                ProcessRuntimeExecutionFence("actual-tool-component")
            )
            try:
                with bind_execution_custody_snapshot((tool_scope,)):
                    await backend.execute(
                        "INSERT INTO effects VALUES (1, 'actual native tool')"
                    )
                if mode == "post-validation":
                    entered.set()
                    await release.wait()
                    runtime.revoke("actual original reader admission retired")
                    lease.require_live()
            except BaseException as error:
                original_sources.append(error)
                source_chains.append(tuple(_explicit_control_chain(error)))
                raise
            finally:
                native_active.reset(token)

        async def _execute_tool_batch_at_stop_boundary(self, *args, **kwargs):
            # Transparent producer observation: same actual source Task and
            # same result from the real batch, not a fabricated receipt.
            source_owners.append(asyncio.current_task())
            result = await super()._execute_tool_batch_at_stop_boundary(*args, **kwargs)
            if isinstance(result, _DeferredToolBatchCancellation):
                batch_receipts.append(result)
            return result

    agent = AgentInput()
    response = LLMResponse(
        content="", tool_calls=[ToolCall(id="native", name="native", arguments={})]
    )
    source = agent._handle_orchestrator_response_streaming(
        response=response,
        feature_tools=[],
        system_prompt="declared component context",
        force_local_only=False,
        effective_model="no-provider-call",
        user_message="component input",
        tool_events=[],
        tool_results=[],
        session_id="local",
        request_id="local",
    )
    scoped = mode in {"scoped-reader-close", "post-validation"}
    iterator = None

    async def read():
        nonlocal iterator
        scopes = (runtime,) if scoped else ()
        with bind_execution_custody_snapshot(scopes):
            async with owned_execution_stream(agent, source) as iterator:
                # Real producer emits the tool-start sentinel before dispatch.
                assert isinstance(await anext(iterator), str)
                try:
                    item = await anext(iterator)
                except asyncio.CancelledError as error:
                    reader_errors.append(error)
                    raise
                received.append(item)
                assert isinstance(item, _DeferredToolBatchCancellation)
                # Actual continuation handshake confirms delivery before the
                # caller's checkpoint, matching StreamingMixin's continue.
                with pytest.raises(StopAsyncIteration):
                    await anext(iterator)
                await observer.execute(
                    f'INSERT INTO "{schema}".handoff_checkpoints VALUES (1)'
                )
                item.raise_outcome()

    reader = create_owned_task(read(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if mode in {
            "reader-close",
            "scoped-reader-close",
            "closer-observer-withdrawal",
            "native-closer-withdrawal",
        }:
            reader.cancel("actual outer reader withdrawal during batch join")
            if mode == "closer-observer-withdrawal":
                await asyncio.wait_for(closer_started.wait(), 5)
                reader.cancel("actual closer observer withdrawal")
            elif mode == "native-closer-withdrawal":
                await asyncio.wait_for(closer_joining.wait(), 5)
                closer_tasks[-1].cancel("actual native closer withdrawal")
        else:
            source_owners[0].cancel("actual batch observer withdrawal before handoff")
        await asyncio.sleep(0)
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(reader), 5)
        assert original_sources and batch_receipts and outcome.error is not None
        assert all(task.done() for task in source_owners)
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
        chain = tuple(_explicit_control_chain(outcome.error))
        assert original_sources[0] in chain, (
            "aborted handoff discarded original native batch receipt"
        )
        assert batch_receipts[0].outcome.cancellation in chain, (
            "aborted handoff discarded actual batch observer withdrawal"
        )
        assert tuple(_explicit_control_chain(original_sources[0])) == source_chains[0]
        assert all(
            tuple(_explicit_control_chain(error)) == before
            for error, before in original_receipts
        ), "outer stream observer mutated an original source-owner receipt"
        if mode != "post-validation":
            assert execution_commit_outcome(outcome.error) == "unknown"
            assert ack_failure in chain
        else:
            assert denied and denied[0] in chain
        if mode in {
            "reader-close",
            "scoped-reader-close",
            "closer-observer-withdrawal",
            "native-closer-withdrawal",
        }:
            assert reader_errors and all(error in chain for error in reader_errors)
            if "withdrawal" in mode:
                assert len(reader_errors) == 2
        assert bool(received) == (mode == "checkpoint")
        assert await observer.fetchval(
            f'SELECT count(*) FROM "{schema}".handoff_checkpoints'
        ) == (1 if mode == "checkpoint" else 0)
    finally:
        release.set()
        await await_owned_task(reader)
        try:
            await source.aclose()
        except BaseException:
            # Expected native receipt was already inspected above.
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["body-and-close", "caused-close-cancel", "consumer-and-close"]
)
async def test_finite_native_iterator_close_preserves_actual_originals(
    native_pg, tmp_path, mode
):
    backend, _, _ = native_pg
    leases = []
    for _ in range(2):
        async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
            lease.require_live()
        leases.append(lease)
    path = tmp_path / "actual-source.sqlite"
    connection = await aiosqlite.connect(path)
    await connection.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    await connection.commit()
    originals, chains = [], []
    closed = asyncio.Event()

    def deny(lease):
        try:
            lease.require_live()
        except BaseException as error:
            originals.append(error)
            chains.append(tuple(_explicit_control_chain(error)))
            raise

    class NativeSource:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await connection.execute("INSERT INTO effects VALUES (1)")
            await connection.commit()
            if mode == "body-and-close":
                deny(leases[0])
            return "actual source item"

        async def aclose(self):
            await connection.close()
            closed.set()
            try:
                if mode == "caused-close-cancel":
                    # Real closed-handle driver error, not an authority cause
                    # that would itself preserve cancellation by default.
                    await connection.execute("SELECT 1")
                    pytest.fail("closed SQLite handle accepted a new query")
                else:
                    deny(leases[1])
            except BaseException as error:
                if mode != "caused-close-cancel":
                    raise
                originals.append(error)
                chains.append(tuple(_explicit_control_chain(error)))
                # Actual source-owner Task cancellation after physical close.
                asyncio.current_task().cancel("actual native source close cancelled")
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError as cancellation:
                    cancellation.__cause__ = error
                    originals.append(cancellation)
                    chains.append(tuple(_explicit_control_chain(cancellation)))
                    raise

    iterator = None
    observed = []
    try:
        if mode == "consumer-and-close":
            try:
                async with owned_execution_stream(object(), NativeSource()) as iterator:
                    assert await anext(iterator) == "actual source item"
                    deny(leases[0])
            except BaseException as error:
                observed.append(error)
        else:
            iterator = OwnedAsyncIterator(NativeSource, operation="actual-native-close")
            if mode == "body-and-close":
                try:
                    await anext(iterator)
                except BaseException as error:
                    observed.append(error)
            else:
                assert await anext(iterator) == "actual source item"
            try:
                await close_execution_stream(object(), iterator)
            except BaseException as error:
                observed.append(error)
        assert closed.is_set() and originals and observed
        for error in observed:
            chain = tuple(_explicit_control_chain(error))
            assert all(original in chain for original in originals), (
                "finite iterator close lost an original native body/closer carrier"
            )
        assert all(
            tuple(_explicit_control_chain(error)) == before
            for error, before in zip(originals, chains)
        )
        async with aiosqlite.connect(path) as observer:
            assert (
                await (
                    await observer.execute("SELECT count(*) FROM effects")
                ).fetchone()
            )[0] == 1
    finally:
        try:
            if iterator is not None:
                await iterator.aclose()
        except BaseException:
            pass
        await connection.close()
