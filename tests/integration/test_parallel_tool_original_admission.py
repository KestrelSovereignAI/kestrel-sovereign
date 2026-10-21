"""Actual tool-batch admission owns each accepted Task through native PG work.

Declared concurrency-safe tool callbacks are local SQL components, not a full
Agent turn, provider answer or verified external effect. Actual Task allocation
refusal carries the original retired PG lease denial, never a fabricated Task.
"""

import asyncio
import inspect
import secrets
from types import SimpleNamespace

import pytest

from kestrel_sovereign import execution_custody
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.streaming import _DeferredToolBatchCancellation
from kestrel_sovereign.execution_custody import _explicit_control_chain
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["children", "batch"])
@pytest.mark.parametrize(
    "kind", ["native-control", "native-cancel", "ordinary", "healthy"]
)
async def test_parallel_tool_admission_joins_original_children_and_closes_refusal(
    native_pg, monkeypatch, stage, kind
):
    backend, observer, schema = native_pg
    await backend.execute("CREATE TABLE batch_effects (id INTEGER PRIMARY KEY)")
    async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
        lease.require_live()
    release = asyncio.Event()
    if kind == "healthy":
        release.set()
    offered, accepted, refused, originals, bodies = [], [], [], [], []
    actual_create = asyncio.create_task

    def create(operation, *args, **kwargs):
        name = kwargs.get("name") or ""
        selected = (
            name.startswith("orchestrator-parallel-tool:")
            or name == "orchestrator-tool-batch"
        )
        refuse = kind != "healthy" and (
            name == "orchestrator-tool-batch"
            if stage == "batch"
            else name == "orchestrator-parallel-tool:1"
        )
        if selected:
            offered.append(operation)
        if refuse:
            refused.append(operation)
            # Canonical admission wraps the original operation. Observe both
            # actual unstarted coroutines before the real factory refusal.
            frame = operation.cr_frame
            original_operation = frame.f_locals.get("operation") if frame else None
            if inspect.iscoroutine(original_operation):
                refused.append(original_operation)
            original = OSError("ordinary Task admission refused")
            if kind.startswith("native"):
                try:
                    lease.require_live()
                except BaseException as native:
                    original = native
                    if kind == "native-cancel":
                        original = asyncio.CancelledError(
                            "actual native admission withdrawal"
                        )
                        original.__cause__ = native
            originals.append(original)
            raise original
        eager_factory = getattr(asyncio, "eager_task_factory", None)
        if name == "orchestrator-parallel-tool:0" and callable(eager_factory):
            # Actual eager native Task: first child can already be awaiting PG
            # when admission of its sibling is refused. No scheduling sleep.
            task = eager_factory(asyncio.get_running_loop(), operation, **kwargs)
        else:
            task = actual_create(operation, *args, **kwargs)
        if selected:
            accepted.append(task)
        return task

    class AgentInput(OrchestratorEngineMixin):
        _direct_tools = {
            str(i): SimpleNamespace(schema=SimpleNamespace(is_concurrency_safe=True))
            for i in range(3)
        }
        _tool_to_feature = {}
        features = {}

        async def _dispatch_tool_call(self, call, *args, **kwargs):
            bodies.append(asyncio.current_task())
            await backend.execute(
                "INSERT INTO batch_effects VALUES (?)", (int(call.name),)
            )
            await release.wait()

    agent = AgentInput()
    calls = [SimpleNamespace(name=str(i), id=str(i), arguments={}) for i in range(3)]
    monkeypatch.setattr(asyncio, "create_task", create)
    outcome_error = None
    try:
        operation = (
            agent._execute_tool_batch_at_stop_boundary
            if stage == "batch"
            else agent._execute_tool_batch
        )
        try:
            await operation(calls, {}, set(), [], 0, "native provider-free component")
        except BaseException as error:
            outcome_error = error
        if kind == "healthy":
            assert outcome_error is None and len(bodies) == 3
            assert (
                await observer.fetchval(
                    f'SELECT count(*) FROM "{schema}".batch_effects'
                )
                == 3
            )
        else:
            assert originals and outcome_error is not None
            assert originals[0] in tuple(_explicit_control_chain(outcome_error))
            assert all(task.done() for task in accepted), (
                "partial tool admission abandoned accepted siblings"
            )
            assert refused and all(
                inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                for coro in refused
            ), "refused original tool admission leaked an unstarted coroutine"
            assert (
                (not bodies)
                if stage == "batch"
                else len(bodies) in ({0, 1} if hasattr(asyncio, "eager_task_factory") else {0})
            )
            # Receipt-gated eager Tasks may still be queued behind canonical
            # publication when the next allocation is refused. A pre-entered
            # body is not required; every accepted original must still join.
            # Native cancellation may retract a query; no committed-ACK claim.
            assert await observer.fetchval(
                f'SELECT count(*) FROM "{schema}".batch_effects'
            ) in {0, 1}
    finally:
        for coro in refused:
            if inspect.getcoroutinestate(coro) == inspect.CORO_CREATED:
                coro.close()
        release.set()
        for task in accepted:
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["children", "batch", "deferred"])
@pytest.mark.parametrize("kind", ["native-control", "native-cancel"])
async def test_parallel_join_keeps_actual_outer_withdrawal_and_source(
    native_pg, monkeypatch, stage, kind
):
    backend, observer, schema = native_pg
    await backend.execute("CREATE TABLE joined_batch_effect (id INTEGER PRIMARY KEY)")
    async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
        lease.require_live()
    ready, release = asyncio.Event(), asyncio.Event()
    bodies, originals, withdrawals, original_chains = [], [], [], []
    caller = None
    actual_terminal = execution_custody.execution_terminal_error

    def terminal(*errors, **kwargs):
        if asyncio.current_task() is caller:
            for error in errors:
                if isinstance(error, asyncio.CancelledError) and error.args == (
                    "actual batch observer withdrawal",
                ):
                    if error not in withdrawals:
                        withdrawals.append(error)
        return actual_terminal(*errors, **kwargs)

    class AgentInput(OrchestratorEngineMixin):
        _direct_tools = {
            str(i): SimpleNamespace(schema=SimpleNamespace(is_concurrency_safe=True))
            for i in range(2)
        }
        _tool_to_feature = {}
        features = {}

        async def _dispatch_tool_call(self, call, *args, **kwargs):
            await backend.execute(
                "INSERT INTO joined_batch_effect VALUES (?)", (int(call.name),)
            )
            bodies.append(asyncio.current_task())
            if len(bodies) == 2:
                ready.set()
            try:
                await release.wait()
            finally:
                if call.name == "1":
                    try:
                        lease.require_live()
                    except BaseException as original:
                        originals.append(original)
                        original_chains.append(tuple(_explicit_control_chain(original)))
                        if kind == "native-cancel":
                            cancellation = asyncio.CancelledError(
                                "actual native tool observation cancelled"
                            )
                            cancellation.__cause__ = original
                            originals.append(cancellation)
                            original_chains.append(
                                tuple(_explicit_control_chain(cancellation))
                            )
                            raise cancellation
                        raise

    from kestrel_sovereign import _async_ownership

    agent = AgentInput()
    operation = (
        agent._execute_tool_batch_at_stop_boundary
        if stage != "children"
        else agent._execute_tool_batch
    )
    calls = [SimpleNamespace(name=str(i), id=str(i), arguments={}) for i in range(2)]
    monkeypatch.setattr(execution_custody, "execution_terminal_error", terminal)

    async def execute():
        # Actual production carrier from the joined batch, not a fabricated
        # outcome. Checkpoint/Agent persistence itself is not claimed here.
        kwargs = (
            {"defer_cancellation_to_persistence": True} if stage == "deferred" else {}
        )
        result = await operation(
            calls, {}, set(), [], 0, "local native callback", **kwargs
        )
        if isinstance(result, _DeferredToolBatchCancellation):
            result.raise_outcome()
        return result

    caller = _async_ownership.create_owned_task(execute(), retain_originals=True)
    try:
        await asyncio.wait_for(ready.wait(), 5)
        caller.cancel("actual batch observer withdrawal")
        await asyncio.sleep(0)
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(caller), 5)
        assert all(task.done() for task in bodies)
        assert (
            await observer.fetchval(
                f'SELECT count(*) FROM "{schema}".joined_batch_effect'
            )
            == 2
        )
        assert (
            originals
            and outcome.error is not None
            and originals[0] in tuple(_explicit_control_chain(outcome.error))
        )
        assert withdrawals and withdrawals[0] in tuple(
            _explicit_control_chain(outcome.error)
        ), "outer tool batch discarded its actual observer withdrawal"
        assert all(
            tuple(_explicit_control_chain(error)) == before
            for error, before in zip(originals, original_chains)
        ), "finite tool observer polluted original source receipt history"
    finally:
        release.set()
        await await_owned_task(caller)
