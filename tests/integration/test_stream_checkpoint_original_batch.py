"""Real streaming consumer keeps batch evidence when its checkpoint fails.

Context, model/tool schema and observational callbacks are declared component
inputs. Native transaction COMMIT/ACK failure, Task cancellation, the actual
batch carrier, streaming consumer and persistence helper execute inside a real
Agent request and journal. No provider response or READY certification is claimed.
"""

import asyncio
import inspect
from contextvars import ContextVar

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent.streaming import (
    _DeferredToolBatchCancellation,
)
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    _explicit_control_chain,
    bind_execution_custody_snapshot,
    execution_commit_outcome,
)
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.utils.native_streaming import streaming_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["stopped", "post-audit-stop", "final"])
@pytest.mark.parametrize("checkpoint", ["sql-error", "admission", "healthy"])
async def test_stream_checkpoint_failure_keeps_original_batch_receipt(
    native_pg, monkeypatch, door, checkpoint, native_tool_effect_agent
):
    backend, observer, schema = native_pg
    await observer.execute(f'CREATE TABLE "{schema}".stream_history (role TEXT)')
    entered, release = asyncio.Event(), asyncio.Event()
    active = ContextVar("actual_native_tool_commit", default=False)
    ack_failure = OSError("actual original tool COMMIT acknowledgement lost")
    actual_exit = asyncpg.transaction.Transaction.__aexit__
    actual_create = asyncio.create_task
    sources, source_chains, carriers, source_tasks, checkpoints, refused = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    input_response = LLMResponse(
        content="", tool_calls=[ToolCall(id="native", name="native", arguments={})]
    )

    async def native_exit(transaction, typ, error, tb):
        result = await actual_exit(transaction, typ, error, tb)
        if active.get() and error is None:
            entered.set()
            await release.wait()
            raise ack_failure
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_exit)

    def create(operation, *args, **kwargs):
        if checkpoint == "admission" and kwargs.get("name") == "persist-assistant-turn":
            refused.append(operation)
            failure = RuntimeError("actual checkpoint Task factory refused")
            checkpoints.append(failure)
            raise failure
        return actual_create(operation, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", create)

    class PrivacyInput:
        async def get_conversation_history(self, **kwargs):
            return []

        async def add_conversation(self, role, content, **kwargs):
            if role == "assistant" and checkpoint == "sql-error":
                try:
                    await observer.execute(
                        f'SELECT * FROM "{schema}".missing_checkpoint'
                    )
                except BaseException as error:
                    checkpoints.append(error)
                    raise
            await observer.execute(
                f'INSERT INTO "{schema}".stream_history VALUES ($1)', role
            )

    agent = await streaming_agent(native_tool_effect_agent, connected_backend=backend)
    privacy = PrivacyInput()
    agent.privacy_agent.get_conversation_history = privacy.get_conversation_history
    agent.privacy_agent.add_conversation = privacy.add_conversation
    agent.hooks_manager = None
    agent.stopped = False
    agent.is_request_cancelled = lambda request_id=None: agent.stopped
    agent._make_inline_tool_executor = lambda *args, **kwargs: None

    async def model_stream(**kwargs):
        yield input_response

    agent.llm_service.stream_with_tool_detection = model_stream

    async def dispatch(*args, **kwargs):
        token = active.set(True)
        scope = ExecutionCustody(ProcessRuntimeExecutionFence(agent.did))
        try:
            with bind_execution_custody_snapshot((scope,)):
                await backend.execute("INSERT INTO effects VALUES (1, 'actual tool')")
        except BaseException as error:
            sources.append(error)
            source_chains.append(tuple(_explicit_control_chain(error)))
            raise
        finally:
            active.reset(token)

    agent._dispatch_tool_call = dispatch

    async def orchestrate(**kwargs):
        source_tasks.append(asyncio.current_task())
        result = await agent._execute_tool_batch_at_stop_boundary(
            input_response.tool_calls, {}, set(), [], 0, "declared component",
            tool_results=kwargs["tool_results"], defer_cancellation_to_persistence=True,
        )
        assert isinstance(result, _DeferredToolBatchCancellation)
        carriers.append(result)
        agent.stopped = door == "stopped"
        yield result

    agent._handle_orchestrator_response_streaming = orchestrate
    post_response = agent._fire_post_response_hook

    async def after_audit(response_text, *args, **kwargs):
        result = await post_response(response_text, *args, **kwargs)
        agent.stopped = door == "post-audit-stop"
        return result

    agent._fire_post_response_hook = after_audit
    stream = agent.process_input_streaming(
        "component input", model_override="declared-model",
        session_id="local-session", request_id="local-request",
    )

    async def consume():
        return [item async for item in stream]

    caller = create_owned_task(consume(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert source_tasks and not source_tasks[0].done()
        source_tasks[0].cancel("actual batch observer withdrawal")
        await asyncio.sleep(0)
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(caller), 5)
        assert carriers and sources and all(task.done() for task in source_tasks)
        assert execution_commit_outcome(sources[0]) == "unknown"
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
        assert outcome.error is not None
        chain = tuple(_explicit_control_chain(outcome.error))
        assert sources[0] in chain, (
            "failed checkpoint discarded original UNKNOWN batch receipt"
        )
        assert carriers[0].outcome.cancellation in chain, (
            "failed checkpoint discarded actual batch caller"
        )
        assert ack_failure in chain
        assert all(error in chain for error in checkpoints)
        assert tuple(_explicit_control_chain(sources[0])) == source_chains[0], (
            "checkpoint consumer polluted original shared batch receipt"
        )
        assert await observer.fetchval(
            f'SELECT count(*) FROM "{schema}".stream_history WHERE role = $1',
            "assistant",
        ) == (1 if checkpoint == "healthy" else 0)
    finally:
        release.set()
        for operation in refused:
            if inspect.getcoroutinestate(operation) == inspect.CORO_CREATED:
                operation.close()
        await await_owned_task(caller)
        try:
            await stream.aclose()
        except BaseException:
            # Original expected native receipt already inspected above.
            pass
