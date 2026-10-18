"""Real streaming consumer keeps batch evidence when its checkpoint fails.

Context, model/tool schema and observational callbacks are declared component
inputs. Native transaction COMMIT/ACK failure, Task cancellation, the actual
batch carrier, streaming consumer and persistence helper execute. This is not
full Agent admission, provider response or READY certification.
"""

import asyncio
import inspect
from contextvars import ContextVar
from types import SimpleNamespace

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.streaming import (
    StreamingMixin,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["stopped", "post-audit-stop", "final"])
@pytest.mark.parametrize("checkpoint", ["sql-error", "admission", "healthy"])
async def test_stream_checkpoint_failure_keeps_original_batch_receipt(
    native_pg, monkeypatch, door, checkpoint
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
        privacy_config = SimpleNamespace(allows_cloud_llm=lambda: True)
        privacy_mode = SimpleNamespace(name="normal")

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

    class ContextInput:
        async def build_context(self, **kwargs):
            return SimpleNamespace(
                system_prompt="component constitution",
                messages=[],
                dynamic_user_context="",
            )

    class Observations:
        async def log_tool_call(self, **kwargs):
            return "actual-local-component-observation"

        async def log_tool_response(self, **kwargs):
            return None

        async def log_metric(self, **kwargs):
            return None

    class ModelInput:
        async def stream_with_tool_detection(self, **kwargs):
            yield input_response

        def resolve_provider_routing(self, **kwargs):
            return [], None

        def effective_request_timeout(self, *args, **kwargs):
            return None

    class StreamInput(StreamingMixin, OrchestratorEngineMixin):
        did = "did:example:declared-stream-component"
        hooks_manager = extension = operator_signal_producer = None
        features = _tool_to_feature = {}
        _direct_tools = {}
        user_prompt_template = "{query}"
        privacy_agent = PrivacyInput()
        context_manager = ContextInput()
        observability_store = Observations()
        llm_service = ModelInput()
        stopped = False

        async def _get_governing_constitution(self):
            return "component constitution, not Agent genesis"

        def is_request_cancelled(self, request_id=None):
            return self.stopped

        def _build_all_tools(self):
            return []

        def _assemble_post_build_system_prompt(self, prompt, *args, **kwargs):
            return prompt

        def _lazy_attachment_hint(self, attachments):
            return ""

        def _make_inline_tool_executor(self, *args, **kwargs):
            return None

        async def _dispatch_tool_call(self, *args, **kwargs):
            token = active.set(True)
            scope = ExecutionCustody(ProcessRuntimeExecutionFence(self.did))
            try:
                with bind_execution_custody_snapshot((scope,)):
                    await backend.execute(
                        "INSERT INTO effects VALUES (1, 'actual tool')"
                    )
            except BaseException as error:
                sources.append(error)
                source_chains.append(tuple(_explicit_control_chain(error)))
                raise
            finally:
                active.reset(token)

        async def _handle_orchestrator_response_streaming(self, **kwargs):
            source_tasks.append(asyncio.current_task())
            result = await self._execute_tool_batch_at_stop_boundary(
                input_response.tool_calls,
                {},
                set(),
                [],
                0,
                "declared component",
                tool_results=kwargs["tool_results"],
                defer_cancellation_to_persistence=True,
            )
            assert isinstance(result, _DeferredToolBatchCancellation)
            carriers.append(result)
            self.stopped = door == "stopped"
            yield result

        async def _fire_post_response_hook(self, response_text, *args, **kwargs):
            result = await super()._fire_post_response_hook(
                response_text, *args, **kwargs
            )
            self.stopped = door == "post-audit-stop"
            return result

    agent = StreamInput()
    stream = agent._process_input_streaming_traced_locked(
        "component input",
        "declared-model",
        "local-session",
        None,
        request_id="local-request",
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
