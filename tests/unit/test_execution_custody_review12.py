"""Production boundary regressions; no provider calls or artificial load."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError, ExecutionCustody, _ExecutionForwarder,
    execution_commit_outcome,
)
from tests.unit.test_execution_custody import Authority


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
async def test_failed_effect_checkpoint_preserves_native_outcome(outcome, carrier):
    from kestrel_sovereign.agent.invocation import (
        bind_async_invocation, mark_current_invocation_effect_completed,
    )
    from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
    ordinary = ValueError("turn failed after completed tool")
    evidence = ExecutionCommitOutcomeError(outcome)
    if carrier != "direct":
        wrapper = asyncio.CancelledError("checkpoint cancelled") if carrier == "cancel" else RuntimeError("checkpoint wrapper")
        wrapper.__cause__ = evidence
        evidence = wrapper
    dispositions = []

    class Owner:
        def register_active_request(self, request_id, *, nested=False):
            pass
        async def await_durable_request_admission(self, request_id):
            return True
        def bind_request_operation(self, request_id, operation):
            pass
        async def _persist_completed_tool_stop_checkpoint(self, **kwargs):
            raise evidence
        def _cleanup_cancelled_request(self, request_id, **kwargs):
            dispositions.append(kwargs.get("disposition"))
        @bind_async_invocation("request_id", track_request_lifecycle=True)
        async def turn(self, request_id=None):
            mark_current_invocation_effect_completed("session")
            raise ordinary

    with pytest.raises(BaseException) as caught:
        await Owner().turn(request_id="checkpoint")
    assert caught.value is evidence
    assert execution_commit_outcome(caught.value) == outcome
    assert dispositions == [RequestCompletionDisposition.ABANDONED]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
@pytest.mark.parametrize("bound", [False, True])
async def test_active_forwarder_close_preserves_cancellation_carried_control(outcome, bound):
    entered, closed = asyncio.Event(), asyncio.Event()
    evidence = asyncio.CancelledError("native commit interrupted")
    evidence.__cause__ = ExecutionCommitOutcomeError(outcome)
    owner = SimpleNamespace(_execution_custody=ExecutionCustody(Authority()) if bound else None)

    async def source():
        try:
            yield "partial"
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise evidence
        finally:
            closed.set()

    stream = _ExecutionForwarder(owner, source())
    advance = None
    try:
        assert await anext(stream) == "partial"
        async def read_terminal():
            try:
                return await anext(stream)
            except BaseException as error:
                return error
        advance = asyncio.create_task(read_terminal())
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(asyncio.CancelledError) as caught:
            await stream.aclose()
        assert caught.value is evidence
        assert execution_commit_outcome(caught.value) == outcome
        assert closed.is_set()
        result = (await asyncio.gather(advance, return_exceptions=True))[0]
        assert execution_commit_outcome(result) == outcome
    finally:
        await stream.aclose()
        if advance is not None:
            if not advance.done():
                advance.cancel()
            await asyncio.gather(advance, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
async def test_owned_iterator_final_close_preserves_control(outcome):
    from kestrel_sovereign._async_ownership import OwnedAsyncIterator
    entered, closed = asyncio.Event(), asyncio.Event()
    evidence = asyncio.CancelledError("native final close interrupted")
    evidence.__cause__ = ExecutionCommitOutcomeError(outcome)

    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration
        async def aclose(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise evidence
            finally:
                closed.set()

    stream = OwnedAsyncIterator(Source, operation="native-close-probe")
    await asyncio.wait_for(entered.wait(), 2)
    with pytest.raises(asyncio.CancelledError) as caught:
        await stream.aclose()
    assert caught.value is evidence
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unknown", "committed"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_owned_iterator_control_survives_later_ordinary_close_failure(outcome, cancel):
    from kestrel_sovereign._async_ownership import OwnedAsyncIterator
    evidence = ExecutionCommitOutcomeError(outcome)
    if cancel:
        wrapper = asyncio.CancelledError("effect cancellation")
        wrapper.__cause__ = evidence
        evidence = wrapper
    closed = asyncio.Event()
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise evidence
        async def aclose(self):
            closed.set()
            raise ValueError("ordinary late close failure")
    stream = OwnedAsyncIterator(Source, operation="native-evidence-precedence")
    with pytest.raises(type(evidence)) as caught:
        await anext(stream)
    assert caught.value is evidence
    assert stream.terminal_error is evidence
    assert closed.is_set()
    await stream.aclose()


@pytest.mark.parametrize("outcome", ["unknown", "committed"])
@pytest.mark.parametrize("source", ["owner", "caller"])
def test_owned_outcome_does_not_replace_control_with_ordinary_cancel_cause(outcome, source):
    from kestrel_sovereign._async_ownership import OwnedTaskOutcome, raise_owned_outcome
    evidence = ExecutionCommitOutcomeError(outcome)
    cancellation = asyncio.CancelledError("caller cancelled")
    if source == "caller":
        cancellation.__cause__ = evidence
        expected = cancellation
        result = OwnedTaskOutcome(None, ValueError("ordinary owned failure"), cancellation)
    else:
        expected = evidence
        result = OwnedTaskOutcome(None, evidence, cancellation)
    with pytest.raises(type(expected)) as caught:
        raise_owned_outcome(result, operation="native-control-probe")
    assert caught.value is expected
    assert execution_commit_outcome(caught.value) == outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["factory", "retired", "ambient", "live", "local"])
async def test_manager_validates_original_custody_before_provider_birth(monkeypatch, tmp_path, failure):
    from kestrel_sovereign.multi_agent import agent_manager as module
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError, bind_execution_custody, current_execution_custody,
    )
    custody = ExecutionCustody(Authority())
    if failure == "local":
        from kestrel_sovereign.execution_custody import ProcessRuntimeExecutionFence
        custody = ExecutionCustody(ProcessRuntimeExecutionFence("did:test:birth"))
    rejection = ExecutionAuthorityError("original host generation denied")
    if failure == "retired":
        custody.revoke(str(rejection))
    def factory(*args):
        if failure == "factory":
            raise rejection
        return custody
    manager = module.AgentManager(base_data_dir=tmp_path, execution_custody_factory=factory)
    config = module.LocalAgentConfig(data_dir=tmp_path / "agent", port=8801)
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://provider-free-probe")
    monkeypatch.setattr(module.LocalAgentConfig, "validate_runtime", lambda *a, **k: [])
    monkeypatch.setattr(module, "read_anchor_agent_did", AsyncMock(return_value="did:test:birth"))
    service = SimpleNamespace(close=AsyncMock(), attach_to_agent=Mock())
    def construct(**kwargs):
        assert custody in current_execution_custody()
        return service
    provider = Mock(side_effect=construct)
    monkeypatch.setattr(module, "LLMService", provider)
    stop = ValueError("construction boundary reached")
    candidate = Mock(side_effect=stop)
    monkeypatch.setattr(module, "KestrelAgent", candidate)
    if failure == "ambient":
        with bind_execution_custody(Authority()) as ambient:
            ambient.revoke("caller admission retired")
            with pytest.raises(ExecutionAuthorityError):
                await manager._initialize_agent("agent", config)
    elif failure in {"live", "local"}:
        with pytest.raises(ValueError) as caught:
            await manager._initialize_agent("agent", config)
        assert caught.value is stop
        provider.assert_called_once()
        assert candidate.call_args.kwargs["execution_custody"] is custody
        service.close.assert_awaited_once()
        if failure == "local":
            with pytest.raises(ExecutionAuthorityError, match="process runtime retired"):
                custody.require_work()
        else:
            custody.require_work()  # Borrowed host generation remains live.
    else:
        with pytest.raises(ExecutionAuthorityError):
            await manager._initialize_agent("agent", config)
    if failure not in {"live", "local"}:
        provider.assert_not_called()
        candidate.assert_not_called()


@pytest.mark.asyncio
async def test_standalone_original_lifetime_is_not_cold_claim_or_shared_tenant(monkeypatch, tmp_path):
    from kestrel_sovereign import server
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError, bind_execution_custody, require_execution_work,
    )
    from tests.unit.test_agent_boot_phases import _make_agent
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://provider-free-probe")
    agent = _make_agent(tmp_path)
    scope = server._standalone_runtime_execution_custody("agent", agent.did, None)
    agent._execution_custody = scope
    seen, release = asyncio.Event(), asyncio.Event()
    async def resident():
        require_execution_work(agent)
        seen.set()
        await release.wait()
        require_execution_work(agent)
    task = None
    try:
        with bind_execution_custody(Authority()):
            task = agent._track_runtime_task(resident(), name="cold-resident")
        agent._runtime_publication_ready.set()
        await asyncio.wait_for(seen.wait(), 2)
        require_execution_work(agent)
        # Instrument the FIRST teardown budget seam of the real shutdown,
        # without inventing a replacement shutdown lifecycle.
        import kestrel_sovereign.kestrel_agent as module
        agent.storage = SimpleNamespace()
        class RetirementObserved(Exception):
            pass
        def second_teardown(storage):
            assert storage is agent.storage
            with pytest.raises(ExecutionAuthorityError, match="process runtime retired"):
                require_execution_work(agent)
            raise RetirementObserved
        monkeypatch.setattr(module, "_minimum_storage_potential_preclose_timeout", second_teardown)
        with pytest.raises(RetirementObserved):
            await agent.shutdown()
        with pytest.raises(ExecutionAuthorityError, match="process runtime retired"):
            require_execution_work(agent)
        release.set()
        with pytest.raises(ExecutionAuthorityError):
            await task
        replacement = server._standalone_runtime_execution_custody("agent", agent.did, None)
        assert replacement is not scope
        replacement.require_work()
    finally:
        release.set()
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.llm_service.close()
