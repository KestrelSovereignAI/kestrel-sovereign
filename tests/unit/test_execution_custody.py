"""Irrevocable native authority across copied contexts (#3569)."""

import asyncio

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    bind_execution_custody,
    require_execution_work,
)
from kestrel_sovereign.turn_scope import capture_turn_scope


class Authority:
    backend_type = "postgres"

    def require_work(self):
        pass

    async def lock_and_validate(self, connection):
        raise AssertionError("the pure context tests must not issue database work")


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["normal", "revoke"])
async def test_copied_child_cannot_regain_authority_after_scope_ends(retirement):
    proceed = asyncio.Event()

    async def child():
        await proceed.wait()
        with pytest.raises(ExecutionAuthorityError):
            require_execution_work()
        # A fresh child scope must not discard its stale ancestor.
        with pytest.raises(ExecutionAuthorityError):
            with bind_execution_custody(Authority()):
                pass

    with bind_execution_custody(Authority()) as scope:
        task = asyncio.create_task(child())
        if retirement == "revoke":
            scope.revoke("lost exact authority session")
    proceed.set()
    await task
    require_execution_work()  # No scope was leaked into the owner context.


def test_nested_ancestor_revocation_is_not_masked_by_live_child():
    with bind_execution_custody(Authority()) as parent:
        with bind_execution_custody(Authority()):
            parent.revoke("parent session lost")
            with pytest.raises(ExecutionAuthorityError, match="parent session lost"):
                require_execution_work()


@pytest.mark.asyncio
async def test_foreign_turn_executor_rebinds_same_mutable_authority():
    # This task exists before the turn publishes, like the inline transport.
    ready = asyncio.Event()
    captured = None

    async def foreign_reader():
        await ready.wait()
        with pytest.raises(ExecutionAuthorityError):
            with captured.bind():
                require_execution_work()

    task = asyncio.create_task(foreign_reader())
    with bind_execution_custody(Authority()) as scope:
        captured = capture_turn_scope(object())
        scope.revoke("turn lost authority")
        ready.set()
        await task


def test_repeated_revoke_never_changes_original_reason_or_revives():
    with bind_execution_custody(Authority()) as scope:
        scope.revoke("first loss")
        scope.revoke("second loss")
        with pytest.raises(ExecutionAuthorityError, match="first loss"):
            require_execution_work()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["tool", "named_subagent", "feature_tool"])
async def test_native_dispatch_rechecks_after_awaited_permission_hook(boundary):
    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
    from kestrel_sovereign.hooks.manager import HooksManager

    entered = asyncio.Event()
    release = asyncio.Event()
    executed = []

    class AwaitPermission(Hook):
        def __init__(self):
            super().__init__("permission", [HookEvent.PRE_TOOL_USE])

        async def execute(self, input):
            entered.set()
            await release.wait()
            return HookOutput()

    class Agent(OrchestratorEngineMixin):
        hooks_manager = HooksManager()

    agent = Agent()
    agent.hooks_manager.register(AwaitPermission())

    async def effect(args):
        executed.append(args)
        return {"success": True}

    async def dispatch_native():
        if boundary == "tool":
            return await agent._execute_tool_with_hooks(
                "effect", "Feature", {}, "session", effect,
            )
        if boundary == "named_subagent":
            from types import SimpleNamespace

            async def denied(_name):
                return set()

            async def execute_as_subagent(**args):
                return await effect(args)

            agent._get_denied_tools = denied
            feature = SimpleNamespace(execute_as_subagent=execute_as_subagent)
            return await agent._execute_named_subagent(
                feature, tool_name="feature", args={"task": "work"}, session_id="session", source="test",
            )
        from types import SimpleNamespace
        from kestrel_sovereign.features.base import Feature

        async def execute(**args):
            return await effect(args)

        feature = SimpleNamespace(agent=agent, name="Feature")
        return await Feature._execute_subagent_tool(
            feature, tool_name="effect", args={},
            tools_by_name={"effect": SimpleNamespace(execute=execute)},
        )

    with bind_execution_custody(Authority()) as scope:
        dispatch = asyncio.create_task(dispatch_native())
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            scope.revoke("authority lost during permission await")
            release.set()
            with pytest.raises(ExecutionAuthorityError):
                await dispatch
        finally:
            release.set()
            await asyncio.gather(dispatch, return_exceptions=True)
    assert executed == []


@pytest.mark.asyncio
async def test_native_dispatch_rechecks_after_final_post_hook():
    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
    from kestrel_sovereign.hooks.manager import HooksManager

    with bind_execution_custody(Authority()) as scope:
        class LoseDuringPost(Hook):
            def __init__(self):
                super().__init__("post", [HookEvent.POST_TOOL_USE])

            async def execute(self, input):
                await asyncio.sleep(0)
                scope.revoke("lost during post hook")
                return HookOutput()

        class Agent(OrchestratorEngineMixin):
            hooks_manager = HooksManager()

        agent = Agent()
        agent.hooks_manager.register(LoseDuringPost())

        async def effect(args):
            return {"success": True}

        with pytest.raises(ExecutionAuthorityError, match="lost during post hook"):
            await agent._execute_tool_with_hooks("effect", "Feature", {}, "session", effect)


@pytest.mark.asyncio
async def test_sqlite_connect_refuses_before_creating_path(tmp_path):
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    path = tmp_path / "absent" / "custody.db"
    backend = SQLiteBackend(str(path))
    with bind_execution_custody(Authority()):
        with pytest.raises(ExecutionAuthorityError, match="backend"):
            await backend.connect()
    assert not path.parent.exists()


@pytest.mark.asyncio
async def test_retained_runtime_guards_native_dispatch_without_ambient_scope():
    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
    from kestrel_sovereign.hooks.manager import HooksManager

    class Agent(OrchestratorEngineMixin):
        hooks_manager = HooksManager()

    agent = Agent()
    agent._execution_custody = ExecutionCustody(Authority())
    agent._execution_custody.revoke("runtime retired")
    ran = []

    async def effect(args):
        ran.append(args)

    with pytest.raises(ExecutionAuthorityError, match="runtime retired"):
        await agent._execute_tool_with_hooks("effect", "Feature", {}, "session", effect)
    assert ran == []


@pytest.mark.asyncio
async def test_sqlalchemy_cached_factory_and_retained_connection_refuse_custody():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from kestrel_sovereign.storage.sqla.session import SovereignSqlaSessionFactory

    factory = SovereignSqlaSessionFactory(create_async_engine("sqlite+aiosqlite:///:memory:"))
    try:
        async with factory.engine.connect() as connection:
            await connection.execute(text("CREATE TABLE effects (id INTEGER)"))
            await connection.commit()
            async with factory.read_session() as session:
                with bind_execution_custody(Authority()):
                    with pytest.raises(ExecutionAuthorityError, match="SQLAlchemy"):
                        await session.execute(text("INSERT INTO effects VALUES (1)"))
                    with pytest.raises(ExecutionAuthorityError, match="SQLAlchemy"):
                        await connection.execute(text("INSERT INTO effects VALUES (2)"))
                    with pytest.raises(ExecutionAuthorityError, match="SQLAlchemy"):
                        await connection.commit()
                    # Retirement never needs a fresh work grant.
                    await session.rollback()
                    await connection.rollback()
            assert await connection.scalar(text("SELECT count(*) FROM effects")) == 0
    finally:
        await factory.close()


@pytest.mark.asyncio
async def test_native_dispatch_does_not_publish_result_after_effect_loses_authority():
    from kestrel_sovereign.agent.invocation import _exact_invocation_scope, current_invocation_effect_checkpoint
    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
    from kestrel_sovereign.hooks.manager import HooksManager

    class Agent(OrchestratorEngineMixin):
        hooks_manager = HooksManager()

    agent = Agent()
    with bind_execution_custody(Authority()) as scope:
        async def effect(args):
            await asyncio.sleep(0)
            scope.revoke("effect lost authority")
            return {"success": True, "parts": [{"type": "text", "text": "private"}]}

        with _exact_invocation_scope("effect-loss", None):
            with pytest.raises(ExecutionAuthorityError, match="effect lost authority"):
                await agent._execute_tool_with_hooks(
                    "effect", "Feature", {}, "session", effect,
                )
            state = current_invocation_effect_checkpoint()
            assert state.completed and not state.checkpointed
            assert state.session_id == "session"


@pytest.mark.asyncio
async def test_cleanup_only_children_and_foreign_snapshot_cannot_regain_work():
    from kestrel_sovereign.execution_custody import bind_execution_cleanup, require_execution_backend

    proceed = asyncio.Event()

    async def child():
        await proceed.wait()
        with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
            require_execution_work()

    with bind_execution_custody(Authority()):
        with bind_execution_cleanup(object()):
            task = asyncio.create_task(child())
            captured = capture_turn_scope(object())
            with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
                require_execution_backend("postgres", ())
        require_execution_work()
        with captured.bind():
            with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
                require_execution_work()
        proceed.set()
        await task


@pytest.mark.asyncio
async def test_stream_closed_by_foreign_task_keeps_cleanup_only_runtime_custody():
    from kestrel_sovereign.agent.invocation import bind_async_generator_invocation

    closed = []

    class Owner:
        _execution_custody = ExecutionCustody(Authority())

        @bind_async_generator_invocation("request_id")
        async def stream(self, request_id=None):
            try:
                require_execution_work(self)
                yield "item"
            finally:
                with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
                    require_execution_work(self)
                closed.append(True)

    owner = Owner()
    iterator = owner.stream(request_id="foreign-close")
    assert await anext(iterator) == "item"
    await asyncio.create_task(iterator.aclose())
    assert closed == [True]
    require_execution_work(owner)  # Closing did not revoke the whole runtime.


@pytest.mark.asyncio
async def test_native_feature_return_records_effect_before_post_return_denial():
    from kestrel_sovereign.features.base import Feature
    from kestrel_sovereign.features.base import tool
    from kestrel_sovereign.agent.invocation import invocation_scope, current_invocation_effect_checkpoint

    class Effects(Feature):
        async def initialize(self):
            pass

        @property
        def tool_description(self):
            return "Test completed-effect custody"

        @tool(name="effect", description="Complete an effect")
        async def effect(self):
            scope.revoke("effect returned after revocation")
            return {"success": True}

    feature = Effects.__new__(Effects)
    feature.agent = None
    feature.disabled_skills = set()
    with bind_execution_custody(Authority()) as scope, invocation_scope("native-effect"):
        with pytest.raises(ExecutionAuthorityError):
            await next(tool for tool in feature.get_tools() if tool.name == "effect").execute()
        assert current_invocation_effect_checkpoint().completed


@pytest.mark.asyncio
async def test_foreign_turn_carries_exact_completed_effect_state():
    from contextvars import Context
    from kestrel_sovereign.agent.invocation import invocation_scope, current_invocation_effect_checkpoint, mark_current_invocation_effect_completed

    async def foreign(captured):
        with captured.bind():
            mark_current_invocation_effect_completed("foreign-session")

    with invocation_scope("foreign-effect"):
        state = current_invocation_effect_checkpoint()
        captured = capture_turn_scope(object())
        await asyncio.create_task(foreign(captured), context=Context())
        assert state.completed
        assert state.session_id == "foreign-session"


@pytest.mark.asyncio
async def test_foreign_stream_cannot_drop_retired_prior_admission():
    from contextvars import Context
    from kestrel_sovereign.agent.invocation import bind_async_generator_invocation

    advanced = []
    closed = []

    class Owner:
        _execution_custody = ExecutionCustody(Authority())

        @bind_async_generator_invocation("request_id")
        async def stream(self, request_id=None):
            try:
                yield "first"
                advanced.append(True)
                yield "private"
            finally:
                with pytest.raises(ExecutionAuthorityError, match="cleanup-only"):
                    require_execution_work(self)
                closed.append(True)

    iterator = Owner().stream(request_id="pinned-stream")
    with bind_execution_custody(Authority()):
        assert await anext(iterator) == "first"
    with pytest.raises(ExecutionAuthorityError):
        await asyncio.create_task(anext(iterator), context=Context())
    assert advanced == []
    assert closed == [True]


@pytest.mark.asyncio
async def test_parent_invocation_rechecks_custody_after_child_completion():
    from kestrel_sovereign.agent.invocation import bind_async_invocation

    class Owner:
        _execution_custody = ExecutionCustody(Authority())

        def register_active_request(self, request_id, *, nested=False):
            pass

        async def await_durable_request_admission(self, request_id):
            return True

        def bind_request_operation(self, request_id, operation):
            operation.add_done_callback(lambda task: self._execution_custody.revoke("lost before parent publication"))

        def _cleanup_cancelled_request(self, request_id, **kwargs):
            pass

        @bind_async_invocation("request_id", track_request_lifecycle=True)
        async def turn(self, request_id=None):
            return "private"

    with pytest.raises(ExecutionAuthorityError, match="lost before parent publication"):
        await Owner().turn(request_id="parent-publication")


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["before", "during", "cleanup"])
async def test_provider_attempt_never_runs_or_publishes_after_custody_loss(loss):
    from kestrel_sovereign.execution_custody import bind_execution_cleanup
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext

    service = LLMService.__new__(LLMService)
    calls = []
    finalized = []

    async def finalize(*args, **kwargs):
        finalized.append(True)

    service._finalize_successful_invocation = finalize
    service._finalize_failed_invocation = finalize

    async def provider():
        calls.append(True)
        if loss == "during":
            scope.revoke("provider custody lost")
            raise ConnectionError("first provider failed")
        return "private"

    with bind_execution_custody(Authority()) as scope:
        from contextlib import nullcontext
        if loss == "before":
            scope.revoke("provider custody lost")
        with bind_execution_cleanup(service) if loss == "cleanup" else nullcontext():
            with pytest.raises(ExecutionAuthorityError):
                await service._run_provider_attempt(provider(), "local-test", "test", path="test", invocation_context=LLMInvocationContext())
    assert calls == ([True] if loss == "during" else [])
    assert finalized == []


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["before", "during"])
async def test_native_stream_provider_refuses_denied_dispatch_and_chunks(loss):
    from kestrel_sovereign.llm.streaming import StreamingMixin

    called = []
    class Adapter:
        async def get_streaming_response(self, **kwargs):
            called.append(True)
            scope.revoke("stream provider lost custody")
            yield "private"

    class Service(StreamingMixin):
        async def _record_streamed_usage(self, *args, **kwargs):
            pass

    service = Service()
    with bind_execution_custody(Authority()) as scope:
        if loss == "before":
            scope.revoke("stream provider lost custody")
        iterator = service._stream_adapter_with_usage(
            adapter=Adapter(), client=None, model="local-test", messages=[],
            provider_name="local-test", path="test", invocation_context=None,
            expose_protocol_events=False,
        )
        try:
            with pytest.raises(ExecutionAuthorityError):
                await anext(iterator)
        finally:
            await iterator.aclose()
    assert called == ([True] if loss == "during" else [])


def test_llm_retains_native_runtime_custody_without_ambient_context():
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService.__new__(LLMService)
    service._owner_agent_did = None
    custody = ExecutionCustody(Authority())
    service.attach_to_agent("did:example:runtime", execution_custody=custody)
    custody.revoke("native provider runtime retired")
    with pytest.raises(ExecutionAuthorityError, match="runtime retired"):
        service._check_policy()
    with pytest.raises(ExecutionAuthorityError, match="discard"):
        service.attach_to_agent("did:example:runtime")


@pytest.mark.asyncio
async def test_hosted_scheduler_preparation_binds_resolved_runtime_for_renewal():
    from types import SimpleNamespace
    from kestrel_sovereign.features.scheduler.runner import HostedSchedulerExecutor, SchedulerExecution
    from kestrel_sovereign.execution_custody import current_execution_custody

    custody = ExecutionCustody(Authority())
    async def effect(*args):
        require_execution_work()

    agent = SimpleNamespace(_execution_custody=custody, features={"SchedulerFeature": SimpleNamespace(enabled=True, _dispatch_scheduled_task=effect)})
    async def resolve(did):
        return agent

    executor = HostedSchedulerExecutor(resolve)
    execution = SchedulerExecution("execution", "schedule", "did:example:runtime", "effect", {}, "occurrence", "idempotency", 1, "owner")
    async with executor.prepare_scheduled(execution):
        assert current_execution_custody() == (custody,)
        custody.revoke("resolved scheduler runtime retired")
        with pytest.raises(ExecutionAuthorityError):
            require_execution_work()


def test_manager_resolves_independent_host_custody_before_agent_boot(tmp_path):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager, LocalAgentConfig

    first, second = ExecutionCustody(Authority()), ExecutionCustody(Authority())
    seen = []
    def factory(name, did, config):
        seen.append((name, did, config))
        return first if did == "did:first" else second

    manager = AgentManager(base_data_dir=tmp_path, execution_custody_factory=factory)
    config = LocalAgentConfig(data_dir=tmp_path / "agent", port=8881)
    assert manager._execution_custody_for_agent("first", "did:first", config) is first
    assert manager._execution_custody_for_agent("second", "did:second", config) is second
    assert len(seen) == 2
    first.revoke("first runtime retired")
    with pytest.raises(ExecutionAuthorityError):
        manager._execution_custody_for_agent("first", "did:first", config)
    assert manager._execution_custody_for_agent("second", "did:second", config) is second


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["before", "during", "cleanup", "discovery"])
async def test_native_decision_provider_preserves_runtime_custody(loss):
    from contextlib import nullcontext
    from kestrel_sovereign.execution_custody import bind_execution_cleanup
    from tests.unit.test_decisions_service import FakeDecisionAdapter, _service, _route, _request, _answer_body

    custody = ExecutionCustody(Authority())
    class Adapter(FakeDecisionAdapter):
        async def list_decision_models(self, client):
            result = await super().list_decision_models(client)
            if loss == "discovery":
                custody.revoke("decision runtime retired during discovery")
            return result

        async def adecide(self, *args, **kwargs):
            self.decide_calls.append("effect")
            if loss == "during":
                custody.revoke("decision runtime retired during provider call")
            return _answer_body()

    adapter = Adapter(["local-test"])
    service = _service([_route("ollama:local", adapter, local=True)])
    service._execution_custody = custody
    if loss == "before":
        custody.revoke("decision runtime already retired")
    with bind_execution_cleanup(service) if loss == "cleanup" else nullcontext():
        with pytest.raises(ExecutionAuthorityError):
            await service.decide(_request(), caller="custody-test", timeout_seconds=5)
    assert adapter.decide_calls == (["effect"] if loss == "during" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["before", "during", "cleanup"])
async def test_native_embedding_service_preserves_runtime_custody(loss):
    from contextlib import nullcontext
    from unittest.mock import AsyncMock
    from kestrel_sovereign.execution_custody import bind_execution_cleanup
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
    from tests.unit.test_embedding_usage_recording import UsageReportingAdapter, _provider

    custody = ExecutionCustody(Authority())
    adapter = UsageReportingAdapter()
    def inside_provider():
        from kestrel_sovereign.execution_custody import current_execution_custody
        assert current_execution_custody() == (custody,)
        if loss == "during":
            custody.revoke("embedding runtime retired during provider call")
    adapter.during_call = inside_provider
    service = LLMService.__new__(LLMService)
    service._execution_custody = custody
    service.snapshot_invocation_context = lambda: LLMInvocationContext()
    service.record_modality_call = AsyncMock()
    embedding = service._new_embedding_service(_provider(adapter))
    if loss == "before":
        custody.revoke("embedding runtime already retired")
    with bind_execution_cleanup(service) if loss == "cleanup" else nullcontext():
        with pytest.raises(ExecutionAuthorityError):
            await embedding.aembed("synthetic local fixture")
    assert adapter.texts == (["synthetic local fixture"] if loss == "during" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["before", "during", "backoff"])
async def test_native_provider_retry_never_reopens_denied_work(monkeypatch, loss):
    from kestrel_sovereign.llm import retry

    calls = []
    class Unavailable(Exception):
        status_code = 503

    async def provider():
        calls.append(True)
        if len(calls) == 1:
            if loss == "during":
                scope.revoke("provider retry admission lost")
            raise Unavailable("temporary provider failure")
        return "private"

    async def wait(_delay):
        scope.revoke("provider retry admission lost during backoff")

    monkeypatch.setattr(retry.asyncio, "sleep", wait)
    with bind_execution_custody(Authority()) as scope:
        if loss == "before":
            scope.revoke("provider retry admission already lost")
        with pytest.raises(ExecutionAuthorityError):
            await retry.with_retry(provider, max_retries=2, base_delay=0.01, max_delay=1)
    assert calls == ([] if loss == "before" else [True])


@pytest.mark.asyncio
async def test_commit_uncertainty_preserves_invocation_without_completed_return():
    from kestrel_sovereign.agent.invocation import bind_async_invocation
    from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    dispositions = []
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
            try:
                raise ExecutionCommitOutcomeError("unknown")
            except Exception as error:
                raise RuntimeError("legacy database wrapper") from error
    with pytest.raises(RuntimeError, match="legacy database wrapper"):
        await Owner().turn(request_id="commit-uncertain")
    assert dispositions == [RequestCompletionDisposition.ABANDONED]


@pytest.mark.asyncio
@pytest.mark.parametrize("loss", ["before", "during"])
async def test_stream_admission_is_inside_retained_runtime_custody(loss):
    from kestrel_sovereign.agent.invocation import bind_async_generator_invocation
    from kestrel_sovereign.execution_custody import current_execution_custody
    registered, bodies, admissions = [], [], []
    class Owner:
        _execution_custody = ExecutionCustody(Authority())
        def register_active_request(self, request_id):
            registered.append(request_id)
        async def await_durable_request_admission(self, request_id):
            admissions.append(current_execution_custody(self))
            self._execution_custody.revoke("lost while admitting")
            return True
        def _cleanup_cancelled_request(self, request_id, **kwargs):
            pass
        @bind_async_generator_invocation("request_id", track_request_lifecycle=True)
        async def stream(self, request_id=None):
            bodies.append(True)
            yield "private"
    owner = Owner()
    if loss == "before":
        owner._execution_custody.revoke("already retired")
    with pytest.raises(ExecutionAuthorityError):
        await anext(owner.stream(request_id="stream-admission"))
    assert bodies == []
    assert registered == ([] if loss == "before" else ["stream-admission"])
    assert admissions == ([] if loss == "before" else [(owner._execution_custody,)])


@pytest.mark.asyncio
async def test_provider_finalization_carries_original_runtime_to_children():
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
    from kestrel_sovereign.execution_custody import current_execution_custody
    service = LLMService.__new__(LLMService)
    service._execution_custody = ExecutionCustody(Authority())
    async def provider():
        return "private"
    async def finalize(*args, **kwargs):
        assert current_execution_custody() == (service._execution_custody,)
        service._execution_custody.revoke("lost during finalization")
        async def separate_sink():
            require_execution_work()
        await asyncio.create_task(separate_sink())
    service._finalize_successful_invocation = finalize
    with pytest.raises(ExecutionAuthorityError, match="lost during finalization"):
        await service._run_provider_attempt(provider(), "local-test", "test", path="test", invocation_context=LLMInvocationContext())


@pytest.mark.asyncio
async def test_native_llm_stream_retains_every_foreign_consumer_admission():
    from contextvars import Context
    from kestrel_sovereign.llm.streaming import StreamingMixin
    advanced = []
    class Adapter:
        async def get_streaming_response(self, **kwargs):
            for item in ("first", "second", "private"):
                advanced.append(item)
                yield item
    class Service(StreamingMixin):
        async def _record_streamed_usage(self, *args, **kwargs):
            pass
    stream = Service()._stream_adapter_with_usage(adapter=Adapter(), client=None, model="local-test", messages=[], provider_name="local-test", path="test", invocation_context=None, expose_protocol_events=False)
    assert await anext(stream) == "first"
    with bind_execution_custody(Authority()):
        assert await anext(stream) == "second"
    with pytest.raises(ExecutionAuthorityError):
        await asyncio.create_task(anext(stream), context=Context())
    assert advanced == ["first", "second"]


@pytest.mark.asyncio
async def test_codex_internal_retry_refuses_second_dispatch_after_loss(monkeypatch):
    from tests.unit.test_codex_retry_on_idle_timeout import _stub_adapter, _transport_error
    from kestrel_sovereign.llm import codex_adapter
    adapter = _stub_adapter()
    calls = []
    async def turn(*args, **kwargs):
        calls.append(True)
        raise _transport_error()
        yield
    async def backoff(*args):
        scope.revoke("Codex retry admission retired")
    adapter._run_turn = turn
    monkeypatch.setattr(codex_adapter.asyncio, "sleep", backoff)
    with bind_execution_custody(Authority()) as scope:
        with pytest.raises(ExecutionAuthorityError, match="retry admission retired"):
            async for _ in adapter._run_turn_with_retry("test", [], None, None, None):
                pass
    assert calls == [True]


@pytest.mark.asyncio
async def test_codex_transport_reader_does_not_inherit_first_occurrence(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock, MagicMock
    from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
    from kestrel_sovereign.execution_custody import current_execution_custody
    monkeypatch.setenv("HOME", str(tmp_path))
    client = CodexAppServerClient.__new__(CodexAppServerClient)
    client._binary = "/test/no-provider"
    client._closed_error = None
    observed = []
    async def reader():
        observed.append(current_execution_custody())
        with bind_execution_custody(Authority()):
            require_execution_work()
    client._read_loop = reader
    client._drain_stderr = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=MagicMock()))
    with bind_execution_custody(Authority()):
        await client._spawn()
    await asyncio.gather(client._reader_task, client._stderr_task)
    assert observed == [()]


@pytest.mark.asyncio
async def test_public_stream_does_not_fallback_after_original_admission_loss():
    from kestrel_sovereign.llm.streaming import StreamingMixin
    calls = []
    class Adapter:
        async def get_streaming_response(self, **kwargs):
            calls.append("first")
            scope.revoke("stream admission lost before fallback")
            raise ConnectionError("ordinary adapter failure")
            yield
    class Service(StreamingMixin):
        def _check_policy(self):
            require_execution_work(self)
        async def _record_streamed_usage(self, *args, **kwargs):
            pass
        async def _get_streaming_response_frozen(self, **kwargs):
            try:
                async for item in self._stream_adapter_with_usage(adapter=Adapter(), client=None, model="test", messages=[], provider_name="local-test", path="test", invocation_context=None, expose_protocol_events=False):
                    yield item
            except ConnectionError:
                calls.append("fallback")
                yield "private"
    with bind_execution_custody(Authority()) as scope:
        with pytest.raises(ExecutionAuthorityError):
            await anext(Service().get_streaming_response("system", "prompt"))
    assert calls == ["first"]


@pytest.mark.asyncio
async def test_aborted_hosted_stream_cannot_issue_usage_or_billing_writes():
    from tests.unit.test_streaming_usage_metering import _FakeService
    from kestrel_sovereign.llm.adapter import LLMResponse
    service = _FakeService()
    service._execution_custody = ExecutionCustody(Authority())
    class Adapter:
        supports_partial_usage_flush = True
        async def get_streaming_response_with_tools(self, **kwargs):
            kwargs["usage_sink"]["input_tokens"] = 12
            yield "first"
            yield LLMResponse(input_tokens=12, output_tokens=1)
    stream = service._stream_adapter_with_usage(adapter=Adapter(), client=None, model="test", messages=[], provider_name="local-test", path="test", invocation_context=None, expose_protocol_events=False)
    assert await anext(stream) == "first"
    await stream.aclose()
    service._track_model_usage.assert_not_awaited()
    service._log_llm_call.assert_not_awaited()
