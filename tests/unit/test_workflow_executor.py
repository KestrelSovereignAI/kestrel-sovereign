"""Workflows execute real decorated tools through native primary task storage."""

import asyncio
import contextlib
import inspect
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from kestrel_sdk.features.base import Feature as SDKFeature
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.native_a2a.types import Task, TaskState, TaskStatus
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sdk.tools import DeferredToolResult
from kestrel_sdk.tools.base import AgentTool, ToolCategory, ToolSchema
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign.a2a.lifecycle import (
    mark_outcome_unknown,
    task_outcome_unknown,
    task_state_label,
)
from kestrel_sovereign.a2a.projections import embedded_json_value, message_text
from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.features.base import Feature, observed_outcome_unknown, tool
from kestrel_sovereign.features.subagent_dispatch import ensure_subagent_dispatch
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.storage.db.interface import QueryError
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.tools.effect_journal import (
    admit_tool_effect_turn, bind_tool_effect_turn, current_tool_effect_turn,
)
from tests.utils.native_a2a import manager_for
from tests.utils.codex_inline import PreTurnReader


@pytest_asyncio.fixture
async def manager(native_a2a_backend_factory, native_tool_effect_agent):
    """Drive the task component with real native host and journal ownership.

    Each top-level skill call is a new component input. Nested skills share
    its original journal. The driver may inspect a persisted unknown Task
    after root retirement refuses completion; it never calls that a successful
    invocation. A missing unknown Task receipt still propagates the terminal
    failure. Full invocation delivery is tested by the tool-effect suites.
    """
    from uuid import uuid4

    backend = native_a2a_backend_factory()
    await backend.connect()
    host = await native_tool_effect_agent(did='did:kestrel:recipient', connected_backend=backend)
    instance = manager_for(backend, NativeRecipientScope(host.did, 'tenant'), owns_backend=False)
    await instance.initialize()
    host.task_manager = instance
    host.observability_store = instance.observability_store
    instance._workflow_host = host
    instance._native_terminal_errors = []
    execute_skill = instance.execute_skill

    async def drive(agent_id, skill_id, args, sync=True, session_id=None):
        if current_tool_effect_turn(host) is not None:
            return await execute_skill(agent_id, skill_id, args, sync=sync, session_id=session_id)
        operation_id = uuid4().hex
        turn = await admit_tool_effect_turn(host, namespace=['workflow-component'],
            client_key=operation_id, request=dict(provider=agent_id, skill=skill_id, arguments=args),
            invocation_id=operation_id)
        with bind_tool_effect_turn((turn, None)):
            try:
                task = await execute_skill(agent_id, skill_id, args, sync=sync, session_id=session_id)
            except BaseException as error:
                await turn.close(error)
                raise AssertionError('Native root retirement discarded an original error')
            try:
                await turn.close()
            except ResourceOutcomeUnknown as terminal:
                assert task_outcome_unknown(task), 'Root uncertainty lacked the original native Task receipt'
                instance._native_terminal_errors.append(terminal)
            return task

    instance.execute_skill = drive
    try:
        yield instance
    finally:
        await instance.close()


@pytest.fixture
def workflow_host(manager):
    return manager._workflow_host


class WorkflowProvider(Feature):
    @property
    def tool_description(self):
        return "Real local workflow tools"

    async def initialize(self):
        pass


class ModelProvider(WorkflowProvider):
    @tool(name="list_models", description="List deterministic test models")
    async def list_models(self, provider: str = "all") -> dict:
        return {"models": ["model-a", "model-b"], "provider": provider}

    @tool(name="get_current_model", description="Read the selected test model")
    async def get_current_model(self) -> dict:
        return {"model": "model-a", "provider": "test-provider"}


class MemoryProvider(WorkflowProvider):
    @tool(name="memory_status", description="Return the selected real tool outcome")
    async def memory_status(self, context: dict | None = None):
        self.calls += 1
        result = self.supplier(self.calls)
        if inspect.isawaitable(result):
            result = await result
        return result if context is None else {"context": context, "value": result}


class WalletProvider(WorkflowProvider):
    @tool(name="check_balance", description="Read a deterministic test balance")
    async def check_balance(self) -> dict:
        return {"balance": "12.50", "currency": "TEST"}


@pytest.fixture
def providers(manager, workflow_host):
    agent = workflow_host
    model, memory, wallet = (
        ModelProvider(agent),
        MemoryProvider(agent),
        WalletProvider(agent),
    )
    memory.calls = 0
    memory.supplier = lambda _attempt: {"episodes": 42, "total_tokens": 150000}
    for name, provider in (
        ("model_agent", model),
        ("memory_feature", memory),
        ("wallet_feature", wallet),
    ):
        provider.name = name
        manager.register_provider(name, provider.get_agent_skills(), provider)
    return SimpleNamespace(model=model, memory=memory, wallet=wallet)


@pytest.fixture
def task_feature(manager, providers, workflow_host):
    feature = TaskFeature(workflow_host)
    feature.set_task_manager(manager)
    return feature


def payload(result, index=0):
    return result.data["results"][index]["result"]["result"]


async def test_single_step_retains_an_official_task_and_result(task_feature, manager):
    result = await task_feature.run_workflow(
        [{"feature": "model_agent", "skill": "get_current_model"}]
    )
    assert result.status is ToolResultStatus.OK
    assert result.data["workflow_steps"] == result.data["completed"] == 1
    assert isinstance(result.data["results"][0]["result"], dict)
    assert result.data["failed"] == 0 and payload(result)["model"] == "model-a"
    [(task_id,)] = await manager.task_store.backend.fetch_all(
        "SELECT id FROM a2a_tasks"
    )
    task = await manager.get_task_for_recipient(task_id, manager.did)
    assert (
        isinstance(task, Task) and task.status.state == TaskState.TASK_STATE_COMPLETED
    )
    recognized, value = embedded_json_value(task.artifacts[0].parts[0])
    assert recognized and value == result.data["results"][0]["result"]


async def test_multi_step_workflow_runs_in_order_and_retains_metadata(task_feature):
    result = await task_feature.run_workflow(
        [
            {"feature": "model_agent", "skill": "get_current_model"},
            {"feature": "memory_feature", "skill": "memory_status"},
            {"feature": "wallet_feature", "skill": "check_balance"},
        ]
    )
    assert result.status is ToolResultStatus.OK
    assert result.data["workflow_steps"] == result.data["completed"] == 3
    assert result.data["failed"] == 0
    assert payload(result, 0)["model"] == "model-a"
    assert payload(result, 1)["episodes"] == 42
    assert payload(result, 2)["balance"] == "12.50"
    assert [
        (step["step"], step["feature"], step["skill"])
        for step in result.data["results"]
    ] == [
        (0, "model_agent", "get_current_model"),
        (1, "memory_feature", "memory_status"),
        (2, "wallet_feature", "check_balance"),
    ]
    assert (
        isinstance(result.data["total_duration_ms"], int)
        and result.data["total_duration_ms"] >= 0
    )
    assert all(
        isinstance(step["duration_ms"], int) and step["duration_ms"] >= 0
        for step in result.data["results"]
    )


@pytest.mark.parametrize(
    "arguments", [{"provider": "selected"}, '{"provider":"selected"}']
)
async def test_object_and_json_encoded_tool_arguments(task_feature, arguments):
    result = await task_feature.run_workflow(
        [{"feature": "model_agent", "skill": "list_models", "args": arguments}]
    )
    assert (
        result.status is ToolResultStatus.OK
        and payload(result)["provider"] == "selected"
    )


@pytest.mark.parametrize("reference", ["{{steps.0.result}}", "{{prev.result}}"])
async def test_references_are_resolved_into_typed_native_arguments(
    task_feature, reference
):
    result = await task_feature.run_workflow(
        [
            {"feature": "model_agent", "skill": "get_current_model"},
            {
                "feature": "memory_feature",
                "skill": "memory_status",
                "args": {"context": reference},
            },
        ]
    )
    assert result.status is ToolResultStatus.OK
    assert payload(result, 1)["context"] == result.data["results"][0]["result"]


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], "list"),
        (None, "NoneType"),
        ('["not","an","object"]', "str decoding to list"),
        ("{not json", "str: invalid JSON"),
    ],
)
async def test_invalid_arguments_keep_the_step_target_and_reason(
    task_feature, arguments, expected
):
    result = await task_feature.run_workflow(
        [{"feature": "model_agent", "skill": "list_models", "args": arguments}]
    )
    assert result.status is ToolResultStatus.ERROR and result.data["failed"] == 1
    step = result.data["results"][0]
    assert (
        step["status"] == "failed"
        and "Step 0 (model_agent.list_models)" in step["error"]
    )
    assert expected in step["error"]


async def test_unknown_provider_has_a_clear_failure(task_feature):
    result = await task_feature.run_workflow(
        [{"feature": "missing", "skill": "missing"}]
    )
    assert result.status is ToolResultStatus.ERROR and result.data["failed"] == 1
    assert "Unknown agent: missing" in result.data["results"][0]["error"]


async def test_native_skill_rejection_keeps_its_reason(task_feature, manager):
    result = await task_feature.run_workflow(
        [{"feature": "model_agent", "skill": "missing"}]
    )
    assert result.status is ToolResultStatus.ERROR and result.data["failed"] == 1
    step = result.data["results"][0]
    assert step["status"] == "failed"
    assert "Unknown skill" in step.get("error", "")
    assert await manager.task_store.backend.fetch_all(
        "SELECT status FROM a2a_tasks"
    ) == [("rejected",)]


async def test_failure_does_not_hide_the_successful_steps(task_feature):
    result = await task_feature.run_workflow(
        [
            {"feature": "model_agent", "skill": "get_current_model"},
            {"feature": "missing", "skill": "missing"},
            {"feature": "wallet_feature", "skill": "check_balance"},
        ]
    )
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["completed"] == 2 and result.data["failed"] == 1
    assert [step["status"] for step in result.data["results"]] == [
        "completed",
        "failed",
        "completed",
    ]


@pytest.mark.parametrize("steps", [[], None, "invalid"])
async def test_invalid_workflow_input_is_rejected(task_feature, steps):
    result = await task_feature.run_workflow(steps)
    assert result.status is ToolResultStatus.ERROR and "non-empty list" in result.error


@pytest.mark.parametrize(
    "step", ["invalid", {"feature": "model_agent"}, {"skill": "list_models"}]
)
async def test_invalid_step_keeps_a_clear_failure(task_feature, step):
    result = await task_feature.run_workflow([step])
    assert result.status is ToolResultStatus.ERROR and result.data["failed"] == 1
    assert result.data["results"][0]["error"]


@pytest.mark.parametrize("method", ["list_available_skills", "run_workflow"])
async def test_a_missing_manager_is_an_explicit_failure(method):
    feature = TaskFeature(agent=None)
    result = await getattr(feature, method)(*([[]] if method == "run_workflow" else []))
    assert result.status is ToolResultStatus.ERROR and "not available" in result.error


@pytest.mark.parametrize(
    "outcome,expected,reason",
    [
        (ToolResult.failed("Unavailable"), ToolResultStatus.ERROR, "Unavailable"),
        (
            ToolResult.partial(confirmation="Some data", error="Incomplete"),
            ToolResultStatus.PARTIAL,
            "Incomplete",
        ),
        (
            {"success": False, "error": "Tool refused the operation"},
            ToolResultStatus.ERROR,
            "Tool refused the operation",
        ),
    ],
)
async def test_returned_semantic_failure_is_not_reported_as_success(
    task_feature, providers, manager, outcome, expected, reason
):
    providers.memory.supplier = lambda _attempt: outcome
    result = await task_feature.run_workflow(
        [{"feature": "memory_feature", "skill": "memory_status"}]
    )
    assert result.status is expected
    assert reason in result.data["results"][0]["error"]
    # Explicit ToolResult errors produce native FAILED receipts. Partial and
    # dictionary results retain their business semantics in the artifact.
    native_status = (
        "failed"
        if isinstance(outcome, ToolResult) and outcome.status is ToolResultStatus.ERROR
        else "completed"
    )
    assert await manager.task_store.backend.fetch_all(
        "SELECT status FROM a2a_tasks"
    ) == [(native_status,)]
    assert providers.memory.calls == 1


async def test_raised_tool_error_is_reported_without_stopping_other_steps(
    task_feature, providers
):
    def unavailable(_attempt):
        raise OSError("Tool storage unavailable")

    providers.memory.supplier = unavailable
    result = await task_feature.run_workflow(
        [
            {"feature": "memory_feature", "skill": "memory_status"},
            {"feature": "model_agent", "skill": "get_current_model"},
        ]
    )
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["results"][0]["outcome_unknown"] is True
    assert result.data["results"][0]["error"] == "tool_effect_outcome_unconfirmed"
    assert result.data["completed"] == result.data["failed"] == 1


async def test_explicit_retry_reports_the_attempt_that_succeeded(
    task_feature, providers
):
    providers.memory.supplier = lambda attempt: (
        ToolResult.failed("Retry")
        if attempt == 1
        else ToolResult.ok("Ready", data={"attempt": attempt})
    )
    result = await task_feature.run_workflow(
        [
            {
                "feature": "memory_feature",
                "skill": "memory_status",
                "max_retries": 1,
                "retry_delay_ms": 0,
            }
        ]
    )
    assert result.status is ToolResultStatus.OK and providers.memory.calls == 2
    assert len(result.data["results"]) == 1
    assert result.data["results"][0]["attempts"] == 2
    assert result.data["results"][0]["result"]["data"]["attempt"] == 2


async def test_execution_control_keeps_its_original_identity_and_is_not_retried(
    task_feature, providers
):
    denial = ExecutionAuthorityError("Original execution authority was withdrawn")

    def deny(_attempt):
        raise denial

    providers.memory.supplier = deny
    with pytest.raises(ExecutionAuthorityError) as observed:
        await task_feature.run_workflow(
            [
                {
                    "feature": "memory_feature",
                    "skill": "memory_status",
                    "max_retries": 2,
                    "retry_delay_ms": 0,
                }
            ]
        )
    assert observed.value is denial and providers.memory.calls == 1


UNKNOWN = "Effect may already have been applied; reconcile before retrying"

# A skill reaches the task boundary as a decorated @tool method (through the
# Core or the SDK wrapper, which return its error as a plain failure) or as the
# feature's own AgentTool (which raises to the boundary). A decorated method is
# a coroutine function, or a plain one returning an awaitable (both wrappers
# await the result), which can raise before returning it. Every form must carry
# an unknown outcome to its task.
EVERY_SKILL_FORM = [
    ("core-feature", "decorated"),
    ("core-feature", "plain-decorated"),
    ("core-feature", "own-tool"),
    ("sdk-feature", "decorated"),
    ("sdk-feature", "plain-decorated"),
    ("sdk-feature", "own-tool"),
]


def skill_form_id(param):
    return "-".join(param)


class EffectSkill(AgentTool):
    """A feature's own skill: it raises to the task boundary unwrapped."""

    def __init__(self, provider):
        self.provider = provider

    @property
    def name(self):
        return "apply_effect"

    @property
    def schema(self):
        return ToolSchema(
            name="apply_effect",
            description="Apply a counted test effect",
            category=ToolCategory.UTILITY,
            parameters=[],
        )

    async def execute(self, **kwargs):
        self.provider.calls += 1
        return await supplied(self.provider)


async def supplied(provider):
    result = provider.supplier(provider.calls)
    return await result if inspect.isawaitable(result) else result


async def completed(value):
    return value


def effect_provider(base, form):
    class EffectProvider(base):
        tool_description = "Applies a counted effect"

        async def initialize(self):
            pass

    if form == "own-tool":

        class OwnToolProvider(EffectProvider):
            def get_tools(self):
                return [EffectSkill(self)]

        return OwnToolProvider

    if form == "plain-decorated":

        class PlainDecoratedProvider(EffectProvider):
            @tool(name="apply_effect", description="Apply a counted test effect")
            def apply_effect(self) -> dict:
                self.calls += 1
                result = self.supplier(self.calls)
                return result if inspect.isawaitable(result) else completed(result)

        return PlainDecoratedProvider

    class DecoratedProvider(EffectProvider):
        @tool(name="apply_effect", description="Apply a counted test effect")
        async def apply_effect(self) -> dict:
            self.calls += 1
            return await supplied(self)

    return DecoratedProvider


@pytest.fixture
def effect(request, manager, workflow_host):
    """A skill with a counted effect, in the form ``request.param`` names."""
    base_name, form = request.param
    base = Feature if base_name == "core-feature" else SDKFeature
    agent = workflow_host
    provider = ensure_subagent_dispatch(effect_provider(base, form))(agent)
    provider.name = "effect_feature"
    provider.calls = 0
    provider.supplier = lambda _attempt: {"applied": True}
    manager.register_provider("effect_feature", provider.get_agent_skills(), provider)
    return provider


def effect_step(max_retries):
    return {
        "feature": "effect_feature",
        "skill": "apply_effect",
        "max_retries": max_retries,
        "retry_delay_ms": 0,
    }


async def stored_tasks(manager):
    rows = await manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks")
    return [await manager.get_task_for_recipient(task_id, manager.did) for (task_id,) in rows]


def raise_unknown(_attempt):
    raise ResourceOutcomeUnknown(UNKNOWN)


def raise_unavailable(_attempt):
    raise OSError("Effect storage unavailable")


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_a_step_whose_outcome_is_unknown_is_never_retried(
    task_feature, effect, manager
):
    effect.supplier = raise_unknown
    result = await task_feature.run_workflow([effect_step(max_retries=2)])
    assert effect.calls == 1
    assert result.status is ToolResultStatus.ERROR
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1 and record["error"] == UNKNOWN
    # The one native receipt stays FAILED with the skill's own message.
    [task] = await stored_tasks(manager)
    assert task.status.state == TaskState.TASK_STATE_FAILED
    assert message_text(task.status.message) == UNKNOWN
    assert task_outcome_unknown(task)


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_an_ordinary_raised_failure_is_not_retried_after_native_admission(task_feature, effect, manager):
    effect.supplier = raise_unavailable
    result = await task_feature.run_workflow([effect_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["attempts"] == 1
    assert record["outcome_unknown"] is True
    tasks = await stored_tasks(manager)
    assert len(tasks) == 1
    assert all(task.status.state == TaskState.TASK_STATE_FAILED for task in tasks)
    assert any(task_outcome_unknown(task) for task in tasks)


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
@pytest.mark.parametrize(
    "supplier, state",
    [
        (lambda _attempt: {"applied": True}, TaskState.TASK_STATE_COMPLETED),
        (lambda _attempt: ToolResult.failed("Refused"), TaskState.TASK_STATE_FAILED),
        (raise_unavailable, TaskState.TASK_STATE_FAILED),
    ],
    ids=["completed", "returned-failure", "raised-failure"],
)
async def test_only_a_post_admission_exception_marks_the_native_task(
    effect, manager, supplier, state
):
    effect.supplier = supplier
    task = await manager.execute_skill("effect_feature", "apply_effect", {}, sync=True)
    assert task.status.state == state
    assert task_outcome_unknown(task) is (supplier is raise_unavailable)
    [stored] = await stored_tasks(manager)
    assert stored.status.state == state
    assert task_outcome_unknown(stored) is (supplier is raise_unavailable)


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_an_unknown_outcome_marks_the_stored_task(effect, manager):
    effect.supplier = raise_unknown
    task = await manager.execute_skill("effect_feature", "apply_effect", {}, sync=True)
    assert task.status.state == TaskState.TASK_STATE_FAILED
    assert task_outcome_unknown(task)
    [stored] = await stored_tasks(manager)
    assert task_outcome_unknown(stored)
    assert message_text(stored.status.message) == UNKNOWN


@pytest.mark.parametrize(
    "effect",
    [
        (base, form)
        for base in ("core-feature", "sdk-feature")
        for form in ("decorated", "plain-decorated")
    ],
    ids=skill_form_id,
    indirect=True,
)
async def test_a_durable_tool_wrapper_preserves_the_original_unknown(effect):
    # A durable host owns normalization and must receive original evidence.
    effect.supplier = raise_unknown
    [skill] = effect.get_tools()
    with pytest.raises(ResourceOutcomeUnknown, match=UNKNOWN):
        await skill.execute()
    assert effect.calls == 1


async def test_an_unknown_outcome_raised_to_the_workflow_is_not_retried(
    task_feature, manager, monkeypatch
):
    calls = []

    async def execute_skill(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OSError("Transient task storage failure")
        raise ResourceOutcomeUnknown(UNKNOWN)

    monkeypatch.setattr(manager, "execute_skill", execute_skill)
    result = await task_feature.run_workflow(
        [
            {
                "feature": "memory_feature",
                "skill": "memory_status",
                "max_retries": 3,
                "retry_delay_ms": 0,
            }
        ]
    )
    assert len(calls) == 2
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 2 and record["error"] == UNKNOWN


async def raise_unknown_from_a_task_group(_attempt):
    """Run the effect concurrently, so its unknown outcome arrives grouped."""

    async def uncertain():
        raise ResourceOutcomeUnknown(UNKNOWN)

    async with asyncio.TaskGroup() as group:
        group.create_task(uncertain())


def raise_unknown_in_a_nested_group(_attempt):
    raise ExceptionGroup(
        "Effects failed",
        [
            OSError("Other effect unavailable"),
            ExceptionGroup("Effect failed", [ResourceOutcomeUnknown(UNKNOWN)]),
        ],
    )


def raise_unknown_as_a_cause(_attempt):
    raise RuntimeError("Effect publication failed") from ResourceOutcomeUnknown(
        UNKNOWN
    )


async def raise_unavailable_from_a_task_group(_attempt):
    async def unavailable():
        raise OSError("Effect storage unavailable")

    async with asyncio.TaskGroup() as group:
        group.create_task(unavailable())


# An unknown outcome another error carries, and that error's own text.
CARRIED_UNKNOWN_OUTCOMES = [
    (raise_unknown_from_a_task_group, "unhandled errors in a TaskGroup (1 sub-exception)"),
    (raise_unknown_in_a_nested_group, "Effects failed (2 sub-exceptions)"),
    (raise_unknown_as_a_cause, "Effect publication failed"),
]


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
@pytest.mark.parametrize(
    "supplier, text", CARRIED_UNKNOWN_OUTCOMES,
    ids=["task-group", "nested-group", "cause"],
)
async def test_an_unknown_outcome_another_error_carries_is_never_retried(
    task_feature, effect, manager, supplier, text
):
    # A skill running its effects in an asyncio.TaskGroup raises an
    # ExceptionGroup; one wrapping its effect's error names it as the cause.
    effect.supplier = supplier
    result = await task_feature.run_workflow([effect_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1 and record["error"] == text
    # The receipt keeps the carrying error's own message.
    [task] = await stored_tasks(manager)
    assert task.status.state == TaskState.TASK_STATE_FAILED
    assert message_text(task.status.message) == text
    assert task_outcome_unknown(task)


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_an_ordinary_exception_group_is_not_retried_after_native_admission(
    task_feature, effect, manager
):
    effect.supplier = raise_unavailable_from_a_task_group
    result = await task_feature.run_workflow([effect_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["attempts"] == 1
    assert record["outcome_unknown"] is True
    assert any(task_outcome_unknown(task) for task in await stored_tasks(manager))


@pytest.mark.parametrize(
    "effect",
    [
        (base, form)
        for base in ("core-feature", "sdk-feature")
        for form in ("decorated", "plain-decorated")
    ],
    ids=skill_form_id,
    indirect=True,
)
async def test_a_durable_tool_wrapper_preserves_original_grouped_uncertainty(effect):
    effect.supplier = raise_unknown_in_a_nested_group
    [skill] = effect.get_tools()
    with pytest.raises(ExceptionGroup) as raised:
        await skill.execute()
    assert len(observed_outcome_unknown(raised.value)) == 1
    assert effect.calls == 1


async def test_a_grouped_unknown_outcome_raised_to_the_workflow_is_not_retried(
    task_feature, manager, monkeypatch
):
    calls = []

    async def execute_skill(**kwargs):
        calls.append(kwargs)
        raise ExceptionGroup(
            "Task execution failed", [ResourceOutcomeUnknown(UNKNOWN)]
        )

    monkeypatch.setattr(manager, "execute_skill", execute_skill)
    result = await task_feature.run_workflow(
        [
            {
                "feature": "memory_feature",
                "skill": "memory_status",
                "max_retries": 2,
                "retry_delay_ms": 0,
            }
        ]
    )
    assert len(calls) == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1
    assert record["error"] == "Task execution failed (1 sub-exception)"


def test_an_unknown_outcome_is_read_from_the_explicit_error_graph():
    bare = ResourceOutcomeUnknown(UNKNOWN)
    assert observed_outcome_unknown(bare) == (bare,)

    member = ResourceOutcomeUnknown(UNKNOWN)
    grouped = ExceptionGroup(
        "outer", [OSError("other"), ExceptionGroup("inner", [member])]
    )
    assert observed_outcome_unknown(grouped) == (member,)

    cause = ResourceOutcomeUnknown(UNKNOWN)
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = cause
    assert observed_outcome_unknown(wrapper) == (cause,)

    # Implicit handling context is not evidence, as for every execution
    # classifier: only an explicit cause or group member counts.
    try:
        try:
            raise ResourceOutcomeUnknown(UNKNOWN)
        except ResourceOutcomeUnknown:
            raise OSError("cleanup failed")
    except OSError as error:
        handled = error
    assert isinstance(handled.__context__, ResourceOutcomeUnknown)
    assert observed_outcome_unknown(handled) == ()

    assert observed_outcome_unknown(ExceptionGroup("ordinary", [OSError("x")])) == ()
    assert observed_outcome_unknown(OSError("ordinary")) == ()
    assert observed_outcome_unknown(None) == ()


class PlainValueProvider(SDKFeature):
    tool_description = "Reads a value"

    async def initialize(self):
        pass

    @tool(name="read_value", description="Read a value")
    def read_value(self) -> dict:
        return {"value": 1}


@pytest.mark.parametrize("form", ["decorated", "plain-decorated"])
def test_the_sdk_outcome_guard_keeps_each_tool_methods_shape(form):
    original = effect_provider(SDKFeature, form)
    adopted = ensure_subagent_dispatch(original)
    guarded = adopted.__dict__["apply_effect"]
    assert guarded.__wrapped__ is original.apply_effect
    assert guarded._tool_schema is original.apply_effect._tool_schema
    assert inspect.signature(guarded) == inspect.signature(original.apply_effect)
    assert inspect.iscoroutinefunction(guarded) is (form == "decorated")


def test_the_sdk_outcome_guard_returns_a_plain_methods_value_unchanged():
    agent = SimpleNamespace(did="did:kestrel:plain", _execution_custody=None)
    feature = ensure_subagent_dispatch(PlainValueProvider)(agent)
    assert feature.read_value() == {"value": 1}


def unreadable_step_results(monkeypatch):
    """Fail reading every step's returned task, after the skill has run."""

    def unreadable(*_args, **_kwargs):
        raise ValueError("Step result unreadable")

    monkeypatch.setattr(TaskFeature, "_classify_step_result", staticmethod(unreadable))


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_a_flagged_step_whose_result_cannot_be_read_is_not_retried(
    task_feature, effect, monkeypatch
):
    effect.supplier = raise_unknown
    unreadable_step_results(monkeypatch)
    result = await task_feature.run_workflow([effect_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1
    assert record["error"] == (
        "Step 0 outcome is unknown; its result could not be read: "
        "Step result unreadable"
    )


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_an_unflagged_step_whose_result_cannot_be_read_is_still_retried(
    task_feature, effect, monkeypatch
):
    effect.supplier = lambda _attempt: ToolResult.failed("Known business failure")
    unreadable_step_results(monkeypatch)
    result = await task_feature.run_workflow([effect_step(max_retries=2)])
    assert effect.calls == 3
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["attempts"] == 3
    assert "outcome_unknown" not in record


def reject_task_writes(manager, monkeypatch, rejected):
    """Fail every terminal write of a task ``rejected`` selects, on either engine."""
    save = manager.task_store.save_recipient_terminal_outcome

    async def refuse(task, **kwargs):
        if rejected(task):
            raise QueryError("Task write rejected")
        return await save(task, **kwargs)

    monkeypatch.setattr(manager.task_store, "save_recipient_terminal_outcome", refuse)


def reject_failed_task_writes(manager, monkeypatch):
    """Fail every terminal write of a FAILED task."""
    reject_task_writes(
        manager, monkeypatch,
        lambda task: task.status.state == TaskState.TASK_STATE_FAILED,
    )


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_an_unknown_outcome_survives_a_failed_task_write(
    task_feature, effect, manager, monkeypatch
):
    effect.supplier = raise_unknown
    reject_failed_task_writes(manager, monkeypatch)
    with pytest.raises(ResourceOutcomeUnknown) as raised:
        await manager.execute_skill("effect_feature", "apply_effect", {}, sync=True)
    # The storage failure stays the evidence for the unrecorded task.
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    assert any(isinstance(error, QueryError) for error in execution_evidence_carriers(raised.value))
    assert str(raised.value) == f"{UNKNOWN}; its failed task could not be recorded"

    result = await task_feature.run_workflow([effect_step(max_retries=2)])
    assert effect.calls == 2
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1 and record["error"] == str(raised.value)


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_post_admission_exception_and_failed_task_write_cannot_trigger_retry(
    task_feature, effect, manager, monkeypatch
):
    effect.supplier = raise_unavailable
    reject_failed_task_writes(manager, monkeypatch)
    with pytest.raises((ResourceOutcomeUnknown, BaseExceptionGroup)):
        await manager.execute_skill("effect_feature", "apply_effect", {}, sync=True)
    result = await task_feature.run_workflow([effect_step(max_retries=2)])
    assert effect.calls == 2
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["attempts"] == 1
    assert record["outcome_unknown"] is True


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_an_unknown_outcome_survives_another_writers_terminal_state(
    task_feature, effect, manager
):
    async def cancel_then_lose_the_outcome(_attempt):
        [running] = [
            task for task in await stored_tasks(manager)
            if task.status.state == TaskState.TASK_STATE_SUBMITTED
        ]
        await manager.cancel_task(
            running.id, reason="Canceled while running", agent_name=manager.did
        )
        raise ResourceOutcomeUnknown(UNKNOWN)

    effect.supplier = cancel_then_lose_the_outcome
    result = await task_feature.run_workflow([effect_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1
    assert record["error"] == (
        f"{UNKNOWN}; another writer recorded the task as canceled"
    )
    [task] = await stored_tasks(manager)
    assert task.status.state == TaskState.TASK_STATE_CANCELED


@pytest.fixture
def nested_workflow(manager, task_feature, workflow_host):
    """The tasks feature registered as a skill provider, as on a real host."""
    agent = workflow_host
    nested = TaskFeature(agent)
    nested.set_task_manager(manager)
    nested.name = "tasks"
    manager.register_provider("tasks", nested.get_agent_skills(), nested)
    return task_feature


def outer_step(inner_steps, max_retries):
    return {
        "feature": "tasks",
        "skill": "run_workflow",
        "args": {"steps": inner_steps},
        "max_retries": max_retries,
        "retry_delay_ms": 0,
    }


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
@pytest.mark.parametrize("persisted", [True, False], ids=["recorded", "write-failed"])
async def test_an_enclosing_workflow_never_reruns_an_unknown_inner_outcome(
    nested_workflow, effect, manager, monkeypatch, persisted
):
    effect.supplier = raise_unknown
    if not persisted:
        # Neither the inner steps' tasks nor the inner workflow's own task
        # can be recorded, so the disposition crosses both levels raised.
        reject_failed_task_writes(manager, monkeypatch)
    # Every inner step fails, so the inner workflow fails as a whole.
    inner = [effect_step(max_retries=2), effect_step(max_retries=2)]
    result = await nested_workflow.run_workflow([outer_step(inner, max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_an_enclosing_workflow_fences_an_ordinary_inner_exception(
    nested_workflow, effect, manager
):
    effect.supplier = raise_unavailable
    result = await nested_workflow.run_workflow(
        [outer_step([effect_step(max_retries=0)], max_retries=2)]
    )
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["attempts"] == 1
    assert record["outcome_unknown"] is True
    run_tasks = [
        task for task in await stored_tasks(manager)
        if task.metadata["skill"] == "run_workflow"
    ]
    assert len(run_tasks) == 1
    assert any(task_outcome_unknown(task) for task in run_tasks)


def is_workflow_task(task):
    return task.metadata["skill"] == "run_workflow"


def mixed_inner_workflow():
    """An inner workflow whose first step succeeds and whose second is uncertain."""
    return [
        {"feature": "wallet_feature", "skill": "check_balance"},
        effect_step(max_retries=2),
    ]


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_a_partial_inner_workflow_carries_its_unknown_outcome(
    nested_workflow, effect, manager
):
    # One step succeeded, so the inner workflow completes with a partial
    # result. The uncertain step still decides whether it may run again.
    effect.supplier = raise_unknown
    result = await nested_workflow.run_workflow(
        [outer_step(mixed_inner_workflow(), max_retries=2)]
    )
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "partial" and record["outcome_unknown"] is True
    assert record["attempts"] == 1
    [inner] = [task for task in await stored_tasks(manager) if is_workflow_task(task)]
    assert inner.status.state == TaskState.TASK_STATE_COMPLETED
    assert task_outcome_unknown(inner)
    assert message_text(inner.status.message) == "Completed run_workflow"


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_a_partial_inner_workflow_whose_task_write_fails_is_not_rerun(
    nested_workflow, effect, manager, monkeypatch
):
    effect.supplier = raise_unknown
    reject_task_writes(manager, monkeypatch, is_workflow_task)
    result = await nested_workflow.run_workflow(
        [outer_step(mixed_inner_workflow(), max_retries=2)]
    )
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1
    assert record["error"] == (
        "The skill's outcome is unknown; its completed task could not be recorded"
    )


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_a_partial_inner_workflow_canceled_meanwhile_is_not_rerun(
    nested_workflow, effect, manager
):
    async def cancel_the_inner_workflow_then_lose_the_outcome(_attempt):
        [running] = [
            task for task in await stored_tasks(manager)
            if is_workflow_task(task)
            and task.status.state == TaskState.TASK_STATE_SUBMITTED
        ]
        await manager.cancel_task(
            running.id, reason="Canceled while running", agent_name=manager.did
        )
        raise ResourceOutcomeUnknown(UNKNOWN)

    effect.supplier = cancel_the_inner_workflow_then_lose_the_outcome
    result = await nested_workflow.run_workflow(
        [outer_step(mixed_inner_workflow(), max_retries=2)]
    )
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1
    assert record["error"] == (
        "The skill's outcome is unknown; another writer recorded the task as canceled"
    )
    [inner] = [task for task in await stored_tasks(manager) if is_workflow_task(task)]
    assert inner.status.state == TaskState.TASK_STATE_CANCELED


def encloses_a_workflow(task):
    """A workflow task one of whose steps ran another workflow."""
    if not is_workflow_task(task) or not task.artifacts:
        return False
    recognized, value = embedded_json_value(task.artifacts[0].parts[0])
    return recognized and any(
        step.get("feature") == "tasks" for step in value["data"]["results"]
    )


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
@pytest.mark.parametrize("persisted", [True, False], ids=["recorded", "write-failed"])
async def test_a_workflow_reports_a_partial_steps_unknown_outcome_upward(
    nested_workflow, effect, manager, monkeypatch, persisted
):
    # The middle workflow's only step is a partial inner workflow. It must pass
    # that step's unknown outcome to its own task even though the step did not
    # fail, so the outermost workflow never reruns it.
    effect.supplier = raise_unknown
    if not persisted:
        reject_task_writes(manager, monkeypatch, encloses_a_workflow)
    middle = [outer_step(mixed_inner_workflow(), max_retries=2)]
    result = await nested_workflow.run_workflow([outer_step(middle, max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["outcome_unknown"] is True and record["attempts"] == 1
    if persisted:
        assert record["status"] == "partial"
        workflow_tasks = [
            task for task in await stored_tasks(manager) if is_workflow_task(task)
        ]
        assert len(workflow_tasks) == 2
        assert all(
            task.status.state == TaskState.TASK_STATE_COMPLETED
            and task_outcome_unknown(task)
            for task in workflow_tasks
        )
    else:
        assert record["status"] == "failed"
        assert record["error"].endswith("its completed task could not be recorded")


def is_effect_task(task):
    return task.metadata["skill"] == "apply_effect"


def coordinator_provider(base):
    class CoordinatorProvider(base):
        tool_description = "Coordinates the effect skill"

        async def initialize(self):
            pass

        @tool(name="coordinate", description="Run the effect skill and report it")
        async def coordinate(self) -> dict:
            # It reads its child task's state, never its flag, and returns an
            # ordinary failure: nothing it returns says the effect is unknown.
            try:
                child = await self.manager.execute_skill(
                    "effect_feature", "apply_effect", {}, sync=True
                )
            except ResourceOutcomeUnknown as error:
                return ToolResult.failed(f"Effect skill failed: {error}")
            return ToolResult.failed(
                f"Effect skill ended {task_state_label(child.status.state)}"
            )

    return CoordinatorProvider


@pytest.fixture(params=["core-feature", "sdk-feature"])
def coordinator(request, manager, workflow_host):
    """A skill, other than a workflow, that runs the effect skill as a task."""
    base = Feature if request.param == "core-feature" else SDKFeature
    agent = workflow_host
    provider = ensure_subagent_dispatch(coordinator_provider(base))(agent)
    provider.name = "coordinator_feature"
    provider.manager = manager
    manager.register_provider(
        "coordinator_feature", provider.get_agent_skills(), provider
    )
    return provider


def coordinator_step(max_retries):
    return {
        "feature": "coordinator_feature",
        "skill": "coordinate",
        "max_retries": max_retries,
        "retry_delay_ms": 0,
    }


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
@pytest.mark.parametrize("persisted", [True, False], ids=["recorded", "write-failed"])
async def test_a_skill_that_ran_an_uncertain_skill_is_never_rerun(
    task_feature, effect, coordinator, manager, monkeypatch, persisted
):
    effect.supplier = raise_unknown
    if not persisted:
        # The child's task cannot be recorded, so the coordinator catches the
        # raised error instead of reading a flagged task.
        reject_task_writes(manager, monkeypatch, is_effect_task)
    result = await task_feature.run_workflow([coordinator_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1
    [coordinating] = [
        task for task in await stored_tasks(manager)
        if task.metadata["skill"] == "coordinate"
    ]
    # The coordinator's own failure message is unchanged.
    assert coordinating.status.state == TaskState.TASK_STATE_FAILED
    assert message_text(coordinating.status.message).startswith("Effect skill ")
    assert task_outcome_unknown(coordinating)


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_a_skill_that_ran_an_ordinarily_failing_skill_is_not_rerun_after_native_admission(
    task_feature, effect, coordinator, manager
):
    effect.supplier = raise_unavailable
    result = await task_feature.run_workflow([coordinator_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["attempts"] == 1
    assert record["outcome_unknown"] is True
    assert any(task_outcome_unknown(task) for task in await stored_tasks(manager))


class RunsTheEffectOnCompletion(Hook):
    """A deferred result's completion policy that runs the effect skill."""

    def __init__(self, manager):
        super().__init__("runs-the-effect-on-completion", [HookEvent.POST_TOOL_USE])
        self.manager = manager

    async def execute(self, value):
        if isinstance(value.tool_response, DeferredToolResult):
            # It allows the result whatever the child did: nothing the
            # completed result says is that the effect is unknown.
            with contextlib.suppress(ResourceOutcomeUnknown):
                await self.manager.execute_skill(
                    "effect_feature", "apply_effect", {}, sync=True
                )
        return HookOutput.allow()


def deferring_provider(base):
    class DeferringProvider(base):
        tool_description = "Returns a failure completed after the skill returns"

        async def initialize(self):
            pass

        @tool(name="defer", description="Return a deferred failure")
        async def defer(self) -> dict:
            return DeferredToolResult(
                ToolResult.failed("Deferred step failed").to_dict(),
                publish=lambda: [],
            )

    return DeferringProvider


@pytest.fixture(params=["core-feature", "sdk-feature"])
async def deferring(request, manager, workflow_host):
    """A skill whose deferred result's completion runs the effect skill."""
    host = workflow_host
    host.hooks_manager.register(RunsTheEffectOnCompletion(manager))
    base = Feature if request.param == "core-feature" else SDKFeature
    provider = ensure_subagent_dispatch(deferring_provider(base))(host)
    provider.name = "deferring_feature"
    host.features[type(provider).__name__] = provider
    manager.register_provider("deferring_feature", provider.get_agent_skills(), provider)
    return provider


def deferring_step(max_retries):
    return {
        "feature": "deferring_feature",
        "skill": "defer",
        "max_retries": max_retries,
        "retry_delay_ms": 0,
    }


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
@pytest.mark.parametrize("persisted", [True, False], ids=["recorded", "write-failed"])
async def test_a_skill_whose_completion_ran_an_uncertain_skill_is_never_rerun(
    task_feature, effect, deferring, manager, monkeypatch, persisted
):
    effect.supplier = raise_unknown
    if not persisted:
        reject_task_writes(manager, monkeypatch, is_effect_task)
    result = await task_feature.run_workflow([deferring_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1
    [deferred] = [
        task for task in await stored_tasks(manager)
        if task.metadata["skill"] == "defer"
    ]
    assert deferred.status.state == TaskState.TASK_STATE_FAILED
    assert message_text(deferred.status.message) == "Deferred step failed"
    assert task_outcome_unknown(deferred)


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_a_skill_whose_completion_ran_an_ordinarily_failing_skill_is_not_rerun_after_native_admission(
    task_feature, effect, deferring, manager
):
    effect.supplier = raise_unavailable
    result = await task_feature.run_workflow([deferring_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["attempts"] == 1
    assert record["outcome_unknown"] is True
    assert any(task_outcome_unknown(task) for task in await stored_tasks(manager))


def delegating_provider(base):
    class DelegatingProvider(base):
        tool_description = "Runs its own effect tool as its subagent does"

        async def initialize(self):
            pass

        @tool(name="delegate", description="Run the effect tool through the subagent step")
        async def delegate(self) -> dict:
            # The subagent's tool step returns a raised error to its model as
            # an ordinary failure, and this skill returns that unchanged.
            return await self._execute_subagent_tool(
                tool_name="apply_effect",
                args={},
                tools_by_name={"apply_effect": EffectSkill(self)},
            )

    return DelegatingProvider


@pytest.fixture(params=["core-feature", "sdk-feature"])
def delegating(request, manager, workflow_host):
    """A skill whose feature-owned tool runs through the subagent tool step."""
    base = Feature if request.param == "core-feature" else SDKFeature
    agent = workflow_host
    provider = ensure_subagent_dispatch(delegating_provider(base))(agent)
    provider.name = "delegating_feature"
    provider.calls = 0
    manager.register_provider(
        "delegating_feature", provider.get_agent_skills(), provider
    )
    return provider


def delegating_step(max_retries):
    return {
        "feature": "delegating_feature",
        "skill": "delegate",
        "max_retries": max_retries,
        "retry_delay_ms": 0,
    }


async def test_a_subagent_tool_step_carries_its_unknown_outcome(
    task_feature, delegating, manager
):
    delegating.supplier = raise_unknown
    result = await task_feature.run_workflow([delegating_step(max_retries=2)])
    assert delegating.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1
    assert record["error"] == UNKNOWN
    [task] = await stored_tasks(manager)
    assert task_outcome_unknown(task)


async def test_a_subagent_tool_steps_ordinary_failure_is_not_retried_after_native_admission(
    task_feature, delegating, manager
):
    delegating.supplier = raise_unavailable
    result = await task_feature.run_workflow([delegating_step(max_retries=2)])
    assert delegating.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["attempts"] == 1
    assert record["outcome_unknown"] is True
    assert any(task_outcome_unknown(task) for task in await stored_tasks(manager))


async def test_a_subagent_tool_step_carries_a_grouped_unknown_outcome(
    task_feature, delegating, manager
):
    delegating.supplier = raise_unknown_from_a_task_group
    result = await task_feature.run_workflow([delegating_step(max_retries=2)])
    assert delegating.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1
    [task] = await stored_tasks(manager)
    assert task_outcome_unknown(task)


class CodexSubagentRunner(Feature):
    """A skill whose feature subagent is a codex turn.

    It builds the effect feature's inline tool executor on its own task, and
    the codex reader runs it on a task that predates the skill, returning a
    flattened failure as the subagent's model would see it.
    """

    tool_description = "Runs the effect tool as a codex subagent turn does"

    async def initialize(self):
        pass

    @tool(name="run_on_reader", description="Run the effect tool on the codex reader")
    async def run_on_reader(self) -> dict:
        executor = self.effect._make_feature_inline_tool_executor(parts_sink=None)
        return await self.reader.call(executor, "apply_effect")


@pytest.fixture
async def codex_runner(manager, effect, workflow_host):
    """The runner, with a codex reader started before any skill runs."""
    agent = workflow_host
    runner = CodexSubagentRunner(agent)
    runner.name = "codex_runner"
    runner.effect = effect
    runner.reader = PreTurnReader()
    await runner.reader.start()
    manager.register_provider("codex_runner", runner.get_agent_skills(), runner)
    try:
        yield runner
    finally:
        await runner.reader.stop()


def codex_runner_step(max_retries):
    return {
        "feature": "codex_runner",
        "skill": "run_on_reader",
        "max_retries": max_retries,
        "retry_delay_ms": 0,
    }


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
@pytest.mark.parametrize("persisted", [True, False], ids=["recorded", "write-failed"])
async def test_an_unknown_outcome_flattened_on_a_codex_reader_is_never_rerun(
    task_feature, effect, codex_runner, manager, monkeypatch, persisted
):
    # The effect's tool wrapper or the subagent's tool step flattens the
    # unknown outcome on the reader's task. Only the turn's scope carries the
    # runner's capture there.
    effect.supplier = raise_unknown
    if not persisted:
        reject_failed_task_writes(manager, monkeypatch)
    result = await task_feature.run_workflow([codex_runner_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["outcome_unknown"] is True
    assert record["attempts"] == 1 and UNKNOWN in record["error"]


@pytest.mark.parametrize("effect", EVERY_SKILL_FORM, ids=skill_form_id, indirect=True)
async def test_an_ordinary_failure_flattened_on_a_codex_reader_is_not_retried_after_native_admission(
    task_feature, effect, codex_runner, manager
):
    effect.supplier = raise_unavailable
    result = await task_feature.run_workflow([codex_runner_step(max_retries=2)])
    assert effect.calls == 1
    [record] = result.data["results"]
    assert record["status"] == "failed" and record["attempts"] == 1
    assert record["outcome_unknown"] is True
    assert any(task_outcome_unknown(task) for task in await stored_tasks(manager))


@pytest.mark.parametrize(
    "effect",
    [("core-feature", "decorated"), ("sdk-feature", "decorated")],
    ids=skill_form_id,
    indirect=True,
)
@pytest.mark.parametrize(
    "member, retried",
    [(lambda: ResourceOutcomeUnknown(UNKNOWN), False),
     (lambda: OSError("Effect storage unavailable"), True)],
    ids=["unknown", "ordinary"],
)
async def test_the_inline_executor_reports_a_grouped_unknown_outcome(
    task_feature, effect, codex_runner, manager, monkeypatch, member, retried
):
    # The subagent's tool step itself raises, grouped, to the executor, and
    # the codex handler returns that error to the model as an ordinary failure.
    async def subagent_step(**_kwargs):
        effect.calls += 1
        raise ExceptionGroup("Subagent step failed", [member()])

    monkeypatch.setattr(effect, "_execute_subagent_tool", subagent_step)
    result = await task_feature.run_workflow([codex_runner_step(max_retries=2)])
    [record] = result.data["results"]
    assert record["status"] == "failed"
    assert record["error"] == "Subagent step failed (1 sub-exception)"
    if retried:
        assert effect.calls == 3 and record["attempts"] == 3
        assert "outcome_unknown" not in record
    else:
        assert effect.calls == 1 and record["attempts"] == 1
        assert record["outcome_unknown"] is True


@pytest.mark.parametrize(
    "state",
    [
        TaskState.TASK_STATE_SUBMITTED,
        TaskState.TASK_STATE_WORKING,
        TaskState.TASK_STATE_INPUT_REQUIRED,
    ],
)
def test_only_a_finished_task_can_carry_an_unknown_outcome(state):
    task = Task(id="task", context_id="context", status=TaskStatus(state=state))
    with pytest.raises(ValueError, match="Only a finished task"):
        mark_outcome_unknown(task)
    assert not task_outcome_unknown(task)


@pytest.mark.parametrize(
    "state",
    [
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    ],
)
def test_a_finished_task_in_any_state_can_carry_an_unknown_outcome(state):
    task = Task(id="task", context_id="context", status=TaskStatus(state=state))
    assert not task_outcome_unknown(task)
    mark_outcome_unknown(task)
    assert task_outcome_unknown(task) and task.status.state == state


def test_an_unknown_outcome_flag_must_be_a_true_boolean():
    task = Task(
        id="task",
        context_id="context",
        status=TaskStatus(state=TaskState.TASK_STATE_FAILED),
    )
    assert not task_outcome_unknown(task)
    for value in ("true", 1, False):
        task.metadata.update({"outcome_unknown": value})
        assert not task_outcome_unknown(task)
    mark_outcome_unknown(task)
    assert task_outcome_unknown(task)


async def test_nested_results_and_task_projections_remain_json_serializable(
    task_feature, providers, manager
):
    providers.memory.supplier = lambda _attempt: {
        "nested": ToolResult.ok("Nested", data={"count": 3})
    }
    result = await task_feature.run_workflow(
        [{"feature": "memory_feature", "skill": "memory_status"}]
    )
    assert result.status is ToolResultStatus.OK
    assert payload(result)["nested"]["data"]["count"] == 3
    json.dumps(result.to_dict(), allow_nan=False)
    [(task_id,)] = await manager.task_store.backend.fetch_all(
        "SELECT id FROM a2a_tasks"
    )
    for projection in (
        await task_feature.check_task_status(task_id),
        await task_feature.get_task_result(task_id),
    ):
        assert projection.status is ToolResultStatus.OK
        json.dumps(projection.to_dict(), allow_nan=False)
    stored = await manager.get_task_for_recipient(task_id, manager.did)
    recognized, value = embedded_json_value(stored.artifacts[0].parts[0])
    assert recognized and value == result.data["results"][0]["result"]


async def test_catalog_names_the_actual_registered_providers_and_skills(task_feature):
    result = await task_feature.list_available_skills()
    assert result.status is ToolResultStatus.OK
    assert result.data["feature_count"] == 3 and result.data["skill_count"] == 4
    assert set(result.data["features"]) == {
        "model_agent",
        "memory_feature",
        "wallet_feature",
    }
    skills = result.data["features"]["model_agent"]["skills"]
    assert {skill["skill"] for skill in skills} == {"list_models", "get_current_model"}
    assert all(
        skill["description"]
        for value in result.data["features"].values()
        for skill in value["skills"]
    )


def test_classify_step_result_unit_table():
    """Classify semantic outcomes across the utility's explicit accepted value forms."""
    from kestrel_sdk.tools.result import ToolResult

    cases = [
        ("completed", {"status": "ok", "confirmation": "ok"}, "completed", None),
        ("completed", {"status": "error", "error": "x"}, "failed", "x"),
        (
            "completed",
            {"status": "partial", "confirmation": "c", "error": "e"},
            "partial",
            "e",
        ),
        ("completed", ToolResult.ok("ok"), "completed", None),
        ("completed", ToolResult.failed("raw err"), "failed", "raw err"),
        (
            "completed",
            ToolResult.partial(confirmation="c", error="raw partial"),
            "partial",
            "raw partial",
        ),
        (
            "completed",
            {
                "success": True,
                "result": {"status": "ok", "confirmation": "ok"},
                "tool": "t",
            },
            "completed",
            None,
        ),
        (
            "completed",
            {
                "success": True,
                "result": {"status": "error", "error": "wrapped err"},
                "tool": "t",
            },
            "failed",
            "wrapped err",
        ),
        (
            "completed",
            {
                "success": True,
                "result": {
                    "status": "partial",
                    "confirmation": "c",
                    "error": "wrapped partial",
                },
                "tool": "t",
            },
            "partial",
            "wrapped partial",
        ),
        (
            "completed",
            {"success": True, "result": ToolResult.ok("ok"), "tool": "t"},
            "completed",
            None,
        ),
        (
            "completed",
            {
                "success": True,
                "result": ToolResult.failed("raw wrapped err"),
                "tool": "t",
            },
            "failed",
            "raw wrapped err",
        ),
        (
            "completed",
            {
                "success": True,
                "result": ToolResult.partial(
                    confirmation="c", error="raw wrapped partial"
                ),
                "tool": "t",
            },
            "partial",
            "raw wrapped partial",
        ),
        ("completed", {"success": False, "error": "legacy"}, "failed", "legacy"),
        ("completed", {"success": True, "model": "m"}, "completed", None),
        ("failed", {"status": "ok"}, "failed", None),
        ("failed", ToolResult.ok("would have been ok"), "failed", None),
        ("completed", "string result", "completed", None),
        ("completed", None, "completed", None),
    ]
    for transport, data, expected_status, expected_err in cases:
        actual_status, actual_err = TaskFeature._classify_step_result(transport, data)
        assert actual_status == expected_status, (
            f"transport={transport!r} data={data!r} → expected {expected_status}, got {actual_status}"
        )
        if expected_err is not None:
            assert actual_err == expected_err, (
                f"transport={transport!r} data={data!r} → expected error {expected_err!r}, got {actual_err!r}"
            )


@pytest.mark.parametrize("domain_status", ["ok", "partial"])
async def test_domain_status_is_not_mistaken_for_a_tool_result_envelope(
    task_feature,
    providers,
    domain_status,
):
    providers.memory.supplier = lambda _attempt: {"status": domain_status, "items": [1]}
    result = await task_feature.run_workflow(
        [
            {"feature": "memory_feature", "skill": "memory_status"},
        ]
    )
    assert result.status is ToolResultStatus.OK
    assert result.data["results"][0]["status"] == "completed"


def test_native_terminal_failures_override_an_optimistic_result():
    for state in ["failed", "canceled", "rejected"]:
        assert TaskFeature._classify_step_result(
            state,
            ToolResult.ok("Returned"),
            failure_message="Native refusal",
        ) == ("failed", "Native refusal")
