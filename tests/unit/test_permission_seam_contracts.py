"""Permission seam contracts across command, A2A, and tool execution paths."""

from datetime import datetime
from types import SimpleNamespace

import aiosqlite
import pytest

from kestrel_sovereign.a2a.agent_card import AgentCapabilities, AgentCard, AgentSkill
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.a2a.types import Artifact, Message, TaskState, TaskStatus, TextPart
from kestrel_sovereign.features.compute.feature import ComputeFeature
from kestrel_sovereign.features.compute.models import (
    ComputePolicy,
    ComputeScript,
    ExecutionRecord,
    ScriptState,
)
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.features.security.permissions import PermissionStore
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sdk.hooks.base import HookEvent, HookInput, HookOutput


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Backend:
    def transaction(self):
        return _Transaction()


class _TaskStore:
    def __init__(self):
        self._backend = _Backend()
        self.saved = []

    async def save(self, task):
        self.saved.append(task)


class _NoopStore:
    async def log_tool_call(self, **kwargs):
        return None


class _SingleScriptStore:
    def __init__(self, script):
        self.script = script
        self.updates = []
        self.executions = []

    async def find_by_id_prefix(self, script_id):
        if self.script.id.startswith(script_id):
            return self.script
        return None

    async def update(self, script):
        self.updates.append((script.state, script.review_notes))
        self.script = script

    async def save_execution(self, record):
        self.executions.append(record)


class _Signer:
    def __init__(self, valid):
        self.valid = valid
        self.calls = 0

    async def verify(self, script):
        self.calls += 1
        return self.valid


class _Executor:
    def __init__(self):
        self.calls = 0

    def supports_language(self, language):
        return True

    async def execute(self, script):
        self.calls += 1
        return ExecutionRecord(
            id="exec-1",
            script_id=script.id,
            completed_at=datetime.now(),
            exit_code=0,
            stdout="executed",
            executor="uv",
        )


class _SideEffectHook:
    name = "side_effect"
    events = [HookEvent.PRE_TOOL_USE]
    priority = 20
    enabled = True
    timeout = 1
    fail_closed = False
    awaits_user_input = False

    def __init__(self):
        self.calls = 0

    def matches(self, tool_name):
        return True

    async def execute(self, hook_input):
        self.calls += 1
        return HookOutput.allow("side effect reached")


class _DenyComputeRunScriptHooks:
    def __init__(self):
        self.inputs: list[HookInput] = []

    async def execute_hooks(self, event, hook_input):
        assert event == HookEvent.PRE_TOOL_USE
        self.inputs.append(hook_input)
        if (
            hook_input.feature_name == "ComputeFeature"
            and hook_input.tool_name == "run_script"
        ):
            return HookOutput.deny("blocked by configured ComputeFeature policy")
        return HookOutput.allow()

    async def execute_hooks_parallel(self, event, hook_input):
        self.inputs.append(hook_input)
        return []


class _ComputeFeatureHandler:
    name = "ComputeFeature"

    def __init__(self):
        self.calls = 0

    async def handle_task(self, task):
        self.calls += 1
        task.status = TaskStatus(state=TaskState.COMPLETED)
        task.artifacts = [Artifact(parts=[TextPart(text="executed")])]
        return task

    def _get_tool_by_name(self, name):
        if name != "run_script":
            return None
        return SimpleNamespace(
            parse_command_args=lambda user_input: {
                "script_id": user_input.split()[-1],
            }
        )


def _make_manager(hooks):
    return TaskManager(
        task_store=_TaskStore(),
        session_service=_NoopStore(),
        observability_store=_NoopStore(),
        hooks_manager=hooks,
    )


@pytest.fixture
def compute_approval_boundary():
    script = ComputeScript(
        id="script-123",
        name="privileged",
        language="python",
        content="print('before')",
        purpose="prove approval boundary",
        state=ScriptState.APPROVED,
        signature="ecdsa:approved",
        signed_by="did:example:agent",
        signed_at=datetime.now(),
    )
    script_store = _SingleScriptStore(script)
    signer = _Signer(valid=True)
    executor = _Executor()
    feature = ComputeFeature(SimpleNamespace(features={}))
    feature.script_store = script_store
    feature.signer = signer
    feature.analyzer = SimpleNamespace(
        analyze=lambda script: SimpleNamespace(
            findings=[],
            risk_score=0,
            has_critical=False,
        )
    )
    feature.policy = ComputePolicy(auto_approve_below_risk=101)
    feature.executors = {"uv": executor}
    feature._initialized = True
    return SimpleNamespace(
        feature=feature,
        script=script,
        script_store=script_store,
        signer=signer,
        executor=executor,
    )


def _compute_agent_card():
    return AgentCard(
        name="compute_feature",
        url="/agents/compute_feature",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        skills=[
            AgentSkill(
                id="run_script",
                name="run_script",
                description="Run a staged compute script",
            )
        ],
    )


@pytest.mark.asyncio
async def test_a2a_skill_uses_feature_class_name_for_permission_hooks():
    hooks = _DenyComputeRunScriptHooks()
    manager = _make_manager(hooks)
    handler = _ComputeFeatureHandler()
    manager.register_agent(
        agent_card=_compute_agent_card(),
        handler=handler,
        command_prefixes={"!compute-run": "run_script"},
    )

    task = await manager.execute_skill(
        agent_id="compute_feature",
        skill_id="run_script",
        args={"script_id": "abc123"},
        sync=True,
    )

    assert hooks.inputs[0].feature_name == "ComputeFeature"
    assert hooks.inputs[0].tool_name == "run_script"
    assert task.status.state == TaskState.FAILED
    assert task.metadata["denied"] is True
    assert "Permission denied" in task.status.message.parts[0].text
    assert handler.calls == 0


@pytest.mark.asyncio
async def test_command_routing_uses_same_permission_identity_as_a2a_skill():
    hooks = _DenyComputeRunScriptHooks()
    manager = _make_manager(hooks)
    handler = _ComputeFeatureHandler()
    manager.register_agent(
        agent_card=_compute_agent_card(),
        handler=handler,
        command_prefixes={"!compute-run": "run_script"},
    )

    result = await manager.execute_command("!compute-run abc123")

    assert hooks.inputs[0].feature_name == "ComputeFeature"
    assert hooks.inputs[0].tool_name == "run_script"
    assert result == {
        "success": False,
        "error": "Permission denied: blocked by configured ComputeFeature policy",
    }
    assert handler.calls == 0


@pytest.mark.asyncio
async def test_stale_approval_rejected_before_executor(compute_approval_boundary):
    boundary = compute_approval_boundary
    boundary.signer.valid = False
    boundary.script.content = "print('mutated after approval')"

    result = await boundary.feature.run_script(boundary.script.id, executor="uv")

    assert result.status == "error"
    assert "invalid signature" in (result.error or "").lower()
    assert boundary.signer.calls == 1
    assert boundary.executor.calls == 0
    assert boundary.script_store.executions == []
    assert boundary.script_store.script.state == ScriptState.REJECTED
    assert "Invalid signature" in boundary.script_store.script.review_notes


@pytest.mark.asyncio
async def test_replayed_approval_rejected_before_executor(compute_approval_boundary):
    boundary = compute_approval_boundary

    first = await boundary.feature.run_script(boundary.script.id, executor="uv")
    assert first.status == "ok"
    assert boundary.executor.calls == 1
    assert boundary.script_store.script.state == ScriptState.COMPLETED

    second = await boundary.feature.run_script(boundary.script.id, executor="uv")

    assert second.status == "error"
    assert "cannot execute" in (second.error or "")
    assert second.data == {"script_id": boundary.script.id, "state": "completed"}
    assert boundary.executor.calls == 1
    assert len(boundary.script_store.executions) == 1


@pytest.mark.asyncio
async def test_malformed_permission_scope_fails_closed_and_audits(tmp_path):
    permission_store = PermissionStore(str(tmp_path / "permissions.db"))
    await permission_store.initialize()
    await permission_store.register_tool("ComputeFeature", "run_script")
    approval_queue = ApprovalQueue(permission_store=permission_store)
    manager = HooksManager()
    manager.register(SecurityHook(permission_store, approval_queue))
    side_effect = _SideEffectHook()
    manager.register(side_effect)

    output = await manager.execute_hooks(
        HookEvent.PRE_TOOL_USE,
        HookInput(
            session_id="red-team",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            feature_name="",
            tool_name="run_script",
            tool_input={"script_id": "script-123"},
        ),
    )

    assert output.continue_execution is False
    assert "Malformed permission scope" in output.stop_reason
    assert side_effect.calls == 0

    logs = await permission_store.get_audit_log(limit=1)
    assert logs[0]["feature"] == "unknown"
    assert logs[0]["tool"] == "run_script"
    assert logs[0]["decision"] == "malformed_scope_denied"

    async with aiosqlite.connect(permission_store.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM security_audit_log")
        assert (await cursor.fetchone())[0] == 1
