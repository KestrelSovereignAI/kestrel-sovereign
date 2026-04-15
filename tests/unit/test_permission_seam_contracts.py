"""Permission seam contracts across command, A2A, and tool execution paths."""

from types import SimpleNamespace

import pytest

from kestrel_sovereign.a2a.agent_card import AgentCapabilities, AgentCard, AgentSkill
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.a2a.types import Artifact, Message, TaskState, TaskStatus, TextPart
from kestrel_sovereign.hooks import HookEvent, HookInput, HookOutput


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
