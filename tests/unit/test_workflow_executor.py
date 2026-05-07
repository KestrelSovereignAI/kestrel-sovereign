"""
Unit tests for TaskFeature.run_workflow — multi-step workflow execution.

Tests the workflow executor that lets the orchestrator submit a plan
of feature skills and get consolidated results back.
"""

import asyncio
import os
import tempfile

import pytest
import pytest_asyncio

from kestrel_sovereign.a2a.types import (
    Task,
    TaskState,
    TaskStatus,
    Message,
    TextPart,
    Artifact,
    DataPart,
)
from kestrel_sovereign.a2a.task_manager import TaskManager, create_task_manager
from kestrel_sovereign.a2a.agent_card import AgentCard, AgentCapabilities, AgentSkill
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.tasks.feature import TaskFeature


# =============================================================================
# Test Helpers
# =============================================================================

class MockHandler:
    """A simple handler that returns predictable results for testing."""

    def __init__(self, skills: dict):
        """
        Args:
            skills: dict of skill_name -> callable(args) -> result_data
        """
        self._skills = skills

    async def handle_task(self, task: Task) -> Task:
        """Execute the skill and return result as artifact."""
        task.status = TaskStatus(state=TaskState.WORKING)

        metadata = task.metadata or {}
        skill_name = metadata.get("skill")
        args = metadata.get("args", {})

        if skill_name not in self._skills:
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(role="agent", parts=[TextPart(text=f"Unknown skill: {skill_name}")])
            )
            return task

        try:
            result = self._skills[skill_name](args)
            if asyncio.iscoroutine(result):
                result = await result

            task.artifacts = [
                Artifact(
                    name=f"{skill_name}_result",
                    parts=[DataPart(data=result)],
                )
            ]
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(role="agent", parts=[TextPart(text=f"Completed {skill_name}")])
            )
        except Exception as e:
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(role="agent", parts=[TextPart(text=str(e))])
            )

        return task


def make_agent_card(name: str, skills: list[str]) -> AgentCard:
    """Create a minimal AgentCard for testing."""
    return AgentCard(
        name=name,
        url=f"/agents/{name}",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        skills=[
            AgentSkill(id=s, name=s, description=f"Test skill: {s}")
            for s in skills
        ],
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def db_path():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


_managers_to_close = []


@pytest_asyncio.fixture(autouse=True)
async def _cleanup():
    """Cleanup task managers after each test."""
    yield
    for m in _managers_to_close:
        try:
            if hasattr(m, 'task_store'):
                await m.task_store.close()
            if hasattr(m, 'session_service'):
                await m.session_service.close()
            if hasattr(m, 'observability_store'):
                await m.observability_store.close()
            if hasattr(m, 'memory_service'):
                await m.memory_service.close()
            if hasattr(m, 'feedback_store'):
                await m.feedback_store.close()
        except Exception:
            pass
    _managers_to_close.clear()


@pytest_asyncio.fixture
async def task_manager(db_path):
    """Create an initialized TaskManager with mock features registered."""
    manager = await create_task_manager(db_path)
    _managers_to_close.append(manager)

    # Register mock "model_agent" with list_models and get_current_model
    model_handler = MockHandler({
        "list_models": lambda args: {
            "success": True,
            "models": ["gpt-4o", "claude-sonnet-4-5-20250929", "llama3"],
            "provider": args.get("provider", "all"),
        },
        "get_current_model": lambda args: {
            "success": True,
            "model": "claude-sonnet-4-5-20250929",
            "provider": "anthropic",
        },
    })
    manager.register_agent(
        agent_card=make_agent_card("model_agent", ["list_models", "get_current_model"]),
        handler=model_handler,
    )

    # Register mock "memory_feature" with memory_status
    memory_handler = MockHandler({
        "memory_status": lambda args: {
            "success": True,
            "episodes": 42,
            "total_tokens": 150000,
        },
    })
    manager.register_agent(
        agent_card=make_agent_card("memory_feature", ["memory_status"]),
        handler=memory_handler,
    )

    # Register mock "wallet_feature" with check_balance
    wallet_handler = MockHandler({
        "check_balance": lambda args: {
            "success": True,
            "balance": "12.50",
            "currency": "FIL",
        },
    })
    manager.register_agent(
        agent_card=make_agent_card("wallet_feature", ["check_balance"]),
        handler=wallet_handler,
    )

    yield manager


@pytest_asyncio.fixture
async def task_feature(task_manager):
    """Create a TaskFeature wired to the task manager."""
    feature = TaskFeature(agent=None)
    feature.set_task_manager(task_manager)
    return feature


# =============================================================================
# Tests
# =============================================================================

class TestRunWorkflow:
    """Tests for TaskFeature.run_workflow."""

    @pytest.mark.asyncio
    async def test_single_step(self, task_feature):
        """Single-step workflow executes and returns result."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "get_current_model"},
        ])

        assert result.status is ToolResultStatus.OK
        data = result.data
        assert data["workflow_steps"] == 1
        assert data["completed"] == 1
        assert data["failed"] == 0
        assert len(data["results"]) == 1

        step = data["results"][0]
        assert step["status"] == "completed"
        assert step["result"]["model"] == "claude-sonnet-4-5-20250929"

    @pytest.mark.asyncio
    async def test_multi_step_workflow(self, task_feature):
        """Multi-step workflow executes all steps and returns consolidated results."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "get_current_model"},
            {"feature": "memory_feature", "skill": "memory_status"},
            {"feature": "wallet_feature", "skill": "check_balance"},
        ])

        assert result.status is ToolResultStatus.OK
        data = result.data
        assert data["workflow_steps"] == 3
        assert data["completed"] == 3
        assert data["failed"] == 0

        assert data["results"][0]["result"]["model"] == "claude-sonnet-4-5-20250929"
        assert data["results"][1]["result"]["episodes"] == 42
        assert data["results"][2]["result"]["balance"] == "12.50"

    @pytest.mark.asyncio
    async def test_step_with_args(self, task_feature):
        """Steps can pass arguments to the skill."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "list_models", "args": {"provider": "openai"}},
        ])

        assert result.status is ToolResultStatus.OK
        step = result.data["results"][0]
        assert step["result"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_unknown_feature_fails_gracefully(self, task_feature):
        """Unknown feature name produces a failed step, not a crash."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "nonexistent_feature", "skill": "do_something"},
        ])

        # Every step failed → ERROR (no successes to surface as PARTIAL).
        assert result.status is ToolResultStatus.ERROR
        assert result.data["failed"] == 1
        step = result.data["results"][0]
        assert step["status"] == "failed"
        assert "Unknown agent" in step["error"]

    @pytest.mark.asyncio
    async def test_unknown_skill_fails_gracefully(self, task_feature):
        """Unknown skill name on a valid feature produces a failed step."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "nonexistent_skill"},
        ])

        assert result.status is ToolResultStatus.ERROR
        assert result.data["failed"] == 1
        step = result.data["results"][0]
        assert step["status"] == "failed"

    @pytest.mark.asyncio
    async def test_partial_failure(self, task_feature):
        """A failed step doesn't prevent other steps from running.

        Mix of success+failure surfaces as PARTIAL — the LLM cannot
        claim "workflow complete" while a step actually failed.
        """
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "get_current_model"},
            {"feature": "nonexistent_feature", "skill": "bad_skill"},
            {"feature": "wallet_feature", "skill": "check_balance"},
        ])

        assert result.status is ToolResultStatus.PARTIAL
        data = result.data
        assert data["completed"] == 2
        assert data["failed"] == 1

        assert data["results"][0]["status"] == "completed"
        assert data["results"][1]["status"] == "failed"
        assert data["results"][2]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_empty_steps(self, task_feature):
        """Empty steps list returns error."""
        result = await task_feature.run_workflow(steps=[])
        assert result.status is ToolResultStatus.ERROR
        assert "non-empty" in result.error

    @pytest.mark.asyncio
    async def test_invalid_step_format(self, task_feature):
        """Non-dict step is handled gracefully."""
        result = await task_feature.run_workflow(steps=["not a dict"])
        assert result.status is ToolResultStatus.ERROR
        assert result.data["results"][0]["status"] == "failed"
        assert "object" in result.data["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_step_missing_required_fields(self, task_feature):
        """Step missing feature or skill gets a clear error."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent"},  # missing skill
        ])

        assert result.status is ToolResultStatus.ERROR
        assert "requires" in result.data["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_no_task_manager(self):
        """Without task_manager, returns error."""
        feature = TaskFeature(agent=None)
        # Don't set task_manager
        result = await feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "list_models"},
        ])

        assert result.status is ToolResultStatus.ERROR
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_duration_tracking(self, task_feature):
        """Each step and total duration are tracked."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "get_current_model"},
        ])

        data = result.data
        assert "total_duration_ms" in data
        assert data["total_duration_ms"] >= 0

        step = data["results"][0]
        assert "duration_ms" in step
        assert step["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_step_metadata_in_results(self, task_feature):
        """Results include feature and skill names for each step."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "list_models"},
            {"feature": "wallet_feature", "skill": "check_balance"},
        ])

        results = result.data["results"]
        assert results[0]["feature"] == "model_agent"
        assert results[0]["skill"] == "list_models"
        assert results[1]["feature"] == "wallet_feature"
        assert results[1]["skill"] == "check_balance"


class TestRunWorkflowSemanticHonesty:
    """Codex round 1+2 P1: A2A's task.status is *transport-level* — a tool
    that returns ToolResult.failed lands here as task.state == COMPLETED.
    The workflow rollup must inspect the wire-data and downgrade those
    steps to "failed" / "partial" so it doesn't claim "Workflow complete"
    while a step semantically failed.

    Wire shapes covered:
      - DynamicTool-wrapped (realistic Feature.handle_task path):
        ``{"success": True, "result": ToolResult.to_dict(), "tool": "..."}``
      - Bare ToolResult envelope (some handlers bypass DynamicTool)
      - Pre-migration dict ``{"success": False, "error": ...}``
    """

    @pytest.mark.asyncio
    async def test_tool_result_failed_dynamictool_wrapped_downgrades_step(self, task_feature):
        """Realistic A2A path: DynamicTool.execute() wraps the @tool's
        return as ``{"success": True, "result": <envelope>, "tool": ...}``.
        Round 2 codex finding: classifier must peek inside ``result``.
        """
        from kestrel_sdk.tools.result import ToolResult
        agent_card, _ = task_feature.task_manager._agents["memory_feature"]
        task_feature.task_manager._agents["memory_feature"] = (
            agent_card,
            MockHandler({
                # Mimic DynamicTool.execute()'s wrapping behavior
                "memory_status": lambda args: {
                    "success": True,
                    "result": ToolResult.failed(
                        "ObservabilityStore not available"
                    ).to_dict(),
                    "tool": "memory_status",
                },
            }),
        )

        result = await task_feature.run_workflow(steps=[
            {"feature": "memory_feature", "skill": "memory_status"},
        ])

        assert result.status is ToolResultStatus.ERROR
        assert result.data["failed"] == 1
        step = result.data["results"][0]
        assert step["status"] == "failed"
        assert "ObservabilityStore not available" in step["error"]

    @pytest.mark.asyncio
    async def test_tool_result_partial_dynamictool_wrapped_marks_step_partial(self, task_feature):
        """ToolResult.partial inside DynamicTool wrapper → step partial."""
        from kestrel_sdk.tools.result import ToolResult
        agent_card, _ = task_feature.task_manager._agents["memory_feature"]
        task_feature.task_manager._agents["memory_feature"] = (
            agent_card,
            MockHandler({
                "memory_status": lambda args: {
                    "success": True,
                    "result": ToolResult.partial(
                        confirmation="Got partial stats",
                        error="rag chunks count unknown",
                    ).to_dict(),
                    "tool": "memory_status",
                },
            }),
        )

        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "get_current_model"},
            {"feature": "memory_feature", "skill": "memory_status"},
        ])

        assert result.status is ToolResultStatus.PARTIAL
        assert result.data["partial"] == 1
        partial_step = result.data["results"][1]
        assert partial_step["status"] == "partial"
        assert "rag chunks count unknown" in partial_step["error"]

    @pytest.mark.asyncio
    async def test_bare_envelope_failed_downgrades_step(self, task_feature):
        """Some handlers bypass DynamicTool and put the bare envelope
        in part.data. The classifier must still detect it."""
        from kestrel_sdk.tools.result import ToolResult
        agent_card, _ = task_feature.task_manager._agents["memory_feature"]
        task_feature.task_manager._agents["memory_feature"] = (
            agent_card,
            MockHandler({
                "memory_status": lambda args: ToolResult.failed(
                    "bare envelope path"
                ).to_dict(),
            }),
        )

        result = await task_feature.run_workflow(steps=[
            {"feature": "memory_feature", "skill": "memory_status"},
        ])

        assert result.status is ToolResultStatus.ERROR
        step = result.data["results"][0]
        assert step["status"] == "failed"
        assert "bare envelope path" in step["error"]

    @pytest.mark.asyncio
    async def test_old_dict_shape_with_success_false_downgrades_step(self, task_feature):
        """Pre-migration tool that returns {"success": False, ...} also
        gets downgraded — the rollup is honest during the migration window."""
        agent_card, _ = task_feature.task_manager._agents["memory_feature"]
        task_feature.task_manager._agents["memory_feature"] = (
            agent_card,
            MockHandler({
                "memory_status": lambda args: {
                    "success": False,
                    "error": "legacy shape failure",
                },
            }),
        )

        result = await task_feature.run_workflow(steps=[
            {"feature": "memory_feature", "skill": "memory_status"},
        ])

        assert result.status is ToolResultStatus.ERROR
        step = result.data["results"][0]
        assert step["status"] == "failed"
        assert "legacy shape failure" in step["error"]

    def test_classify_step_result_unit_table(self):
        """Direct table-test of the classifier against every wire shape.

        Defends against future regressions in the dispatch path's
        wire format — adding a new wire layer should be matched by
        a new row here.
        """
        from kestrel_sdk.tools.result import ToolResult
        # (transport_state, result_data, expected_status, expected_error)
        cases = [
            # Bare envelope shapes
            ("completed", {"status": "ok", "confirmation": "ok"}, "completed", None),
            ("completed", {"status": "error", "error": "x"}, "failed", "x"),
            ("completed", {"status": "partial", "confirmation": "c", "error": "e"},
             "partial", "e"),
            # DynamicTool-wrapped shapes
            ("completed", {
                "success": True,
                "result": {"status": "ok", "confirmation": "ok"},
                "tool": "t",
            }, "completed", None),
            ("completed", {
                "success": True,
                "result": {"status": "error", "error": "wrapped err"},
                "tool": "t",
            }, "failed", "wrapped err"),
            ("completed", {
                "success": True,
                "result": {"status": "partial", "confirmation": "c", "error": "wrapped partial"},
                "tool": "t",
            }, "partial", "wrapped partial"),
            # Legacy dict
            ("completed", {"success": False, "error": "legacy"}, "failed", "legacy"),
            ("completed", {"success": True, "model": "m"}, "completed", None),
            # Transport failure overrides everything
            ("failed", {"status": "ok"}, "failed", None),
            # Non-dict result
            ("completed", "string result", "completed", None),
            ("completed", None, "completed", None),
        ]
        for transport, data, expected_status, expected_err in cases:
            actual_status, actual_err = TaskFeature._classify_step_result(transport, data)
            assert actual_status == expected_status, (
                f"transport={transport!r} data={data!r} → "
                f"expected {expected_status}, got {actual_status}"
            )
            if expected_err is not None:
                assert actual_err == expected_err, (
                    f"transport={transport!r} data={data!r} → "
                    f"expected error {expected_err!r}, got {actual_err!r}"
                )


class TestListAvailableSkills:
    """Tests for TaskFeature.list_available_skills."""

    @pytest.mark.asyncio
    async def test_lists_all_features(self, task_feature):
        """Returns all registered features with their skills."""
        result = await task_feature.list_available_skills()

        assert result.status is ToolResultStatus.OK
        data = result.data
        assert data["feature_count"] == 3
        assert "model_agent" in data["features"]
        assert "memory_feature" in data["features"]
        assert "wallet_feature" in data["features"]

    @pytest.mark.asyncio
    async def test_lists_skills_per_feature(self, task_feature):
        """Each feature includes its skill list with descriptions."""
        result = await task_feature.list_available_skills()

        model = result.data["features"]["model_agent"]
        skill_names = [s["skill"] for s in model["skills"]]
        assert "list_models" in skill_names
        assert "get_current_model" in skill_names

        # Each skill has a description
        for skill in model["skills"]:
            assert "description" in skill
            assert skill["description"] is not None

    @pytest.mark.asyncio
    async def test_no_task_manager(self):
        """Without task_manager, returns error."""
        feature = TaskFeature(agent=None)
        result = await feature.list_available_skills()
        assert result.status is ToolResultStatus.ERROR
        assert "not available" in result.error
