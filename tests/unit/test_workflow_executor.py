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

        assert result["success"] is True
        assert result["workflow_steps"] == 1
        assert result["completed"] == 1
        assert result["failed"] == 0
        assert len(result["results"]) == 1

        step = result["results"][0]
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

        assert result["success"] is True
        assert result["workflow_steps"] == 3
        assert result["completed"] == 3
        assert result["failed"] == 0

        # Check each step has results
        assert result["results"][0]["result"]["model"] == "claude-sonnet-4-5-20250929"
        assert result["results"][1]["result"]["episodes"] == 42
        assert result["results"][2]["result"]["balance"] == "12.50"

    @pytest.mark.asyncio
    async def test_step_with_args(self, task_feature):
        """Steps can pass arguments to the skill."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "list_models", "args": {"provider": "openai"}},
        ])

        assert result["success"] is True
        step = result["results"][0]
        assert step["result"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_unknown_feature_fails_gracefully(self, task_feature):
        """Unknown feature name produces a failed step, not a crash."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "nonexistent_feature", "skill": "do_something"},
        ])

        assert result["success"] is False
        assert result["failed"] == 1
        step = result["results"][0]
        assert step["status"] == "failed"
        assert "Unknown agent" in step["error"]

    @pytest.mark.asyncio
    async def test_unknown_skill_fails_gracefully(self, task_feature):
        """Unknown skill name on a valid feature produces a failed step."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "nonexistent_skill"},
        ])

        assert result["success"] is False
        assert result["failed"] == 1
        step = result["results"][0]
        assert step["status"] == "failed"

    @pytest.mark.asyncio
    async def test_partial_failure(self, task_feature):
        """A failed step doesn't prevent other steps from running."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "get_current_model"},
            {"feature": "nonexistent_feature", "skill": "bad_skill"},
            {"feature": "wallet_feature", "skill": "check_balance"},
        ])

        assert result["success"] is False  # overall fails because one step failed
        assert result["completed"] == 2
        assert result["failed"] == 1

        # First and third steps succeeded
        assert result["results"][0]["status"] == "completed"
        assert result["results"][1]["status"] == "failed"
        assert result["results"][2]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_empty_steps(self, task_feature):
        """Empty steps list returns error."""
        result = await task_feature.run_workflow(steps=[])
        assert result["success"] is False
        assert "non-empty" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_step_format(self, task_feature):
        """Non-dict step is handled gracefully."""
        result = await task_feature.run_workflow(steps=["not a dict"])
        assert result["success"] is False
        assert result["results"][0]["status"] == "failed"
        assert "object" in result["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_step_missing_required_fields(self, task_feature):
        """Step missing feature or skill gets a clear error."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent"},  # missing skill
        ])

        assert result["success"] is False
        assert "requires" in result["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_no_task_manager(self):
        """Without task_manager, returns error."""
        feature = TaskFeature(agent=None)
        # Don't set task_manager
        result = await feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "list_models"},
        ])

        assert result["success"] is False
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_duration_tracking(self, task_feature):
        """Each step and total duration are tracked."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "get_current_model"},
        ])

        assert "total_duration_ms" in result
        assert result["total_duration_ms"] >= 0

        step = result["results"][0]
        assert "duration_ms" in step
        assert step["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_step_metadata_in_results(self, task_feature):
        """Results include feature and skill names for each step."""
        result = await task_feature.run_workflow(steps=[
            {"feature": "model_agent", "skill": "list_models"},
            {"feature": "wallet_feature", "skill": "check_balance"},
        ])

        assert result["results"][0]["feature"] == "model_agent"
        assert result["results"][0]["skill"] == "list_models"
        assert result["results"][1]["feature"] == "wallet_feature"
        assert result["results"][1]["skill"] == "check_balance"


class TestListAvailableSkills:
    """Tests for TaskFeature.list_available_skills."""

    @pytest.mark.asyncio
    async def test_lists_all_features(self, task_feature):
        """Returns all registered features with their skills."""
        result = await task_feature.list_available_skills()

        assert result["success"] is True
        assert result["feature_count"] == 3
        assert "model_agent" in result["features"]
        assert "memory_feature" in result["features"]
        assert "wallet_feature" in result["features"]

    @pytest.mark.asyncio
    async def test_lists_skills_per_feature(self, task_feature):
        """Each feature includes its skill list with descriptions."""
        result = await task_feature.list_available_skills()

        model = result["features"]["model_agent"]
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
        assert result["success"] is False
        assert "not available" in result["error"]
