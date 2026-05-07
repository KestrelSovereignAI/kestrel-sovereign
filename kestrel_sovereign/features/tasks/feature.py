"""
Task Feature - Execute workflows and monitor background tasks.

This feature allows the agent to:
- Execute multi-step workflows across features in a single tool call
- Check status of background tasks
- List pending/running/completed tasks
- Get artifacts from completed tasks
- Cancel tasks

This bridges the gap between the A2A task system (HTTP endpoints)
and the agent's tool system, and provides workflow execution for
multi-step operations.
"""

import asyncio
import copy
import logging
import re
import time
from typing import Dict, Any, Optional, List

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory

logger = logging.getLogger(__name__)

# Pattern for step-output references: {{steps.0.result}}, {{prev.result}}
_STEP_REF_PATTERN = re.compile(r"\{\{(steps\.(\d+)\.(\w+)|prev\.(\w+))\}\}")


class TaskFeature(Feature):
    """
    Feature for executing workflows and managing A2A background tasks.

    Gives the agent the ability to:
    - Execute multi-step plans across features
    - Monitor async operations (selfie generation, LoRA training, etc.)
    - Query task status and results
    """

    def __init__(self, agent=None):
        if agent is not None:
            super().__init__(agent)
        else:
            self.agent = None
            self.name = self.__class__.__name__

        self.task_manager = None

    @property
    def tool_description(self) -> str:
        return (
            "Execute multi-step workflows and monitor background tasks - "
            "run plans across features, check task status, get results"
        )

    async def initialize(self):
        """Initialize with task manager reference."""
        # Task manager will be set by server startup if available
        self.enabled = True

    def set_task_manager(self, task_manager):
        """Set the A2A task manager for querying tasks."""
        self.task_manager = task_manager
        logger.info("TaskFeature connected to TaskManager")

    @tool(
        name="list_available_skills",
        description=(
            "List all available features and their skills that can be used with "
            "run_workflow. Returns feature names, skill names, and descriptions. "
            "Call this first to discover what skills are available before building "
            "a workflow plan."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!list-skills"
    )
    async def list_available_skills(self) -> Dict[str, Any]:
        """
        List all registered features and their individual skills.

        Returns:
            Dict with features and their skills for use with run_workflow.
        """
        if not self.task_manager:
            return {
                "success": False,
                "error": "Task manager not available"
            }

        features = {}
        for agent_id, (agent_card, _handler) in self.task_manager._agents.items():
            skills = []
            for skill in agent_card.skills:
                skills.append({
                    "skill": skill.id,
                    "description": skill.description,
                })
            features[agent_id] = {
                "description": agent_card.description,
                "skills": skills,
            }

        return {
            "success": True,
            "feature_count": len(features),
            "features": features,
        }

    @tool(
        name="run_workflow",
        description=(
            "Execute a multi-step plan across features. Each step runs a specific "
            "feature skill with arguments. All steps execute sequentially and results "
            "are returned together. Use this instead of making individual subagent calls "
            "when you need to gather information from multiple features. "
            "Steps format: [{\"feature\": \"feature_name\", \"skill\": \"skill_name\", \"args\": {}}]. "
            "Feature names match the tool names shown in your available tools (e.g., model_agent, "
            "memory_feature, wallet_feature). Skill names are the individual tool methods within "
            "each feature (e.g., list_models, memory_status, check_balance). "
            "Args can reference prior step outputs with {{steps.N.result}} or {{prev.result}}. "
            "Steps can optionally include max_retries (default 0) and retry_delay_ms (default 1000)."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!run-workflow"
    )
    async def run_workflow(self, steps: list) -> Dict[str, Any]:
        """
        Execute a multi-step workflow plan.

        Args:
            steps: List of workflow steps. Each step is an object with 'feature' (the feature's tool_name like 'model_agent'), 'skill' (the tool method name like 'list_models'), and optional 'args' (dict of arguments to pass).
                   Args values can include {{steps.N.result}} or {{prev.result}} placeholders
                   to reference outputs from earlier steps.
                   Optional 'max_retries' (int, default 0) and 'retry_delay_ms' (int, default 1000).

        Returns:
            Consolidated results from all steps with per-step status.
        """
        if not self.task_manager:
            return {
                "success": False,
                "error": "Task manager not available"
            }

        if not steps or not isinstance(steps, list):
            return {
                "success": False,
                "error": "Steps must be a non-empty list of {feature, skill, args} objects"
            }

        workflow_start = time.time()
        results = []

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                results.append({
                    "step": i,
                    "status": "failed",
                    "error": f"Step must be an object, got {type(step).__name__}"
                })
                continue

            feature_name = step.get("feature")
            skill_name = step.get("skill")
            raw_args = step.get("args", {})
            max_retries = step.get("max_retries", 0)
            retry_delay_ms = step.get("retry_delay_ms", 1000)

            if not feature_name or not skill_name:
                results.append({
                    "step": i,
                    "status": "failed",
                    "error": "Step requires 'feature' and 'skill' fields"
                })
                continue

            # Resolve step-output references in args
            args = self._resolve_step_refs(
                raw_args if isinstance(raw_args, dict) else {},
                results,
                i,
            )

            step_start = time.time()
            last_error = None
            attempts = 1 + max(0, int(max_retries))

            for attempt in range(attempts):
                try:
                    task = await self.task_manager.execute_skill(
                        agent_id=feature_name,
                        skill_id=skill_name,
                        args=args,
                        sync=True,
                    )

                    # Extract result data from task artifacts
                    result_data = None
                    if task.artifacts:
                        for artifact in task.artifacts:
                            if artifact.parts:
                                for part in artifact.parts:
                                    if hasattr(part, 'data'):
                                        result_data = part.data

                    step_duration = int((time.time() - step_start) * 1000)
                    results.append({
                        "step": i,
                        "feature": feature_name,
                        "skill": skill_name,
                        "status": task.status.state.value,
                        "result": result_data,
                        "duration_ms": step_duration,
                        "attempts": attempt + 1,
                    })
                    logger.info(
                        f"[WORKFLOW] Step {i}: {feature_name}.{skill_name} "
                        f"-> {task.status.state.value} ({step_duration}ms, attempt {attempt + 1})"
                    )
                    last_error = None
                    break  # Success — exit retry loop

                except Exception as e:
                    last_error = e
                    if attempt < attempts - 1:
                        delay_s = retry_delay_ms / 1000.0
                        logger.warning(
                            f"[WORKFLOW] Step {i}: {feature_name}.{skill_name} "
                            f"attempt {attempt + 1} failed: {e}, retrying in {delay_s}s"
                        )
                        await asyncio.sleep(delay_s)

            # All attempts exhausted
            if last_error is not None:
                step_duration = int((time.time() - step_start) * 1000)
                results.append({
                    "step": i,
                    "feature": feature_name,
                    "skill": skill_name,
                    "status": "failed",
                    "error": str(last_error),
                    "duration_ms": step_duration,
                    "attempts": attempts,
                })
                logger.error(
                    f"[WORKFLOW] Step {i}: {feature_name}.{skill_name} "
                    f"failed after {attempts} attempt(s): {last_error} ({step_duration}ms)"
                )

        total_duration = int((time.time() - workflow_start) * 1000)
        completed = sum(1 for r in results if r.get("status") == "completed")
        failed = sum(1 for r in results if r.get("status") == "failed")

        logger.info(
            f"[WORKFLOW] Complete: {completed}/{len(steps)} succeeded, "
            f"{failed} failed, {total_duration}ms total"
        )

        return {
            "success": failed == 0,
            "workflow_steps": len(steps),
            "completed": completed,
            "failed": failed,
            "total_duration_ms": total_duration,
            "results": results,
        }

    @staticmethod
    def _resolve_step_refs(
        args: Dict[str, Any],
        prior_results: list,
        current_step: int,
    ) -> Dict[str, Any]:
        """
        Resolve {{steps.N.field}} and {{prev.field}} references in step args.

        Performs a deep copy so original step definitions are not mutated.
        Only string values are resolved; non-string values pass through unchanged.
        """
        if not prior_results:
            return args

        resolved = copy.deepcopy(args)

        def _resolve_value(val):
            if not isinstance(val, str):
                return val

            def _replacer(match):
                full = match.group(0)
                if match.group(2) is not None:
                    # {{steps.N.field}}
                    idx = int(match.group(2))
                    field = match.group(3)
                else:
                    # {{prev.field}}
                    idx = current_step - 1
                    field = match.group(4)

                if idx < 0 or idx >= len(prior_results):
                    logger.warning(f"[WORKFLOW] Unresolved ref {full}: step {idx} not available")
                    return full
                step_result = prior_results[idx]
                if field not in step_result:
                    logger.warning(f"[WORKFLOW] Unresolved ref {full}: field '{field}' not in step {idx}")
                    return full

                replacement = step_result[field]
                # If the entire string is a single reference, return the raw value
                # (preserves dicts/lists instead of stringifying)
                if match.start() == 0 and match.end() == len(val):
                    return replacement
                return str(replacement)

            result = _STEP_REF_PATTERN.sub(_replacer, val)
            # Handle the case where _replacer returned a non-string (whole-value reference)
            # re.sub always returns a string, so check if we had a single whole-value ref
            single_match = _STEP_REF_PATTERN.fullmatch(val)
            if single_match:
                return _replacer(single_match)
            return result

        def _resolve_dict(d):
            for key, val in d.items():
                if isinstance(val, str):
                    d[key] = _resolve_value(val)
                elif isinstance(val, dict):
                    _resolve_dict(val)
                elif isinstance(val, list):
                    _resolve_list(val)

        def _resolve_list(lst):
            for i, val in enumerate(lst):
                if isinstance(val, str):
                    lst[i] = _resolve_value(val)
                elif isinstance(val, dict):
                    _resolve_dict(val)
                elif isinstance(val, list):
                    _resolve_list(val)

        _resolve_dict(resolved)
        return resolved

    @tool(
        name="check_task_status",
        description="Check the status of a background task by ID.",
        category=ToolCategory.UTILITY,
        command_prefix="!task-status"
    )
    async def check_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Check the status of a specific task.

        Args:
            task_id: The task ID to check

        Returns:
            {
                "task_id": str,
                "status": "submitted|working|completed|failed|canceled",
                "message": str (status message),
                "progress": float (0-1 if available),
                "artifacts": list (if completed),
                "error": str (if failed)
            }
        """
        if not self.task_manager:
            return {
                "success": False,
                "error": "Task manager not available"
            }

        try:
            task = await self.task_manager.get_task(task_id)

            if not task:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found"
                }

            # Extract status message
            status_message = None
            if task.status.message and task.status.message.parts:
                status_message = task.status.message.parts[0].text

            # Extract artifacts
            artifacts = []
            if task.artifacts:
                for artifact in task.artifacts:
                    artifact_data = {
                        "name": artifact.name,
                        "description": artifact.description,
                    }
                    # Extract data from parts
                    if artifact.parts:
                        for part in artifact.parts:
                            if hasattr(part, 'data'):
                                artifact_data["data"] = part.data
                    artifacts.append(artifact_data)

            return {
                "success": True,
                "task_id": task_id,
                "status": task.status.state.value,
                "message": status_message,
                "task_type": task.metadata.get("task_type") if task.metadata else None,
                "artifacts": artifacts,
                "created_at": task.id[:8],  # Task ID prefix contains timestamp info
            }

        except Exception as e:
            logger.error(f"Failed to check task {task_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @tool(
        name="list_my_tasks",
        description="List background tasks, optionally filtered by status or type.",
        category=ToolCategory.UTILITY,
        command_prefix="!tasks"
    )
    async def list_my_tasks(
        self,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        List tasks, optionally filtered.

        Args:
            status: Filter by status (submitted, working, completed, failed, canceled)
            task_type: Filter by type (selfie_generation, lora_training, etc.)
            limit: Maximum number of tasks to return

        Returns:
            {
                "success": True,
                "tasks": [
                    {"task_id": str, "status": str, "task_type": str, "message": str},
                    ...
                ],
                "total": int
            }
        """
        if not self.task_manager:
            return {
                "success": False,
                "error": "Task manager not available"
            }

        try:
            from kestrel_sovereign.a2a.types import TaskState

            # Get pending tasks (TaskManager uses get_pending_tasks, not list_tasks)
            tasks = await self.task_manager.get_pending_tasks(limit=limit)

            # Filter by status if specified
            if status:
                try:
                    task_state = TaskState(status)
                    tasks = [t for t in tasks if t.status.state == task_state]
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid status: {status}. Valid: submitted, working, completed, failed, canceled"
                    }

            # Filter by type if specified
            if task_type:
                tasks = [t for t in tasks if t.metadata and t.metadata.get("task_type") == task_type]

            # Format response
            task_list = []
            for task in tasks:
                status_msg = None
                if task.status.message and task.status.message.parts:
                    status_msg = task.status.message.parts[0].text

                task_list.append({
                    "task_id": task.id,
                    "status": task.status.state.value,
                    "task_type": task.metadata.get("task_type") if task.metadata else None,
                    "message": status_msg,
                })

            return {
                "success": True,
                "tasks": task_list,
                "total": len(task_list)
            }

        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @tool(
        name="get_task_result",
        description="Get the result/artifacts from a completed task.",
        category=ToolCategory.UTILITY,
        command_prefix="!task-result"
    )
    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        Get results from a completed task.

        Args:
            task_id: The task ID to get results from

        Returns:
            {
                "success": True,
                "status": "completed",
                "artifacts": [
                    {"name": str, "data": {...}},
                    ...
                ]
            }
        """
        result = await self.check_task_status(task_id)

        if not result.get("success"):
            return result

        if result.get("status") != "completed":
            return {
                "success": False,
                "status": result.get("status"),
                "error": f"Task not completed yet. Current status: {result.get('status')}"
            }

        return {
            "success": True,
            "status": "completed",
            "task_type": result.get("task_type"),
            "artifacts": result.get("artifacts", []),
            "message": result.get("message")
        }

    @tool(
        name="cancel_task",
        description="Cancel a pending or running task.",
        category=ToolCategory.UTILITY,
        command_prefix="!cancel-task"
    )
    async def cancel_task(self, task_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel a task.

        Args:
            task_id: The task ID to cancel
            reason: Optional reason for cancellation

        Returns:
            {"success": True, "status": "canceled"}
        """
        if not self.task_manager:
            return {
                "success": False,
                "error": "Task manager not available"
            }

        try:
            # Check task exists and is cancelable
            task = await self.task_manager.get_task(task_id)

            if not task:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found"
                }

            from kestrel_sovereign.a2a.types import TaskState
            if task.status.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED):
                return {
                    "success": False,
                    "error": f"Cannot cancel task in state: {task.status.state.value}"
                }

            # Cancel it
            await self.task_manager.cancel_task(task_id, reason=reason)

            return {
                "success": True,
                "task_id": task_id,
                "status": "canceled"
            }

        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @tool(
        name="wait_for_task",
        description="Wait for a task to complete and return its result.",
        category=ToolCategory.UTILITY,
        command_prefix="!wait-task"
    )
    async def wait_for_task(
        self,
        task_id: str,
        timeout_seconds: int = 300,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete.

        Args:
            task_id: The task ID to wait for
            timeout_seconds: Maximum time to wait (default 5 minutes)
            poll_interval: Seconds between status checks

        Returns:
            Task result if completed, or timeout error
        """
        import asyncio

        if not self.task_manager:
            return {
                "success": False,
                "error": "Task manager not available"
            }

        elapsed = 0
        while elapsed < timeout_seconds:
            result = await self.check_task_status(task_id)

            if not result.get("success"):
                return result

            status = result.get("status")

            if status == "completed":
                return {
                    "success": True,
                    "status": "completed",
                    "task_type": result.get("task_type"),
                    "artifacts": result.get("artifacts", []),
                    "message": result.get("message"),
                    "waited_seconds": elapsed
                }

            if status in ("failed", "canceled"):
                return {
                    "success": False,
                    "status": status,
                    "error": result.get("message", f"Task {status}"),
                    "waited_seconds": elapsed
                }

            # Still running, wait and poll again
            logger.info(f"Task {task_id} status: {status}, waiting...")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return {
            "success": False,
            "error": f"Timeout after {timeout_seconds}s. Task status: {status}",
            "task_id": task_id
        }
