"""
Task Monitoring Feature - Agent access to A2A task status.

This feature allows the agent to:
- Check status of background tasks it submitted
- List pending/running/completed tasks
- Get artifacts from completed tasks
- Cancel tasks

This bridges the gap between the A2A task system (HTTP endpoints)
and the agent's tool system.
"""

import logging
from typing import Dict, Any, Optional, List

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class TaskFeature(Feature):
    """
    Feature for monitoring and managing A2A background tasks.

    Gives the agent visibility into async operations like:
    - Selfie generation
    - LoRA training
    - Image processing
    - Any other A2A task
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
            "Monitor background tasks - check status of async operations, "
            "list pending tasks, get results from completed tasks"
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
