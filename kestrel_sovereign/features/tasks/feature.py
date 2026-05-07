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

@tool methods return ``kestrel_sdk.tools.result.ToolResult`` per the
kestrel-sovereign #1042 narration-honesty contract (see #1061).
"""

import asyncio
import copy
import logging
import re
import time
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)

# Pattern for step-output references: {{steps.0.result}}, {{prev.result}}
_STEP_REF_PATTERN = re.compile(r"\{\{(steps\.(\d+)\.(\w+)|prev\.(\w+))\}\}")

# Terminal task states that block cancellation. The set is duplicated
# here (rather than imported at module load) because importing
# ``kestrel_sovereign.a2a.types`` at decoration time triggers a circular
# import in some test fixtures; the values themselves are stable wire
# tokens.
_TERMINAL_STATES = {"completed", "failed", "canceled"}


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

    # ------------------------------------------------------------------
    # Internal helpers
    #
    # The @tool methods are thin ToolResult-returning wrappers around
    # these dict-returning helpers. Keeping the helpers private means
    # tools that invoke other tools internally (e.g. ``wait_for_task``
    # polling ``check_task_status``) don't have to unpack a ToolResult
    # envelope just to read state.
    # ------------------------------------------------------------------

    async def _get_task_status_data(self, task_id: str) -> Dict[str, Any]:
        """Fetch a task and shape its status into a dict.

        Returns either ``{"ok": False, "error": str}`` or
        ``{"ok": True, "task_id": str, "status": str, "message": str|None,
        "task_type": str|None, "artifacts": list, "created_at": str}``.

        ``ok`` is the local discriminator — the @tool wrappers translate
        it into ToolResult.ok / ToolResult.failed.
        """
        if not self.task_manager:
            return {"ok": False, "error": "Task manager not available"}

        try:
            task = await self.task_manager.get_task(task_id)
        except Exception as e:
            logger.error(f"Failed to fetch task {task_id}: {e}")
            return {"ok": False, "error": str(e)}

        if not task:
            return {"ok": False, "error": f"Task {task_id} not found"}

        status_message = None
        if task.status.message and task.status.message.parts:
            status_message = task.status.message.parts[0].text

        artifacts: List[Dict[str, Any]] = []
        if task.artifacts:
            for artifact in task.artifacts:
                artifact_data: Dict[str, Any] = {
                    "name": artifact.name,
                    "description": artifact.description,
                }
                if artifact.parts:
                    for part in artifact.parts:
                        if hasattr(part, "data"):
                            artifact_data["data"] = part.data
                artifacts.append(artifact_data)

        return {
            "ok": True,
            "task_id": task_id,
            "status": task.status.state.value,
            "message": status_message,
            "task_type": task.metadata.get("task_type") if task.metadata else None,
            "artifacts": artifacts,
            "created_at": task.id[:8],  # task ID prefix encodes timestamp
        }

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
    async def list_available_skills(self) -> ToolResult:
        """
        List all registered features and their individual skills.

        Returns:
            ToolResult.ok with feature/skill catalog when the task manager
            is available; ToolResult.failed otherwise.
        """
        if not self.task_manager:
            return ToolResult.failed(
                "Task manager not available",
                data={"reason": "TaskFeature has no connected TaskManager"},
            )

        features: Dict[str, Dict[str, Any]] = {}
        try:
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
        except Exception as e:
            logger.error(f"list_available_skills failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        total_skills = sum(len(f["skills"]) for f in features.values())
        return ToolResult.ok(
            confirmation=(
                f"Catalog: {len(features)} feature(s), {total_skills} skill(s) "
                f"available for run_workflow"
            ),
            data={
                "feature_count": len(features),
                "skill_count": total_skills,
                "features": features,
            },
        )

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
    async def run_workflow(self, steps: list) -> ToolResult:
        """
        Execute a multi-step workflow plan.

        Args:
            steps: List of workflow steps. Each step is an object with 'feature' (the feature's tool_name like 'model_agent'), 'skill' (the tool method name like 'list_models'), and optional 'args' (dict of arguments to pass).
                   Args values can include {{steps.N.result}} or {{prev.result}} placeholders
                   to reference outputs from earlier steps.
                   Optional 'max_retries' (int, default 0) and 'retry_delay_ms' (int, default 1000).

        Returns:
            ToolResult — OK if every step completed, PARTIAL if any step
            failed (so the LLM cannot claim "all steps ran" while a step
            errored), ERROR if input was malformed or no steps ran.
        """
        if not self.task_manager:
            return ToolResult.failed("Task manager not available")

        if not steps or not isinstance(steps, list):
            return ToolResult.failed(
                "Steps must be a non-empty list of {feature, skill, args} objects",
                data={"received_type": type(steps).__name__},
            )

        workflow_start = time.time()
        results: List[Dict[str, Any]] = []

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
            max_retries_raw = step.get("max_retries", 0)
            retry_delay_ms_raw = step.get("retry_delay_ms", 1000)

            if not feature_name or not skill_name:
                results.append({
                    "step": i,
                    "status": "failed",
                    "error": "Step requires 'feature' and 'skill' fields"
                })
                continue

            # Coerce numeric step controls — accept str-typed JSON inputs
            # but reject anything that doesn't read as a non-negative int.
            try:
                max_retries = max(0, int(max_retries_raw))
            except (TypeError, ValueError):
                results.append({
                    "step": i,
                    "status": "failed",
                    "error": f"max_retries must be an integer, got {max_retries_raw!r}",
                })
                continue
            try:
                retry_delay_ms = max(0, int(retry_delay_ms_raw))
            except (TypeError, ValueError):
                results.append({
                    "step": i,
                    "status": "failed",
                    "error": f"retry_delay_ms must be an integer, got {retry_delay_ms_raw!r}",
                })
                continue

            args = self._resolve_step_refs(
                raw_args if isinstance(raw_args, dict) else {},
                results,
                i,
            )

            step_start = time.time()
            last_error: Optional[BaseException] = None
            attempts = 1 + max_retries

            for attempt in range(attempts):
                try:
                    task = await self.task_manager.execute_skill(
                        agent_id=feature_name,
                        skill_id=skill_name,
                        args=args,
                        sync=True,
                    )

                    result_data = None
                    if task.artifacts:
                        for artifact in task.artifacts:
                            if artifact.parts:
                                for part in artifact.parts:
                                    if hasattr(part, 'data'):
                                        result_data = part.data

                    step_duration = int((time.time() - step_start) * 1000)

                    # Honesty: A2A's task.status reports the *transport*
                    # outcome — the call returned without raising — not
                    # the *semantic* outcome. A migrated step that
                    # returned ``ToolResult.failed`` lands here with
                    # ``task.status.state == COMPLETED`` and would be
                    # counted as a success in the workflow rollup.
                    # Inspect the wire-shape (status / error fields)
                    # and downgrade. Old-style dict tools with
                    # ``success: False`` are also surfaced as failed so
                    # the rollup is honest during the migration window.
                    semantic_status, semantic_error = self._classify_step_result(
                        task.status.state.value, result_data,
                    )
                    step_record = {
                        "step": i,
                        "feature": feature_name,
                        "skill": skill_name,
                        "status": semantic_status,
                        "result": result_data,
                        "duration_ms": step_duration,
                        "attempts": attempt + 1,
                    }
                    if semantic_error is not None:
                        step_record["error"] = semantic_error
                    results.append(step_record)
                    logger.info(
                        f"[WORKFLOW] Step {i}: {feature_name}.{skill_name} "
                        f"-> {semantic_status} (transport={task.status.state.value}, "
                        f"{step_duration}ms, attempt {attempt + 1})"
                    )
                    last_error = None
                    if semantic_status != "failed":
                        break
                    # Semantic failure with retries left: fall through
                    # to retry like a transport exception.
                    if attempt < attempts - 1:
                        delay_s = retry_delay_ms / 1000.0
                        logger.warning(
                            f"[WORKFLOW] Step {i}: {feature_name}.{skill_name} "
                            f"attempt {attempt + 1} returned failed: {semantic_error}, "
                            f"retrying in {delay_s}s"
                        )
                        # Pop the failed record so a successful retry
                        # writes a clean record (mirrors the transport
                        # retry path which also overwrites).
                        results.pop()
                        await asyncio.sleep(delay_s)
                    else:
                        break

                except Exception as e:
                    last_error = e
                    if attempt < attempts - 1:
                        delay_s = retry_delay_ms / 1000.0
                        logger.warning(
                            f"[WORKFLOW] Step {i}: {feature_name}.{skill_name} "
                            f"attempt {attempt + 1} failed: {e}, retrying in {delay_s}s"
                        )
                        await asyncio.sleep(delay_s)

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
        partial = sum(1 for r in results if r.get("status") == "partial")

        logger.info(
            f"[WORKFLOW] Complete: {completed}/{len(steps)} succeeded, "
            f"{partial} partial, {failed} failed, {total_duration}ms total"
        )

        data = {
            "workflow_steps": len(steps),
            "completed": completed,
            "partial": partial,
            "failed": failed,
            "total_duration_ms": total_duration,
            "results": results,
        }

        # All-clean: every step OK. Cleanest path → OK.
        if failed == 0 and partial == 0 and completed == len(steps):
            return ToolResult.ok(
                confirmation=(
                    f"Workflow complete: {completed}/{len(steps)} step(s) "
                    f"succeeded in {total_duration}ms"
                ),
                data=data,
            )
        # All failures, no successes or partials. PARTIAL would require
        # a confirmation to be honestly speakable — there is none.
        if completed == 0 and partial == 0:
            return ToolResult.failed(
                f"Workflow failed: 0/{len(steps)} step(s) succeeded "
                f"({failed} failed)",
                data=data,
            )
        # Anything else (any mixture of succeeded/partial/failed)
        # → PARTIAL forces the LLM to surface the failed/partial half
        # rather than claim "workflow complete".
        error_parts = []
        if failed:
            error_parts.append(f"{failed} step(s) failed")
        if partial:
            error_parts.append(f"{partial} step(s) partially completed")
        return ToolResult.partial(
            confirmation=(
                f"Workflow partially complete: {completed}/{len(steps)} "
                f"step(s) cleanly succeeded in {total_duration}ms"
            ),
            error=(
                "; ".join(error_parts) + "; see results[*].error for details"
            ),
            data=data,
        )

    @staticmethod
    def _classify_step_result(
        transport_state: str,
        result_data: Any,
    ) -> tuple[str, Optional[str]]:
        """Translate the A2A task state + tool wire-data into a
        workflow-step status.

        A2A's ``task.status.state`` is transport-level: COMPLETED means
        "the python call returned." A migrated tool that returns
        ``ToolResult.failed`` is *transport-completed* but *semantically
        failed*. To keep the workflow rollup honest, this helper
        inspects the wire-shape:

          - ToolResult envelope (``{"status": "ok"|"error"|"partial",
            ...}``) → use the envelope's status
          - Old-style dict with ``success: False`` → "failed"
          - Old-style dict with ``error: "..."`` → "failed"
          - Anything else → defer to the transport state

        Returns ``(status, error_or_None)``. ``status`` is one of
        ``"completed" | "failed" | "partial"``.
        """
        if transport_state == "failed":
            return "failed", None
        if isinstance(result_data, dict):
            envelope_status = result_data.get("status")
            if envelope_status in ("ok",):
                return "completed", None
            if envelope_status == "error":
                return "failed", result_data.get("error")
            if envelope_status == "partial":
                return "partial", result_data.get("error")
            # Pre-migration dict shape — `success: False` or `error: ...`
            if result_data.get("success") is False:
                return "failed", result_data.get("error")
            if result_data.get("error"):
                return "failed", result_data.get("error")
        return transport_state if transport_state != "completed" else "completed", None

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
                    idx = int(match.group(2))
                    field = match.group(3)
                else:
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
                if match.start() == 0 and match.end() == len(val):
                    return replacement
                return str(replacement)

            result = _STEP_REF_PATTERN.sub(_replacer, val)
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
    async def check_task_status(self, task_id: str) -> ToolResult:
        """
        Check the status of a specific task.

        Args:
            task_id: The task ID to check
        """
        data = await self._get_task_status_data(task_id)
        if not data["ok"]:
            return ToolResult.failed(data["error"])

        return ToolResult.ok(
            confirmation=(
                f"Task {task_id[:8]} status: {data['status']}"
                + (f" — {data['message']}" if data["message"] else "")
            ),
            data={
                "task_id": data["task_id"],
                "status": data["status"],
                "message": data["message"],
                "task_type": data["task_type"],
                "artifacts": data["artifacts"],
                "created_at": data["created_at"],
            },
        )

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
    ) -> ToolResult:
        """
        List tasks, optionally filtered.

        Args:
            status: Filter by status (submitted, working, completed, failed, canceled)
            task_type: Filter by type (selfie_generation, lora_training, etc.)
            limit: Maximum number of tasks to return (the request — actual
                   count returned may be lower if fewer tasks exist).
        """
        if not self.task_manager:
            return ToolResult.failed("Task manager not available")

        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"limit must be an integer, got {limit!r}"
            )
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")

        try:
            from kestrel_sovereign.a2a.types import TaskState

            tasks = await self.task_manager.get_pending_tasks(limit=limit_val)

            if status:
                try:
                    task_state = TaskState(status)
                except ValueError:
                    return ToolResult.failed(
                        f"Invalid status: {status!r}. Valid: submitted, "
                        "working, completed, failed, canceled"
                    )
                tasks = [t for t in tasks if t.status.state == task_state]

            if task_type:
                tasks = [
                    t for t in tasks
                    if t.metadata and t.metadata.get("task_type") == task_type
                ]

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
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        # Honesty: phrase the confirmation as the request + the actual
        # count so the LLM cannot claim "Retrieved N tasks" when fewer
        # came back.
        filter_clause = ""
        if status:
            filter_clause += f" with status={status}"
        if task_type:
            filter_clause += f" of type={task_type}"
        return ToolResult.ok(
            confirmation=(
                f"Retrieved {len(task_list)} task(s){filter_clause} "
                f"(limit requested: {limit_val})"
            ),
            data={
                "tasks": task_list,
                "count": len(task_list),
                "limit_requested": limit_val,
                "filter_status": status,
                "filter_task_type": task_type,
            },
        )

    @tool(
        name="get_task_result",
        description="Get the result/artifacts from a completed task.",
        category=ToolCategory.UTILITY,
        command_prefix="!task-result"
    )
    async def get_task_result(self, task_id: str) -> ToolResult:
        """
        Get results from a completed task.

        Args:
            task_id: The task ID to get results from
        """
        data = await self._get_task_status_data(task_id)
        if not data["ok"]:
            return ToolResult.failed(data["error"])

        if data["status"] != "completed":
            # Surface the actual status in the error so the LLM doesn't
            # narrate a "results retrieved" success.
            return ToolResult.failed(
                f"Task not completed yet. Current status: {data['status']}",
                data={
                    "task_id": data["task_id"],
                    "status": data["status"],
                    "message": data["message"],
                },
            )

        return ToolResult.ok(
            confirmation=(
                f"Retrieved {len(data['artifacts'])} artifact(s) from "
                f"completed task {task_id[:8]}"
            ),
            data={
                "task_id": data["task_id"],
                "status": data["status"],
                "task_type": data["task_type"],
                "artifacts": data["artifacts"],
                "message": data["message"],
            },
        )

    @tool(
        name="cancel_task",
        description="Cancel a pending or running task.",
        category=ToolCategory.UTILITY,
        command_prefix="!cancel-task"
    )
    async def cancel_task(self, task_id: str, reason: Optional[str] = None) -> ToolResult:
        """
        Cancel a task.

        Args:
            task_id: The task ID to cancel
            reason: Optional reason for cancellation
        """
        if not self.task_manager:
            return ToolResult.failed("Task manager not available")

        try:
            task = await self.task_manager.get_task(task_id)
        except Exception as e:
            logger.error(f"Failed to fetch task {task_id} for cancel: {e}")
            return ToolResult.failed(str(e))

        if not task:
            return ToolResult.failed(f"Task {task_id} not found")

        current_state = task.status.state.value
        if current_state in _TERMINAL_STATES:
            return ToolResult.failed(
                f"Cannot cancel task in state: {current_state}",
                data={"task_id": task_id, "status": current_state},
            )

        try:
            await self.task_manager.cancel_task(task_id, reason=reason)
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}", exc_info=True)
            return ToolResult.failed(
                str(e),
                data={"task_id": task_id, "status_before": current_state},
            )

        return ToolResult.ok(
            confirmation=f"Cancelled task {task_id[:8]} (was: {current_state})",
            data={
                "task_id": task_id,
                "status": "canceled",
                "status_before": current_state,
                "reason": reason,
            },
        )

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
    ) -> ToolResult:
        """
        Wait for a task to complete.

        Args:
            task_id: The task ID to wait for
            timeout_seconds: Maximum time to wait (default 5 minutes)
            poll_interval: Seconds between status checks
        """
        if not self.task_manager:
            return ToolResult.failed("Task manager not available")

        try:
            timeout_val = int(timeout_seconds)
            poll_val = int(poll_interval)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"timeout_seconds and poll_interval must be integers, "
                f"got {timeout_seconds!r}, {poll_interval!r}"
            )
        if timeout_val < 0 or poll_val <= 0:
            return ToolResult.failed(
                "timeout_seconds must be >= 0 and poll_interval must be > 0"
            )

        elapsed = 0
        last_status: Optional[str] = None
        while elapsed < timeout_val:
            data = await self._get_task_status_data(task_id)
            if not data["ok"]:
                return ToolResult.failed(
                    data["error"],
                    data={"task_id": task_id, "waited_seconds": elapsed},
                )

            last_status = data["status"]

            if last_status == "completed":
                return ToolResult.ok(
                    confirmation=(
                        f"Task {task_id[:8]} completed after {elapsed}s "
                        f"({len(data['artifacts'])} artifact(s))"
                    ),
                    data={
                        "task_id": data["task_id"],
                        "status": "completed",
                        "task_type": data["task_type"],
                        "artifacts": data["artifacts"],
                        "message": data["message"],
                        "waited_seconds": elapsed,
                    },
                )

            if last_status in ("failed", "canceled"):
                return ToolResult.failed(
                    data.get("message") or f"Task {last_status}",
                    data={
                        "task_id": data["task_id"],
                        "status": last_status,
                        "waited_seconds": elapsed,
                    },
                )

            logger.info(f"Task {task_id} status: {last_status}, waiting...")
            await asyncio.sleep(poll_val)
            elapsed += poll_val

        return ToolResult.failed(
            f"Timeout after {timeout_val}s. Task status: {last_status}",
            data={
                "task_id": task_id,
                "status": last_status,
                "waited_seconds": elapsed,
                "timeout_seconds": timeout_val,
            },
        )
