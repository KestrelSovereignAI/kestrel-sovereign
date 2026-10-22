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
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from a2a.helpers.proto_helpers import new_text_artifact, new_text_message
from a2a.types import Role, TaskState
from a2a.utils.errors import InvalidAgentResponseError
from google.protobuf.json_format import MessageToDict
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign.a2a.lifecycle import (
    TERMINAL_STATES,
    task_state_from_label,
    task_state_label,
)
from kestrel_sovereign.a2a.projections import (
    embedded_json_value,
    message_text,
    task_presentation,
)
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.features.base import Feature, is_flat_toolresult_envelope, tool
from kestrel_sovereign.features.tasks.wait_provider import TaskWaitable
from kestrel_sovereign.storage.db.interface import DatabaseError

logger = logging.getLogger(__name__)

# Pattern for step-output references: {{steps.0.result}}, {{prev.result}}
_STEP_REF_PATTERN = re.compile(r"\{\{(steps\.(\d+)\.(\w+)|prev\.(\w+))\}\}")

# ``list_my_tasks`` filters ``task_type`` (a metadata field, no SQL column) in
# Python. When that filter is active we over-fetch at least this many rows so
# the caller's ``limit`` bounds matching rows, not pre-filter rows (a larger
# requested limit is still honoured). A floor, not a ceiling — it just keeps a
# small requested limit from missing matches beyond the first page.
_TASK_TYPE_FILTER_FETCH_CAP = 1000


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

    @property
    def promote_tools_on_startup(self) -> bool:
        return True

    async def initialize(self):
        """Initialize with task manager reference."""
        # Task manager will be set by server startup if available
        self.enabled = True

    async def post_all_features_loaded(self, agent):
        """Register the ``task:`` Waitable provider with the wait engine.

        Lets the generic ``wait("task:<task_id>")`` tool dispatch here, and
        lets the wait reconciler enumerate this kind for signal-resume.
        """
        registry = getattr(agent, "wait_registry", None)
        if registry is not None:
            # Record ownership so base shutdown()/boot rollback unregisters it.
            self._register_wait_provider(registry, TaskWaitable(self), replace=True)

    def set_task_manager(self, task_manager):
        """Set the A2A task manager for querying tasks."""
        self.task_manager = task_manager
        logger.info("TaskFeature connected to TaskManager")

    def _recipient_agent_id(self) -> str:
        """Return the runtime-bound inbox principal for every task operation."""

        value = self._durable_agent_id()
        if value is None:
            raise ValueError("Task recipient identity unavailable")
        return value

    # ------------------------------------------------------------------
    # Internal helpers
    #
    # The @tool methods are thin ToolResult-returning wrappers around
    # these dict-returning helpers. Keeping the helpers private means
    # internal callers (e.g. the TaskWaitable provider reading task status)
    # don't have to unpack a ToolResult envelope just to read state.
    # ------------------------------------------------------------------

    def _durable_agent_id(self) -> Optional[str]:
        """Return the host's durable identity, never a display-derived name.

        The agent's ``did`` through the shared guard; ``None`` when it has
        none. This used to accept ``agent_id`` as a second choice. On every
        real agent that is a property returning the DID, so the fallback
        only ever admitted a test double carrying a bare ``agent_id`` — the
        shape the guard refuses everywhere else (#3246).
        """

        from kestrel_sovereign.features.storage_access import (
            AgentIdentityUnavailable,
            resolve_scoped_agent_did,
        )

        try:
            return resolve_scoped_agent_did(self.agent)
        except AgentIdentityUnavailable:
            return None

    async def _get_task_status_data(self, task_id: str) -> Dict[str, Any]:
        """Present a native task with its committed sender and real status time."""
        if not self.task_manager:
            return {"ok": False, "error": "Task manager not available"}
        try:
            snapshot = await self.task_manager.get_task_snapshot(
                task_id, recipient_agent_id=self._recipient_agent_id(),
            )
        except DatabaseError as error:
            if is_execution_control_error(error):
                raise
            logger.error("Failed to fetch task %s: %s", task_id, error)
            return {"ok": False, "error": str(error)}
        if snapshot is None:
            return {"ok": False, "error": f"Task {task_id} not found"}
        return {"ok": True, **task_presentation(snapshot)}

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

        features = {
            provider_id: {"skills": [{"skill": skill.id, "description": skill.description}
                                     for skill in skills]}
            for provider_id, skills in self.task_manager.get_provider_skills().items()
        }

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
            "Use the exact feature identifiers returned by list_available_skills. "
            "Skill names are the individual tool methods within "
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
            steps: List of workflow steps. Each step is an object with 'feature' (the registered identifier returned by list_available_skills), 'skill' (the tool method name like 'list_models'), and optional 'args' (dict of arguments to pass).
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

            try:
                normalized_args = self._normalize_step_args(
                    raw_args,
                    step_index=i,
                    feature_name=feature_name,
                    skill_name=skill_name,
                )
            except ValueError as e:
                if is_execution_control_error(e):
                    raise
                results.append({
                    "step": i,
                    "feature": feature_name,
                    "skill": skill_name,
                    "status": "failed",
                    "error": str(e),
                })
                continue

            args = self._resolve_step_refs(normalized_args, results, i)

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
                                    recognized, value = embedded_json_value(part)
                                    if recognized:
                                        result_data = value

                    step_duration = int((time.time() - step_start) * 1000)

                    # Preserve native rejection/failure reasons and inspect
                    # the business result of a completed tool before rollup.
                    semantic_status, semantic_error = self._classify_step_result(
                        task_state_label(task.status.state), result_data,
                        failure_message=(message_text(task.status.message)
                                         if task.status.HasField("message") else None),
                    )
                    # All native artifacts are decoded through the shared
                    # JSON projector before constructing the workflow result.
                    serialized_result = self._serialize_step_payload(result_data)
                    step_record = {
                        "step": i,
                        "feature": feature_name,
                        "skill": skill_name,
                        "status": semantic_status,
                        "result": serialized_result,
                        "duration_ms": step_duration,
                        "attempts": attempt + 1,
                    }
                    if semantic_error is not None:
                        step_record["error"] = semantic_error
                    results.append(step_record)
                    logger.info(
                        f"[WORKFLOW] Step {i}: {feature_name}.{skill_name} "
                        f"-> {semantic_status} (transport={task_state_label(task.status.state)}, "
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
                    if is_execution_control_error(e):
                        raise
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
    def _normalize_step_args(
        raw_args: Any,
        *,
        step_index: int,
        feature_name: str,
        skill_name: str,
    ) -> Dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                decoded = json.loads(raw_args)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Step {step_index} ({feature_name}.{skill_name}) args must be "
                    f"an object/dict or JSON object string, got str: "
                    f"invalid JSON at char {e.pos}"
                ) from e
            if isinstance(decoded, dict):
                return decoded
            raise ValueError(
                f"Step {step_index} ({feature_name}.{skill_name}) args must be "
                f"an object/dict or JSON object string, got str decoding to "
                f"{type(decoded).__name__}"
            )
        raise ValueError(
            f"Step {step_index} ({feature_name}.{skill_name}) args must be "
            f"an object/dict or JSON object string, got {type(raw_args).__name__}"
        )

    @staticmethod
    def _serialize_step_payload(payload: Any) -> Any:
        """Walk a step's result payload and replace any ToolResult
        instance with its dict form.

        Without this, a workflow whose step returned a real ToolResult
        (the production in-process path: DynamicTool.execute stores
        the raw object) ends up with a non-JSON-serializable object
        embedded in run_workflow's data dict. The LLM tool-history
        wire then fails to serialize the whole workflow result.
        """
        if isinstance(payload, ToolResult):
            return payload.to_dict()
        if isinstance(payload, dict):
            return {k: TaskFeature._serialize_step_payload(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [TaskFeature._serialize_step_payload(v) for v in payload]
        if isinstance(payload, tuple):
            return [TaskFeature._serialize_step_payload(v) for v in payload]
        return payload

    @staticmethod
    def _classify_step_result(
        transport_state: str,
        result_data: Any,
        *,
        failure_message: str | None = None,
    ) -> tuple[str, Optional[str]]:
        """Combine native execution state with the returned business outcome.

        A failed/canceled/rejected native task always fails its workflow step.
        A completed tool can still return a partial or unsuccessful business
        result. Recognize the canonical ToolResult envelope before inspecting
        a dictionary return carried by DynamicTool's result wrapper.
        """
        native_state = task_state_from_label(transport_state)
        if native_state in TERMINAL_STATES and native_state != TaskState.TASK_STATE_COMPLETED:
            return "failed", failure_message

        def normalize(candidate: Any) -> Any:
            return candidate.to_dict() if isinstance(candidate, ToolResult) else candidate

        result_data = normalize(result_data)
        if not isinstance(result_data, dict):
            return transport_state, None
        candidates = [result_data]
        if "tool" in result_data and "success" in result_data:
            inner = normalize(result_data.get("result"))
            if isinstance(inner, dict):
                candidates.append(inner)

        for candidate in candidates:
            if not is_flat_toolresult_envelope(candidate):
                continue
            status = candidate["status"]
            if status == ToolResultStatus.OK:
                return "completed", None
            if status == ToolResultStatus.ERROR:
                return "failed", candidate.get("error")
            if status == ToolResultStatus.PARTIAL:
                return "partial", candidate.get("error")

        for candidate in candidates:
            if candidate.get("success") is False or (
                candidate.get("error") and candidate.get("success") is not True
            ):
                return "failed", candidate.get("error")
        return transport_state, None

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

            single_match = _STEP_REF_PATTERN.fullmatch(val)
            if single_match:
                return _replacer(single_match)
            result = _STEP_REF_PATTERN.sub(_replacer, val)
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
        description="Check the status of a background task by ID. Returns two distinct content fields: request_content (what was ASKED — the inbound sender's message) and message (the REPLY that has been written back, if any).",
        category=ToolCategory.UTILITY,
        command_prefix="!task-status"
    )
    async def check_task_status(self, task_id: str) -> ToolResult:
        """
        Check the status of a specific task.

        Args:
            task_id: The task ID to check

        Returns two distinct content fields in ``data``:
            request_content: What was ASKED — the inbound sender's original
                             message (load-bearing for tasks awaiting a reply).
            message: The REPLY that has been written back to the task, if any.
        """
        data = await self._get_task_status_data(task_id)
        if not data["ok"]:
            return ToolResult.failed(data["error"])

        # The confirmation must show the LLM what was ASKED (for inbound
        # tasks awaiting a reply, this is the load-bearing fact) AND any
        # reply that's been written. The data dict keeps both fields so
        # downstream rendering can show either; the confirmation prefers
        # request_content for non-terminal states and status_message for
        # terminal ones — the actionable text on each side.
        bits = [f"Task {task_id[:8]} status: {data['status']}"]
        if data["sender"]:
            bits.append(f"from {data['sender']}")
        if data["request_content"]:
            preview = data["request_content"]
            if len(preview) > 200:
                preview = preview[:197] + "..."
            bits.append(f"request: {preview!r}")
        if data["message"]:
            preview = data["message"]
            if len(preview) > 200:
                preview = preview[:197] + "..."
            bits.append(f"reply: {preview!r}")
        return ToolResult.ok(
            confirmation=" — ".join(bits),
            data={key: value for key, value in data.items() if key != "ok"},
        )

    @tool(
        name="list_my_tasks",
        description="List background tasks, optionally filtered by status or type. status is a native lifecycle label such as submitted, working, input-required, auth-required, completed, failed, canceled, or rejected (case-insensitive). With no status, returns the pending (submitted) inbox; a status filter queries tasks across ALL states.",
        category=ToolCategory.UTILITY,
        command_prefix="!tasks"
    )
    async def list_my_tasks(
        self, status: Optional[str] = None, task_type: Optional[str] = None,
        limit: int = 10,
    ) -> ToolResult:
        """List native inbox tasks using committed sender ownership."""
        if not self.task_manager:
            return ToolResult.failed("Task manager not available")
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(f"limit must be an integer, got {limit!r}")
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")
        try:
            task_state = task_state_from_label(status.strip()) if status else TaskState.TASK_STATE_SUBMITTED
        except ValueError:
            return ToolResult.failed(f"Invalid task status: {status!r}")
        fetch_limit = max(limit_val, _TASK_TYPE_FILTER_FETCH_CAP) if task_type else limit_val
        try:
            snapshots = await self.task_manager.list_task_snapshots(
                recipient_agent_id=self._recipient_agent_id(), status=task_state,
                limit=fetch_limit,
            )
        except DatabaseError as error:
            if is_execution_control_error(error):
                raise
            logger.error("Failed to list tasks: %s", error, exc_info=True)
            return ToolResult.failed(str(error))
        task_list = []
        for snapshot in snapshots:
            task = snapshot.task
            metadata = MessageToDict(task.metadata)
            if task_type and metadata.get("task_type") != task_type:
                continue
            task_list.append(task_presentation(snapshot, include_artifacts=False))
            if len(task_list) == limit_val:
                break
        filters = (f" with status={status}" if status else "") + (f" of type={task_type}" if task_type else "")
        return ToolResult.ok(
            confirmation=f"Retrieved {len(task_list)} task(s){filters} (limit requested: {limit_val})",
            data={"tasks": task_list, "count": len(task_list), "limit_requested": limit_val,
                  "filter_status": status, "filter_task_type": task_type},
        )

    @tool(
        name="get_task_result",
        description="Get the result/artifacts from a completed task. The returned message field is the REPLY content, distinct from request_content (what was originally ASKED) surfaced by check_task_status.",
        category=ToolCategory.UTILITY,
        command_prefix="!task-result"
    )
    async def get_task_result(self, task_id: str) -> ToolResult:
        """
        Get results from a completed task.

        Args:
            task_id: The task ID to get results from

        The returned ``message`` is the REPLY content written back to the
        task — distinct from ``request_content`` (what was originally ASKED),
        which check_task_status surfaces.
        """
        data = await self._get_task_status_data(task_id)
        if not data["ok"]:
            return ToolResult.failed(data["error"])

        if data["status"] != "completed":
            # Surface the actual status in the error so the LLM doesn't
            # narrate a "results retrieved" success.
            return ToolResult.failed(
                f"Task is not completed. Current status: {data['status']}",
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
                "request_content": data["request_content"],
                "request": data["request"],
                "context_id": data["context_id"],
                "skill_hint": data["skill_hint"],
                "sender": data["sender"],
            },
        )

    @tool(
        name="respond_to_a2a_task",
        description=(
            "Respond to an incoming A2A task in your inbox by transitioning "
            "it to a terminal state with your reply text. Use this when "
            "another agent sent you a task via send_a2a_question "
            "(fire-and-resume — sender's turn ended, they wake on the "
            "a2a.question_answered signal when you transition), "
            "send_a2a_message (FYI, brief receipt), or send_a2a_task "
            "(delegated work, full result). The sender's resident question "
            "recovery reads your native result and queues a durable resumption "
            "signal after your terminal reply. Unanswered questions use their "
            "recorded deadline; a later reply produces a late-answer wake."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a respond",
    )
    async def respond_to_a2a_task(
        self, task_id: str, content: str, state: str = "completed",
    ) -> ToolResult:
        """Respond to an inbox task using native terminal states and message IDs."""
        if not self.task_manager:
            return ToolResult.failed("Task manager not available")
        try:
            terminal = task_state_from_label((state or "completed").strip())
        except ValueError:
            return ToolResult.failed(f"Invalid terminal state: {state!r}")
        if terminal not in TERMINAL_STATES:
            return ToolResult.failed(f"State {state!r} is not terminal")
        if terminal == TaskState.TASK_STATE_CANCELED:
            return await self.cancel_task(task_id, reason=content)
        actor = self._durable_agent_id()
        if actor is None:
            return ToolResult.failed("Task response requires this agent's durable identity")
        task = await self.task_manager.get_task_for_recipient(task_id, self._recipient_agent_id())
        if task is None:
            return ToolResult.failed(f"Task {task_id} not found in this agent's task store")
        current = task.status.state
        if current in TERMINAL_STATES:
            return ToolResult.failed(f"Task {task_id} is already terminal: {task_state_label(current)}")
        response = new_text_message(content, role=Role.ROLE_AGENT,
                                    task_id=task.id, context_id=task.context_id)
        try:
            updated = await self.task_manager.update_status(
                task_id=task_id, new_state=terminal, message=response,
                agent_name=actor, recipient_agent_id=actor,
            )
        except (ValueError, PermissionError, DatabaseError) as error:
            if is_execution_control_error(error):
                raise
            logger.error("Failed to respond to task %s: %s", task_id, error)
            return ToolResult.failed(str(error), data={"task_id": task_id, "state_before": task_state_label(current)})
        label = task_state_label(updated.status.state)
        return ToolResult.ok(
            confirmation=f"Responded to task {task_id[:8]} ({task_state_label(current)} → {label})",
            data={"task_id": task_id, "state": label, "state_before": task_state_label(current), "response": content},
        )

    @tool(
        name="attach_artifact_to_a2a_task",
        description=(
            "Attach text output to an incoming task. The first call returns an artifact_id. "
            "For longer output, pass that same artifact_id with each later chunk; "
            "set last_chunk=False until the final chunk. Finish the task with respond_to_a2a_task."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a attach",
    )
    async def attach_artifact_to_a2a_task(
        self, task_id: str, name: str, content: str,
        artifact_id: Optional[str] = None, last_chunk: bool = True,
    ) -> ToolResult:
        """Create an artifact or append text to the artifact ID already returned."""
        if not self.task_manager:
            return ToolResult.failed("Task manager not available")
        actor = self._durable_agent_id()
        if actor is None:
            return ToolResult.failed("Task artifact attachment requires this agent's durable identity")
        if artifact_id is not None and (not isinstance(artifact_id, str) or not artifact_id.strip()):
            return ToolResult.failed("artifact_id must be the non-empty ID returned by the first chunk")
        artifact = new_text_artifact(name, content, artifact_id=artifact_id)
        append = artifact_id is not None
        try:
            updated = await self.task_manager.add_artifact(
                task_id, artifact, agent_name=actor, recipient_agent_id=actor,
                append=append, last_chunk=last_chunk,
            )
        except (ValueError, TypeError, PermissionError, DatabaseError, InvalidAgentResponseError) as error:
            if is_execution_control_error(error):
                raise
            logger.error("Failed to attach artifact to task %s: %s", task_id, error)
            return ToolResult.failed(str(error), data={"task_id": task_id, "artifact_id": artifact.artifact_id})
        return ToolResult.ok(
            confirmation=f"Attached {len(content)} chars to artifact {artifact.artifact_id} on task {task_id[:8]}",
            data={"task_id": task_id, "artifact_id": artifact.artifact_id,
                  "append": append, "last_chunk": last_chunk,
                  "content_chars": len(content), "artifact_count": len(updated.artifacts)},
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

        actor_agent_id = self._durable_agent_id()
        if actor_agent_id is None:
            return ToolResult.failed(
                "Task cancellation requires this agent's durable identity",
                data={"task_id": task_id},
            )

        # Check the sender-owned route before touching the local task table.
        # SQLite agents normally cannot see a recipient's row, but hosted
        # agents may share one PostgreSQL table.  In that topology, using mere
        # row visibility as proof that this is a recipient-local task bypasses
        # current peer-scope reauthorization and recipient-side notification.
        # ``None`` is the PeersFeature contract for an exact absent outbound
        # route; every unsafe/unreadable route is a fail-closed ToolResult.
        try:
            local_recipient_match = await self.task_manager.is_task_recipient(
                task_id,
                actor_agent_id,
            )
        except Exception as error:
            if is_execution_control_error(error):
                raise
            logger.error(
                "Could not resolve task direction for %s: %s",
                task_id,
                error,
                exc_info=True,
            )
            return ToolResult.failed(
                "Could not resolve task cancellation direction",
                data={"task_id": task_id},
            )
        features = getattr(self.agent, "features", None)
        values = (
            features.values()
            if hasattr(features, "values")
            else features or ()
        )
        outbound_router_available = False
        for feature in values:
            cancel_outbound = getattr(feature, "cancel_outbound_task", None)
            if callable(cancel_outbound):
                outbound_router_available = True
                outbound_result = await cancel_outbound(
                    task_id,
                    reason=reason,
                    local_recipient_match=local_recipient_match,
                )
                if outbound_result is not None:
                    return outbound_result

        if not local_recipient_match:
            return ToolResult.failed(
                f"Task {task_id} not found"
                if not outbound_router_available
                else "Task cancellation direction is not unambiguous",
                data={"task_id": task_id},
            )

        try:
            task = await self.task_manager.cancel_task(
                task_id,
                reason=reason,
                agent_name=actor_agent_id,
                recipient_agent_id=actor_agent_id,
            )
        except ValueError as e:
            if is_execution_control_error(e):
                raise
            logger.error(f"Failed to cancel task {task_id}: {e}", exc_info=True)
            return ToolResult.failed(
                str(e),
                data={"task_id": task_id},
            )
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.error(f"Failed to cancel task {task_id}: {e}", exc_info=True)
            return ToolResult.failed(
                str(e),
                data={"task_id": task_id},
            )

        receipt = MessageToDict(task.metadata).get("cancellation_receipt") or {}
        current_state = receipt.get("status_before")
        durable_reason = receipt.get("reason")

        return ToolResult.ok(
            confirmation=f"Cancelled task {task_id[:8]} (was: {current_state})",
            data={
                "task_id": task_id,
                "status": "canceled",
                "status_before": current_state,
                "reason": durable_reason,
            },
        )
