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

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.waits import run_wait_loop
from kestrel_sovereign.waits.engine import MAX_HANDLE_WAIT_SECONDS
from kestrel_sovereign.features.tasks.wait_provider import TaskWaitable
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

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

    # Conservative ceiling on a single blocking ``wait``. A pause longer
    # than this should be a scheduled/cron resume, not a held agent turn.
    _MAX_WAIT_SECONDS = 1800

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

        Lets ``wait("task:<task_id>")`` dispatch here, and lets the
        Wave-2 reconciler enumerate this kind. ``wait_for_task`` calls
        the engine directly and does not depend on this registration.
        """
        registry = getattr(agent, "wait_registry", None)
        if registry is not None:
            registry.register(TaskWaitable(self), replace=True)

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

        # ``status.message`` is the AGENT'S response message (set when the
        # receiver calls ``respond_to_a2a_task``). For a fresh inbound task
        # it's None until the agent replies — DO NOT mistake it for the
        # sender's question. The incoming request body lives in
        # ``task.history[0]`` (where ``create_task`` puts ``params.message``).
        # Surface both so the LLM inspecting an inbox task sees what was
        # ASKED (request_content) AND what was answered if anything
        # (status_message). Conflating the two caused #1433 — the receiver
        # read None and concluded the body was null when in fact it just
        # hadn't been replied to yet.
        status_message = None
        if task.status.message and task.status.message.parts:
            status_message = task.status.message.parts[0].text

        request_content: Optional[str] = None
        sender: Optional[str] = None
        history = getattr(task, "history", None) or []
        if history:
            first = history[0]
            parts = getattr(first, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    request_content = text
                    break
        if task.metadata and isinstance(task.metadata, dict):
            raw_sender = task.metadata.get("sender")
            if isinstance(raw_sender, str) and raw_sender:
                sender = raw_sender

        artifacts: List[Dict[str, Any]] = []
        if task.artifacts:
            for artifact in task.artifacts:
                artifact_data: Dict[str, Any] = {
                    "name": artifact.name,
                    "description": artifact.description,
                }
                if artifact.parts:
                    text_segments: List[str] = []
                    for part in artifact.parts:
                        if hasattr(part, "data"):
                            # Same hazard as run_workflow: a migrated
                            # tool's artifact data can carry a raw
                            # ToolResult inside the DynamicTool wrapper
                            # (only serialized on wire output, not on
                            # in-process reads). Serialize here so
                            # check_task_status / get_task_result /
                            # wait_for_task all return JSON-clean
                            # payloads. Round 6 codex finding.
                            artifact_data["data"] = self._serialize_step_payload(
                                part.data
                            )
                        elif getattr(part, "text", None) is not None:
                            # Surface text-part bodies too so a recipient
                            # can read sender-attached text artifacts
                            # (e.g. a handoff plan) — not just structured
                            # data parts (#1525). Concatenated in part
                            # order to reassemble a chunked body.
                            text_segments.append(part.text or "")
                    if text_segments:
                        artifact_data["text"] = "".join(text_segments)
                artifacts.append(artifact_data)

        return {
            "ok": True,
            "task_id": task_id,
            "status": task.status.state.value,
            "message": status_message,
            "request_content": request_content,
            "sender": sender,
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

            try:
                normalized_args = self._normalize_step_args(
                    raw_args,
                    step_index=i,
                    feature_name=feature_name,
                    skill_name=skill_name,
                )
            except ValueError as e:
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
                    # Serialize the result before storing it in the
                    # workflow's data dict. ``DynamicTool.execute`` may
                    # store a raw ``ToolResult`` under ``result``; if
                    # we leave that in place, downstream JSON
                    # serialization for the LLM tool-history wire will
                    # blow up. Walk the dict and ``.to_dict()`` any
                    # ToolResult instances we find. (Round 5 codex
                    # finding.)
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
    ) -> tuple[str, Optional[str]]:
        """Translate the A2A task state + tool wire-data into a
        workflow-step status.

        A2A's ``task.status.state`` is transport-level: COMPLETED means
        "the python call returned." A migrated tool that returns
        ``ToolResult.failed`` is *transport-completed* but *semantically
        failed*. To keep the workflow rollup honest, this helper
        inspects the wire-shape.

        Wire shapes seen in the wild (in order of how widely they
        appear in production paths):

          1. **In-process DynamicTool wrapper, raw ToolResult**
             ``{"success": True, "result": <ToolResult instance>,
             "tool": "<name>"}``.
             ``Feature.handle_task`` → ``DynamicTool.execute``
             stores the raw return value in ``result``. The
             ToolResult object is only converted to a dict when
             the artifact is serialized for the wire (Pydantic
             ``model_dump()``); a synchronous in-process workflow
             reads the raw object before that happens. **This is
             the path codex round 4 caught.**
          2. **Wire-serialized DynamicTool wrapper, dict ToolResult**
             ``{"success": True, "result": <ToolResult.to_dict()>,
             "tool": "..."}``. Same path after serialization.
          3. **Bare ToolResult envelope** (handlers that bypass
             DynamicTool): ``{"status": "ok"|"error"|"partial", ...}``.
          4. **Bare ToolResult instance** (rare; some custom handlers).
          5. **Pre-migration dict** ``{"success": False, "error": ...}``.
          6. Anything else → defer to the transport state.

        Returns ``(status, error_or_None)``. ``status`` is one of
        ``"completed" | "failed" | "partial"``.
        """
        if transport_state == "failed":
            return "failed", None

        # Normalize a raw ToolResult instance to its dict form so the
        # rest of the helper is purely structural.
        def _normalize(candidate: Any) -> Any:
            if isinstance(candidate, ToolResult):
                return candidate.to_dict()
            return candidate

        result_data = _normalize(result_data)

        if not isinstance(result_data, dict):
            return ("completed" if transport_state == "completed"
                    else transport_state), None

        # Resolve to whichever shape carries the @tool's return:
        #  - bare envelope: result_data IS the ToolResult.to_dict()
        #  - wrapped: result_data['result'] is the ToolResult — either
        #    a dict (post-serialize) or a raw ToolResult instance
        #    (pre-serialize, in-process)
        candidates: List[Any] = [result_data]
        if "tool" in result_data and "success" in result_data:
            inner = _normalize(result_data.get("result"))
            if isinstance(inner, dict):
                candidates.append(inner)

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            envelope_status = candidate.get("status")
            # Coerce the enum to its wire token for comparison so
            # both raw ToolResultStatus values (post-normalize from a
            # raw ToolResult) and bare strings (post-serialize) match.
            if isinstance(envelope_status, ToolResultStatus):
                envelope_status = envelope_status.value
            if envelope_status == "ok":
                # The @tool said OK explicitly. Even if the outer
                # wrapper had its own success flag, this is the
                # source of truth.
                return "completed", None
            if envelope_status == "error":
                return "failed", candidate.get("error")
            if envelope_status == "partial":
                return "partial", candidate.get("error")

        # Pre-migration dict shape on the OUTER result_data only —
        # the wrapper's own ``success: True`` is not authoritative
        # (it just means DynamicTool.execute didn't raise), so don't
        # treat outer success=True as a clean completion if we
        # didn't find an envelope above.
        if result_data.get("success") is False:
            return "failed", result_data.get("error")
        if result_data.get("error") and result_data.get("success") is not True:
            return "failed", result_data.get("error")

        return ("completed" if transport_state == "completed"
                else transport_state), None

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
            data={
                "task_id": data["task_id"],
                "status": data["status"],
                "message": data["message"],
                "request_content": data["request_content"],
                "sender": data["sender"],
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

                # Same conflation guard as `_get_task_status_data`: surface
                # the incoming sender content from `history[0]` so an inbox
                # listing actually shows what was ASKED, not just the
                # currently-empty reply slot. See #1433.
                request_text = None
                history = getattr(task, "history", None) or []
                if history:
                    parts = getattr(history[0], "parts", None) or []
                    for part in parts:
                        text = getattr(part, "text", None)
                        if isinstance(text, str) and text:
                            request_text = text
                            break
                sender_name = None
                if task.metadata and isinstance(task.metadata, dict):
                    raw_sender = task.metadata.get("sender")
                    if isinstance(raw_sender, str) and raw_sender:
                        sender_name = raw_sender

                task_list.append({
                    "task_id": task.id,
                    "status": task.status.state.value,
                    "task_type": task.metadata.get("task_type") if task.metadata else None,
                    "message": status_msg,
                    "request_content": request_text,
                    "sender": sender_name,
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
                "request_content": data["request_content"],
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
            "(delegated work, full result). The sender's subscription "
            "supervisor on the SSE stream picks up your terminal frame "
            "and fires their resumption signal. Without this tool the "
            "sender's send_a2a_question lineage never resumes until the "
            "hourly expiry sweep fires a state='expired' signal."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a respond",
    )
    async def respond_to_a2a_task(
        self,
        task_id: str,
        content: str,
        state: str = "completed",
    ) -> ToolResult:
        """
        Receiver-side completion of an A2A task.

        Transitions the named task to a terminal state (COMPLETED by
        default; FAILED or CANCELED via the ``state`` argument) and
        attaches ``content`` as the response text in
        ``status.message.parts[].text``. The sender's subscription
        supervisor on ``GET /tasks/{id}/subscribe`` picks up the
        terminal SSE frame and fires their ``a2a.question_answered``
        resumption signal (#1444).

        A2A state machine constraint: SUBMITTED cannot go directly to
        COMPLETED — it must pass through WORKING first. This tool
        chains the two transitions automatically when the current
        state is SUBMITTED, so the caller can ignore the intermediate
        bookkeeping.

        Args:
            task_id: The id of the incoming task to respond to.
                Find it via ``list_my_tasks(status='submitted')``
                or directly from the ``a2a.task_submitted`` signal
                payload that woke this cognition turn.
            content: The response text (one part, plain text).
                Becomes ``status.message.parts[0].text`` on the task.
            state: Terminal state. ``"completed"`` (default) for
                normal success; ``"failed"`` if you couldn't fulfill
                the request and want the sender to see the error;
                ``"canceled"`` to decline the task.
        """
        from kestrel_sovereign.a2a.types import (
            Message,
            TaskState,
            TextPart,
        )

        if not self.task_manager:
            return ToolResult.failed("Task manager not available")

        normalized_state = (state or "completed").strip().lower()
        try:
            terminal = TaskState(normalized_state)
        except ValueError:
            return ToolResult.failed(
                f"Invalid state {state!r}. Use 'completed', 'failed', or 'canceled'.",
                data={"task_id": task_id, "state": state},
            )
        if terminal not in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED):
            return ToolResult.failed(
                f"State {terminal.value!r} is not terminal — must be "
                f"completed, failed, or canceled.",
                data={"task_id": task_id, "state": terminal.value},
            )

        try:
            task = await self.task_manager.get_task(task_id)
        except Exception as e:
            logger.error(f"Failed to fetch task {task_id} for respond: {e}")
            return ToolResult.failed(str(e))
        if not task:
            return ToolResult.failed(
                f"Task {task_id} not found in this agent's task store. "
                f"(Senders see their own outbound; you can only respond "
                f"to incoming tasks that landed in YOUR store.)",
                data={"task_id": task_id},
            )

        current = task.status.state
        if current in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED):
            return ToolResult.failed(
                f"Task {task_id} is already terminal: {current.value}",
                data={"task_id": task_id, "state": current.value},
            )

        agent_name = getattr(self.agent, "did", None) or type(self.agent).__name__
        response_message = Message(
            role="agent",
            parts=[TextPart(text=content)],
        )

        # SUBMITTED → COMPLETED is not a valid direct transition
        # (VALID_TRANSITIONS in task_manager); SUBMITTED must pass
        # through WORKING first. Chain automatically so the receiver
        # doesn't need to know about the intermediate step.
        try:
            if current == TaskState.SUBMITTED:
                await self.task_manager.update_status(
                    task_id=task_id,
                    new_state=TaskState.WORKING,
                    agent_name=agent_name,
                )
            updated = await self.task_manager.update_status(
                task_id=task_id,
                new_state=terminal,
                message=response_message,
                agent_name=agent_name,
            )
        except ValueError as e:
            # Transition validator caught an illegal sequence (rare —
            # only happens if another path mutates the task between
            # our get_task and update_status).
            return ToolResult.failed(
                str(e),
                data={"task_id": task_id, "state_before": current.value},
            )
        except Exception as e:
            logger.error(
                f"Failed to respond to task {task_id}: {e}", exc_info=True,
            )
            return ToolResult.failed(
                str(e),
                data={"task_id": task_id, "state_before": current.value},
            )

        return ToolResult.ok(
            confirmation=(
                f"Responded to task {task_id[:8]} "
                f"({current.value} → {updated.status.state.value}); "
                f"sender's poll will now see your answer."
            ),
            data={
                "task_id": task_id,
                "state": updated.status.state.value,
                "state_before": current.value,
                "response": content,
            },
        )

    @tool(
        name="attach_artifact_to_a2a_task",
        description=(
            "RESPONDER-SIDE artifact attach: the RECIPIENT of an "
            "incoming A2A task uses this to attach its own output. To "
            "attach payload as the SENDER of an outgoing task, pass "
            "``artifacts``/``references`` to send_a2a_task / "
            "send_a2a_question instead. "
            "Attach one chunk of long-form output as an Artifact to "
            "an incoming A2A task BEFORE calling respond_to_a2a_task. "
            "Use this when your reply exceeds the per-tool argument "
            "cap (10K chars) — chunk the body into segments of <=9000 "
            "chars each, call this tool once per segment with "
            "monotonically-increasing index (0, 1, 2, ...) and "
            "last_chunk=False on every segment except the final one. "
            "The sender's get_peer_task_result returns the artifacts "
            "in order so the resumed turn can reassemble the full "
            "body. After all segments are attached, call "
            "respond_to_a2a_task with a SHORT content like "
            "'See attached artifacts (N segments).' so the sender "
            "knows where to look."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a attach",
    )
    async def attach_artifact_to_a2a_task(
        self,
        task_id: str,
        name: str,
        content: str,
        index: int = 0,
        last_chunk: bool = True,
    ) -> ToolResult:
        """Attach a single artifact segment to an incoming A2A task.

        For replies up to ~9K chars one call with ``last_chunk=True``
        is enough. For longer replies, chunk into segments and call
        repeatedly with monotonically-increasing ``index`` and
        ``last_chunk=False`` until the final segment.

        Args:
            task_id: The incoming task to attach to.
            name: Artifact name. Use the SAME name across all
                segments of one logical body so the sender's
                resumed turn can group them — e.g.
                ``"reply_body"``.
            content: The segment text. Bounded by the system-wide
                10K tool-arg cap; keep segments at ~9000 chars to
                leave headroom.
            index: Segment order, 0-based. Matters because the
                sender's reassembly walks artifacts in index order.
            last_chunk: ``True`` on the final segment (or on the
                only segment for short replies). The sender's
                resumed-turn prompt uses this to decide whether the
                body it received is complete.
        """
        from kestrel_sovereign.a2a.types import Artifact, TextPart

        if not self.task_manager:
            return ToolResult.failed(
                "No A2A task manager available — not running in a "
                "multi_agent environment with inbox support.",
                data={"task_id": task_id},
            )

        agent_name = (
            getattr(self.agent, "did", None) or type(self.agent).__name__
        )
        artifact = Artifact(
            name=name,
            parts=[TextPart(text=content)],
            index=index,
            lastChunk=last_chunk,
        )
        try:
            updated = await self.task_manager.add_artifact(
                task_id=task_id,
                artifact=artifact,
                agent_name=agent_name,
            )
        except ValueError as e:
            return ToolResult.failed(
                f"Failed to attach artifact to task {task_id}: {e}",
                data={"task_id": task_id, "name": name, "index": index},
            )
        except Exception as e:
            logger.error(
                "attach_artifact_to_a2a_task failed for task=%s: %s",
                task_id, e, exc_info=True,
            )
            return ToolResult.failed(
                str(e),
                data={"task_id": task_id, "name": name, "index": index},
            )

        return ToolResult.ok(
            confirmation=(
                f"Attached artifact '{name}' (index={index}, "
                f"last_chunk={last_chunk}, {len(content)} chars) to "
                f"task {task_id[:8]}."
            ),
            data={
                "task_id": task_id,
                "name": name,
                "index": index,
                "last_chunk": last_chunk,
                "content_chars": len(content),
                "artifact_count": len(updated.artifacts or []),
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

        # Thin wrapper over the generic wait engine: the TaskWaitable
        # provider classifies one status read, the engine owns the loop,
        # the cap, and the ToolResult mapping. ``wait_for_task`` keeps its
        # name so existing callers keep working.
        return await run_wait_loop(
            TaskWaitable(self),
            task_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval,
        )

    @tool(
        name="wait",
        description=(
            "The one wait. Two modes:\n"
            "• Pass `target` as `\"<kind>:<handle>\"` to block until that "
            "thing reaches a terminal state — e.g. `\"talon:<job_id>\"`, "
            "`\"task:<task_id>\"`. Polls the right feature's provider and "
            "returns the terminal outcome (or a still-pending result on "
            "timeout). This replaces the per-feature waiters (talon_wait, "
            "wait_for_task).\n"
            "• Pass `duration_seconds` with no target for a plain bounded "
            "pause — the native alternative to shelling out to `sleep` "
            "between polls in an autonomous loop.\n"
            "Long unattended waits should use the signal-resume path, not "
            "a held turn."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!wait",
    )
    async def wait(
        self,
        target: str = "",
        duration_seconds: int = 0,
        timeout_seconds: int = 600,
        poll_interval_seconds: int = 5,
        reason: str = "",
    ) -> ToolResult:
        """
        Block on a handle, or pause for a bounded duration.

        Args:
            target: ``"<kind>:<handle>"`` to wait on (e.g.
                ``"talon:job_42"``). When set, ``duration_seconds`` is
                ignored and the wait is driven by the registered provider.
            duration_seconds: Seconds to pause when no ``target`` is given
                (0 to the enforced maximum).
            timeout_seconds: Max seconds to block on a ``target`` before
                returning a still-pending result.
            poll_interval_seconds: Seconds between polls of a ``target``.
            reason: Optional human-readable note (recorded in the result).
        """
        # The first positional accepts BOTH forms so the interface stays
        # one tool: `!wait 5` (bare number) is a bounded sleep, while
        # `!wait talon:job_42` is a handle wait. parse_command_args binds
        # positional CLI tokens in signature order, so a numeric target is
        # the legacy `!wait <seconds>` command — route it to the pause.
        target = str(target).strip() if target else ""
        if target and target.lstrip("-").isdigit():
            duration_seconds = int(target)
            target = ""

        if target:
            registry = getattr(self.agent, "wait_registry", None) if self.agent else None
            if registry is None:
                return ToolResult.failed(
                    "wait engine unavailable: no wait_registry on the agent"
                )
            return await registry.wait(
                target,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )

        # No target: bounded idle pause (the legacy generic `wait`).
        try:
            duration = int(duration_seconds)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"duration_seconds must be an integer, got {duration_seconds!r}"
            )
        if duration < 0:
            return ToolResult.failed(
                f"duration_seconds must be >= 0, got {duration}"
            )
        if duration > self._MAX_WAIT_SECONDS:
            return ToolResult.failed(
                f"duration_seconds {duration} exceeds the maximum "
                f"{self._MAX_WAIT_SECONDS}s for a single wait; schedule a "
                f"resume instead of holding the turn",
                data={
                    "requested_seconds": duration,
                    "max_seconds": self._MAX_WAIT_SECONDS,
                },
            )

        start = time.monotonic()
        await asyncio.sleep(duration)
        elapsed = round(time.monotonic() - start, 3)

        confirmation = f"Waited {elapsed}s"
        if reason:
            confirmation += f" ({reason})"
        return ToolResult.ok(
            confirmation=confirmation,
            data={
                "requested_seconds": duration,
                "elapsed_seconds": elapsed,
                "reason": reason,
            },
        )
