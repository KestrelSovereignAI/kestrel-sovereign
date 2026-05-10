"""Agent-callable Workflows feature surface."""

from __future__ import annotations

import logging
from typing import Any, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.features.workflows.models import (
    RunStatus,
    WorkflowSpec,
)
from kestrel_sovereign.features.workflows.runner import (
    WorkflowRunner,
    WorkflowRunnerError,
)
from kestrel_sovereign.features.workflows.schema import validate_spec_payload
from kestrel_sovereign.features.workflows.signing import sign_workflow_spec
from kestrel_sovereign.features.workflows.store import WorkflowStore
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)

logger = logging.getLogger(__name__)


class WorkflowsFeature(Feature):
    """First-class signed workflow definitions and runs."""

    @property
    def tool_description(self) -> str:
        return (
            "Define, run, inspect, cancel, and audit signed multi-stage "
            "workflows that orchestrate registered signal sources."
        )

    async def initialize(self):
        self._db = resolve_feature_database(self.agent)
        self.store: Optional[WorkflowStore] = None
        self.runner: Optional[WorkflowRunner] = None

        if self._db is None:
            logger.warning("WorkflowsFeature: database not available")
            return

        backend = getattr(self._db, "_backend", self._db)
        self.store = WorkflowStore(backend)
        await self.store.initialize()
        self._build_runner()
        logger.info("WorkflowsFeature initialized")

    def _build_runner(self) -> None:
        if self.store is None:
            return
        identity = getattr(self.agent, "identity", None)
        dispatcher = getattr(self.agent, "dispatcher", None)
        registry = getattr(self.agent, "signal_registry", None)
        if identity is None or dispatcher is None or registry is None:
            logger.warning(
                "WorkflowsFeature: runner unavailable "
                "(identity=%s dispatcher=%s registry=%s)",
                identity is not None,
                dispatcher is not None,
                registry is not None,
            )
            return
        self.runner = WorkflowRunner(
            store=self.store,
            dispatcher=dispatcher,
            registry=registry,
            agent_identity=identity,
            public_key_resolver=self._public_key_resolver,
            verification_methods_resolver=self._verification_methods_resolver,
            consent_collect_provider=self._consent_collect_provider,
        )

    def _public_key_resolver(self, did: str) -> bytes:
        external = getattr(self.agent, "workflow_public_key_resolver", None)
        if external is not None:
            try:
                return external(did)
            except KeyError:
                pass

        identity = getattr(self.agent, "identity", None)
        if identity is not None and did == identity.legacy_did:
            suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
            return suite.serialize_public_key(identity.legacy_keypair.public_key)
        raise KeyError(did)

    def _verification_methods_resolver(self, did: str) -> list:
        external = getattr(self.agent, "workflow_verification_methods_resolver", None)
        if external is not None:
            try:
                return list(external(did))
            except KeyError:
                pass

        identity = getattr(self.agent, "identity", None)
        if (
            identity is not None
            and identity.is_hybrid
            and did == identity.signing_did
        ):
            return list(identity.new_verification_methods or [])
        raise KeyError(did)

    async def _consent_collect_provider(
        self,
        gate: Any,
        run: Any,
        stage: Any,
        link: Any,
    ) -> dict[str, Any]:
        external = getattr(self.agent, "workflow_consent_collect_provider", None)
        if external is not None:
            result = external(gate, run, stage, link)
            if hasattr(result, "__await__"):
                result = await result
            return result

        request_id = _pending_consent_request_id(link.gate_reason)
        if request_id is None:
            return {
                "scope": gate.params["scope"],
                "approved": False,
                "reason": "missing approval request id",
            }

        security = _security_feature(self.agent)
        approval_queue = getattr(security, "approval_queue", None)
        if approval_queue is None:
            return {
                "scope": gate.params["scope"],
                "approved": False,
                "reason": "approval queue unavailable",
            }

        request = approval_queue.get_request(request_id)
        if request is None:
            return {
                "scope": gate.params["scope"],
                "approved": False,
                "reason": f"approval request not found:{request_id}",
            }

        status = getattr(getattr(request, "status", None), "value", None)
        if not _approval_request_matches_scope(request, gate.params["scope"]):
            return {
                "scope": gate.params["scope"],
                "approved": False,
                "reason": "approval request scope mismatch",
            }
        if status == "pending":
            return {
                "scope": gate.params["scope"],
                "status": "pending",
                "approval_id": request_id,
            }
        if status == "approved":
            return {
                "scope": gate.params["scope"],
                "approved": True,
                "approval_id": request_id,
                "approval_scope": getattr(request, "user_decision", None),
            }
        return {
            "scope": gate.params["scope"],
            "approved": False,
            "reason": status or "approval request denied",
        }

    def _require_store(self) -> WorkflowStore:
        if self.store is None:
            raise WorkflowRunnerError("Workflows store is not initialized")
        return self.store

    def _require_runner(self) -> WorkflowRunner:
        if self.runner is None:
            self._build_runner()
        if self.runner is None:
            raise WorkflowRunnerError(
                "Workflow runner is not available; agent identity, "
                "dispatcher, and signal registry are required"
            )
        return self.runner

    @tool(
        name="workflow_define",
        description="Register and sign a workflow definition.",
        category=ToolCategory.SYSTEM,
        command_prefix="!workflow-define",
    )
    async def workflow_define(self, spec: dict) -> ToolResult:
        """
        Register and sign a workflow definition.

        Args:
            spec: WorkflowSpec JSON payload without author_sig/spec_hash.
        """
        try:
            store = self._require_store()
            identity = getattr(self.agent, "identity", None)
            if identity is None:
                return ToolResult.failed("Agent identity is not available")
            validate_spec_payload(spec)
            workflow = WorkflowSpec.from_dict(spec)
            signed = sign_workflow_spec(workflow, identity, use_hybrid=True)
            await store.put_definition(signed)
            return ToolResult.ok(
                f"Workflow '{signed.name}' v{signed.version} registered.",
                data={
                    "name": signed.name,
                    "version": signed.version,
                    "spec_hash": signed.spec_hash,
                    "author_did": signed.author_did,
                },
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return ToolResult.failed(str(exc))

    @tool(
        name="workflow_run",
        description="Start a workflow run and execute the current foundation runner.",
        category=ToolCategory.SYSTEM,
        command_prefix="!workflow-run",
    )
    async def workflow_run(
        self,
        name: str,
        params: dict = None,
        version: int = 0,
    ) -> ToolResult:
        """
        Run a workflow.

        Args:
            name: Workflow definition name.
            params: Run parameters.
            version: Specific definition version, or latest active.
        """
        try:
            result = await self._require_runner().run_to_completion(
                name=name,
                params={} if params is None else params,
                version=version or None,
            )
            return ToolResult.ok(
                f"Workflow run {result.run_id} finished with {result.status.value}.",
                data={"run_id": result.run_id, "status": result.status.value},
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return ToolResult.failed(str(exc))

    @tool(
        name="workflow_status",
        description="Show the current status for a workflow run.",
        category=ToolCategory.SYSTEM,
        command_prefix="!workflow-status",
    )
    async def workflow_status(self, run_id: str) -> ToolResult:
        """
        Show workflow run status.

        Args:
            run_id: Workflow run id.
        """
        try:
            run = await self._require_store().get_run(run_id)
            if run is None:
                return ToolResult.failed(f"Unknown workflow run: {run_id}")
            return ToolResult.ok(
                f"Workflow run {run_id} is {run.status.value}.",
                data=run.to_dict(),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failed(str(exc))

    @tool(
        name="workflow_cancel",
        description=(
            "Set the workflow cancellation barrier and compensate completed stages."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!workflow-cancel",
    )
    async def workflow_cancel(self, run_id: str) -> ToolResult:
        """
        Cancel a workflow run.

        Args:
            run_id: Workflow run id.
        """
        try:
            status = await self._require_runner().cancel_run(run_id)
            return ToolResult.ok(
                f"Workflow run {run_id} cancelled with status {status.value}.",
                data={"run_id": run_id, "status": status.value},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failed(str(exc))

    @tool(
        name="workflow_pause",
        description="Pause a workflow at its current stage boundary.",
        category=ToolCategory.SYSTEM,
        command_prefix="!workflow-pause",
    )
    async def workflow_pause(self, run_id: str) -> ToolResult:
        """
        Pause a workflow run.

        Args:
            run_id: Workflow run id.
        """
        try:
            store = self._require_store()
            run = await store.get_run(run_id)
            if run is None:
                return ToolResult.failed(f"Unknown workflow run: {run_id}")
            if run.status not in (RunStatus.RUNNING, RunStatus.WAITING):
                return ToolResult.failed(
                    f"Workflow run {run_id} cannot be paused from "
                    f"status {run.status.value}."
                )
            await store.update_run_status(run_id, RunStatus.PAUSED)
            return ToolResult.ok(
                f"Workflow run {run_id} paused.",
                data={"run_id": run_id, "status": RunStatus.PAUSED.value},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failed(str(exc))

    @tool(
        name="workflow_resume",
        description="Resume a paused workflow run.",
        category=ToolCategory.SYSTEM,
        command_prefix="!workflow-resume",
    )
    async def workflow_resume(self, run_id: str) -> ToolResult:
        """
        Resume a workflow run.

        Args:
            run_id: Workflow run id.
        """
        try:
            result = await self._require_runner().continue_run(run_id)
            return ToolResult.ok(
                f"Workflow run {run_id} resumed and finished with "
                f"{result.status.value}.",
                data={"run_id": run_id, "status": result.status.value},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failed(str(exc))

    @tool(
        name="workflow_remediate",
        description="Record or apply remediation for a failed workflow stage.",
        category=ToolCategory.SYSTEM,
        command_prefix="!workflow-remediate",
    )
    async def workflow_remediate(
        self,
        run_id: str,
        stage: str,
        action: str,
    ) -> ToolResult:
        """
        Remediate a workflow stage.

        Args:
            run_id: Workflow run id.
            stage: Stage name.
            action: Remediation action.
        """
        if action != "retry":
            return ToolResult.failed(
                "workflow_remediate currently supports action='retry' only; "
                f"received action={action!r}."
            )
        try:
            result = await self._require_runner().retry_stage(run_id, stage)
            return ToolResult.ok(
                f"Workflow run {run_id} retried stage {stage} and finished "
                f"with {result.status.value}.",
                data={
                    "run_id": run_id,
                    "stage": stage,
                    "action": action,
                    "status": result.status.value,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failed(str(exc))

    @tool(
        name="workflow_history",
        description="Return the audit trail for a workflow run.",
        category=ToolCategory.SYSTEM,
        command_prefix="!workflow-history",
    )
    async def workflow_history(self, run_id: str) -> ToolResult:
        """
        Show workflow stage history.

        Args:
            run_id: Workflow run id.
        """
        try:
            links = await self._require_store().list_stage_links(run_id)
            return ToolResult.ok(
                f"Workflow run {run_id} has {len(links)} stage event(s).",
                data={"run_id": run_id, "links": [l.to_dict() for l in links]},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failed(str(exc))

    @tool(
        name="workflow_list_definitions",
        description="List registered workflow definitions.",
        category=ToolCategory.SYSTEM,
        command_prefix="!workflow-list-definitions",
    )
    async def workflow_list_definitions(self) -> ToolResult:
        """List workflow definitions."""
        try:
            rows = await self._require_store().list_definitions()
            data = {"definitions": [_json_ready(row) for row in rows]}
            return ToolResult.ok(
                f"{len(rows)} workflow definition(s) registered.",
                data=data,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failed(str(exc))

    @tool(
        name="workflow_list_runs",
        description="List workflow runs.",
        category=ToolCategory.SYSTEM,
        command_prefix="!workflow-list-runs",
    )
    async def workflow_list_runs(
        self,
        workflow_name: str = "",
        status: str = "",
        limit: int = 50,
    ) -> ToolResult:
        """
        List workflow runs.

        Args:
            workflow_name: Optional workflow name filter.
            status: Optional run status filter.
            limit: Maximum rows to return.
        """
        try:
            runs = await self._require_store().list_runs(
                workflow_name=workflow_name or None,
                status=status or None,
                limit=limit,
            )
            return ToolResult.ok(
                f"{len(runs)} workflow run(s) found.",
                data={"runs": [run.to_dict() for run in runs]},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failed(str(exc))


def _json_ready(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def _pending_consent_request_id(reason: Any) -> Optional[str]:
    if not isinstance(reason, str):
        return None
    prefix = "consent_collect_pending:"
    if not reason.startswith(prefix):
        return None
    request_id = reason[len(prefix) :].strip()
    return request_id or None


def _security_feature(agent: Any) -> Any:
    direct = getattr(agent, "security_feature", None)
    if direct is not None:
        return direct
    features = getattr(agent, "features", None)
    if isinstance(features, dict):
        return (
            features.get("SecurityFeature")
            or features.get("Security")
            or features.get("security")
        )
    return None


def _approval_request_matches_scope(request: Any, scope: str) -> bool:
    if getattr(request, "feature_name", None) != "WorkflowsFeature":
        return False
    tool_args = getattr(request, "tool_args", None)
    if not isinstance(tool_args, dict):
        return False
    return tool_args.get("scope") == scope


__all__ = ["WorkflowsFeature"]
