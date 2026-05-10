"""Workflow runner foundation.

Phase 1 composes workflow stages onto the existing SignalDispatcher. This
module intentionally starts with the smallest executable surface:

- refuse-before-side-effect run-start validation against SourceRegistry;
- DID signature re-verification for the pinned definition;
- deterministic idempotency-key derivation with stored engine nonce;
- sequential graph walking with the default ``signal_status_ok`` gate;
- cancellation barrier detection and reverse-order compensation.

Later Phase 1/2 chunks can extend ``_evaluate_gate`` and ``_next_stages``
for richer gates and graph shapes without changing the storage/audit
contract established here.
"""

from __future__ import annotations

import asyncio
import ast
import base64
import hashlib
import inspect
import json
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from kestrel_sdk.signals import (
    CausationFrame,
    Signal,
    SignalMode,
    Status,
    Urgency,
    Visibility,
)

from kestrel_sovereign.features.workflows.models import (
    EdgeKind,
    Gate,
    GateOutcome,
    RunStatus,
    Stage,
    StageLink,
    WorkflowDefinitionError,
    WorkflowRun,
    WorkflowSpec,
)
from kestrel_sovereign.features.workflows.metrics import (
    record_compensation_state,
    record_gate_outcome,
)
from kestrel_sovereign.features.workflows.signing import (
    PublicKeyResolver,
    VerificationMethodsResolver,
    sign_stage_transition,
    verify_workflow_spec,
)
from kestrel_sovereign.features.workflows.store import WorkflowStore
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    CryptoSuiteError,
    get_suite,
)
from kestrel_sovereign.signals import SignalDispatcher, SourceRegistry


_RUNNER_GATE_TYPES = frozenset(
    {
        "signal_status_ok",
        "tests_pass",
        "ci_green",
        "lint_clean",
        "signature_collected",
        "constitutional_boundary_clean",
    }
)

CiGreenProvider = Callable[[Gate, Any], Awaitable[Any] | Any]


class WorkflowRunnerError(RuntimeError):
    """Raised when the runner refuses a workflow before firing signals."""


@dataclass(frozen=True)
class WorkflowRunResult:
    run_id: str
    status: RunStatus


class WorkflowRunner:
    """Execute signed workflow definitions through SignalDispatcher."""

    def __init__(
        self,
        *,
        store: WorkflowStore,
        dispatcher: SignalDispatcher,
        registry: SourceRegistry,
        agent_identity: AgentIdentity,
        public_key_resolver: PublicKeyResolver,
        verification_methods_resolver: Optional[
            VerificationMethodsResolver
        ] = None,
        ci_green_provider: Optional[CiGreenProvider] = None,
    ) -> None:
        self.store = store
        self.dispatcher = dispatcher
        self.registry = registry
        self.agent_identity = agent_identity
        self.public_key_resolver = public_key_resolver
        self.verification_methods_resolver = verification_methods_resolver
        self.ci_green_provider = ci_green_provider or _default_ci_green_provider

    async def start_run(
        self,
        *,
        name: str,
        params: Optional[dict] = None,
        version: Optional[int] = None,
        parent_run_id: Optional[str] = None,
        scheduler_task_id: Optional[str] = None,
    ) -> WorkflowRun:
        spec = await self._load_startable_definition(name, version)
        self._validate_run_start_contract(spec)
        run_params = {} if params is None else params
        self._validate_run_params(spec, run_params)

        run = WorkflowRun(
            run_id=str(uuid4()),
            workflow_name=spec.name,
            workflow_ver=spec.version,
            params=run_params,
            status=RunStatus.RUNNING,
            engine_nonce=secrets.token_hex(16),
            current_stages=[self._start_stage(spec).name],
            parent_run_id=parent_run_id,
            started_by_did=self.agent_identity.legacy_did,
            scheduler_task_id=scheduler_task_id,
            started_at=datetime.now(timezone.utc),
        )
        await self.store.insert_run(run)
        return run

    async def run_to_completion(
        self,
        *,
        name: str,
        params: Optional[dict] = None,
        version: Optional[int] = None,
    ) -> WorkflowRunResult:
        run = await self.start_run(name=name, params=params, version=version)
        return await self.continue_run(run.run_id)

    async def continue_run(self, run_id: str) -> WorkflowRunResult:
        run = await self.store.get_run(run_id)
        if run is None:
            raise WorkflowRunnerError(f"unknown workflow run: {run_id}")
        if run.status in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.CANCELLED_WITH_IRREVERSIBLE_RESIDUE,
        ):
            raise WorkflowRunnerError(
                f"workflow run {run_id} is terminal ({run.status.value})"
            )
        spec = await self._load_pinned_definition(run)
        self._validate_run_start_contract(spec)
        if run.status == RunStatus.PAUSED:
            await self.store.update_run_status(run_id, RunStatus.RUNNING)
            run = await self.store.get_run(run_id)
            if run is None:
                raise WorkflowRunnerError(f"workflow run missing: {run_id}")
        return await self._continue_run(run, spec)

    async def _continue_run(
        self, run: WorkflowRun, spec: WorkflowSpec
    ) -> WorkflowRunResult:
        current = list(run.current_stages)
        while current:
            run_snapshot = await self.store.get_run(run.run_id)
            if run_snapshot is None:
                raise WorkflowRunnerError(f"workflow run missing: {run.run_id}")
            if run_snapshot.cancel_barrier_at is not None:
                status = await self._compensate(run_snapshot, spec)
                return WorkflowRunResult(run.run_id, status)

            stage = self._stage_by_name(spec, current.pop(0))
            gate = await self._dispatch_stage(run_snapshot, spec, stage)
            post_dispatch_run = await self.store.get_run(run.run_id)
            if post_dispatch_run is None:
                raise WorkflowRunnerError(f"workflow run missing: {run.run_id}")
            if post_dispatch_run.cancel_barrier_at is not None:
                status = await self._compensate(post_dispatch_run, spec)
                return WorkflowRunResult(run.run_id, status)
            if gate != GateOutcome.PASS:
                status = await self._compensate(
                    run_snapshot,
                    spec,
                    success_status=RunStatus.FAILED,
                    residue_status=RunStatus.FAILED,
                )
                return WorkflowRunResult(run.run_id, status)

            next_current = [*current, *self._next_stages(spec, stage.name)]
            if post_dispatch_run.status == RunStatus.PAUSED:
                await self.store.update_run_status(
                    run.run_id,
                    RunStatus.PAUSED,
                    current_stages=next_current,
                )
                return WorkflowRunResult(run.run_id, RunStatus.PAUSED)

            current = next_current
            await self.store.update_run_status(
                run.run_id,
                RunStatus.RUNNING,
                current_stages=current,
            )

        await self.store.update_run_status(
            run.run_id,
            RunStatus.COMPLETED,
            current_stages=[],
            finished_at=datetime.now(timezone.utc),
        )
        return WorkflowRunResult(run.run_id, RunStatus.COMPLETED)

    async def cancel_run(self, run_id: str) -> RunStatus:
        run = await self.store.get_run(run_id)
        if run is None:
            raise WorkflowRunnerError(f"unknown workflow run: {run_id}")
        if run.status in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.CANCELLED_WITH_IRREVERSIBLE_RESIDUE,
        ):
            raise WorkflowRunnerError(
                f"workflow run {run_id} is terminal ({run.status.value})"
            )
        immediate_compensation = run.status in (
            RunStatus.PENDING,
            RunStatus.PAUSED,
            RunStatus.WAITING,
        )
        await self.store.set_cancel_barrier(run_id)
        spec = await self._load_pinned_definition(run)
        run = await self.store.get_run(run_id)
        if run is None:
            raise WorkflowRunnerError(f"workflow run missing after cancel: {run_id}")
        if not immediate_compensation:
            return RunStatus.COMPENSATING
        return await self._compensate(run, spec)

    async def retry_stage(self, run_id: str, stage_name: str) -> WorkflowRunResult:
        run = await self.store.get_run(run_id)
        if run is None:
            raise WorkflowRunnerError(f"unknown workflow run: {run_id}")
        if run.status != RunStatus.FAILED:
            raise WorkflowRunnerError(
                f"workflow run {run_id} must be failed before retry; "
                f"got {run.status.value}"
            )
        spec = await self._load_pinned_definition(run)
        self._validate_run_start_contract(spec)
        failed_stage_name = await self._latest_failed_stage_name(run_id)
        if stage_name != failed_stage_name:
            raise WorkflowRunnerError(
                f"workflow run {run_id} failed at stage {failed_stage_name!r}; "
                f"cannot retry {stage_name!r}"
            )
        await self._ensure_retry_has_no_compensation_residue(run_id)
        await self.store.update_run_status(
            run_id,
            RunStatus.RUNNING,
            current_stages=[self._start_stage(spec).name],
            clear_finished_at=True,
        )
        run = await self.store.get_run(run_id)
        if run is None:
            raise WorkflowRunnerError(f"workflow run missing after retry: {run_id}")
        return await self._continue_run(run, spec)

    async def _load_startable_definition(
        self, name: str, version: Optional[int]
    ) -> WorkflowSpec:
        if version is None:
            spec = await self.store.get_latest_definition(name)
            row = None
        else:
            row = await self.store.get_definition_row(name, version)
            spec = (
                WorkflowSpec.from_dict(json.loads(row["spec_json"]))
                if row is not None
                else None
            )
        if spec is None:
            raise WorkflowRunnerError(f"workflow definition not found: {name}")
        if row is not None and row["deleted_at"] is not None:
            raise WorkflowRunnerError(
                f"workflow definition is revoked: {name} v{version}"
            )
        self._verify_workflow_spec(spec)
        return spec

    def _verify_workflow_spec(self, spec: WorkflowSpec) -> None:
        if verify_workflow_spec(
            spec,
            self.public_key_resolver,
            verification_methods_resolver=self.verification_methods_resolver,
        ):
            return
        raise WorkflowRunnerError(
            f"workflow definition signature failed verification: "
            f"{spec.name} v{spec.version}"
        )

    async def _load_pinned_definition(self, run: WorkflowRun) -> WorkflowSpec:
        row = await self.store.get_definition_row(
            run.workflow_name,
            run.workflow_ver,
        )
        if row is None:
            raise WorkflowRunnerError(
                f"workflow definition missing for run {run.run_id}: "
                f"{run.workflow_name} v{run.workflow_ver}"
            )
        spec = WorkflowSpec.from_dict(json.loads(row["spec_json"]))
        self._verify_workflow_spec(spec)
        if row["deleted_at"] is not None and not run.signature_post_revocation:
            await self.store.mark_run_signature_post_revocation(run.run_id)
        return spec

    def _validate_run_start_contract(self, spec: WorkflowSpec) -> None:
        self._validate_phase1_sequential_graph(spec)
        for stage in spec.stages:
            if stage.gate.type not in _RUNNER_GATE_TYPES:
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} gate {stage.gate.type!r} "
                    "is not implemented in the workflow runner"
                )
            if stage.gate.type in {
                "tests_pass",
                "ci_green",
                "lint_clean",
                "signature_collected",
            } and stage.signal_mode != SignalMode.ACTION:
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} gate {stage.gate.type!r} "
                    "requires signal_mode=ACTION"
                )
            registration = self.registry.get(stage.signal_source)
            if registration is None:
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} references unregistered source "
                    f"{stage.signal_source!r}"
                )
            if stage.signal_mode not in registration.allowed_modes:
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} mode {stage.signal_mode.value!r} "
                    f"is not allowed by source {stage.signal_source!r}"
                )
            if stage.compensate not in (
                "noop_idempotent",
                "compensate_record_only",
            ):
                comp_reg = self.registry.get(stage.compensate)
                if comp_reg is None:
                    raise WorkflowRunnerError(
                        f"stage {stage.name!r} compensation source "
                        f"{stage.compensate!r} is not registered"
                    )
                if SignalMode.ACTION not in comp_reg.allowed_modes:
                    raise WorkflowRunnerError(
                        f"stage {stage.name!r} compensation source "
                        f"{stage.compensate!r} must allow ACTION"
                    )

    def _validate_phase1_sequential_graph(self, spec: WorkflowSpec) -> None:
        start = self._start_stage(spec)
        adjacency: dict[str, list[str]] = {stage.name: [] for stage in spec.stages}
        incoming_counts: dict[str, int] = {stage.name: 0 for stage in spec.stages}
        for edge in spec.edges:
            if edge.kind != EdgeKind.SEQUENTIAL:
                continue
            adjacency[edge.from_stage].append(edge.to_stage)
            incoming_counts[edge.to_stage] += 1

        for stage_name, next_stages in adjacency.items():
            if len(next_stages) > 1:
                raise WorkflowRunnerError(
                    f"workflow graph has sequential fan-out at stage "
                    f"{stage_name!r}; Phase 1 foundation supports only "
                    "linear sequential graphs"
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(stage_name: str, path: list[str]) -> None:
            if stage_name in visiting:
                cycle = " -> ".join([*path, stage_name])
                raise WorkflowRunnerError(
                    f"workflow graph contains a cycle: {cycle}"
                )
            if stage_name in visited:
                return
            visiting.add(stage_name)
            for next_stage in adjacency.get(stage_name, []):
                visit(next_stage, [*path, stage_name])
            visiting.remove(stage_name)
            visited.add(stage_name)

        visit(start.name, [])
        unreachable = sorted(set(adjacency) - visited)
        if unreachable:
            raise WorkflowRunnerError(
                "workflow graph has unreachable stages: "
                f"{', '.join(unreachable)}"
            )
        for stage_name, incoming_count in incoming_counts.items():
            if incoming_count > 1:
                raise WorkflowRunnerError(
                    f"workflow graph has sequential fan-in at stage "
                    f"{stage_name!r}; Phase 1 foundation supports only "
                    "linear sequential graphs"
                )

    async def _latest_failed_stage_name(self, run_id: str) -> str:
        links = await self.store.list_stage_links(run_id)
        failed_links = [
            link for link in links if link.gate_outcome == GateOutcome.FAIL
        ]
        if not failed_links:
            raise WorkflowRunnerError(
                f"workflow run {run_id} has no failed stage to retry"
            )
        return failed_links[-1].stage_name

    async def _ensure_retry_has_no_compensation_residue(
        self, run_id: str
    ) -> None:
        links = await self.store.list_stage_links(run_id)
        residue = [
            (link.stage_name, link.compensate_state)
            for link in links
            if link.compensate_state in {"record_only", "failed"}
        ]
        if residue:
            stage_list = ", ".join(
                f"{stage!r} ({state})" for stage, state in residue
            )
            raise WorkflowRunnerError(
                f"workflow run {run_id} cannot be retried because "
                f"compensation residue remains: {stage_list}"
            )

    @staticmethod
    def _validate_run_params(spec: WorkflowSpec, params: object) -> None:
        if not isinstance(params, dict):
            raise WorkflowRunnerError("workflow run params must be an object")
        if not spec.params_schema:
            return
        schema = spec.to_dict()["params_schema"]
        try:
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(params)
        except SchemaError as exc:
            raise WorkflowRunnerError(
                f"workflow {spec.name!r} params_schema is invalid: {exc.message}"
            ) from exc
        except ValidationError as exc:
            raise WorkflowRunnerError(
                f"workflow {spec.name!r} params failed validation: {exc.message}"
            ) from exc

    async def _dispatch_stage(
        self, run: WorkflowRun, spec: WorkflowSpec, stage: Stage
    ) -> GateOutcome:
        attempt_number = await self.store.next_attempt_number(
            run.run_id, stage.name
        )
        idem = derive_stage_idempotency_key(
            run_id=run.run_id,
            stage=stage,
            attempt_number=attempt_number,
            engine_nonce=run.engine_nonce,
        )
        actor_did, actor_sig = sign_stage_transition(
            run_id=run.run_id,
            stage_name=stage.name,
            attempt_number=attempt_number,
            signal_id=None,
            gate_outcome=None,
            agent_identity=self.agent_identity,
            use_hybrid=True,
        )
        link = StageLink(
            link_id=str(uuid4()),
            run_id=run.run_id,
            stage_name=stage.name,
            attempt_number=attempt_number,
            idempotency_key=idem,
            actor_did=actor_did,
            actor_sig=actor_sig,
            compensate_state="pending",
            occurred_at=datetime.now(timezone.utc),
        )
        await self.store.insert_stage_link(link)

        payload = {**stage.to_dict()["params"], **run.to_dict()["params"]}
        if stage.gate.type in {
            "tests_pass",
            "ci_green",
            "lint_clean",
            "signature_collected",
        }:
            payload.update(stage.gate.to_dict()["params"])
        if stage.gate.type == "signature_collected":
            payload.update(
                {
                    "workflow_run_id": run.run_id,
                    "workflow_stage_name": stage.name,
                    "workflow_attempt_number": attempt_number,
                    "signature_payload": canonical_signature_ack_payload(
                        run_id=run.run_id,
                        stage_name=stage.name,
                        attempt_number=attempt_number,
                        did=stage.gate.params["did"],
                    ).decode("utf-8"),
                }
            )

        signal = Signal(
            source=stage.signal_source,
            kind="workflow.stage",
            mode=stage.signal_mode,
            payload=payload,
            target_agent=run.started_by_did,
            visibility=Visibility.INTERNAL,
            session_id=run.run_id,
            urgency=Urgency.NORMAL,
            dedupe_key=idem,
            causation_chain=[
                CausationFrame(
                    agent_id=run.started_by_did,
                    source=f"workflow.{spec.name}.{stage.name}",
                    signal_id=run.run_id,
                    turn_id=None,
                    depth=0,
                    emitted_at=datetime.now(timezone.utc),
                )
            ],
        )
        result = await self.dispatcher.dispatch_signal(signal)
        gate_outcome, gate_reason = await self._evaluate_gate(
            stage,
            result,
            run=run,
            attempt_number=attempt_number,
        )
        actor_did, actor_sig = sign_stage_transition(
            run_id=run.run_id,
            stage_name=stage.name,
            attempt_number=attempt_number,
            signal_id=result.signal_id,
            gate_outcome=gate_outcome.value,
            agent_identity=self.agent_identity,
            use_hybrid=True,
        )
        latest_run = await self.store.get_run(run.run_id)
        post_cancel = (
            latest_run is not None and latest_run.cancel_barrier_at is not None
        )
        await self.store.update_stage_link_transition(
            link.link_id,
            signal_id=result.signal_id,
            gate_outcome=gate_outcome,
            gate_reason=gate_reason,
            actor_did=actor_did,
            actor_sig=actor_sig,
            post_cancel=post_cancel,
        )
        if gate_outcome == GateOutcome.PASS and stage.compensate == "noop_idempotent":
            await self.store.update_compensate_state(link.link_id, "not_required")
        record_gate_outcome(spec.name, stage.name, gate_outcome.value)
        return gate_outcome

    async def _evaluate_gate(
        self,
        stage: Stage,
        result,
        *,
        run: WorkflowRun,
        attempt_number: int,
    ) -> tuple[GateOutcome, Optional[str]]:
        if result.status != Status.OK:
            return GateOutcome.FAIL, result.error or result.status.value
        if stage.gate.type == "signal_status_ok":
            return GateOutcome.PASS, None
        if stage.gate.type in {"tests_pass", "lint_clean"}:
            return _evaluate_exit_marker_gate(stage.gate, result)
        if stage.gate.type == "ci_green":
            return await self._evaluate_ci_green_gate(stage.gate, result)
        if stage.gate.type == "signature_collected":
            return self._evaluate_signature_collected_gate(
                stage.gate,
                result,
                run_id=run.run_id,
                stage_name=stage.name,
                attempt_number=attempt_number,
            )
        if stage.gate.type == "constitutional_boundary_clean":
            return self._evaluate_constitutional_boundary_gate(stage, result)
        if stage.gate.type not in _RUNNER_GATE_TYPES:
            return (
                GateOutcome.FAIL,
                f"gate {stage.gate.type!r} is not implemented in the workflow runner",
            )
        return GateOutcome.FAIL, f"gate {stage.gate.type!r} failed closed"

    async def _evaluate_ci_green_gate(
        self, gate: Gate, result: Any
    ) -> tuple[GateOutcome, Optional[str]]:
        if result.mode != SignalMode.ACTION:
            return GateOutcome.FAIL, "ci_green_requires_action_result"
        try:
            marker = self.ci_green_provider(gate, result)
            if inspect.isawaitable(marker):
                marker = await marker
        except Exception as exc:  # pragma: no cover - defensive boundary
            return GateOutcome.FAIL, f"ci_green_error:{exc}"
        if _ci_marker_green(gate, marker):
            return GateOutcome.PASS, None
        return GateOutcome.FAIL, f"ci_green_failed:{_ci_marker_reason(gate, marker)}"

    def _evaluate_signature_collected_gate(
        self,
        gate: Gate,
        result: Any,
        *,
        run_id: str,
        stage_name: str,
        attempt_number: int,
    ) -> tuple[GateOutcome, Optional[str]]:
        return _evaluate_signature_collected_gate(
            gate,
            result,
            run_id=run_id,
            stage_name=stage_name,
            attempt_number=attempt_number,
            public_key_resolver=self.public_key_resolver,
            verification_methods_resolver=self.verification_methods_resolver,
        )

    def _evaluate_constitutional_boundary_gate(
        self, stage: Stage, result
    ) -> tuple[GateOutcome, Optional[str]]:
        forbidden = tuple(
            dict.fromkeys(
                _normalize_module_path(item)
                for item in (
                    *stage.forbidden_modules,
                    *stage.gate.params["forbidden_modules"],
                )
            )
        )
        for source in _iter_emitted_code(result):
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                return (
                    GateOutcome.FAIL,
                    f"constitutional_boundary_unparseable:{exc.msg}",
                )
            for imported in _iter_imported_modules(tree):
                normalized = _normalize_module_path(imported)
                violation = _matching_forbidden_module(normalized, forbidden)
                if violation is not None:
                    return (
                        GateOutcome.FAIL,
                        f"constitutional_boundary_violation:{normalized}",
                    )
        return GateOutcome.PASS, None

    async def _compensate(
        self,
        run: WorkflowRun,
        spec: WorkflowSpec,
        *,
        success_status: RunStatus = RunStatus.CANCELLED,
        residue_status: RunStatus = RunStatus.CANCELLED_WITH_IRREVERSIBLE_RESIDUE,
    ) -> RunStatus:
        await self.store.update_run_status(
            run.run_id,
            RunStatus.COMPENSATING,
            current_stages=[],
        )
        residue = False
        failed = False
        links = await self.store.list_stage_links(run.run_id)
        for link in reversed(
            [l for l in links if l.gate_outcome == GateOutcome.PASS]
        ):
            stage = self._stage_by_name(spec, link.stage_name)
            if link.compensate_state in {
                "complete",
                "not_required",
                "record_only",
            }:
                record_compensation_state(
                    spec.name, stage.name, link.compensate_state
                )
                if link.compensate_state == "record_only":
                    residue = True
                continue
            if stage.compensate == "noop_idempotent":
                await self.store.update_compensate_state(
                    link.link_id, "not_required"
                )
                record_compensation_state(
                    spec.name, stage.name, "not_required"
                )
                continue
            if stage.compensate == "compensate_record_only":
                residue = True
                await self.store.update_compensate_state(
                    link.link_id, "record_only"
                )
                record_compensation_state(spec.name, stage.name, "record_only")
                continue
            result = await self.dispatcher.dispatch_signal(
                Signal(
                    source=stage.compensate,
                    kind="workflow.compensate",
                    mode=SignalMode.ACTION,
                    payload={
                        **stage.to_dict()["params"],
                        **run.to_dict()["params"],
                        "compensate": True,
                    },
                    target_agent=run.started_by_did,
                    visibility=Visibility.INTERNAL,
                    session_id=run.run_id,
                    urgency=Urgency.NORMAL,
                    dedupe_key=f"{link.idempotency_key}:compensate",
                )
            )
            state = "complete" if result.status == Status.OK else "failed"
            failed = failed or state == "failed"
            await self.store.update_compensate_state(link.link_id, state)
            record_compensation_state(spec.name, stage.name, state)

        status = (
            RunStatus.FAILED
            if failed
            else residue_status
            if residue
            else success_status
        )
        await self.store.update_run_status(
            run.run_id,
            status,
            current_stages=[],
            finished_at=datetime.now(timezone.utc),
        )
        return status

    def _start_stage(self, spec: WorkflowSpec) -> Stage:
        incoming = set()
        for edge in spec.edges:
            if edge.kind == EdgeKind.SEQUENTIAL:
                incoming.add(edge.to_stage)
            elif edge.kind in (EdgeKind.BRANCH, EdgeKind.PARALLEL):
                raise WorkflowRunnerError(
                    f"edge kind {edge.kind.value!r} is not implemented in "
                    "Phase 1 foundation"
                )
            elif edge.kind == EdgeKind.SUBWORKFLOW:
                raise WorkflowRunnerError(
                    "subworkflow edges are not implemented in Phase 1 foundation"
                )
        starts = [stage for stage in spec.stages if stage.name not in incoming]
        if len(starts) != 1:
            raise WorkflowRunnerError(
                f"workflow {spec.name!r} must have exactly one start stage; "
                f"found {[s.name for s in starts]}"
            )
        return starts[0]

    def _next_stages(self, spec: WorkflowSpec, stage_name: str) -> list[str]:
        out = []
        for edge in spec.edges:
            if edge.from_stage != stage_name:
                continue
            if edge.kind == EdgeKind.SEQUENTIAL:
                out.append(edge.to_stage)
            else:
                raise WorkflowRunnerError(
                    f"edge kind {edge.kind.value!r} is not implemented in "
                    "Phase 1 foundation"
                )
        return [name for name in out if name is not None]

    @staticmethod
    def _stage_by_name(spec: WorkflowSpec, name: str) -> Stage:
        for stage in spec.stages:
            if stage.name == name:
                return stage
        raise WorkflowDefinitionError(f"stage {name!r} not found in spec")


_CODE_RESULT_KEYS = frozenset(
    {
        "code",
        "source",
        "source_code",
        "content",
        "text",
    }
)
_PATCH_RESULT_KEYS = frozenset({"patch", "diff"})
_CONTAINER_RESULT_KEYS = frozenset(
    {
        "artifact",
        "artifacts",
        "file",
        "files",
        "output",
        "outputs",
        "result",
        "results",
    }
)
_EXIT_CODE_KEYS = ("exit_code", "returncode", "return_code")
_FAILURE_COUNT_KEYS = (
    "failed",
    "failures",
    "failure_count",
    "errors",
    "error_count",
    "violations",
    "violation_count",
)


def _iter_emitted_code(result: Any) -> Iterable[str]:
    for payload in (
        getattr(result, "action_result", None),
        getattr(result, "artifact", None),
    ):
        yield from _iter_code_payload(payload, root=True)


def _evaluate_exit_marker_gate(
    gate: Gate, result: Any
) -> tuple[GateOutcome, Optional[str]]:
    if result.mode != SignalMode.ACTION:
        return GateOutcome.FAIL, f"{gate.type}_requires_action_result"
    marker = getattr(result, "action_result", None)
    if marker is None:
        return GateOutcome.FAIL, f"{gate.type}_missing_result"
    if _result_marker_passed(gate, marker):
        contract_reason = _quality_gate_contract_reason(gate, marker)
        if contract_reason is not None:
            return GateOutcome.FAIL, contract_reason
        return GateOutcome.PASS, None
    reason = _result_marker_reason(marker)
    return GateOutcome.FAIL, f"{gate.type}_failed:{reason}"


def _evaluate_signature_collected_gate(
    gate: Gate,
    result: Any,
    *,
    run_id: str,
    stage_name: str,
    attempt_number: int,
    public_key_resolver: PublicKeyResolver,
    verification_methods_resolver: Optional[VerificationMethodsResolver],
) -> tuple[GateOutcome, Optional[str]]:
    if result.mode != SignalMode.ACTION:
        return GateOutcome.FAIL, "signature_collected_requires_action_result"
    marker = getattr(result, "action_result", None)
    if not isinstance(marker, dict):
        return GateOutcome.FAIL, "signature_collected_missing_result"

    expected_did = gate.params["did"].strip()
    observed_did = _signature_marker_did(marker)
    if observed_did is None:
        return GateOutcome.FAIL, "signature_collected_missing_did"
    if observed_did != expected_did:
        return GateOutcome.FAIL, f"signature_collected_did_mismatch:{expected_did}"
    if _has_explicit_bad_status(marker):
        reason = _result_marker_reason(marker)
        return GateOutcome.FAIL, f"signature_collected_failed:{reason}"
    signatures = _signature_marker_signatures(marker)
    if signatures is None:
        return GateOutcome.FAIL, "signature_collected_missing_signature"
    payload = canonical_signature_ack_payload(
        run_id=run_id,
        stage_name=stage_name,
        attempt_number=attempt_number,
        did=expected_did,
    )
    if not _verify_signature_collected_marker(
        did=expected_did,
        signatures=signatures,
        payload=payload,
        public_key_resolver=public_key_resolver,
        verification_methods_resolver=verification_methods_resolver,
    ):
        return GateOutcome.FAIL, "signature_collected_invalid_signature"
    return GateOutcome.PASS, None


def _signature_marker_did(marker: dict[str, Any]) -> Optional[str]:
    for key in ("did", "signer_did", "signed_by", "author_did", "actor_did"):
        value = marker.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    signer = marker.get("signer")
    if isinstance(signer, dict):
        for key in ("did", "id"):
            value = signer.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _signature_marker_signatures(
    marker: dict[str, Any],
) -> Optional[str | list[Mapping[str, str]]]:
    for key in ("signature", "sig"):
        value = marker.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("proof", "receipt"):
        value = marker.get(key)
        if isinstance(value, dict):
            nested = value.get("signature") or value.get("sig")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
            signatures = value.get("signatures")
            if isinstance(signatures, list) and signatures:
                return signatures
    signatures = marker.get("signatures")
    if isinstance(signatures, list):
        if signatures and all(isinstance(item, dict) for item in signatures):
            return signatures
        for item in signatures:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                nested = item.get("signature") or item.get("sig")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return None


def canonical_signature_ack_payload(
    *, run_id: str, stage_name: str, attempt_number: int, did: str
) -> bytes:
    body = json.dumps(
        {
            "run_id": run_id,
            "stage_name": stage_name,
            "attempt_number": attempt_number,
            "did": did,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"workflow.signature_collected.v1\n{body}".encode("utf-8")


def _verify_signature_collected_marker(
    *,
    did: str,
    signatures: str | list[Mapping[str, str]],
    payload: bytes,
    public_key_resolver: PublicKeyResolver,
    verification_methods_resolver: Optional[VerificationMethodsResolver],
) -> bool:
    if isinstance(signatures, list):
        return _verify_signature_collected_hybrid(
            did=did,
            signatures=signatures,
            payload=payload,
            verification_methods_resolver=verification_methods_resolver,
        )
    signature = signatures
    if signature.startswith("hybrid:"):
        try:
            decoded = base64.b64decode(signature[len("hybrid:") :]).decode()
            decoded_signatures = json.loads(decoded)
        except Exception:
            return False
        if not isinstance(decoded_signatures, list):
            return False
        return _verify_signature_collected_hybrid(
            did=did,
            signatures=decoded_signatures,
            payload=payload,
            verification_methods_resolver=verification_methods_resolver,
        )
    if signature.startswith("ecdsa:"):
        signature = signature[len("ecdsa:") :]
    try:
        public_key_bytes = public_key_resolver(did)
        public_key = get_suite(
            ALG_ECDSA_SECP256K1_SHA256
        ).deserialize_public_key(public_key_bytes)
        signature_bytes = bytes.fromhex(signature)
    except Exception:
        return False
    try:
        return get_suite(ALG_ECDSA_SECP256K1_SHA256).verify(
            payload,
            signature_bytes,
            public_key,
        )
    except CryptoSuiteError:
        return False


def _verify_signature_collected_hybrid(
    *,
    did: str,
    signatures: list[Mapping[str, str]],
    payload: bytes,
    verification_methods_resolver: Optional[VerificationMethodsResolver],
) -> bool:
    if verification_methods_resolver is None:
        return False
    try:
        from kestrel_sovereign.identity.hybrid_keypair import verify_hybrid

        return verify_hybrid(
            payload,
            signatures,
            verification_methods_resolver(did),
        ).ok
    except Exception:
        return False


async def _default_ci_green_provider(gate: Gate, result: Any) -> Any:
    del result
    interval = gate.params.get("poll_interval_seconds", 10)
    max_wait = gate.params.get("max_wait_seconds", 600)
    deadline = time.monotonic() + max_wait
    marker: Any = None

    while True:
        marker = await asyncio.to_thread(_fetch_github_ci_marker, gate)
        if _ci_marker_green(gate, marker):
            return marker
        if not _ci_marker_pending(gate, marker) or time.monotonic() >= deadline:
            break
        await asyncio.sleep(min(interval, max(0.0, deadline - time.monotonic())))

    if _ci_marker_pending(gate, marker):
        if isinstance(marker, dict):
            return {
                **marker,
                "status": "timeout",
                "message": "ci_green timed out waiting for checks to complete",
            }
        return {
            "status": "timeout",
            "message": "ci_green timed out waiting for checks to start",
        }
    return marker


def _fetch_github_ci_marker(gate: Gate) -> dict[str, Any]:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        return {
            "status": "error",
            "message": "GITHUB_TOKEN or GH_TOKEN is required for ci_green",
        }

    repo = gate.params["repo"]
    branch = gate.params["branch"]
    repo_path = _quote_repo_path(repo)
    branch_ref = quote(branch, safe="")
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "kestrel-workflows-ci-green",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    branch_doc = _github_json(
        f"https://api.github.com/repos/{repo_path}/branches/{branch_ref}",
        headers=headers,
    )
    sha = branch_doc["commit"]["sha"]
    check_runs = _github_paginated_items(
        f"https://api.github.com/repos/{repo_path}/commits/{sha}/check-runs?per_page=100",
        headers=headers,
        item_key="check_runs",
    )
    statuses_doc = _github_json(
        f"https://api.github.com/repos/{repo_path}/commits/{sha}/status",
        headers=headers,
    )
    required_checks = tuple(gate.params.get("required_checks") or ())
    required_doc = None
    if not required_checks:
        required_doc = _github_json_optional(
            (
                "https://api.github.com/repos/"
                f"{repo_path}/branches/{branch_ref}/protection/required_status_checks"
            ),
            headers=headers,
        )
    if not required_checks and isinstance(required_doc, dict):
        required_checks = _github_required_check_names(required_doc)

    return {
        "repo": repo,
        "branch": branch,
        "sha": sha,
        "check_runs": check_runs,
        "statuses": statuses_doc.get("statuses", []),
        "required_checks": required_checks,
    }


def _quote_repo_path(repo: str) -> str:
    owner, _, name = repo.partition("/")
    if not owner or not name or "/" in name:
        raise WorkflowRunnerError(
            "ci_green params.repo must use 'owner/repo' GitHub syntax"
        )
    return f"{quote(owner, safe='')}/{quote(name, safe='')}"


def _github_json(url: str, *, headers: dict[str, str]) -> Any:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=15) as response:  # noqa: S310 - GitHub API only
        return json.loads(response.read().decode("utf-8"))


def _github_paginated_items(
    url: str, *, headers: dict[str, str], item_key: str
) -> list[Any]:
    items: list[Any] = []
    page = 1
    separator = "&" if "?" in url else "?"
    while True:
        payload = _github_json(f"{url}{separator}page={page}", headers=headers)
        page_items = payload.get(item_key, [])
        if not isinstance(page_items, list):
            return items
        items.extend(page_items)
        total_count = payload.get("total_count")
        if not isinstance(total_count, int) or len(items) >= total_count:
            return items
        if not page_items:
            return items
        page += 1


def _github_required_check_names(required_doc: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for item in required_doc.get("contexts") or ():
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
    checks = required_doc.get("checks")
    if isinstance(checks, list):
        for item in checks:
            if not isinstance(item, dict):
                continue
            name = item.get("context") or item.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return tuple(dict.fromkeys(names))


def _github_json_optional(url: str, *, headers: dict[str, str]) -> Any:
    try:
        return _github_json(url, headers=headers)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _ci_marker_green(gate: Gate, marker: Any) -> bool:
    if isinstance(marker, bool):
        return marker
    if isinstance(marker, str):
        return marker.strip().lower() in {"ok", "pass", "passed", "success", "green"}
    if not isinstance(marker, dict):
        return False

    if _has_explicit_bad_status(marker):
        return False
    check_entries = _ci_check_entries(marker)
    required_names = _ci_required_names(gate, marker)
    if check_entries:
        return _ci_checks_green(check_entries, required_names)
    if required_names:
        return False
    if any(marker.get(key) is True for key in ("ok", "success", "passed", "green")):
        return True
    state = marker.get("state") or marker.get("status") or marker.get("conclusion")
    return isinstance(state, str) and state.strip().lower() in {
        "ok",
        "pass",
        "passed",
        "success",
        "green",
    }


def _ci_marker_pending(gate: Gate, marker: Any) -> bool:
    if not isinstance(marker, dict):
        return False
    if _has_explicit_bad_status(marker):
        return False
    check_entries = _ci_check_entries(marker)
    required_names = _ci_required_names(gate, marker)
    if not check_entries:
        return True
    relevant = [
        entry
        for name, entry in check_entries
        if not required_names or name in required_names
    ]
    if any(
        _ci_check_failure_reason(entry) is not None
        for entry in relevant
        if not _ci_check_pending(entry)
    ):
        return False
    if required_names and not required_names.issubset(
        {name for name, _ in check_entries}
    ):
        return True
    return any(_ci_check_pending(entry) for entry in relevant)


def _ci_marker_reason(gate: Gate, marker: Any) -> str:
    if isinstance(marker, dict):
        check_entries = _ci_check_entries(marker)
        required_names = _ci_required_names(gate, marker)
        observed_names = {name for name, _ in check_entries}
        for name, entry in check_entries:
            if required_names and name not in required_names:
                continue
            reason = _ci_check_failure_reason(entry)
            if reason is not None:
                return f"{name}:{reason}"
        missing = sorted(required_names - observed_names)
        if missing:
            return "missing_required_checks:" + ",".join(missing)
        for key in ("message", "error", "summary", "status", "state", "conclusion"):
            reason = marker.get(key)
            if isinstance(reason, str) and reason.strip():
                return reason.strip()
    return repr(marker)


def _ci_required_names(gate: Gate, marker: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    names.update(_marker_strings(gate.params.get("required_checks")))
    names.update(_marker_strings(marker.get("required_checks")))
    return names


def _ci_check_entries(marker: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for key in ("checks", "check_runs", "statuses"):
        value = marker.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("context") or item.get("check_name")
            if isinstance(name, str) and name.strip():
                entries.append((name.strip(), item))
    return entries


def _ci_checks_green(
    check_entries: list[tuple[str, dict[str, Any]]], required_names: set[str]
) -> bool:
    relevant = [
        (name, entry)
        for name, entry in check_entries
        if not required_names or name in required_names
    ]
    if not relevant:
        return False
    if required_names and not required_names.issubset({name for name, _ in relevant}):
        return False
    return all(_ci_check_success(entry) for _, entry in relevant)


def _ci_check_failure_reason(entry: dict[str, Any]) -> Optional[str]:
    if _ci_check_pending(entry):
        return None
    status = entry.get("status")
    conclusion = entry.get("conclusion") or entry.get("state")
    if isinstance(status, str) and status.strip().lower() not in {
        "completed",
        "success",
        "ok",
    }:
        return f"status={status.strip()}"
    if conclusion is None and isinstance(status, str):
        return None
    if isinstance(conclusion, str) and conclusion.strip().lower() in {
        "success",
        "neutral",
        "skipped",
    }:
        return None
    return f"conclusion={conclusion!r}"


def _ci_check_success(entry: dict[str, Any]) -> bool:
    if _ci_check_pending(entry):
        return False
    status = entry.get("status")
    conclusion = entry.get("conclusion") or entry.get("state")
    if isinstance(status, str) and status.strip().lower() not in {
        "completed",
        "success",
        "ok",
    }:
        return False
    return isinstance(conclusion, str) and conclusion.strip().lower() in {
        "success",
        "neutral",
        "skipped",
    }


def _ci_check_pending(entry: dict[str, Any]) -> bool:
    status = entry.get("status")
    conclusion = entry.get("conclusion") or entry.get("state")
    if isinstance(status, str) and status.strip().lower() in {
        "queued",
        "in_progress",
        "pending",
        "requested",
        "waiting",
    }:
        return True
    if conclusion is None and isinstance(status, str):
        return status.strip().lower() not in {"completed", "success", "ok"}
    if isinstance(conclusion, str) and conclusion.strip().lower() in {
        "pending",
        "queued",
        "in_progress",
    }:
        return True
    return False


def _result_marker_passed(gate: Gate, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 0
    if isinstance(value, str):
        return value.lower() in {"ok", "pass", "passed", "success", "clean"}
    if isinstance(value, dict):
        if _has_explicit_nonzero_exit(value):
            return False
        if _has_explicit_failure_count(value):
            return False
        if _has_explicit_bad_status(value):
            return False
        if any(_is_zero(value.get(key)) for key in _EXIT_CODE_KEYS):
            return True
        if any(value.get(key) is True for key in ("ok", "success", "passed", "clean")):
            return True
        status = value.get("status")
        if isinstance(status, str) and status.strip().lower() in {
            "ok",
            "pass",
            "passed",
            "success",
            "clean",
        }:
            return True
        if gate.type == "tests_pass" and _has_zero_test_counts(value):
            return True
        if gate.type == "lint_clean" and _has_zero_lint_counts(value):
            return True
    return False


def _has_explicit_bad_status(value: dict[str, Any]) -> bool:
    status = value.get("status")
    if not isinstance(status, str):
        return False
    return status.strip().lower() in {
        "error",
        "errored",
        "fail",
        "failed",
        "failure",
        "timeout",
        "timed_out",
    }


def _has_explicit_nonzero_exit(value: dict[str, Any]) -> bool:
    for key in _EXIT_CODE_KEYS:
        code = value.get(key)
        if isinstance(code, int) and not isinstance(code, bool) and code != 0:
            return True
    return False


def _has_explicit_failure_count(value: dict[str, Any]) -> bool:
    for key in _FAILURE_COUNT_KEYS:
        count = value.get(key)
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            return True
    return False


def _has_zero_test_counts(value: dict[str, Any]) -> bool:
    has_failure_count = any(
        key in value and _is_zero(value.get(key))
        for key in ("failed", "failures", "failure_count")
    )
    has_error_count = any(
        key in value and _is_zero(value.get(key))
        for key in ("errors", "error_count")
    )
    return has_failure_count and has_error_count


def _has_zero_lint_counts(value: dict[str, Any]) -> bool:
    has_violation_count = any(
        key in value and _is_zero(value.get(key))
        for key in ("violations", "violation_count")
    )
    has_error_count = any(
        key in value and _is_zero(value.get(key))
        for key in ("errors", "error_count")
    )
    return has_violation_count and has_error_count


def _quality_gate_contract_reason(gate: Gate, marker: Any) -> Optional[str]:
    if not isinstance(marker, dict):
        # Design §3.3 defines tests_pass(suite) as "pytest ... exit 0
        # = pass"; the runner supplies the suite to the ACTION source
        # payload, and simple command-style sources may return only the
        # scalar process exit marker. lint_clean still needs structured
        # scope coverage, so it remains dict-only below.
        if gate.type == "tests_pass":
            return None
        return f"{gate.type}_missing_contract_echo"
    if gate.type == "tests_pass":
        expected_suite = gate.params["suite"]
        if not any(
            key in marker
            for key in ("suite", "suites", "test_suite", "test_suites")
        ):
            return None
        if _marker_contains_string(
            marker,
            expected_suite,
            keys=("suite", "suites", "test_suite", "test_suites"),
        ):
            return None
        return f"tests_pass_suite_mismatch:{expected_suite}"
    if gate.type == "lint_clean":
        expected_scopes = tuple(gate.params["scopes"])
        if _marker_contains_all_strings(
            marker,
            expected_scopes,
            keys=("scope", "scopes", "checked_scopes", "covered_scopes", "lint_scopes"),
        ):
            return None
        return "lint_clean_scope_mismatch:" + ",".join(expected_scopes)
    return None


def _marker_contains_string(
    marker: dict[str, Any], expected: str, *, keys: tuple[str, ...]
) -> bool:
    for key in keys:
        if _marker_value_contains(marker.get(key), expected):
            return True
    return False


def _marker_contains_all_strings(
    marker: dict[str, Any], expected: tuple[str, ...], *, keys: tuple[str, ...]
) -> bool:
    observed: set[str] = set()
    for key in keys:
        observed.update(_marker_strings(marker.get(key)))
    return set(expected).issubset(observed)


def _marker_value_contains(value: Any, expected: str) -> bool:
    return expected in _marker_strings(value)


def _marker_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return {stripped} if stripped else set()
    if isinstance(value, (list, tuple, set)):
        values: set[str] = set()
        for item in value:
            values.update(_marker_strings(item))
        return values
    return set()


def _is_zero(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _result_marker_reason(value: Any) -> str:
    if isinstance(value, dict):
        status = value.get("status")
        if isinstance(status, str) and _has_explicit_bad_status(value):
            return status.strip()
        for key in _EXIT_CODE_KEYS:
            code = value.get(key)
            if isinstance(code, int) and not isinstance(code, bool) and code != 0:
                return f"{key}={code!r}"
        for key in _FAILURE_COUNT_KEYS:
            count = value.get(key)
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                return f"{key}={count!r}"
        for key in ("error", "message", "summary", "status"):
            reason = value.get(key)
            if isinstance(reason, str) and reason.strip():
                return reason.strip()
        for key in _EXIT_CODE_KEYS:
            if key in value:
                return f"{key}={value[key]!r}"
    return repr(value)


def _iter_code_payload(value: Any, *, root: bool = False) -> Iterable[str]:
    if isinstance(value, str):
        if root:
            if _looks_like_patch_source(value):
                yield _python_source_from_patch(value)
                return
            if not _looks_like_python_source(value):
                return
        yield value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = key if isinstance(key, str) else ""
            normalized_key = key_text.lower()
            if normalized_key in _CODE_RESULT_KEYS:
                yield from _iter_code_payload(item)
            elif normalized_key in _PATCH_RESULT_KEYS and isinstance(item, str):
                yield _python_source_from_patch(item)
            elif normalized_key in _CONTAINER_RESULT_KEYS:
                yield from _iter_code_payload(item, root=True)
            elif key_text.endswith(".py") and isinstance(item, str):
                yield item
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_code_payload(item, root=True)


def _looks_like_python_source(value: str) -> bool:
    return any(
        line.lstrip().startswith(("import ", "from "))
        for line in value.splitlines()
    )


def _looks_like_patch_source(value: str) -> bool:
    return any(
        line.startswith("+")
        and not line.startswith("+++")
        and line[1:].lstrip().startswith(("import ", "from "))
        for line in value.splitlines()
    )


def _python_source_from_patch(value: str) -> str:
    added_imports = [
        code
        for line in value.splitlines()
        if line.startswith("+") and not line.startswith("+++")
        for code in (line[1:].lstrip(),)
        if code.startswith("import ") or code.startswith("from ")
    ]
    return "\n".join(added_imports)


def _iter_imported_modules(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            module = "" if node.module is None else node.module
            prefix = "." * node.level
            base = f"{prefix}{module}"
            if module:
                yield base
            for alias in node.names:
                if alias.name == "*":
                    continue
                separator = "" if base in ("", prefix) else "."
                yield f"{base}{separator}{alias.name}"


def _normalize_module_path(value: str) -> str:
    normalized = value.strip().replace("/", ".")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    if normalized.startswith("."):
        return normalized.rstrip(".")
    return normalized.strip(".")


def _matching_forbidden_module(
    imported: str, forbidden_modules: Iterable[str]
) -> Optional[str]:
    for forbidden in forbidden_modules:
        relative_imported = imported.lstrip(".") if imported.startswith(".") else None
        if (
            imported == forbidden
            or imported.startswith(f"{forbidden}.")
            or imported.endswith(f".{forbidden}")
            or f".{forbidden}." in imported
            or (
                relative_imported is not None
                and (
                    relative_imported == forbidden
                    or forbidden.endswith(f".{relative_imported}")
                    or f".{relative_imported}." in forbidden
                )
            )
        ):
            return forbidden
    return None


def derive_stage_idempotency_key(
    *,
    run_id: str,
    stage: Stage,
    attempt_number: int,
    engine_nonce: str,
) -> str:
    canonical_stage_input = json.dumps(
        stage.to_dict()["params"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    input_hash = hashlib.sha256(
        f"{canonical_stage_input}|{attempt_number}|{engine_nonce}".encode("utf-8")
    ).hexdigest()
    return hashlib.sha256(
        f"{run_id}|{stage.name}|{input_hash}".encode("utf-8")
    ).hexdigest()


__all__ = [
    "WorkflowRunResult",
    "WorkflowRunner",
    "WorkflowRunnerError",
    "derive_stage_idempotency_key",
]
