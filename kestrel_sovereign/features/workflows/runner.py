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

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
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
from kestrel_sovereign.signals import SignalDispatcher, SourceRegistry


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
    ) -> None:
        self.store = store
        self.dispatcher = dispatcher
        self.registry = registry
        self.agent_identity = agent_identity
        self.public_key_resolver = public_key_resolver
        self.verification_methods_resolver = verification_methods_resolver

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
            if gate != GateOutcome.PASS:
                status = await self._compensate(
                    run_snapshot,
                    spec,
                    success_status=RunStatus.FAILED,
                    residue_status=RunStatus.FAILED,
                )
                return WorkflowRunResult(run.run_id, status)

            post_dispatch_run = await self.store.get_run(run.run_id)
            if post_dispatch_run is None:
                raise WorkflowRunnerError(f"workflow run missing: {run.run_id}")
            if post_dispatch_run.cancel_barrier_at is not None:
                status = await self._compensate(post_dispatch_run, spec)
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
            if stage.gate.type != "signal_status_ok":
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} gate {stage.gate.type!r} "
                    "is not implemented in Phase 1 foundation"
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

        signal = Signal(
            source=stage.signal_source,
            kind="workflow.stage",
            mode=stage.signal_mode,
            payload={**stage.to_dict()["params"], **run.to_dict()["params"]},
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
        gate_outcome, gate_reason = self._evaluate_gate(stage, result)
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

    def _evaluate_gate(self, stage: Stage, result) -> tuple[GateOutcome, Optional[str]]:
        if stage.gate.type != "signal_status_ok":
            return (
                GateOutcome.FAIL,
                f"gate {stage.gate.type!r} is not implemented in Phase 1 foundation",
            )
        if result.status == Status.OK:
            return GateOutcome.PASS, None
        return GateOutcome.FAIL, result.error or result.status.value

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
