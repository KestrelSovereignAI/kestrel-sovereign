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
import math
import os
import re
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
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

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
    RevocationReason,
    RunStatus,
    Stage,
    StageLink,
    WorkflowDefinitionError,
    WorkflowRun,
    WorkflowSpec,
    _DID_RE,
)
from kestrel_sovereign.features.workflows.metrics import (
    record_compensation_state,
    record_gate_outcome,
)
from kestrel_sovereign.features.workflows.signing import (
    PublicKeyResolver,
    VerificationMethodsResolver,
    sign_stage_transition,
    sign_definition_revocation,
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

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


_RUNNER_GATE_TYPES = frozenset(
    {
        "signal_status_ok",
        "tests_pass",
        "ci_green",
        "lint_clean",
        "red_team_clear",
        "council_approve",
        "consent_collect",
        "signature_collected",
        "script",
        "constitution_echo_verified",
        "constitutional_boundary_clean",
    }
)

CiGreenProvider = Callable[[Gate, Any], Awaitable[Any] | Any]
ConsentCollectProvider = Callable[
    [Gate, WorkflowRun, Stage, StageLink],
    Awaitable[Any] | Any,
]
CouncilApproveProvider = Callable[
    [Gate, WorkflowRun, Stage, StageLink],
    Awaitable[Any] | Any,
]
ScriptGateProvider = Callable[[Gate, Any], Awaitable[Any] | Any]
ScriptArtifactResolver = Callable[[Gate], Awaitable[Any] | Any]
RedTeamAttestationResolver = Callable[[str], Awaitable[Mapping[str, Any]] | Mapping[str, Any]]
RedTeamPromptPackResolver = Callable[
    [str], Awaitable[Mapping[str, Any]] | Mapping[str, Any]
]


class WorkflowRunnerError(RuntimeError):
    """Raised when the runner refuses a workflow before firing signals."""


@dataclass(frozen=True)
class WorkflowRunResult:
    run_id: str
    status: RunStatus


@dataclass(frozen=True)
class WorkflowRevocationResult:
    changed: bool
    reason: RevocationReason
    force_revoked_run_ids: tuple[str, ...] = ()


_TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.CANCELLED_WITH_IRREVERSIBLE_RESIDUE,
    }
)


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
        consent_collect_provider: Optional[ConsentCollectProvider] = None,
        council_approve_provider: Optional[CouncilApproveProvider] = None,
        script_gate_provider: Optional[ScriptGateProvider] = None,
        script_artifact_resolver: Optional[ScriptArtifactResolver] = None,
        red_team_attestation_resolver: Optional[
            RedTeamAttestationResolver
        ] = None,
        red_team_prompt_pack_resolver: Optional[
            RedTeamPromptPackResolver
        ] = None,
        red_team_max_total_tokens: Optional[int] = None,
        red_team_max_total_cost_usd: Optional[float] = None,
    ) -> None:
        self.store = store
        self.dispatcher = dispatcher
        self.registry = registry
        self.agent_identity = agent_identity
        self.public_key_resolver = public_key_resolver
        self.verification_methods_resolver = verification_methods_resolver
        self.ci_green_provider = ci_green_provider or _default_ci_green_provider
        self.consent_collect_provider = consent_collect_provider
        self.council_approve_provider = council_approve_provider
        self.script_gate_provider = script_gate_provider
        self.script_artifact_resolver = script_artifact_resolver
        self.red_team_attestation_resolver = red_team_attestation_resolver
        self.red_team_prompt_pack_resolver = red_team_prompt_pack_resolver
        self.red_team_max_total_tokens = _validate_red_team_operator_tokens(
            red_team_max_total_tokens
        )
        self.red_team_max_total_cost_usd = _validate_red_team_operator_cost(
            red_team_max_total_cost_usd
        )

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
        await self._validate_run_start_contract(spec)
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
        if run.status in _TERMINAL_RUN_STATUSES:
            raise WorkflowRunnerError(
                f"workflow run {run_id} is terminal ({run.status.value})"
            )
        spec = await self._load_pinned_definition(run)
        run = await self.store.get_run(run_id)
        if run is None:
            raise WorkflowRunnerError(f"workflow run missing: {run_id}")
        await self._validate_run_start_contract(spec)
        if run.status == RunStatus.WAITING or (
            run.status == RunStatus.PAUSED
            and await self._has_pending_waiting_stage(run, spec)
        ):
            await self.store.update_run_status(run_id, RunStatus.RUNNING)
            run = await self.store.get_run(run_id)
            if run is None:
                raise WorkflowRunnerError(f"workflow run missing: {run_id}")
            return await self._continue_waiting_run(run, spec)
        if run.status == RunStatus.PAUSED:
            await self.store.update_run_status(run_id, RunStatus.RUNNING)
            run = await self.store.get_run(run_id)
            if run is None:
                raise WorkflowRunnerError(f"workflow run missing: {run_id}")
        return await self._continue_run(run, spec)

    async def _continue_waiting_run(
        self, run: WorkflowRun, spec: WorkflowSpec
    ) -> WorkflowRunResult:
        if run.cancel_barrier_at is not None:
            status = await self._compensate(run, spec)
            return WorkflowRunResult(run.run_id, status)
        current = list(run.current_stages)
        if not current:
            return await self._continue_run(run, spec)

        stage = self._stage_by_name(spec, current.pop(0))
        gate = await self._evaluate_waiting_stage(run, spec, stage)
        if gate is None:
            return await self._continue_run(run, spec)
        latest = await self.store.get_run(run.run_id)
        if latest is None:
            raise WorkflowRunnerError(f"workflow run missing: {run.run_id}")
        if latest.cancel_barrier_at is not None:
            status = await self._compensate(latest, spec)
            return WorkflowRunResult(run.run_id, status)
        if latest.status == RunStatus.PAUSED:
            if gate == GateOutcome.PENDING:
                await self.store.update_run_status(
                    run.run_id,
                    RunStatus.PAUSED,
                    current_stages=[stage.name, *current],
                )
                return WorkflowRunResult(run.run_id, RunStatus.PAUSED)
            if gate == GateOutcome.PASS:
                await self.store.update_run_status(
                    run.run_id,
                    RunStatus.PAUSED,
                    current_stages=[*current, *self._next_stages(spec, stage.name)],
                )
                return WorkflowRunResult(run.run_id, RunStatus.PAUSED)
        if gate == GateOutcome.PENDING:
            await self.store.update_run_status(
                run.run_id,
                RunStatus.WAITING,
                current_stages=[stage.name, *current],
            )
            return WorkflowRunResult(run.run_id, RunStatus.WAITING)
        if gate != GateOutcome.PASS:
            status = await self._compensate(
                latest,
                spec,
                success_status=RunStatus.FAILED,
                residue_status=RunStatus.FAILED,
            )
            return WorkflowRunResult(run.run_id, status)

        next_current = [*current, *self._next_stages(spec, stage.name)]
        await self.store.update_run_status(
            run.run_id,
            RunStatus.RUNNING,
            current_stages=next_current,
        )
        resumed = await self.store.get_run(run.run_id)
        if resumed is None:
            raise WorkflowRunnerError(f"workflow run missing: {run.run_id}")
        return await self._continue_run(resumed, spec)

    async def _has_pending_waiting_stage(
        self, run: WorkflowRun, spec: WorkflowSpec
    ) -> bool:
        current = list(run.current_stages)
        if not current:
            return False
        stage = self._stage_by_name(spec, current[0])
        return (await self._pending_waiting_link(run.run_id, stage)) is not None

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
            if gate == GateOutcome.PENDING:
                await self.store.update_run_status(
                    run.run_id,
                    RunStatus.WAITING,
                    current_stages=[stage.name, *current],
                )
                return WorkflowRunResult(run.run_id, RunStatus.WAITING)
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
        if run.status in _TERMINAL_RUN_STATUSES:
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

    async def revoke_definition(
        self,
        name: str,
        version: int,
        *,
        reason: RevocationReason | str,
    ) -> WorkflowRevocationResult:
        reason_value = RevocationReason(reason)
        revoked_at = datetime.now(timezone.utc)
        revoked_at_wire = revoked_at.isoformat()
        authority_did, authority_sig = sign_definition_revocation(
            name=name,
            version=version,
            reason=reason_value,
            revoked_at=revoked_at_wire,
            agent_identity=self.agent_identity,
        )
        changed = await self.store.revoke_definition(
            name,
            version,
            reason=reason_value,
            authority_did=authority_did,
            authority_sig=authority_sig,
            revoked_at=revoked_at,
        )
        if reason_value is not RevocationReason.COMPROMISED:
            return WorkflowRevocationResult(changed=changed, reason=reason_value)

        if not changed:
            row = await self.store.get_definition_row(name, version)
            if (
                row is None
                or row["revocation_reason"] != RevocationReason.COMPROMISED.value
            ):
                return WorkflowRevocationResult(
                    changed=False,
                    reason=reason_value,
                )

        runs = await self.store.list_runs_for_definition(
            name,
            version,
            statuses=set(RunStatus) - set(_TERMINAL_RUN_STATUSES),
        )
        force_revoked: list[str] = []
        for run in runs:
            status = await self._force_cancel_run(run.run_id)
            if status is None:
                continue
            force_revoked.append(run.run_id)
        return WorkflowRevocationResult(
            changed=changed,
            reason=reason_value,
            force_revoked_run_ids=tuple(force_revoked),
        )

    async def _force_cancel_run(self, run_id: str) -> RunStatus | None:
        run = await self.store.get_run(run_id)
        if run is None:
            raise WorkflowRunnerError(f"unknown workflow run: {run_id}")
        if run.status in _TERMINAL_RUN_STATUSES:
            return None
        try:
            status = await self.cancel_run(run_id)
        except WorkflowRunnerError:
            run = await self.store.get_run(run_id)
            if run is not None and run.status in _TERMINAL_RUN_STATUSES:
                return None
            raise
        if status != RunStatus.COMPENSATING:
            return status
        run = await self.store.get_run(run_id)
        if run is None:
            raise WorkflowRunnerError(f"workflow run missing after cancel: {run_id}")
        if run.status in _TERMINAL_RUN_STATUSES:
            return run.status
        spec = await self._load_pinned_definition(run)
        run = await self.store.get_run(run_id)
        if run is None:
            raise WorkflowRunnerError(f"workflow run missing after cancel: {run_id}")
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
        await self._validate_run_start_contract(spec)
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
        if row["deleted_at"] is not None:
            if (
                row["revocation_reason"] == RevocationReason.COMPROMISED.value
                and run.cancel_barrier_at is None
                and run.status not in _TERMINAL_RUN_STATUSES
            ):
                await self.store.set_cancel_barrier(run.run_id)
            if (
                row["revocation_reason"] in {reason.value for reason in RevocationReason}
                and run.status == RunStatus.FAILED
            ):
                raise WorkflowRunnerError(
                    f"workflow definition is revoked "
                    f"({row['revocation_reason']}): "
                    f"{run.workflow_name} v{run.workflow_ver}"
                )
            if not run.signature_post_revocation:
                await self.store.mark_run_signature_post_revocation(run.run_id)
        return spec

    async def _validate_run_start_contract(self, spec: WorkflowSpec) -> None:
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
                "consent_collect",
                "signature_collected",
            } and stage.signal_mode != SignalMode.ACTION:
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} gate {stage.gate.type!r} "
                    "requires signal_mode=ACTION"
                )
            if (
                stage.gate.type == "council_approve"
                and stage.signal_mode not in {SignalMode.ACTION, SignalMode.ARTIFACT}
            ):
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} gate {stage.gate.type!r} "
                    "requires signal_mode=ACTION or ARTIFACT"
                )
            if (
                stage.gate.type == "script"
                and stage.signal_mode not in {SignalMode.ACTION, SignalMode.COGNITION}
            ):
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} gate {stage.gate.type!r} "
                    "requires signal_mode=ACTION or COGNITION"
                )
            if (
                stage.gate.type == "red_team_clear"
                and stage.signal_mode not in {SignalMode.ACTION, SignalMode.COGNITION}
            ):
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} gate {stage.gate.type!r} "
                    "requires signal_mode=ACTION or COGNITION"
                )
            if (
                stage.gate.type == "constitution_echo_verified"
                and stage.signal_mode != SignalMode.COGNITION
            ):
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} gate {stage.gate.type!r} "
                    "requires signal_mode=COGNITION"
                )
            registration = self.registry.get(stage.signal_source)
            if registration is None:
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} references unregistered source "
                    f"{stage.signal_source!r}"
                )
            if (
                stage.gate.type == "constitution_echo_verified"
                and not registration.require_constitution_echo
            ):
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} gate constitution_echo_verified "
                    f"requires source {stage.signal_source!r} to set "
                    "require_constitution_echo=True"
                )
            if stage.signal_mode not in registration.allowed_modes:
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} mode {stage.signal_mode.value!r} "
                    f"is not allowed by source {stage.signal_source!r}"
                )
            if stage.gate.type == "red_team_clear":
                _validate_red_team_budget_policy(
                    stage,
                    operator_token_limit=self.red_team_max_total_tokens,
                    operator_cost_limit=self.red_team_max_total_cost_usd,
                )
                await self._validate_red_team_clear_start(spec, stage)
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

    async def _validate_red_team_clear_start(
        self, spec: WorkflowSpec, stage: Stage
    ) -> None:
        if self.red_team_prompt_pack_resolver is None:
            raise WorkflowRunnerError(
                f"stage {stage.name!r} gate red_team_clear requires "
                "a prompt-pack resolver"
            )
        if self.red_team_attestation_resolver is None:
            raise WorkflowRunnerError(
                f"stage {stage.name!r} gate red_team_clear requires "
                "a reviewer attestation resolver"
            )

        await self._resolve_red_team_prompt_pack(stage)

        reviewer_pool = _red_team_reviewer_pool(stage.gate)
        reviewer_ids = tuple(
            _red_team_reviewer_identity(reviewer) for reviewer in reviewer_pool
        )
        if len(set(reviewer_ids)) != len(reviewer_ids):
            raise WorkflowRunnerError(
                f"stage {stage.name!r} red_team_clear reviewer identities "
                "must be distinct"
            )
        reviewer_set = set(reviewer_ids)
        forbidden = {self.agent_identity.legacy_did, spec.author_did}
        if self.agent_identity.is_hybrid:
            forbidden.add(self.agent_identity.signing_did)
        overlap = sorted(reviewer_set & {did for did in forbidden if did})
        if overlap:
            raise WorkflowRunnerError(
                f"stage {stage.name!r} red_team_clear reviewers must be "
                f"distinct from proposer/author DIDs: {overlap}"
            )

        families: set[str] = set()
        for reviewer in reviewer_pool:
            reviewer_id = _red_team_reviewer_identity(reviewer)
            source = _red_team_reviewer_source(reviewer)
            registration = self.registry.get(source)
            if registration is None:
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} red_team_clear reviewer source "
                    f"{source!r} is not registered"
                )
            if SignalMode.COGNITION not in registration.allowed_modes:
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} red_team_clear reviewer source "
                    f"{source!r} must allow COGNITION"
                )
            try:
                attestation = self.red_team_attestation_resolver(reviewer_id)
                if inspect.isawaitable(attestation):
                    attestation = await attestation
            except Exception as exc:
                raise WorkflowRunnerError(
                    f"stage {stage.name!r} red_team_clear reviewer "
                    f"{reviewer_id!r} attestation could not be resolved: {exc}"
                ) from exc
            family = _red_team_attested_family(stage, reviewer_id, attestation)
            families.add(family)

        if len(families) < 2:
            raise WorkflowRunnerError(
                f"stage {stage.name!r} red_team_clear requires at least "
                "two distinct attested model families"
            )

    async def _resolve_red_team_prompt_pack(self, stage: Stage) -> Mapping[str, Any]:
        if self.red_team_prompt_pack_resolver is None:
            raise WorkflowRunnerError(
                f"stage {stage.name!r} gate red_team_clear requires "
                "a prompt-pack resolver"
            )
        constraint = stage.gate.params["prompt_pack_constraint"]
        try:
            prompt_pack = self.red_team_prompt_pack_resolver(constraint)
            if inspect.isawaitable(prompt_pack):
                prompt_pack = await prompt_pack
        except Exception as exc:
            raise WorkflowRunnerError(
                f"stage {stage.name!r} red_team_clear prompt pack "
                f"could not be resolved: {exc}"
            ) from exc
        _validate_red_team_prompt_pack(stage, prompt_pack)
        return prompt_pack

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

        preflight_outcome, preflight_reason = await self._preflight_stage_gate(stage)
        if preflight_outcome is not None:
            actor_did, actor_sig = sign_stage_transition(
                run_id=run.run_id,
                stage_name=stage.name,
                attempt_number=attempt_number,
                signal_id=None,
                gate_outcome=preflight_outcome.value,
                agent_identity=self.agent_identity,
                use_hybrid=True,
            )
            await self.store.update_stage_link_transition(
                link.link_id,
                signal_id=None,
                gate_outcome=preflight_outcome,
                gate_reason=preflight_reason,
                actor_did=actor_did,
                actor_sig=actor_sig,
                post_cancel=False,
            )
            if stage.compensate == "noop_idempotent":
                await self.store.update_compensate_state(
                    link.link_id, "not_required"
                )
            record_gate_outcome(spec.name, stage.name, preflight_outcome.value)
            return preflight_outcome

        payload = {**stage.to_dict()["params"], **run.to_dict()["params"]}
        if stage.gate.type in {
            "tests_pass",
            "ci_green",
            "lint_clean",
            "council_approve",
            "consent_collect",
            "red_team_clear",
            "signature_collected",
            "script",
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

    async def _preflight_stage_gate(
        self, stage: Stage
    ) -> tuple[Optional[GateOutcome], Optional[str]]:
        if stage.gate.type != "script":
            return None, None
        if self.script_gate_provider is None:
            return GateOutcome.FAIL, "script_no_resolver"
        outcome, reason = await self._verify_script_gate_artifact(stage.gate)
        if outcome == GateOutcome.PASS:
            return None, None
        return outcome, reason

    async def _evaluate_waiting_stage(
        self, run: WorkflowRun, spec: WorkflowSpec, stage: Stage
    ) -> Optional[GateOutcome]:
        link = await self._pending_waiting_link(run.run_id, stage)
        if link is None:
            return None
        gate_outcome, gate_reason = await self._evaluate_pending_link(
            stage.gate, run, stage, link
        )
        actor_did, actor_sig = sign_stage_transition(
            run_id=run.run_id,
            stage_name=stage.name,
            attempt_number=link.attempt_number,
            signal_id=link.signal_id,
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
            signal_id=link.signal_id,
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

    async def _pending_waiting_link(
        self, run_id: str, stage: Stage
    ) -> Optional[StageLink]:
        if stage.gate.type not in {"consent_collect", "council_approve"}:
            return None
        links = await self.store.list_stage_links(run_id)
        pending = [
            link
            for link in links
            if link.stage_name == stage.name
            and link.gate_outcome == GateOutcome.PENDING
        ]
        return pending[-1] if pending else None

    async def _evaluate_pending_link(
        self,
        gate: Gate,
        run: WorkflowRun,
        stage: Stage,
        link: StageLink,
    ) -> tuple[GateOutcome, Optional[str]]:
        if gate.type == "consent_collect":
            return await self._evaluate_pending_consent_link(gate, run, stage, link)
        if gate.type == "council_approve":
            return await self._evaluate_pending_council_link(gate, run, stage, link)
        return GateOutcome.FAIL, f"gate {gate.type!r} cannot resume from pending"

    async def _evaluate_pending_consent_link(
        self,
        gate: Gate,
        run: WorkflowRun,
        stage: Stage,
        link: StageLink,
    ) -> tuple[GateOutcome, Optional[str]]:
        if self.consent_collect_provider is None:
            return GateOutcome.FAIL, "consent_collect_no_resolver"
        try:
            marker = self.consent_collect_provider(gate, run, stage, link)
            if inspect.isawaitable(marker):
                marker = await marker
        except Exception as exc:  # pragma: no cover - defensive boundary
            return GateOutcome.PENDING, f"consent_collect_pending_error:{exc}"
        return _evaluate_consent_collect_marker(gate, marker)

    async def _evaluate_pending_council_link(
        self,
        gate: Gate,
        run: WorkflowRun,
        stage: Stage,
        link: StageLink,
    ) -> tuple[GateOutcome, Optional[str]]:
        if _stage_link_timed_out(link, gate.params["timeout"]):
            return GateOutcome.FAIL, "council_approve_timeout"
        if self.council_approve_provider is None:
            return GateOutcome.FAIL, "council_approve_no_resolver"
        try:
            marker = self.council_approve_provider(gate, run, stage, link)
            if inspect.isawaitable(marker):
                marker = await marker
        except Exception as exc:  # pragma: no cover - defensive boundary
            return GateOutcome.PENDING, f"council_approve_pending_error:{exc}"
        return _evaluate_council_approve_marker(gate, marker)

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
        if stage.gate.type == "consent_collect":
            return _evaluate_consent_collect_gate(stage.gate, result)
        if stage.gate.type == "council_approve":
            return _evaluate_council_approve_gate(stage.gate, result)
        if stage.gate.type == "red_team_clear":
            return await self._evaluate_red_team_clear_gate(
                stage.gate,
                result,
                run=run,
                stage=stage,
                attempt_number=attempt_number,
            )
        if stage.gate.type == "script":
            return await self._evaluate_script_gate(stage.gate, result)
        if stage.gate.type == "signature_collected":
            return self._evaluate_signature_collected_gate(
                stage.gate,
                result,
                run_id=run.run_id,
                stage_name=stage.name,
                attempt_number=attempt_number,
            )
        if stage.gate.type == "constitution_echo_verified":
            if result.mode != SignalMode.COGNITION:
                return GateOutcome.FAIL, "constitution_echo_requires_cognition_result"
            return GateOutcome.PASS, None
        if stage.gate.type == "constitutional_boundary_clean":
            return self._evaluate_constitutional_boundary_gate(stage, result)
        if stage.gate.type not in _RUNNER_GATE_TYPES:
            return (
                GateOutcome.FAIL,
                f"gate {stage.gate.type!r} is not implemented in the workflow runner",
            )
        return GateOutcome.FAIL, f"gate {stage.gate.type!r} failed closed"

    async def _evaluate_red_team_clear_gate(
        self,
        gate: Gate,
        result: Any,
        *,
        run: WorkflowRun,
        stage: Stage,
        attempt_number: int,
    ) -> tuple[GateOutcome, Optional[str]]:
        if result.mode not in {SignalMode.ACTION, SignalMode.COGNITION}:
            return GateOutcome.FAIL, "red_team_clear_requires_action_or_cognition_result"
        if self.red_team_attestation_resolver is None:
            return GateOutcome.FAIL, "red_team_clear_no_attestation_resolver"
        try:
            prompt_pack = await self._resolve_red_team_prompt_pack(stage)
        except WorkflowRunnerError as exc:
            return GateOutcome.FAIL, f"red_team_prompt_pack_error:{exc}"

        canary = _red_team_canary(run, stage, attempt_number)
        try:
            stage_output = _red_team_stage_output(result)
        except (TypeError, ValueError) as exc:
            return GateOutcome.FAIL, f"red_team_unserializable_output:{exc}"
        reviewer_results: list[tuple[str, Mapping[str, Any]]] = []
        for reviewer in _red_team_reviewer_pool(gate):
            review_outcome, review_reason, marker = await self._dispatch_red_team_reviewer(
                gate,
                run=run,
                stage=stage,
                attempt_number=attempt_number,
                reviewer=reviewer,
                canary=canary,
                stage_output=stage_output,
                prompt_pack=prompt_pack,
                operator_token_limit=self.red_team_max_total_tokens,
                operator_cost_limit=self.red_team_max_total_cost_usd,
            )
            if review_outcome != GateOutcome.PASS:
                return review_outcome, review_reason
            reviewer_results.append((reviewer, marker))
            budget = _red_team_budget_reason(
                gate,
                reviewer_results,
                operator_token_limit=self.red_team_max_total_tokens,
                operator_cost_limit=self.red_team_max_total_cost_usd,
            )
            if budget is not None:
                return GateOutcome.FAIL, budget

        return GateOutcome.PASS, None

    async def _dispatch_red_team_reviewer(
        self,
        gate: Gate,
        *,
        run: WorkflowRun,
        stage: Stage,
        attempt_number: int,
        reviewer: str,
        canary: str,
        stage_output: str,
        prompt_pack: Mapping[str, Any],
        operator_token_limit: Optional[int],
        operator_cost_limit: Optional[float],
    ) -> tuple[GateOutcome, Optional[str], Mapping[str, Any]]:
        source = _red_team_reviewer_source(reviewer)
        reviewer_id = _red_team_reviewer_identity(reviewer)
        signal = Signal(
            source=source,
            kind="workflow.red_team_review",
            mode=SignalMode.COGNITION,
            payload={
                "pr_diff": _fence_untrusted(stage_output),
                "canary": canary,
                "rubric": _red_team_rubric(
                    gate,
                    prompt_pack,
                    operator_token_limit=operator_token_limit,
                    operator_cost_limit=operator_cost_limit,
                ),
                "reviewer": reviewer_id,
                "workflow_run_id": run.run_id,
                "workflow_stage_name": stage.name,
                "workflow_attempt_number": attempt_number,
            },
            target_agent=run.started_by_did,
            visibility=Visibility.INTERNAL,
            session_id=run.run_id,
            urgency=Urgency.NORMAL,
            dedupe_key=(
                f"{run.run_id}:{stage.name}:{attempt_number}:"
                f"red_team:{reviewer_id}"
            ),
            causation_chain=[
                CausationFrame(
                    agent_id=run.started_by_did,
                    source=f"workflow.{run.workflow_name}.{stage.name}",
                    signal_id=run.run_id,
                    turn_id=None,
                    depth=0,
                    emitted_at=datetime.now(timezone.utc),
                ),
                CausationFrame(
                    agent_id=run.started_by_did,
                    source=f"workflow.red_team_clear.{stage.name}",
                    signal_id=run.run_id,
                    turn_id=None,
                    depth=1,
                    emitted_at=datetime.now(timezone.utc),
                ),
            ],
        )
        review_result = await self.dispatcher.dispatch_signal(signal)
        if review_result.status != Status.OK:
            return (
                GateOutcome.FAIL,
                "red_team_reviewer_failed:"
                f"{source}:{review_result.error or review_result.status.value}",
                {},
            )
        marker = _red_team_review_marker(review_result.artifact)
        if marker is None:
            return GateOutcome.FAIL, f"red_team_malformed_response:{source}", {}

        try:
            attestation = self.red_team_attestation_resolver(reviewer_id)
            if inspect.isawaitable(attestation):
                attestation = await attestation
        except Exception as exc:
            return (
                GateOutcome.FAIL,
                f"red_team_attestation_error:{reviewer_id}:{exc}",
                {},
            )
        outcome, reason = _evaluate_red_team_marker(
            reviewer_id,
            source,
            marker,
            canary=canary,
            attestation=attestation,
        )
        if outcome != GateOutcome.PASS:
            return outcome, reason, {}
        return GateOutcome.PASS, None, marker

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

    async def _evaluate_script_gate(
        self, gate: Gate, result: Any
    ) -> tuple[GateOutcome, Optional[str]]:
        if result.mode not in {SignalMode.ACTION, SignalMode.COGNITION}:
            return GateOutcome.FAIL, "script_requires_action_or_cognition_result"
        if self.script_gate_provider is None:
            return GateOutcome.FAIL, "script_no_resolver"
        try:
            marker = self.script_gate_provider(gate, result)
            if inspect.isawaitable(marker):
                marker = await marker
        except Exception as exc:  # pragma: no cover - defensive boundary
            return GateOutcome.FAIL, f"script_error:{exc}"
        outcome, reason = _evaluate_script_gate_marker(gate, marker)
        if outcome != GateOutcome.PASS:
            return outcome, reason
        return await self._verify_script_gate_artifact(gate)

    async def _verify_script_gate_artifact(
        self, gate: Gate
    ) -> tuple[GateOutcome, Optional[str]]:
        if self.script_artifact_resolver is None:
            return GateOutcome.FAIL, "script_no_script_resolver"
        try:
            artifact = self.script_artifact_resolver(gate)
            if inspect.isawaitable(artifact):
                artifact = await artifact
        except Exception as exc:  # pragma: no cover - defensive boundary
            return GateOutcome.FAIL, f"script_resolver_error:{exc}"
        return _verify_script_gate_artifact(
            gate,
            artifact,
            public_key_resolver=self.public_key_resolver,
            verification_methods_resolver=self.verification_methods_resolver,
        )

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


def _evaluate_consent_collect_gate(
    gate: Gate, result: Any
) -> tuple[GateOutcome, Optional[str]]:
    if result.mode != SignalMode.ACTION:
        return GateOutcome.FAIL, "consent_collect_requires_action_result"
    marker = getattr(result, "action_result", None)
    return _evaluate_consent_collect_marker(gate, marker)


def _red_team_reviewer_pool(gate: Gate) -> tuple[str, ...]:
    return tuple(gate.params["reviewer_pool"])


def _red_team_reviewer_source(reviewer: str) -> str:
    if reviewer.startswith("review."):
        return reviewer
    return f"review.{reviewer}"


def _red_team_reviewer_identity(reviewer: str) -> str:
    if reviewer.startswith("review."):
        return reviewer.removeprefix("review.")
    return reviewer


def _validate_red_team_prompt_pack(stage: Stage, prompt_pack: Any) -> None:
    if not isinstance(prompt_pack, Mapping):
        raise WorkflowRunnerError(
            f"stage {stage.name!r} red_team_clear prompt pack resolver "
            "must return a mapping"
        )
    version = prompt_pack.get("version")
    if not isinstance(version, str) or not version.strip():
        raise WorkflowRunnerError(
            f"stage {stage.name!r} red_team_clear prompt pack is missing "
            "a version"
        )
    constraint = stage.gate.params["prompt_pack_constraint"]
    try:
        if Version(version) not in SpecifierSet(constraint):
            raise WorkflowRunnerError(
                f"stage {stage.name!r} red_team_clear prompt pack version "
                f"{version!r} does not satisfy {constraint!r}"
            )
    except (InvalidSpecifier, InvalidVersion) as exc:
        raise WorkflowRunnerError(
            f"stage {stage.name!r} red_team_clear prompt pack constraint "
            f"or version is invalid: {exc}"
        ) from exc
    name = prompt_pack.get("name")
    if not isinstance(name, str) or not name.strip():
        raise WorkflowRunnerError(
            f"stage {stage.name!r} red_team_clear prompt pack is missing name"
        )
    prompt_hash = prompt_pack.get("prompt_hash")
    if not isinstance(prompt_hash, str) or not _HASH_RE.match(prompt_hash):
        raise WorkflowRunnerError(
            f"stage {stage.name!r} red_team_clear prompt pack is missing "
            "a prompt_hash"
        )


def _validate_red_team_operator_tokens(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkflowRunnerError(
            "red_team operator max_total_tokens must be a positive integer"
        )
    return value


def _validate_red_team_operator_cost(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise WorkflowRunnerError(
            "red_team operator max_total_cost_usd must be a positive finite number"
        )
    return float(value)


def _validate_red_team_budget_policy(
    stage: Stage,
    *,
    operator_token_limit: Optional[int],
    operator_cost_limit: Optional[float],
) -> None:
    definition_tokens = stage.gate.params.get("max_total_tokens")
    if (
        operator_token_limit is not None
        and definition_tokens is not None
        and definition_tokens > operator_token_limit
    ):
        raise WorkflowRunnerError(
            f"stage {stage.name!r} red_team_clear max_total_tokens "
            f"{definition_tokens} exceeds operator ceiling {operator_token_limit}"
        )

    definition_cost = stage.gate.params.get("max_total_cost_usd")
    if (
        operator_cost_limit is not None
        and definition_cost is not None
        and float(definition_cost) > operator_cost_limit
    ):
        raise WorkflowRunnerError(
            f"stage {stage.name!r} red_team_clear max_total_cost_usd "
            f"{definition_cost} exceeds operator ceiling {operator_cost_limit:g}"
        )


def _effective_red_team_token_limit(
    gate: Gate, operator_token_limit: Optional[int]
) -> Optional[int]:
    definition_limit = gate.params.get("max_total_tokens")
    if definition_limit is None:
        return operator_token_limit
    if operator_token_limit is None:
        return definition_limit
    return min(operator_token_limit, definition_limit)


def _effective_red_team_cost_limit(
    gate: Gate, operator_cost_limit: Optional[float]
) -> Optional[float]:
    definition_limit = gate.params.get("max_total_cost_usd")
    if definition_limit is None:
        return operator_cost_limit
    if operator_cost_limit is None:
        return float(definition_limit)
    return min(operator_cost_limit, float(definition_limit))


def _red_team_attested_family(
    stage: Stage, reviewer: str, attestation: Any
) -> str:
    if not isinstance(attestation, Mapping):
        raise WorkflowRunnerError(
            f"stage {stage.name!r} red_team_clear reviewer {reviewer!r} "
            "attestation resolver must return a mapping"
        )
    family = attestation.get("model_family")
    if not isinstance(family, str) or not family.strip():
        raise WorkflowRunnerError(
            f"stage {stage.name!r} red_team_clear reviewer {reviewer!r} "
            "attestation is missing model_family"
        )
    constitution_hash = attestation.get("constitution_hash")
    if not isinstance(constitution_hash, str) or not _HASH_RE.match(
        constitution_hash
    ):
        raise WorkflowRunnerError(
            f"stage {stage.name!r} red_team_clear reviewer {reviewer!r} "
            "attestation is missing a constitution_hash"
        )
    return family.strip()


def _red_team_canary(
    run: WorkflowRun, stage: Stage, attempt_number: int
) -> str:
    seed = "|".join(
        (
            run.run_id,
            stage.name,
            str(attempt_number),
            secrets.token_hex(16),
        )
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _red_team_stage_output(result: Any) -> str:
    if result.mode == SignalMode.ACTION:
        payload = getattr(result, "action_result", None)
    else:
        payload = getattr(result, "artifact", None)
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fence_untrusted(value: str) -> str:
    token = secrets.token_hex(16)
    begin = f"<<<UNTRUSTED_BEGIN:{token}"
    end = f"UNTRUSTED_END:{token}>>>"
    while begin in value or end in value:
        token = secrets.token_hex(16)
        begin = f"<<<UNTRUSTED_BEGIN:{token}"
        end = f"UNTRUSTED_END:{token}>>>"
    return f"{begin}\n{value}\n{end}"


def _red_team_rubric(
    gate: Gate,
    prompt_pack: Mapping[str, Any],
    *,
    operator_token_limit: Optional[int] = None,
    operator_cost_limit: Optional[float] = None,
) -> dict[str, Any]:
    return {
        "blockers": gate.params.get("blockers", "zero"),
        "prompt_pack_constraint": gate.params["prompt_pack_constraint"],
        "prompt_pack_name": prompt_pack["name"],
        "prompt_pack_version": prompt_pack["version"],
        "prompt_hash": prompt_pack["prompt_hash"],
        "max_total_cost_usd": _effective_red_team_cost_limit(
            gate, operator_cost_limit
        ),
        "max_total_tokens": _effective_red_team_token_limit(
            gate, operator_token_limit
        ),
    }


def _red_team_review_marker(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    return decoded


def _evaluate_red_team_marker(
    reviewer: str,
    source: str,
    marker: Mapping[str, Any],
    *,
    canary: str,
    attestation: Mapping[str, Any],
) -> tuple[GateOutcome, Optional[str]]:
    observed_canary = _first_marker_string(marker, ("canary", "echoed_canary"))
    if observed_canary != canary:
        return GateOutcome.FAIL, f"red_team_missing_canary:{source}"

    observed_reviewer = _first_marker_string(
        marker, ("reviewer", "reviewer_did", "did")
    )
    if observed_reviewer is not None and observed_reviewer != reviewer:
        return GateOutcome.FAIL, f"red_team_reviewer_mismatch:{source}"

    family = str(attestation["model_family"]).strip()
    observed_family = _first_marker_string(marker, ("model_family", "family"))
    observed_model = _first_marker_string(marker, ("model",))
    if observed_family is not None:
        if observed_family != family:
            return GateOutcome.FAIL, f"red_team_model_family_mismatch:{source}"
    if observed_model is None or not observed_model.startswith(family):
        return GateOutcome.FAIL, f"red_team_model_family_mismatch:{source}"

    blockers = marker.get("blockers")
    if not isinstance(blockers, list):
        return GateOutcome.FAIL, f"red_team_malformed_blockers:{source}"
    if blockers:
        return GateOutcome.FAIL, f"red_team_blockers:{source}:{len(blockers)}"
    return GateOutcome.PASS, None


def _red_team_budget_reason(
    gate: Gate,
    reviewer_results: list[tuple[str, Mapping[str, Any]]],
    *,
    operator_token_limit: Optional[int] = None,
    operator_cost_limit: Optional[float] = None,
) -> Optional[str]:
    token_limit = _effective_red_team_token_limit(gate, operator_token_limit)
    cost_limit = _effective_red_team_cost_limit(gate, operator_cost_limit)
    total_tokens = 0
    total_cost = 0.0
    for _, marker in reviewer_results:
        tokens = marker.get("tokens")
        cost = marker.get("cost_usd")
        if token_limit is not None and tokens is None:
            return "red_team_malformed_cost"
        if cost_limit is not None and cost is None:
            return "red_team_malformed_cost"
        if tokens is not None:
            if (
                isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens < 0
            ):
                return "red_team_malformed_cost"
            total_tokens += tokens
        if cost is not None:
            if (
                isinstance(cost, bool)
                or not isinstance(cost, (int, float))
            ):
                return "red_team_malformed_cost"
            cost_value = float(cost)
            if not math.isfinite(cost_value) or cost_value < 0:
                return "red_team_malformed_cost"
            total_cost += cost_value
    if token_limit is not None and total_tokens > token_limit:
        return "red_team_budget_exhausted:tokens"
    if cost_limit is not None and total_cost > cost_limit:
        return "red_team_budget_exhausted:cost"
    return None


def _evaluate_consent_collect_marker(
    gate: Gate, marker: Any
) -> tuple[GateOutcome, Optional[str]]:
    if not isinstance(marker, dict):
        return GateOutcome.FAIL, "consent_collect_missing_result"

    expected_scope = gate.params["scope"]
    observed_scope = _first_marker_string(
        marker, ("scope", "consent_scope", "requested_scope")
    )
    if observed_scope is None:
        return GateOutcome.FAIL, "consent_collect_missing_scope"
    if observed_scope != expected_scope:
        return GateOutcome.FAIL, f"consent_collect_scope_mismatch:{expected_scope}"

    if _consent_marker_pending(marker):
        return GateOutcome.PENDING, _consent_marker_pending_reason(marker)
    if _consent_marker_denied(marker):
        reason = _consent_marker_reason(marker)
        return GateOutcome.FAIL, f"consent_collect_denied:{reason}"
    if _consent_marker_approved(marker):
        return GateOutcome.PASS, None
    return GateOutcome.FAIL, "consent_collect_missing_approval"


def _evaluate_council_approve_gate(
    gate: Gate, result: Any
) -> tuple[GateOutcome, Optional[str]]:
    if result.mode == SignalMode.ACTION:
        marker = getattr(result, "action_result", None)
    elif result.mode == SignalMode.ARTIFACT:
        marker = getattr(result, "artifact", None)
    else:
        return GateOutcome.FAIL, "council_approve_requires_action_or_artifact_result"
    return _evaluate_council_approve_marker(gate, marker)


def _evaluate_council_approve_marker(
    gate: Gate, marker: Any
) -> tuple[GateOutcome, Optional[str]]:
    if not isinstance(marker, dict):
        return GateOutcome.FAIL, "council_approve_missing_result"

    if _has_explicit_bad_status(marker) or _council_marker_denied(marker):
        reason = _council_marker_reason(marker)
        return GateOutcome.FAIL, f"council_approve_denied:{reason}"
    if _council_marker_pending(marker):
        return GateOutcome.PENDING, _council_marker_pending_reason(marker)

    quorum = gate.params["quorum"]
    approved_dids = _council_approved_dids(marker)
    if not approved_dids:
        return GateOutcome.FAIL, "council_approve_missing_approvals"
    if len(approved_dids) < quorum:
        return (
            GateOutcome.FAIL,
            f"council_approve_quorum_not_met:{len(approved_dids)}/{quorum}",
        )
    return GateOutcome.PASS, None


def _evaluate_script_gate_marker(
    gate: Gate, marker: Any
) -> tuple[GateOutcome, Optional[str]]:
    if not isinstance(marker, dict):
        return GateOutcome.FAIL, "script_missing_result"
    mismatch = _script_marker_contract_mismatch(gate, marker)
    if mismatch is not None:
        return GateOutcome.FAIL, mismatch
    if _has_explicit_bad_status(marker):
        return GateOutcome.FAIL, f"script_failed:{_result_marker_reason(marker)}"
    if _result_marker_passed(gate, marker):
        return GateOutcome.PASS, None
    return GateOutcome.FAIL, f"script_failed:{_result_marker_reason(marker)}"


def _verify_script_gate_artifact(
    gate: Gate,
    artifact: Any,
    *,
    public_key_resolver: PublicKeyResolver,
    verification_methods_resolver: Optional[VerificationMethodsResolver],
) -> tuple[GateOutcome, Optional[str]]:
    if artifact is None:
        return GateOutcome.FAIL, "script_missing_artifact"

    expected_language = gate.params["language"]
    observed_language = _script_artifact_string(artifact, "language")
    if observed_language != expected_language:
        return GateOutcome.FAIL, f"script_language_mismatch:{expected_language}"

    expected_hash = gate.params["src_hash"]
    observed_hash = _script_artifact_content_address(artifact)
    if observed_hash != expected_hash:
        return GateOutcome.FAIL, f"script_src_hash_unresolved:{expected_hash}"

    expected_signature = gate.params["signature"]
    observed_signature = _script_artifact_string(artifact, "signature")
    if observed_signature != expected_signature:
        return GateOutcome.FAIL, "script_signature_mismatch"

    expected_did = gate.params["signing_did"]
    observed_did = _script_artifact_string(artifact, "signed_by")
    if observed_did != expected_did:
        return GateOutcome.FAIL, f"script_signing_did_mismatch:{expected_did}"

    payload = _script_signature_payload(artifact)
    if payload is None or not _verify_script_artifact_signature(
        did=expected_did,
        signature=expected_signature,
        payload=payload,
        public_key_resolver=public_key_resolver,
        verification_methods_resolver=verification_methods_resolver,
    ):
        return GateOutcome.FAIL, "script_invalid_signature"
    return GateOutcome.PASS, None


def _script_artifact_string(artifact: Any, field: str) -> Optional[str]:
    if isinstance(artifact, Mapping):
        value = artifact.get(field)
    else:
        value = getattr(artifact, field, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _script_artifact_content_address(artifact: Any) -> Optional[str]:
    content_hash = _script_artifact_content_hash(artifact)
    if content_hash is None:
        return None
    return f"sha256:{content_hash}"


def _script_signature_payload(artifact: Any) -> Optional[bytes]:
    content_hash = _script_artifact_content_hash(artifact)
    if content_hash is None:
        return None
    return hashlib.sha256(content_hash.encode()).digest()


def _script_artifact_content_hash(artifact: Any) -> Optional[str]:
    name = _script_artifact_raw_string(artifact, "name")
    language = _script_artifact_raw_string(artifact, "language")
    content = _script_artifact_raw_string(artifact, "content")
    purpose = _script_artifact_raw_string(artifact, "purpose")
    if None in {name, language, content, purpose}:
        return None
    canonical = f"{name}|{language}|{content}|{purpose}"
    return hashlib.sha256(canonical.encode()).hexdigest()


def _script_artifact_raw_string(artifact: Any, field: str) -> Optional[str]:
    if isinstance(artifact, Mapping):
        value = artifact.get(field)
    else:
        value = getattr(artifact, field, None)
    return value if isinstance(value, str) else None


def _verify_script_artifact_signature(
    *,
    did: str,
    signature: str,
    payload: bytes,
    public_key_resolver: PublicKeyResolver,
    verification_methods_resolver: Optional[VerificationMethodsResolver],
) -> bool:
    if signature.startswith("hybrid:"):
        try:
            decoded = base64.b64decode(signature[len("hybrid:") :]).decode()
            signatures = json.loads(decoded)
        except Exception:
            return False
        if not isinstance(signatures, list):
            return False
        return _verify_signature_collected_hybrid(
            did=did,
            signatures=signatures,
            payload=payload,
            verification_methods_resolver=verification_methods_resolver,
        )
    if not signature.startswith("ecdsa:"):
        return False
    try:
        public_key_bytes = public_key_resolver(did)
        public_key = get_suite(
            ALG_ECDSA_SECP256K1_SHA256
        ).deserialize_public_key(public_key_bytes)
        signature_bytes = base64.b64decode(signature[len("ecdsa:") :])
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


def _script_marker_contract_mismatch(
    gate: Gate, marker: dict[str, Any]
) -> Optional[str]:
    aliases = {
        "language": ("language", "script_language"),
        "src_hash": ("src_hash", "script_hash", "content_hash"),
        "signature": ("signature", "script_signature"),
        "signing_did": ("signing_did", "signed_by", "signer_did"),
        "sandbox": ("sandbox", "executor", "execution_sandbox"),
    }
    for param_key, marker_keys in aliases.items():
        observed = _first_marker_string(marker, marker_keys)
        expected = gate.params[param_key]
        if observed is None:
            return f"script_missing_{param_key}"
        if observed != expected:
            return f"script_{param_key}_mismatch:{expected}"
    return None


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


def _first_marker_string(
    value: dict[str, Any], keys: tuple[str, ...]
) -> Optional[str]:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _consent_marker_approved(value: dict[str, Any]) -> bool:
    if any(value.get(key) is True for key in ("approved", "consent", "accepted")):
        return True
    decision = _first_marker_string(value, ("decision", "status", "state", "outcome"))
    return decision is not None and decision.lower() in {
        "allow",
        "allowed",
        "approve",
        "approved",
        "accept",
        "accepted",
    }


def _consent_marker_denied(value: dict[str, Any]) -> bool:
    if any(value.get(key) is False for key in ("approved", "consent", "accepted")):
        return True
    if _has_explicit_bad_status(value):
        return True
    decision = _first_marker_string(value, ("decision", "status", "state", "outcome"))
    return decision is not None and decision.lower() in {
        "deny",
        "denied",
        "reject",
        "rejected",
        "decline",
        "declined",
        "cancel",
        "cancelled",
        "canceled",
    }


def _consent_marker_pending(value: dict[str, Any]) -> bool:
    decision = _first_marker_string(value, ("decision", "status", "state", "outcome"))
    return decision is not None and decision.lower() in {
        "pending",
        "waiting",
        "queued",
        "needs_approval",
        "needs_review",
    }


def _consent_marker_pending_reason(value: dict[str, Any]) -> str:
    request_id = _first_marker_string(
        value,
        (
            "approval_id",
            "approval_request_id",
            "request_id",
            "consent_request_id",
        ),
    )
    if request_id is None:
        return "consent_collect_pending"
    return f"consent_collect_pending:{request_id}"


def _consent_marker_reason(value: dict[str, Any]) -> str:
    reason = _first_marker_string(value, ("reason", "message", "error", "summary"))
    return reason if reason is not None else _result_marker_reason(value)


def _council_marker_denied(value: dict[str, Any]) -> bool:
    if any(value.get(key) is False for key in ("approved", "accepted", "passed")):
        return True
    decision = _first_marker_string(value, ("decision", "status", "state", "outcome"))
    return decision is not None and decision.lower() in {
        "deny",
        "denied",
        "reject",
        "rejected",
        "decline",
        "declined",
        "veto",
        "vetoed",
        "blocked",
        "cancel",
        "cancelled",
        "canceled",
    }


def _council_marker_pending(value: dict[str, Any]) -> bool:
    decision = _first_marker_string(value, ("decision", "status", "state", "outcome"))
    return decision is not None and decision.lower() in {
        "pending",
        "waiting",
        "queued",
        "in_progress",
        "needs_approval",
        "needs_review",
    }


def _council_marker_pending_reason(value: dict[str, Any]) -> str:
    request_id = _first_marker_string(
        value,
        (
            "council_request_id",
            "approval_id",
            "approval_request_id",
            "request_id",
            "vote_request_id",
        ),
    )
    if request_id is None:
        return "council_approve_pending"
    return f"council_approve_pending:{request_id}"


def _council_marker_reason(value: dict[str, Any]) -> str:
    reason = _first_marker_string(value, ("reason", "message", "error", "summary"))
    return reason if reason is not None else _result_marker_reason(value)


def _council_approved_dids(value: dict[str, Any]) -> tuple[str, ...]:
    approved: list[str] = []
    for key in ("approved_dids", "approver_dids", "approval_dids"):
        dids = _marker_did_sequence(value.get(key), require_approval=False)
        if dids:
            approved.extend(dids)
    for key in ("approvals", "votes", "ballots"):
        dids = _marker_did_sequence(value.get(key), require_approval=True)
        if dids:
            approved.extend(dids)
    return tuple(dict.fromkeys(approved))


def _marker_did_sequence(value: Any, *, require_approval: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    dids: list[str] = []
    for item in value:
        if isinstance(item, str):
            did = item.strip()
            if _DID_RE.fullmatch(did):
                dids.append(did)
            continue
        if not isinstance(item, dict):
            continue
        did = _first_marker_string(
            item,
            ("did", "approver_did", "voter_did", "member_did", "actor_did"),
        )
        if did is None or not _DID_RE.fullmatch(did):
            continue
        if require_approval and not _vote_marker_approved(item):
            continue
        dids.append(did)
    return tuple(dids)


def _vote_marker_approved(value: dict[str, Any]) -> bool:
    if any(value.get(key) is True for key in ("approved", "approve", "accepted")):
        return True
    decision = _first_marker_string(value, ("decision", "status", "state", "vote"))
    return decision is not None and decision.lower() in {
        "approve",
        "approved",
        "accept",
        "accepted",
        "yes",
        "y",
        "aye",
    }


def _stage_link_timed_out(link: StageLink, timeout_seconds: int) -> bool:
    occurred_at = link.occurred_at
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    elapsed = (datetime.now(timezone.utc) - occurred_at).total_seconds()
    return elapsed >= timeout_seconds


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
