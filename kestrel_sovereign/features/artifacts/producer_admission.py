"""Finite, explicit native-output policy owned by the canonical artifact feature.

Plans and audit rows are descriptive. Only this live feature's confirmed user
admission can bind a WRITE operation on its existing ArtifactAuthority. Nothing
here retains a decision caller, restores an admission, or attaches HOST outputs.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from hashlib import sha256
import json
import math
from copy import deepcopy
import time
from uuid import uuid4

from kestrel_sdk.artifacts import ArtifactCommit, ArtifactFile, ArtifactManifest
from kestrel_sdk.operator import OperatorContext, ServiceRequirement, ServiceScope
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown, ResourceScope
from kestrel_sovereign.artifacts.service import ArtifactAccessDenied, ArtifactAction
from kestrel_sovereign.features.security.permissions import PermissionStore
from kestrel_sovereign.operator import ServiceDiscoveryIdentity
from kestrel_sovereign.security.host_authority import HostAuthorityError, stable_sovereign_secret
from kestrel_sovereign.security.sovereign_key import sovereign_key_fingerprint
from .producers import _TEXT_TYPES


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class NativeOutputPublicationUnknown(ResourceOutcomeUnknown):
    """Original intent/key retained; this supplies no publication receipt."""
    def __init__(self, intent):
        super().__init__("original native output publication requires reconciliation")
        self.intent = deepcopy(intent)


class NativeOutputReceiptPending(NativeOutputPublicationUnknown):
    """Known canonical receipt, unknown independent native acknowledgment.

    This carries facts only; reconciliation still requires fresh independent
    run and artifact authority. It cannot acknowledge or attach a revision.
    """
    def __init__(self, receipt, intent):
        super().__init__(intent)
        self.receipt = receipt


@dataclass(frozen=True)
class NativeOutputLimits:
    """One explicit finite window; limits cannot be inferred from stage intent."""
    window_seconds: int
    max_publications: int
    max_total_bytes: int
    max_file_bytes: int
    max_attempts: int = 1

    def __post_init__(self):
        bounds = {"window_seconds": 900, "max_publications": 64,
                  "max_total_bytes": 16 * 1024 * 1024,
                  "max_file_bytes": 1024 * 1024, "max_attempts": 16}
        if any(type(getattr(self, name)) is not int or
               not 0 < getattr(self, name) <= ceiling for name, ceiling in bounds.items()):
            raise ValueError("native output approval needs explicit finite limits")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("native output file limit exceeds its total")


@dataclass(frozen=True)
class NativeOutputEffectRestriction:
    """Restriction metadata in the current loop's clock; never native Work.

    The genuine source frame and original native publisher still own all
    admission. Copying this value cannot issue, renew or restore a grant.
    """
    check_current: object
    deadline: float
    _issuance: object = field(default=None,init=False,repr=False,compare=False)

    def __post_init__(self):
        if not callable(self.check_current) or type(self.deadline) is not float or not math.isfinite(self.deadline):
            raise TypeError("finite original output restriction required")

    @classmethod
    def _issue(cls,current,deadline,bind):
        value = cls(current,deadline)
        object.__setattr__(value,"_issuance",_EffectIssuance(value,bind))
        return value

    def require_issued(self):
        owner = self._issuance
        if (type(owner) is not _EffectIssuance or owner.value is not self
                or not owner.active or self.check_current is not owner.current
                or self.deadline != owner.deadline):
            raise ArtifactAccessDenied("original finite output restriction is unavailable")
        if owner.current() is not None:
            raise ArtifactAccessDenied("original finite output checker must return None")

    def bind_original_owner(self,execution):
        self.require_issued()
        owner = self._issuance
        if asyncio.current_task() is not owner.task or owner.bound:
            raise ArtifactAccessDenied("original finite effect owner is unavailable")
        owner.bind(execution)
        object.__setattr__(owner,"bound",True)


class _EffectIssuance:
    """Finite same-object integrity only; no native Work or content grant."""
    __slots__ = ("value","current","deadline","bind","task","active","bound")
    def __init__(self,value,bind):
        for name,item in (("value",value),("current",value.check_current),("deadline",value.deadline),
                ("bind",bind),("task",asyncio.current_task()),("active",True),("bound",False)):
            object.__setattr__(self,name,item)

    def __setattr__(self,name,value):
        raise TypeError("original output restriction is immutable")

    def __repr__(self):
        return "<original output restriction>"

    def __reduce_ex__(self,protocol):
        raise TypeError("original output restriction cannot be serialized")


@dataclass(eq=False, repr=False)
class _Admission:
    admission_id: str
    plan: dict
    plan_sha256: str
    scope: tuple
    generation: int
    bridge: object
    source_feature: object
    discovery_identity: ServiceDiscoveryIdentity
    privacy_epoch: str
    state: str = "proposed"
    key_fingerprint: str | None = None
    expires_at: object = None
    monotonic_deadline: float | None = None
    activation: tuple | None = None
    reservations: dict = field(default_factory=dict)
    retirements: list = field(default_factory=list)
    pending_effect_frames: int = 0
    cleanup_confirmed: bool = False
    cleanup_unknown: bool = False


class NativeArtifactAdmissions:
    """Feature-owned live policy, not a persistent grant store or executor."""
    api_version = 1
    _max_plans = 128

    def __init__(self, feature):
        self.feature = feature
        self._records = {}
        self._by_run = {}
        self._producers = {}

    def retire(self):
        for record in self._records.values():
            self._retire_record(record,"retired")

    def _retire_record(self,record,state):
        record.state = state
        # These exact original handles only schedule their existing Stop in
        # their captured native-owner context. Never execute a user callback
        # or acquire native/data authority in this revocation caller.
        for owner in tuple(record.retirements):
            owner.request_output_retirement()

    def _cleanup_state(self,record):
        if (record.retirements or record.pending_effect_frames or any(
                producer._record is record for producer in self._producers.values())):
            return "pending"
        if record.cleanup_unknown:
            return "unknown"
        if record.cleanup_confirmed:
            return "joined"
        return "not_started" if record.activation is None else "unknown"

    def _epoch(self):
        epoch = self.feature.agent._get_privacy_policy_epoch()
        if type(epoch) is not str or not epoch:
            raise ArtifactAccessDenied("native output privacy epoch is unavailable")
        return epoch

    def _live(self, record):
        self.feature._require_live(None, record.generation)
        if self._records.get(record.admission_id) is not record:
            raise ArtifactAccessDenied("native output admission is unavailable")
        if record.privacy_epoch != self._epoch():
            self._retire_record(record,"retired")
        if record.key_fingerprint is not None:
            try:
                matches = sovereign_key_fingerprint(stable_sovereign_secret("native output producer").decode()) == record.key_fingerprint
            except HostAuthorityError:
                matches = False
            if not matches:
                self._retire_record(record,"retired")
        if record.monotonic_deadline is not None and (
            time.monotonic() >= record.monotonic_deadline
            or self.feature._service.authority.clock() >= record.expires_at
        ):
            self._retire_record(record,"expired")
        if record.state != "approved":
            raise ArtifactAccessDenied("native output admission is unavailable")

    @asynccontextmanager
    async def _source(self, feature_name, run_id, *, expected=None, foreground=False, request=None):
        agent = self.feature.agent
        source = agent.features.get(feature_name)
        bridge = getattr(source, "native_bridge", None)
        if (source is None or source.agent is not agent
                or getattr(bridge, "artifact_producer_bridge_api_version", None) != 1
                or bridge.feature is not source
                or bridge.backend is not self.feature._service.repository.db.backend
                or getattr(agent, "host_service_identity_api_version", None) != 1):
            raise ArtifactAccessDenied("native output source is unavailable")
        # Obtain the genuine operation ourselves. A supplied ServiceOperation,
        # copied discovery identity or no-op callback is never accepted here.
        requirement = ServiceRequirement("workflows.runs", "1.2.0", ServiceScope.HOST)
        async with agent.host_service_operation(requirement) as original:
            original.require_issued()
            identity = original.discovery_identity
            if type(identity) is not ServiceDiscoveryIdentity:
                raise ArtifactAccessDenied("native output host discovery is unavailable")
            if expected is not None and (source is not expected.source_feature
                    or bridge is not expected.bridge
                    or not identity.matches(expected.discovery_identity)):
                raise ArtifactAccessDenied("native output registration changed")
            if foreground:
                plane = getattr(original.service, "native_runs", None)
                if getattr(plane, "producer_plan_bridge_api_version", None) != 1:
                    raise ArtifactAccessDenied("original native run policy bridge is unavailable")
                frame = plane.producer_plan_operation(source, run_id,
                    check_current=original.require_issued, request=request)
            else:
                frame = bridge.prepared_output_operation(run_id)
            async with frame as native:
                def check():
                    original.require_issued()
                    native.check_current()
                    if agent.features.get(feature_name) is not source or source.native_bridge is not bridge:
                        raise ArtifactAccessDenied("native output source changed")
                check()
                yield native, identity, source, bridge, check
                check()

    def _view(self, record):
        # No handle, caller, discovery identity, backend or live checker leaves
        # the host. A returned ID still requires fresh policy for every action.
        state = record.state
        if state == "approved":
            if record.privacy_epoch != self._epoch() or record.generation != self.feature._generation:
                state = "retired"
            elif (time.monotonic() >= record.monotonic_deadline
                    or self.feature._service.authority.clock() >= record.expires_at):
                state = "expired"
            else:
                try:
                    fingerprint = sovereign_key_fingerprint(stable_sovereign_secret("native output status").decode())
                    if fingerprint != record.key_fingerprint:
                        state = "retired"
                except HostAuthorityError:
                    state = "retired"
        return {"admission_id": record.admission_id, "plan": deepcopy(record.plan),
                "plan_sha256": record.plan_sha256, "state": state,
                "expires_at": None if record.expires_at is None else record.expires_at.isoformat(),
                "activated": record.activation is not None,
                "reserved_publications": len(record.reservations),
                "reserved_bytes": sum(value[1] for value in record.reservations.values())}

    async def propose(self, feature_name, run_id, limits, *, request=None):
        if type(limits) is not NativeOutputLimits:
            raise TypeError("typed finite native output limits required")
        if len(self._records) >= self._max_plans:
            raise ArtifactAccessDenied("native output proposal bound reached")
        async with self._source(feature_name, run_id, foreground=True, request=request) as (native, identity, source, bridge, check):
            async with self.feature.artifact_operation(request) as original:
                return self._propose_current(feature_name, limits, native, identity, source, bridge, check, original)

    def _propose_current(self, feature_name, limits, native, identity, source, bridge, check, original):
        original.check_current(); check()
        self._check_source_scope(native, original)
        if type(limits) is not NativeOutputLimits or not native.prepared:
            raise ArtifactAccessDenied("native output preparation is unavailable")
        # Recheck after awaited policy acquisition, immediately before insert.
        if len(self._records) >= self._max_plans:
            raise ArtifactAccessDenied("native output proposal bound reached")
        plan = deepcopy({"feature_name": feature_name, **native.plan, "limits": asdict(limits)})
        record = _Admission(uuid4().hex, plan, _digest(plan),
            (original.context.tenant_id, original.context.principal_id, original.context.acting_agent_id),
            self.feature._generation, bridge, source, identity, self._epoch())
        self._records[record.admission_id] = record
        return self._view(record)

    async def decide(self, admission_id, plan_sha256, *, allow, request=None):
        if type(allow) is not bool:
            raise ValueError("explicit native output decision required")
        record = self._records.get(admission_id)
        if record is None or plan_sha256 != record.plan_sha256:
            raise ResourceConflict("native output plan changed or is unavailable")
        async with self._source(record.plan["feature_name"], record.plan["run_id"], expected=record,
                                foreground=True, request=request) as (native, _, _, _, check):
            async with self.feature.artifact_operation(request) as original:
                return await self._decide_current(record, native, check, original, allow)

    async def _decide_current(self, record, native, check, original, allow):
        self._check_source_scope(native, original)
        if type(allow) is not bool:
            raise ValueError("explicit native output decision required")
        async with self.feature._service.authority.operation(original.context, ArtifactAction.WRITE,
                        self.feature._service.repository.storage_id) as write_check:
            # Schema/envelope admission precedes native ACTION dispatch.
            # Lazy DDL inside a running stage would invalidate that same
            # backend's original signal-journal relation seal.
            write_check()
            await self.feature._service.repository.ensure_initialized()
            write_check()
            if record.scope != (original.context.tenant_id, original.context.principal_id, original.context.acting_agent_id):
                raise ArtifactAccessDenied("native output decision owner changed")
            original.check_current(); check()
            if record.state != "proposed":
                return self._view(record)  # Unknown audit is never re-issued.
            if not native.prepared or record.privacy_epoch != self._epoch():
                raise ArtifactAccessDenied("native output preparation changed")
            security = self.feature.agent.features.get("SecurityFeature")
            audit = getattr(security, "permission_store", None)
            if not isinstance(audit, PermissionStore) or security.agent is not self.feature.agent:
                raise ArtifactAccessDenied("native output approval audit is unavailable")
            # Mark pending BEFORE the first awaited audit. An ambiguous
            # audit response must never be interpreted as a fresh grant.
            record.state = "approval_pending"
            seconds = record.plan["limits"]["window_seconds"]
            record.expires_at = self.feature._service.authority.clock() + timedelta(seconds=seconds)
            record.monotonic_deadline = time.monotonic() + seconds
            fingerprint = sovereign_key_fingerprint(stable_sovereign_secret("native output approval").decode())
            await audit.log_decision("ArtifactsFeature", "producer_admission",
                "artifact_producer_admission", "user_approved" if allow else "denied",
                user_choice="finite_plan", args_summary=json.dumps({
                    "decision_id": record.admission_id, "plan_sha256": record.plan_sha256,
                    "issuer_fingerprint": fingerprint, "limits": record.plan["limits"],
                    "expires_at": record.expires_at.isoformat(),
                    "feature_generation": record.generation, "privacy_epoch_sha256": _digest(record.privacy_epoch)}))
            original.check_current(); check()
            if record.privacy_epoch != self._epoch():
                raise ArtifactAccessDenied("native output privacy policy changed")
            if not allow:
                record.state = "denied"
            else:
                previous = self._by_run.get((record.plan["feature_name"], record.plan["run_id"]))
                if previous is not None:
                    self._retire_record(previous,"retired")
                record.key_fingerprint = fingerprint
                record.state = "approved"
                self._by_run[(record.plan["feature_name"], record.plan["run_id"])] = record
                self._live(record)
            return self._view(record)

    async def revoke(self, admission_id, *, request=None):
        async with self.feature.artifact_operation(request) as original:
            record = self._records.get(admission_id)
            if record is None:
                raise ArtifactAccessDenied("native output admission is unavailable")
            self._revoke_current(record, original)
        async with self.feature._gate:
            pass
        return self._view(record)

    def _check_owner(self, record, original):
        original.check_current()
        if (original.service is not self.feature._service or record.scope !=
                (original.context.tenant_id, original.context.principal_id, original.context.acting_agent_id)):
            raise ArtifactAccessDenied("native output decision owner changed")

    def _check_source_scope(self, native, original):
        original.check_current(); native.check_current()
        expected = ResourceScope(tenant_id=original.context.tenant_id,
            principal_id=original.context.principal_id, agent_id=original.context.acting_agent_id)
        if type(native.operator_scope) is not ResourceScope or native.operator_scope != expected:
            raise ArtifactAccessDenied("original native and artifact output scopes differ")

    def _revoke_current(self, record, original):
        self._check_owner(record, original)
        self._retire_record(record,"revoked")
        value = self._view(record)
        value["publication_cleanup"] = self._cleanup_state(record)
        return value

    @asynccontextmanager
    async def start_operation(self, feature_name, run_id):
        record = self._by_run.get((feature_name, run_id))
        if record is None:
            raise ArtifactAccessDenied("native output admission is unavailable")
        self._live(record)
        async with self._source(feature_name, run_id, expected=record) as (native, _, _, _, source_check):
            def check():
                source_check(); self._live(record)
            check()
            yield record.admission_id, check
            check()

    async def activate(self, feature_name, run_id):
        """Only the original applied native RESUME receipt can activate effects."""
        record = self._by_run.get((feature_name, run_id))
        if record is None:
            raise ArtifactAccessDenied("native output admission is unavailable")
        async with self.start_operation(feature_name, run_id) as (admission_id, check):
            activation = await record.bridge.accepted_output_activation(run_id, admission_id)
            check()
            if record.activation is not None and record.activation != activation:
                raise ArtifactAccessDenied("native output activation changed")
            record.activation = activation

    async def require_effect(self, feature_name, run_id):
        record = self._by_run.get((feature_name, run_id))
        if record is None:
            raise ArtifactAccessDenied("native output admission is unavailable")
        async with self.start_operation(feature_name, run_id) as (_, check):
            if record.activation is None or await record.bridge.output_activation(run_id) != record.activation:
                raise ArtifactAccessDenied("original native output Start is unavailable")
            check()

    @asynccontextmanager
    async def effect_operation(self, feature_name, run_id):
        record = self._by_run.get((feature_name, run_id))
        if record is None:
            raise ArtifactAccessDenied("native output admission is unavailable")
        async with self.start_operation(feature_name, run_id) as (_, source_check):
            await self.require_effect(feature_name, run_id)
            activation = record.activation
            checking = False
            def current():
                nonlocal checking
                # Discovery independently requires the same original native
                # Work, which calls this restriction again. All reentries still
                # validate live policy; only the synchronous discovery walk is
                # bounded. No await/task handoff occurs while checking is true.
                self._live(record)
                if record.activation is not activation or activation is None:
                    raise ArtifactAccessDenied("native output activation changed")
                if checking:
                    return
                checking = True
                try:
                    source_check()
                    record.bridge.check_effect_selection(run_id, record.plan)
                finally:
                    checking = False
            current()
            loop_now = asyncio.get_running_loop().time()
            remaining = min(record.monotonic_deadline - time.monotonic(),
                (record.expires_at - self.feature._service.authority.clock()).total_seconds())
            record.pending_effect_frames += 1
            record.cleanup_confirmed = False
            accepted = None
            def bind(execution):
                nonlocal accepted
                current()
                if len(record.retirements) >= 128:
                    raise ArtifactAccessDenied("original output retirement bound reached")
                def acknowledge(original):
                    if accepted is not original or not any(value is original for value in record.retirements):
                        raise ArtifactAccessDenied("original output join acknowledgment differs")
                    record.retirements = [value for value in record.retirements if value is not original]
                    record.cleanup_confirmed = not (record.retirements or record.pending_effect_frames or record.cleanup_unknown)
                accepted = record.bridge.retain_output_owner(run_id,execution,acknowledge=acknowledge)
                record.retirements.append(accepted)
                record.pending_effect_frames -= 1
            restriction = NativeOutputEffectRestriction._issue(current,loop_now + max(0.,remaining),bind)
            try:
                yield restriction
            finally:
                object.__setattr__(restriction._issuance,"active",False)
                if not restriction._issuance.bound:
                    record.pending_effect_frames -= 1
                    record.cleanup_unknown = True
            # Do not turn an original stage's completed/unknown publication
            # into a newly claimed success when its policy closes here.

    @asynccontextmanager
    async def producer_operation(self, feature_name, run_id, slot):
        record = self._by_run.get((feature_name, run_id))
        if record is None:
            raise ArtifactAccessDenied("native output admission is unavailable")
        async with self.start_operation(feature_name, run_id) as (_, source_check):
            await self.require_effect(feature_name, run_id)
            async with record.bridge.output_operation(slot) as (native, native_check):
                if (native.claim.run_id != run_id or (native.binding.stage_name, slot) not in
                        tuple(tuple(value) for value in record.plan["output_slots"])
                        or native.binding.attempt_number > record.plan["limits"]["max_attempts"]):
                    raise ArtifactAccessDenied("native output slot is outside its finite plan")
                service = self.feature._service
                now = service.authority.clock()
                context = OperatorContext(tenant_id=record.scope[0], principal_id=record.scope[1],
                    acting_agent_id=record.scope[2], granted_actions=frozenset({ArtifactAction.WRITE.value}),
                    granted_capabilities=frozenset({"artifacts"}),
                    permitted_boundary_ids=frozenset({service.repository.storage_id}),
                    correlation_id=record.admission_id, issued_at=now, expires_at=record.expires_at)
                active = True
                def current():
                    if not active:
                        raise ArtifactAccessDenied("native output operation ended")
                    source_check(); native_check(); self._live(record)
                    if record.activation is None:
                        raise ArtifactAccessDenied("native output activation ended")
                def policy(ctx, action, link):
                    current()
                    if ctx is not context or action is not ArtifactAction.WRITE or link is not None:
                        raise ArtifactAccessDenied("native output action is outside its plan")
                try:
                    with service.authority.bind_authenticated(context, privacy_holder=self.feature.agent,
                            check_current=policy, authority_gate=self.feature._gate):
                        async with service.authority.operation(context, ArtifactAction.WRITE, service.repository.storage_id):
                            async with service.restrict_current(context, current):
                                current()
                                task = asyncio.current_task()
                                if task in self._producers:
                                    raise ArtifactAccessDenied("nested native output producer unavailable")
                                producer = _NativeProducer(self, record, context, current, native, slot)
                                self._producers[task] = producer
                                try:
                                    yield producer
                                    current()
                                finally:
                                    self._producers.pop(task, None)
                finally:
                    active = False


    def require_publication_receipt(self, native, slot, receipt):
        """A copied receipt/operation DTO cannot manufacture a saved fact."""
        producer = self._producers.get(asyncio.current_task())
        if (producer is None or producer._native is not native or producer._slot != slot
                or producer._accepted_receipt is not receipt):
            raise ArtifactAccessDenied("original canonical output receipt is unavailable")
        producer._check()

    def require_publication_intent(self, native, slot, intent):
        producer = self._producers.get(asyncio.current_task())
        if (producer is None or producer._native is not native or producer._slot != slot
                or producer._accepted_intent is not intent
                or intent["publication_key"] not in producer._record.reservations):
            raise ArtifactAccessDenied("original admitted publication intent is unavailable")
        producer._check()


class _NativeProducer:
    """Only create a declared original publication; no reads, edits or links."""
    def __init__(self, owner, record, context, check, native, slot):
        self._owner, self._record, self._context, self._check = owner, record, context, check
        self._native, self._slot = native, slot
        self._accepted_receipt = None
        self._accepted_intent = None

    def _retain_unknown_current(self, error):
        from kestrel_sovereign.execution_custody import execution_terminal_error
        try:
            self._check()
        except BaseException as ended:
            combined = execution_terminal_error(error, ended,
                retain_originals=True, isolate_control_originals=True)
            raise ResourceOutcomeUnknown("native output acknowledgment is unavailable") from combined

    async def publish_text(self, content, *, title, path, media_type):
        self._check()
        if type(content) is not bytes or media_type not in _TEXT_TYPES:
            raise ValueError("bounded UTF-8 native output bytes required")
        content.decode("utf-8")
        limits = self._record.plan["limits"]
        if len(content) > limits["max_file_bytes"]:
            raise ArtifactAccessDenied("native output file limit exceeded")
        file = ArtifactFile(path=path, media_type=media_type, size_bytes=len(content), sha256=sha256(content).hexdigest())
        manifest = ArtifactManifest(files=(file,), entrypoint=path)
        intent = await self._record.bridge.record_output_intent(self._slot, manifest_sha256=manifest.digest())
        self._check()
        key = intent["publication_key"]
        request = ArtifactCommit(idempotency_key=key, title=title, manifest=manifest)
        digest = request.digest()
        reservations = self._record.reservations
        old = reservations.get(key)
        if old is not None and old != (digest, len(content)):
            raise ResourceConflict("native output publication changed")
        if old is None:
            if (len(reservations) >= limits["max_publications"] or
                    sum(value[1] for value in reservations.values()) + len(content) > limits["max_total_bytes"]):
                raise ArtifactAccessDenied("native output finite budget exhausted")
            reservations[key] = (digest, len(content))  # Unknown native ACK stays charged.
        self._accepted_intent = intent
        await self._record.bridge.mark_output_publication_pending(self._slot,intent,check_current=self._check)
        self._check()
        intent["state"] = "publication_pending"
        service = self._owner.feature._service
        try:
            receipt = await service.publication_status(self._context, key)
            if receipt is not None:
                if receipt.request_sha256 != digest:
                    raise ResourceConflict("native output publication changed")
            else:
                await service.stage(self._context, content, publication_key=key, path=path, media_type=media_type)
                receipt = await service.commit(self._context, request)
        except ResourceOutcomeUnknown as error:
            from kestrel_sovereign.execution_custody import is_execution_control_error, execution_terminal_error
            if is_execution_control_error(error):
                raise
            self._retain_unknown_current(error)
            # One bounded read of the SAME original key, never another write,
            # native ACK, stage attempt or admission. Missing/error stays debt.
            try:
                recovered = await service.publication_status(self._context,key)
                self._check()
            except BaseException as status_error:
                combined = execution_terminal_error(error,status_error,
                    retain_originals=True,isolate_control_originals=True)
                self._retain_unknown_current(combined)
                raise NativeOutputPublicationUnknown(intent) from combined
            if recovered is not None and recovered.request_sha256 == digest:
                raise NativeOutputReceiptPending(recovered,intent) from error
            raise NativeOutputPublicationUnknown(intent) from error
        try:
            self._check()
        except Exception as ended:
            # Keep the original refusal/control graph; a saved fact does not
            # grant permission to deliver its revision after currentness loss.
            raise ResourceOutcomeUnknown("native output publication acknowledgment is unavailable") from ended
        self._accepted_receipt = receipt
        try:
            await self._record.bridge.record_committed_output(self._slot, receipt, check_current=self._check)
        except Exception as error:
            from kestrel_sovereign.execution_custody import is_execution_control_error
            if is_execution_control_error(error):
                raise
            self._retain_unknown_current(error)
            raise NativeOutputReceiptPending(receipt, intent) from error
        self._check()
        return receipt, intent
