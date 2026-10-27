"""One original current visual step; its row is an intent, never an SDK input receipt.

The durable claim precedes inference. A restarted or concurrent caller can only
inspect an existing intent. Absence alone never arms inference: a fresh native
claim ACK must serialize against every original writer before planning.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

from kestrel_sdk.resources import (
    ResourceConflict,
    ResourceNotFound,
    ResourceOutcomeUnknown,
    ResourceScope,
)
from pydantic import BaseModel, ConfigDict, Field

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    require_execution_work,
)


class StepIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    schema_version: Literal[1] = 1
    scope: ResourceScope
    session_id: str
    action_id: str
    prompt: str = Field(min_length=1, max_length=16384)
    configuration: str = Field(max_length=32768)
    source_sha256: str


class PlanningRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    kind: Literal["visual-step-v1"] = "visual-step-v1"
    intent: StepIntent
    session_sha256: str
    state: Literal["planning", "finished"] = "planning"
    answer: str | None = Field(default=None, max_length=2000)

    def pending(self):
        return self.state == "planning"


@dataclass(frozen=True, eq=False, repr=False)
class _StepAdmission:
    issuer: object
    runtime: object
    context: object
    task: object
    intent: object
    session: object
    serial: int
    act: object
    observe: object
    configuration: object
    database: object
    backend: object
    key: object
    backing: object
    generation: object
    backend_scope: object
    work_scopes: tuple
    expires_at: object


class _OriginalVisualStep:
    """Closed issuer used only in the original feature/runtime lexical call."""

    def __init__(self, runtime, context, intent, session, act, observe, configuration):
        from .runtime import VisualComputerRuntime

        if type(runtime) is not VisualComputerRuntime or type(intent) is not StepIntent:
            raise ExecutionAuthorityError("Original visual step host required")
        from kestrel_sovereign.execution_custody import current_execution_custody
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

        database = runtime.repository.db
        backend = database.backend
        require_execution_work(backend)
        self._admission = _StepAdmission(
            self,
            runtime,
            context,
            asyncio.current_task(),
            intent,
            session.model_copy(deep=True),
            runtime._control_serials.get(
                runtime._key(intent.scope, intent.session_id), 0
            ),
            act,
            observe,
            configuration,
            database,
            backend,
            runtime.repository._key,
            backend._connection if type(backend) is SQLiteBackend else backend._pool,
            backend.connection_generation if type(backend) is SQLiteBackend else None,
            getattr(backend, "_execution_custody", None),
            current_execution_custody(backend),
            min(
                context.expires_at,
                session.expires_at,
                runtime._now() + timedelta(seconds=runtime.policy.operation_seconds),
            ),
        )
        self._active, self._phase = True, "issued"
        self._mutation = None
        self._claim = self._planning = self._proposal = None
        runtime.repository._original_results[self] = self._admission
        try:
            self._check_original()
        except BaseException:
            runtime.repository._original_results.pop(self, None)
            self._active = False
            raise

    def _check_original(self):
        a = self._admission
        runtime = a.runtime
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

        if (
            type(self) is not _OriginalVisualStep
            or a.issuer is not self
            or not self._active
            or runtime.repository._original_results.get(self) is not a
            or runtime.repository.db is not a.database
            or a.database.backend is not a.backend
            or runtime.repository._key is not a.key
            or getattr(a.backend, "_execution_custody", None) is not a.backend_scope
            or (
                type(a.backend) is SQLiteBackend
                and (
                    a.backend._connection is not a.backing
                    or a.backend.connection_generation != a.generation
                    or a.backend._closing
                )
            )
            or (
                type(a.backend) is not SQLiteBackend
                and a.backend._pool is not a.backing
            )
            or runtime.scope(a.context) != a.intent.scope
            or runtime._control_serials.get(
                runtime._key(a.intent.scope, a.intent.session_id), 0
            )
            != a.serial
            or runtime._now() >= a.expires_at
        ):
            raise ResourceConflict("Original visual step retired")
        for scope in a.work_scopes:
            scope.require_work()
        if a.backend_scope is not None:
            a.backend_scope.require_work()
        a.act()
        a.observe()
        a.configuration()

    def _require_owner(self):
        if asyncio.current_task() is not self._admission.task:
            raise ExecutionAuthorityError("Original visual step task required")
        self._check_original()
        require_execution_work(self._admission.runtime.repository.db.backend)

    def _mutation_plain(self, mutation):
        self._check_original()
        if (
            self._mutation is not mutation._acceptance
            or mutation._operation is not self
        ):
            raise ExecutionAuthorityError("Original step mutation differs")
        record = json.loads(mutation._plain)
        pending = int(record.get("state") != "finished")
        return mutation._plain, pending

    async def _write(self, record, parent, *, mode, expected=b""):
        from kestrel_sovereign.storage.current_visual_step import (
            _AcceptedCurrentStep,
            _CurrentVisualStepMutation,
        )
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

        self._require_owner()
        repository = self._admission.runtime.repository
        intent = self._admission.intent
        identity = (
            repository._owner(intent.scope),
            repository._address("session", intent.session_id),
            intent.action_id,
        )
        raw = record.model_dump_json().encode()
        acceptance = _AcceptedCurrentStep(
            self, asyncio.current_task(), identity, expected, raw, parent.revision, mode
        )
        self._mutation = acceptance
        repository._original_results[acceptance] = None
        mutation = None
        try:
            mutation = _CurrentVisualStepMutation(repository, acceptance)
            backend = repository.db.backend
            if type(backend) is SQLiteBackend:
                mutation._sqlite_relation = (
                    await backend._read_original_signal_outcome_relation(
                        mutation, capture=True
                    )
                )
                mutation._aux_relations = tuple(
                    [
                        await backend._read_original_signal_outcome_relation(
                            mutation, capture=True, role=role
                        )
                        for role in ("owners", "requests")
                    ]
                )
                valid = mutation._sqlite_relation is not None
            else:
                mutation._postgres_relation = (
                    await backend._capture_original_signal_outcome_relation(mutation)
                )
                mutation._aux_relations = tuple(
                    [
                        await backend._capture_original_signal_outcome_relation(
                            mutation, role=role
                        )
                        for role in ("owners", "requests")
                    ]
                )
                valid = mutation._postgres_relation is not None
            if not valid or any(item is None for item in mutation._aux_relations):
                raise ExecutionAuthorityError(
                    "Current step requires its canonical native relations"
                )
            mutation.bind_relations()
            mutation._query, mutation._match_query = mutation._fixed_queries()
            ack = await mutation.commit_step()
            self._require_owner()
            if (
                ack is not mutation._ack
                or ack.owner is not mutation
                or mutation._phase != "acknowledged"
            ):
                raise ExecutionAuthorityError("Original visual step native ACK absent")
            return bytes(
                mutation._entry.values[3]
                if mode == "claim"
                else mutation._entry.values[0]
            )
        finally:
            if mutation is not None:
                mutation.close()
            else:
                repository._original_results.pop(acceptance, None)
            self._mutation = None

    async def claim(self, parent):
        self._require_owner()
        if self._phase != "issued" or parent.session != self._admission.session:
            raise ExecutionAuthorityError("Original visual step claim differs")
        self._planning = PlanningRecord(
            intent=self._admission.intent,
            session_sha256=hashlib.sha256(
                parent.session.model_dump_json().encode()
            ).hexdigest(),
        )
        self._claim = await self._write(self._planning, parent, mode="claim")
        self._phase = "planning"

    def proposal_action_id(self):
        self._require_owner()
        if self._phase != "planning" or self._claim is None:
            raise ExecutionAuthorityError("Original visual step claim ACK required")
        return self._admission.intent.action_id

    def action_deadline(self):
        self.proposal_action_id()
        return self._admission.expires_at

    def require_plan(self, runtime, context, session_id, prompt):
        self.proposal_action_id()
        a = self._admission
        if (
            runtime is not a.runtime
            or context is not a.context
            or session_id != a.intent.session_id
            or prompt != a.intent.prompt
        ):
            raise ExecutionAuthorityError("Original visual planning frame differs")

    def bind_proposal(self, proposal):
        self._require_owner()
        if self._phase != "planning" or self._proposal is not None:
            raise ExecutionAuthorityError("Original visual step proposal differs")
        if proposal is not None:
            if (
                proposal.action_id != self.proposal_action_id()
                or proposal.session_id != self._admission.session.session_id
            ):
                raise ExecutionAuthorityError(
                    "Original visual step proposal identity differs"
                )
            self._proposal = proposal.model_copy(deep=True)

    def own_pending(self, scope, session_id, action_id, payload):
        self._require_owner()
        return (
            self._phase == "planning"
            and self._claim is not None
            and (scope, session_id, action_id)
            == (
                self._admission.intent.scope,
                self._admission.intent.session_id,
                self._admission.intent.action_id,
            )
            and bytes(payload) == self._claim
        )

    async def promote(self, parent, request):
        from .repository import ActionRecord

        self._require_owner()
        if (
            self._phase != "planning"
            or self._proposal != request
            or parent.session != self._admission.session
        ):
            raise ExecutionAuthorityError(
                "Only the original step may promote its proposal"
            )
        await self._write(
            ActionRecord(
                request=request,
                dispatch_started=False,
                step_intent=self._admission.intent,
            ),
            parent,
            mode="update",
            expected=self._claim,
        )
        self._phase = "promoted"

    async def finish(self, parent, answer):
        self._require_owner()
        if (
            self._phase != "planning"
            or self._proposal is not None
            or parent.session != self._admission.session
        ):
            raise ExecutionAuthorityError("Original no-action step differs")
        terminal = PlanningRecord.model_validate(
            {
                **self._planning.model_dump(mode="python"),
                "state": "finished",
                "answer": answer,
            }
        )
        await self._write(terminal, parent, mode="update", expected=self._claim)
        self._phase = "finished"
        return {
            "action_id": self._admission.intent.action_id,
            "state": "finished",
            "answer": answer,
            "receipt": None,
        }

    def close(self):
        if (
            self._admission.issuer is not self
            or asyncio.current_task() is not self._admission.task
        ):
            raise ExecutionAuthorityError("Original visual step closes in its owner")
        self._active = False
        self._admission.runtime.repository._original_results.pop(self, None)


def step_intent(runtime, context, session_id, action_id, prompt, configuration):
    if (
        type(action_id) is not str
        or not action_id.strip()
        or len(action_id) > 128
        or type(prompt) is not str
        or not prompt.strip()
        or len(prompt.encode()) > 16384
    ):
        raise ValueError("Visual step requires bounded stable action ID and prompt")
    # A descriptive source/configuration binding, never an admission grant.
    directory = Path(__file__).parent
    source = b"".join(
        (directory / name).read_bytes()
        for name in ("step.py", "inference.py", "runtime.py")
    )
    source += (
        directory.parent / "features" / "visual_computer" / "feature.py"
    ).read_bytes()
    normalized = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if len(normalized.encode()) > 32768:
        raise ValueError("Visual step configuration exceeds bounds")
    return StepIntent(
        scope=runtime.scope(context),
        session_id=session_id,
        action_id=action_id,
        prompt=prompt,
        configuration=normalized,
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


async def run_step(
    runtime,
    context,
    session_id,
    action_id,
    prompt,
    *,
    planner,
    configuration,
    configuration_current,
):
    from .repository import ActionRecord
    from .runtime import VisualAction

    intent = step_intent(runtime, context, session_id, action_id, prompt, configuration)
    async with runtime.operation(context, VisualAction.ACT) as act:  # noqa: SIM117 - retain both explicit independent grants
        async with runtime.operation(context, VisualAction.OBSERVE) as observe:
            scope = intent.scope
            async with runtime._lock(session_id):
                parent = await runtime.repository.get(
                    scope, session_id=session_id, guard=act
                )
                try:
                    previous = await runtime.repository.action(
                        scope, session_id, action_id, guard=act
                    )
                except ResourceNotFound:
                    previous = None
                if previous is not None:
                    original = (
                        previous.intent
                        if type(previous) is PlanningRecord
                        else previous.step_intent
                    )
                    if original != intent:
                        raise ResourceConflict(
                            "Visual step ID already binds another intent"
                        )
                    act()
                    observe()
                    configuration_current()
                    if type(previous) is PlanningRecord:
                        if previous.state == "finished":
                            return {
                                "action_id": action_id,
                                "state": "finished",
                                "answer": previous.answer,
                                "receipt": None,
                            }
                        raise ResourceOutcomeUnknown(
                            "Original visual step planning requires reconciliation; never replan"
                        )
                    if type(previous) is not ActionRecord:
                        raise ResourceConflict("Original visual step record differs")
                    if (
                        previous.receipt is not None
                        and previous.receipt.outcome != "unknown"
                    ):
                        return {
                            "action_id": action_id,
                            "state": previous.receipt.outcome,
                            "receipt": previous.receipt.model_dump(mode="json"),
                        }
                    reconcile = previous.request
                else:
                    runtime._ready(parent)
                    if await runtime.repository.has_pending_actions(
                        scope, session_id, guard=act
                    ):
                        raise ResourceOutcomeUnknown(
                            "Reconcile original session input before a new step"
                        )
                    owner = _OriginalVisualStep(
                        runtime,
                        context,
                        intent,
                        parent.session,
                        act,
                        observe,
                        configuration_current,
                    )
                    try:
                        await owner.claim(parent)
                    except BaseException:
                        owner.close()
                        raise
                    reconcile = None
            if reconcile is not None:
                receipt = await runtime.action_status(context, session_id, action_id)
                return {
                    "action_id": action_id,
                    "state": receipt.outcome,
                    "receipt": receipt.model_dump(mode="json"),
                }
            try:
                async with asyncio.timeout_at(
                    asyncio.get_running_loop().time()
                    + max(
                        0,
                        (owner._admission.expires_at - runtime._now()).total_seconds(),
                    )
                ):
                    owner.proposal_action_id()  # ACK and original owner before any model/observation work.
                    result = await planner(owner)
                    owner._require_owner()
                    from kestrel_sdk.computer_use import ActionRequest

                    request = result["action_request"]
                    if request is not None:
                        request = ActionRequest.model_validate_json(json.dumps(request))
                    owner.bind_proposal(request)
                    if request is None:
                        async with runtime._lock(session_id):
                            parent = await runtime.repository.get(
                                scope, session_id=session_id, guard=act
                            )
                            return await owner.finish(parent, result["answer"])
                    receipt = await runtime.act(context, request, _step_owner=owner)
                    owner._require_owner()
                    return {
                        "action_id": action_id,
                        "state": receipt.outcome,
                        "receipt": receipt.model_dump(mode="json"),
                    }
            finally:
                owner.close()
