"""Original native non-dispatch facts, never authority or a retry permission.

The host's one-shot CAS owns one invocation. Only a concrete driver which
accounts for its physical native children and actual input write can prove that
invocation never sent input. A copied exception or a generic refusal cannot.
"""
from contextlib import contextmanager
from contextvars import ContextVar

from kestrel_sdk.computer_use import ActionRequest, ComputerSession
from kestrel_sdk.resources import ResourceOutcomeUnknown

_ORIGIN = ContextVar('visual-original-dispatch', default=None)
_HUMAN_ORIGIN = ContextVar('visual-original-human-dispatch', default=None)
_TRACE = ContextVar('visual-native-dispatch', default=None)


class NativeActionNotDispatched(Exception):
    def __init__(self, trace, original):
        super().__init__('original_native_action_not_dispatched')
        self._trace, self.original = trace, original


class _OriginalDispatch:
    def __init__(self, session, request):
        self.session = ComputerSession.model_validate_json(session.model_dump_json())
        self.request = ActionRequest.model_validate_json(request.model_dump_json())
        self.active, self.trace = True, None
        self.driver_entered = False

    def enter_driver(self):
        if not self.active or _ORIGIN.get() is not self or self.driver_entered or self.trace is not None:
            raise ResourceOutcomeUnknown('original_driver_entry_differs')
        self.driver_entered = True

    def refuse_unentered(self, error):
        if not self.active or _ORIGIN.get() is not self or self.driver_entered or self.trace is not None:
            raise ResourceOutcomeUnknown('original_host_dispatch_may_have_entered')
        trace = self.trace = _NativeDispatch(self)
        trace.finished = True
        return trace.refusal(error)

    def require(self, fact):
        trace = getattr(fact, '_trace', None)
        if (type(fact) is not NativeActionNotDispatched or not self.active
                or _ORIGIN.get() is not self or trace is not self.trace or trace is None
                or trace.origin is not self or trace.proof is not fact or not trace.finished
                or not trace.no_input()):
            raise ResourceOutcomeUnknown('native_non_dispatch_fact_not_original')


class _NativeDispatch:
    def __init__(self, origin):
        self.origin = origin
        self.commands = 0
        self.input_started = self.rpc_started = self.rpc_settled = self.finished = False
        self.proof = None

    def no_input(self):
        return (self.commands == 0 and not self.input_started
                and (not self.rpc_started or self.rpc_settled))

    def refusal(self, original):
        if self.origin.active and self.origin.trace is self and self.no_input():
            self.proof = NativeActionNotDispatched(self, original)
            return self.proof
        return None


class _OriginalHumanDispatch:
    """One normalized original human call, distinct from agent provenance."""
    def __init__(self, session, lease, request, frame):
        from kestrel_sdk.computer_use.human_control import (
            HumanActionRequest, HumanControlLease, HumanScreenFrame,
        )
        self.session = ComputerSession.model_validate_json(session.model_dump_json())
        self.lease = HumanControlLease.model_validate_json(lease.model_dump_json())
        self.request = HumanActionRequest.model_validate_json(request.model_dump_json())
        self.frame = HumanScreenFrame.model_validate_json(frame.model_dump_json())
        self.binding = (self.session, self.lease, self.request, self.frame)
        self.active, self.trace, self.driver_entered = True, None, False

    def enter_driver(self):
        if not self.active or _HUMAN_ORIGIN.get() is not self or self.driver_entered or self.trace is not None:
            raise ResourceOutcomeUnknown('original_human_driver_entry_differs')
        self.driver_entered = True

    def refuse_unentered(self, error):
        if not self.active or _HUMAN_ORIGIN.get() is not self or self.driver_entered or self.trace is not None:
            raise ResourceOutcomeUnknown('original_human_dispatch_may_have_entered')
        trace = self.trace = _NativeDispatch(self)
        trace.finished = True
        return trace.refusal(error)

    def require(self, fact):
        trace = getattr(fact, '_trace', None)
        if (type(fact) is not NativeActionNotDispatched or not self.active
                or _HUMAN_ORIGIN.get() is not self or trace is not self.trace or trace is None
                or trace.origin is not self or trace.proof is not fact or not trace.finished
                or not trace.no_input()):
            raise ResourceOutcomeUnknown('native_human_non_dispatch_fact_not_original')


@contextmanager
def original_human_dispatch(session, lease, request, frame):
    origin = _OriginalHumanDispatch(session, lease, request, frame)
    token = _HUMAN_ORIGIN.set(origin)
    try:
        yield origin
    finally:
        origin.active = False
        _HUMAN_ORIGIN.reset(token)


@contextmanager
def native_human_dispatch(session, lease, request, driver_id):
    origin = _HUMAN_ORIGIN.get()
    if origin is None:
        yield None
        return
    if (type(origin) is not _OriginalHumanDispatch or not origin.active
            or origin.trace is not None or origin.session != session
            or origin.lease != lease or origin.request != request or session.driver_id != driver_id):
        raise ResourceOutcomeUnknown('native_human_dispatch_invocation_not_original')
    trace = origin.trace = _NativeDispatch(origin)
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        trace.finished = True
        _TRACE.reset(token)


@contextmanager
def original_dispatch(session, request):
    origin = _OriginalDispatch(session, request)
    token = _ORIGIN.set(origin)
    try:
        yield origin
    finally:
        origin.active = False
        _ORIGIN.reset(token)


@contextmanager
def native_dispatch(session, request, driver_id):
    origin = _ORIGIN.get()
    if origin is None:
        yield None  # Bare SDK calls do not acquire host receipt provenance.
        return
    if (not origin.active or origin.trace is not None or origin.session != session
            or origin.request != request or session.driver_id != driver_id):
        raise ResourceOutcomeUnknown('native_dispatch_invocation_not_original')
    trace = origin.trace = _NativeDispatch(origin)
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        trace.finished = True
        _TRACE.reset(token)


def native_trace():
    return _TRACE.get()


def raise_driver_borrow_outcome(outcome, *, operation, retain_originals=False):
    """A revoked native borrower is UNKNOWN, not cancellation of its caller.

    Preserve original caller cancellation and all native control/commit facts.
    This is the single result boundary shared by the Docker and worker drivers.
    """
    import asyncio
    from kestrel_sovereign._async_ownership import raise_owned_outcome
    error = outcome.error
    from types import TracebackType

    from kestrel_sovereign.execution_custody import (
        execution_commit_outcome,
        is_execution_control_error,
    )
    def descriptive_fields_only(node):
        # Absence of a recognized control predicate is not proof that an
        # original carrier has no other admission/native-source facts.
        fields = vars(node)
        if any(type(key) is not str or key not in {
                "__notes__", "_kestrel_retained_traceback"} for key in fields):
            return False
        notes = fields.get("__notes__", [])
        trace = fields.get("_kestrel_retained_traceback")
        return (type(notes) is list and all(type(note) is str for note in notes)
                and (trace is None or type(trace) is TracebackType))
    cause = error.__cause__ if type(error) is asyncio.CancelledError else None
    ordinary_cause = (cause is None or (type(cause) is ResourceOutcomeUnknown
                      and cause.__cause__ is None and descriptive_fields_only(cause)))
    if (outcome.cancellation is None and type(error) is asyncio.CancelledError
            and ordinary_cause and descriptive_fields_only(error)
            and not is_execution_control_error(error)
            and execution_commit_outcome(error) is None):
        # No receipt or no-effect claim follows from an interrupted driver.
        # The original ledger/native dispatch owner still decides custody.
        raise ResourceOutcomeUnknown("original_visual_operation_interrupted") from error
    return raise_owned_outcome(outcome, operation=operation, retain_originals=retain_originals)
