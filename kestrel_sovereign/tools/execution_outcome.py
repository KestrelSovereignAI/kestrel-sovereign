"""Private host-generated failures, distinct from an audit receipt."""

from copy import deepcopy


TOOL_OUTCOME_UNCONFIRMED = (
    "Tool outcome is unconfirmed; reconcile before retrying. Do not replay"
)


TOOL_PRECONDITION_CHANGED = (
    "Tool precondition changed; inspect current resource state before retrying"
)


class _PrivateToolResult(dict):
    """Host-owned facts never survive copying or public serialization."""

    def __copy__(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return deepcopy(dict(self), memo)

    def __reduce_ex__(self, protocol):
        return dict, (dict(self),)


class _HostToolFailure(_PrivateToolResult):
    """A host-generated failure marker, carrying no audit authority."""


class _UnconfirmedToolResult(_HostToolFailure):
    """The host could not confirm execution or delivery."""

    outcome = "unconfirmed"
    error_class = "ResourceOutcomeUnknown"
    message = TOOL_OUTCOME_UNCONFIRMED


class _PreconditionFailedToolResult(_HostToolFailure):
    """A tool reported a known precondition refusal after hooks ran."""

    outcome = "precondition_failed"
    error_class = "ResourceConflict"
    message = TOOL_PRECONDITION_CHANGED
