"""Private host-generated failures, distinct from an audit receipt."""

from copy import deepcopy


TOOL_OUTCOME_UNCONFIRMED = (
    "Tool outcome is unconfirmed; reconcile before retrying. Do not replay"
)


TOOL_PRECONDITION_CHANGED = (
    "Tool precondition changed; inspect current resource state before retrying"
)


class _HostToolFailure(dict):
    """An in-process host marker; wire data alone cannot create this fact.

    It does not prove persistence. Copies lose the marker just as copies of
    an audited receipt lose their ability to suppress another dispatch row.
    """

    def __copy__(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return deepcopy(dict(self), memo)

    def __reduce_ex__(self, protocol):
        return dict, (dict(self),)


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
