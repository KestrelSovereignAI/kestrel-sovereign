"""Private host-generated uncertainty, distinct from an audit receipt."""

from copy import deepcopy


TOOL_OUTCOME_UNCONFIRMED = (
    "Tool outcome is unconfirmed; reconcile before retrying. Do not replay"
)


class _UnconfirmedToolResult(dict):
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
