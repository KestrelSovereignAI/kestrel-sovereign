"""Negative control: a match that forgets a variant fails ``assert_never``."""

from typing import assert_never

from kestrel_sovereign.tools.invocation_outcome import (
    Cancelled, ControlDeny, Deferred, Success, ToolInvocationOutcome,
)


def name_without_error(outcome: ToolInvocationOutcome) -> str:
    match outcome:
        case Success():
            return "success"
        case ControlDeny():
            return "control"
        case Cancelled():
            return "cancelled"
        case Deferred():
            return "deferred"
        case _:
            assert_never(outcome)
