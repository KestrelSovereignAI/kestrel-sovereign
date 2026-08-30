"""Stable vocabulary for cooperative Stop requests."""

from enum import Enum


class StopScope(str, Enum):
    """The addressable work boundary affected by a cooperative Stop.

    Process termination is intentionally absent. It is a runtime lifecycle
    operation, not a cooperative cancellation scope.
    """

    HOST = "host"
    AGENT = "agent"
    TURN = "turn"
    TOOL_CALL = "tool_call"
