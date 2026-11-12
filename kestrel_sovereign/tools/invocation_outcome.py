"""One tool invocation's outcome, as a closed set of immutable variants (#3977).

Today a path represents an invocation's outcome as a return value, a
``DeferredToolResult``, a host-private failure marker, an exception classified
by ``is_execution_control_error`` or a string in a failure envelope, and each
path flattens a different one. This is the one contract a host executor and
the normalized tool-execution envelope consume:

* ``Success``: the invocation returned. Its ``result`` may be a host-private
  dict (``_PrivateToolResult``); those facts never reach a copy or the public
  projection.
* ``ControlDeny``: execution control (an ``is_execution_control_error``
  exception) or a blocking hook decision (approval required, policy denial),
  kept as the object itself with its ``error_class``, never as a string.
* ``Cancelled``: a cancellation or process-control carrier (anything that is
  not an ``Exception``), kept as its cause.
* ``Deferred``: the ``DeferredToolResult`` itself, its one-shot private
  publication untouched.
* ``Error``: an ordinary exception with its ``error_class``.

The variant says how the invocation ended, never what its result reports: a
returned failure envelope is a ``Success`` whose public result says it failed.
``outcome_unknown`` is a fact on every variant, never a variant of its own.
It records whether execution met a ``ResourceOutcomeUnknown``, the only thing
retry may depend on (#3946).

``classify_tool_invocation`` maps what a path produces today to a variant and
``project_public`` maps a variant back to exactly what the model sees on main.
No path consumes this yet; that migration is a separate change.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeAlias, assert_never, cast

from kestrel_sdk.resources import observed_outcome_unknown
from kestrel_sdk.tools.deferred import DeferredToolResult

from kestrel_sovereign.execution_custody import (
    is_execution_control_error,
    is_tool_execution_terminal_error,
)
from kestrel_sovereign.hooks.decision_gate import BlockedDecision
from kestrel_sovereign.tools.execution_outcome import (
    _PrivateToolResult,
    _UnconfirmedToolResult,
)

__all__ = [
    "Cancelled",
    "ControlDeny",
    "Deferred",
    "Error",
    "Success",
    "ToolInvocationOutcome",
    "classify_tool_invocation",
    "project_public",
]


def _require_flag(outcome_unknown: object) -> None:
    if type(outcome_unknown) is not bool:
        raise TypeError("outcome_unknown must be a bool")


def _require_exception(error: object) -> None:
    if isinstance(error, BaseException) and not isinstance(error, Exception):
        raise TypeError("A cancellation carrier is a Cancelled outcome")
    if not isinstance(error, Exception):
        raise TypeError("An outcome's error must be an exception")


class _Outcome:
    """Behaviour every variant shares; the variants themselves are dataclasses."""

    __slots__ = ()

    def project_public(self) -> Any:
        """What the model sees for this outcome; see :func:`project_public`."""
        return project_public(cast("ToolInvocationOutcome", self))


def _public_result(result: Any) -> Any:
    """A host-private result's public dict: ``_PrivateToolResult``'s copy rule."""
    return dict(result) if isinstance(result, _PrivateToolResult) else result


def _public_success(result: Any, outcome_unknown: bool) -> Success:
    return Success(result, outcome_unknown=outcome_unknown)


@dataclass(frozen=True, slots=True)
class Success(_Outcome):
    """The invocation returned ``result``, whatever its envelope reports.

    ``result`` is the object the path returned, so host code can still read a
    host-private result's facts (a failure marker's ``error_class``, a dispatch
    receipt). Copying or pickling a ``Success`` keeps only the public result,
    as copying the ``_PrivateToolResult`` itself does.
    """

    result: Any
    outcome_unknown: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        _require_flag(self.outcome_unknown)
        if isinstance(self.result, DeferredToolResult):
            raise TypeError(
                "A deferred tool result is a Deferred outcome, not a Success"
            )

    def __copy__(self) -> Success:
        return _public_success(_public_result(self.result), self.outcome_unknown)

    def __deepcopy__(self, memo: dict) -> Success:
        return _public_success(
            deepcopy(_public_result(self.result), memo), self.outcome_unknown,
        )

    def __reduce_ex__(self, protocol: Any) -> tuple:
        return _public_success, (_public_result(self.result), self.outcome_unknown)


@dataclass(frozen=True, slots=True)
class ControlDeny(_Outcome):
    """Execution control, or a blocking hook decision, as the object itself.

    ``error`` is an ``Exception`` that ``is_execution_control_error`` reads as
    control, or the ``BlockedDecision`` a PRE hook chain produced. Copies keep
    the same object: a copy is never a second carrier of the same evidence.
    """

    error: Exception | BlockedDecision
    outcome_unknown: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        _require_flag(self.outcome_unknown)
        if isinstance(self.error, BlockedDecision):
            return
        _require_exception(self.error)
        if not is_execution_control_error(self.error):
            raise TypeError("Only execution control is a ControlDeny outcome")

    @property
    def error_class(self) -> str:
        if isinstance(self.error, BlockedDecision):
            return self.error.error_class
        return type(self.error).__name__

    def __copy__(self) -> ControlDeny:
        return self

    def __deepcopy__(self, memo: dict) -> ControlDeny:
        return self


@dataclass(frozen=True, slots=True)
class Cancelled(_Outcome):
    """A cancellation or process-control carrier, kept as its exact cause."""

    cause: BaseException
    outcome_unknown: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        _require_flag(self.outcome_unknown)
        if not isinstance(self.cause, BaseException) or isinstance(self.cause, Exception):
            raise TypeError("Only a non-Exception carrier is a Cancelled outcome")

    def __copy__(self) -> Cancelled:
        return self

    def __deepcopy__(self, memo: dict) -> Cancelled:
        return self


@dataclass(frozen=True, slots=True)
class Deferred(_Outcome):
    """The deferred result itself, with its private publication unconsumed.

    Completion (post-tool policy, required dispatch persistence, then
    ``finalize`` at the host-selected destination) belongs to the executor; its
    completed result is a ``Success``. A ``Deferred`` cannot be deep-copied or
    pickled, as no copy may duplicate or silently drop the one-shot factory.
    """

    result: DeferredToolResult
    outcome_unknown: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        _require_flag(self.outcome_unknown)
        if not isinstance(self.result, DeferredToolResult):
            raise TypeError("A Deferred outcome requires a DeferredToolResult")

    def __copy__(self) -> Deferred:
        return self

    def __deepcopy__(self, memo: dict) -> Deferred:
        raise TypeError("A deferred outcome's private publication cannot be copied")

    def __reduce_ex__(self, protocol: Any) -> tuple:
        raise TypeError("A deferred outcome's private publication cannot be pickled")


@dataclass(frozen=True, slots=True)
class Error(_Outcome):
    """An ordinary exception the invocation raised, never execution control."""

    error: Exception
    outcome_unknown: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        _require_flag(self.outcome_unknown)
        _require_exception(self.error)
        if is_execution_control_error(self.error):
            raise TypeError("Execution control is a ControlDeny outcome, never an Error")

    @property
    def error_class(self) -> str:
        return type(self.error).__name__

    def __copy__(self) -> Error:
        return self

    def __deepcopy__(self, memo: dict) -> Error:
        return self


ToolInvocationOutcome: TypeAlias = Success | ControlDeny | Cancelled | Deferred | Error


_NOTHING = object()


def classify_tool_invocation(
    *,
    returned: Any = _NOTHING,
    raised: BaseException | None = None,
    refused: BlockedDecision | None = None,
    observed: Iterable[BaseException] = (),
) -> ToolInvocationOutcome:
    """The outcome of what one invocation produced: exactly one of three.

    ``returned`` is the value the path returned, ``raised`` the exception it
    raised and ``refused`` the blocking decision its PRE hook chain produced
    (a tool's own return value cannot attest to a host decision). Anything not
    an ``Exception`` is ``Cancelled``; an exception ``is_execution_control_error``
    reads as control is ``ControlDeny``; every other exception is an ``Error``.

    ``observed`` names errors the path met and turned into its result, such as
    a wrapper's flattened failure. ``outcome_unknown`` is set when the raised
    error or any observed one carries a ``ResourceOutcomeUnknown``
    (``observed_outcome_unknown``: the error itself, an explicit cause or an
    exception-group member, never implicit context), or when the path returned
    the host's unconfirmed-outcome marker.
    """
    supplied = (returned is not _NOTHING) + (raised is not None) + (refused is not None)
    if supplied != 1:
        raise TypeError("Classify exactly one of returned=, raised= or refused=")
    unknown = any(observed_outcome_unknown(error) for error in observed)
    if refused is not None:
        if not isinstance(refused, BlockedDecision):
            raise TypeError("refused= requires the BlockedDecision a hook chain produced")
        return ControlDeny(refused, outcome_unknown=unknown)
    if raised is not None:
        if not isinstance(raised, BaseException):
            raise TypeError("raised= requires an exception")
        unknown = unknown or bool(observed_outcome_unknown(raised))
        if not isinstance(raised, Exception):
            return Cancelled(raised, outcome_unknown=unknown)
        if is_execution_control_error(raised):
            return ControlDeny(raised, outcome_unknown=unknown)
        return Error(raised, outcome_unknown=unknown)
    unknown = unknown or isinstance(returned, _UnconfirmedToolResult)
    if isinstance(returned, DeferredToolResult):
        return Deferred(returned, outcome_unknown=unknown)
    return Success(returned, outcome_unknown=unknown)


def project_public(outcome: ToolInvocationOutcome) -> Any:
    """Exactly what the model sees for ``outcome`` on main.

    A returned result is its public value (a host-private dict becomes a plain
    dict) and a deferred result its public dictionary, a plain copy that leaves
    the factory with the original. A blocking decision is its canonical
    envelope. Control, cancellation and a terminal error (one that requires
    reconciliation, ``is_tool_execution_terminal_error``) are never a result:
    the projection raises the original object, unchanged, as every path does.
    Only an ordinary error becomes ``{"success": False, "error": str(error)}``,
    the envelope each wrapper builds its own on.
    """
    match outcome:
        case Success(result=result):
            return _public_result(result)
        case Deferred(result=result):
            return dict(result)
        case ControlDeny(error=error):
            if isinstance(error, BlockedDecision):
                return error.envelope
            raise error
        case Cancelled(cause=cause):
            raise cause
        case Error(error=error):
            if is_tool_execution_terminal_error(error):
                raise error
            return {"success": False, "error": str(error)}
        case _:
            assert_never(outcome)
