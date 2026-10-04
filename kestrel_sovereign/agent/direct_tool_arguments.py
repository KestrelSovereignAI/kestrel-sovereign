"""Argument check for direct tool calls (#3396).

A direct tool is one ``AgentTool`` called with the model's arguments as
keyword arguments: a promoted ``@tool`` method, an unpromoted one resolved
straight off its feature, or a dynamically mounted tool such as an MCP
server's. A feature's subagent dispatcher takes ``task`` and ``context``,
and callers carry those two names over to direct tools by habit. A ``@tool``
method that does not declare them raised ``TypeError`` before it ran
(``shell() got an unexpected keyword argument 'context'``), losing whatever
else the call asked for, such as ``capture_output``.

Every direct-tool dispatch path applies two rules, here and nowhere else:

* ``task`` and ``context`` are dropped when the tool does not declare them.
  They are orchestration arguments, not data.
* Any other undeclared argument refuses the call, with an error naming the
  tool's parameters and the nearest one when a close match exists. It is
  never dropped: a misspelt argument must not run the tool with a default.

The parameters a tool declares are the ones its advertised schema names. An
SDK ``ToolSchema`` lists them, and that list is closed: it is generated from
the method signature, which takes nothing else. A JSON-Schema ``parameters``
object (the MCP shape) is closed only when it sets ``additionalProperties``
to ``false`` and declares no ``patternProperties``, as JSON Schema defines.
A tool whose schema is neither shape is passed its arguments unchanged,
because nothing says what it accepts.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Orchestration arguments a feature's subagent dispatcher takes. A direct
#: tool that does not declare one has it dropped rather than refused.
GENERIC_ORCHESTRATION_ARGUMENTS = frozenset({"task", "context"})

# Argument names come from the model and are echoed back to it, so a
# pathological name, or a pathological number of them, is bounded rather than
# repeated whole.
_ARGUMENT_NAME_REPR_LIMIT = 64
_MAX_NAMED_ARGUMENTS = 10


@dataclass(frozen=True)
class _DeclaredParameters:
    names: Tuple[str, ...]
    closed: bool


def _declared_parameters(tool: Any) -> Optional[_DeclaredParameters]:
    """The parameters ``tool`` advertises, or ``None`` when unknowable."""
    parameters = getattr(getattr(tool, "schema", None), "parameters", None)
    if isinstance(parameters, (list, tuple)):
        names = tuple(getattr(parameter, "name", None) for parameter in parameters)
        if not all(isinstance(name, str) for name in names):
            return None
        return _DeclaredParameters(names=names, closed=True)
    if isinstance(parameters, dict):
        properties = parameters.get("properties")
        names = tuple(properties) if isinstance(properties, dict) else ()
        if not all(isinstance(name, str) for name in names):
            return None
        # ``patternProperties`` admits names beyond ``properties``; matching
        # them is the server's job, so such a schema is treated as open.
        return _DeclaredParameters(
            names=names,
            closed=(
                parameters.get("additionalProperties") is False
                and not parameters.get("patternProperties")
            ),
        )
    return None


def _render_name(name: Any) -> str:
    rendered = repr(name)
    if len(rendered) > _ARGUMENT_NAME_REPR_LIMIT:
        rendered = f"{rendered[:_ARGUMENT_NAME_REPR_LIMIT]}..."
    return rendered


def _abbreviates(short: str, name: str) -> bool:
    """Whether ``short`` is ``name`` with letters left out (``cmd``, ``command``)."""
    if len(short) < 3 or len(short) >= len(name) or short[0] != name[0]:
        return False
    remaining = iter(name)
    return all(char in remaining for char in short)


def _closest_parameter(name: str, candidates: Sequence[str]) -> Optional[str]:
    """The parameter ``name`` most likely meant, or ``None``.

    An abbreviation wins over spelling similarity: by similarity alone
    ``cmd`` is nearer ``cwd`` than ``command``.
    """
    abbreviated = [candidate for candidate in candidates if _abbreviates(name, candidate)]
    if abbreviated:
        return difflib.get_close_matches(name, abbreviated, n=1, cutoff=0.0)[0]
    matches = difflib.get_close_matches(name, candidates, n=1)
    return matches[0] if matches else None


def _unknown_arguments_error(
    tool_name: str,
    unknown: Sequence[str],
    declared: Sequence[str],
    supplied: Sequence[str],
) -> str:
    # A parameter the call already supplied is not what a misspelling meant.
    candidates = [name for name in declared if name not in supplied]
    described = []
    for name in unknown[:_MAX_NAMED_ARGUMENTS]:
        # JSON keys are strings; an in-process caller's need not be.
        match = _closest_parameter(name, candidates) if isinstance(name, str) else None
        hint = f" (did you mean {match!r}?)" if match else ""
        described.append(f"{_render_name(name)}{hint}")
    if len(unknown) > _MAX_NAMED_ARGUMENTS:
        described.append(f"and {len(unknown) - _MAX_NAMED_ARGUMENTS} more")
    noun = "argument" if len(unknown) == 1 else "arguments"
    valid = ", ".join(declared) if declared else "none (call it with no arguments)"
    return (
        f"{tool_name} does not accept {noun} {', '.join(described)}. "
        f"Valid parameters: {valid}."
    )


def normalize_direct_tool_arguments(
    tool_name: str,
    tool: Any,
    arguments: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Apply the direct-tool argument rules to one call.

    Returns:
        ``(arguments, error)``. ``error`` is ``None`` when the call may run
        with the returned arguments; otherwise it names the undeclared
        arguments and the tool's parameters, and the call must not run.
    """
    declared = _declared_parameters(tool)
    if declared is None or not isinstance(arguments, dict):
        return arguments, None
    accepted = set(declared.names)
    undeclared = [name for name in arguments if name not in accepted]
    if not undeclared:
        return arguments, None

    unknown = [
        name for name in undeclared if name not in GENERIC_ORCHESTRATION_ARGUMENTS
    ]
    if unknown and declared.closed:
        return arguments, _unknown_arguments_error(
            tool_name, unknown, declared.names, list(arguments)
        )

    dropped = [name for name in undeclared if name in GENERIC_ORCHESTRATION_ARGUMENTS]
    if not dropped:
        return arguments, None
    logger.info(
        "[DIRECT-TOOL] %s: dropped undeclared orchestration argument(s) %s",
        tool_name,
        ", ".join(sorted(dropped)),
    )
    return {
        name: value for name, value in arguments.items() if name not in dropped
    }, None
