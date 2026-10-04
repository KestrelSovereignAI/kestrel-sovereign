"""Argument check for direct tool calls (#3396).

A direct tool is one ``AgentTool`` called with the model's arguments as
keyword arguments: a promoted ``@tool`` method, an unpromoted one resolved
straight off its feature, or a dynamically mounted tool such as an MCP
server's. A feature's subagent dispatcher takes ``task`` and ``context``,
and callers carry those two names over to direct tools by habit. A ``@tool``
method that does not declare them raised ``TypeError`` before it ran
(``shell() got an unexpected keyword argument 'context'``), losing whatever
else the call asked for, such as ``capture_output``.

Every direct-tool dispatch path applies these rules, here and nowhere else.
An argument the tool's schema accepts always passes through unchanged. For
an argument it does not accept:

* ``task`` and ``context`` are dropped. They are orchestration arguments,
  not data.
* Any other name refuses the call, with an error naming the tool's
  parameters and the nearest one when a close match exists. It is never
  dropped: a misspelt argument must not run the tool with a default.

Which names a schema accepts depends on its shape. An SDK ``ToolSchema``
lists its parameters, and that list is closed: it is generated from the
method signature, which takes nothing else. A JSON-Schema ``parameters``
object (the MCP shape) follows JSON Schema. It accepts every name in
``properties`` and every name a ``patternProperties`` pattern matches. When
``additionalProperties`` is anything but ``false`` (``true``, absent, or a
schema object), it accepts every other name too, and validating those is the
tool's job. A tool whose schema is neither shape is passed its arguments
unchanged, because nothing says what it accepts.

A server's pattern is never run as a regular expression: ``(?:x?){4000000000}``
exhausts memory matching even ``task``, and ``^(a+)+$`` backtracks for
minutes on a long name the model chose, all on the event loop. Patterns are
compared with ``task`` and ``context`` only, and only a literal one,
optionally anchored with ``^`` and ``$`` (``^x-``, ``^context$``), is
evaluated, by string comparison. A generic name that any other pattern might
match passes through, and so does any other name a closed schema leaves to
its patterns. The tool validates them.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Orchestration arguments a feature's subagent dispatcher takes. A direct
#: tool whose schema does not accept one has it dropped rather than refused.
GENERIC_ORCHESTRATION_ARGUMENTS = frozenset({"task", "context"})

# Argument names come from the model and are echoed back to it, so a
# pathological name, or a pathological number of them, is bounded rather than
# repeated whole.
_ARGUMENT_NAME_REPR_LIMIT = 64
_MAX_NAMED_ARGUMENTS = 10

#: Characters with a meaning in an ECMA-262 regular expression. A pattern
#: with none of them inside its optional ``^``/``$`` anchors is a literal.
_REGEX_SYNTAX = frozenset("\\.^$*+?()[]{}|")


@dataclass(frozen=True)
class _DeclaredParameters:
    """The argument names a tool's schema accepts."""

    names: Tuple[str, ...]
    #: The names accepted beyond ``names``: ``None`` accepts every name;
    #: otherwise the ``patternProperties`` patterns that admit one.
    patterns: Optional[Tuple[str, ...]]
    _listed: FrozenSet[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_listed", frozenset(self.names))

    def accepts(self, name: Any) -> bool:
        """Whether the schema accepts ``name``, or might.

        Only a generic name is compared with ``patterns`` (see the module
        docstring); any other name a pattern might admit is left to the tool.
        """
        if name in self._listed or self.patterns is None:
            return True
        if not self.patterns:
            return False
        if isinstance(name, str) and name in GENERIC_ORCHESTRATION_ARGUMENTS:
            return any(
                _literal_pattern_matches(pattern, name) is not False
                for pattern in self.patterns
            )
        return True


def _literal_pattern_matches(pattern: str, name: str) -> Optional[bool]:
    """Whether ``pattern`` matches ``name``; ``None`` when it is not a literal.

    JSON Schema patterns are unanchored, so an unanchored literal matches
    anywhere in the name.
    """
    anchored_start = pattern.startswith("^")
    body = pattern[1:] if anchored_start else pattern
    anchored_end = body.endswith("$")
    if anchored_end:
        body = body[:-1]
    if not _REGEX_SYNTAX.isdisjoint(body):
        return None
    if anchored_start and anchored_end:
        return name == body
    if anchored_start:
        return name.startswith(body)
    if anchored_end:
        return name.endswith(body)
    return body in name


def _pattern_properties(pattern_properties: Any) -> Optional[Tuple[str, ...]]:
    """A closed schema's ``patternProperties`` patterns.

    ``None`` when they cannot be read: the schema then admits names this
    check cannot identify, so it accepts every name and leaves refusing one
    to the tool.
    """
    if not pattern_properties:
        return ()
    if not isinstance(pattern_properties, dict):
        return None
    patterns = tuple(pattern_properties)
    if not all(isinstance(pattern, str) for pattern in patterns):
        return None
    return patterns


def _declared_parameters(tool: Any) -> Optional[_DeclaredParameters]:
    """The parameters ``tool`` advertises, or ``None`` when unknowable."""
    parameters = getattr(getattr(tool, "schema", None), "parameters", None)
    if isinstance(parameters, (list, tuple)):
        names = tuple(getattr(parameter, "name", None) for parameter in parameters)
        if not all(isinstance(name, str) for name in names):
            return None
        return _DeclaredParameters(names=names, patterns=())
    if isinstance(parameters, dict):
        properties = parameters.get("properties")
        names = tuple(properties) if isinstance(properties, dict) else ()
        if not all(isinstance(name, str) for name in names):
            return None
        if parameters.get("additionalProperties") is not False:
            return _DeclaredParameters(names=names, patterns=None)
        return _DeclaredParameters(
            names=names,
            patterns=_pattern_properties(parameters.get("patternProperties")),
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
    supplied: FrozenSet[Any],
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
        with the returned arguments; otherwise it names the arguments the
        schema does not accept and the tool's parameters, and the call must
        not run.
    """
    declared = _declared_parameters(tool)
    if declared is None or not isinstance(arguments, dict):
        return arguments, None
    rejected = [name for name in arguments if not declared.accepts(name)]
    if not rejected:
        return arguments, None

    unknown = [
        name for name in rejected if name not in GENERIC_ORCHESTRATION_ARGUMENTS
    ]
    if unknown:
        return arguments, _unknown_arguments_error(
            tool_name, unknown, declared.names, frozenset(arguments)
        )

    logger.info(
        "[DIRECT-TOOL] %s: dropped undeclared orchestration argument(s) %s",
        tool_name,
        ", ".join(sorted(rejected)),
    )
    return {
        name: value for name, value in arguments.items() if name not in rejected
    }, None
