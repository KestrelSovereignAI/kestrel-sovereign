"""Captured native actions share ordinary tool governance, never authority."""

from collections.abc import Mapping
from dataclasses import dataclass
import json

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable

from kestrel_sdk.tools import NativeAction, NativeActionContext
from kestrel_sovereign.security.input_guardrails import MAX_TOOL_ARG_LENGTH, validate_tool_arguments


def snapshot_action_context(context: NativeActionContext) -> NativeActionContext:
    """Revalidate and detach identity before a caller or hook can retain it."""
    if type(context) is not NativeActionContext:
        raise TypeError('Native actions require their original identity context')
    return NativeActionContext(**context.to_dict())


def capture_native_actions(tool) -> tuple[NativeAction, ...]:
    """Prepare the entire static declaration before changing the registry."""
    declared = getattr(tool, 'native_actions', {})
    if not isinstance(declared, Mapping):
        raise TypeError('Native action declarations must be a mapping')
    captured = []
    for name, declaration in list(declared.items()):
        if type(declaration) is not NativeAction or name != declaration.name:
            raise ValueError('Native action mapping key must match its declaration')
        sealed = NativeAction(declaration.name, declaration.method,
                              declaration.schema_json, declaration.handler)
        Draft202012Validator.check_schema(sealed.schema)
        captured.append(sealed)
    return tuple(captured)


def action_audit(context: NativeActionContext, arguments) -> dict:
    """Content-free evidence; exact form values stay in encrypted custody."""
    return {'native_action': context.to_dict(), **action_parameter_audit(arguments)}


def action_parameter_audit(arguments) -> dict:
    """Bounded parameter names only; canonical rows keep identity separately."""
    names, size = [], 2  # The JSON array's brackets.
    if type(arguments) is dict:
        for key in sorted(key for key in arguments if type(key) is str):
            if len(key) > MAX_TOOL_ARG_LENGTH:
                continue
            addition = len(json.dumps(key)) + (2 if names else 0)
            if size + addition <= MAX_TOOL_ARG_LENGTH:
                names.append(key)
                size += addition
    result = {'parameter_names': names, 'parameters_type': type(arguments).__name__}
    if type(arguments) is dict and len(names) < len(arguments):
        result['parameter_names_omitted'] = len(arguments) - len(names)
    return result


@dataclass(frozen=True)
class NativeActionInvocation:
    declaration: NativeAction
    context: NativeActionContext

    def validate(self, arguments) -> dict:
        """Validate finite JSON against captured bytes without coercion or I/O."""
        message = 'Native action parameters do not match the captured schema'
        if type(arguments) is not dict:
            raise ValueError(message)
        pending, visited = [arguments], set()
        while pending:
            value = pending.pop()
            kind = type(value)
            if kind in (dict, list):
                if id(value) in visited:
                    continue
                visited.add(id(value))
                if kind is dict:
                    if any(type(key) is not str for key in value):
                        raise ValueError(message)
                    pending.extend(value.values())
                else:
                    pending.extend(value)
            elif value is not None and kind not in (str, int, float, bool):
                raise ValueError(message)
        try:
            snapshot = json.loads(json.dumps(arguments, allow_nan=False))
            # An explicit empty public Registry cannot retrieve external refs.
            Draft202012Validator(self.declaration.schema, registry=Registry()).validate(snapshot)
        except (ValueError, TypeError, RecursionError, ValidationError, Unresolvable):
            raise ValueError(message) from None
        valid, _ = validate_tool_arguments(self.context.tool_name, snapshot, redact_paths=True)
        if not valid:
            raise ValueError(message)
        return snapshot
