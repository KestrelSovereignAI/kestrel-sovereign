"""Tool parameters as a Gemini ``Schema``, shared by the Google and Vertex adapters.

Every adapter receives a tool's parameters as the JSON Schema in an
OpenAI-format definition's ``function.parameters``. Most providers take JSON
Schema as it is. google-genai instead validates
``FunctionDeclaration.parameters`` as its own ``Schema``, an OpenAPI subset,
and refuses the whole request over one keyword outside it. An isolated
service's schema reaches the adapters whole (#3925), so a single tool saying
``"type": ["string", "null"]`` or ``"$ref"`` would otherwise fail every
Gemini turn it is offered on.

:func:`gemini_parameters` is the one place such a schema is weakened for
Gemini. Every rule below either keeps what the schema says or admits more
than it does; none tells Gemini a type or constraint the schema does not:

* a ``type`` list becomes one ``type`` plus ``nullable: true``, or, naming
  several types, an ``anyOf`` with one branch per type, each keeping the
  keywords that apply to it, and naming a repeated type once. A keyword
  that applies to none of the types is dropped. A list with a member that
  is not a JSON type name is dropped, as is a ``type`` that is not one;
* a local ``$ref`` (``#`` or ``#/...``) is replaced by the schema its JSON
  pointer names, but only when the whole schema is one resource: one where
  no object below the root, whatever keyword holds it, has a string ``$id``
  (or draft-04's ``id``) and none anywhere uses ``$anchor``,
  ``$dynamicAnchor``, ``$dynamicRef``, ``$recursiveAnchor`` or
  ``$recursiveRef``. In any other schema a pointer
  may be read against a base this projection does not track, so no
  reference is resolved. One that is not local, one that cycles back into
  itself, one in such a schema, and any past the expansion budget become
  ``{}`` (any value). Nothing is ever fetched;
* beside a ``$ref``, only the annotations ``title``, ``description``,
  ``default`` and ``example`` are kept: drafts disagree on whether the
  others apply at all, and laying them over the referenced schema could
  close an object the schema leaves open;
* a subschema nested more than :data:`MAX_PROJECTION_DEPTH` deep becomes
  ``{}``, and a keyword whose value nests deeper than the depth left is
  dropped, so the projection's own recursion stays bounded whatever the
  schema;
* ``oneOf`` becomes ``anyOf``, so exclusivity is lost; a string ``const``
  becomes a one-member ``enum``;
* an ``enum`` with a member other than a string is dropped (Gemini's are
  strings), except that ``null`` members become ``nullable: true``;
* a property whose schema is ``false`` is omitted, and one whose schema is
  ``true`` becomes ``{}``;
* ``additionalProperties`` is dropped beside ``patternProperties``, and
  ``items`` beside ``prefixItems``: with their partner gone they would
  constrain values the schema admits;
* a keyword whose value is not the JSON type ``Schema`` takes for it (a
  ``description`` that is not a string, a ``minLength`` that is not a
  non-negative integer, a ``minimum`` that is not finite or that a float
  cannot hold exactly, ...) is dropped;
* every other keyword ``Schema`` lacks (``allOf``, ``not``,
  ``patternProperties``, ``$defs``, ``$schema``, ``examples``, ...) is
  dropped.

A schema already inside the dialect comes back equal to the one given. The
result shares no mutable value with its input. The projection never raises:
one tool's schema must not fail every Gemini turn it is offered on. When a
reference or the depth bound costs the model a constraint, one warning names
the tool and counts what was lost, never the schema's own text.
"""

import copy
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import unquote

from kestrel_sovereign.json_walk import json_containers

logger = logging.getLogger(__name__)

#: The names a JSON Schema ``type`` may use.
_JSON_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})

#: The largest count google-genai's ``Schema`` (an int64 on the wire) takes.
_MAX_COUNT = 2**63 - 1


def _is_text(value: Any) -> bool:
    return isinstance(value, str)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_COUNT


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a bound the SDK's float field holds exactly. An
    integer a float rounds (``2**53 + 1``) would move the bound inward."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        as_float = float(value)
    except OverflowError:
        return False
    return math.isfinite(as_float) and as_float == value


def _is_flag(value: Any) -> bool:
    return isinstance(value, bool)


def _is_names(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(member, str) for member in value)


#: The JSON spelling of each google-genai ``Schema`` field copied as it is,
#: with the shape the field takes. ``type``, ``properties``, ``items``,
#: ``additionalProperties``, ``anyOf`` and ``enum`` are projected instead;
#: ``ref`` and ``defs`` are left out because a local reference is inlined.
_VALUE_KEYWORDS: Dict[str, Callable[[Any], bool]] = {
    "description": _is_text,
    "format": _is_text,
    "pattern": _is_text,
    "title": _is_text,
    "maxItems": _is_count,
    "maxLength": _is_count,
    "maxProperties": _is_count,
    "minItems": _is_count,
    "minLength": _is_count,
    "minProperties": _is_count,
    "maximum": _is_number,
    "minimum": _is_number,
    "nullable": _is_flag,
    "propertyOrdering": _is_names,
    "required": _is_names,
}

#: ``Schema`` fields whose value is an instance of the schema: data of any
#: shape, copied when it is plain JSON shallow enough.
_DATA_KEYWORDS = frozenset({"default", "example"})

#: The keywords kept beside a ``$ref``: they describe the value, and
#: constrain nothing.
_ANNOTATION_KEYWORDS = ("title", "description", "default", "example")

#: The keywords that apply to a value of each JSON type, for splitting a
#: ``type`` list into ``anyOf`` branches.
_TYPE_KEYWORDS = {
    "object": (
        "properties", "required", "additionalProperties", "minProperties",
        "maxProperties", "propertyOrdering",
    ),
    "array": ("items", "minItems", "maxItems"),
    "string": ("enum", "format", "minLength", "maxLength", "pattern"),
    "number": ("enum", "format", "minimum", "maximum"),
    "integer": ("enum", "format", "minimum", "maximum"),
}
_TYPED_KEYWORDS = frozenset(keyword for keywords in _TYPE_KEYWORDS.values() for keyword in keywords)

#: Keywords that make a pointer's meaning depend on more than the document
#: it is in, wherever they appear.
_RESOURCE_KEYWORDS = ("$anchor", "$dynamicAnchor", "$dynamicRef", "$recursiveAnchor", "$recursiveRef")
#: Keywords that give an object below the root its own base URI; draft-04
#: spells it ``id``. Only a string is one: a property or definition named
#: ``$id`` or ``id`` has a schema as its value.
_IDENTIFIER_KEYWORDS = ("$id", "id")

#: How many ``$ref`` expansions one schema may make. Inlining copies the
#: target, so references to references could otherwise grow exponentially.
MAX_REFERENCE_EXPANSIONS = 256

#: How deep one projected subschema may sit in another, counting each
#: ``properties`` entry, ``items``, ``additionalProperties`` and ``anyOf``
#: branch, and each inlined ``$ref``. The expansion budget bounds how many
#: references are inlined, not how deep: a short schema whose references
#: chain through object properties would otherwise project, and recurse, as
#: deep as the chain is long, and exhaust the interpreter stack on every turn
#: the tool is offered (#3925).
MAX_PROJECTION_DEPTH = 64

#: What a projection can weaken past the dialect's own mapping, in the order
#: its warning reports them.
RECURSIVE_REFERENCE = "recursive $ref (now any value)"
UNRESOLVED_REFERENCE = "$ref to no schema object (now any value)"
MULTI_RESOURCE_REFERENCE = "$ref in a schema with embedded resources or anchors (now any value)"
REFERENCE_PAST_BUDGET = "$ref past the expansion budget (now any value)"
SUBSCHEMA_PAST_DEPTH = f"subschema past depth {MAX_PROJECTION_DEPTH} (now any value)"
VALUE_PAST_DEPTH = f"keyword value past depth {MAX_PROJECTION_DEPTH} (dropped)"
_WEAKENINGS = (
    RECURSIVE_REFERENCE,
    UNRESOLVED_REFERENCE,
    MULTI_RESOURCE_REFERENCE,
    REFERENCE_PAST_BUDGET,
    SUBSCHEMA_PAST_DEPTH,
    VALUE_PAST_DEPTH,
)

#: How much of a tool's name the warning quotes: a service chooses it.
_TOOL_NAME_LOG_LIMIT = 200


def gemini_parameters(schema: Any, *, tool_name: str) -> Any:
    """``schema`` as a google-genai ``FunctionDeclaration.parameters``.

    ``tool_name`` is the tool the schema belongs to, for the warning a
    weakened reference or depth bound logs. A value that is not a dict is
    returned unchanged, for the SDK to judge.
    """
    if not isinstance(schema, dict):
        return schema
    projection = _Projection(schema)
    parameters = projection.project(schema, (), 1)
    weakened = projection.weakened
    if weakened:
        logger.warning(
            "Gemini is offered weaker parameters for tool %r than it declares: %s",
            str(tool_name)[:_TOOL_NAME_LOG_LIMIT],
            "; ".join(
                f"{reason}: {weakened[reason]}" for reason in _WEAKENINGS if reason in weakened
            ),
        )
    return parameters


def _single_resource(schema: Dict[str, Any]) -> bool:
    """Whether a local JSON pointer in ``schema`` names the same value
    however its schema is read: ``schema`` is one resource, and uses no
    anchor or dynamic reference.

    Every object is looked at, whatever keyword holds it, so a ``$id`` under
    ``default`` counts as well: that only resolves fewer references.
    """
    for path, container in json_containers(schema):
        if not isinstance(container, dict):
            continue
        if any(keyword in container for keyword in _RESOURCE_KEYWORDS):
            return False
        if path and any(isinstance(container.get(key), str) for key in _IDENTIFIER_KEYWORDS):
            return False
    return True


#: The ids of the reference targets being inlined above a node: a cycle
#: re-enters one of them, however its reference is spelled.
_Active = Tuple[int, ...]


class _Projection:
    def __init__(self, root: Dict[str, Any]) -> None:
        self._root = root
        self._resolves = _single_resource(root)
        self._expansions_left = MAX_REFERENCE_EXPANSIONS
        #: How often each of :data:`_WEAKENINGS` happened.
        self.weakened: Dict[str, int] = {}

    def _weaken(self, reason: str) -> None:
        self.weakened[reason] = self.weakened.get(reason, 0) + 1

    def project(self, node: Dict[str, Any], active: _Active, depth: int) -> Dict[str, Any]:
        """``node`` projected, ``depth`` deep in the result (the root is 1)."""
        if depth > MAX_PROJECTION_DEPTH:
            if node:
                self._weaken(SUBSCHEMA_PAST_DEPTH)
            return {}
        if "$ref" not in node:
            return self._keywords(node, active, depth)
        projected = self._referenced(node["$ref"], active, depth)
        annotations = {key: node[key] for key in _ANNOTATION_KEYWORDS if key in node}
        projected.update(self._keywords(annotations, active, depth))
        return projected

    def _referenced(self, reference: Any, active: _Active, depth: int) -> Dict[str, Any]:
        """The projection of the schema ``reference`` names, or ``{}``."""
        if not (isinstance(reference, str) and reference.startswith("#")):
            self._weaken(UNRESOLVED_REFERENCE)
            return {}
        if not self._resolves:
            self._weaken(MULTI_RESOURCE_REFERENCE)
            return {}
        target = _pointer_target(self._root, reference)
        if target is True:
            # The schema that admits any value: ``{}`` loses nothing.
            return {}
        if not isinstance(target, dict):
            self._weaken(UNRESOLVED_REFERENCE)
            return {}
        if id(target) in active:
            self._weaken(RECURSIVE_REFERENCE)
            return {}
        if self._expansions_left <= 0:
            self._weaken(REFERENCE_PAST_BUDGET)
            return {}
        self._expansions_left -= 1
        # One level deeper, though it lands in the same place in the result:
        # the depth bounds the recursion, not just the result.
        return self.project(target, active + (id(target),), depth + 1)

    def _subschema(self, value: Any, active: _Active, depth: int) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return self.project(value, active, depth + 1)

    def _branches(self, value: Any, active: _Active, depth: int) -> Optional[List[Dict[str, Any]]]:
        if not (isinstance(value, list) and value):
            return None
        return [self._subschema(branch, active, depth) for branch in value]

    def _keywords(self, node: Dict[str, Any], active: _Active, depth: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        nullable = False
        several_types: List[str] = []
        for key, value in node.items():
            if key == "type":
                if isinstance(value, str):
                    if value in _JSON_TYPES:
                        out["type"] = value
                    continue
                if not (
                    isinstance(value, list)
                    and value
                    and all(isinstance(member, str) and member in _JSON_TYPES for member in value)
                ):
                    continue
                # Each type once: a branch per repeat would copy the same
                # subschemas, doubling the result at every level that repeats.
                named = list(dict.fromkeys(member for member in value if member != "null"))
                nullable = nullable or "null" in value
                if len(named) == 1:
                    out["type"] = named[0]
                elif not named:
                    out["type"] = "null"
                else:
                    several_types = named
            elif key == "properties":
                if isinstance(value, dict):
                    out[key] = {
                        name: self._subschema(subschema, active, depth)
                        for name, subschema in value.items()
                        if subschema is not False
                    }
            elif key == "items":
                # Beside ``prefixItems`` it applies only past the prefix.
                if isinstance(value, dict) and "prefixItems" not in node:
                    out[key] = self._subschema(value, active, depth)
            elif key == "additionalProperties":
                # It applies to the names ``properties`` and
                # ``patternProperties`` leave: alone, it would reach names
                # ``patternProperties`` admits.
                if "patternProperties" in node or not isinstance(node.get("properties", {}), dict):
                    continue
                if isinstance(value, bool):
                    out[key] = value
                elif isinstance(value, dict):
                    out[key] = self._subschema(value, active, depth)
            elif key == "anyOf":
                branches = self._branches(value, active, depth)
                if branches is not None:
                    out[key] = branches
            elif key == "enum":
                members, enum_nullable = _string_enum(value)
                if members is not None:
                    out[key] = members
                nullable = nullable or enum_nullable
            elif key in _VALUE_KEYWORDS:
                if _VALUE_KEYWORDS[key](value):
                    out[key] = copy.deepcopy(value)
            elif key in _DATA_KEYWORDS:
                # Copying the value recurses through it.
                if _nests_deeper_than(value, MAX_PROJECTION_DEPTH + 1 - depth):
                    self._weaken(VALUE_PAST_DEPTH)
                elif _is_plain_data(value):
                    out[key] = copy.deepcopy(value)
        if "anyOf" not in out:
            branches = self._branches(node.get("oneOf"), active, depth)
            if branches is not None:
                out["anyOf"] = branches
        const = node.get("const")
        if "enum" not in out and isinstance(const, str):
            out["enum"] = [const]
        if several_types and "anyOf" not in out:
            out["anyOf"] = _type_branches(out, several_types)
        if nullable:
            out["nullable"] = True
        return out


def _pointer_target(root: Dict[str, Any], reference: str) -> Any:
    """The value the local JSON-pointer ``reference`` names in ``root``, or
    ``None`` for a plain-name anchor (``#name``) and for a pointer that
    leads nowhere.

    A ``$ref`` is a URI, so its fragment is percent-decoded whole before it
    is read as a JSON pointer (RFC 6901, section 6): ``%2F`` is a separator
    like ``/``, and ``#/$defs/a%2F$defs%2Fb`` names ``b`` under ``$defs``
    under ``a``. Only then does each token unescape ``~1`` and ``~0``, so
    ``~1`` is the one spelling of a ``/`` inside a name.
    """
    try:
        pointer = unquote(reference[1:], errors="strict")
    except UnicodeDecodeError:
        return None
    if pointer == "":
        return root
    if not pointer.startswith("/"):
        return None
    current: Any = root
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and _is_array_index(token, len(current)):
            current = current[int(token)]
        else:
            return None
    return current


def _type_branches(out: Dict[str, Any], types: List[str]) -> List[Dict[str, Any]]:
    """One ``anyOf`` branch per distinct type in ``types``, each taking the
    keywords of ``out`` that apply to it. Every type-specific keyword then
    leaves ``out``: one no branch took applies to none of the types.

    Several types share a keyword (``number`` and ``integer`` both take
    ``enum``): the first branch takes the value and each later one a copy,
    so no branch aliases another's.
    """
    branches = []
    taken = set()
    for member in types:
        branch: Dict[str, Any] = {"type": member}
        for keyword in _TYPE_KEYWORDS.get(member, ()):
            if keyword in out:
                branch[keyword] = copy.deepcopy(out[keyword]) if keyword in taken else out[keyword]
                taken.add(keyword)
        branches.append(branch)
    for keyword in _TYPED_KEYWORDS:
        out.pop(keyword, None)
    return branches


def _string_enum(value: Any) -> Tuple[Optional[List[str]], bool]:
    """``value``'s string members, or ``None`` to drop it; and whether a
    ``null`` member made the schema nullable."""
    if not isinstance(value, list):
        return None, False
    members = [member for member in value if isinstance(member, str)]
    others = [member for member in value if not isinstance(member, str)]
    if not members:
        return None, False
    if not others:
        return members, False
    if all(member is None for member in others):
        return members, True
    return None, False


def _nests_deeper_than(value: Any, limit: int) -> bool:
    """Whether ``value`` has an object or array more than ``limit`` deep (a
    scalar is 0 deep, ``{}`` is 1)."""
    return any(len(path) + 1 > limit for path, _ in json_containers(value))


def _is_plain_data(value: Any) -> bool:
    """Whether ``value`` is JSON a request can carry: no float that is not
    finite, which JSON cannot spell, anywhere in it."""
    if isinstance(value, float):
        return math.isfinite(value)
    for _, container in json_containers(value):
        members = container.values() if isinstance(container, dict) else container
        if any(isinstance(member, float) and not math.isfinite(member) for member in members):
            return False
    return True


def _is_array_index(token: str, length: int) -> bool:
    """Whether ``token`` is a JSON-pointer index into a list of ``length``:
    ``0``, or ASCII digits without a leading zero, naming a member.

    ``str.isdigit`` alone also admits ``²``, which ``int`` refuses; and
    ``int`` refuses a string of more than 4300 digits, so a token longer
    than any index of the list is refused before it is converted."""
    return (
        token.isascii()
        and token.isdigit()
        and (token == "0" or token[0] != "0")
        and len(token) <= len(str(length))
        and int(token) < length
    )
