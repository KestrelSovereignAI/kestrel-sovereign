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
Gemini:

* a ``type`` list becomes one ``type`` plus ``nullable: true``, or, naming
  several types, an ``anyOf`` with one branch per type, each keeping the
  keywords that apply to it;
* a local ``$ref`` (``#/...``) is replaced by the schema it points at, with
  the reference's sibling keywords laid over it. One that is not local, one
  that cycles back into itself, and any past the expansion budget become
  ``{}`` (any value). Nothing is ever fetched;
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
* every other keyword ``Schema`` lacks (``allOf``, ``not``,
  ``patternProperties``, ``$defs``, ``$schema``, ``examples``, ...) is
  dropped.

A schema already inside the dialect comes back equal to the one given. The
result shares no mutable value with its input.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

#: The JSON spelling of each google-genai ``Schema`` field, less ``ref`` and
#: ``defs``: a local reference is inlined instead.
_SCHEMA_KEYWORDS = frozenset({
    "additionalProperties", "anyOf", "default", "description", "enum",
    "example", "format", "items", "maxItems", "maxLength", "maxProperties",
    "maximum", "minItems", "minLength", "minProperties", "minimum", "nullable",
    "pattern", "properties", "propertyOrdering", "required", "title", "type",
})

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


def gemini_parameters(schema: Any) -> Any:
    """``schema`` as a google-genai ``FunctionDeclaration.parameters``.

    A value that is not a dict is returned unchanged, for the SDK to judge.
    """
    if not isinstance(schema, dict):
        return schema
    return _Projection(schema).project(schema, (), 1)


#: The ids of the reference targets being inlined above a node: a cycle
#: re-enters one of them, however its reference is spelled.
_Active = Tuple[int, ...]


class _Projection:
    def __init__(self, root: Dict[str, Any]) -> None:
        self._root = root
        self._expansions_left = MAX_REFERENCE_EXPANSIONS

    def project(self, node: Dict[str, Any], active: _Active, depth: int) -> Dict[str, Any]:
        """``node`` projected, ``depth`` deep in the result (the root is 1)."""
        if depth > MAX_PROJECTION_DEPTH:
            return {}
        if "$ref" not in node:
            return self._keywords(node, active, depth)
        reference = node["$ref"]
        target = None
        if isinstance(reference, str) and self._expansions_left > 0:
            target = _local_target(self._root, reference)
        if isinstance(target, dict) and id(target) not in active:
            self._expansions_left -= 1
            # One level deeper, though it lands in the same place in the
            # result: the depth bounds the recursion, not just the result.
            projected = self.project(target, active + (id(target),), depth + 1)
        else:
            projected = {}
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        projected.update(self._keywords(siblings, active, depth))
        return projected

    def _subschema(self, value: Any, active: _Active, depth: int) -> Dict[str, Any]:
        return self.project(value, active, depth + 1) if isinstance(value, dict) else {}

    def _keywords(self, node: Dict[str, Any], active: _Active, depth: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        nullable = False
        several_types: List[str] = []
        for key, value in node.items():
            if key == "type" and isinstance(value, list):
                named = [member for member in value if member != "null"]
                nullable = nullable or len(named) < len(value)
                if len(named) == 1:
                    out["type"] = named[0]
                elif not named:
                    out["type"] = "null"
                else:
                    several_types = named
            elif key == "properties" and isinstance(value, dict):
                out[key] = {
                    name: self._subschema(subschema, active, depth)
                    for name, subschema in value.items()
                    if subschema is not False
                }
            elif key == "items":
                if isinstance(value, dict):
                    out[key] = self._subschema(value, active, depth)
            elif key == "additionalProperties":
                if isinstance(value, bool):
                    out[key] = value
                elif isinstance(value, dict):
                    out[key] = self._subschema(value, active, depth)
            elif key == "anyOf" and isinstance(value, list):
                out[key] = [self._subschema(branch, active, depth) for branch in value]
            elif key == "enum":
                members, enum_nullable = _string_enum(value)
                if members is not None:
                    out[key] = members
                nullable = nullable or enum_nullable
            elif key in _SCHEMA_KEYWORDS:
                # ``default`` and ``example`` are data of any shape, and
                # copying one recurses through it.
                if not _nests_deeper_than(value, MAX_PROJECTION_DEPTH + 1 - depth):
                    out[key] = copy.deepcopy(value)
        one_of = node.get("oneOf")
        if "anyOf" not in out and isinstance(one_of, list):
            out["anyOf"] = [self._subschema(branch, active, depth) for branch in one_of]
        const = node.get("const")
        if "enum" not in out and isinstance(const, str):
            out["enum"] = [const]
        if several_types and "anyOf" not in out:
            out["anyOf"] = [
                {
                    "type": member,
                    **{
                        keyword: out[keyword]
                        for keyword in _TYPE_KEYWORDS.get(member, ())
                        if keyword in out
                    },
                }
                for member in several_types
            ]
            for member in several_types:
                for keyword in _TYPE_KEYWORDS.get(member, ()):
                    out.pop(keyword, None)
        if nullable:
            out["nullable"] = True
        return out


def _string_enum(value: Any) -> Tuple[Optional[List[str]], bool]:
    """``value``'s string members, or ``None`` to drop it; and whether a
    ``null`` member made the schema nullable."""
    if not isinstance(value, list):
        return None, False
    members = [member for member in value if isinstance(member, str)]
    others = [member for member in value if not isinstance(member, str)]
    if not others:
        return members, False
    if members and all(member is None for member in others):
        return members, True
    return None, False


def _nests_deeper_than(value: Any, limit: int) -> bool:
    """Whether ``value`` has an object or array more than ``limit`` deep (a
    scalar is 0 deep, ``{}`` is 1). Iterative, so a deep value cannot
    exhaust the stack it is checked to protect."""
    pending = [(value, 1)]
    while pending:
        current, level = pending.pop()
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, list):
            children = current
        else:
            continue
        if level > limit:
            return True
        pending.extend((child, level + 1) for child in children)
    return False


def _local_target(root: Dict[str, Any], reference: str) -> Any:
    """The value a local JSON-pointer ``reference`` names in ``root``.

    ``None`` for a reference that is not a fragment of this schema, for a
    plain-name anchor (``#name``), and for a pointer that leads nowhere.

    A ``$ref`` is a URI, so its fragment is percent-decoded whole before it
    is read as a JSON pointer (RFC 6901, section 6): ``%2F`` is a separator
    like ``/``, and ``#/$defs/a%2F$defs%2Fb`` names ``b`` under ``$defs``
    under ``a``. Only then does each token unescape ``~1`` and ``~0``, so
    ``~1`` is the one spelling of a ``/`` inside a name.
    """
    if not reference.startswith("#"):
        return None
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
        elif isinstance(current, list) and _is_array_index(token) and int(token) < len(current):
            current = current[int(token)]
        else:
            return None
    return current


def _is_array_index(token: str) -> bool:
    """Whether ``token`` is a JSON-pointer array index: ``0``, or ASCII
    digits without a leading zero. ``str.isdigit`` alone also admits ``²``,
    which ``int`` refuses."""
    return token.isascii() and token.isdigit() and (token == "0" or token[0] != "0")
