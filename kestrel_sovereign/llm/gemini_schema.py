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
  keywords that apply to it, and naming a repeated type once. A list with a
  member that is not a string is dropped;
* a local ``$ref`` (``#/...``) is replaced by the schema it points at, with
  the reference's sibling keywords laid over it. One that is not local, one
  that cycles back into itself, and any past the expansion budget become
  ``{}`` (any value). Nothing is ever fetched;
* a local ``$ref`` is read in the schema resource it sits in, as JSON
  Schema 2020-12 reads it: a subschema whose ``$id`` names another resource
  (``"item.json"``, ``"https://example.com/item"``) is its own root, so a
  ``#/$defs/Value`` inside it names that subschema's ``$defs``, not the
  document's, and the schema a reference lands in carries its resource
  along. A ``$id`` of ``""`` or ``"#"`` names the enclosing resource's own
  URI, so no resource can be told apart for a reference inside it: such a
  reference becomes ``{}``. A ``$id`` holding only a plain-name fragment
  (``"#name"``, draft-07's anchor) keeps the enclosing resource, and one
  that is not a string identifies nothing;
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
result shares no mutable value with its input. The projection never raises:
one tool's schema must not fail every Gemini turn it is offered on. When a
reference or the depth bound costs the model a constraint, one warning names
the tool and counts what was lost, never the schema's own text.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

logger = logging.getLogger(__name__)

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

#: What a projection can weaken past the dialect's own mapping, in the order
#: its warning reports them.
RECURSIVE_REFERENCE = "recursive $ref (now any value)"
UNRESOLVED_REFERENCE = "$ref to no schema object (now any value)"
AMBIGUOUS_REFERENCE = "$ref in a schema resource with an ambiguous $id (now any value)"
REFERENCE_PAST_BUDGET = "$ref past the expansion budget (now any value)"
SUBSCHEMA_PAST_DEPTH = f"subschema past depth {MAX_PROJECTION_DEPTH} (now any value)"
VALUE_PAST_DEPTH = f"keyword value past depth {MAX_PROJECTION_DEPTH} (dropped)"
_WEAKENINGS = (
    RECURSIVE_REFERENCE,
    UNRESOLVED_REFERENCE,
    AMBIGUOUS_REFERENCE,
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
    parameters = projection.project(schema, (), 1, schema)
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


#: The ids of the reference targets being inlined above a node: a cycle
#: re-enters one of them, however its reference is spelled.
_Active = Tuple[int, ...]

#: The resource of a subschema whose ``$id`` names the enclosing resource's
#: own URI: a local reference inside it names no one schema.
_AMBIGUOUS = object()


class _Projection:
    def __init__(self, root: Dict[str, Any]) -> None:
        self._root = root
        self._expansions_left = MAX_REFERENCE_EXPANSIONS
        #: How often each of :data:`_WEAKENINGS` happened.
        self.weakened: Dict[str, int] = {}

    def _weaken(self, reason: str) -> None:
        self.weakened[reason] = self.weakened.get(reason, 0) + 1

    def project(
        self, node: Dict[str, Any], active: _Active, depth: int, resource: Any,
    ) -> Dict[str, Any]:
        """``node`` projected, ``depth`` deep in the result (the root is 1).

        ``resource`` is the schema resource ``node`` sits in: the root, a
        subschema with its own ``$id``, or :data:`_AMBIGUOUS`.
        """
        if depth > MAX_PROJECTION_DEPTH:
            if node:
                self._weaken(SUBSCHEMA_PAST_DEPTH)
            return {}
        # Before its own ``$ref``: a ``$id`` sets the base that reads it.
        resource = self._resource(node, resource)
        if "$ref" not in node:
            return self._keywords(node, active, depth, resource)
        projected = self._referenced(node["$ref"], active, depth, resource)
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        projected.update(self._keywords(siblings, active, depth, resource))
        return projected

    def _resource(self, node: Any, enclosing: Any) -> Any:
        """The schema resource ``node`` sits in, inside ``enclosing``.

        Only a string ``$id`` is an identifier: a ``properties`` or ``$defs``
        member named ``$id`` holds a schema, never a string.
        """
        if node is self._root or not isinstance(node, dict):
            return enclosing
        identifier = node.get("$id")
        if not isinstance(identifier, str):
            return enclosing
        base, _, fragment = identifier.partition("#")
        if base:
            return node
        if fragment:
            return enclosing
        return _AMBIGUOUS

    def _referenced(
        self, reference: Any, active: _Active, depth: int, resource: Any,
    ) -> Dict[str, Any]:
        """The projection of the schema ``reference`` names in ``resource``,
        or ``{}``."""
        if not (isinstance(reference, str) and reference.startswith("#")):
            self._weaken(UNRESOLVED_REFERENCE)
            return {}
        if resource is _AMBIGUOUS:
            self._weaken(AMBIGUOUS_REFERENCE)
            return {}
        target, target_resource = self._local_target(resource, reference)
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
        return self.project(target, active + (id(target),), depth + 1, target_resource)

    def _local_target(self, resource: Dict[str, Any], reference: str) -> Tuple[Any, Any]:
        """The value the local JSON-pointer ``reference`` names in
        ``resource``, and the resource that value sits in.

        The value is ``None`` for a plain-name anchor (``#name``) and for a
        pointer that leads nowhere. A pointer may pass into a subschema with
        its own ``$id`` on its way: the value then sits in that resource.

        A ``$ref`` is a URI, so its fragment is percent-decoded whole before
        it is read as a JSON pointer (RFC 6901, section 6): ``%2F`` is a
        separator like ``/``, and ``#/$defs/a%2F$defs%2Fb`` names ``b`` under
        ``$defs`` under ``a``. Only then does each token unescape ``~1`` and
        ``~0``, so ``~1`` is the one spelling of a ``/`` inside a name.
        """
        try:
            pointer = unquote(reference[1:], errors="strict")
        except UnicodeDecodeError:
            return None, resource
        if pointer == "":
            return resource, resource
        if not pointer.startswith("/"):
            return None, resource
        current: Any = resource
        within = resource
        for token in pointer[1:].split("/"):
            within = self._resource(current, within)
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and token in current:
                current = current[token]
            elif isinstance(current, list) and _is_array_index(token, len(current)):
                current = current[int(token)]
            else:
                return None, within
        return current, within

    def _subschema(self, value: Any, active: _Active, depth: int, resource: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return self.project(value, active, depth + 1, resource)

    def _keywords(
        self, node: Dict[str, Any], active: _Active, depth: int, resource: Any,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        nullable = False
        several_types: List[str] = []
        for key, value in node.items():
            if key == "type" and isinstance(value, list):
                if not all(isinstance(member, str) for member in value):
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
            elif key == "properties" and isinstance(value, dict):
                out[key] = {
                    name: self._subschema(subschema, active, depth, resource)
                    for name, subschema in value.items()
                    if subschema is not False
                }
            elif key == "items":
                if isinstance(value, dict):
                    out[key] = self._subschema(value, active, depth, resource)
            elif key == "additionalProperties":
                if isinstance(value, bool):
                    out[key] = value
                elif isinstance(value, dict):
                    out[key] = self._subschema(value, active, depth, resource)
            elif key == "anyOf" and isinstance(value, list):
                out[key] = [self._subschema(branch, active, depth, resource) for branch in value]
            elif key == "enum":
                members, enum_nullable = _string_enum(value)
                if members is not None:
                    out[key] = members
                nullable = nullable or enum_nullable
            elif key in _SCHEMA_KEYWORDS:
                # ``default`` and ``example`` are data of any shape, and
                # copying one recurses through it.
                if _nests_deeper_than(value, MAX_PROJECTION_DEPTH + 1 - depth):
                    self._weaken(VALUE_PAST_DEPTH)
                else:
                    out[key] = copy.deepcopy(value)
        one_of = node.get("oneOf")
        if "anyOf" not in out and isinstance(one_of, list):
            out["anyOf"] = [self._subschema(branch, active, depth, resource) for branch in one_of]
        const = node.get("const")
        if "enum" not in out and isinstance(const, str):
            out["enum"] = [const]
        if several_types and "anyOf" not in out:
            out["anyOf"] = _type_branches(out, several_types)
        if nullable:
            out["nullable"] = True
        return out


def _type_branches(out: Dict[str, Any], types: List[str]) -> List[Dict[str, Any]]:
    """One ``anyOf`` branch per distinct type in ``types``, each taking the
    keywords of ``out`` that apply to it, which leave ``out``.

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
    for keyword in taken:
        del out[keyword]
    return branches


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
