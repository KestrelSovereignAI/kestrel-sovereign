"""Every object and array in a JSON value, visited without recursion.

A tool's JSON Schema arrives from an isolated service or an MCP server, and
the JSON decoder that read it admits documents nested far deeper than the
interpreter stack. Code that must look at every part of such a value,
whatever keyword holds it (#3925), walks it with :func:`json_containers`.
"""

from typing import Any, Iterator, List, Tuple


def json_pointer_token(name: Any) -> str:
    """``name`` as one JSON-pointer reference token (RFC 6901): ``~`` is
    written ``~0`` and ``/`` is written ``~1``."""
    return str(name).replace("~", "~0").replace("/", "~1")


def json_members(container: Any) -> Iterator[Tuple[str, Any]]:
    """Each member of the dict or list ``container`` with its JSON-pointer
    token: an escaped key, or an index."""
    if isinstance(container, dict):
        return ((json_pointer_token(key), item) for key, item in container.items())
    return ((str(index), item) for index, item in enumerate(container))


def json_containers(value: Any) -> Iterator[Tuple[List[str], Any]]:
    """Each dict and list in ``value``, ``value`` itself first, depth first
    in document order, with the JSON-pointer tokens of its path from
    ``value``.

    The path is one list the walk reuses: copy it to keep it. A container
    ``len(path) + 1`` deep is yielded before anything inside it, so a
    caller that stops at one never has its contents walked. A scalar
    ``value`` yields nothing.

    Iterative, holding one member iterator per level, so a deep value
    cannot exhaust the stack.
    """
    if not isinstance(value, (dict, list)):
        return
    path: List[str] = []
    yield path, value
    members = [json_members(value)]
    while members:
        member = next(members[-1], None)
        if member is None:
            members.pop()
            if path:
                path.pop()
            continue
        token, item = member
        if not isinstance(item, (dict, list)):
            continue
        path.append(token)
        yield path, item
        members.append(json_members(item))
