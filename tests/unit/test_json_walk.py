"""``json_containers`` visits every object and array, without recursion (#3925)."""

import sys

from kestrel_sovereign.json_walk import json_containers, json_members, json_pointer_token


def _visited(value) -> list:
    return [("/".join(path), container) for path, container in json_containers(value)]


def test_every_container_is_visited_in_document_order_with_its_path():
    value = {"a": {"b": [1, {"c": 2}]}, "d/e~f": [], "g": 3}

    assert _visited(value) == [
        ("", value),
        ("a", value["a"]),
        ("a/b", value["a"]["b"]),
        ("a/b/1", {"c": 2}),
        ("d~1e~0f", []),
    ]


def test_a_scalar_has_no_containers():
    assert _visited(3) == []
    assert _visited("x") == []


class _Untouchable(dict):
    def items(self):
        raise AssertionError("the walk looked inside a container its caller stopped at")


def test_a_caller_that_stops_never_has_the_contents_walked():
    """What bounds a depth check: it stops at the first container past its
    limit, which is never looked inside."""
    inner = _Untouchable(x={"y": 1})
    walk = json_containers({"a": inner})

    next(walk)
    path, container = next(walk)
    walk.close()

    assert path == ["a"] and container is inner


def test_a_value_deeper_than_the_stack_is_walked():
    depth = sys.getrecursionlimit() * 3
    value: list = []
    for _ in range(depth):
        value = [value]

    deepest = max(len(path) for path, _ in json_containers(value))

    assert deepest == depth


def test_members_carry_escaped_tokens():
    assert list(json_members({"a/b": 1, "c~d": 2})) == [("a~1b", 1), ("c~0d", 2)]
    assert list(json_members(["x", "y"])) == [("0", "x"), ("1", "y")]
    assert json_pointer_token("~/") == "~0~1"
