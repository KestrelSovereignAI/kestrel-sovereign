"""An isolated tool's own JSON Schema is what the model is offered (#3925).

``IsolatedFeatureTool.schema`` used to rebuild the model-facing
``function.parameters`` from the flat ``ToolParameter`` list derived from the
service's ``input_schema``. That projection kept only each top-level
property's type, description, enum, default and items, so a service that
required ``record.id``, closed its objects and made ``note`` nullable was
advertised as ``record: object`` and ``note: string``.

These tests pin:

* the advertised ``function.parameters`` equals the service's schema, as a
  copy, on the direct-tool mount and the feature-subagent path alike;
* ``schema.parameters`` stays flat for command help and name checks, and
  marks a ``[T, "null"]`` property nullable;
* a schema with a non-local ``$ref`` is refused at registration with
  :class:`IsolatedToolSchemaError`, opening no connection, while ``$ref``
  keys that are data are advertised unchanged;
* so is a schema nested more than ``MAX_ISOLATED_TOOL_SCHEMA_DEPTH`` deep,
  which copying it for the model would recurse through;
* the real Telegram package's tools are advertised as before.
"""

from __future__ import annotations

import asyncio
import copy
import json
import socket
from unittest.mock import Mock, patch

import pytest
from kestrel_sdk.isolated_feature import ToolMetadata
from kestrel_sdk.tools.base import ToolCategory, ToolSchema

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.isolated_runtime import (
    MAX_ISOLATED_TOOL_SCHEMA_DEPTH,
    IsolatedFeatureTool,
    IsolatedRuntimePreparationError,
    IsolatedToolSchemaError,
    ProxyFeature,
    _input_schema_to_parameters,
    safe_isolated_runtime_preparation_diagnostic,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from tests.utils.tool_schema_corpus import ISSUE_3925_SCHEMA, SCHEMA_CORPUS

TOOL_NAME = "record_put"


class _Service:
    """The ``ProxyFeature`` surface an ``IsolatedFeatureTool`` forwards to."""

    async def call_isolated_tool(self, name, args):
        return {"success": True}


def _tool(input_schema, *, name: str = TOOL_NAME) -> IsolatedFeatureTool:
    return IsolatedFeatureTool(
        _Service(),
        ToolMetadata(name=name, description="Store a record", input_schema=input_schema),
    )


def _advertised(tool) -> dict:
    return tool.schema.to_openai_format()["function"]["parameters"]


# --- the canonical schema reaches the model ---------------------------------


@pytest.mark.parametrize("schema", list(SCHEMA_CORPUS.values()), ids=list(SCHEMA_CORPUS))
def test_the_advertised_parameters_are_the_service_schema(schema):
    service_schema = copy.deepcopy(schema)

    definition = _tool(schema).schema.to_openai_format()

    assert definition["type"] == "function"
    assert definition["function"]["name"] == TOOL_NAME
    assert definition["function"]["description"] == "Store a record"
    assert definition["function"]["parameters"] == service_schema


def test_the_issue_reproduction_keeps_every_nested_constraint():
    """The model is no longer offered ``record: object`` and ``note: string``."""
    parameters = _advertised(_tool(ISSUE_3925_SCHEMA))

    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["record"]
    record = parameters["properties"]["record"]
    assert record["properties"] == {"id": {"type": "string"}}
    assert record["required"] == ["id"]
    assert record["additionalProperties"] is False
    assert parameters["properties"]["note"] == {"type": ["string", "null"]}


def test_each_definition_is_a_copy_of_the_service_schema():
    """A consumer that edits its definition (``register_dynamic_tools``
    renames it) cannot reach the service's metadata or the next definition."""
    schema = copy.deepcopy(ISSUE_3925_SCHEMA)
    tool = _tool(schema)

    first = _advertised(tool)
    first["properties"]["record"]["required"].append("label")
    first["properties"].pop("note")

    assert schema == ISSUE_3925_SCHEMA
    assert _advertised(tool) == ISSUE_3925_SCHEMA


def test_a_tool_without_a_schema_object_keeps_the_flat_definition():
    tool = IsolatedFeatureTool(_Service(), {"name": TOOL_NAME, "description": "d"})

    assert type(tool.schema) is ToolSchema
    assert _advertised(tool) == {"type": "object", "properties": {}, "required": []}


def test_the_flat_parameters_stay_for_command_help_and_mark_nullable():
    parameters = {p.name: p for p in _tool(ISSUE_3925_SCHEMA).schema.parameters}

    assert list(parameters) == ["record", "note"]
    assert (parameters["record"].type, parameters["record"].required) == ("object", True)
    assert parameters["record"].nullable is False
    assert (parameters["note"].type, parameters["note"].required) == ("string", False)
    assert parameters["note"].nullable is True


def test_a_direct_tool_mount_advertises_the_service_schema():
    """``register_dynamic_tools`` is how an explored feature's tools reach
    the model as direct tools."""
    with patch("kestrel_sovereign.kestrel_agent.LLMService"):
        agent = KestrelAgent(did="did:test:agent")
    agent.features = {}

    assert agent.register_dynamic_tools("RecordFeature", [_tool(ISSUE_3925_SCHEMA)]) == 1

    (definition,) = [
        d for d in agent._direct_tool_defs if d["function"]["name"] == TOOL_NAME
    ]
    assert definition["function"]["parameters"] == ISSUE_3925_SCHEMA


# --- references are data ------------------------------------------------------


def _forbid_network(monkeypatch) -> list:
    attempts: list = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("a schema reference opened a connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return attempts


def _with(reference_holder: dict) -> dict:
    return {"type": "object", "properties": {"record": reference_holder}}


NON_LOCAL_REFERENCES = {
    "remote URL": (
        _with({"$ref": "https://schemas.example.com/record.json"}),
        "#/properties/record/$ref",
    ),
    "remote URL with fragment": (
        _with({"$ref": "https://schemas.example.com/defs.json#/$defs/Record"}),
        "#/properties/record/$ref",
    ),
    "relative document": (_with({"$ref": "record.json"}), "#/properties/record/$ref"),
    "URN": (_with({"$ref": "urn:example:record"}), "#/properties/record/$ref"),
    "not a string": (_with({"$ref": {"url": "x"}}), "#/properties/record/$ref"),
    "under items": (
        {"type": "object", "properties": {"lines": {
            "type": "array", "items": {"$ref": "https://example.com/line.json"},
        }}},
        "#/properties/lines/items/$ref",
    ),
    "under anyOf": (
        _with({"anyOf": [{"type": "null"}, {"$ref": "https://example.com/r.json"}]}),
        "#/properties/record/anyOf/1/$ref",
    ),
    "under $defs": (
        {"type": "object", "$defs": {"R": {"$ref": "https://example.com/r.json"}},
         "properties": {"record": {"$ref": "#/$defs/R"}}},
        "#/$defs/R/$ref",
    ),
    "root": ({"$ref": "https://example.com/schema.json"}, "#/$ref"),
    "dynamic reference": (
        _with({"$dynamicRef": "https://example.com/meta#node"}),
        "#/properties/record/$dynamicRef",
    ),
}


@pytest.mark.parametrize(
    "schema, location",
    list(NON_LOCAL_REFERENCES.values()),
    ids=list(NON_LOCAL_REFERENCES),
)
def test_a_non_local_reference_is_refused_at_registration(monkeypatch, schema, location):
    attempts = _forbid_network(monkeypatch)

    with pytest.raises(IsolatedToolSchemaError) as refused:
        _tool(schema)

    assert attempts == []
    assert isinstance(refused.value, IsolatedRuntimePreparationError)
    assert (refused.value.tool_name, refused.value.location) == (TOOL_NAME, location)
    assert refused.value.problem == IsolatedToolSchemaError.NON_LOCAL_REFERENCE
    # The service chose the reference, and a URL can carry credentials.
    assert "example" not in str(refused.value)
    assert TOOL_NAME in str(refused.value)


def test_the_camel_case_schema_spelling_is_checked_too(monkeypatch):
    _forbid_network(monkeypatch)
    with pytest.raises(IsolatedToolSchemaError):
        IsolatedFeatureTool(_Service(), {
            "name": TOOL_NAME,
            "description": "d",
            "inputSchema": _with({"$ref": "https://example.com/r.json"}),
        })


REFERENCES_AS_DATA = {
    "a default": _with({"type": "object", "default": {"$ref": "https://example.com"}}),
    "an enum member": _with({"enum": [{"$ref": "https://example.com"}]}),
    "examples": _with({"type": "object", "examples": [{"$ref": "https://example.com"}]}),
    "an unknown keyword": _with({"x-origin": {"$ref": "https://example.com"}}),
    "a property named $ref": {
        "type": "object",
        "properties": {"$ref": {"type": "string", "format": "uri"}},
    },
    "a local anchor": _with({"$ref": "#record"}),
}


@pytest.mark.parametrize(
    "schema", list(REFERENCES_AS_DATA.values()), ids=list(REFERENCES_AS_DATA)
)
def test_a_reference_key_that_is_data_is_advertised_unchanged(monkeypatch, schema):
    attempts = _forbid_network(monkeypatch)

    assert _advertised(_tool(schema)) == schema
    assert attempts == []


# --- nesting is bounded -------------------------------------------------------


def _nested_items(depth: int) -> dict:
    """A schema whose objects nest exactly ``depth`` deep, the root being 1."""
    node: dict = {}
    for _ in range(depth - 1):
        node = {"type": "array", "items": node}
    return node


def test_a_schema_at_the_nesting_bound_is_advertised_whole():
    schema = _nested_items(MAX_ISOLATED_TOOL_SCHEMA_DEPTH)

    assert _advertised(_tool(schema)) == schema


def _deep_value(depth: int):
    value: list = []
    for _ in range(depth - 1):
        value = [value]
    return value


TOO_DEEP = {
    # Without the bound, copying it for the model raises RecursionError on
    # every turn: the transport's JSON decoder admits far deeper documents.
    "deep subschemas": (
        _nested_items(600),
        "#" + "/items" * MAX_ISOLATED_TOOL_SCHEMA_DEPTH,
    ),
    "one level past the bound": (
        _nested_items(MAX_ISOLATED_TOOL_SCHEMA_DEPTH + 1),
        "#" + "/items" * MAX_ISOLATED_TOOL_SCHEMA_DEPTH,
    ),
    # Data is copied too. ``record`` is 3 deep and ``default`` 4.
    "deep data under default": (
        _with({"type": "array", "default": _deep_value(MAX_ISOLATED_TOOL_SCHEMA_DEPTH)}),
        "#/properties/record/default"
        + "/0" * (MAX_ISOLATED_TOOL_SCHEMA_DEPTH - 3),
    ),
    "a name that needs escaping": (
        {"properties": {"a/b~c": _nested_items(MAX_ISOLATED_TOOL_SCHEMA_DEPTH)}},
        "#/properties/a~1b~0c" + "/items" * (MAX_ISOLATED_TOOL_SCHEMA_DEPTH - 2),
    ),
}


@pytest.mark.parametrize(
    "schema, location", list(TOO_DEEP.values()), ids=list(TOO_DEEP)
)
def test_a_schema_nested_past_the_bound_is_refused_at_registration(schema, location):
    with pytest.raises(IsolatedToolSchemaError) as refused:
        _tool(schema)

    assert isinstance(refused.value, IsolatedRuntimePreparationError)
    assert refused.value.problem == IsolatedToolSchemaError.TOO_DEEP
    assert (refused.value.tool_name, refused.value.location) == (TOOL_NAME, location)
    assert f"more than {MAX_ISOLATED_TOOL_SCHEMA_DEPTH} levels" in str(refused.value)


def test_nesting_is_checked_before_references():
    """Both are wrong; the depth walk runs first because it is the bounded one."""
    schema = _with({"$ref": "https://example.com/r.json", "default": _deep_value(200)})

    with pytest.raises(IsolatedToolSchemaError) as refused:
        _tool(schema)

    assert refused.value.problem == IsolatedToolSchemaError.TOO_DEEP


def test_an_unknown_refusal_problem_is_a_core_bug():
    with pytest.raises(ValueError):
        IsolatedToolSchemaError(TOOL_NAME, "#", "surprise")


@pytest.mark.parametrize(
    "problem, named",
    [
        (IsolatedToolSchemaError.NON_LOCAL_REFERENCE, "non-local schema reference"),
        (IsolatedToolSchemaError.TOO_DEEP, "levels deep"),
    ],
)
def test_the_startup_diagnostic_names_the_refusal_without_service_text(problem, named):
    error = IsolatedToolSchemaError("https://secret@example.com", "#/secret/$ref", problem)

    diagnostic = safe_isolated_runtime_preparation_diagnostic(error)

    assert named in diagnostic
    assert "example.com" not in diagnostic
    assert "secret" not in diagnostic


class _RemoteReferenceClient:
    """An isolated client whose service advertises a remote ``$ref``."""

    def __init__(self, **kwargs):
        self.started = False
        self.stopped = False

    @property
    def capabilities(self):
        return {"inbound_producer": False}

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def health(self):
        return True

    async def list_tools(self):
        return [{
            "name": TOOL_NAME,
            "description": "Store a record",
            "input_schema": _with({"$ref": "https://example.com/record.json"}),
        }]

    async def call_tool(self, name, args):  # pragma: no cover - never reached
        raise AssertionError("a refused tool was called")

    def on_event(self, handler):
        pass


@pytest.mark.asyncio
async def test_a_service_advertising_a_remote_reference_is_retired_unpublished(
    monkeypatch, tmp_path
):
    """Registration refuses the tools, and the child that advertised them is
    stopped rather than left running with nothing able to reach it."""
    agent = Mock(did="did:test:isolated-schema")
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    runtime = InstalledFeatureRuntime(
        class_name="TestFeature",
        entry_point="test_pkg.feature:TestFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test_service",
        description="Test proxy",
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = _RemoteReferenceClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, runtime, client_factory=client_factory)

    with pytest.raises(IsolatedToolSchemaError):
        await feature.initialize()

    assert len(clients) == 1
    assert clients[0].started is True
    assert clients[0].stopped is True
    assert feature.get_tools() == []
    await feature.shutdown()


# --- the real Telegram package ------------------------------------------------


def _without_flat_fill_ins(parameters: dict) -> dict:
    """The flat projection's additions to a flat schema: an empty
    ``description`` on every property and an empty ``required`` list."""
    result = copy.deepcopy(parameters)
    for prop in result.get("properties", {}).values():
        if prop.get("description") == "":
            del prop["description"]
    if result.get("required") == []:
        del result["required"]
    return result


def test_the_real_telegram_tools_are_advertised_as_before(monkeypatch):
    """Telegram's tools take flat strings or nothing, so the projection lost
    nothing from them: their advertised schemas are the service's, and say
    what the flat definitions said."""
    service_module = pytest.importorskip("kestrel_channel_telegram.service")
    # The loop exists before the network is refused: on Windows its self-pipe
    # is a connected socket pair.
    loop = asyncio.new_event_loop()
    try:
        attempts = _forbid_network(monkeypatch)
        # Constructed only: never started, so it opens no connection. The
        # explicit empty token keeps it from reading one from the environment.
        advertised = loop.run_until_complete(
            service_module.TelegramService({"token": ""}).get_tools()
        )
    finally:
        loop.close()

    names = []
    for metadata in advertised:
        # The host sees the tool as it arrives on the wire.
        wire = ToolMetadata.from_dict(json.loads(json.dumps(metadata.to_dict())))
        tool = IsolatedFeatureTool(_Service(), wire)
        before = ToolSchema(
            name=tool.name,
            description=wire.description,
            category=ToolCategory.COMMUNICATION,
            parameters=_input_schema_to_parameters(wire.input_schema),
            command_prefix=wire.command_prefix,
        ).to_openai_format()

        after = tool.schema.to_openai_format()

        assert after["function"]["parameters"] == metadata.input_schema
        assert after["function"]["parameters"] == _without_flat_fill_ins(
            before["function"]["parameters"]
        )
        assert {k: v for k, v in after["function"].items() if k != "parameters"} == {
            k: v for k, v in before["function"].items() if k != "parameters"
        }
        assert [p.name for p in tool.schema.parameters] == list(
            metadata.input_schema["properties"]
        )
        names.append(tool.name)

    assert {"telegram_send", "telegram_status", "telegram_link"} <= set(names)
    assert attempts == []
