"""What each provider adapter sends for an isolated tool's own schema (#3925).

An isolated tool's ``function.parameters`` is now its service's JSON Schema,
whole. Each provider adapter still converts tool definitions into its own
request, and where a provider cannot express a keyword, that conversion is
the one place the schema is weakened. These tests pin, for every schema in
:data:`tests.utils.tool_schema_corpus.SCHEMA_CORPUS`, the parameters each
adapter puts on its request:

* OpenAI (and OpenRouter, its subclass), Anthropic (and Claude Max) and Codex
  send the schema verbatim;
* the Ollama adapter projects it into what the Ollama SDK's models accept,
  which then keep only what they declare, so nested ``properties``,
  ``additionalProperties``, ``anyOf`` and ``$ref`` do not reach Ollama;
* the Google and Vertex adapters project it into google-genai's ``Schema``
  (:func:`kestrel_sovereign.llm.gemini_schema.gemini_parameters`), which the
  SDK refuses the whole request without: a ``type`` list becomes ``nullable``
  and a local ``$ref`` is inlined, but only in a schema that is one
  resource. That projection never raises on a schema registration accepts,
  a recursive or over-deep reference or a value of the wrong shape
  included, never tells Gemini a type or constraint the schema does not,
  and logs one warning naming the tool when a reference or the depth bound
  costs the model a constraint.

Every projection these tests pin keeps what the schema says or admits more
than it does.

Nothing leaves the process: every SDK runs over a fake client or a mock
transport.
"""

from __future__ import annotations

import copy
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from kestrel_sdk.isolated_feature import ToolMetadata

from kestrel_sovereign.features.isolated_runtime import IsolatedFeatureTool
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter
from kestrel_sovereign.llm.codex_adapter import _convert_tools_to_codex_dynamic_tools
from kestrel_sovereign.llm.gemini_schema import (
    MAX_PROJECTION_DEPTH,
    MAX_REFERENCE_EXPANSIONS,
    MULTI_RESOURCE_REFERENCE,
    RECURSIVE_REFERENCE,
    REFERENCE_PAST_BUDGET,
    SUBSCHEMA_PAST_DEPTH,
    UNRESOLVED_REFERENCE,
    VALUE_PAST_DEPTH,
    gemini_parameters,
)
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter
from tests.utils.anthropic_client import anthropic_client
from tests.utils.ollama_client import ollama_client
from tests.utils.tool_schema_corpus import ISSUE_3925_SCHEMA, SCHEMA_CORPUS

TOOL_NAME = "record_put"
MESSAGES = [{"role": "user", "content": "Store the record."}]
CASES = list(SCHEMA_CORPUS)


class _Service:
    async def call_isolated_tool(self, name, args):
        return {"success": True}


def _definition(schema: dict) -> dict:
    """The model-facing definition of an isolated tool advertising ``schema``."""
    tool = IsolatedFeatureTool(
        _Service(),
        ToolMetadata(name=TOOL_NAME, description="Store a record", input_schema=schema),
    )
    return tool.schema.to_openai_format()


# --- what each adapter puts on its request ------------------------------------


async def _openai_parameters(adapter_type, schema: dict) -> dict:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="ok", tool_calls=None, reasoning_content=None),
            finish_reason="stop",
        )],
        usage=MagicMock(prompt_tokens=10, completion_tokens=1, total_tokens=11),
    ))
    await adapter_type().get_response(
        client=client, model="gpt-4o-mini", messages=MESSAGES,
        tools=[_definition(schema)],
    )
    (tool,) = client.chat.completions.create.call_args.kwargs["tools"]
    assert tool["function"]["name"] == TOOL_NAME
    return tool["function"]["parameters"]


async def _openai(schema):
    return await _openai_parameters(OpenAIAdapter, schema)


async def _openrouter(schema):
    return await _openai_parameters(OpenRouterAdapter, schema)


async def _anthropic_parameters(adapter_type, schema: dict) -> dict:
    client = anthropic_client(MagicMock(
        content=[MagicMock(type="text", text="ok")],
        stop_reason="end_turn",
        usage=MagicMock(input_tokens=10, output_tokens=1),
    ))
    await adapter_type().get_response(
        client=client, model="claude-sonnet-4-5-20250929", messages=MESSAGES,
        tools=[_definition(schema)],
    )
    (tool,) = client.messages.stream.call_args.kwargs["tools"]
    assert tool["name"] == TOOL_NAME
    return tool["input_schema"]


async def _anthropic(schema):
    return await _anthropic_parameters(AnthropicAdapter, schema)


async def _claude_max(schema):
    return await _anthropic_parameters(ClaudeMaxAdapter, schema)


async def _ollama(schema: dict) -> dict:
    """The native Ollama SDK's serialized request: its wire, not its input."""
    reply = {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": "ok"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 1,
    }
    requests = []
    async with ollama_client(reply, requests=requests) as client:
        await OllamaAdapter().get_response(
            client=client, model="llama3.2:3b", messages=MESSAGES,
            tools=[_definition(schema)],
        )
    (request,) = requests
    (tool,) = request["tools"]
    assert tool["function"]["name"] == TOOL_NAME
    return tool["function"]["parameters"]


def _genai_client() -> SimpleNamespace:
    response = SimpleNamespace(
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=[SimpleNamespace(text="ok", function_call=None)]),
            finish_reason="STOP",
        )],
        text="ok",
        usage_metadata=None,
    )
    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content=AsyncMock(return_value=response),
    )))


async def _genai_parameters(adapter_type, schema: dict) -> dict:
    client = _genai_client()
    await adapter_type().get_response(
        client=client, model="gemini-2.5-flash", messages=MESSAGES,
        tools=[_definition(schema)], max_tokens=256,
    )
    config = client.aio.models.generate_content.call_args.kwargs["config"]
    (tool,) = config["tools"]
    (declaration,) = tool["function_declarations"]
    assert declaration["name"] == TOOL_NAME
    return declaration["parameters"]


async def _google(schema):
    return await _genai_parameters(GoogleAdapter, schema)


async def _vertex(schema):
    return await _genai_parameters(VertexAIAdapter, schema)


async def _codex(schema: dict) -> dict:
    (tool,) = _convert_tools_to_codex_dynamic_tools([_definition(schema)])
    assert tool["name"] == TOOL_NAME
    return tool["inputSchema"]


#: The case's own schema, unchanged.
VERBATIM = object()

_ADDRESS = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}

#: Gemini: only the nullable unions and the local ``$ref`` change.
GEMINI_PARAMETERS = {
    "nested required object": VERBATIM,
    "array of objects": VERBATIM,
    "nullable type": {
        "type": "object",
        "properties": {
            "note": {"type": "string", "description": "Optional note", "nullable": True},
        },
    },
    "enum": VERBATIM,
    "anyOf": VERBATIM,
    "closed objects": VERBATIM,
    "local $ref": {
        "type": "object",
        "properties": {
            "home": _ADDRESS,
            "work": {**_ADDRESS, "description": "Work address"},
        },
        "required": ["home"],
    },
    "issue #3925 reproduction": {
        **ISSUE_3925_SCHEMA,
        "properties": {
            **ISSUE_3925_SCHEMA["properties"],
            "note": {"type": "string", "nullable": True},
        },
    },
}

#: Ollama's SDK: a property keeps only ``type``, ``items``, ``description``
#: and ``enum``, and the root drops ``additionalProperties``.
OLLAMA_PARAMETERS = {
    "nested required object": {
        "type": "object",
        "properties": {"record": {"type": "object"}},
        "required": ["record"],
    },
    "array of objects": VERBATIM,
    "nullable type": VERBATIM,
    "enum": VERBATIM,
    "anyOf": {"type": "object", "properties": {"target": {}}, "required": ["target"]},
    "closed objects": {
        "type": "object",
        "properties": {"record": {"type": "object"}},
        "required": ["record"],
    },
    "local $ref": {
        "type": "object",
        "$defs": SCHEMA_CORPUS["local $ref"]["$defs"],
        "properties": {"home": {}, "work": {"description": "Work address"}},
        "required": ["home"],
    },
    "issue #3925 reproduction": {
        "type": "object",
        "properties": {"record": {"type": "object"}, "note": {"type": ["string", "null"]}},
        "required": ["record"],
    },
}

ALL_VERBATIM = dict.fromkeys(CASES, VERBATIM)

ADAPTERS = {
    "openai": (_openai, ALL_VERBATIM),
    "openrouter": (_openrouter, ALL_VERBATIM),
    "anthropic": (_anthropic, ALL_VERBATIM),
    "claude_max": (_claude_max, ALL_VERBATIM),
    "codex": (_codex, ALL_VERBATIM),
    "ollama": (_ollama, OLLAMA_PARAMETERS),
    "google": (_google, GEMINI_PARAMETERS),
    "vertex": (_vertex, GEMINI_PARAMETERS),
}


def test_every_adapter_pins_every_case():
    for adapter, (_, expected) in ADAPTERS.items():
        assert list(expected) == CASES, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("adapter", list(ADAPTERS))
async def test_each_adapter_sends_the_pinned_parameters(adapter, case):
    run, expected = ADAPTERS[adapter]
    schema = copy.deepcopy(SCHEMA_CORPUS[case])
    pinned = expected[case]

    sent = await run(schema)

    assert sent == (SCHEMA_CORPUS[case] if pinned is VERBATIM else pinned)
    assert schema == SCHEMA_CORPUS[case], "the adapter edited the tool definition"


# --- the Ollama projection is what the Ollama SDK accepts ---------------------


def _one_property(schema) -> dict:
    return {"type": "object", "properties": {"q": schema}}


#: Schemas registration accepts that the Ollama SDK's models refuse, failing
#: every Ollama turn the tool is offered on, and the parameters Ollama is sent.
OLLAMA_PROJECTED = {
    "a property that admits anything": (
        {"type": "object", "properties": {"anything": True, "q": {"type": "string"}}},
        {"type": "object", "properties": {"anything": {}, "q": {"type": "string"}}},
    ),
    "a property that admits nothing": (
        {"type": "object", "properties": {"never": False, "q": {"type": "string"}}},
        {"type": "object", "properties": {"q": {"type": "string"}}},
    ),
    "a property schema that is a list": (_one_property([]), _one_property({})),
    "a nullable root": (
        {"type": ["object", "null"], "properties": {"q": {"type": "string"}}},
        {"type": "object", "properties": {"q": {"type": "string"}}},
    ),
    "a root that is not an object": (
        {"type": "string", "properties": {"q": {"type": "string"}}},
        {"type": "object", "properties": {"q": {"type": "string"}}},
    ),
    "a required that is a string": (
        {"type": "object", "required": "q", "properties": {"q": {"type": "string"}}},
        {"type": "object", "properties": {"q": {"type": "string"}}},
    ),
    "a required naming a number": (
        {"type": "object", "required": ["q", 1], "properties": {"q": {"type": "string"}}},
        {"type": "object", "properties": {"q": {"type": "string"}}},
    ),
    "properties that are a list": (
        {"type": "object", "properties": ["q"]},
        {"type": "object"},
    ),
    "a property type that is a number": (
        _one_property({"type": 5, "description": "Q"}), _one_property({"description": "Q"}),
    ),
    "a property type list naming a number": (
        _one_property({"type": ["string", 1]}), _one_property({}),
    ),
    "a property description that is a number": (
        _one_property({"type": "string", "description": 5}), _one_property({"type": "string"}),
    ),
    "a property enum that is not a list": (
        _one_property({"type": "string", "enum": 5}), _one_property({"type": "string"}),
    ),
}

#: Schemas the Ollama SDK accepts but would tighten: it drops a keyword
#: whose partner it keeps.
OLLAMA_PARTNERS = {
    # ``items`` applies only past ``prefixItems``: alone it would refuse
    # ``["a", 1]``.
    "items beside prefixItems": (
        _one_property({"type": "array", "prefixItems": [{"type": "string"}],
                       "items": {"type": "integer"}}),
        _one_property({"type": "array"}),
    ),
    # In draft-07 nothing beside a ``$ref`` applies: ``type`` alone would
    # refuse the referenced number.
    "constraints beside a property's $ref": (
        {"type": "object", "definitions": {"n": {"type": "number"}}, "properties": {"q": {
            "$ref": "#/definitions/n", "type": "string", "enum": ["a"], "items": {},
            "description": "Q",
        }}},
        {"type": "object", "properties": {"q": {"description": "Q"}}},
    ),
    "constraints beside the root's $ref": (
        {"$ref": "#/$defs/Args", "$defs": {"Args": {"type": "object"}}, "type": "object",
         "properties": {"q": {"type": "string"}}, "required": ["q"]},
        {"$defs": {"Args": {"type": "object"}}, "type": "object"},
    ),
}


@pytest.mark.parametrize(
    "case", list(OLLAMA_PROJECTED), ids=list(OLLAMA_PROJECTED),
)
def test_the_ollama_sdk_refuses_those_schemas_unprojected(case):
    """Why the Ollama adapter projects: sent as given, each fails the request."""
    pydantic = pytest.importorskip("pydantic")
    ollama_types = pytest.importorskip("ollama._types")
    schema, _ = OLLAMA_PROJECTED[case]

    with pytest.raises(pydantic.ValidationError):
        ollama_types.Tool.model_validate(_definition(schema))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", list({**OLLAMA_PROJECTED, **OLLAMA_PARTNERS}),
)
async def test_the_ollama_adapter_sends_a_schema_its_sdk_refuses_or_tightens_weakened(case):
    schema, expected = {**OLLAMA_PROJECTED, **OLLAMA_PARTNERS}[case]
    given = copy.deepcopy(schema)

    sent = await _ollama(given)

    assert sent == expected
    assert given == schema, "the adapter edited the tool definition"


# --- the Gemini projection is what google-genai accepts -----------------------


def _validate_declaration(parameters: dict) -> None:
    types = pytest.importorskip("google.genai.types")
    types.GenerateContentConfig.model_validate({"tools": [{"function_declarations": [
        {"name": TOOL_NAME, "description": "Store a record", "parameters": parameters},
    ]}]})


@pytest.mark.parametrize("case", CASES)
def test_google_genai_accepts_every_projected_schema(case):
    _validate_declaration(gemini_parameters(SCHEMA_CORPUS[case], tool_name=TOOL_NAME))


@pytest.mark.parametrize("case", ["nullable type", "local $ref", "issue #3925 reproduction"])
def test_google_genai_refuses_those_schemas_unprojected(case):
    """Why the projection exists: sent as given, these fail the request."""
    pydantic = pytest.importorskip("pydantic")
    with pytest.raises(pydantic.ValidationError):
        _validate_declaration(SCHEMA_CORPUS[case])


def _real_genai_client(vertex: bool, handler):
    genai = pytest.importorskip("google.genai")
    types = pytest.importorskip("google.genai.types")
    if "httpx_async_client" not in types.HttpOptions.model_fields:
        pytest.skip("installed google-genai cannot take an httpx client")
    http_options = types.HttpOptions(
        httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    if not vertex:
        return genai.Client(api_key="test-key", http_options=http_options)
    oauth2 = pytest.importorskip("google.oauth2.credentials")
    return genai.Client(
        vertexai=True, project="test-project", location="us-central1",
        # A static token: nothing refreshes it, so nothing leaves the process.
        credentials=oauth2.Credentials(token="test-token"),
        http_options=http_options,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", [GoogleAdapter, VertexAIAdapter])
async def test_the_real_sdk_sends_the_whole_corpus_in_one_request(adapter_type):
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "finishReason": "STOP",
        }]})

    tools = [
        {**_definition(schema), "function": {
            **_definition(schema)["function"], "name": f"tool_{index}",
        }}
        for index, schema in enumerate(SCHEMA_CORPUS.values())
    ]
    client = _real_genai_client(adapter_type is VertexAIAdapter, handler)

    await adapter_type().get_response(
        client=client, model="gemini-2.5-flash", messages=MESSAGES, tools=tools,
        max_tokens=256,
    )

    (body,) = sent
    (wire_tool,) = body["tools"]
    declarations = {d["name"]: d["parameters"] for d in wire_tool["functionDeclarations"]}
    assert len(declarations) == len(SCHEMA_CORPUS)
    note = declarations[f"tool_{CASES.index('nullable type')}"]["properties"]["note"]
    assert note == {"type": "STRING", "description": "Optional note", "nullable": True}
    home = declarations[f"tool_{CASES.index('local $ref')}"]["properties"]["home"]
    assert home == {
        "type": "OBJECT", "properties": {"city": {"type": "STRING"}}, "required": ["city"],
    }
    closed = declarations[f"tool_{CASES.index('closed objects')}"]
    assert _wire_field(closed, "additionalProperties") is False
    assert _wire_field(closed["properties"]["record"], "additionalProperties") is False
    target = declarations[f"tool_{CASES.index('anyOf')}"]["properties"]["target"]
    assert _wire_field(target, "anyOf") == [{"type": "STRING"}, {"type": "INTEGER"}]


def _wire_field(schema: dict, camel_name: str):
    """A field google-genai may write by its proto name: it sends some nested
    ``Schema`` fields as ``any_of``/``additional_properties``, which proto3
    JSON accepts as well as the camelCase name."""
    snake_name = "".join(f"_{c.lower()}" if c.isupper() else c for c in camel_name)
    (present,) = [name for name in (camel_name, snake_name) if name in schema]
    return schema[present]


# --- the Gemini projection itself ---------------------------------------------


def test_a_schema_inside_the_dialect_comes_back_equal_and_unshared():
    schema = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "Query", "default": "x"},
            "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
            "opts": {"type": "object", "properties": {}, "additionalProperties": {"type": "integer"}},
        },
        "required": ["q"],
    }

    projected = gemini_parameters(schema, tool_name=TOOL_NAME)

    assert projected == schema
    projected["properties"]["tags"]["items"]["enum"].append("c")
    projected["required"].append("tags")
    assert schema["properties"]["tags"]["items"]["enum"] == ["a", "b"]
    assert schema["required"] == ["q"]


def test_a_type_list_naming_several_types_becomes_an_any_of():
    schema = {"type": ["string", "array", "null"], "description": "One or many",
              "items": {"type": "string"}, "maxLength": 9}

    assert gemini_parameters(schema, tool_name=TOOL_NAME) == {
        "description": "One or many",
        "anyOf": [
            {"type": "string", "maxLength": 9},
            {"type": "array", "items": {"type": "string"}},
        ],
        "nullable": True,
    }


@pytest.mark.parametrize(
    "members, expected",
    [
        (["string", "string"], {"type": "string"}),
        (["string", "null", "string"], {"type": "string", "nullable": True}),
        (["integer", "string", "integer"], {"anyOf": [{"type": "integer"}, {"type": "string"}]}),
    ],
)
def test_a_repeated_type_is_one_type(members, expected):
    assert gemini_parameters({"type": members}, tool_name=TOOL_NAME) == expected


def test_a_repeated_type_at_every_level_projects_linearly():
    """A branch per repeat copied the level below into each, doubling the
    result at every level: 40 levels, which registration accepts, made 2**40
    dicts. Twelve make the difference plain while staying fast."""
    schema: dict = {}
    for _ in range(12):
        schema = {"type": ["array", "array"], "items": schema}
    expected: dict = {}
    for _ in range(12):
        expected = {"type": "array", "items": expected}

    assert gemini_parameters(schema, tool_name=TOOL_NAME) == expected


def test_the_any_of_branches_share_no_value():
    """``number`` and ``integer`` both take ``enum``: each branch has its own."""
    projected = gemini_parameters(
        {"type": ["number", "integer"], "enum": ["1", "2"]}, tool_name=TOOL_NAME
    )

    first, second = projected["anyOf"]
    assert first == {"type": "number", "enum": ["1", "2"]}
    assert second == {"type": "integer", "enum": ["1", "2"]}
    first["enum"].append("3")
    assert second["enum"] == ["1", "2"]


def test_the_keywords_gemini_cannot_express_are_weakened():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "kind": {"const": "record"},
            "either": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            "level": {"type": "integer", "enum": [1, 2]},
            "state": {"type": "string", "enum": ["on", "off", None]},
            "all": {"allOf": [{"type": "string"}], "type": "string"},
            "anything": True,
            "never": False,
            "map": {"type": "object", "patternProperties": {"^x": {"type": "string"}}},
        },
        "examples": [{"kind": "record"}],
    }

    assert gemini_parameters(schema, tool_name=TOOL_NAME) == {
        "type": "object",
        "properties": {
            "kind": {"enum": ["record"]},
            "either": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "level": {"type": "integer"},
            "state": {"type": "string", "enum": ["on", "off"], "nullable": True},
            "all": {"type": "string"},
            "anything": {},
            "map": {"type": "object"},
        },
    }
    _validate_declaration(gemini_parameters(schema, tool_name=TOOL_NAME))


def test_a_reference_that_cannot_be_inlined_becomes_any_value(monkeypatch):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("a schema reference opened a connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    schema = {
        "type": "object",
        "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
        "properties": {
            "remote": {"$ref": "https://example.com/schema.json", "description": "Remote"},
            "missing": {"$ref": "#/$defs/Absent"},
            "anchor": {"$ref": "#node"},
            "list": {"$ref": "#/$defs/Node"},
        },
    }

    assert gemini_parameters(schema, tool_name=TOOL_NAME) == {
        "type": "object",
        "properties": {
            "remote": {"description": "Remote"},
            "missing": {},
            "anchor": {},
            "list": {"type": "object", "properties": {"next": {}}},
        },
    }


def test_reference_expansion_is_bounded():
    """Each level references the next twice: inlining it whole would double
    the schema at every level."""
    depth = 40
    defs = {f"L{i}": {"type": "object", "properties": {
        "a": {"$ref": f"#/$defs/L{i + 1}"}, "b": {"$ref": f"#/$defs/L{i + 1}"},
    }} for i in range(depth)}
    defs[f"L{depth}"] = {"type": "string"}
    schema = {"type": "object", "$defs": defs, "properties": {"root": {"$ref": "#/$defs/L0"}}}

    projected = gemini_parameters(schema, tool_name=TOOL_NAME)

    expansions = json.dumps(projected).count('"properties"') - 1
    assert expansions <= MAX_REFERENCE_EXPANSIONS
    _validate_declaration(projected)


def test_a_json_pointer_is_unescaped():
    schema = {
        "type": "object",
        "$defs": {"a/b": {"type": "string"}, "c~d": {"type": "integer"}, "e f": {"type": "boolean"}},
        "properties": {
            "slash": {"$ref": "#/$defs/a~1b"},
            "tilde": {"$ref": "#/$defs/c~0d"},
            "space": {"$ref": "#/$defs/e%20f"},
        },
    }

    assert gemini_parameters(schema, tool_name=TOOL_NAME)["properties"] == {
        "slash": {"type": "string"}, "tilde": {"type": "integer"}, "space": {"type": "boolean"},
    }


def test_the_whole_fragment_is_percent_decoded_before_it_is_split():
    """RFC 6901, section 6: a ``$ref`` fragment is percent-decoded, then read
    as a JSON pointer, then each token's ``~1``/``~0`` is unescaped. Decoding
    each token after splitting made ``%2F`` part of a name."""
    schema = {
        "type": "object",
        "$defs": {
            # Where the review's reference went: the wrong schema.
            "a/$defs/b": {"type": "string"},
            "a": {"$defs": {"b": {"type": "integer"}}},
            "Record": {"type": "boolean"},
            "c/d": {"type": "number"},
            "e%2Ff": {"type": "array", "items": {"type": "string"}},
        },
        "properties": {
            "encoded separators": {"$ref": "#/$defs/a%2F$defs%2Fb"},
            "fully encoded": {"$ref": "#%2F$defs%2FRecord"},
            "encoded escape": {"$ref": "#/$defs/c%7E1d"},
            "escaped name": {"$ref": "#/$defs/c~1d"},
            "decoded once": {"$ref": "#/$defs/e%252Ff"},
        },
    }

    assert gemini_parameters(schema, tool_name=TOOL_NAME)["properties"] == {
        "encoded separators": {"type": "integer"},
        "fully encoded": {"type": "boolean"},
        "encoded escape": {"type": "number"},
        "escaped name": {"type": "number"},
        "decoded once": {"type": "array", "items": {"type": "string"}},
    }


def test_a_pointer_that_names_nothing_is_any_value():
    schema = {
        "type": "object",
        "$defs": {"\ufffd": {"type": "string"}, "list": [{"type": "integer"}]},
        "properties": {
            # Invalid UTF-8 is not the replacement character.
            "invalid utf-8": {"$ref": "#/$defs/%FF"},
            "index": {"$ref": "#/$defs/list/0"},
            "leading zero": {"$ref": "#/$defs/list/00"},
            # ``"²".isdigit()``, but ``int("²")`` raises.
            "superscript digit": {"$ref": "#/$defs/list/²"},
            "past the end": {"$ref": "#/$defs/list/1"},
            "plain-name anchor": {"$ref": "#%41"},
        },
    }

    assert gemini_parameters(schema, tool_name=TOOL_NAME)["properties"] == {
        "invalid utf-8": {},
        "index": {"type": "integer"},
        "leading zero": {},
        "superscript digit": {},
        "past the end": {},
        "plain-name anchor": {},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("run", [_google, _vertex], ids=["google", "vertex"])
async def test_an_index_too_long_to_convert_names_nothing(run):
    """``int`` refuses a string of more than 4300 digits with ``ValueError``,
    which failed every Gemini turn the tool was offered on."""
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"$ref": "#/required/" + "1" * 4301, "description": "A"}},
    }

    sent = await run(schema)

    assert sent == {"type": "object", "required": ["a"], "properties": {"a": {"description": "A"}}}
    _validate_declaration(sent)


def test_a_reference_to_the_true_schema_is_any_value_and_loses_nothing(caplog):
    schema = {
        "type": "object",
        "$defs": {"Anything": True},
        "properties": {"x": {"$ref": "#/$defs/Anything", "description": "Any"}},
    }

    with caplog.at_level(logging.WARNING, logger=GEMINI_SCHEMA_LOGGER):
        projected = gemini_parameters(schema, tool_name=TOOL_NAME)

    assert projected["properties"] == {"x": {"description": "Any"}}
    assert _weakening_warnings(caplog) == []


def test_a_cycle_is_caught_however_its_reference_is_spelled():
    schema = {
        "type": "object",
        "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/%24defs/Node"}}}},
        "properties": {"list": {"$ref": "#/$defs/Node"}},
    }

    assert gemini_parameters(schema, tool_name=TOOL_NAME)["properties"] == {
        "list": {"type": "object", "properties": {"next": {}}},
    }


# --- a local reference is resolved only in a one-resource schema -------------


def _embedded_resource(identifier) -> dict:
    """The review's reproduction: the document and a subschema with its own
    ``$id`` each define ``Value``, as different types."""
    return {
        "type": "object",
        "$defs": {"Value": {"type": "string"}},
        "properties": {
            "outer": {"$ref": "#/$defs/Value"},
            "embedded": {
                "$id": identifier,
                "type": "object",
                "$defs": {"Value": {"type": "integer"}},
                "properties": {"inner": {"$ref": "#/$defs/Value"}},
            },
        },
    }


#: Every reference in a schema with more than one resource is any value.
_EMBEDDED_RESOURCE_SENT = {
    "outer": {},
    "embedded": {"type": "object", "properties": {"inner": {}}},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("run", [_google, _vertex], ids=["google", "vertex"])
async def test_conflicting_outer_and_embedded_definitions_reach_the_request_unconstrained(
    run, caplog,
):
    """Resolving every fragment against the document root sent ``inner`` as
    the outer ``string``, and warned of nothing. No reference in a schema
    with an embedded resource is resolved: each is any value, with a
    warning."""
    with caplog.at_level(logging.WARNING, logger=GEMINI_SCHEMA_LOGGER):
        sent = await run(_embedded_resource("https://example.com/schemas/embedded"))

    assert sent["properties"] == _EMBEDDED_RESOURCE_SENT
    _validate_declaration(sent)
    (warning,) = _weakening_warnings(caplog)
    assert repr(TOOL_NAME) in warning
    assert warning.endswith(f": {MULTI_RESOURCE_REFERENCE}: 2")


def _resolvable() -> dict:
    """A schema whose one reference resolves, to which each case adds."""
    return {
        "type": "object",
        "$defs": {"Value": {"type": "string"}},
        "properties": {"value": {"$ref": "#/$defs/Value"}},
    }


def _adding(path: tuple, key: str, value) -> dict:
    schema = _resolvable()
    node = schema
    for step in path:
        node = node.setdefault(step, {})
    node[key] = value
    return schema


#: What makes a schema more than one resource, wherever it sits.
MORE_THAN_ONE_RESOURCE = {
    "a $id below the root": _adding(("properties", "other"), "$id", "other.json"),
    "a $id that is a fragment": _adding(("properties", "other"), "$id", "#other"),
    "an empty $id": _adding(("properties", "other"), "$id", ""),
    "a $id in a definition": _adding(("$defs", "Value"), "$id", "value.json"),
    "a $id under a default": _adding(("properties", "other", "default"), "$id", "x"),
    "draft-04's id": _adding(("properties", "other"), "id", "other.json"),
    "an $anchor": _adding(("properties", "other"), "$anchor", "other"),
    "an $anchor at the root": _adding((), "$anchor", "root"),
    "a $dynamicAnchor": _adding((), "$dynamicAnchor", "meta"),
    "a $dynamicRef": _adding(("properties", "other"), "$dynamicRef", "#meta"),
    "a $recursiveAnchor": _adding((), "$recursiveAnchor", True),
    "a $recursiveRef": _adding(("properties", "other"), "$recursiveRef", "#"),
}


@pytest.mark.parametrize(
    "schema", list(MORE_THAN_ONE_RESOURCE.values()), ids=list(MORE_THAN_ONE_RESOURCE)
)
def test_no_reference_is_resolved_in_a_schema_of_more_than_one_resource(schema, caplog):
    with caplog.at_level(logging.WARNING, logger=GEMINI_SCHEMA_LOGGER):
        projected = gemini_parameters(schema, tool_name=TOOL_NAME)

    assert projected["properties"]["value"] == {}
    (warning,) = _weakening_warnings(caplog)
    assert warning.endswith(f": {MULTI_RESOURCE_REFERENCE}: 1")


#: Schemas that stay one resource.
ONE_RESOURCE = {
    "a $id at the root": _adding((), "$id", "https://example.com/root.json"),
    "draft-04's id at the root": _adding((), "id", "https://example.com/root.json"),
    "a property named $id": _adding(("properties",), "$id", {"type": "boolean"}),
    "a property named id": _adding(("properties",), "id", {"type": "string"}),
    "a definition named $id": _adding(("$defs",), "$id", {"type": "boolean"}),
    "a $id that is not a string": _adding(("properties", "other"), "$id", 7),
}


@pytest.mark.parametrize("schema", list(ONE_RESOURCE.values()), ids=list(ONE_RESOURCE))
def test_a_reference_in_a_one_resource_schema_is_resolved(schema, caplog):
    with caplog.at_level(logging.WARNING, logger=GEMINI_SCHEMA_LOGGER):
        projected = gemini_parameters(schema, tool_name=TOOL_NAME)

    assert projected["properties"]["value"] == {"type": "string"}
    assert _weakening_warnings(caplog) == []


def test_a_pointer_may_name_a_property_called_id():
    schema = {
        "type": "object",
        "$defs": {"Leaf": {"type": "number"}},
        "properties": {
            "$id": {"type": "string"},
            "id": {"$ref": "#/$defs/Leaf"},
            "id property": {"$ref": "#/properties/$id"},
        },
    }

    assert gemini_parameters(schema, tool_name=TOOL_NAME)["properties"] == {
        "$id": {"type": "string"},
        "id": {"type": "number"},
        "id property": {"type": "string"},
    }


# --- beside a reference, only annotations are kept ---------------------------


_CLOSED = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "additionalProperties": False,
}


def test_constraints_beside_a_reference_are_dropped():
    """Laid over the referenced schema, the sibling ``properties`` replaced
    the referenced ones and left ``additionalProperties: false``: ``{"a":
    "x"}``, which the schema admits, would have been refused. Annotations
    describe the value and stay."""
    schema = {
        "type": "object",
        "$defs": {"Closed": _CLOSED},
        "properties": {"x": {
            "$ref": "#/$defs/Closed",
            "properties": {"b": {"type": "integer"}},
            "required": ["b"],
            "type": "string",
            "title": "X",
            "description": "An x",
            "default": {"a": "y"},
            "example": {"a": "z"},
        }},
    }

    assert gemini_parameters(schema, tool_name=TOOL_NAME)["properties"]["x"] == {
        **_CLOSED,
        "title": "X",
        "description": "An x",
        "default": {"a": "y"},
        "example": {"a": "z"},
    }


def test_constraints_beside_an_unresolved_reference_are_dropped():
    schema = {"type": "object", "properties": {
        "x": {"$ref": "#/$defs/Absent", "type": "string", "description": "An x"},
    }}

    assert gemini_parameters(schema, tool_name=TOOL_NAME)["properties"]["x"] == {
        "description": "An x",
    }


# --- the projection's recursion is bounded ------------------------------------


def _reference_chain(length: int) -> dict:
    """The review's reproduction: ``length`` definitions, each referencing
    the next through an object property. Shallow as written, and inside the
    expansion budget, yet as deep as it is long once inlined."""
    defs = {
        f"D{i}": {"type": "object", "properties": {"next": {"$ref": f"#/$defs/D{i + 1}"}}}
        for i in range(length)
    }
    defs[f"D{length}"] = {"type": "string"}
    return {"type": "object", "$defs": defs, "properties": {"root": {"$ref": "#/$defs/D0"}}}


def _chain_links(parameters: dict) -> int:
    """How many ``D`` links the projected chain kept before it became ``{}``."""
    links = 0
    node = parameters["properties"]["root"]
    while node != {}:
        assert node == {"type": "object", "properties": {"next": node["properties"]["next"]}}
        node = node["properties"]["next"]
        links += 1
    return links


def _schema_depth(schema: dict) -> int:
    """How deep subschemas nest in a projected schema, the root being 1."""
    deepest = 0
    pending = [(schema, 1)]
    while pending:
        node, depth = pending.pop()
        deepest = max(deepest, depth)
        children = list(node.get("properties", {}).values()) + node.get("anyOf", [])
        children += [node[key] for key in ("items", "additionalProperties")
                     if isinstance(node.get(key), dict)]
        pending.extend((child, depth + 1) for child in children)
    return deepest


GEMINI_SCHEMA_LOGGER = "kestrel_sovereign.llm.gemini_schema"


def _weakening_warnings(caplog) -> list:
    return [
        record.getMessage() for record in caplog.records
        if record.name == GEMINI_SCHEMA_LOGGER and record.levelno == logging.WARNING
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("run", [_google, _vertex], ids=["google", "vertex"])
async def test_a_long_reference_chain_reaches_the_request_bounded(run, caplog):
    """Without a depth bound this raised RecursionError before the SDK call,
    failing every turn the tool was offered on."""
    with caplog.at_level(logging.WARNING, logger=GEMINI_SCHEMA_LOGGER):
        sent = await run(_reference_chain(MAX_REFERENCE_EXPANSIONS))

    assert _schema_depth(sent) <= MAX_PROJECTION_DEPTH
    # Truncated, not collapsed: the chain is inlined until the bound.
    assert _chain_links(sent) == (MAX_PROJECTION_DEPTH - 1) // 2
    _validate_declaration(sent)
    (warning,) = _weakening_warnings(caplog)
    assert repr(TOOL_NAME) in warning
    assert warning.endswith(f": {SUBSCHEMA_PAST_DEPTH}: 1")


@pytest.mark.parametrize("last, warned", [({}, False), ({"type": "string"}, True)])
def test_the_depth_bound_warns_only_when_it_cuts_a_constraint(last, warned, caplog):
    """Chain link ``D31`` is inlined one level past the bound: ``{}`` there
    loses nothing, a ``string`` does."""
    schema = _reference_chain(31)
    schema["$defs"]["D31"] = last

    with caplog.at_level(logging.WARNING, logger=GEMINI_SCHEMA_LOGGER):
        projected = gemini_parameters(schema, tool_name=TOOL_NAME)

    assert _chain_links(projected) == 31
    assert len(_weakening_warnings(caplog)) == (1 if warned else 0)


_TREE = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "children": {"type": "array", "items": {"$ref": "#"}},
    },
    "required": ["name"],
}

#: Schemas whose ``$ref`` re-enters itself, and what Gemini is sent: each is
#: inlined until the reference recurs, which becomes any value.
SELF_RECURSIVE = {
    "a tree whose items are the root": (
        _TREE,
        {
            **_TREE,
            "properties": {
                **_TREE["properties"],
                "children": {"type": "array", "items": {
                    **_TREE,
                    "properties": {
                        **_TREE["properties"],
                        "children": {"type": "array", "items": {}},
                    },
                }},
            },
        },
    ),
    "a definition that is its own reference": (
        {
            "type": "object",
            "$defs": {"Loop": {"$ref": "#/$defs/Loop"}},
            "properties": {"x": {"$ref": "#/$defs/Loop", "description": "Loops"}},
        },
        {"type": "object", "properties": {"x": {"description": "Loops"}}},
    ),
    # Beside a ``$ref`` only annotations are kept: the root's own
    # ``properties`` are a constraint laid beside its reference to itself.
    "a root that references itself": (
        {"$ref": "#", "type": "object", "description": "Loops",
         "properties": {"q": {"type": "string"}}},
        {"description": "Loops"},
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(SELF_RECURSIVE))
@pytest.mark.parametrize("run", [_google, _vertex], ids=["google", "vertex"])
async def test_a_self_recursive_reference_reaches_the_request_weakened(run, case, caplog):
    """Registration accepts these (``_definition`` registers the tool), so
    projecting one must not raise: it would fail every Gemini turn the tool
    is offered on, whichever tool the turn uses."""
    schema, expected = SELF_RECURSIVE[case]

    with caplog.at_level(logging.WARNING, logger=GEMINI_SCHEMA_LOGGER):
        sent = await run(copy.deepcopy(schema))

    assert sent == expected
    _validate_declaration(sent)
    (warning,) = _weakening_warnings(caplog)
    assert repr(TOOL_NAME) in warning
    assert warning.endswith(f": {RECURSIVE_REFERENCE}: 1")


#: ``type`` lists registration accepts, which the projection raised
#: ``TypeError`` on (an unhashable member) or turned into an ``anyOf`` the SDK
#: refuses (a number member).
MALFORMED_TYPE_LISTS = {
    "a list member": [["string"], "null"],
    "an object member": [{"type": "string"}, "string"],
    "a number member": [1, "string"],
    "a name that is not a JSON type": ["string", "text"],
    "an empty list": [],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("run", [_google, _vertex], ids=["google", "vertex"])
async def test_a_type_list_with_a_member_that_is_not_a_string_is_dropped(run):
    schema = {"type": "object", "properties": {
        name: {"type": members, "description": name}
        for name, members in MALFORMED_TYPE_LISTS.items()
    }}

    sent = await run(schema)

    assert sent == {"type": "object", "properties": {
        name: {"description": name} for name in MALFORMED_TYPE_LISTS
    }}
    _validate_declaration(sent)


#: Values registration accepts (they are JSON) for a keyword ``Schema``
#: copies, which google-genai refuses, failing every Gemini turn the tool is
#: offered on, or which no JSON Schema allows (a count past int64, an empty
#: ``enum``). Each is dropped.
MALFORMED_VALUES = {
    "a description that is a number": {"description": 5},
    "a title that is a list": {"title": ["x"]},
    "a pattern that is a number": {"type": "string", "pattern": 5},
    "a format that is an object": {"type": "string", "format": {"x": 1}},
    "a nullable that is a string": {"type": "string", "nullable": "x"},
    "a minimum that is a string": {"type": "number", "minimum": "x"},
    "a maximum too large for a float": {"type": "number", "maximum": 10**400},
    "a minimum that is a flag": {"type": "number", "minimum": True},
    "a fractional minLength": {"type": "string", "minLength": 2.5},
    "a negative minLength": {"type": "string", "minLength": -1},
    "a minLength that is a flag": {"type": "string", "minLength": True},
    "a maxItems past int64": {"type": "array", "maxItems": 2**63},
    "a maxProperties that is a string": {"type": "object", "maxProperties": "3"},
    "a required that is a string": {"type": "object", "required": "a"},
    "a required naming a number": {"type": "object", "required": [1]},
    "a propertyOrdering that is a string": {"type": "object", "propertyOrdering": "a"},
    "a type that is a number": {"type": 5},
    "a type that is not a JSON type": {"type": "text"},
    "an empty enum": {"type": "string", "enum": []},
    "an empty anyOf": {"anyOf": []},
    "an empty oneOf": {"oneOf": []},
}

#: Where only a native call can make a value JSON cannot spell: an isolated
#: tool's is refused at registration.
NOT_JSON_VALUES = {
    "a NaN minimum": {"type": "number", "minimum": float("nan")},
    "an infinite maximum": {"type": "number", "maximum": float("inf")},
    "a NaN default": {"type": "number", "default": float("nan")},
    "an infinity inside an example": {"type": "array", "example": [1, [float("-inf")]]},
}


def _with_each(values: dict) -> dict:
    return {"type": "object", "properties": {
        name: {**value, "title": name} if "title" not in value else value
        for name, value in values.items()
    }}


def test_google_genai_refuses_values_of_the_wrong_shape_unprojected():
    """Why the projection checks them: sent as given, each fails the request."""
    pydantic = pytest.importorskip("pydantic")
    for name in (
        "a description that is a number", "a fractional minLength",
        "a required that is a string", "a type that is a number",
        "a maximum too large for a float",
    ):
        with pytest.raises(pydantic.ValidationError):
            _validate_declaration({"type": "object", "properties": {"x": MALFORMED_VALUES[name]}})


@pytest.mark.asyncio
@pytest.mark.parametrize("run", [_google, _vertex], ids=["google", "vertex"])
async def test_a_value_of_the_wrong_shape_is_dropped(run):
    sent = await run(_with_each(MALFORMED_VALUES))

    expected = {}
    for name, value in MALFORMED_VALUES.items():
        kept = {"title": name}
        if value.get("type") in ("string", "number", "array", "object"):
            kept["type"] = value["type"]
        expected[name] = kept
    expected["a title that is a list"] = {}
    assert sent == {"type": "object", "properties": expected}
    _validate_declaration(sent)


def test_an_integer_bound_a_float_cannot_hold_is_dropped():
    """google-genai's bound fields are floats: ``2**53 + 1`` arrives as
    ``2**53``, refusing a value the schema admits."""
    schema = {"type": "object", "properties": {
        "rounded maximum": {"type": "integer", "maximum": 2**53 + 1},
        "rounded minimum": {"type": "integer", "minimum": -(2**53) - 1},
        "exact": {"type": "integer", "minimum": 2**53, "maximum": 2**60},
    }}

    assert gemini_parameters(schema, tool_name=TOOL_NAME)["properties"] == {
        "rounded maximum": {"type": "integer"},
        "rounded minimum": {"type": "integer"},
        "exact": {"type": "integer", "minimum": 2**53, "maximum": 2**60},
    }


def test_a_number_json_cannot_spell_is_dropped():
    projected = gemini_parameters(_with_each(NOT_JSON_VALUES), tool_name=TOOL_NAME)

    assert projected == {"type": "object", "properties": {
        name: {"type": value["type"], "title": name}
        for name, value in NOT_JSON_VALUES.items()
    }}


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", [GoogleAdapter, VertexAIAdapter])
async def test_the_real_sdk_sends_values_of_the_wrong_shape_as_valid_json(adapter_type):
    """The real google-genai validates and encodes the request: nothing it
    refuses reaches it, and its body is JSON a server can read."""
    sent = []

    def refuse_constant(constant):
        raise AssertionError(f"the request carries {constant}, which JSON cannot spell")

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content, parse_constant=refuse_constant))
        return httpx.Response(200, json={"candidates": [{
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "finishReason": "STOP",
        }]})

    parameters = _with_each({**MALFORMED_VALUES, **NOT_JSON_VALUES})
    client = _real_genai_client(adapter_type is VertexAIAdapter, handler)

    await adapter_type().get_response(
        client=client, model="gemini-2.5-flash", messages=MESSAGES, max_tokens=256,
        tools=[{"type": "function", "function": {
            "name": TOOL_NAME, "description": "Store a record", "parameters": parameters,
        }}],
    )

    (body,) = sent
    ((declaration,),) = [t["functionDeclarations"] for t in body["tools"]]
    assert set(declaration["parameters"]["properties"]) == set(parameters["properties"])


# --- keywords whose meaning depends on a partner Gemini drops -----------------


def test_additional_properties_beside_pattern_properties_is_dropped():
    """``patternProperties`` admits ``x-a``; ``additionalProperties: false``
    alone would refuse it."""
    schema = {"type": "object", "properties": {
        "map": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "patternProperties": {"^x-": {"type": "string"}},
            "additionalProperties": False,
        },
        "closed": {"type": "object", "additionalProperties": False},
        "malformed": {"type": "object", "properties": ["id"], "additionalProperties": False},
    }}

    assert gemini_parameters(schema, tool_name=TOOL_NAME)["properties"] == {
        "map": {"type": "object", "properties": {"id": {"type": "string"}}},
        "closed": {"type": "object", "additionalProperties": False},
        "malformed": {"type": "object"},
    }


def test_items_beside_prefix_items_is_dropped():
    """``items`` applies only past ``prefixItems``: alone, it would refuse
    ``["a", 1]``, which the schema admits."""
    schema = {"type": "object", "properties": {
        "pair": {
            "type": "array",
            "prefixItems": [{"type": "string"}],
            "items": {"type": "integer"},
            "maxItems": 2,
        },
        "list": {"type": "array", "items": {"type": "integer"}},
    }}

    assert gemini_parameters(schema, tool_name=TOOL_NAME)["properties"] == {
        "pair": {"type": "array", "maxItems": 2},
        "list": {"type": "array", "items": {"type": "integer"}},
    }


def test_a_keyword_that_applies_to_none_of_the_listed_types_is_dropped():
    """Left beside the branches, an object's ``properties`` could read as
    making a string-or-integer value an object."""
    schema = {
        "type": ["string", "integer"],
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "items": {"type": "string"},
        "maxLength": 3,
        "description": "A code",
    }

    assert gemini_parameters(schema, tool_name=TOOL_NAME) == {
        "description": "A code",
        "anyOf": [{"type": "string", "maxLength": 3}, {"type": "integer"}],
    }


def test_a_schema_the_dialect_can_take_whole_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger=GEMINI_SCHEMA_LOGGER):
        for schema in SCHEMA_CORPUS.values():
            gemini_parameters(schema, tool_name=TOOL_NAME)

    assert _weakening_warnings(caplog) == []


def test_one_warning_counts_every_weakening_and_quotes_no_schema_text(monkeypatch, caplog):
    from kestrel_sovereign.llm import gemini_schema

    monkeypatch.setattr(gemini_schema, "MAX_REFERENCE_EXPANSIONS", 3)
    deep_value: list = []
    for _ in range(100):
        deep_value = [deep_value]
    schema = {
        "type": "object",
        "description": "private-text",
        "$defs": {
            "Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}},
            "Leaf": {"type": "string", "description": "private-text"},
        },
        "properties": {
            "recursive": {"$ref": "#/$defs/Node"},
            "missing": {"$ref": "#/$defs/private-text"},
            "first": {"$ref": "#/$defs/Leaf"},
            "second": {"$ref": "#/$defs/Leaf"},
            "past the budget": {"$ref": "#/$defs/Leaf"},
            "deep": _nested_items(MAX_PROJECTION_DEPTH + 5),
            "deep default": {"type": "array", "default": deep_value},
        },
    }

    with caplog.at_level(logging.WARNING, logger=GEMINI_SCHEMA_LOGGER):
        gemini_parameters(schema, tool_name=TOOL_NAME)

    (warning,) = _weakening_warnings(caplog)
    assert warning == (
        f"Gemini is offered weaker parameters for tool {TOOL_NAME!r} than it declares: "
        f"{RECURSIVE_REFERENCE}: 1; {UNRESOLVED_REFERENCE}: 1; "
        f"{REFERENCE_PAST_BUDGET}: 1; {SUBSCHEMA_PAST_DEPTH}: 1; {VALUE_PAST_DEPTH}: 1"
    )
    assert "private-text" not in warning


def test_the_warning_quotes_at_most_200_characters_of_the_tool_name(caplog):
    with caplog.at_level(logging.WARNING, logger=GEMINI_SCHEMA_LOGGER):
        gemini_parameters({"$ref": "#/$defs/Absent"}, tool_name="n" * 500)

    (warning,) = _weakening_warnings(caplog)
    assert repr("n" * 200) in warning
    assert "n" * 201 not in warning


def _nested_items(depth: int) -> dict:
    """A schema whose subschemas nest ``depth`` deep, the root being 1."""
    node: dict = {}
    for _ in range(depth - 1):
        node = {"type": "array", "items": node}
    return node


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", [GoogleAdapter, VertexAIAdapter])
async def test_the_real_sdk_sends_a_long_reference_chain(adapter_type):
    """google-genai's own schema processing recurses too: the bound leaves
    it room."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "finishReason": "STOP",
        }]})

    client = _real_genai_client(adapter_type is VertexAIAdapter, handler)

    await adapter_type().get_response(
        client=client, model="gemini-2.5-flash", messages=MESSAGES,
        tools=[_definition(_reference_chain(MAX_REFERENCE_EXPANSIONS))], max_tokens=256,
    )

    (body,) = sent
    ((declaration,),) = [t["functionDeclarations"] for t in body["tools"]]
    node, links = declaration["parameters"]["properties"]["root"], 0
    while node.get("properties"):
        node, links = node["properties"]["next"], links + 1
    assert links == (MAX_PROJECTION_DEPTH - 1) // 2


def test_a_deep_schema_is_cut_off_at_the_bound():
    """A schema from anywhere, not only an isolated service, may be deep."""
    schema: dict = {"type": "string"}
    for _ in range(1000):
        schema = {"type": "array", "items": schema}

    projected = gemini_parameters(schema, tool_name=TOOL_NAME)

    node, depth = projected, 1
    while node != {}:
        assert node["type"] == "array"
        node, depth = node["items"], depth + 1
    # The first subschema past the bound is any value.
    assert depth == MAX_PROJECTION_DEPTH + 1
    _validate_declaration(projected)


def test_data_nested_deeper_than_the_depth_left_is_dropped():
    deep: list = []
    for _ in range(1000):
        deep = [deep]
    schema = {
        "type": "object",
        "properties": {
            "deep": {"type": "array", "default": deep, "description": "kept"},
            "shallow": {"type": "array", "default": [[1]]},
        },
        "required": ["deep"],
    }

    assert gemini_parameters(schema, tool_name=TOOL_NAME) == {
        "type": "object",
        "properties": {
            "deep": {"type": "array", "description": "kept"},
            "shallow": {"type": "array", "default": [[1]]},
        },
        "required": ["deep"],
    }


@pytest.mark.parametrize("value", [None, "not a schema", []])
def test_parameters_that_are_not_an_object_are_left_for_the_sdk(value):
    assert gemini_parameters(value, tool_name=TOOL_NAME) is value
