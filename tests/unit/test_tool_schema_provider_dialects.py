"""What each provider adapter sends for an isolated tool's own schema (#3925).

An isolated tool's ``function.parameters`` is now its service's JSON Schema,
whole. Each provider adapter still converts tool definitions into its own
request, and where a provider cannot express a keyword, that conversion is
the one place the schema is weakened. These tests pin, for every schema in
:data:`tests.utils.tool_schema_corpus.SCHEMA_CORPUS`, the parameters each
adapter puts on its request:

* OpenAI (and OpenRouter, its subclass), Anthropic (and Claude Max) and Codex
  send the schema verbatim;
* Ollama's SDK models keep only what they declare, so nested ``properties``,
  ``additionalProperties``, ``anyOf`` and ``$ref`` do not reach Ollama;
* the Google and Vertex adapters project it into google-genai's ``Schema``
  (:func:`kestrel_sovereign.llm.gemini_schema.gemini_parameters`), which the
  SDK refuses the whole request without: a ``type`` list becomes ``nullable``
  and a local ``$ref`` is inlined.

Nothing leaves the process: every SDK runs over a fake client or a mock
transport.
"""

from __future__ import annotations

import copy
import json
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


# --- the Gemini projection is what google-genai accepts -----------------------


def _validate_declaration(parameters: dict) -> None:
    types = pytest.importorskip("google.genai.types")
    types.GenerateContentConfig.model_validate({"tools": [{"function_declarations": [
        {"name": TOOL_NAME, "description": "Store a record", "parameters": parameters},
    ]}]})


@pytest.mark.parametrize("case", CASES)
def test_google_genai_accepts_every_projected_schema(case):
    _validate_declaration(gemini_parameters(SCHEMA_CORPUS[case]))


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

    projected = gemini_parameters(schema)

    assert projected == schema
    projected["properties"]["tags"]["items"]["enum"].append("c")
    projected["required"].append("tags")
    assert schema["properties"]["tags"]["items"]["enum"] == ["a", "b"]
    assert schema["required"] == ["q"]


def test_a_type_list_naming_several_types_becomes_an_any_of():
    schema = {"type": ["string", "array", "null"], "description": "One or many",
              "items": {"type": "string"}, "maxLength": 9}

    assert gemini_parameters(schema) == {
        "description": "One or many",
        "anyOf": [
            {"type": "string", "maxLength": 9},
            {"type": "array", "items": {"type": "string"}},
        ],
        "nullable": True,
    }


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

    assert gemini_parameters(schema) == {
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
    _validate_declaration(gemini_parameters(schema))


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

    assert gemini_parameters(schema) == {
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

    projected = gemini_parameters(schema)

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

    assert gemini_parameters(schema)["properties"] == {
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

    assert gemini_parameters(schema)["properties"] == {
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
            "plain-name anchor": {"$ref": "#%41"},
        },
    }

    assert gemini_parameters(schema)["properties"] == {
        "invalid utf-8": {},
        "index": {"type": "integer"},
        "leading zero": {},
        "superscript digit": {},
        "plain-name anchor": {},
    }


def test_a_cycle_is_caught_however_its_reference_is_spelled():
    schema = {
        "type": "object",
        "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/%24defs/Node"}}}},
        "properties": {"list": {"$ref": "#/$defs/Node"}},
    }

    assert gemini_parameters(schema)["properties"] == {
        "list": {"type": "object", "properties": {"next": {}}},
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


@pytest.mark.asyncio
@pytest.mark.parametrize("run", [_google, _vertex], ids=["google", "vertex"])
async def test_a_long_reference_chain_reaches_the_request_bounded(run):
    """Without a depth bound this raised RecursionError before the SDK call,
    failing every turn the tool was offered on."""
    sent = await run(_reference_chain(MAX_REFERENCE_EXPANSIONS))

    assert _schema_depth(sent) <= MAX_PROJECTION_DEPTH
    # Truncated, not collapsed: the chain is inlined until the bound.
    assert _chain_links(sent) == (MAX_PROJECTION_DEPTH - 1) // 2
    _validate_declaration(sent)


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

    projected = gemini_parameters(schema)

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

    assert gemini_parameters(schema) == {
        "type": "object",
        "properties": {
            "deep": {"type": "array", "description": "kept"},
            "shallow": {"type": "array", "default": [[1]]},
        },
        "required": ["deep"],
    }


@pytest.mark.parametrize("value", [None, "not a schema", []])
def test_parameters_that_are_not_an_object_are_left_for_the_sdk(value):
    assert gemini_parameters(value) is value
