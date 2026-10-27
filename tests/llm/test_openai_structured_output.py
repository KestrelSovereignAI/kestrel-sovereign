"""Original OpenAI client serialization, with a scripted model transport.

These checks prove the three outbound adapter contracts, not provider service
acceptance or a live model's ability to choose a visual action.
"""
import json

import httpx
import openai
from pydantic import BaseModel, ConfigDict, ValidationError
import pytest

from kestrel_sdk.computer_use import Click, Key, Navigate, Scroll, TypeText, Wait
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.visual_computer.inference import VisualProposal


class NestedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str | None = None


class StructuredAnswer(BaseModel):
    results: list[NestedAnswer]
    explanation: str = ""


def _assert_strict_objects(value):
    if isinstance(value, dict):
        assert "oneOf" not in value and "discriminator" not in value
        if value.get("type") == "object":
            assert set(value["required"]) == set(value["properties"])
            assert value["additionalProperties"] is False
        for child in value.values():
            _assert_strict_objects(child)
    elif isinstance(value, list):
        for child in value:
            _assert_strict_objects(child)


@pytest.mark.parametrize("path", ["get_response", "get_streaming_response", "get_streaming_response_with_tools"])
@pytest.mark.parametrize("model", [VisualProposal, StructuredAnswer])
@pytest.mark.asyncio
async def test_original_openai_client_serializes_strict_nested_schema_on_every_door(path, model):
    posts = []
    content = '{"action":null,"answer":"done"}'

    async def server(request):
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        posts.append(body)
        common = {"id": "synthetic-completion", "created": 1, "model": "synthetic-vision"}
        if body.get("stream"):
            chunk = {**common, "object": "chat.completion.chunk", "choices": [
                {"index": 0, "finish_reason": "stop", "delta": {"content": content}}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                content=("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
        return httpx.Response(200, json={**common, "object": "chat.completion", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]})

    async with openai.AsyncOpenAI(api_key="synthetic-only", base_url="http://owned-model.test/v1",
        max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(server), trust_env=False)) as client:
        method = getattr(OpenAIAdapter(), path)
        result = method(client, "synthetic-vision", [{"role": "user", "content": "inspect"}], response_format=model)
        if path == "get_response":
            assert (await result).content == content
        else:
            assert [item async for item in result]
    assert client.is_closed()
    assert len(posts) == 1
    response_format = posts[0]["response_format"]
    assert response_format["type"] == "json_schema"
    definition = response_format["json_schema"]
    assert definition["strict"] is True and definition["name"] == model.__name__
    _assert_strict_objects(definition["schema"])
    if model is VisualProposal:
        variants = definition["schema"]["$defs"].values()
        assert {item["properties"]["kind"]["const"] for item in variants} == {
            "click", "type", "key", "scroll", "wait", "navigate"}
        assert all("kind" in item["required"] for item in variants)


@pytest.mark.parametrize("action", [
    {"kind": "click", "x": 10, "y": 20},
    {"kind": "type", "text": "hello"},
    {"kind": "key", "key": "Enter"},
    {"kind": "scroll", "delta_y": 10},
    {"kind": "wait", "milliseconds": 100},
    {"kind": "navigate", "url": "https://example.test/"},
    None,
])
def test_proposal_schema_does_not_change_canonical_sdk_models_or_parser(action):
    models = (Click, TypeText, Key, Scroll, Wait, Navigate)
    before = [model.model_json_schema() for model in models]
    proposal = VisualProposal.model_validate({"action": action})
    schema = VisualProposal.model_json_schema()
    assert [model.model_json_schema() for model in models] == before
    if action is not None:
        assert type(proposal.action) is next(model for model in models if model.model_fields["kind"].default == action["kind"])
    else:
        assert proposal.action is None
    assert all("kind" in variant["required"] for variant in schema["$defs"].values())


@pytest.mark.parametrize("value,error_type", [
    ({"action": {"x": 10, "y": 20}}, "union_tag_not_found"),
    ({"action": {"kind": "shell", "command": "unsafe"}}, "union_tag_invalid"),
    ({"action": {"kind": "click", "x": 10, "y": 20}, "owner": "model-selected"}, "extra_forbidden"),
    ({"action": {"kind": "click", "x": 10, "y": 20, "lease_id": "model-selected"}}, "extra_forbidden"),
])
def test_proposal_never_guesses_an_action_kind_or_accepts_model_authority(value, error_type):
    with pytest.raises(ValidationError) as error:
        VisualProposal.model_validate(value)
    assert error_type in {item["type"] for item in error.value.errors()}
