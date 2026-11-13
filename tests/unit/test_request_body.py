"""Native ASGI streamed JSON budgets precede private schema admission."""

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.requests import Request

from kestrel_sovereign.security.request_body import InvalidJSONRequest, read_json_model


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: str


def request(chunks, *, media="application/json"):
    remaining = list(chunks)

    async def receive():
        content = remaining.pop(0)
        return {"type": "http.request", "body": content, "more_body": bool(remaining)}

    return Request({"type": "http", "method": "POST", "path": "/owned",
        "headers": [(b"content-type", media.encode())]}, receive), remaining


@pytest.mark.asyncio
async def test_streamed_json_accepts_exact_budget_and_original_schema():
    r, remaining = request([b'{"decision":', b'"approve"}'])
    value = await read_json_model(r, Selection, max_bytes=len(b'{"decision":"approve"}'))
    assert value.decision == "approve"
    assert remaining == []


@pytest.mark.asyncio
async def test_streamed_json_rejects_over_budget_before_reading_the_rest():
    r, remaining = request([b'{"decision":', b'"too-long"}', b'private-tail'])
    with pytest.raises(InvalidJSONRequest, match="byte limit"):
        await read_json_model(r, Selection, max_bytes=12)
    assert remaining == [b'private-tail']


@pytest.mark.asyncio
async def test_wrong_media_type_does_not_read_private_body():
    r, remaining = request([b'private-body'], media="text/plain")
    with pytest.raises(InvalidJSONRequest, match="JSON request"):
        await read_json_model(r, Selection, max_bytes=30)
    assert remaining == [b'private-body']


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b'{', b'{"decision":1}', b'{"decision":"approve","extra":1}'])
async def test_original_schema_errors_remain_explicit(content):
    r, _remaining = request([content])
    with pytest.raises(ValidationError):
        await read_json_model(r, Selection, max_bytes=100)
