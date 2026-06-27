import io
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sdk.llm import (
    BatchRequest,
    CacheMarker,
    CodeExecOptions,
    ProviderCapabilities,
    RequestOptions,
    WebSearchOptions,
)

from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter


def _chat_response(content="ok"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
        ),
    )


def _client():
    client = SimpleNamespace()
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=AsyncMock(return_value=_chat_response()))
    )
    client.files = SimpleNamespace(
        create=AsyncMock(
            return_value=SimpleNamespace(
                id="file_1",
                filename="batch.jsonl",
                purpose="batch",
                bytes=10,
                created_at=1,
            )
        ),
        list=AsyncMock(return_value=SimpleNamespace(data=[])),
        retrieve=AsyncMock(return_value=SimpleNamespace(id="file_1")),
        delete=AsyncMock(return_value=SimpleNamespace(id="file_1", deleted=True)),
        content=AsyncMock(
            return_value=io.BytesIO(
                (
                    json.dumps(
                        {
                            "custom_id": "a",
                            "response": {
                                "body": {
                                    "choices": [
                                        {"message": {"content": "first"}}
                                    ],
                                    "usage": {
                                        "prompt_tokens": 1,
                                        "completion_tokens": 1,
                                        "total_tokens": 2,
                                    },
                                }
                            },
                        }
                    )
                    + "\n"
                ).encode("utf-8")
            )
        ),
    )
    client.batches = SimpleNamespace(
        create=AsyncMock(
            return_value=SimpleNamespace(
                id="batch_1",
                status="in_progress",
                created_at=1,
                expires_at=2,
                output_file_id="file_out",
            )
        ),
        retrieve=AsyncMock(
            return_value=SimpleNamespace(
                id="batch_1",
                status="completed",
                output_file_id="file_out",
            )
        ),
        cancel=AsyncMock(return_value=SimpleNamespace(id="batch_1", status="cancelled")),
    )
    client.responses = SimpleNamespace(create=AsyncMock(return_value={"id": "resp_1"}))
    return client


def test_openai_v5_capability_flags_and_round_trip():
    caps = OpenAIAdapter(native_openai=True).provider_capabilities()

    assert caps.supports_token_counting is True
    assert caps.supports_batch is True
    assert caps.supports_files is True
    assert caps.supports_prompt_cache is True
    assert caps.supports_reasoning_control is True
    assert caps.supports_raw_passthrough is True
    # web_search / code_execution are Responses/Assistants-API server tools, not
    # chat.completions features — this adapter does not advertise them.
    assert caps.supports_web_search is False
    assert caps.supports_code_execution is False
    assert caps.reasoning_control_mode.value == "effort"
    assert caps.batch_mode.value == "file_based"
    assert caps.files_mode.value == "upload"
    assert "responses.create" in caps.raw_operations
    assert ProviderCapabilities.from_mapping(caps.to_dict()) == caps


def test_openai_compatible_route_does_not_advertise_native_only_caps():
    # A compatible route (Kimi/DeepSeek/OpenRouter via custom base_url) is
    # constructed with supports_embeddings=False; it must NOT advertise the
    # OpenAI-only /batches, /files, /responses surface (would 404).
    compat = OpenAIAdapter(name="openrouter", supports_embeddings=False)
    caps = compat.provider_capabilities()

    assert caps.supports_batch is False
    assert caps.supports_files is False
    assert caps.supports_prompt_cache is False
    assert caps.supports_raw_passthrough is False
    assert caps.batch_mode.value == "none"
    assert caps.files_mode.value == "none"
    assert caps.raw_operations == ()
    # Endpoint-agnostic capabilities remain.
    assert caps.supports_token_counting is True
    assert caps.supports_reasoning_control is True

    features = compat.contract_features()
    assert "batch" not in features and "files" not in features
    assert "raw_passthrough" not in features
    assert "token_counting" in features


def test_openai_contract_features_match_advertised_optional_methods():
    features = OpenAIAdapter(native_openai=True).contract_features()

    assert {
        "token_counting",
        "batch",
        "files",
        "prompt_cache",
        "reasoning_control",
        "raw_passthrough",
    }.issubset(features)
    assert "server_tools" not in features


def test_apply_request_options_mutates_outbound_kwargs():
    adapter = OpenAIAdapter()
    kwargs = {"tools": [{"type": "function", "function": {"name": "x"}}]}
    options = RequestOptions(
        cache_markers=[CacheMarker(index=0, label="system")],
        reasoning_effort="high",
        web_search=WebSearchOptions(max_results=3, search_context_size="low"),
        code_execution=CodeExecOptions(container="auto", timeout_seconds=30),
        raw={"extra_body": {"custom": True}},
    )

    out = adapter.apply_request_options(kwargs, options, model="gpt-5")

    assert out is kwargs
    # reasoning_effort is forwarded directly; no top-level `reasoning` kwarg
    # (chat.completions rejects it).
    assert out["reasoning_effort"] == "high"
    assert "reasoning" not in out
    # Only prompt_cache_key is sent (chat.completions has no cache_markers field).
    assert "cache_markers" not in out["extra_body"]
    assert out["extra_body"]["prompt_cache_key"].startswith("kestrel:gpt-5:")
    assert out["extra_body"]["custom"] is True
    # web_search / code_execution options are ignored — not chat.completions
    # features. The only tools present are the caller's own function tools.
    assert out["tools"] == [{"type": "function", "function": {"name": "x"}}]
    assert "web_search_options" not in out
    assert "code_execution" not in out["extra_body"]


@pytest.mark.asyncio
async def test_get_response_applies_request_options_to_chat_completion():
    adapter = OpenAIAdapter()
    client = _client()

    await adapter.get_response(
        client,
        "gpt-5",
        [{"role": "user", "content": "hello"}],
        request_options=RequestOptions(reasoning_effort="low"),
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert "reasoning" not in kwargs


@pytest.mark.asyncio
async def test_count_tokens_returns_estimated_token_count():
    result = await OpenAIAdapter().count_tokens(
        _client(),
        "gpt-5",
        [{"role": "user", "content": "hello world"}],
    )

    assert result.input_tokens > 0
    assert result.total_tokens == result.input_tokens


@pytest.mark.asyncio
async def test_files_api_wrappers_use_openai_client():
    adapter = OpenAIAdapter()
    client = _client()

    uploaded = await adapter.file_upload(client, ("test.txt", b"hello"), purpose="assistants")
    fetched = await adapter.file_get(client, uploaded.id)
    deleted = await adapter.file_delete(client, uploaded.id)

    assert uploaded.id == "file_1"
    assert fetched.id == "file_1"
    assert deleted is True
    assert adapter.file_reference(uploaded) == {"file_id": "file_1"}
    client.files.create.assert_awaited_once()
    client.files.retrieve.assert_awaited_once_with("file_1")
    client.files.delete.assert_awaited_once_with("file_1")


@pytest.mark.asyncio
async def test_batch_submit_and_results_are_keyed_by_custom_id():
    adapter = OpenAIAdapter()
    client = _client()
    request = BatchRequest(
        custom_id="a",
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "one"}],
    )

    handle = await adapter.batch_submit(client, [request])
    results = await adapter.batch_results(client, handle)

    assert handle.id == "batch_1"
    client.files.create.assert_awaited_once()
    uploaded_file = client.files.create.await_args.kwargs["file"]
    assert uploaded_file[0] == "kestrel-openai-batch.jsonl"
    line = json.loads(uploaded_file[1].decode("utf-8").strip())
    assert line["custom_id"] == "a"
    assert line["url"] == "/v1/chat/completions"
    assert line["body"]["model"] == "gpt-5-mini"
    assert [item.custom_id for item in results] == ["a"]
    assert results[0].response.content == "first"


@pytest.mark.asyncio
async def test_raw_request_dispatches_provider_unique_operations():
    client = _client()
    result = await OpenAIAdapter().raw_request(
        client,
        "responses.create",
        {"model": "gpt-5", "input": "hi"},
    )

    assert result.operation == "responses.create"
    assert result.data == {"id": "resp_1"}
    client.responses.create.assert_awaited_once_with(model="gpt-5", input="hi")


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("KESTREL_LIVE_TESTS") or not os.environ.get("OPENAI_API_KEY"),
    reason="live smoke requires explicit opt-in: set KESTREL_LIVE_TESTS=1 and OPENAI_API_KEY",
)
@pytest.mark.asyncio
async def test_live_openai_smoke_count_tokens_batch_and_files():
    import openai

    adapter = OpenAIAdapter()
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_SMOKE_MODEL", "gpt-5-mini")
    messages = [{"role": "user", "content": "hello world"}]

    count = await adapter.count_tokens(client, model, messages)
    assert count.input_tokens and count.input_tokens > 0

    uploaded = await adapter.file_upload(
        client,
        ("kestrel-openai-smoke.txt", b"hello from kestrel smoke"),
        purpose="assistants",
    )
    assert uploaded.id
    assert adapter.file_reference(uploaded) == {"file_id": uploaded.id}

    batch = await adapter.batch_submit(
        client,
        [
            BatchRequest(custom_id="one", model=model, messages=messages),
            BatchRequest(
                custom_id="two",
                model=model,
                messages=[{"role": "user", "content": "say two"}],
            ),
        ],
    )
    assert batch.id
