"""Chat attachments (#1662) — upload-ref sanitization + user-turn persistence."""
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock


# --- _sanitize_attachments (endpoint, security-relevant) --------------------

def test_sanitize_attachments_keeps_only_valid_refs():
    from kestrel_sovereign.endpoints.agent import _sanitize_attachments
    h = "a" * 64
    out = _sanitize_attachments([
        {"hash": h, "kind": "image", "mime": "image/png", "name": "shot.png"},
    ])
    assert out == [{"hash": h, "kind": "image", "mime": "image/png",
                    "name": "shot.png", "inline": False}]


def test_sanitize_attachments_drops_malformed_and_bounds_count():
    from kestrel_sovereign.endpoints.agent import _sanitize_attachments
    h = "b" * 64
    out = _sanitize_attachments([
        {"hash": "not-a-sha"},          # bad hash → dropped
        {"kind": "image"},               # no hash → dropped
        "nope",                          # not a dict → dropped
        {"hash": h, "kind": "weird"},    # bad kind → defaults to document
        {"hash": h, "mime": "evil/x"},   # disallowed mime → nulled
    ])
    assert [a["hash"] for a in out] == [h, h]
    assert out[0]["kind"] == "document"   # bad kind coerced
    assert out[1]["mime"] is None         # disallowed mime nulled
    # Count is bounded.
    assert len(_sanitize_attachments([{"hash": h}] * 50)) == 10


def test_sanitize_attachments_non_list_is_empty():
    from kestrel_sovereign.endpoints.agent import _sanitize_attachments
    assert _sanitize_attachments(None) == []
    assert _sanitize_attachments("x") == []
    assert _sanitize_attachments({"hash": "a" * 64}) == []


# --- attachments persist on the user turn -----------------------------------

@asynccontextmanager
async def _passthrough():
    yield


@pytest.mark.asyncio
async def test_attachments_persist_on_user_turn_metadata():
    """process_input_streaming threads attachments → the user-row metadata, so
    the composer's images/docs survive reload. The file bytes stay in the
    encrypted store; only the refs ride in the conversation row."""
    from kestrel_sovereign.agent.streaming import StreamingMixin
    from kestrel_sovereign.llm.adapter import LLMResponse

    add_convo_calls = []
    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock(
        side_effect=lambda role, content, **kw: add_convo_calls.append(
            {"role": role, "content": content, **kw})
    )
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    mock_agent = MagicMock()
    mock_agent.privacy_agent = privacy_agent
    mock_agent.features = {}
    mock_agent.did = "test-did"
    mock_agent.extension = None
    mock_agent._cached_features_prompt = ""
    mock_agent.is_request_cancelled = MagicMock(return_value=False)
    mock_agent._maybe_audit = AsyncMock()
    mock_agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    mock_agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    mock_agent.hooks_manager = None
    mock_agent._get_governing_constitution = AsyncMock(return_value="")
    mock_agent.check_solvency = AsyncMock(return_value="test-model")
    mock_agent._build_all_tools = MagicMock(return_value=[])
    mock_agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    mock_agent.user_prompt_template = MagicMock()
    mock_agent.user_prompt_template.format.return_value = "rendered prompt"

    context_result = MagicMock()
    context_result.system_prompt = "system"
    context_result.dynamic_user_context = "ctx"
    context_result.messages = []
    mock_agent.context_manager = MagicMock()
    mock_agent.context_manager.build_context = AsyncMock(return_value=context_result)

    mock_agent.observability_store = MagicMock()
    mock_agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    mock_agent.observability_store.log_tool_response = AsyncMock()
    mock_agent.observability_store.log_metric = AsyncMock()

    async def mock_stream(**kwargs):
        yield "Looking at your screenshot. "
        yield LLMResponse(content="", tool_calls=[])

    mock_agent.llm_service = MagicMock()
    mock_agent.llm_service.stream_with_tool_detection = mock_stream

    mock_agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(mock_agent)
    mock_agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(mock_agent))
    mock_agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(mock_agent))
    mock_agent._resolve_eager_images = (
        StreamingMixin._resolve_eager_images.__get__(mock_agent))

    h = "c" * 64
    attachments = [{"hash": h, "kind": "image", "mime": "image/png", "name": "shot.png"}]
    async for _ in mock_agent.process_input_streaming(
        "what's in this image?", session_id="sess-1", attachments=attachments,
    ):
        pass

    user_rows = [c for c in add_convo_calls if c["role"] == "user"]
    assert len(user_rows) == 1
    meta = user_rows[0].get("metadata") or {}
    assert meta.get("attachments") == attachments


@pytest.mark.asyncio
async def test_no_attachments_means_no_attachments_key():
    """A turn without attachments must not add an empty attachments key."""
    from kestrel_sovereign.agent.streaming import StreamingMixin
    from kestrel_sovereign.llm.adapter import LLMResponse

    add_convo_calls = []
    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock(
        side_effect=lambda role, content, **kw: add_convo_calls.append(
            {"role": role, "content": content, **kw})
    )
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    mock_agent = MagicMock()
    mock_agent.privacy_agent = privacy_agent
    mock_agent.features = {}
    mock_agent.did = "test-did"
    mock_agent.extension = None
    mock_agent._cached_features_prompt = ""
    mock_agent.is_request_cancelled = MagicMock(return_value=False)
    mock_agent._maybe_audit = AsyncMock()
    mock_agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    mock_agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    mock_agent.hooks_manager = None
    mock_agent._get_governing_constitution = AsyncMock(return_value="")
    mock_agent.check_solvency = AsyncMock(return_value="test-model")
    mock_agent._build_all_tools = MagicMock(return_value=[])
    mock_agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    mock_agent.user_prompt_template = MagicMock()
    mock_agent.user_prompt_template.format.return_value = "rendered prompt"
    context_result = MagicMock()
    context_result.system_prompt = "system"
    context_result.dynamic_user_context = "ctx"
    context_result.messages = []
    mock_agent.context_manager = MagicMock()
    mock_agent.context_manager.build_context = AsyncMock(return_value=context_result)
    mock_agent.observability_store = MagicMock()
    mock_agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    mock_agent.observability_store.log_tool_response = AsyncMock()
    mock_agent.observability_store.log_metric = AsyncMock()

    async def mock_stream(**kwargs):
        yield "Hello."
        yield LLMResponse(content="", tool_calls=[])

    mock_agent.llm_service = MagicMock()
    mock_agent.llm_service.stream_with_tool_detection = mock_stream
    mock_agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(mock_agent)
    mock_agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(mock_agent))
    mock_agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(mock_agent))
    mock_agent._resolve_eager_images = (
        StreamingMixin._resolve_eager_images.__get__(mock_agent))

    async for _ in mock_agent.process_input_streaming("hi", session_id="s"):
        pass

    user_rows = [c for c in add_convo_calls if c["role"] == "user"]
    assert len(user_rows) == 1
    assert "attachments" not in (user_rows[0].get("metadata") or {})
