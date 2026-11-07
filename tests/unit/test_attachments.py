"""Chat attachments (#1662) — upload-ref sanitization + user-turn persistence."""
import pytest
from base64 import b64decode
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace
import pytest_asyncio


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

@pytest_asyncio.fixture
async def attachment_agent(native_tool_effect_agent, monkeypatch):
    """Real encrypted conversation/journal; provider and context are explicit seams."""
    from kestrel_sovereign.llm.adapter import LLMResponse
    from kestrel_sovereign.agent.context_manager import ContextResult

    agent = await native_tool_effect_agent(did="did:test:attachments")
    agent.extension = None
    agent._cached_features_prompt = ""
    agent._maybe_audit = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._fire_post_response_hook = AsyncMock(side_effect=lambda text, sid, **_: text)
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered prompt"
    agent.context_manager = SimpleNamespace(build_context=AsyncMock(return_value=ContextResult(
        system_prompt="system", dynamic_user_context="ctx", messages=[],
        total_tokens=0, budget_summary={},
    )))

    async def stream(**kwargs):
        yield "Looking at your attachment. "
        yield LLMResponse(content="", tool_calls=[])

    monkeypatch.setattr(agent.llm_service, "stream_with_tool_detection", stream)
    return agent


@pytest.mark.asyncio
async def test_attachments_persist_on_user_turn_metadata(attachment_agent):
    """Encrypted file bytes and the user row's upload reference survive the turn."""
    agent = attachment_agent
    content = b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    digest = await agent._raw_storage.store_file(content, "shot.png")
    attachments = [{"hash": digest, "kind": "image", "mime": "image/png", "name": "shot.png"}]
    async for _ in agent.process_input_streaming(
        "what is in this attachment?", session_id="sess-1", attachments=attachments,
    ):
        pass

    rows = await agent.privacy_agent.get_conversation_history(session_id="sess-1")
    user_rows = [row for row in rows if row["role"] == "user"]
    assert len(user_rows) == 1
    assert user_rows[0]["metadata"]["attachments"] == attachments
    assert await agent._raw_storage.retrieve_file(digest) == content
    assert await agent._raw_storage.db.fetchval("SELECT state FROM tool_effect_operations") == "CLOSED"


@pytest.mark.asyncio
async def test_no_attachments_means_no_attachments_key(attachment_agent):
    """A turn without attachments must not add an empty attachments key."""
    agent = attachment_agent
    async for _ in agent.process_input_streaming("hi", session_id="s"):
        pass
    rows = await agent.privacy_agent.get_conversation_history(session_id="s")
    user_rows = [row for row in rows if row["role"] == "user"]
    assert len(user_rows) == 1
    assert "attachments" not in (user_rows[0].get("metadata") or {})
    assert await agent._raw_storage.db.fetchval("SELECT state FROM tool_effect_operations") == "CLOSED"
