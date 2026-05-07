"""
Regression: memory recall must strip sent-form wrappers from user rows.

User turns are persisted in fully-rendered prompt form so the
conversation-history loader can replay byte-exact bytes for prompt-
cache stability:

    <retrieved_context>
    <memories>...</memories>
    </retrieved_context>
    <user_input>
    {raw user text}
    </user_input>

That shape is correct for prompt replay. It is *wrong* when fed back
to the LLM as a memory-recall tool result: the model treats the
``user``-role content as the user's actual words and ends up
paraphrasing the previous turn's retrieved-context block back at the
conversation. Real-world symptom: April 28 cluster of "Based on the
retrieved context, I can tell you that..." memories.

The fix is local to the recall path. ``search_memory`` /
``recall_recent`` / ``recall_emotional`` strip sent-form via
``extract_raw_user_content`` for ``role == "user"`` rows; assistant
rows pass through unchanged.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.memory.feature import (
    MemoryFeature,
    _strip_sent_form_for_recall,
)


SENT_FORM = (
    "\n<retrieved_context>\n<memories>April 28 — discussion about the constitution"
    "</memories>\n</retrieved_context>\n<user_input>\nwhat about Article IV?\n</user_input>"
)
RAW_USER = "what about Article IV?"


def test_strip_helper_pulls_raw_text_from_user_rows_only():
    rows = [
        {"role": "user", "content": SENT_FORM, "metadata": None},
        {"role": "assistant", "content": "The constitution defines...", "metadata": None},
        {"role": "user", "content": "plain legacy row no wrappers", "metadata": None},
    ]
    out = _strip_sent_form_for_recall(rows)

    assert out[0]["content"] == RAW_USER, (
        "user-role rows must have <retrieved_context> + <user_input> wrappers stripped"
    )
    assert out[1]["content"] == "The constitution defines...", (
        "assistant-role rows must pass through unchanged"
    )
    assert out[2]["content"] == "plain legacy row no wrappers", (
        "extract_raw_user_content is idempotent on legacy raw rows"
    )


def test_strip_helper_leaves_metadata_untouched():
    rows = [
        {"role": "user", "content": SENT_FORM, "metadata": {"sent_form": True, "session_id": "s1"}},
    ]
    out = _strip_sent_form_for_recall(rows)
    assert out[0]["metadata"] == {"sent_form": True, "session_id": "s1"}


def test_strip_helper_does_not_mutate_input():
    rows = [{"role": "user", "content": SENT_FORM}]
    original_content = rows[0]["content"]
    _strip_sent_form_for_recall(rows)
    assert rows[0]["content"] == original_content, (
        "strip helper must return new dicts, not mutate the input"
    )


@pytest.mark.asyncio
async def test_search_memory_returns_raw_user_text_to_llm():
    """When the LLM calls search_memory, user-role rows in the result
    must contain raw user text — not the rendered sent-form prompt."""
    feature = MemoryFeature.__new__(MemoryFeature)

    fake_store = MagicMock()
    fake_store.search_history = AsyncMock(return_value=[
        {"role": "user", "content": SENT_FORM, "metadata": None},
        {"role": "assistant", "content": "I remember.", "metadata": None},
    ])
    feature._get_conversation_store = lambda: fake_store

    out = await feature.search_memory(query="Article IV", limit=10)

    assert out.status is ToolResultStatus.OK
    results = out.data["results"]
    user_rows = [r for r in results if r["role"] == "user"]
    assert user_rows[0]["content"] == RAW_USER
    # Assistant unchanged — guards against over-stripping.
    assistant_rows = [r for r in results if r["role"] == "assistant"]
    assert assistant_rows[0]["content"] == "I remember."


@pytest.mark.asyncio
async def test_recall_recent_returns_raw_user_text_to_llm():
    feature = MemoryFeature.__new__(MemoryFeature)

    fake_storage = MagicMock()
    fake_storage.get_conversation_history = AsyncMock(return_value=[
        {"role": "user", "content": SENT_FORM},
        {"role": "assistant", "content": "Got it."},
    ])
    feature.storage = fake_storage

    out = await feature.recall_recent(limit=10)

    assert out.status is ToolResultStatus.OK
    messages = out.data["messages"]
    assert messages[0]["content"] == RAW_USER
    assert messages[1]["content"] == "Got it."


def test_idempotent_on_assistant_content_that_happens_to_contain_user_input_tag():
    """Defense-in-depth: if an assistant row's persisted text accidentally
    starts with ``<user_input>`` (e.g. the model emitted it as part of
    its reply), the strip helper must NOT touch it — only role=user rows
    are eligible. This guards against losing assistant content."""
    rows = [
        {"role": "assistant", "content": "<user_input>\nQuoting back\n</user_input>"},
    ]
    out = _strip_sent_form_for_recall(rows)
    assert out[0]["content"] == "<user_input>\nQuoting back\n</user_input>", (
        "assistant content must never be stripped, even when it contains the wrapper tags"
    )
