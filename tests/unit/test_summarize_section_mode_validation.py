"""
Regression (#1946, surfaced by #1925 dogfooding): summarize_section.mode

The ``mode`` argument accepts exactly {time_range, topic, messages, last_n}.
Pre-fix, an invalid mode fell through ``get_messages_for_selection``'s
elif-chain, returned ``[]``, and surfaced as a misleading
"No messages found for {mode}={criteria}" — hiding the real problem (a bad
mode) behind an empty-result message.

The fix validates ``mode`` up front and returns ``ToolResult.failed``
listing the four valid modes. Valid modes still work.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.context.feature import ContextFeature


def _make_feature(messages=None):
    feature = ContextFeature.__new__(ContextFeature)
    cm = MagicMock()
    cm.get_messages_for_selection = AsyncMock(return_value=messages or [])
    cm.summarize_messages = AsyncMock(return_value={
        "ok": True,
        "summary_id": "sum-1",
        "replaced_count": 2,
    })
    feature.context_manager = cm  # test-only setter pushes onto agent stub
    feature.llm_service = MagicMock()
    return feature, cm


@pytest.mark.asyncio
async def test_invalid_mode_returns_error_not_no_messages_found():
    feature, cm = _make_feature()

    out = await feature.summarize_section(mode="bogus", criteria="whatever")

    assert out.status is ToolResultStatus.ERROR
    # Must name the bad mode and the valid set; must NOT be the misleading
    # "No messages found" message.
    assert "bogus" in out.error
    for valid in ("time_range", "topic", "messages", "last_n"):
        assert valid in out.error
    assert "No messages found" not in out.error
    # Short-circuits before touching the conversation manager.
    cm.get_messages_for_selection.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_mode_echoes_request_in_data():
    feature, _ = _make_feature()
    out = await feature.summarize_section(mode="LAST", criteria="5")
    assert out.status is ToolResultStatus.ERROR
    assert out.data["mode_requested"] == "LAST"
    assert "last_n" in out.data["valid_modes"]


@pytest.mark.asyncio
async def test_valid_mode_is_normalized_and_runs():
    """A valid mode (with surrounding whitespace/case) is normalized and
    flows through to selection + summarization."""
    feature, cm = _make_feature(messages=[
        {"id": 1, "role": "user", "content": "a"},
        {"id": 2, "role": "assistant", "content": "b"},
    ])

    out = await feature.summarize_section(mode="  Last_N  ", criteria="2")

    assert out.status is ToolResultStatus.OK
    # Normalized mode handed to the selection layer.
    call_kwargs = cm.get_messages_for_selection.call_args.kwargs
    assert call_kwargs["mode"] == "last_n"
