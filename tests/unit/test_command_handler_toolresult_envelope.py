"""CLI command path must unwrap ToolResult envelopes (#1078 round 4).

When a migrated `@tool` is invoked via its `command_prefix`
(e.g. `!security-list`), the dispatch path goes through
`Feature.handle_task` → `DynamicTool.execute` (which serializes
the ToolResult) → `task_manager.execute_command` → back to
`CommandHandler.handle`. Without unwrapping, the user sees the
raw JSON envelope (`{"status": "ok", "confirmation": "...", ...}`)
instead of the human-readable confirmation.

`CommandHandler.handle` now detects the envelope shape and renders
appropriate CLI text per `ToolResult` status.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.command_handler import CommandHandler


def _handler_with_task_result(envelope: dict) -> CommandHandler:
    """Wire a CommandHandler whose task_manager returns the given
    DynamicTool-shaped wrapper. The wrapper's ``result`` field
    carries the ToolResult envelope under test."""
    agent = MagicMock()
    task_manager = MagicMock()
    task_manager.execute_command = AsyncMock(return_value=envelope)
    handler = CommandHandler(agent, task_manager=task_manager)
    return handler


@pytest.mark.asyncio
async def test_ok_envelope_renders_confirmation_only():
    handler = _handler_with_task_result({
        "success": True,
        "tool": "list_permissions",
        "result": {
            "status": "ok",
            "confirmation": "Security Permissions:\n  ☑ WalletAgent",
            "data": {"feature_count": 1},
        },
    })

    text = await handler.handle("!security-list")
    assert text == "Security Permissions:\n  ☑ WalletAgent"
    # No JSON envelope leakage
    assert "status" not in text
    assert "data" not in text


@pytest.mark.asyncio
async def test_error_envelope_renders_with_error_prefix():
    handler = _handler_with_task_result({
        "success": False,
        "tool": "approve",
        "result": {"status": "error", "error": "Request 'abc' not found"},
        "error": "Request 'abc' not found",
    })

    text = await handler.handle("!security-approve abc")
    assert text.startswith("❌ Error:")
    assert "Request 'abc' not found" in text


@pytest.mark.asyncio
async def test_partial_envelope_renders_both_confirmation_and_caveat():
    """PARTIAL must surface BOTH halves so the user sees the action
    that completed AND the half that didn't (#1042 honesty contract)."""
    handler = _handler_with_task_result({
        "success": True,
        "tool": "approve",
        "result": {
            "status": "partial",
            "confirmation": "Approved abc12345 for this request",
            "error": "scope=session persistence is asynchronous and unverified",
        },
        "error": "scope=session persistence is asynchronous and unverified",
    })

    text = await handler.handle("!security-approve abc")
    # Both halves must appear
    assert "Approved abc12345" in text
    assert "Caveat" in text
    assert "asynchronous" in text


@pytest.mark.asyncio
async def test_envelope_with_unknown_status_falls_back_to_repr():
    """Defensive: if status doesn't match the contract, don't crash —
    render something the user can paste back."""
    handler = _handler_with_task_result({
        "success": True,
        "tool": "weird",
        "result": {"status": "unknown", "data": {}},
    })

    text = await handler.handle("!weird")
    # Should not raise; output is something string-ish
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.asyncio
async def test_legacy_dict_result_falls_through_to_format_result():
    """Pre-migration tools that return Dict[str, Any] (no envelope)
    still go through the legacy _format_result path."""
    handler = _handler_with_task_result({
        "success": True,
        "tool": "legacy_tool",
        "result": {"some_field": "some_value"},  # no "status" key
    })

    text = await handler.handle("!legacy")
    assert isinstance(text, str)
    # Falls through to JSON format (legacy behavior unchanged)
    assert "some_field" in text


@pytest.mark.asyncio
async def test_legacy_string_result_unchanged():
    """Pre-migration tools that return str pass through verbatim."""
    handler = _handler_with_task_result({
        "success": True,
        "tool": "legacy_str_tool",
        "result": "plain string output",
    })

    text = await handler.handle("!legacy")
    assert text == "plain string output"


# ---------------------------------------------------------------------------
# Read-payload rendering (regression caught by codex on PR #1093)
# ---------------------------------------------------------------------------
#
# Without this, !recall etc would print only "Found 3 match(es)" and
# drop the actual results — a real regression vs the legacy
# json.dumps rendering. The formatter now appends a JSON block when
# data carries a list-of-dicts or non-empty dict-of-dicts payload.

@pytest.mark.asyncio
async def test_ok_envelope_with_results_list_renders_data_block():
    handler = _handler_with_task_result({
        "success": True,
        "tool": "recall",
        "result": {
            "status": "ok",
            "confirmation": "Found 1 match(es) for 'foo'",
            "data": {
                "query": "foo",
                "result_count": 1,
                "results": [
                    {"id": "i-1", "name": "thing", "score": 0.9},
                ],
            },
        },
    })

    text = await handler.handle("!recall foo")
    assert "Found 1 match(es)" in text
    # Read payload must be visible to the user.
    assert '"id": "i-1"' in text
    assert '"name": "thing"' in text


@pytest.mark.asyncio
async def test_ok_envelope_with_items_list_renders_data_block():
    handler = _handler_with_task_result({
        "success": True,
        "tool": "recall_list",
        "result": {
            "status": "ok",
            "confirmation": "Listed 2 saved item(s)",
            "data": {
                "count": 2,
                "items": [
                    {"id": "a", "name": "alpha"},
                    {"id": "b", "name": "beta"},
                ],
            },
        },
    })

    text = await handler.handle("!recall list")
    assert "Listed 2 saved item(s)" in text
    assert '"alpha"' in text
    assert '"beta"' in text


@pytest.mark.asyncio
async def test_ok_envelope_with_nested_dict_renders_data_block():
    handler = _handler_with_task_result({
        "success": True,
        "tool": "recall_get",
        "result": {
            "status": "ok",
            "confirmation": "Retrieved item i-1",
            "data": {"item": {"id": "i-1", "name": "thing", "content": "hello"}},
        },
    })

    text = await handler.handle("!recall get i-1")
    assert "Retrieved item i-1" in text
    assert '"content": "hello"' in text


@pytest.mark.asyncio
async def test_ok_envelope_with_scalar_only_data_skips_json_block():
    """Write-style tools (save_*, !security-approve, etc) return
    scalar-only data; the confirmation already conveys the action,
    so the JSON dump would be noise."""
    handler = _handler_with_task_result({
        "success": True,
        "tool": "save_stash",
        "result": {
            "status": "ok",
            "confirmation": "Saved stash 'x' as item y (3 messages)",
            "data": {
                "saved_item_id": "y",
                "name": "x",
                "message_count": 3,
                "has_embedding": True,
            },
        },
    })

    text = await handler.handle("!save stash")
    assert text == "Saved stash 'x' as item y (3 messages)"
    assert "saved_item_id" not in text


@pytest.mark.asyncio
async def test_ok_envelope_with_empty_results_list_skips_json_block():
    """Zero-result recall already says so in the confirmation; the
    empty list adds nothing."""
    handler = _handler_with_task_result({
        "success": True,
        "tool": "recall",
        "result": {
            "status": "ok",
            "confirmation": "No matches for query: 'xyz'",
            "data": {"query": "xyz", "result_count": 0, "results": []},
        },
    })

    text = await handler.handle("!recall xyz")
    assert text == "No matches for query: 'xyz'"


@pytest.mark.asyncio
async def test_partial_envelope_with_data_block_includes_caveat_and_payload():
    handler = _handler_with_task_result({
        "success": True,
        "tool": "save_excerpt",
        "result": {
            "status": "partial",
            "confirmation": "Saved excerpt 'thin' as item exc-2 (1 messages)",
            "error": "requested last_5 but only 1 messages were available",
            "data": {
                "saved_item_id": "exc-2",
                "message_count": 1,
                "requested_count": 5,
                "shortfall": 4,
                "has_embedding": True,
            },
        },
        "error": "requested last_5 but only 1 messages were available",
    })

    text = await handler.handle("!save excerpt last_5 thin")
    assert "Saved excerpt 'thin'" in text
    assert "Caveat" in text
    # Scalar-only data → no JSON block (shortfall is implicit in caveat).
    assert "saved_item_id" not in text
