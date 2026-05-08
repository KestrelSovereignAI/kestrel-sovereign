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
