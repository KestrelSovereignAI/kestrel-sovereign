"""Pre-response failure-result rewrite at the codex adapter (#1563)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.codex_adapter import (
    CodexAdapter,
    _result_to_codex_response,
    classify_and_render_failure,
)


def test_codex_rejected_string_renders_as_sandbox_blocked():
    raw = "CreateProcess { message: \"Rejected(\\\"rejected by user\\\")\" }"
    block = classify_and_render_failure(raw, tool_name="bash")
    assert "Outcome: sandbox_blocked" in block
    assert "Recovery:" in block
    assert raw in block
    assert "do NOT echo verbatim" in block


def test_recovery_hint_per_outcome_distinguishes_user_denied():
    sandbox_block = classify_and_render_failure(
        "rejected by user", tool_name="bash",
    )
    assert "sandbox" in sandbox_block.lower()
    assert "NOT a user denial" in sandbox_block

    binary_missing = classify_and_render_failure(
        "binary not found: gh", tool_name="gh",
    )
    assert "Outcome: tooling_error" in binary_missing
    assert "binary" in binary_missing.lower()
    assert "NOT attribute this to a user denial" in binary_missing

    policy_timeout = classify_and_render_failure(
        "approval timed out", tool_name="bash",
    )
    assert "Outcome: policy_blocked" in policy_timeout


def test_audit_backed_user_denied_renders_with_respect_hint():
    audit = [{
        "feature": "shell", "tool": "bash",
        "decision": "user_denied", "user_choice": "user_denied",
    }]
    block = classify_and_render_failure(
        "Rejected(\"rejected by user\")",
        tool_name="bash", feature_name="shell",
        recent_decisions=audit,
    )
    assert "Outcome: user_denied" in block
    assert "respect the user's denial" in block


def test_empty_raw_error_renders_as_unconfirmed_block():
    block = classify_and_render_failure("", tool_name="bash")
    assert "Outcome: unconfirmed" in block
    assert "could not be confirmed" in block


def test_success_result_is_NOT_rewritten():
    out = _result_to_codex_response(
        {"success": True, "result": "ok"},
        tool_name="x",
    )
    assert out["success"] is True
    text = out["contentItems"][0]["text"]
    assert text == "ok"
    assert "Outcome:" not in text


def test_failure_result_with_codex_rejection_is_rewritten():
    raw = "CreateProcess { message: \"Rejected(\\\"rejected by user\\\")\" }"
    out = _result_to_codex_response(
        {"success": False, "error": raw},
        tool_name="gh",
    )
    assert out["success"] is False
    text = out["contentItems"][0]["text"]
    assert "Outcome: sandbox_blocked" in text
    assert "Recovery:" in text
    assert raw in text
    assert "do NOT echo verbatim" in text


def test_failure_result_missing_explicit_error_field_still_rewrites():
    out = _result_to_codex_response(
        {"success": False, "result": "binary not found: codex"},
        tool_name="codex",
    )
    text = out["contentItems"][0]["text"]
    assert "Outcome: tooling_error" in text


@pytest.mark.asyncio
async def test_adapter_recent_security_decisions_no_agent_yields_empty():
    adapter = CodexAdapter()
    rows = await adapter._recent_security_decisions()
    assert rows == []


@pytest.mark.asyncio
async def test_adapter_recent_security_decisions_missing_security_feature_yields_empty():
    adapter = CodexAdapter()
    agent = MagicMock()
    agent.features = {}
    adapter.attach_agent_for_audit(agent)
    rows = await adapter._recent_security_decisions()
    assert rows == []


@pytest.mark.asyncio
async def test_adapter_recent_security_decisions_with_real_audit():
    adapter = CodexAdapter()
    agent = MagicMock()
    security = MagicMock()
    permission_store = MagicMock()
    permission_store.get_audit_log = AsyncMock(return_value=[{
        "feature": "shell", "tool": "bash",
        "decision": "user_denied", "user_choice": "user_denied",
    }])
    security.permission_store = permission_store
    agent.features = {"SecurityFeature": security}
    adapter.attach_agent_for_audit(agent)

    rows = await adapter._recent_security_decisions()
    assert len(rows) == 1
    assert rows[0]["decision"] == "user_denied"


@pytest.mark.asyncio
async def test_adapter_audit_lookup_swallows_exceptions():
    adapter = CodexAdapter()
    agent = MagicMock()
    security = MagicMock()
    permission_store = MagicMock()
    permission_store.get_audit_log = AsyncMock(
        side_effect=RuntimeError("disk full"),
    )
    security.permission_store = permission_store
    agent.features = {"SecurityFeature": security}
    adapter.attach_agent_for_audit(agent)

    rows = await adapter._recent_security_decisions()
    assert rows == []


def test_toolresult_failed_with_codex_rejection_is_rewritten():
    """codex P1 round 1: most tools (after #1061 PR-E) return
    ``ToolResult`` objects, not legacy ``{success: bool}`` dicts. A
    direct repro showed the rewrite missed these entirely and
    stringified the misleading raw text with ``success=True``.

    A ``ToolResult.failed(...)`` MUST be detected as failure and
    routed through the classifier.
    """
    from kestrel_sdk.tools.result import ToolResult

    raw = "CreateProcess { message: \"Rejected(\\\"rejected by user\\\")\" }"
    tr = ToolResult.failed(raw)
    out = _result_to_codex_response(tr, tool_name="bash")
    assert out["success"] is False, (
        "ToolResult.failed must marshal as success=False"
    )
    text = out["contentItems"][0]["text"]
    assert "Outcome: sandbox_blocked" in text
    assert "Recovery:" in text
    assert raw in text


def test_toolresult_ok_is_NOT_rewritten():
    """The mirror: a ``ToolResult.ok(...)`` must pass through with
    its confirmation/data text unchanged — only failures route
    through the classifier."""
    from kestrel_sdk.tools.result import ToolResult

    out = _result_to_codex_response(
        ToolResult.ok(confirmation="all set"), tool_name="x",
    )
    assert out["success"] is True
    text = out["contentItems"][0]["text"]
    assert "Outcome:" not in text
    assert "all set" in text


def test_toolresult_partial_is_treated_as_failure():
    """``ToolResultStatus.PARTIAL`` indicates a mixed outcome the
    LLM should NOT narrate as success. Treat it as failure for the
    rewrite so the classifier annotates it."""
    from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

    tr = ToolResult(
        status=ToolResultStatus.PARTIAL,
        confirmation="mixed result",
        error="some step failed",
    )
    out = _result_to_codex_response(tr, tool_name="x")
    assert out["success"] is False
    text = out["contentItems"][0]["text"]
    assert "Outcome:" in text


@pytest.mark.asyncio
async def test_handler_passes_audit_for_toolresult_failures():
    """codex P1 round 2: ``_make_tool_call_handler`` originally only
    fetched the security audit slice for legacy dict failures, so a
    real ``ToolResult.failed`` with a Codex rejection bypassed the
    audit cross-check. The handler MUST mirror
    ``_result_to_codex_response``'s failure detection for both
    shapes.
    """
    from kestrel_sdk.tools.result import ToolResult

    adapter = CodexAdapter()
    agent = MagicMock()
    security = MagicMock()
    permission_store = MagicMock()
    permission_store.get_audit_log = AsyncMock(return_value=[{
        "feature": "shell", "tool": "bash",
        "decision": "user_denied", "user_choice": "user_denied",
    }])
    security.permission_store = permission_store
    agent.features = {"SecurityFeature": security}
    adapter.attach_agent_for_audit(agent)

    raw = "Rejected(\"rejected by user\")"

    async def fake_executor(name, args):
        return ToolResult.failed(raw)

    handler = adapter._make_tool_call_handler(
        executor=fake_executor,
        thread_id="t1",
        allowed_tools=frozenset({"bash"}),
        executed_log=[],
    )
    result = await handler({
        "threadId": "t1",
        "callId": "c1", "name": "bash", "arguments": "{}",
    })
    text = result["contentItems"][0]["text"]
    # Audit row backs user denial → block renders USER_DENIED, not
    # SANDBOX_BLOCKED. Without the round-2 fix this would still
    # have classified as sandbox_blocked.
    assert "Outcome: user_denied" in text
    # And get_audit_log was actually called (proof we crossed the
    # is_failure branch for a ToolResult).
    assert permission_store.get_audit_log.await_count == 1
