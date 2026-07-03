"""DynamicTool.execute() must serialize a ToolResult-returning @tool
at the wrap site (#1070).

#1066 codex rounds 5 and 6 caught two surfaces (run_workflow,
check_task_status) where raw ``ToolResult`` instances embedded in
``DynamicTool.execute()``'s wrapper dict broke downstream JSON
serialization. PR-E pilot worked around this with per-callsite
``_serialize_step_payload`` helpers. This test pins the
framework-level fix: ``DynamicTool.execute()`` itself converts a
``ToolResult`` return to its dict form, and reflects the inner
status in the wrapper's transport-level ``success`` flag.

Unified wire shape (#F025): the ToolResult envelope is spread TOP-LEVEL
(matching the SDK wrapper), with a retained ``success`` back-compat flag:

  - @tool returns ``ToolResult.ok(...)`` → wrapper:
    ``{"status": "ok", "confirmation": ..., "data": ..., "tool": "...", "success": True}``
  - @tool returns ``ToolResult.failed(...)`` → wrapper:
    ``{"status": "error", "error": "...", "data": ..., "tool": "...", "success": False}``
  - @tool returns ``ToolResult.partial(c, e)`` → wrapper:
    ``{"status": "partial", "confirmation": c, "error": e, "data": ..., "tool": "...", "success": True}``
  - @tool returns plain dict (pre-migration) → wrapper unchanged:
    ``{"success": True, "result": {dict}, "tool": "..."}``

Honesty: ``status`` is the canonical signal (top-level, so
``summarize_tool_result_for_audit`` sees a PARTIAL — #F001), and
``success`` reflects the *semantic* outcome for migrated tools
(`error` ⇒ False), not just whether the call raised. command_handler
renders every feature's ``!command`` from this one shape (#F002).
"""

import json

import pytest

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign.features.base import Feature, tool


class _FixtureFeature(Feature):
    """Test feature with one tool per ToolResult status + a legacy dict."""

    @property
    def tool_description(self) -> str:
        return "test"

    async def initialize(self):
        pass

    @tool("returns_ok", "returns ToolResult.ok", category=ToolCategory.UTILITY)
    async def returns_ok(self) -> ToolResult:
        return ToolResult.ok("done", data={"n": 1})

    @tool("returns_failed", "returns ToolResult.failed", category=ToolCategory.UTILITY)
    async def returns_failed(self) -> ToolResult:
        return ToolResult.failed("boom", data={"reason": "test"})

    @tool("returns_partial", "returns ToolResult.partial", category=ToolCategory.UTILITY)
    async def returns_partial(self) -> ToolResult:
        return ToolResult.partial(
            confirmation="half done",
            error="other half failed",
            data={"phase": "1of2"},
        )

    @tool("returns_legacy_dict", "returns plain dict", category=ToolCategory.UTILITY)
    async def returns_legacy_dict(self):  # no -> annotation: pre-migration
        return {"some": "data"}

    @tool("raises_exception", "raises", category=ToolCategory.UTILITY)
    async def raises_exception(self):
        raise RuntimeError("the tool blew up")


@pytest.fixture
def feature():
    feat = _FixtureFeature.__new__(_FixtureFeature)
    feat.disabled_skills = frozenset()  # set by Feature.__init__ normally
    return feat


def _tool(feature, name):
    for t in feature.get_tools():
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not in {[t.name for t in feature.get_tools()]}")


# ---------------------------------------------------------------------------
# ToolResult.ok
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ok_serializes_to_dict_and_keeps_success_true(feature):
    result = await _tool(feature, "returns_ok").execute()
    assert result["success"] is True
    assert result["tool"] == "returns_ok"
    # Envelope spread top-level (unified shape) — NOT nested under "result".
    assert "result" not in result
    assert result["status"] == "ok"
    assert result["confirmation"] == "done"
    assert result["data"] == {"n": 1}
    # No error on a clean OK
    assert "error" not in result


@pytest.mark.asyncio
async def test_ok_wire_payload_is_json_clean(feature):
    """The whole wrapper must round-trip through json.dumps."""
    result = await _tool(feature, "returns_ok").execute()
    json.dumps(result)


# ---------------------------------------------------------------------------
# ToolResult.failed — the !command honesty leak
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_flips_wrapper_success_to_false(feature):
    """Pre-fix: wrapper success was True (no exception raised) and the
    !command path branched on it, telling the user "succeeded" while
    the ToolResult said error. The fix derives wrapper success from the
    inner status."""
    result = await _tool(feature, "returns_failed").execute()
    assert result["success"] is False
    assert result["error"] == "boom"
    assert result["tool"] == "returns_failed"
    # Envelope spread top-level.
    assert "result" not in result
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_failed_wire_payload_is_json_clean(feature):
    result = await _tool(feature, "returns_failed").execute()
    json.dumps(result)


# ---------------------------------------------------------------------------
# ToolResult.partial — both confirmation AND error must be reachable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_keeps_success_true_and_surfaces_error(feature):
    """PARTIAL means the action partly succeeded — wrapper success
    stays True so callers don't drop the result, but the wrapper's
    top-level error field carries the caveat so legacy callers that
    only read ``error`` (e.g. command_handler) still see the partial
    half."""
    result = await _tool(feature, "returns_partial").execute()
    assert result["success"] is True
    assert result["error"] == "other half failed"
    # Envelope spread top-level with BOTH confirmation and error, and
    # status=partial visible top-level so the honesty layer flags it.
    assert "result" not in result
    assert result["status"] == "partial"
    assert result["confirmation"] == "half done"


@pytest.mark.asyncio
async def test_partial_wire_payload_is_json_clean(feature):
    result = await _tool(feature, "returns_partial").execute()
    json.dumps(result)


@pytest.mark.asyncio
async def test_partial_is_visible_to_honesty_layer(feature):
    """#F001: a PARTIAL in-tree tool result must reach the constitutional
    honesty layer as a failure, so a past-tense success claim over it is
    flagged. Before unification the sovereign wrapper nested ``status`` under
    ``result`` and exposed only a derived ``success=True`` top-level, so
    ``summarize_tool_result_for_audit`` (which reads top-level ``status``) saw
    ``success=True`` and let the false claim through. This is the exact path the
    orchestrator runs (summarize → analyze_narration)."""
    from kestrel_sovereign.security.narration_check import (
        summarize_tool_result_for_audit,
        _result_indicates_failure,
        analyze_narration,
    )

    wrapper = await _tool(feature, "returns_partial").execute()
    summary = summarize_tool_result_for_audit(wrapper)
    assert summary.get("status") == "partial"
    assert _result_indicates_failure(summary) is True

    verdict = analyze_narration(
        pre_tool_prose="Saved your note successfully.",
        tool_results=[{"name": "returns_partial", "result": summary}],
    )
    assert verdict.risk_boost > 0


@pytest.mark.asyncio
async def test_ok_is_not_flagged_by_honesty_layer(feature):
    """Complement: a clean OK must NOT be flagged, so a truthful past-tense
    success claim passes (no spurious audit elevation)."""
    from kestrel_sovereign.security.narration_check import (
        summarize_tool_result_for_audit,
        analyze_narration,
    )

    wrapper = await _tool(feature, "returns_ok").execute()
    summary = summarize_tool_result_for_audit(wrapper)
    assert summary.get("status") == "ok"
    verdict = analyze_narration(
        pre_tool_prose="Saved your note successfully.",
        tool_results=[{"name": "returns_ok", "result": summary}],
    )
    assert verdict.risk_boost == 0


# ---------------------------------------------------------------------------
# Pre-migration dict — wrapper unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_dict_return_keeps_original_wrapper(feature):
    """A tool that hasn't been migrated to ToolResult yet still gets
    the original wrapper. The success flag remains transport-level
    (`True` because the call didn't raise) — the #1061 bulk waves
    migrate these tools one by one."""
    result = await _tool(feature, "returns_legacy_dict").execute()
    assert result == {
        "success": True,
        "result": {"some": "data"},
        "tool": "returns_legacy_dict",
    }


# ---------------------------------------------------------------------------
# Exceptions — original error path unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_path_unchanged(feature):
    result = await _tool(feature, "raises_exception").execute()
    assert result["success"] is False
    assert "the tool blew up" in result["error"]
    assert result["tool"] == "raises_exception"


# ---------------------------------------------------------------------------
# Round-trip through ToolResult and back — check the no-double-wrap path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_raw_toolresult_in_any_wire_field(feature):
    """Every field of the wrapper must be JSON-clean for every status."""
    for tool_name in ("returns_ok", "returns_failed", "returns_partial"):
        result = await _tool(feature, tool_name).execute()
        # Walk the dict and assert no ToolResult instances anywhere.
        def _walk(node):
            assert not isinstance(node, ToolResult), (
                f"{tool_name}: raw ToolResult leaked into wrapper at "
                f"{node!r}"
            )
            if isinstance(node, dict):
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)
        _walk(result)
