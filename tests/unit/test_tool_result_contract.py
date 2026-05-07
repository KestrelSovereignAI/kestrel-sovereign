"""Tests for kestrel_sovereign.tools.result_contract.

The validator pins the #1042 layer 4 promise — every @tool method on a
migrated feature must annotate ``-> ToolResult`` (and at runtime, return
one). The pilot allowlist lives in ``MIGRATED_FEATURE_MODULES``; these
tests assert the three pilot features are clean and that the validator
catches a synthetic regression.
"""

from __future__ import annotations

import pytest

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.context.feature import ContextFeature
from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.tools.result_contract import (
    MIGRATED_FEATURE_MODULES,
    ToolResultContractError,
    assert_feature_returns_tool_result,
    enforce_tool_result_contract,
    find_violations,
    is_migrated_module,
)


# ---------------------------------------------------------------------------
# Pilot feature classes — must be clean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("feature_cls", [TaskFeature, MemoryFeature, ContextFeature])
def test_pilot_features_pass_contract(feature_cls):
    """Every pilot-allowlisted feature returns ToolResult on every @tool."""
    # Validator works on instances; instantiate without calling initialize()
    # so we don't need fully-wired storage.
    feature = feature_cls.__new__(feature_cls)
    violations = find_violations(feature)
    assert violations == [], (
        f"{feature_cls.__name__} has uncaught contract violations:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


@pytest.mark.parametrize("feature_cls", [TaskFeature, MemoryFeature, ContextFeature])
def test_pilot_modules_are_in_allowlist(feature_cls):
    """The allowlist must mention every pilot feature's module."""
    assert feature_cls.__module__ in MIGRATED_FEATURE_MODULES


@pytest.mark.parametrize("feature_cls", [TaskFeature, MemoryFeature, ContextFeature])
def test_assert_feature_returns_tool_result_does_not_raise(feature_cls):
    """The test-friendly form is a no-op for clean features."""
    feature = feature_cls.__new__(feature_cls)
    assert_feature_returns_tool_result(feature)  # raises if dirty


# ---------------------------------------------------------------------------
# Synthetic regression: a non-migrated module shipping a bad tool is OK
# ---------------------------------------------------------------------------


class _NonMigratedFeature(Feature):
    """Pre-migration features may still return Dict[str, Any]; the
    validator does NOT reject them. The allowlist is opt-in."""

    @property
    def tool_description(self) -> str:
        return "test"

    async def initialize(self):
        pass

    @tool("legacy_tool", "still returns a dict", category=ToolCategory.UTILITY)
    async def legacy_tool(self) -> dict:
        return {"success": True}


def test_non_migrated_module_is_not_validated():
    """A feature whose module is NOT in the allowlist must not raise."""
    feature = _NonMigratedFeature.__new__(_NonMigratedFeature)
    # Confirm test setup: this synthetic class lives in the test module,
    # which is not on the allowlist.
    assert not is_migrated_module(_NonMigratedFeature.__module__)
    enforce_tool_result_contract(feature)  # silent no-op


# ---------------------------------------------------------------------------
# Synthetic regression: a migrated module with a bad tool MUST raise
# ---------------------------------------------------------------------------


class _BadMigratedFeature(Feature):
    """Synthetic feature that *claims* to be in a migrated module but
    ships a tool returning ``dict`` — exactly the regression the
    validator must catch.

    We can't fake-import this into ``kestrel_sovereign.features.tasks.feature``
    so we instead test the validator's helpers directly: passing this
    class through ``find_violations`` should turn up exactly one
    annotation violation.
    """

    @property
    def tool_description(self) -> str:
        return "test"

    async def initialize(self):
        pass

    @tool("bad_tool", "returns the wrong type", category=ToolCategory.UTILITY)
    async def bad_tool(self) -> dict:
        return {"success": True}


def test_find_violations_catches_dict_return():
    feature = _BadMigratedFeature.__new__(_BadMigratedFeature)
    violations = find_violations(feature)
    assert len(violations) == 1
    assert "bad_tool" in violations[0]
    assert "ToolResult" in violations[0]


def test_find_violations_catches_missing_annotation():
    class _UnannotatedFeature(Feature):
        @property
        def tool_description(self) -> str:
            return "test"

        async def initialize(self):
            pass

        @tool("unannotated_tool", "has no return annotation",
              category=ToolCategory.UTILITY)
        async def unannotated_tool(self):  # no -> annotation
            return ToolResult.ok("ok")

    feature = _UnannotatedFeature.__new__(_UnannotatedFeature)
    violations = find_violations(feature)
    assert len(violations) == 1
    assert "unannotated_tool" in violations[0]
    assert "missing return annotation" in violations[0]


def test_find_violations_rejects_optional_tool_result():
    """``Optional[ToolResult]`` means the tool can also return None —
    the contract forbids that. The validator must reject it."""
    from typing import Optional

    class _OptionalFeature(Feature):
        @property
        def tool_description(self) -> str:
            return "test"

        async def initialize(self):
            pass

        @tool("optional_tool", "returns Optional[ToolResult]",
              category=ToolCategory.UTILITY)
        async def optional_tool(self) -> Optional[ToolResult]:
            return None

    feature = _OptionalFeature.__new__(_OptionalFeature)
    violations = find_violations(feature)
    assert len(violations) == 1
    assert "optional_tool" in violations[0]


def test_assert_feature_returns_tool_result_raises_on_dirty():
    feature = _BadMigratedFeature.__new__(_BadMigratedFeature)
    with pytest.raises(ToolResultContractError, match="bad_tool"):
        assert_feature_returns_tool_result(feature)


def test_warn_only_env_downgrades_violation_to_log(caplog, monkeypatch):
    """The escape hatch must log instead of raise when set."""
    monkeypatch.setenv("KESTREL_TOOL_RESULT_CONTRACT_WARN_ONLY", "1")

    # Pretend BadMigratedFeature lives in a migrated module by patching
    # the allowlist for this single test.
    import kestrel_sovereign.tools.result_contract as result_contract
    original = result_contract.MIGRATED_FEATURE_MODULES
    monkeypatch.setattr(
        result_contract,
        "MIGRATED_FEATURE_MODULES",
        original | frozenset({_BadMigratedFeature.__module__}),
    )

    feature = _BadMigratedFeature.__new__(_BadMigratedFeature)
    with caplog.at_level("WARNING", logger="kestrel_sovereign.tools.result_contract"):
        enforce_tool_result_contract(feature)  # must not raise
    assert any("bad_tool" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_pilot_tool_actually_returns_tool_result_at_runtime():
    """Sanity-check one tool from each pilot feature — the annotation
    promises ToolResult, this asserts the actual return value matches
    (defends against a future change that updates annotations but not
    return statements).
    """
    # MemoryFeature.search_memory: store-unavailable path returns
    # ToolResult.failed (no agent state needed beyond the missing store).
    mem = MemoryFeature.__new__(MemoryFeature)
    mem._get_conversation_store = lambda: None  # type: ignore[assignment]
    result = await mem.search_memory(query="x", limit=5)
    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR

    # TaskFeature.list_available_skills with no task manager.
    feat = TaskFeature(agent=None)
    result = await feat.list_available_skills()
    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR

    # ContextFeature.context_status with no context_manager.
    cf = ContextFeature.__new__(ContextFeature)
    cf.context_manager = None  # type: ignore[assignment]
    cf.llm_service = None  # type: ignore[assignment]
    result = await cf.context_status()
    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
