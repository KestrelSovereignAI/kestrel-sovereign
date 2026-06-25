"""
Unit tests for cleanup_models dry-run safety (#1946).

cleanup_models is DESTRUCTIVE (deletes model files from disk). Surfaced by
the #1925 dogfooding sweep: dry_run used to default to False, so a naive
``cleanup_models()`` call deleted immediately, and the @tool description
documented none of dry_run / threshold_days / min_free_space_pct. The fix
flips the default to dry_run=True (preview) — mirroring the well-regarded
empty_trash(dry_run=True) pattern — and documents the params.
"""
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.model.feature import ModelAgent


async def _feature() -> tuple[ModelAgent, MagicMock]:
    llm_service = MagicMock()
    llm_service.cleanup_unused_models = AsyncMock(
        return_value={"would_delete": ["m1", "m2"], "freed_bytes": 0}
    )
    agent = SimpleNamespace(llm_service=llm_service, features={})
    feature = ModelAgent(agent)
    await feature.initialize()
    return feature, llm_service


class TestCleanupModelsDryRunDefault:
    @pytest.mark.asyncio
    async def test_default_call_does_not_delete(self):
        """A naive cleanup_models() must PREVIEW, not delete."""
        feature, llm_service = await _feature()

        result = await feature.cleanup_models()

        # Preview => PARTIAL with the dry-run caveat, never OK.
        assert result.status is ToolResultStatus.PARTIAL
        assert "dry_run=False" in result.error
        # The underlying service must have been called with dry_run=True.
        _, kwargs = llm_service.cleanup_unused_models.call_args
        assert kwargs["dry_run"] is True

    @pytest.mark.asyncio
    async def test_explicit_dry_run_false_deletes(self):
        """dry_run=False still performs the real deletion (OK status)."""
        feature, llm_service = await _feature()

        result = await feature.cleanup_models(dry_run=False)

        assert result.status is ToolResultStatus.OK
        _, kwargs = llm_service.cleanup_unused_models.call_args
        assert kwargs["dry_run"] is False

    @pytest.mark.asyncio
    async def test_threshold_days_passed_through(self):
        feature, llm_service = await _feature()

        await feature.cleanup_models(threshold_days=7)

        _, kwargs = llm_service.cleanup_unused_models.call_args
        assert kwargs["threshold_days"] == 7


class TestCleanupModelsDocs:
    @property
    def _schema(self):
        return ModelAgent.cleanup_models._tool_schema

    def _param(self, name):
        return next(p for p in self._schema["parameters"] if p.name == name)

    def test_dry_run_default_is_true_in_schema(self):
        assert self._param("dry_run").default is True

    def test_description_documents_safety_knobs(self):
        desc = self._schema["description"].lower()
        assert "destructive" in desc
        assert "dry_run" in desc or "dry-run" in desc
        assert "threshold_days" in desc
        assert "min_free_space_pct" in desc

    def test_param_docs_present(self):
        assert "preview" in self._param("dry_run").description.lower()
        assert "30" in self._param("threshold_days").description
