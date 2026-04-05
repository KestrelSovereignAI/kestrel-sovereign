"""Tests for large tool result persistence — preview with head+tail."""

import pytest

from kestrel_sovereign.agent.orchestrator_engine import (
    _build_persisted_preview,
    PREVIEW_HEAD_CHARS,
    PREVIEW_TAIL_CHARS,
)


class TestBuildPersistedPreview:
    def test_small_result_head_only(self):
        """Result smaller than head+tail threshold gets head only."""
        result = "x" * 2200
        preview = _build_persisted_preview(result, "Read", len(result))
        assert "[Large tool output" in preview
        assert "Read" in preview
        assert "2,200 chars" in preview
        assert f"first {PREVIEW_HEAD_CHARS} chars" in preview
        # No tail section for small results
        assert "Tail" not in preview

    def test_large_result_has_head_and_tail(self):
        """Result larger than head+tail gets both sections."""
        result = "H" * 3000 + "M" * 5000 + "T" * 1000
        preview = _build_persisted_preview(result, "Bash", len(result))
        assert "[Large tool output" in preview
        assert "Bash" in preview
        assert f"first {PREVIEW_HEAD_CHARS} chars" in preview
        assert f"last {PREVIEW_TAIL_CHARS} chars" in preview
        # Head starts with H's
        assert "HHH" in preview
        # Tail ends with T's
        assert "TTT" in preview

    def test_omitted_count_correct(self):
        """The omitted chars count is accurate."""
        total = 10000
        result = "x" * total
        preview = _build_persisted_preview(result, "Grep", total)
        omitted = total - PREVIEW_HEAD_CHARS - PREVIEW_TAIL_CHARS
        assert f"{omitted:,} chars omitted" in preview

    def test_preserves_tool_name(self):
        result = "x" * 5000
        preview = _build_persisted_preview(result, "web_search", len(result))
        assert "web_search" in preview

    def test_formats_large_numbers_with_commas(self):
        result = "x" * 50000
        preview = _build_persisted_preview(result, "Bash", 50000)
        assert "50,000" in preview

    def test_exact_boundary_head_plus_tail(self):
        """Result exactly at head+tail boundary — no tail needed."""
        exact = PREVIEW_HEAD_CHARS + PREVIEW_TAIL_CHARS
        result = "x" * exact
        preview = _build_persisted_preview(result, "Read", exact)
        # At exact boundary, tail is empty string (len check)
        assert "Tail" not in preview

    def test_one_char_over_boundary(self):
        """Result one char over head+tail — tail appears."""
        over = PREVIEW_HEAD_CHARS + PREVIEW_TAIL_CHARS + 1
        result = "A" * PREVIEW_HEAD_CHARS + "B" + "C" * PREVIEW_TAIL_CHARS
        preview = _build_persisted_preview(result, "Read", over)
        assert "Tail" in preview
