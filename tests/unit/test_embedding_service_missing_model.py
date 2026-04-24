"""Soften the first-run UX when Ollama hasn't pulled the embedding model
yet — #657.

Before: every embed call flooded the log with ``ERROR - Embedding
failed: model "nomic-embed-text" not found, try pulling it first``,
and during ``kestrel create`` that's dozens of red lines.

After: the first time a call hits the 404, the service logs ONE
WARNING with the exact fix command and suppresses follow-up lines
from the same instance. Anything else (real runtime error, Ollama
unreachable, etc.) still surfaces as ERROR.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.llm.embedding_service import EmbeddingService


class _FakeOllamaResponseError(Exception):
    """Mimics ``ollama._types.ResponseError`` — the class is private in
    the SDK, so we replicate its shape rather than importing it."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class TestModelNotFoundIsWarningNotError:

    def test_first_404_logs_warning_with_pull_command(self, caplog):
        """A 404 'not found, try pulling it first' is a setup issue —
        log ONE WARNING naming the fix command, never ERROR."""
        svc = EmbeddingService(model="nomic-embed-text")
        svc._client = MagicMock()
        svc._client.embed.side_effect = _FakeOllamaResponseError(
            'model "nomic-embed-text" not found, try pulling it first',
            status_code=404,
        )

        with caplog.at_level(logging.DEBUG):
            result = svc.embed("anything")

        assert result is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(warnings) == 1, f"Expected one WARNING, got {len(warnings)}"
        assert "nomic-embed-text" in warnings[0].getMessage()
        assert "ollama pull nomic-embed-text" in warnings[0].getMessage()
        assert errors == [], "404 must not surface as an ERROR"

    def test_repeat_404_is_deduplicated(self, caplog):
        """Agent startup calls embed() dozens of times (RAG indexing,
        memory seeding). One log line is plenty — further calls must
        not re-warn."""
        svc = EmbeddingService(model="nomic-embed-text")
        svc._client = MagicMock()
        svc._client.embed.side_effect = _FakeOllamaResponseError(
            'model "nomic-embed-text" not found, try pulling it first',
            status_code=404,
        )

        with caplog.at_level(logging.WARNING):
            for _ in range(10):
                svc.embed("x")

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, (
            f"First call should log once; subsequent 9 must be silent. "
            f"Got {len(warnings)} warnings."
        )

    def test_non_404_errors_still_surface_as_error(self, caplog):
        """Don't blanket-silence embed errors — a real runtime failure
        (connection refused, 500, malformed response) still deserves a
        loud ERROR so the user/oncall notices."""
        svc = EmbeddingService(model="nomic-embed-text")
        svc._client = MagicMock()
        svc._client.embed.side_effect = ConnectionRefusedError(
            "ollama server unreachable"
        )

        with caplog.at_level(logging.ERROR):
            svc.embed("x")

        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        assert "ollama server unreachable" in errors[0].getMessage()

    def test_batch_embed_also_softens_404(self, caplog):
        """The batch path hits the same Ollama endpoint — it must apply
        the same softening so a missing model during RAG indexing
        doesn't produce one ERROR per chunk."""
        svc = EmbeddingService(model="nomic-embed-text")
        svc._client = MagicMock()
        svc._client.embed.side_effect = _FakeOllamaResponseError(
            'model "nomic-embed-text" not found, try pulling it first',
            status_code=404,
        )

        with caplog.at_level(logging.WARNING):
            result = svc.embed_batch(["a", "b", "c"])

        assert result == [None, None, None]
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(warnings) == 1
        assert errors == []

    @pytest.mark.asyncio
    async def test_async_embed_also_softens_404(self, caplog):
        """Async path matches the sync path."""
        from unittest.mock import AsyncMock

        svc = EmbeddingService(model="nomic-embed-text")
        svc._async_client = MagicMock()
        svc._async_client.embed = AsyncMock(
            side_effect=_FakeOllamaResponseError(
                'model "nomic-embed-text" not found, try pulling it first',
                status_code=404,
            )
        )

        with caplog.at_level(logging.WARNING):
            await svc.aembed("x")

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(warnings) == 1
        assert errors == []
