"""
Regression tests proving ensemble mode (CW-005) stays gone.

The original suite either permanently skipped on CI (no Ollama) or used
``hasattr(...)`` guards that quietly passed when the attribute had been
re-added. This rewrite is fully deterministic and independent of any
LLM provider:

1.  Static source scan over the production package: no symbol that
    historically belonged to the ensemble pathway may reappear.
2.  Live ``KestrelAgent`` API surface check: the ensemble attributes /
    methods MUST NOT exist (no ``hasattr`` escape hatch).
3.  Persisted conversation rows must never carry ensemble metadata.

The previous "exactly N LLM calls per query" tests were permaskipped
because driving ``process_input`` through a fake LLM requires stubbing
the full provider surface (state_of_mind, audit, generate_with_messages,
streaming, model registry, …). They added no signal beyond what the
static scan + surface check already prove and have been removed rather
than left as dead skip markers.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode


# ---------------------------------------------------------------------------
# Static source scan
# ---------------------------------------------------------------------------

# Symbols that only ever existed in the now-removed ensemble pathway. Any
# reintroduction is an immediate regression — we want the test to fail loudly,
# not silently pass.
ENSEMBLE_FORBIDDEN = (
    "process_ensemble_query",
    "_process_ensemble_query",
    "ensemble_size",
    "ensemble_consensus",
    "EnsembleAgent",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "kestrel_sovereign"


def _iter_source_files() -> list[Path]:
    return [p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_ensemble_symbols_in_package_source():
    """No production source file references any ensemble-only symbol."""
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, ENSEMBLE_FORBIDDEN)) + r")\b")
    offenders: list[tuple[Path, int, str, str]] = []

    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = pattern.search(line)
            if match:
                offenders.append(
                    (path.relative_to(REPO_ROOT), lineno, match.group(1), line.strip())
                )

    assert offenders == [], (
        "Ensemble symbols reintroduced into production source:\n"
        + "\n".join(
            f"  {p}:{lineno} [{sym}] {snippet}" for p, lineno, sym, snippet in offenders
        )
    )


# ---------------------------------------------------------------------------
# Live agent API surface
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest_asyncio.fixture
async def real_llm_agent(temp_db):
    """Agent backed by a real LLMService — for surface checks only (no calls made)."""
    llm = LLMService()
    agent = KestrelAgent(
        did="did:pkh:eip155:1:test_ensemble_surface",
        storage_path=temp_db,
        llm_service=llm,
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()
    try:
        yield agent
    finally:
        await agent.shutdown()
        await llm.close()


@pytest.mark.asyncio
async def test_kestrel_agent_has_no_ensemble_attributes(real_llm_agent):
    """The live agent MUST NOT expose any ensemble attribute or method."""
    for name in ENSEMBLE_FORBIDDEN:
        assert not hasattr(real_llm_agent, name), (
            f"KestrelAgent unexpectedly exposes '{name}' — ensemble code reintroduced?"
        )


@pytest.mark.asyncio
async def test_no_ensemble_metadata_in_conversation(real_llm_agent):
    """Stored conversation rows must never carry ensemble_consensus metadata."""
    await real_llm_agent.privacy_agent.add_conversation(
        role="assistant",
        content="Test response",
        metadata={"test": True},
    )

    history = await real_llm_agent.storage.get_conversation_history(limit=10)
    assert history, "expected at least one conversation row"
    for msg in history:
        metadata = msg.get("metadata", {}) or {}
        assert "ensemble_consensus" not in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
