"""Recency-driven model visibility (#2015).

These tests pin the behaviour that replaced "feature every undated lineage
root forever": featuring/ranking now key off the provider-supplied
``created_at`` (naming-agnostic) instead of parsing a version out of the model
id. That makes the system correct for numbered (``gpt-5.5``), codenamed
(``claude-opus-4-8``, future ``gpt-5.6-sol``), and dated schemes alike.
"""
from __future__ import annotations

from pathlib import Path

from kestrel_sovereign.llm.model_catalog import ModelCatalogService
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from kestrel_sovereign.llm.model_selection import _rank_cached_candidates

# Reference unix timestamps (UTC) used across scenarios.
_TS_2023 = "1677628800"   # 2023-03-01 — genuinely stale
_TS_2025_08 = "1754006400"  # 2025-08-01
_TS_2026_04 = "1776844800"  # 2026-04-23
_TS_2026_06 = "1780272000"  # 2026-06-01 — newest


def _catalog() -> ModelCatalogService:
    """A catalog with no on-disk overrides — exercises pure computed logic."""
    return ModelCatalogService(config_path=Path("/nonexistent-model-catalog.toml"))


def _chat(model_id: str, provider: str, created_at: str) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        provider=provider,
        display_name=model_id,
        category=ModelCategory.CHAT,
        supports_tools=True,
        created_at=created_at,
    )


def test_stale_canonical_roots_are_not_featured_and_are_deprecated():
    """gpt-3.5-turbo / gpt-4 must NOT be featured when a newer line exists, and
    must be flagged deprecated — the original symptom in #2015."""
    models = [
        _chat("gpt-3.5-turbo", "openai", _TS_2023),
        _chat("gpt-3.5-turbo-0125", "openai", _TS_2023),  # dated sibling
        _chat("gpt-4", "openai", _TS_2023),
        _chat("gpt-5.5", "openai", _TS_2026_06),
        _chat("gpt-5.5-2026-04-23", "openai", _TS_2026_04),  # dated snapshot
    ]
    enriched = {m.id: m for m in _catalog().enrich_models(models)}

    assert enriched["gpt-5.5"].is_featured is True
    assert enriched["gpt-3.5-turbo"].is_featured is False
    assert enriched["gpt-4"].is_featured is False
    assert enriched["gpt-3.5-turbo"].is_deprecated is True
    assert enriched["gpt-4"].is_deprecated is True
    # The undated alias is preferred over its dated snapshot.
    assert enriched["gpt-5.5-2026-04-23"].is_featured is False


def test_default_seed_is_best_featured_never_alphabetical():
    """The vendor's top-ranked featured chat model is a current model, not the
    alphabetical accident (gpt-3.5-turbo) that used to seed the dropdown."""
    models = [
        _chat("gpt-3.5-turbo", "openai", _TS_2023),
        _chat("gpt-3.5-turbo-0125", "openai", _TS_2023),
        _chat("gpt-5.5", "openai", _TS_2026_06),
        _chat("gpt-5.4", "openai", _TS_2026_04),
    ]
    enriched = _catalog().enrich_models(models)
    featured_chat = [m for m in enriched if m.is_featured and m.category == ModelCategory.CHAT]
    best = _rank_cached_candidates(featured_chat)[0]
    assert best.id != "gpt-3.5-turbo"
    assert best.id == "gpt-5.5"


def test_created_at_primary_ranking_survives_codename_tiers():
    """A codenamed newest model (gpt-5.6 Sol/Terra/Luna scheme) outranks an
    older numbered model even though its id carries no comparable version
    number — because ranking keys on created_at, not name-parsing."""
    older_numbered = _chat("gpt-5.5", "openai", _TS_2026_04)
    newer_codenamed = _chat("gpt-5.6-sol", "openai", _TS_2026_06)
    bare_codename = _chat("luna", "openai", _TS_2026_06)

    ranked = _rank_cached_candidates([older_numbered, newer_codenamed])
    assert ranked[0].id == "gpt-5.6-sol"

    # Even a bare codename with no number must not sink below an older numbered
    # model (the pre-#2015 _numeric_rank bug would have ranked "luna" last).
    ranked2 = _rank_cached_candidates([older_numbered, bare_codename])
    assert ranked2[0].id == "luna"


def test_anthropic_codenames_get_featured():
    """Codename schemes (opus/sonnet/haiku) previously yielded ZERO featured
    models because canonical-alias detection never fired. Recency featuring
    fixes that."""
    models = [
        _chat("claude-opus-4-8", "anthropic", "2026-05-28T00:00:00Z"),
        _chat("claude-sonnet-4-6", "anthropic", "2026-02-17T00:00:00Z"),
        _chat("claude-opus-4-1-20250805", "anthropic", "2025-08-05T00:00:00Z"),
    ]
    enriched = _catalog().enrich_models(models)
    featured = [m.id for m in enriched if m.is_featured]
    assert featured, "expected at least one featured anthropic model"
    assert "claude-opus-4-8" in featured


def test_non_chat_models_reclassified_and_unfeatured():
    """Audio/image/video/realtime models that OpenAI's API stamps as CHAT are
    inferred to their real category and kept out of the featured chat set."""
    models = [
        _chat("gpt-5.5", "openai", _TS_2026_06),
        _chat("tts-1", "openai", _TS_2026_06),
        _chat("gpt-4o-mini-transcribe", "openai", _TS_2026_06),
        _chat("gpt-realtime", "openai", _TS_2026_06),
        _chat("gpt-image-2", "openai", _TS_2026_06),
        _chat("grok-imagine-video", "xai", _TS_2026_06),
    ]
    enriched = {m.id: m for m in _catalog().enrich_models(models)}
    assert enriched["tts-1"].category == ModelCategory.AUDIO
    assert enriched["gpt-4o-mini-transcribe"].category == ModelCategory.AUDIO
    assert enriched["gpt-realtime"].category == ModelCategory.AUDIO
    assert enriched["gpt-image-2"].category == ModelCategory.IMAGE
    assert enriched["grok-imagine-video"].category == ModelCategory.IMAGE
    for non_chat in ("tts-1", "gpt-4o-mini-transcribe", "gpt-realtime", "gpt-image-2"):
        assert enriched[non_chat].is_featured is False


def test_stale_cached_featured_flag_does_not_survive():
    """Regression for the codex P2 on #2015: a model arriving with a stale
    ``is_featured=True`` (e.g. loaded from a pre-upgrade discovery cache) must
    be RE-decided by the recency rules, not preserved. Otherwise gpt-3.5-turbo
    stays featured on upgrade and bypasses the per-vendor cap."""
    stale = _chat("gpt-3.5-turbo", "openai", _TS_2023)
    stale.is_featured = True  # as if loaded from the old all-featured cache
    fresh = _chat("gpt-5.5", "openai", _TS_2026_06)
    fresh.is_featured = True

    enriched = {m.id: m for m in _catalog().enrich_models([stale, fresh])}
    assert enriched["gpt-3.5-turbo"].is_featured is False
    assert enriched["gpt-5.5"].is_featured is True


def test_pinned_configured_model_stays_featured_without_created_at():
    """Regression for codex r2 on #2015: a concrete model configured in
    kestrel.toml (passed as ``pinned_featured``) must stay featured even when it
    is not in the recency top-N and carries no ``created_at`` — operator intent
    survives the authoritative recompute."""
    cat = _catalog()
    cap = cat._featured_per_vendor
    # Fill the recency top-N with newer models so the pinned one is outside it.
    models = [
        _chat(f"gpt-5.{i}", "openai", str(int(_TS_2026_06) - i * 1000))
        for i in range(cap + 2)
    ]
    pinned = ModelInfo(
        id="gpt-private-beta", provider="openai",
        display_name="gpt-private-beta", category=ModelCategory.CHAT,
        supports_tools=True, created_at=None,
    )
    models.append(pinned)

    enriched = {m.id: m for m in cat.enrich_models(
        models, pinned_featured={"openai": {"gpt-private-beta"}})}
    assert enriched["gpt-private-beta"].is_featured is True


def test_featured_set_is_bounded_per_vendor():
    """Featuring is capped by the visibility dial — never 'everything featured'
    (the regression that overwhelmed the dropdown)."""
    cat = _catalog()
    cap = cat._featured_per_vendor
    # More recent chat models than the cap.
    models = [
        _chat(f"gpt-5.{i}", "openai", str(int(_TS_2026_06) - i * 1000))
        for i in range(cap + 6)
    ]
    enriched = cat.enrich_models(models)
    featured = [m for m in enriched if m.is_featured]
    assert len(featured) <= cap
