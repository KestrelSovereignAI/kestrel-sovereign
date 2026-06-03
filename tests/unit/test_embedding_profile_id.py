"""End-to-end tests for #1477 embedding_profile_id stamping + kNN filter.

Covers:
- Deterministic profile id derivation.
- Profile id changes when any input changes (provider, model, dim,
  space_id, normalized).
- ``EmbeddingProfile.profile_id`` is 12 hex chars.
- ``ProviderEmbeddingService.describe()`` builds the right profile
  from capability metadata; returns None when metadata missing.
- ``EmbeddingService.describe()`` (legacy Ollama path) handles the
  two known models + returns None for unknown.
- ``upsert_embedding_profile`` writes the registry row idempotently;
  in-process cache prevents repeat UPSERTs.
- ``cosine_similarity`` length guard returns 0.0 instead of running.

These tests are pure-Python / SQLite — no real DB needed.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.embedding_service import (
    EmbeddingProfile,
    EmbeddingService,
    ProviderEmbeddingService,
    cosine_similarity,
    derive_embedding_profile,
)
from kestrel_sovereign.storage.sqla.embedding_profile import (
    _clear_profile_upsert_cache_for_tests,
    upsert_embedding_profile,
)


# --- derive_embedding_profile ------------------------------------------------


def test_profile_id_is_deterministic():
    a = derive_embedding_profile(provider="openai", model="text-embedding-3-small", dim=1536)
    b = derive_embedding_profile(provider="openai", model="text-embedding-3-small", dim=1536)
    assert a.profile_id == b.profile_id


def test_profile_id_changes_on_provider_change():
    a = derive_embedding_profile(provider="openai", model="m", dim=768)
    b = derive_embedding_profile(provider="vertex", model="m", dim=768)
    assert a.profile_id != b.profile_id


def test_profile_id_changes_on_model_change():
    a = derive_embedding_profile(provider="openai", model="m1", dim=768)
    b = derive_embedding_profile(provider="openai", model="m2", dim=768)
    assert a.profile_id != b.profile_id


def test_profile_id_changes_on_dim_change():
    a = derive_embedding_profile(provider="openai", model="m", dim=768)
    b = derive_embedding_profile(provider="openai", model="m", dim=1536)
    assert a.profile_id != b.profile_id


def test_profile_id_changes_on_normalized_flag():
    a = derive_embedding_profile(provider="openai", model="m", dim=768, normalized=False)
    b = derive_embedding_profile(provider="openai", model="m", dim=768, normalized=True)
    assert a.profile_id != b.profile_id


def test_profile_id_changes_on_space_id_override():
    """Operators that force-merge two providers must change the id."""
    a = derive_embedding_profile(provider="openai", model="m", dim=768)
    b = derive_embedding_profile(
        provider="openai", model="m", dim=768, space_id="custom-merged-space"
    )
    assert a.profile_id != b.profile_id


def test_profile_id_is_12_hex_chars():
    p = derive_embedding_profile(provider="openai", model="m", dim=768)
    assert len(p.profile_id) == 12
    int(p.profile_id, 16)  # raises if non-hex


def test_derive_rejects_blank_fields():
    with pytest.raises(ValueError):
        derive_embedding_profile(provider="", model="m", dim=768)
    with pytest.raises(ValueError):
        derive_embedding_profile(provider="openai", model="", dim=768)
    with pytest.raises(ValueError):
        derive_embedding_profile(provider="openai", model="m", dim=0)
    with pytest.raises(ValueError):
        derive_embedding_profile(provider="openai", model="m", dim=-1)


def test_derive_space_id_defaults_to_provider_colon_model():
    p = derive_embedding_profile(provider="openai", model="my-model", dim=768)
    assert p.space_id == "openai:my-model"


def test_space_id_override_force_merges_distinct_providers():
    """#1477 codex P2 regression: two profiles with the SAME explicit
    ``space_id`` (override) must hash to the same id even when their
    provider / model labels differ. This is the whole point of the
    override — it tells the system "these two routes wrap the same
    upstream embedding space, treat their rows as compatible."
    """
    a = derive_embedding_profile(
        provider="openai",
        model="text-embedding-3-small",
        dim=1536,
        space_id="upstream-bge-large",
    )
    b = derive_embedding_profile(
        provider="openrouter",
        model="openai/text-embedding-3-small",
        dim=1536,
        space_id="upstream-bge-large",
    )
    assert a.profile_id == b.profile_id, (
        "space_id override must force-merge — same space_id + dim + "
        "normalized → same profile id regardless of provider/model labels."
    )


def test_same_space_id_but_different_dim_yields_different_ids():
    """Dim is part of the hash even under override — vectors of
    different lengths can never live in the same space."""
    a = derive_embedding_profile(provider="openai", model="m", dim=768, space_id="custom")
    b = derive_embedding_profile(provider="openai", model="m", dim=1536, space_id="custom")
    assert a.profile_id != b.profile_id


# --- ProviderEmbeddingService.describe ---------------------------------------


def _stub_provider(*, vendor="openai", model="text-embedding-3-small", dim=1536, **extra):
    return {
        "name": f"{vendor}:api",
        "vendor": vendor,
        "adapter": MagicMock(),
        "client": object(),
        "capabilities": {
            "embedding_model": model,
            "embedding_dim": dim,
            **extra,
        },
    }


def test_provider_embedding_service_describe_uses_capability_metadata():
    svc = ProviderEmbeddingService(_stub_provider())
    profile = svc.describe()
    assert profile is not None
    assert profile.provider == "openai"
    assert profile.model == "text-embedding-3-small"
    assert profile.dim == 1536
    assert profile.normalized is False
    assert profile.space_id == "openai:text-embedding-3-small"


def test_provider_embedding_service_describe_honors_normalized_flag():
    svc = ProviderEmbeddingService(
        _stub_provider(embedding_normalized=True)
    )
    profile = svc.describe()
    assert profile is not None
    assert profile.normalized is True


def test_provider_embedding_service_describe_honors_space_id_override():
    svc = ProviderEmbeddingService(
        _stub_provider(embedding_space_id="upstream-bge-base")
    )
    profile = svc.describe()
    assert profile is not None
    assert profile.space_id == "upstream-bge-base"


def test_provider_embedding_service_describe_returns_none_without_model():
    p = _stub_provider()
    p["capabilities"]["embedding_model"] = None
    svc = ProviderEmbeddingService(p)
    assert svc.describe() is None
    assert svc.current_profile_id() is None


def test_provider_embedding_service_describe_returns_none_without_dim():
    p = _stub_provider()
    p["capabilities"]["embedding_dim"] = None
    svc = ProviderEmbeddingService(p)
    assert svc.describe() is None


def test_provider_embedding_service_falls_back_to_name_when_no_vendor():
    p = _stub_provider()
    del p["vendor"]
    svc = ProviderEmbeddingService(p)
    profile = svc.describe()
    assert profile is not None
    assert profile.provider == "openai:api"  # falls back to ``name``


# --- EmbeddingService (legacy Ollama) describe --------------------------------


def test_embedding_service_describe_known_model_nomic():
    svc = EmbeddingService.__new__(EmbeddingService)
    svc.model = "nomic-embed-text"
    profile = svc.describe()
    assert profile is not None
    assert profile.provider == "ollama"
    assert profile.dim == 768


def test_embedding_service_describe_known_model_mxbai():
    svc = EmbeddingService.__new__(EmbeddingService)
    svc.model = "mxbai-embed-large"
    profile = svc.describe()
    assert profile is not None
    assert profile.dim == 1024


def test_embedding_service_describe_unknown_model_returns_none():
    svc = EmbeddingService.__new__(EmbeddingService)
    svc.model = "some-custom-model"
    assert svc.describe() is None
    assert svc.current_profile_id() is None


# --- cosine_similarity length guard ------------------------------------------


def test_cosine_similarity_mismatched_lengths_returns_zero():
    """#1477 defense-in-depth — mixed-dim vectors must not cosine."""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_empty_returns_zero():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0], []) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_cosine_similarity_normal_case():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


# --- upsert_embedding_profile ------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_caches_per_process():
    """Idempotency: second call for the same id is a no-op via cache."""
    _clear_profile_upsert_cache_for_tests()

    db = MagicMock()
    db.backend_type = "sqlite"
    db.execute = AsyncMock()

    profile = derive_embedding_profile(
        provider="openai", model="text-embedding-3-small", dim=1536
    )
    svc = MagicMock()
    svc.describe = MagicMock(return_value=profile)

    await upsert_embedding_profile(db, svc, profile.profile_id)
    await upsert_embedding_profile(db, svc, profile.profile_id)

    assert db.execute.await_count == 1, (
        "Second call must hit the per-process cache and skip the UPSERT."
    )


@pytest.mark.asyncio
async def test_upsert_handles_postgres_dialect():
    _clear_profile_upsert_cache_for_tests()

    db = MagicMock()
    db.backend_type = "postgres"
    db.execute = AsyncMock()

    profile = derive_embedding_profile(provider="openai", model="m", dim=768)
    svc = MagicMock()
    svc.describe = MagicMock(return_value=profile)

    await upsert_embedding_profile(db, svc, profile.profile_id)
    sql = db.execute.call_args.args[0]
    assert "ON CONFLICT" in sql


@pytest.mark.asyncio
async def test_upsert_swallows_failure():
    """A failed upsert must NEVER bubble — registry is operator-visibility-only."""
    _clear_profile_upsert_cache_for_tests()

    db = MagicMock()
    db.backend_type = "sqlite"
    db.execute = AsyncMock(side_effect=RuntimeError("table missing"))

    profile = derive_embedding_profile(provider="openai", model="m", dim=768)
    svc = MagicMock()
    svc.describe = MagicMock(return_value=profile)

    # Must not raise.
    await upsert_embedding_profile(db, svc, profile.profile_id)


@pytest.mark.asyncio
async def test_upsert_skips_when_service_describes_to_none():
    """A service that can't describe itself yields a no-op."""
    _clear_profile_upsert_cache_for_tests()

    db = MagicMock()
    db.backend_type = "sqlite"
    db.execute = AsyncMock()

    svc = MagicMock()
    svc.describe = MagicMock(return_value=None)

    await upsert_embedding_profile(db, svc, "some-id")
    assert db.execute.await_count == 0
