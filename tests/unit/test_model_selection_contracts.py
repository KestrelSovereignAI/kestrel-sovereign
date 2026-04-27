"""Contracts for shared config-driven model selection helpers."""

from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from kestrel_sovereign.llm.model_selection import (
    _numeric_rank,
    _rank_cached_candidates,
    resolve_provider_default,
)


def test_resolve_provider_default_prefers_explicit_model():
    resolved = resolve_provider_default(
        "openai",
        llm_config={"openai": {"model": "gpt-5.1"}},
        catalog_config={},
    )

    assert resolved == "gpt-5.1"


def test_resolve_provider_default_uses_selection_hints_against_cached_models():
    resolved = resolve_provider_default(
        "anthropic",
        llm_config={"anthropic": {"model": "auto", "selection_hints": ["opus", "sonnet"]}},
        catalog_config={},
        cached_models=[
            ModelInfo(
                id="claude-sonnet-4-6",
                provider="anthropic",
                display_name="Claude Sonnet 4.6",
                category=ModelCategory.CHAT,
                supports_tools=True,
            ),
            ModelInfo(
                id="claude-opus-4-5-20251101",
                provider="anthropic",
                display_name="Claude Opus 4.5",
                category=ModelCategory.CHAT,
                supports_tools=True,
            ),
        ],
    )

    assert resolved == "claude-opus-4-5-20251101"


def test_resolve_provider_default_prefers_newest_matching_model_from_discovery():
    resolved = resolve_provider_default(
        "anthropic",
        llm_config={"anthropic": {"model": "auto", "selection_hints": ["sonnet"]}},
        catalog_config={},
        cached_models=[
            ModelInfo(
                id="claude-sonnet-4-20250514",
                provider="anthropic",
                display_name="Claude Sonnet 4",
                category=ModelCategory.CHAT,
                supports_tools=True,
                created_at="2025-05-14T00:00:00Z",
            ),
            ModelInfo(
                id="claude-sonnet-4-5-20250929",
                provider="anthropic",
                display_name="Claude Sonnet 4.5",
                category=ModelCategory.CHAT,
                supports_tools=True,
                created_at="2025-09-29T00:00:00Z",
            ),
            ModelInfo(
                id="claude-sonnet-4-6",
                provider="anthropic",
                display_name="Claude Sonnet 4.6",
                category=ModelCategory.CHAT,
                supports_tools=True,
                created_at="2026-04-13T00:00:00Z",
            ),
        ],
    )

    assert resolved == "claude-sonnet-4-6"


def test_resolve_provider_default_uses_vendor_catalog_for_subscription_route():
    """A vendor's routes share the discovery catalog.

    Under the vendor/route architecture, ``openai:plan`` (ChatGPT subscription)
    is a route on the ``openai`` vendor — not a separate provider. Model
    selection must read the vendor's catalog, not look for a separate
    pseudo-provider.
    """
    resolved = resolve_provider_default(
        "openai:plan",
        llm_config={
            "vendors": {
                "openai": {
                    "routes": {
                        "api": {"model": "auto", "adapter": "OpenAIAdapter"},
                        "plan": {"model": "auto", "adapter": "CodexAdapter",
                                 "selection_hints": ["gpt-5.4"]},
                    }
                }
            }
        },
        catalog_config={},
        cached_models=[
            ModelInfo(
                id="gpt-5.4",
                provider="openai",
                display_name="GPT-5.4",
                category=ModelCategory.CHAT,
                supports_tools=True,
            ),
            ModelInfo(
                id="gpt-5.4-mini",
                provider="openai",
                display_name="GPT-5.4 Mini",
                category=ModelCategory.CHAT,
                supports_tools=True,
            ),
        ],
    )

    assert resolved == "gpt-5.4"


# ----------------------------------------------------------------------
# Regressions for the ranker (post #702 fixes)
# ----------------------------------------------------------------------


def test_rank_cached_candidates_handles_unix_epoch_created_at():
    """Newer Unix-epoch ``created_at`` must rank before older.

    OpenAIAdapter stores ``created_at = str(model.created)`` — i.e. a
    stringified Unix epoch like ``"1700000000"``. The previous
    ``_rank_auto_candidates`` ranker called
    ``datetime.fromisoformat(created_at)`` which always raised on those
    values; the except returned 0.0 for every model and the function
    fell through to alphabetical order. Consolidating onto
    ``_rank_cached_candidates`` (which uses ``re.findall(r"\\d+", ...)``)
    makes recency rank correctly for both ISO and epoch formats.
    """
    older = ModelInfo(
        id="gpt-4-mini",
        provider="openai",
        display_name="GPT-4 Mini",
        category=ModelCategory.CHAT,
        supports_tools=True,
        created_at="1600000000",  # 2020-09-13
    )
    newer = ModelInfo(
        id="gpt-5-mini",
        provider="openai",
        display_name="GPT-5 Mini",
        category=ModelCategory.CHAT,
        supports_tools=True,
        created_at="1700000000",  # 2023-11-14
    )

    # Pass them in the wrong order on purpose — the ranker has to fix it.
    ranked = _rank_cached_candidates([older, newer])

    assert ranked[0].id == "gpt-5-mini"
    assert ranked[1].id == "gpt-4-mini"


def test_rank_cached_candidates_also_handles_iso_created_at():
    """ISO timestamps still rank correctly — the ranker is format-agnostic."""
    older = ModelInfo(
        id="claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        category=ModelCategory.CHAT,
        supports_tools=True,
        created_at="2025-09-29T00:00:00Z",
    )
    newer = ModelInfo(
        id="claude-sonnet-4-6",
        provider="anthropic",
        display_name="Claude Sonnet 4.6",
        category=ModelCategory.CHAT,
        supports_tools=True,
        created_at="2026-03-15T00:00:00Z",
    )

    ranked = _rank_cached_candidates([older, newer])

    assert ranked[0].id == "claude-sonnet-4-6"


def test_rank_auto_candidates_delegates_to_cached_candidates():
    """``_rank_auto_candidates`` must produce the same order as the canonical
    ranker. Previously each had its own implementation that disagreed on the
    epoch-string case; consolidation keeps any future ranker fix in one place.
    """
    from kestrel_sovereign.llm.model_discovery import ModelDiscoveryMixin

    class _Harness(ModelDiscoveryMixin):
        pass

    harness = _Harness()
    models = [
        ModelInfo(
            id="gpt-4-mini",
            provider="openai",
            display_name="GPT-4 Mini",
            category=ModelCategory.CHAT,
            supports_tools=True,
            created_at="1600000000",
        ),
        ModelInfo(
            id="gpt-5-mini",
            provider="openai",
            display_name="GPT-5 Mini",
            category=ModelCategory.CHAT,
            supports_tools=True,
            created_at="1700000000",
        ),
    ]

    discovery_ranked = harness._rank_auto_candidates(models)
    canonical_ranked = _rank_cached_candidates(models)

    assert [m.id for m in discovery_ranked] == [m.id for m in canonical_ranked]
    assert discovery_ranked[0].id == "gpt-5-mini"


def test_numeric_rank_strips_iso_dates_in_all_separators():
    """``_numeric_rank``'s date stripper must catch contiguous, dashed, and
    space-separated forms.

    The display-name path normalizes a model id like
    ``gpt-audio-mini-2025-12-15`` into ``"Gpt Audio Mini 2025 12 15"``. The
    legacy regex ``r"20\\d{6}"`` only matched the contiguous form; the
    spaced version leaked ``2025`` into the rank tuple, which would outrank
    the actual model version (a 5 or 6 digit). Broaden the regex to handle
    all three separators.
    """
    plain = _numeric_rank("Gpt 5.4 Mini")
    contiguous = _numeric_rank("Gpt 5.4 Mini 20251215")
    dashed = _numeric_rank("Gpt 5.4 Mini 2025-12-15")
    underscored = _numeric_rank("Gpt 5.4 Mini 2025_12_15")
    spaced = _numeric_rank("Gpt 5.4 Mini 2025 12 15")

    assert plain == contiguous == dashed == underscored == spaced
    # Sanity: the model version (5.4) made it into the rank, the date didn't.
    assert plain == (-5, -4, 0, 0)


def test_rank_cached_candidates_uses_id_not_display_name_for_numeric_rank():
    """When the same model has display_name with embedded date but a clean
    id, ranking on ``id`` (not ``display_name``) prevents the date from
    leaking into the version comparison.

    Pre-fix scenario: two GPT-5 variants where the dated one had the same
    underlying version but a date-stamped id. The display_name's spaced
    date polluted the rank. Switching to ``id`` is canonical.
    """
    dated = ModelInfo(
        id="gpt-5.4-mini-2025-12-15",
        provider="openai",
        display_name="Gpt 5.4 Mini 2025 12 15",
        category=ModelCategory.CHAT,
        supports_tools=True,
        created_at="1700000000",
    )
    canonical = ModelInfo(
        id="gpt-5.4-mini",
        provider="openai",
        display_name="GPT-5.4 Mini",
        category=ModelCategory.CHAT,
        supports_tools=True,
        # Older creation time — but rankers should still treat both as the
        # same generation since the date got stripped out of the numeric rank.
        created_at="1600000000",
    )

    ranked = _rank_cached_candidates([dated, canonical])

    # The numeric rank (5.4) is identical for both, so the next tiebreaker
    # — created_at — wins. The newer dated id sorts first, but the spaced
    # date in display_name does not promote the dated entry artificially.
    # If the ranker leaked the spaced "2025" into the numeric rank, the
    # dated entry would crush the canonical one regardless of created_at,
    # which is exactly the bug.
    assert ranked[0].id == "gpt-5.4-mini-2025-12-15"  # newer epoch wins
    # Now flip created_at to confirm rank is driven by epoch, not date-leak.
    dated.created_at = "1500000000"  # older
    canonical.created_at = "1700000000"  # newer
    ranked_flipped = _rank_cached_candidates([dated, canonical])
    assert ranked_flipped[0].id == "gpt-5.4-mini"
