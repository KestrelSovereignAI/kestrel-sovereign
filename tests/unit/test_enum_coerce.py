"""Unit tests for the shared enum-parameter coercion helpers (#1923)."""

from __future__ import annotations

import pytest

from kestrel_sovereign.features.enum_coerce import (
    LOW_NORMAL_HIGH_URGENT_ALIASES,
    STATUS_DONE_CANCELLED_ALIASES,
    coerce_enum,
    normalize_choice,
)


class TestNormalizeChoice:
    def test_lowercases_and_strips(self):
        assert normalize_choice("  HIGH ") == "high"

    def test_maps_known_synonym(self):
        assert normalize_choice("medium", LOW_NORMAL_HIGH_URGENT_ALIASES) == "normal"
        assert normalize_choice("Critical", LOW_NORMAL_HIGH_URGENT_ALIASES) == "urgent"

    def test_unknown_passes_through_lowercased(self):
        # Genuine typos survive (lower-cased) so the caller's validator still
        # rejects them with a helpful, value-listing message.
        assert normalize_choice("supercritical", LOW_NORMAL_HIGH_URGENT_ALIASES) == "supercritical"

    def test_non_string_passes_through_untouched(self):
        assert normalize_choice(None) is None
        assert normalize_choice(7) == 7

    def test_no_alias_map_just_casefolds(self):
        assert normalize_choice("Done") == "done"

    def test_status_aliases(self):
        for syn, canon in [
            ("completed", "done"), ("complete", "done"), ("finished", "done"),
            ("canceled", "cancelled"), ("abandoned", "cancelled"),
        ]:
            assert normalize_choice(syn, STATUS_DONE_CANCELLED_ALIASES) == canon


class TestCoerceEnum:
    def test_accepts_canonical(self):
        value, err = coerce_enum("high", ["low", "normal", "high"], field="priority")
        assert value == "high"
        assert err is None

    def test_normalizes_alias(self):
        value, err = coerce_enum(
            "medium", ["low", "normal", "high"], field="priority",
            aliases=LOW_NORMAL_HIGH_URGENT_ALIASES,
        )
        assert value == "normal"
        assert err is None

    def test_rejects_typo_with_value_listing_error(self):
        value, err = coerce_enum("huge", ["low", "normal", "high"], field="priority")
        assert value is None
        assert err == "priority must be one of: low, normal, high (got 'huge')"

    def test_default_used_for_blank_or_none(self):
        for blank in (None, "", "   "):
            value, err = coerce_enum(blank, ["a", "b"], field="x", default="a")
            assert (value, err) == ("a", None)

    def test_blank_without_default_is_rejected(self):
        value, err = coerce_enum("", ["a", "b"], field="x")
        assert value is None
        assert "must be one of: a, b" in err
