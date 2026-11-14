"""Trusted resource provenance has a byte budget independent of retry IDs."""

import pytest

from kestrel_sovereign.agent.invocation import (
    MAX_INVOCATION_ID_LENGTH,
    MAX_PROVENANCE_LOCATOR_BYTES,
    request_provenance,
    validate_invocation_id,
)


def test_resource_provenance_and_invocation_ids_keep_distinct_size_contracts():
    locator = "https://kestrel.example/" + "r" * MAX_INVOCATION_ID_LENGTH
    provenance = request_provenance(source_kind="http_request", source_locator=locator)
    assert provenance.source_locator == locator
    with pytest.raises(ValueError, match="invocation id"):
        validate_invocation_id(locator)
    with pytest.raises(ValueError, match="provenance source locator"):
        request_provenance(source_kind="http_request", source_locator="r" * (MAX_PROVENANCE_LOCATOR_BYTES + 1))


def test_configured_provenance_budget_counts_actual_utf8_bytes():
    locator = "https://kestrel.example/" + "雪" * 300
    budget = len(locator.encode("utf-8"))
    assert request_provenance(source_kind="http_request", source_locator=locator,
        source_locator_max_bytes=budget).source_locator == locator
    with pytest.raises(ValueError, match="bytes"):
        request_provenance(source_kind="http_request", source_locator=locator, source_locator_max_bytes=budget - 1)


@pytest.mark.parametrize("locator", [None, 1, "", "\ud800"])
def test_provenance_locator_rejects_nontext_empty_and_invalid_unicode(locator):
    with pytest.raises(ValueError, match="provenance source locator"):
        request_provenance(source_kind="http_request", source_locator=locator)


@pytest.mark.parametrize("budget", [True, 0, -1, 3.5])
def test_provenance_locator_budget_must_be_a_positive_integer(budget):
    with pytest.raises(ValueError, match="positive integer"):
        request_provenance(source_kind="http_request", source_locator="https://kestrel.example/",
            source_locator_max_bytes=budget)
