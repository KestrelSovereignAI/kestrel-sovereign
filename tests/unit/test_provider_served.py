"""What a provider's own response says served a call is kept only as short
identifiers (#4198)."""

from types import SimpleNamespace

import pytest

from kestrel_sovereign.llm.provider_served import (
    MAX_IDENTIFIER_CHARS,
    attach_provider_served,
    provider_served,
    response_provider_served,
)


def test_the_recorded_shape_names_the_api_model_and_response_id():
    response = attach_provider_served(
        SimpleNamespace(),
        provider_served("responses", model="gpt-6.2-sol-2026-11-01", response_id="resp_1"),
    )

    assert response_provider_served(response) == {
        "api": "responses", "model": "gpt-6.2-sol-2026-11-01", "response_id": "resp_1",
    }


@pytest.mark.parametrize("value", [
    None, "", 7, ["gpt"], object(), "x" * (MAX_IDENTIFIER_CHARS + 1),
], ids=["none", "empty", "int", "list", "object", "too-long"])
def test_a_value_that_is_not_a_short_identifier_is_not_recorded(value):
    served = provider_served("responses", model=value, response_id="resp_1")

    assert served.record() == {"api": "responses", "model": None, "response_id": "resp_1"}
    assert provider_served("responses", model=value, response_id=value) is None


def test_a_response_with_no_evidence_records_none():
    response = attach_provider_served(SimpleNamespace(), None)

    assert response_provider_served(response) is None
    assert response_provider_served(SimpleNamespace(_kestrel_provider_served="forged")) is None


def test_a_usage_only_copy_keeps_what_served_the_response():
    """A failed attempt is recorded from a usage-only copy of its response;
    the provider's model and response id survive the copy, its text does not."""
    from kestrel_sdk.llm import LLMResponse

    from kestrel_sovereign.llm.completion_protocol import response_usage_only

    served = provider_served("responses", model="gpt-6.2-sol-2026-11-01", response_id="resp_1")
    response = attach_provider_served(
        LLMResponse(content="secret text", input_tokens=3, output_tokens=4), served,
    )

    copy = response_usage_only(response)

    assert copy.content is None
    assert (copy.input_tokens, copy.output_tokens) == (3, 4)
    assert response_provider_served(copy) == served.record()
    assert response_provider_served(response_usage_only(LLMResponse(input_tokens=1))) is None


def _evidence(response_id: str):
    from kestrel_sdk.llm import LLMResponse

    return attach_provider_served(
        LLMResponse(input_tokens=5, output_tokens=6),
        provider_served("responses", model="gpt-6.2-sol", response_id=response_id),
    )


def test_reported_usage_reads_a_providers_own_failure_through_its_wrappers():
    from kestrel_sovereign.llm.output_ceiling import ProviderReportedFailure, reported_usage

    failure = ProviderReportedFailure("the provider failed the response")
    failure.usage = _evidence("resp_failed")
    wrapper = RuntimeError("route failed")
    wrapper.__cause__ = failure

    assert reported_usage(failure) is failure.usage
    assert reported_usage(wrapper) is failure.usage
    assert reported_usage(ProviderReportedFailure("no evidence")) is None


def test_an_incomplete_generation_is_read_before_a_provider_failure():
    from kestrel_sovereign.llm.output_ceiling import (
        IncompleteGenerationError,
        ProviderReportedFailure,
        reported_usage,
    )

    failure = ProviderReportedFailure("the provider failed the response")
    failure.usage = _evidence("resp_failed")
    incomplete = IncompleteGenerationError("unfinished")
    incomplete.usage = _evidence("resp_unfinished")
    incomplete.__cause__ = failure

    assert reported_usage(incomplete) is incomplete.usage


def test_a_providers_failure_behind_a_route_aggregate_is_not_read():
    """An aggregate of several routes' errors states its own verdict; the last
    route's failure behind it describes neither the call nor the routes before."""
    from kestrel_sovereign.llm.output_ceiling import ProviderReportedFailure, reported_usage

    failure = ProviderReportedFailure("the last route's provider failed")
    failure.usage = _evidence("resp_last_route")
    aggregate = RuntimeError("every route failed")
    aggregate.declined_wait = None
    aggregate.__cause__ = failure

    assert reported_usage(aggregate) is None
