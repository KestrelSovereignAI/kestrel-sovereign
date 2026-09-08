"""The advised-wait contract after review round 1 of #3127.

* The budget is a running bound on the SUM of waits, not a price per attempt:
  advice that fit each remaining-attempt price could add up to 3360 s.
* A header the provider got wrong (an epoch timestamp, an exponent, ``inf``)
  cannot raise ``OverflowError`` or name a year decades out: ``inf``/``nan``
  are no advice at all, and a reset time is computed from the advice clamped
  to a one-week horizon and said to be a floor.
* Only explicit links are followed when a surface looks for the decline:
  ``__cause__``, ``original_error``, ``underlying``. A chain severed with
  ``from None`` and an unrelated exception raised while a decline was being
  handled do not render as a rate limit.
* Aggregate errors raised after every route failed carry the earliest
  decline as their cause.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from kestrel_sovereign.api_errors import rate_limited_until
from kestrel_sovereign.llm.error_handling import (
    LLMAllProvidersFailedError,
    LLMProviderQuotaError,
)
from kestrel_sovereign.llm.retry import (
    ADVISED_WAIT_HORIZON_SECONDS,
    MAX_DELAY,
    MAX_RETRIES,
    THROTTLE_MAX_DELAY,
    THROTTLE_MAX_RETRIES,
    AdvisedWaitExceedsRetryBudget,
    advised_wait_exceeding_budget,
    earliest_declined_wait,
    retry_after_seconds,
    with_retry,
)
from kestrel_sovereign.llm.streaming import LLMStreamingError
from kestrel_sovereign.llm.streaming_errors import safe_streaming_error_message

THROTTLE_BUDGET = THROTTLE_MAX_DELAY * (THROTTLE_MAX_RETRIES - 1)  # 840
TIGHT_BUDGET = MAX_DELAY * (MAX_RETRIES - 1)  # 240


class _FakeResponse:
    def __init__(self, headers):
        self.headers = headers


class _FakeRateLimit(Exception):
    def __init__(self, message="429 slow down", *, status_code=429, headers=None):
        super().__init__(message)
        self.status_code = status_code
        if headers is not None:
            self.response = _FakeResponse(headers)


def _throttle(advised) -> _FakeRateLimit:
    return _FakeRateLimit(headers={"retry-after": str(advised)})


async def _drive(errors_by_attempt, **kwargs):
    """Run with_retry over scripted per-attempt errors; return (result, sleeps)."""
    calls = {"n": 0}
    sleeps: list[float] = []

    async def op():
        i = calls["n"]
        calls["n"] += 1
        item = errors_by_attempt(i) if callable(errors_by_attempt) else errors_by_attempt[i]
        if isinstance(item, Exception):
            raise item
        return item

    async def fake_sleep(delay):
        sleeps.append(delay)

    with (
        patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep),
        patch("kestrel_sovereign.llm.retry.random.uniform", return_value=0.0),
    ):
        result = await with_retry(op, **kwargs)
    return result, sleeps


# ---------------------------------------------------------------------------
# P1: the budget bounds the sum of waits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advice_that_fits_each_attempts_price_cannot_sum_past_the_budget():
    """The review's script: advice of exactly each remaining-attempt price slept
    [840, 720, 600, 480, 360, 240, 120] = 3360 s. Now: 840 s, then decline."""
    with pytest.raises(AdvisedWaitExceedsRetryBudget) as info:
        await _drive(lambda i: _throttle(THROTTLE_MAX_DELAY * (THROTTLE_MAX_RETRIES - i - 1)))
    sleeps = info.value  # the error; sleeps captured below via a second drive
    assert sleeps.budget_seconds == 0.0

    slept: list[float] = []

    async def op():
        raise _throttle(THROTTLE_MAX_DELAY * (THROTTLE_MAX_RETRIES - len(slept) - 1))

    async def fake_sleep(d):
        slept.append(d)

    with (
        patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep),
        patch("kestrel_sovereign.llm.retry.random.uniform", return_value=0.0),
        pytest.raises(AdvisedWaitExceedsRetryBudget),
    ):
        await with_retry(op)
    assert slept == [THROTTLE_BUDGET]
    assert sum(slept) <= THROTTLE_BUDGET


@pytest.mark.asyncio
async def test_a_constant_advice_is_declined_once_the_budget_is_spent():
    """600 s advised every time: one 600 s wait fits, the second does not
    (240 s left); the loop declines instead of sleeping 1800 s."""
    slept: list[float] = []

    async def op():
        raise _throttle(600)

    async def fake_sleep(d):
        slept.append(d)

    with (
        patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep),
        patch("kestrel_sovereign.llm.retry.random.uniform", return_value=0.0),
        pytest.raises(AdvisedWaitExceedsRetryBudget) as info,
    ):
        await with_retry(op)
    assert slept == [600.0]
    assert info.value.budget_seconds == THROTTLE_BUDGET - 600


@pytest.mark.asyncio
async def test_a_guessed_delay_after_a_large_advised_wait_is_clamped_to_what_is_left():
    """800 s advised then a throttle with no advice: the backoff guess may not
    exceed the 40 s left, so the total stays within 840 s."""
    result, sleeps = await _drive([_throttle(800), _FakeRateLimit(), "ok"], base_delay=100.0)
    assert result == "ok"
    assert sleeps == [800.0, 40.0]
    assert sum(sleeps) == THROTTLE_BUDGET


@pytest.mark.asyncio
async def test_advice_equal_to_the_remaining_budget_is_taken_whole():
    """The boundary: 840 s fits exactly; 840.5 s does not."""
    result, sleeps = await _drive([_throttle(THROTTLE_BUDGET), "ok"])
    assert result == "ok" and sleeps == [THROTTLE_BUDGET]

    with pytest.raises(AdvisedWaitExceedsRetryBudget):
        await _drive([_throttle(THROTTLE_BUDGET + 0.5), "ok"])


@pytest.mark.asyncio
async def test_the_budget_follows_the_error_type_per_attempt():
    """A tight-budget error after a throttle wait sees the tight budget less
    what was already slept."""

    class _Overloaded(Exception):
        status_code = 503

        def __init__(self, advised=None):
            super().__init__("503 overloaded")
            if advised is not None:
                self.response = _FakeResponse({"retry-after": str(advised)})

    with pytest.raises(AdvisedWaitExceedsRetryBudget) as info:
        await _drive([_throttle(200), _Overloaded(50), "ok"])
    assert info.value.budget_seconds == TIGHT_BUDGET - 200


# ---------------------------------------------------------------------------
# P2: a wrong header cannot overflow or name a far-off year
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["inf", "1e999", "nan", "-inf"])
def test_a_non_finite_retry_after_is_no_advice(value):
    assert retry_after_seconds(_FakeRateLimit(headers={"retry-after": value})) is None
    assert retry_after_seconds(_FakeRateLimit(headers={"retry-after-ms": value})) is None


def test_a_boolean_retry_after_attribute_is_ignored():
    err = _FakeRateLimit()
    err.retry_after = True
    assert retry_after_seconds(err) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["inf", "1e999"])
async def test_a_non_finite_header_falls_back_to_backoff_instead_of_raising(value):
    result, sleeps = await _drive(
        [_FakeRateLimit(headers={"retry-after": value}), "ok"], base_delay=3.0
    )
    assert result == "ok" and sleeps == [3.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["1756241645000", "1756241645", "1e15"])
async def test_a_huge_header_declines_with_a_horizon_floor_instead_of_overflowing(value):
    before = datetime.now(UTC)
    with pytest.raises(AdvisedWaitExceedsRetryBudget) as info:
        await _drive([_FakeRateLimit(headers={"retry-after": value})])
    err = info.value
    assert err.advised_seconds == float(value)
    assert err.beyond_horizon is True
    assert (err.retry_at - before).total_seconds() == pytest.approx(
        ADVISED_WAIT_HORIZON_SECONDS, abs=5
    )
    assert err.retry_after_header_seconds == ADVISED_WAIT_HORIZON_SECONDS
    assert "for more than 7 days, until at least" in str(err)


def test_advice_within_the_horizon_is_an_exact_time():
    err = AdvisedWaitExceedsRetryBudget(
        _throttle(6832), advised_seconds=6832, budget_seconds=840,
        retry_at=datetime(2026, 8, 26, 21, 14, 5, tzinfo=UTC),
    )
    assert err.beyond_horizon is False
    assert err.reset_phrase() == "until 2026-08-26T21:14:05+00:00"
    assert err.retry_after_header_seconds == 6832
    assert rate_limited_until(err).headers == {"Retry-After": "6832"}
    assert "until 2026-08-26T21:14:05+00:00" in rate_limited_until(err).message


def test_the_surfaces_say_at_least_when_the_advice_exceeded_the_horizon():
    floor = datetime(2026, 9, 15, 0, 0, tzinfo=UTC)
    err = AdvisedWaitExceedsRetryBudget(
        _throttle(1756241645), advised_seconds=1756241645, budget_seconds=840, retry_at=floor,
    )
    http = rate_limited_until(err)
    assert http.headers == {"Retry-After": str(ADVISED_WAIT_HORIZON_SECONDS)}
    assert "for more than 7 days, until at least 2026-09-15T00:00:00+00:00" in http.message
    assert "2082" not in http.message
    stream = safe_streaming_error_message(err)
    assert "for more than 7 days, until at least 2026-09-15T00:00:00+00:00" in stream
    assert "2082" not in stream


# ---------------------------------------------------------------------------
# P2: only explicit links reach a decline
# ---------------------------------------------------------------------------


def _declined() -> AdvisedWaitExceedsRetryBudget:
    return AdvisedWaitExceedsRetryBudget(
        _throttle(6832), advised_seconds=6832, budget_seconds=840,
        retry_at=datetime(2026, 8, 26, 21, 0, tzinfo=UTC),
    )


def test_a_chain_severed_with_from_none_does_not_surface_the_decline():
    try:
        try:
            raise _declined()
        except AdvisedWaitExceedsRetryBudget:
            raise RuntimeError("deliberately unchained") from None
    except RuntimeError as severed:
        assert severed.__suppress_context__ is True
        assert advised_wait_exceeding_budget(severed) is None


def test_an_unrelated_error_raised_while_handling_a_decline_is_not_a_rate_limit():
    try:
        try:
            raise _declined()
        except AdvisedWaitExceedsRetryBudget:
            raise ValueError("fallback summarizer: no providers configured")
    except ValueError as unrelated:
        assert unrelated.__context__ is not None  # Python chained it implicitly
        assert advised_wait_exceeding_budget(unrelated) is None


def test_a_failure_in_a_finally_while_a_decline_propagates_is_not_a_rate_limit():
    try:
        try:
            raise _declined()
        finally:
            raise OSError("cleanup failed")
    except OSError as cleanup:
        assert advised_wait_exceeding_budget(cleanup) is None


def test_explicit_links_are_followed():
    declined = _declined()
    quota = LLMProviderQuotaError("anthropic", "Quota exceeded", declined)
    assert advised_wait_exceeding_budget(quota) is declined
    streaming = LLMStreamingError("Selected route failed", provider="p", underlying=declined)
    assert advised_wait_exceeding_budget(streaming) is declined
    try:
        raise RuntimeError("Model x failed") from quota
    except RuntimeError as wrapped:
        assert advised_wait_exceeding_budget(wrapped) is declined


def test_earliest_declined_wait_picks_the_soonest_reset():
    late = _declined()
    soon = AdvisedWaitExceedsRetryBudget(
        _throttle(900), advised_seconds=900, budget_seconds=840,
        retry_at=late.retry_at - timedelta(hours=1),
    )
    errors = [RuntimeError("unrelated"), late, LLMProviderQuotaError("p", "Quota exceeded", soon)]
    assert earliest_declined_wait(errors) is soon
    assert earliest_declined_wait([RuntimeError("a"), ValueError("b")]) is None


def test_the_all_providers_aggregate_carries_the_earliest_decline():
    soon = _declined()
    late = AdvisedWaitExceedsRetryBudget(
        _throttle(90000), advised_seconds=90000, budget_seconds=840,
        retry_at=soon.retry_at + timedelta(days=1),
    )
    errors = {"a": RuntimeError("boom"), "b": late, "c": soon}
    try:
        raise LLMAllProvidersFailedError(errors) from earliest_declined_wait(errors.values())
    except LLMAllProvidersFailedError as aggregate:
        assert advised_wait_exceeding_budget(aggregate) is soon
