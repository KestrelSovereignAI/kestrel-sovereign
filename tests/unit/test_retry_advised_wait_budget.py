"""A server-advised cool-down is honoured, or declined at once (#3127).

``with_retry`` used to clamp ``Retry-After`` to the per-attempt cap, so a 429
advising a 13-hour wait was retried eight times at 120 s each: sixteen minutes
of attempts that could not succeed, the conversation lock held throughout, then
the same failure. On one host 112 of 130 advised waits exceeded the cap.

The contract now:

* advice that fits the budget the loop has left is taken as ONE wait, not
  several capped ones;
* advice that does not fit ends the loop immediately with
  ``AdvisedWaitExceedsRetryBudget``, which carries the reset time and
  classifies as the 429 it stands for, but is not itself retryable;
* a guessed delay (no advice) is still clamped to the per-attempt cap.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.retry import (
    THROTTLE_MAX_DELAY,
    THROTTLE_MAX_RETRIES,
    AdvisedWaitExceedsRetryBudget,
    advised_wait_exceeding_budget,
    is_retryable_error,
    retry_after_seconds,
    with_retry,
)


class _FakeResponse:
    def __init__(self, headers):
        self.headers = headers


class _FakeRateLimit(Exception):
    def __init__(self, message="rate limited", *, status_code=429, headers=None):
        super().__init__(message)
        self.status_code = status_code
        if headers is not None:
            self.response = _FakeResponse(headers)


def _throttle(advised: float) -> _FakeRateLimit:
    return _FakeRateLimit(
        "429 slow down", status_code=429, headers={"retry-after": str(advised)}
    )


async def _run(error_sequence, **kwargs):
    """Drive with_retry over a scripted sequence; sleeps are captured, not slept."""
    calls = {"n": 0}
    sleeps: list[float] = []

    async def op():
        i = calls["n"]
        calls["n"] += 1
        item = error_sequence[i]
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
    return result, sleeps, calls["n"]


# ---------------------------------------------------------------------------
# Advice that fits is one wait
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advice_above_the_cap_but_within_budget_is_one_whole_wait():
    """300 s of advice with 8 x 120 s left: one 300 s sleep, then success —
    not three capped 120 s attempts against advice."""
    result, sleeps, calls = await _run([_throttle(300), "ok"])
    assert result == "ok"
    assert sleeps == [300.0]
    assert calls == 2


@pytest.mark.asyncio
async def test_advice_under_the_cap_is_still_honoured_exactly():
    result, sleeps, _ = await _run([_throttle(5), "ok"])
    assert result == "ok" and sleeps == [5.0]


@pytest.mark.asyncio
async def test_a_guessed_delay_is_still_clamped_to_the_cap():
    """No Retry-After: exponential backoff, capped per attempt as before."""
    no_advice = [_FakeRateLimit("429", status_code=429) for _ in range(7)] + ["ok"]
    result, sleeps, _ = await _run(no_advice, base_delay=100.0)
    assert result == "ok"
    assert sleeps == [100.0] + [THROTTLE_MAX_DELAY] * 6


# ---------------------------------------------------------------------------
# Advice that does not fit ends the loop now
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advice_beyond_the_remaining_budget_raises_immediately_without_sleeping():
    """Emma's instance: advised 6832 s against a 16-minute budget. Zero sleeps,
    one attempt, the reset time on the error."""
    before = datetime.now(UTC)
    with pytest.raises(AdvisedWaitExceedsRetryBudget) as info:
        await _run([_throttle(6832)])
    err = info.value
    assert err.advised_seconds == 6832.0
    assert err.budget_seconds == THROTTLE_MAX_DELAY * (THROTTLE_MAX_RETRIES - 1)
    assert (err.retry_at - before).total_seconds() == pytest.approx(6832, abs=5)
    assert isinstance(err.__cause__, _FakeRateLimit)


@pytest.mark.asyncio
async def test_the_attempt_that_raises_did_not_sleep_first():
    calls = {"n": 0}
    slept: list[float] = []

    async def op():
        calls["n"] += 1
        raise _throttle(46774)

    async def fake_sleep(delay):
        slept.append(delay)

    with (
        patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep),
        pytest.raises(AdvisedWaitExceedsRetryBudget),
    ):
        await with_retry(op)
    assert calls["n"] == 1 and slept == []


@pytest.mark.asyncio
async def test_the_budget_is_what_the_loop_could_still_wait_not_the_whole_budget():
    """After spending attempts, less budget remains: 500 s of advice fits at
    attempt 0 (7 x 120 = 840 left) and does not fit at attempt 6 (120 left)."""
    seq = [_throttle(10)] * 6 + [_throttle(500)]
    with pytest.raises(AdvisedWaitExceedsRetryBudget) as info:
        await _run(seq)
    assert info.value.budget_seconds == THROTTLE_MAX_DELAY * 1

    result, sleeps, _ = await _run([_throttle(500), "ok"])
    assert result == "ok" and sleeps == [500.0]


@pytest.mark.asyncio
async def test_a_non_throttle_transient_with_advice_uses_the_tight_budget():
    """A 503 advising 400 s against 5 x 60 s: does not fit the tight budget, so
    it is declined at once too; failover is the caller's move."""

    class _Overloaded(Exception):
        status_code = 503

        def __init__(self):
            super().__init__("503 overloaded")
            self.response = _FakeResponse({"retry-after": "400"})

    with pytest.raises(AdvisedWaitExceedsRetryBudget) as info:
        await _run([_Overloaded()])
    assert info.value.budget_seconds == 60.0 * 4
    assert info.value.status_code == 429  # it stands for a wait, classified as a throttle


# ---------------------------------------------------------------------------
# The error's shape at every downstream door
# ---------------------------------------------------------------------------


def _declined() -> AdvisedWaitExceedsRetryBudget:
    return AdvisedWaitExceedsRetryBudget(
        _throttle(6832),
        advised_seconds=6832,
        budget_seconds=840,
        retry_at=datetime(2026, 8, 26, 21, 0, tzinfo=UTC),
    )


def test_the_declined_error_classifies_as_the_throttle_it_stands_for():
    err = _declined()
    assert err.status_code == 429
    assert err.retry_after == 6832.0
    assert retry_after_seconds(err) == 6832.0
    assert "429" in str(err) and "rate limit" in str(err)
    assert "2026-08-26T21:00:00+00:00" in str(err)
    assert err.response is err.__cause__.response


def test_the_declined_error_is_not_retryable_by_an_outer_loop():
    assert is_retryable_error(_declined()) is False


@pytest.mark.asyncio
async def test_an_outer_retry_loop_does_not_re_enter_the_declined_wait():
    async def inner():
        raise _throttle(6832)

    async def outer():
        return await with_retry(inner)

    calls = {"n": 0}

    async def counted():
        calls["n"] += 1
        return await outer()

    with (
        patch("kestrel_sovereign.llm.retry.asyncio.sleep") as sleep,
        pytest.raises(AdvisedWaitExceedsRetryBudget),
    ):
        await with_retry(counted)
    assert calls["n"] == 1
    sleep.assert_not_called()


def test_the_reset_time_survives_a_runtime_error_wrapper():
    """The service wraps provider errors as RuntimeError(...) from e; the
    surface reads the decline back out of the chain."""
    declined = _declined()
    try:
        try:
            raise declined
        except AdvisedWaitExceedsRetryBudget as e:
            raise RuntimeError("Model x failed: ...") from e
    except RuntimeError as wrapped:
        assert advised_wait_exceeding_budget(wrapped) is declined
    assert advised_wait_exceeding_budget(RuntimeError("unrelated")) is None


def test_the_chain_walk_terminates_on_a_cycle():
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert advised_wait_exceeding_budget(a) is None
