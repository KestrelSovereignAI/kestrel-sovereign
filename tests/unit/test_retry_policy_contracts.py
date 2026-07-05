"""Contract tests for the retry policy.

The retry layer must:
  * treat permanent failures (401/403/404/422/400, invalid key, quota, model
    not found) as non-retryable — retrying a dead key burns wall-time and
    triggers abuse detection,
  * retry transient failures (429, 5xx, timeout, "rate", "try again"),
  * default to non-retryable for unknown errors (explicit is safer than the
    old "if any pattern matches, retry" which let oddly-worded messages
    through).
"""

from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.retry import (
    is_retryable_error,
    retry_after_seconds,
    with_retry,
)


class _FakeResponse:
    def __init__(self, headers):
        self.headers = headers


class _FakeRateLimit(Exception):
    """Shaped like an SDK ``RateLimitError``: carries a structured
    ``status_code`` and a response with headers, regardless of message text."""

    def __init__(self, message="rate limited", *, status_code=429, headers=None):
        super().__init__(message)
        self.status_code = status_code
        if headers is not None:
            self.response = _FakeResponse(headers)


NON_RETRYABLE_CASES = [
    # OpenRouter dead-key / anti-abuse — the specific error that caused the
    # 401 storm in reflection cycles.
    ("Error code: 401 - {'error': {'message': 'User not found.', 'code': 401}}"),
    # Standard auth failures.
    ("Error code: 401 - Unauthorized"),
    ("Error code: 403 - Forbidden"),
    ("AuthenticationError: Invalid API key"),
    # Quota exhausted — permanent until refill; not rate-limit transience.
    ("Quota exceeded for this key"),
    ("insufficient_quota"),
    # Caller errors.
    ("Error code: 400 - Bad Request: invalid parameter"),
    ("Error code: 404 - model not found"),
    ("Error code: 422 - Unprocessable Entity"),
    # Explicit permission denied.
    ("permission_denied"),
    # Model-level permanent.
    ("model_not_found: 'gpt-5-mini' does not exist"),
]


RETRYABLE_CASES = [
    ("Error code: 429 - Rate limit exceeded. Try again later."),
    ("Error code: 500 - Internal server error"),
    ("Error code: 502 - Bad Gateway"),
    ("Error code: 503 - Service Unavailable"),
    ("Error code: 504 - Gateway Timeout"),
    ("Connection timeout occurred"),
    ("Server overloaded, try again"),
    ("capacity exceeded momentarily"),
]


UNKNOWN_CASES_DEFAULT_NO_RETRY = [
    # Unknown error shape — explicit non-retryable default prevents the old
    # "accidentally retryable because message contains 'internal'" trap.
    ("UnknownWeirdError: something went sideways"),
    ("A generic error with no keywords"),
]


@pytest.mark.parametrize("msg", NON_RETRYABLE_CASES)
def test_non_retryable_classifications(msg):
    assert is_retryable_error(Exception(msg)) is False, (
        f"{msg!r} must NOT be retryable (would burn wall-time on a permanent failure)"
    )


@pytest.mark.parametrize("msg", RETRYABLE_CASES)
def test_retryable_classifications(msg):
    assert is_retryable_error(Exception(msg)) is True, (
        f"{msg!r} must be retryable (transient)"
    )


@pytest.mark.parametrize("msg", UNKNOWN_CASES_DEFAULT_NO_RETRY)
def test_unknown_defaults_non_retryable(msg):
    assert is_retryable_error(Exception(msg)) is False, (
        f"{msg!r} is unknown — default must be no-retry"
    )


def test_non_retryable_wins_over_retryable_pattern():
    """A 401 message that happens to contain the word 'internal' must still
    be classified non-retryable — the old policy would have retried it
    because 'internal' is a retryable pattern."""
    err = Exception("Error code: 401 - Unauthorized (internal detail: ...)")
    assert is_retryable_error(err) is False


def test_status_code_word_boundary():
    """A 64-character hex string happening to contain '401' as a substring
    must not be treated as a 401 status code. Word-boundary match only."""
    err = Exception("hash=abc401def retry_token=xyz")
    # No structural 401-status indicator (no "401 ", " 401", "code: 401", etc.)
    # so this falls through to unknown → no retry by default.
    assert is_retryable_error(err) is False


# --- Structured-status classification (classify on the ACTUAL error) ---------


def test_structured_429_retryable_even_without_429_in_text():
    """A real RateLimitError may not contain '429' in its message; we must
    classify on the structured status code, not a hopeful substring."""
    err = _FakeRateLimit("You're sending requests too quickly", status_code=429)
    assert is_retryable_error(err) is True


def test_structured_401_non_retryable_even_with_retry_wording():
    err = _FakeRateLimit("rate stuff but actually auth", status_code=401)
    assert is_retryable_error(err) is False


# --- Retry-After extraction --------------------------------------------------


def test_retry_after_header_seconds():
    assert retry_after_seconds(_FakeRateLimit(headers={"retry-after": "12"})) == 12.0


def test_retry_after_header_millis():
    assert retry_after_seconds(_FakeRateLimit(headers={"retry-after-ms": "2500"})) == 2.5


def test_retry_after_absent_returns_none():
    assert retry_after_seconds(_FakeRateLimit(headers={})) is None
    assert retry_after_seconds(Exception("plain error, no response")) is None


# --- with_retry actually retries a throttle (instead of propagating) ---------


@pytest.mark.asyncio
async def test_with_retry_rides_out_throttle_then_succeeds():
    """The core fix: a 429 is retried (honoring Retry-After) and ultimately
    succeeds — it does NOT propagate to the caller's fallback chain on the
    first throttle."""
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FakeRateLimit("429 slow down", status_code=429, headers={"retry-after": "5"})
        return "ok"

    slept = []

    async def fake_sleep(d):
        slept.append(d)

    with patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep):
        result = await with_retry(flaky)

    assert result == "ok"
    assert calls["n"] == 3
    # Honored the server's 5s cool-down (+ <1s jitter), not blind 1s/2s backoff.
    assert all(5.0 <= d < 6.0 for d in slept), slept


@pytest.mark.asyncio
async def test_with_retry_caps_advised_delay_at_max():
    async def throttled():
        raise _FakeRateLimit("429", status_code=429, headers={"retry-after": "9999"})

    slept = []

    async def fake_sleep(d):
        slept.append(d)

    with patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep):
        with pytest.raises(_FakeRateLimit):
            # A throttle is bounded by throttle_max_retries, not max_retries.
            await with_retry(throttled, throttle_max_retries=2)

    assert slept and all(d <= 120.0 for d in slept)


class _FakeServerError(Exception):
    """A transient 5xx (e.g. ollama 503 while a model loads) — retryable but NOT
    a throttle, so it must use the tight budget, not the patient plan one."""

    def __init__(self, message="503 service unavailable", *, status_code=503):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.asyncio
async def test_non_throttle_uses_tight_budget_for_fast_failover():
    """#2074 regression: a transient 5xx must NOT get the patient 8x120 plan
    budget — it uses the tight default (5 attempts, 60s cap) so the fallback
    chain advances to a healthy route in ~seconds, not ~2 minutes."""
    attempts = {"n": 0}

    async def always_503():
        attempts["n"] += 1
        raise _FakeServerError()

    slept = []

    async def fake_sleep(d):
        slept.append(d)

    with patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep):
        with pytest.raises(_FakeServerError):
            await with_retry(always_503)

    # Tight default: 5 total attempts (4 sleeps), each capped at 60s.
    assert attempts["n"] == 5, attempts
    assert len(slept) == 4, slept
    assert all(d <= 60.0 for d in slept), slept


@pytest.mark.asyncio
async def test_throttle_uses_patient_budget():
    """A 429 gets the patient plan-route budget (8 attempts) so a rate-limit
    window is ridden out instead of downgrading to a metered route."""
    attempts = {"n": 0}

    async def always_429():
        attempts["n"] += 1
        raise _FakeRateLimit("429", status_code=429)

    slept = []

    async def fake_sleep(d):
        slept.append(d)

    with patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep):
        with pytest.raises(_FakeRateLimit):
            await with_retry(always_429)

    assert attempts["n"] == 8, attempts  # patient budget
    assert all(d <= 120.0 for d in slept), slept
