"""Contract tests for the retry policy.

The retry layer must:
  * treat permanent failures (401/403/404/422/400, invalid key, quota, model
    not found) as non-retryable — retrying a dead key burns wall-time and
    triggers abuse detection,
  * EXCEPT a subscription plan-limit window, which Anthropic's plan endpoint
    reports as a 400 with billing wording rather than a 429 — that is the same
    transient condition as the 429 form and must be ridden out, not failed,
  * retry transient failures (429, 5xx, timeout, "rate", "try again"),
  * default to non-retryable for unknown errors (explicit is safer than the
    old "if any pattern matches, retry" which let oddly-worded messages
    through).
"""

from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.retry import (
    is_plan_limit_error,
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
async def test_with_retry_propagates_non_retryable_exception_unchanged():
    error = RuntimeError("permanent failure")
    calls = 0

    async def fail():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(RuntimeError) as raised:
        await with_retry(fail)

    assert raised.value is error
    assert calls == 1


@pytest.mark.asyncio
async def test_with_retry_caps_advised_delay_at_max():
    error = _FakeRateLimit("429", status_code=429, headers={"retry-after": "9999"})

    async def throttled():
        raise error

    slept = []

    async def fake_sleep(d):
        slept.append(d)

    with patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep):
        with pytest.raises(_FakeRateLimit) as raised:
            # A throttle is bounded by throttle_max_retries, not max_retries.
            await with_retry(throttled, throttle_max_retries=2)

    assert raised.value is error
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


# ---------------------------------------------------------------------------
# Plan-limit window wearing a 400 (the shape #2074 did not cover)
# ---------------------------------------------------------------------------

# Verbatim from logs/host.log, 2026-09-04 16:31 UTC, Emma on anthropic:plan.
PLAN_LIMIT_400 = (
    "Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': \"You're out of extra usage. Add more "
    "at claude.ai/settings/usage and keep going.\"}, "
    "'request_id': 'req_011Ceiioyx9256gSqt4DStZL'}"
)


def test_plan_limit_400_is_recognized():
    assert is_plan_limit_error(_FakeRateLimit(PLAN_LIMIT_400, status_code=400))


def test_plan_limit_400_is_retryable_despite_the_status_code():
    """The regression this fixes: the structured-400 short-circuit classified
    a transient plan window as a permanent caller error, so it got ZERO
    retries and surfaced as a hard route failure."""
    err = _FakeRateLimit(PLAN_LIMIT_400, status_code=400)
    assert is_retryable_error(err) is True


# Near-miss negatives. These share WORDING with the positive case, not just a
# status code, so they can kill a pattern broadened toward a realistic
# near-miss ("usage", "extra usage") — which fixtures sharing only the status
# code cannot do.
USAGE_WORDED_400 = (
    "Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'usage of the extra field is "
    "invalid for this request'}}"
)
PLAN_LIMIT_WORDING_ON_401 = (
    "Error code: 401 - {'type': 'error', 'error': {'type': "
    "'authentication_error', 'message': \"Invalid API key. You're out of "
    "extra usage.\"}}"
)
PLAN_LIMIT_400_SHOUTED = (
    "ERROR CODE: 400 - YOU'RE OUT OF EXTRA USAGE. ADD MORE AT "
    "CLAUDE.AI/SETTINGS/USAGE AND KEEP GOING."
)


def test_a_400_merely_mentioning_usage_is_not_a_plan_limit():
    """Kills a pattern broadened to 'usage'."""
    assert is_plan_limit_error(
        _FakeRateLimit(USAGE_WORDED_400, status_code=400)
    ) is False


def test_extra_usage_wording_alone_is_not_enough():
    """Kills a pattern broadened to 'extra usage': the phrase has to be the
    plan-limit sentence, not any sentence containing those two words."""
    err = _FakeRateLimit(
        "Error code: 400 - {'message': 'extra usage fields are not allowed'}",
        status_code=400,
    )
    assert is_plan_limit_error(err) is False


def test_plan_limit_wording_on_a_401_is_not_retryable():
    """Scoping guard: a dead key whose message happens to carry the phrase
    must stay permanent. `with_retry` is shared by every adapter, so an
    unscoped match would hand one a retry budget."""
    err = _FakeRateLimit(PLAN_LIMIT_WORDING_ON_401, status_code=401)
    assert is_plan_limit_error(err) is False
    assert is_retryable_error(err) is False


def test_structured_status_wins_over_400_in_the_message_text():
    """The structured `status_code` is authoritative, mirroring
    `test_structured_401_non_retryable_even_with_retry_wording`.

    A provider that reports 401 while its rendered body still carries "Error
    code: 400" and the plan-limit sentence must stay permanent. Without this,
    dropping the structured-status branch is undetectable: every other
    negative fixture happens to omit "400", so the text-based fallback
    returns the right answer for the wrong reason.
    """
    err = _FakeRateLimit(
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': \"You're out of extra usage.\"}}",
        status_code=401,
    )
    assert is_plan_limit_error(err) is False
    assert is_retryable_error(err) is False


def test_plan_limit_match_is_case_insensitive():
    """Exercises the `.lower()` fold, which the lowercase fixture alone never
    reached."""
    assert is_plan_limit_error(
        _FakeRateLimit(PLAN_LIMIT_400_SHOUTED, status_code=400)
    ) is True


def test_plan_limit_is_not_treated_as_a_429_throttle():
    """The patient 8-attempt/131s throttle budget would overrun the 180s
    orchestrator turn watchdog. Plan-limit gets its own tighter budget."""
    from kestrel_sovereign.llm.retry import _is_throttle_error
    assert _is_throttle_error(_FakeRateLimit(PLAN_LIMIT_400, status_code=400)) is False


def test_plan_limit_retry_budget_fits_inside_the_turn_watchdog():
    """Behavioural guard on the constants themselves: worst-case sleep must
    stay a small fraction of the watchdog, or a window that clears late is
    killed as `timeout after 180s` instead of returning its answer."""
    from kestrel_sovereign.llm.retry import (
        PLAN_LIMIT_MAX_RETRIES,
        PLAN_LIMIT_MAX_DELAY,
    )
    from kestrel_sovereign.agent.orchestrator_engine import (
        ORCHESTRATOR_TURN_TIMEOUT_SECS,
    )

    worst_case = sum(
        min(1.0 * (2 ** a) + 1.0, PLAN_LIMIT_MAX_DELAY)
        for a in range(PLAN_LIMIT_MAX_RETRIES - 1)
    )
    assert worst_case < ORCHESTRATOR_TURN_TIMEOUT_SECS * 0.25, (
        f"plan-limit sleep budget {worst_case}s is too much of the "
        f"{ORCHESTRATOR_TURN_TIMEOUT_SECS}s turn watchdog"
    )


@pytest.mark.asyncio
async def test_plan_limit_uses_its_own_attempt_budget():
    """Pins the budget by COUNTING attempts, not by asserting a predicate.
    A white-box assertion on `_is_throttle_error` stayed green when the
    budget selection was broken."""
    from kestrel_sovereign.llm.retry import PLAN_LIMIT_MAX_RETRIES
    calls = {"n": 0}

    async def always_plan_limited():
        calls["n"] += 1
        raise _FakeRateLimit(PLAN_LIMIT_400, status_code=400)

    async def fake_sleep(d):
        pass

    with patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep):
        with pytest.raises(_FakeRateLimit):
            await with_retry(always_plan_limited)

    assert calls["n"] == PLAN_LIMIT_MAX_RETRIES
    # And specifically NOT the patient throttle budget.
    from kestrel_sovereign.llm.retry import THROTTLE_MAX_RETRIES
    assert calls["n"] != THROTTLE_MAX_RETRIES


def test_ordinary_400_stays_non_retryable():
    """Guard on the narrowness of the fix: a real malformed request must not
    become retryable just because it shares the status code."""
    err = _FakeRateLimit(
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'messages.0: unexpected role'}}",
        status_code=400,
    )
    assert is_plan_limit_error(err) is False
    assert is_retryable_error(err) is False


def test_prompt_too_long_400_stays_non_retryable():
    """Emma's other real 400 (context over the model ceiling) is genuinely
    permanent — retrying re-sends the same oversized prompt."""
    err = _FakeRateLimit(
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'prompt is too long: "
        "1104046 tokens > 1000000 maximum'}}",
        status_code=400,
    )
    assert is_retryable_error(err) is False


@pytest.mark.asyncio
async def test_with_retry_rides_out_a_plan_limit_400_then_succeeds():
    """End to end: the window closes and the call goes through, so the
    operator never sees it — exactly how the 429 form already behaves."""
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FakeRateLimit(PLAN_LIMIT_400, status_code=400)
        return "recovered"

    async def fake_sleep(d):
        pass

    with patch("kestrel_sovereign.llm.retry.asyncio.sleep", fake_sleep):
        assert await with_retry(flaky) == "recovered"
    assert calls["n"] == 3
