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

import pytest

from kestrel_sovereign.llm.retry import is_retryable_error


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
