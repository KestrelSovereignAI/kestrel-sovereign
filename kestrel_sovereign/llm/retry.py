"""
Retry Utilities for LLM Adapters

Provides exponential backoff with jitter for handling transient errors
like rate limiting (429) and server errors (5xx).
"""
import asyncio
import logging
import random
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Retry configuration.
#
# The DEFAULT budget is tight (5 attempts, 60s cap): a transient infra error
# (503 while an ollama model loads, a 5xx, a timeout) on ANY route should fail
# over to the next route quickly — waiting minutes buys nothing when a healthy
# fallback exists. with_retry is the default for EVERY adapter call site
# (ollama/openai-compat/vertex/anthropic), so a patient default would stall
# failover on all of them (that was the #2074 regression).
MAX_RETRIES = 5
BASE_DELAY = 1.0      # seconds
MAX_DELAY = 60.0      # seconds

# THROTTLE budget — applied only to a 429 / rate-limit throttle. A 429 on a
# subscription plan route (Claude Max / ChatGPT plan) is "you're going too fast
# — wait", NOT "this route is dead". Riding out a normal rate-limit window on
# the plan route (patient: 8 attempts, 2-minute cap) avoids silently downgrading
# to a metered :api route. Honoring the server's ``Retry-After`` keeps the waits
# accurate. Scoped to throttles by error type so it does NOT slow non-throttle
# failover on local/metered routes.
THROTTLE_MAX_RETRIES = 8
THROTTLE_MAX_DELAY = 120.0

# HTTP status codes that warrant retry. 401/403/404 are permanent and MUST
# NOT appear here — retrying a dead API key burns wall-time on an error that
# will never recover.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# HTTP status codes that MUST NOT retry (caller error or auth failure).
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}

# Error message patterns that indicate retryable (transient) errors.
RETRYABLE_PATTERNS = [
    "rate",
    "timeout",
    "resource_exhausted",
    "unavailable",
    "internal",
    "overloaded",
    "capacity",
    "try again",
]

# Error message patterns that indicate permanent failure. Matched FIRST —
# takes precedence over RETRYABLE_PATTERNS to prevent a 401 message that
# happens to contain "internal server error text" from being retried.
NON_RETRYABLE_PATTERNS = [
    "user not found",          # OpenRouter dead-key / anti-abuse
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "unauthorized",
    "permission_denied",
    "permission denied",
    "quota exceeded",           # Distinct from rate-limit: permanent until refill
    "model_not_found",
    "model not found",
    "insufficient_quota",
]

T = TypeVar('T')


def retry_after_seconds(error: Exception) -> Optional[float]:
    """Extract the server-advised cool-down from a provider exception.

    Honors the actual ``Retry-After`` (or ``retry-after-ms``) response header
    the SDK exception carries, so we wait the amount the *server* asked for
    instead of guessing with exponential backoff. Returns ``None`` when no
    usable value is present (caller falls back to exponential backoff).
    """
    # SDK exceptions (anthropic/openai) carry the httpx response.
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            getter = headers.get
        except AttributeError:
            getter = None
        if getter is not None:
            ms = getter("retry-after-ms")
            if ms:
                try:
                    return max(0.0, float(ms) / 1000.0)
                except (TypeError, ValueError):
                    pass
            secs = getter("retry-after")
            if secs:
                try:
                    # Numeric seconds. (An HTTP-date form is rare here and not
                    # worth parsing — exponential backoff covers that case.)
                    return max(0.0, float(secs))
                except (TypeError, ValueError):
                    pass
    # Some SDKs surface a parsed attribute directly.
    attr = getattr(error, "retry_after", None)
    if isinstance(attr, (int, float)):
        return max(0.0, float(attr))
    return None


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is transient and should be retried.

    Order of checks:
      1. NON_RETRYABLE_STATUS_CODES (401/403/404/422/400) → no retry.
      2. NON_RETRYABLE_PATTERNS (auth/quota/invalid) → no retry.
      3. RETRYABLE_STATUS_CODES (429/5xx) → retry.
      4. RETRYABLE_PATTERNS (rate/timeout/etc) → retry.
      5. Default → no retry.

    Permanent-failure checks win first so a 401 error message that happens
    to contain the word "internal" in a nested detail doesn't trigger a
    misguided retry loop.

    The provider's *structured* status code (``error.status_code`` on the
    Anthropic/OpenAI SDK exceptions) is consulted before any string matching,
    so a real ``RateLimitError`` is recognized as a 429 even when its message
    text doesn't literally contain the number — we classify on the actual
    error, not a hopeful substring.
    """
    # 0. Structured status code from the SDK exception (most authoritative).
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return False
        if status_code in RETRYABLE_STATUS_CODES:
            return True

    error_str = str(error).lower()

    # 1. Hard non-retryable status codes — word-boundary check to avoid
    #    matching "429" inside a 64-character token or hash.
    for code in NON_RETRYABLE_STATUS_CODES:
        token = str(code)
        # Match "401 - ..." or "code: 401" or "401,"/"401 " etc.
        if f" {token}" in error_str or f"{token} " in error_str or f"{token}:" in error_str or f"{token}," in error_str or f"{token}-" in error_str or f"code: {token}" in error_str or error_str.startswith(f"{token}"):
            return False

    # 2. Explicit non-retryable message patterns.
    for pattern in NON_RETRYABLE_PATTERNS:
        if pattern in error_str:
            return False

    # 3. Retryable status codes.
    for code in RETRYABLE_STATUS_CODES:
        if str(code) in error_str:
            return True

    # 4. Retryable message patterns.
    for pattern in RETRYABLE_PATTERNS:
        if pattern in error_str:
            return True

    # 5. Default: don't retry unknown errors.
    return False


def _is_throttle_error(error: Exception) -> bool:
    """True when ``error`` is a 429 / rate-limit throttle (a "slow down", not a
    dead route). Only throttles get the patient plan-route budget; every other
    retryable error (5xx, timeout, "unavailable") uses the tight default so
    failover to a healthy route isn't stalled for minutes (#2074 regression)."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429
    error_str = str(error).lower()
    # No structured code: match 429 with the same word-boundary care
    # is_retryable_error uses, plus the explicit rate-limit phrasings.
    if (" 429" in error_str or "429 " in error_str or "429:" in error_str
            or "429," in error_str or error_str.startswith("429")):
        return True
    return "rate limit" in error_str or "rate_limit" in error_str


async def with_retry(
    func: Callable[..., Any],
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    throttle_max_retries: int = THROTTLE_MAX_RETRIES,
    throttle_max_delay: float = THROTTLE_MAX_DELAY,
    **kwargs
) -> T:
    """
    Execute an async function with exponential backoff retry.

    Retries on rate limiting (429) and server errors (5xx).
    Uses exponential backoff with jitter to prevent thundering herd.

    Args:
        func: The async function to execute
        *args: Positional arguments to pass to func
        max_retries: Retry attempts for a NON-throttle transient error (default 5)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Delay cap for a non-throttle error (default: 60.0)
        throttle_max_retries: Retry attempts for a 429/rate-limit throttle
            (default 8 — patient, to ride out a plan-route rate limit)
        throttle_max_delay: Delay cap for a throttle (default 120.0)
        **kwargs: Keyword arguments to pass to func

    The budget is chosen per error: a 429/rate-limit throttle gets the patient
    plan-route budget; every other retryable error (5xx/timeout/unavailable) gets
    the tight default so failover to a healthy route isn't stalled for minutes.

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    attempt = 0

    while True:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if not is_retryable_error(e):
                raise

            # Pick the budget from the error type: patient for a throttle,
            # tight for any other transient error.
            throttled = _is_throttle_error(e)
            eff_max_retries = throttle_max_retries if throttled else max_retries
            eff_max_delay = throttle_max_delay if throttled else max_delay

            if attempt >= eff_max_retries - 1:
                raise

            # Prefer the server-advised cool-down; otherwise exponential
            # backoff + jitter. Both are capped at the effective max delay.
            advised = retry_after_seconds(e)
            if advised is not None:
                delay = min(advised + random.uniform(0, 1), eff_max_delay)
            else:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), eff_max_delay)

            # Log the *actual* error — type, status code, and advised wait —
            # so a throttle that's being ridden out (rather than billed via
            # fallback) is visible, not silent.
            status_code = getattr(e, "status_code", None)
            logger.warning(
                "LLM retry %d/%d after %.1fs (status=%s, throttle=%s, retry_after=%s): %s: %s",
                attempt + 1, eff_max_retries, delay, status_code, throttled,
                advised, type(e).__name__, e,
            )
            await asyncio.sleep(delay)
            attempt += 1

    raise last_exception


async def retry_with_backoff(
    func: Callable[..., Any],
    *args,
    **kwargs
) -> T:
    """
    Alias for with_retry for backwards compatibility.

    This maintains compatibility with the existing Vertex adapter.
    """
    return await with_retry(func, *args, **kwargs)
