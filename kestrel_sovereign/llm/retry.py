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
# Tuned for *plan-route* throttling: a 429 on a subscription route (Claude
# Max / ChatGPT plan) is "you're going too fast — wait", NOT "this route is
# dead, fall back to the paid API". We must ride out a normal rate-limit
# window on the plan route rather than silently downgrade to metered billing,
# so the budget is deliberately patient (8 attempts, 2-minute cap). Honoring
# the server's ``Retry-After`` (below) keeps the actual waits accurate.
MAX_RETRIES = 8       # was 5
BASE_DELAY = 1.0      # seconds
MAX_DELAY = 120.0     # seconds — was LLM_RETRY_MAX_DELAY_SECONDS (60.0)

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


async def with_retry(
    func: Callable[..., Any],
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    **kwargs
) -> T:
    """
    Execute an async function with exponential backoff retry.

    Retries on rate limiting (429) and server errors (5xx).
    Uses exponential backoff with jitter to prevent thundering herd.

    Args:
        func: The async function to execute
        *args: Positional arguments to pass to func
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        **kwargs: Keyword arguments to pass to func

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            # Check if this is a retryable error
            if not is_retryable_error(e) or attempt == max_retries - 1:
                raise

            # Prefer the server-advised cool-down; otherwise exponential
            # backoff + jitter. Both are capped at ``max_delay``.
            advised = retry_after_seconds(e)
            if advised is not None:
                delay = min(advised + random.uniform(0, 1), max_delay)
            else:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)

            # Log the *actual* error — type, status code, and advised wait —
            # so a throttle that's being ridden out (rather than billed via
            # fallback) is visible, not silent.
            status_code = getattr(e, "status_code", None)
            logger.warning(
                "LLM retry %d/%d after %.1fs (status=%s, retry_after=%s): %s: %s",
                attempt + 1, max_retries, delay, status_code, advised,
                type(e).__name__, e,
            )
            await asyncio.sleep(delay)

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
