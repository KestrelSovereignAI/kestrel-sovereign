"""
Retry Utilities for LLM Adapters

Provides exponential backoff with jitter for handling transient errors
like rate limiting (429) and server errors (5xx).
"""
import asyncio
import logging
import random
from typing import Any, Callable, TypeVar

from kestrel_sovereign.kestrel_config.constants import LLM_RETRY_MAX_DELAY_SECONDS

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
MAX_DELAY = LLM_RETRY_MAX_DELAY_SECONDS

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
    """
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

            # Calculate delay with exponential backoff + jitter
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            logger.warning(
                f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}"
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
