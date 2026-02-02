"""
Retry Utilities for LLM Adapters

Provides exponential backoff with jitter for handling transient errors
like rate limiting (429) and server errors (5xx).
"""
import asyncio
import logging
import random
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds

# HTTP status codes that warrant retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Error message patterns that indicate retryable errors
RETRYABLE_PATTERNS = [
    "429",
    "rate",
    "timeout",
    "resource_exhausted",
    "unavailable",
    "internal",
    "overloaded",
    "capacity",
    "try again",
]

T = TypeVar('T')


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error is transient and should be retried
    """
    error_str = str(error).lower()

    # Check for retryable patterns in error message
    for pattern in RETRYABLE_PATTERNS:
        if pattern in error_str:
            return True

    # Check for status codes in error message
    for code in RETRYABLE_STATUS_CODES:
        if str(code) in error_str:
            return True

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
