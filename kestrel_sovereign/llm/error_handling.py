"""
Error handling decorators and utilities for LLM operations.

This module provides standardized error handling patterns to replace
broad exception catches throughout the LLM service code.
"""
import logging
import time
import asyncio
from typing import Any, Callable, Optional, Type, Union, Dict
from functools import wraps

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class for LLM-related errors."""
    pass


class LLMProviderError(LLMError):
    """Error from a specific LLM provider."""
    def __init__(self, provider: str, message: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"Provider {provider}: {message}")


class LLMProviderTimeoutError(LLMProviderError):
    """Timeout error from an LLM provider."""
    pass


class LLMProviderAuthError(LLMProviderError):
    """Authentication error from an LLM provider."""
    pass


class LLMProviderQuotaError(LLMProviderError):
    """Quota/rate limit error from an LLM provider."""
    pass


class LLMAllProvidersFailedError(LLMError):
    """All configured providers have failed."""
    def __init__(self, errors: Dict[str, Exception]):
        self.errors = errors
        error_summary = "; ".join([f"{provider}: {str(error)}" for provider, error in errors.items()])
        super().__init__(f"All providers failed: {error_summary}")


def handle_llm_errors(
    provider_name: Optional[str] = None,
    log_errors: bool = True,
    reraise_as: Optional[Type[Exception]] = None
):
    """Decorator to handle LLM-specific errors with proper logging and classification.

    Args:
        provider_name: Name of the provider (if known at decoration time)
        log_errors: Whether to log errors (default: True)
        reraise_as: Exception type to reraise as (default: keep original structure)

    Example:
        @handle_llm_errors(provider_name="openai")
        async def call_openai_api(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            provider = provider_name or kwargs.get('provider_name', 'unknown')
            try:
                return await func(*args, **kwargs)
            except asyncio.TimeoutError as e:
                error = LLMProviderTimeoutError(provider, "Request timeout", e)
                if log_errors:
                    logger.error(f"Timeout in {func.__name__} for provider {provider}: {e}")
                if reraise_as:
                    raise reraise_as(str(error)) from e
                raise error
            except Exception as e:
                # Classify common provider errors
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['unauthorized', 'invalid key', 'authentication']):
                    error = LLMProviderAuthError(provider, "Authentication failed", e)
                elif any(keyword in error_msg for keyword in ['quota', 'rate limit', 'too many requests']):
                    error = LLMProviderQuotaError(provider, "Quota exceeded", e)
                elif 'timeout' in error_msg:
                    error = LLMProviderTimeoutError(provider, "Provider timeout", e)
                else:
                    error = LLMProviderError(provider, str(e), e)

                if log_errors:
                    logger.error(f"Error in {func.__name__} for provider {provider}: {e}")

                if reraise_as:
                    raise reraise_as(str(error)) from e
                raise error

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            provider = provider_name or kwargs.get('provider_name', 'unknown')
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Same error classification as async version
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['unauthorized', 'invalid key', 'authentication']):
                    error = LLMProviderAuthError(provider, "Authentication failed", e)
                elif any(keyword in error_msg for keyword in ['quota', 'rate limit', 'too many requests']):
                    error = LLMProviderQuotaError(provider, "Quota exceeded", e)
                elif 'timeout' in error_msg:
                    error = LLMProviderTimeoutError(provider, "Provider timeout", e)
                else:
                    error = LLMProviderError(provider, str(e), e)

                if log_errors:
                    logger.error(f"Error in {func.__name__} for provider {provider}: {e}")

                if reraise_as:
                    raise reraise_as(str(error)) from e
                raise error

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def handle_observability_errors(func: Callable) -> Callable:
    """Decorator to handle observability-related errors without breaking LLM calls.

    Observability failures should never prevent LLM operations from completing.

    Example:
        @handle_observability_errors
        async def log_llm_call(...):
            ...
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Observability operation failed in {func.__name__}: {e}")
            return None  # Return None to indicate failure without raising

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Observability operation failed in {func.__name__}: {e}")
            return None

    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def handle_storage_errors(operation_name: str = "storage operation"):
    """Decorator to handle storage-related errors with proper logging.

    Args:
        operation_name: Description of the storage operation for logging

    Example:
        @handle_storage_errors("usage tracking")
        async def track_usage(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Storage error in {operation_name} ({func.__name__}): {e}")
                # For storage errors, we usually want to continue operation
                # but log the failure for investigation
                return None

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Storage error in {operation_name} ({func.__name__}): {e}")
                return None

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def handle_crypto_errors(func: Callable) -> Callable:
    """Decorator to handle cryptographic errors with proper security logging.

    Crypto errors should be logged with high priority as they may indicate
    security issues or misconfigurations.

    Example:
        @handle_crypto_errors
        def decrypt_api_key(...):
            ...
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Cryptographic operation failed in {func.__name__}: {e}")
            # Reraise crypto errors as they're usually critical
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Cryptographic operation failed in {func.__name__}: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry operations with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Backoff multiplier for delay

    Example:
        @with_retry(max_attempts=3, delay=1.0)
        async def unreliable_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                        logger.debug(f"Attempt {attempt + 1} failed in {func.__name__}: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed in {func.__name__}: {e}")

            raise last_error

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        logger.debug(f"Attempt {attempt + 1} failed in {func.__name__}: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed in {func.__name__}: {e}")

            raise last_error

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def handle_provider_fallback(providers: list):
    """Decorator to handle provider fallback logic.

    This decorator automatically tries multiple providers in order and collects
    all errors for comprehensive error reporting.

    Args:
        providers: List of provider info objects to try in order

    Example:
        @handle_provider_fallback(available_providers)
        async def try_llm_request(provider_info, ...):
            # This function will be called for each provider until one succeeds
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            errors = {}

            for provider_info in providers:
                try:
                    # Call the function with the current provider
                    return await func(provider_info, *args, **kwargs)
                except Exception as e:
                    provider_name = getattr(provider_info, 'name', str(provider_info))
                    errors[provider_name] = e
                    logger.warning(f"Provider {provider_name} failed: {e}")
                    continue

            # All providers failed
            raise LLMAllProvidersFailedError(errors)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            errors = {}

            for provider_info in providers:
                try:
                    return func(provider_info, *args, **kwargs)
                except Exception as e:
                    provider_name = getattr(provider_info, 'name', str(provider_info))
                    errors[provider_name] = e
                    logger.warning(f"Provider {provider_name} failed: {e}")
                    continue

            raise LLMAllProvidersFailedError(errors)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator