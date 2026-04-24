"""Session-level route disable on permanent auth failure (#655).

Problem: OpenRouter (or any route) with an invalid/dead API key returns
401 on every user message. The per-call retry layer already skips 401
(see retry.NON_RETRYABLE_PATTERNS), but each NEW user message restarts
the fallback chain from the top, re-attempting the dead route and
logging a red error ~1s before falling through. These tests cover the
fix: the first permanent auth failure disables the route for the
lifetime of the LLMService instance.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from kestrel_sovereign.llm.service import LLMService


class _FakeAuthError(Exception):
    """Mimics an OpenAI/httpx auth error where both a status code and a
    descriptive message are present."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class TestIsPermanentAuthError:
    """Predicate used to decide whether a failed call should disable its
    route for the session."""

    def test_401_status_is_permanent(self):
        exc = _FakeAuthError("some unhelpful body", status_code=401)
        assert LLMService._is_permanent_auth_error(exc) is True

    def test_403_status_is_permanent(self):
        exc = _FakeAuthError("forbidden", status_code=403)
        assert LLMService._is_permanent_auth_error(exc) is True

    def test_user_not_found_message_is_permanent(self):
        """OpenRouter's anti-abuse error on dead keys looks like this even
        without an HTTP status attached on some SDK versions."""
        exc = Exception(
            "Error code: 401 - {'error': {'message': 'User not found.'}}"
        )
        assert LLMService._is_permanent_auth_error(exc) is True

    def test_invalid_api_key_message_is_permanent(self):
        exc = Exception("invalid api key provided")
        assert LLMService._is_permanent_auth_error(exc) is True

    def test_unauthorized_message_is_permanent(self):
        exc = Exception("Unauthorized: token rejected")
        assert LLMService._is_permanent_auth_error(exc) is True

    def test_500_is_not_permanent(self):
        """Server errors are transient — don't disable on them."""
        exc = _FakeAuthError("Internal Server Error", status_code=500)
        assert LLMService._is_permanent_auth_error(exc) is False

    def test_400_bad_request_is_not_permanent(self):
        """400 = caller sent junk. The ROUTE is fine; don't disable."""
        exc = _FakeAuthError("context_length_exceeded", status_code=400)
        assert LLMService._is_permanent_auth_error(exc) is False

    def test_404_model_not_found_is_not_permanent(self):
        """Model 404 ≠ route 401. Don't disable the whole route — the
        user might switch to a different model on that same vendor."""
        exc = _FakeAuthError("model does not exist", status_code=404)
        assert LLMService._is_permanent_auth_error(exc) is False

    def test_rate_limit_is_not_permanent(self):
        exc = _FakeAuthError("Rate limit exceeded", status_code=429)
        assert LLMService._is_permanent_auth_error(exc) is False


class TestMaybeDisableRoute:
    """Records a route as disabled on a permanent auth error. No-op for
    transient or malformed-request errors."""

    def _fresh_service(self) -> LLMService:
        svc = LLMService.__new__(LLMService)
        svc._disabled_routes = {}
        return svc

    def test_permanent_auth_error_adds_route_to_disabled(self):
        svc = self._fresh_service()
        provider = {"name": "openrouter:api"}
        exc = Exception("Error code: 401 - User not found.")
        svc._maybe_disable_route(provider, exc)
        assert "openrouter:api" in svc._disabled_routes

    def test_transient_error_does_not_disable(self):
        svc = self._fresh_service()
        provider = {"name": "openrouter:api"}
        exc = _FakeAuthError("Service unavailable", status_code=503)
        svc._maybe_disable_route(provider, exc)
        assert "openrouter:api" not in svc._disabled_routes

    def test_disable_is_idempotent_on_repeat_failure(self):
        """If the caller hits the same dead route twice (racing fallbacks
        in streaming+non-streaming paths, say), the second call is a
        no-op — no duplicate log lines, no overwrite of the first reason."""
        svc = self._fresh_service()
        provider = {"name": "openrouter:api"}
        first_exc = Exception("401 - User not found")
        second_exc = Exception("401 - something else slightly different")
        svc._maybe_disable_route(provider, first_exc)
        svc._maybe_disable_route(provider, second_exc)
        assert len(svc._disabled_routes) == 1
        # first reason wins (preserves the original log line)
        assert "User not found" in svc._disabled_routes["openrouter:api"]

    def test_provider_without_name_is_safely_ignored(self):
        svc = self._fresh_service()
        svc._maybe_disable_route({}, Exception("401 unauthorized"))
        assert svc._disabled_routes == {}


class TestAvailableProviders:
    """`_available_providers()` filters out routes recorded as disabled."""

    def _fresh_service(self, providers):
        svc = LLMService.__new__(LLMService)
        svc._disabled_routes = {}
        svc.providers = providers
        return svc

    def test_no_disabled_returns_all_providers(self):
        svc = self._fresh_service([
            {"name": "anthropic:api"},
            {"name": "openai:api"},
        ])
        assert [p["name"] for p in svc._available_providers()] == [
            "anthropic:api", "openai:api",
        ]

    def test_skips_disabled_routes(self):
        svc = self._fresh_service([
            {"name": "openrouter:api"},
            {"name": "anthropic:api"},
            {"name": "openai:api"},
        ])
        svc._disabled_routes["openrouter:api"] = "401 User not found"
        assert [p["name"] for p in svc._available_providers()] == [
            "anthropic:api", "openai:api",
        ]

    def test_all_disabled_returns_empty_list(self):
        """Pathological but well-defined: every route disabled → empty
        list. Caller raises a clear 'all providers failed' rather than
        pretending a dead provider is usable."""
        svc = self._fresh_service([{"name": "openrouter:api"}])
        svc._disabled_routes["openrouter:api"] = "401"
        assert svc._available_providers() == []

    def test_explicit_providers_arg_respects_disabled_set(self):
        """Callers that have their own narrowed list (e.g. after a
        mandate filter) can still use the helper to subtract disabled
        routes."""
        svc = self._fresh_service([])  # self.providers irrelevant here
        svc._disabled_routes["openrouter:api"] = "401"
        narrowed = [{"name": "openrouter:api"}, {"name": "anthropic:api"}]
        assert [p["name"] for p in svc._available_providers(narrowed)] == [
            "anthropic:api",
        ]
