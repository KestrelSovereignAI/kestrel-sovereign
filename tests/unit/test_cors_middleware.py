"""Tests for CORS origin resolution and the wildcard-credentials guard.

Origin resolution moved to ``kestrel_sovereign.config.build_cors_origins`` so
server.py and host.py share one allowlist + guard (#1679). These tests drive
that real function rather than re-inlining the parsing.
"""
import os
from unittest.mock import patch

import pytest

from kestrel_sovereign.config import build_cors_origins, _DEFAULT_CORS_ORIGINS


class TestCORSDefaults:
    """Defaults must be explicit origins, never a wildcard."""

    def test_default_origins_are_explicit(self):
        assert isinstance(_DEFAULT_CORS_ORIGINS, list)
        assert "*" not in _DEFAULT_CORS_ORIGINS
        assert _DEFAULT_CORS_ORIGINS
        for origin in _DEFAULT_CORS_ORIGINS:
            assert origin.startswith(("http://", "https://")), (
                f"Origin must be a full URL, got: {origin}"
            )

    def test_default_origins_include_localhost(self):
        assert "http://localhost:8080" in _DEFAULT_CORS_ORIGINS
        assert "http://127.0.0.1:8080" in _DEFAULT_CORS_ORIGINS
        assert "http://localhost:3000" in _DEFAULT_CORS_ORIGINS
        assert "http://127.0.0.1:3000" in _DEFAULT_CORS_ORIGINS

    def test_empty_env_falls_back_to_defaults(self):
        with patch.dict(os.environ):
            os.environ.pop("KESTREL_CORS_ORIGINS", None)
            assert build_cors_origins() == _DEFAULT_CORS_ORIGINS


class TestCORSEnvironmentOverride:
    """KESTREL_CORS_ORIGINS overrides the defaults."""

    def test_env_override_parses_comma_separated(self):
        with patch.dict(
            os.environ,
            {"KESTREL_CORS_ORIGINS": "https://app.example.com,https://staging.example.com"},
        ):
            assert build_cors_origins() == [
                "https://app.example.com",
                "https://staging.example.com",
            ]

    def test_env_override_strips_whitespace(self):
        with patch.dict(os.environ, {"KESTREL_CORS_ORIGINS": " https://a.com , https://b.com "}):
            assert build_cors_origins() == ["https://a.com", "https://b.com"]


class TestCORSWildcardGuard:
    """A wildcard origin must fail closed — browsers forbid wildcard + credentials,
    so accepting it would silently mis-secure rather than 'open' CORS (#1679)."""

    def test_wildcard_origin_is_rejected(self):
        with patch.dict(os.environ, {"KESTREL_CORS_ORIGINS": "*"}):
            with pytest.raises(RuntimeError, match="(?i)wildcard|credential"):
                build_cors_origins()

    def test_wildcard_among_explicit_origins_is_rejected(self):
        with patch.dict(os.environ, {"KESTREL_CORS_ORIGINS": "https://a.com,*"}):
            with pytest.raises(RuntimeError):
                build_cors_origins()
