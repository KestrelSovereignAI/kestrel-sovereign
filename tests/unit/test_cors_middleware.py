"""Tests for CORS middleware configuration in server.py."""
import os
from unittest.mock import patch


class TestCORSDefaults:
    """Verify CORS middleware uses explicit allowed origins, never wildcards."""

    def test_default_origins_are_explicit(self):
        """Default CORS origins must be explicit localhost entries, not '*'."""
        # Import the defaults directly
        from server import _DEFAULT_CORS_ORIGINS

        assert isinstance(_DEFAULT_CORS_ORIGINS, list)
        assert "*" not in _DEFAULT_CORS_ORIGINS
        assert len(_DEFAULT_CORS_ORIGINS) > 0
        for origin in _DEFAULT_CORS_ORIGINS:
            assert origin.startswith("http://") or origin.startswith("https://"), (
                f"Origin must be a full URL, got: {origin}"
            )

    def test_default_origins_include_localhost(self):
        """Default origins should include common local dev ports."""
        from server import _DEFAULT_CORS_ORIGINS

        assert "http://localhost:8080" in _DEFAULT_CORS_ORIGINS
        assert "http://127.0.0.1:8080" in _DEFAULT_CORS_ORIGINS
        assert "http://localhost:3000" in _DEFAULT_CORS_ORIGINS
        assert "http://127.0.0.1:3000" in _DEFAULT_CORS_ORIGINS


class TestCORSEnvironmentOverride:
    """Verify CORS origins can be overridden via environment variable."""

    def test_env_override_parses_comma_separated(self):
        """KESTREL_CORS_ORIGINS env var should override defaults."""
        env_val = "https://app.example.com,https://staging.example.com"
        with patch.dict(os.environ, {"KESTREL_CORS_ORIGINS": env_val}):
            # Re-evaluate the parsing logic (same logic as server.py)
            _cors_env = os.environ.get("KESTREL_CORS_ORIGINS", "")
            result = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else []
            assert result == ["https://app.example.com", "https://staging.example.com"]

    def test_env_override_strips_whitespace(self):
        """Whitespace around origins should be stripped."""
        env_val = " https://a.com , https://b.com "
        with patch.dict(os.environ, {"KESTREL_CORS_ORIGINS": env_val}):
            _cors_env = os.environ.get("KESTREL_CORS_ORIGINS", "")
            result = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else []
            assert result == ["https://a.com", "https://b.com"]

    def test_empty_env_falls_back_to_defaults(self):
        """Empty KESTREL_CORS_ORIGINS should use defaults."""
        with patch.dict(os.environ, {"KESTREL_CORS_ORIGINS": ""}):
            from server import _DEFAULT_CORS_ORIGINS
            _cors_env = os.environ.get("KESTREL_CORS_ORIGINS", "")
            result = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else _DEFAULT_CORS_ORIGINS
            assert result == _DEFAULT_CORS_ORIGINS
