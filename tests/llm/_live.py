"""Explicit opt-in gate for tests that call paid, network-backed provider APIs.

A live test needs both ``KESTREL_LIVE_TESTS=1`` and the provider's
credentials. Credentials alone never authorize a paid request: a developer
who exports a provider key and runs ``pytest tests/llm`` must not spend money.
Call the gate before constructing any provider client.
"""

import os

import pytest

LIVE_TESTS_ENV = "KESTREL_LIVE_TESTS"
PROVIDER_CREDENTIAL_ENV_VARS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "vertex_ai": ("GOOGLE_API_KEY", "GCP_PROJECT_ID"),
}


def provider_available(provider_name: str) -> bool:
    """Return whether credentials for a declared provider exist."""
    return any(
        os.environ.get(variable, "").strip()
        for variable in PROVIDER_CREDENTIAL_ENV_VARS.get(provider_name, ())
    )


def require_live_opt_in() -> None:
    """Skip before external client use unless live tests are explicitly enabled."""
    if os.environ.get(LIVE_TESTS_ENV) != "1":
        pytest.skip(f"live test requires explicit opt-in: set {LIVE_TESTS_ENV}=1")


def require_live_provider(provider_name: str) -> None:
    """Require explicit live opt-in and credentials for one provider."""
    require_live_opt_in()
    if not provider_available(provider_name):
        credential_names = " or ".join(PROVIDER_CREDENTIAL_ENV_VARS[provider_name])
        pytest.skip(f"{provider_name} requires {credential_names}")
