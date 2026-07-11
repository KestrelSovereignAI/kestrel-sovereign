"""Tests for the centralized route-credential resolver.

This is the single source of truth that keeps ``kestrel setup``,
``kestrel setup --check`` and ``kestrel doctor`` from disagreeing about
which ``.env`` vars satisfy an LLM route (#2245).
"""

from __future__ import annotations

from kestrel_sovereign.llm.route_credentials import accepted_credential_envs


def test_primary_api_key_only():
    envs = accepted_credential_envs("openai:api", {"api_key_env": "OPENAI_API_KEY"})
    assert envs == ["OPENAI_API_KEY"]


def test_local_route_needs_no_credential():
    assert accepted_credential_envs("ollama:local", {"adapter": "OllamaAdapter"}) == []


def test_openrouter_alt_key_applies_without_declared_management_env():
    # The route TOML omits management_api_key_env; the vendor fallback
    # still adds the management key so setup/check/doctor agree (#2245).
    envs = accepted_credential_envs(
        "openrouter:api", {"api_key_env": "OPENROUTER_API_KEY"}
    )
    assert envs == ["OPENROUTER_API_KEY", "OPENROUTER_MANAGEMENT_API_KEY"]


def test_declared_management_env_not_duplicated():
    envs = accepted_credential_envs(
        "openrouter:api",
        {
            "api_key_env": "OPENROUTER_API_KEY",
            "management_api_key_env": "OPENROUTER_MANAGEMENT_API_KEY",
        },
    )
    assert envs == ["OPENROUTER_API_KEY", "OPENROUTER_MANAGEMENT_API_KEY"]


def test_management_key_only_route():
    envs = accepted_credential_envs(
        "openrouter:api",
        {"management_api_key_env": "OPENROUTER_MANAGEMENT_API_KEY"},
    )
    assert envs == ["OPENROUTER_MANAGEMENT_API_KEY"]


def test_inline_api_key_needs_no_env():
    # The runtime falls back to an inline ``api_key`` when the env var is
    # unset (``_resolve_secret``), so check/doctor must not demand .env.
    envs = accepted_credential_envs(
        "openrouter:api",
        {"api_key_env": "OPENROUTER_API_KEY", "api_key": "sk-or-inline"},
    )
    assert envs == []


def test_inline_management_key_needs_no_env():
    envs = accepted_credential_envs(
        "openrouter:api", {"management_api_key": "sk-or-mgmt-inline"}
    )
    assert envs == []


def test_custom_management_env_suppresses_vendor_default():
    # The runtime reads ``management_api_key_env or <default>`` — a route
    # that declares a custom name is served only from that name, so the
    # default must NOT satisfy setup/check/doctor for it.
    envs = accepted_credential_envs(
        "openrouter:api",
        {
            "api_key_env": "OPENROUTER_API_KEY",
            "management_api_key_env": "MY_CUSTOM_MGMT_KEY",
        },
    )
    assert envs == ["OPENROUTER_API_KEY", "MY_CUSTOM_MGMT_KEY"]
