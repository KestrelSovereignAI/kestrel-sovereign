"""Local-route client timeout (issue #1954).

Local models (llama.cpp) can legitimately generate for many minutes (large
reasoning models at low tok/s). The OpenAI SDK's ~600s default timeout cancels
those mid-flight — which surfaced as an agent "stopping after tools" when the
request was silently cancelled. The registry must give is_cloud=false vendors a
generous, route-configurable timeout.
"""

import pytest

from kestrel_sovereign.llm.provider_registry import ProviderRegistry


def _registry(route_extra=None):
    route = {
        "adapter": "OpenAIAdapter",
        "base_url": "http://localhost:8001/v1",
        "model": "auto",
    }
    if route_extra:
        route.update(route_extra)
    config = {
        "route_priority": ["llama_cpp:local"],
        "vendors": {
            "llama_cpp": {"is_cloud": False, "routes": {"local": route}},
        },
    }
    return ProviderRegistry(config=config)


def _llama_client(providers):
    return next(p for p in providers if p.vendor == "llama_cpp").client


def test_local_client_gets_generous_default_timeout():
    """is_cloud=false vendor → 1800s default instead of the SDK's 600s."""
    providers = _registry().initialize_providers()
    assert _llama_client(providers).timeout == 1800.0


def test_local_client_timeout_is_route_configurable():
    """Operators can raise/lower it per route via `timeout`."""
    providers = _registry({"timeout": 3600}).initialize_providers()
    assert _llama_client(providers).timeout == 3600.0


def test_local_flag_alone_also_triggers_timeout():
    """A route flagged local on an otherwise-cloud vendor still gets it."""
    config = {
        "route_priority": ["custom:local"],
        "vendors": {
            "custom": {
                "is_cloud": True,
                "routes": {
                    "local": {
                        "adapter": "OpenAIAdapter",
                        "base_url": "http://localhost:9999/v1",
                        "model": "auto",
                        "local": True,
                    }
                },
            }
        },
    }
    providers = ProviderRegistry(config=config).initialize_providers()
    client = next(p for p in providers if p.vendor == "custom").client
    assert client.timeout == 1800.0
