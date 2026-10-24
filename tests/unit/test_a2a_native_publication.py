"""Public native discovery requires explicit, validated host configuration."""

import pytest

from kestrel_sovereign.a2a.native_publication import NativeHostingConfiguration


def test_operator_only_host_does_not_guess_a_public_peer_url():
    assert NativeHostingConfiguration.from_environment({}) is None
    assert (
        NativeHostingConfiguration.from_environment(
            {
                "HOST": "0.0.0.0",
                "PORT": "8888",
                "KESTREL_HOST_URL": "http://localhost:8888",
            }
        )
        is None
    )


def test_public_discovery_recipient_is_an_explicit_registration_key():
    configuration = NativeHostingConfiguration.from_environment(
        {
            "KESTREL_BASE_URL": "https://public.test/proxy",
            "KESTREL_A2A_DISCOVERY_AGENT": "Stable Key",
        }
    )
    assert configuration.public_base_url == "https://public.test/proxy"
    assert configuration.discovery_routing_key == "Stable Key"
    assert (
        NativeHostingConfiguration.from_environment(
            {
                "KESTREL_BASE_URL": "https://public.test",
            }
        ).discovery_routing_key
        is None
    )


@pytest.mark.parametrize(
    "environment",
    [
        {"KESTREL_A2A_DISCOVERY_AGENT": "configured"},
        {"KESTREL_BASE_URL": ""},
        {"KESTREL_BASE_URL": "https://user:password@host.test"},
        {"KESTREL_BASE_URL": "https://host.test#fragment"},
        {"KESTREL_BASE_URL": "https://host.test?token=secret"},
        {"KESTREL_BASE_URL": "https://host.test", "KESTREL_A2A_DISCOVERY_AGENT": ""},
    ],
)
def test_invalid_discovery_configuration_refuses_publication(environment):
    with pytest.raises(ValueError):
        NativeHostingConfiguration.from_environment(environment)
