"""Public native discovery requires explicit, validated host configuration."""

import pytest

from kestrel_sovereign.a2a.native_publication import (
    NativeHostingConfiguration,
    unpublished_cohost_warning,
    unpublished_peer_detail,
)


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


def test_cohosted_fleet_without_base_url_is_named_with_its_remedy():
    warning = unpublished_cohost_warning({"PORT": "8899"}, agents=2, host_port=8899)
    assert warning is not None
    assert warning.startswith("KESTREL_BASE_URL is unset: none of this host's 2 agents")
    assert "cannot send A2A tasks, messages, questions or stops" in warning
    assert "KESTREL_BASE_URL=http://127.0.0.1:8899 and" in warning
    assert "KESTREL_A2A_PUSH_ENABLED=false" in warning
    # The port is advice only: the same environment still publishes nothing.
    assert NativeHostingConfiguration.from_environment({"PORT": "8899"}) is None


@pytest.mark.parametrize(
    ("environment", "agents"),
    [
        ({}, 0),
        ({}, 1),
        ({"KESTREL_BASE_URL": "http://127.0.0.1:8899"}, 2),
        # An invalid value is set, so onboarding refuses it rather than this
        # warning describing it as unset.
        ({"KESTREL_BASE_URL": ""}, 3),
    ],
)
def test_no_cohosted_warning_for_one_agent_or_a_set_base_url(environment, agents):
    assert (
        unpublished_cohost_warning(environment, agents=agents, host_port=8899)
        is None
    )


def test_unpublished_peer_names_a_host_without_base_url():
    detail = unpublished_peer_detail({})
    assert detail.startswith("Peer has no published native A2A interface: ")
    assert "this host has no KESTREL_BASE_URL" in detail
    assert (
        unpublished_peer_detail({"KESTREL_BASE_URL": "http://127.0.0.1:8899"})
        == "Peer has no published native A2A interface"
    )
