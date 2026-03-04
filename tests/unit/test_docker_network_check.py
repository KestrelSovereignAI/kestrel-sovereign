"""Tests for the _is_docker_network helper in server.py."""

import ipaddress
import pytest


def _is_docker_network(host: str) -> bool:
    """Mirror of the helper in server.py for isolated unit testing."""
    try:
        return ipaddress.ip_address(host) in ipaddress.ip_network("172.16.0.0/12")
    except ValueError:
        return False


class TestIsDockerNetwork:
    """Verify _is_docker_network accepts only Docker's 172.16.0.0/12 range."""

    # --- Should accept: Docker bridge network addresses ---

    def test_default_bridge_gateway(self):
        assert _is_docker_network("172.17.0.1") is True

    def test_default_bridge_container(self):
        assert _is_docker_network("172.17.0.2") is True

    def test_custom_network_range(self):
        assert _is_docker_network("172.18.0.1") is True

    def test_start_of_range(self):
        assert _is_docker_network("172.16.0.0") is True

    def test_end_of_range(self):
        assert _is_docker_network("172.31.255.255") is True

    # --- Should reject: IPs outside Docker's 172.16.0.0/12 ---

    def test_reject_172_0(self):
        """172.0.0.1 is outside 172.16.0.0/12 -- the old bug."""
        assert _is_docker_network("172.0.0.1") is False

    def test_reject_172_15(self):
        """172.15.255.255 is just below the Docker range."""
        assert _is_docker_network("172.15.255.255") is False

    def test_reject_172_32(self):
        """172.32.0.0 is just above the Docker range."""
        assert _is_docker_network("172.32.0.0") is False

    def test_reject_public_172(self):
        assert _is_docker_network("172.100.0.1") is False

    def test_reject_private_10(self):
        assert _is_docker_network("10.0.0.1") is False

    def test_reject_private_192(self):
        assert _is_docker_network("192.168.1.1") is False

    def test_reject_localhost(self):
        assert _is_docker_network("127.0.0.1") is False

    def test_reject_public_ip(self):
        assert _is_docker_network("8.8.8.8") is False

    # --- Edge cases ---

    def test_invalid_ip_returns_false(self):
        assert _is_docker_network("not-an-ip") is False

    def test_empty_string_returns_false(self):
        assert _is_docker_network("") is False

    def test_hostname_returns_false(self):
        assert _is_docker_network("localhost") is False

    def test_ipv6_loopback_returns_false(self):
        assert _is_docker_network("::1") is False
