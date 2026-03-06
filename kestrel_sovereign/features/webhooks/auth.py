"""
Authentication handlers for incoming webhooks.

Each handler validates a request against webhook-specific credentials.
All comparison operations use timing-safe algorithms to prevent
side-channel attacks.
"""

import hashlib
import hmac
import ipaddress
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class WebhookAuth(ABC):
    """Base class for webhook authentication strategies."""

    @abstractmethod
    def validate(
        self,
        *,
        headers: Dict[str, str],
        body: bytes,
        source_ip: str,
    ) -> bool:
        """Return True if the request is authenticated.

        Args:
            headers: HTTP request headers (lower-cased keys).
            body: Raw request body bytes.
            source_ip: Client IP address string.
        """
        ...


class NoAuth(WebhookAuth):
    """No authentication -- accepts every request.

    Suitable only for testing or trusted-network deployments.
    """

    def validate(
        self,
        *,
        headers: Dict[str, str],
        body: bytes,
        source_ip: str,
    ) -> bool:
        return True


class BearerTokenAuth(WebhookAuth):
    """Validate a Bearer token in the Authorization header.

    Expected auth_config:
        {"token": "<shared-secret>"}
    """

    def __init__(self, auth_config: Dict[str, Any]):
        self.expected_token: str = auth_config.get("token", "")
        if not self.expected_token:
            raise ValueError("BearerTokenAuth requires a non-empty 'token' in auth_config")

    def validate(
        self,
        *,
        headers: Dict[str, str],
        body: bytes,
        source_ip: str,
    ) -> bool:
        auth_header = headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            logger.debug("BearerTokenAuth: missing or malformed Authorization header")
            return False

        provided_token = auth_header[7:]  # strip "Bearer "
        return hmac.compare_digest(provided_token, self.expected_token)


class HMACSignatureAuth(WebhookAuth):
    """Validate an HMAC-SHA256 signature of the request body.

    This is the pattern used by GitHub, Stripe, and many other webhook
    providers. The sender computes ``HMAC-SHA256(secret, body)`` and
    sends the hex digest in a configurable header.

    Expected auth_config:
        {
            "secret": "<shared-hmac-secret>",
            "header": "X-Hub-Signature-256"   # optional, defaults to X-Hub-Signature-256
            "prefix": "sha256="               # optional, defaults to "sha256="
        }
    """

    def __init__(self, auth_config: Dict[str, Any]):
        self.secret: str = auth_config.get("secret", "")
        if not self.secret:
            raise ValueError("HMACSignatureAuth requires a non-empty 'secret' in auth_config")
        self.header_name: str = auth_config.get("header", "x-hub-signature-256").lower()
        self.prefix: str = auth_config.get("prefix", "sha256=")

    def validate(
        self,
        *,
        headers: Dict[str, str],
        body: bytes,
        source_ip: str,
    ) -> bool:
        signature_header = headers.get(self.header_name, "")
        if not signature_header:
            logger.debug("HMACSignatureAuth: missing signature header '%s'", self.header_name)
            return False

        # Strip the prefix (e.g. "sha256=")
        if self.prefix and signature_header.startswith(self.prefix):
            provided_hex = signature_header[len(self.prefix) :]
        else:
            provided_hex = signature_header

        expected_hex = hmac.new(
            self.secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(provided_hex, expected_hex)


class IPAllowlistAuth(WebhookAuth):
    """Validate that the source IP is in a configured allowlist.

    Supports individual IPs and CIDR ranges (e.g. ``192.168.1.0/24``).

    Expected auth_config:
        {"allowed_ips": ["192.168.1.0/24", "10.0.0.5"]}
    """

    def __init__(self, auth_config: Dict[str, Any]):
        raw_ips = auth_config.get("allowed_ips", [])
        if not raw_ips:
            raise ValueError("IPAllowlistAuth requires a non-empty 'allowed_ips' list")

        self.networks = []
        for entry in raw_ips:
            try:
                self.networks.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                logger.warning("IPAllowlistAuth: ignoring invalid network '%s'", entry)

        if not self.networks:
            raise ValueError("IPAllowlistAuth: no valid networks after parsing 'allowed_ips'")

    def validate(
        self,
        *,
        headers: Dict[str, str],
        body: bytes,
        source_ip: str,
    ) -> bool:
        try:
            addr = ipaddress.ip_address(source_ip)
        except ValueError:
            logger.debug("IPAllowlistAuth: invalid source IP '%s'", source_ip)
            return False

        for network in self.networks:
            if addr in network:
                return True

        logger.debug("IPAllowlistAuth: source IP '%s' not in allowlist", source_ip)
        return False


def create_auth_handler(auth_type_value: str, auth_config: Dict[str, Any]) -> WebhookAuth:
    """Factory function to create the correct auth handler from stored config.

    Args:
        auth_type_value: The string value of a WebhookAuthType enum member.
        auth_config: Auth-specific configuration dict.

    Returns:
        An instantiated WebhookAuth subclass.

    Raises:
        ValueError: If the auth_type is unknown.
    """
    from .models import WebhookAuthType

    try:
        auth_type = WebhookAuthType(auth_type_value)
    except ValueError:
        raise ValueError(f"Unknown webhook auth type: {auth_type_value}")

    if auth_type == WebhookAuthType.NONE:
        return NoAuth()
    elif auth_type == WebhookAuthType.BEARER_TOKEN:
        return BearerTokenAuth(auth_config)
    elif auth_type == WebhookAuthType.HMAC_SHA256:
        return HMACSignatureAuth(auth_config)
    elif auth_type == WebhookAuthType.IP_ALLOWLIST:
        return IPAllowlistAuth(auth_config)
    else:
        raise ValueError(f"Unsupported webhook auth type: {auth_type_value}")
