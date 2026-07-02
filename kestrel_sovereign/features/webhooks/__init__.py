"""
Generic Webhook Receiver feature for Kestrel agents.

Allows features to register custom HTTP webhook endpoints with
configurable authentication (Bearer token, HMAC-SHA256, IP allowlist,
or none). All webhook receives are logged for security audit.

Usage:
    The WebhookFeature is auto-discovered by the feature loader, and its
    receiver router is mounted automatically via the standard
    ``Feature.get_router()`` contract — no manual wiring in server.py is
    required.
"""

from .feature import WebhookFeature

__all__ = ["WebhookFeature"]
