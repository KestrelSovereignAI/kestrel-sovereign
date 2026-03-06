"""
Generic Webhook Receiver feature for Kestrel agents.

Allows features to register custom HTTP webhook endpoints with
configurable authentication (Bearer token, HMAC-SHA256, IP allowlist,
or none). All webhook receives are logged for security audit.

Usage:
    The WebhookFeature is auto-discovered by the feature loader.
    To mount the webhook router in server.py:

        # In server.py lifespan or after agent init:
        webhook_feature = get_feature_by_name(agent.features, "WebhookFeature")
        if webhook_feature:
            app.include_router(webhook_feature.get_webhook_router())
"""

from .feature import WebhookFeature

__all__ = ["WebhookFeature"]
