"""Backward-compatible re-exports for SDK channel contracts.

Feature and channel packages should import these contracts from
``kestrel_sdk.channels`` directly. Sovereign re-exports them here while the
in-tree channels feature migrates to the SDK-owned public surface.
"""

from kestrel_sdk.channels import (  # noqa: F401
    ChannelConfig,
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
    MessageCallback,
    MessageDirection,
)

__all__ = [
    "ChannelConfig",
    "ChannelMessage",
    "DeliveryReceipt",
    "DeliveryStatus",
    "MessageCallback",
    "MessageDirection",
]
