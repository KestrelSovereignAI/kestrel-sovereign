"""Pluggable messaging channel adapters for the agent."""

from .feature import ChannelFeature
from .route_ownership import ChannelRouteOwnershipStore

__all__ = ["ChannelFeature", "ChannelRouteOwnershipStore"]
