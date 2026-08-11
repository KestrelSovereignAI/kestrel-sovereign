"""Pluggable messaging channel adapters for the agent."""

from .feature import ChannelFeature
from .route_ownership import ChannelRouteClaim, ChannelRouteOwnershipStore

__all__ = ["ChannelFeature", "ChannelRouteClaim", "ChannelRouteOwnershipStore"]
