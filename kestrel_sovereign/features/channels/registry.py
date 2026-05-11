"""
Channel Registry -- central manager for channel adapters.

Handles adapter registration, lookup, and inbound message routing.

External packages can register channel adapters via entry_points::

    [project.entry-points."kestrel_sovereign.channel_adapters"]
    TelegramAdapter = "kestrel_channel_telegram:TelegramAdapter"
"""

import logging
from typing import Awaitable, Callable, Dict, List, Optional

from kestrel_sdk.channels import CHANNEL_ADAPTER_ENTRY_POINT_GROUP
from kestrel_sovereign.entrypoints import discover_entry_point_classes
from .adapter import ChannelAdapter
from .models import ChannelMessage, MessageDirection

logger = logging.getLogger(__name__)

# Callback type for routing inbound messages to an agent/session
InboundRouter = Callable[[ChannelMessage], Awaitable[None]]


class ChannelRegistry:
    """
    Manages the set of active channel adapters for an agent.

    Responsibilities:
    - Register / unregister adapters
    - Look up adapters by channel_type
    - List all channels with status
    - Route inbound messages to the correct handler
    - Discover external adapters via entry_points
    """

    def __init__(self):
        self._adapters: Dict[str, ChannelAdapter] = {}
        self._adapter_classes: Dict[str, type] = {}
        self._inbound_router: Optional[InboundRouter] = None
        self._discover_entrypoint_adapters()

    # ------------------------------------------------------------------
    # Entry Point Discovery
    # ------------------------------------------------------------------

    def _discover_entrypoint_adapters(self) -> None:
        """Discover external ChannelAdapter classes via entry_points.

        Discovered classes are stored in ``_adapter_classes`` for later
        instantiation.  They are NOT auto-connected — call
        ``create_adapter(channel_type, config)`` to instantiate and register.
        """
        classes = discover_entry_point_classes(
            CHANNEL_ADAPTER_ENTRY_POINT_GROUP, ChannelAdapter,
        )
        for ep_name, cls in classes.items():
            self._adapter_classes[ep_name] = cls
            logger.info("Discovered entry_point channel adapter: %s", ep_name)

    def get_adapter_class(self, name: str) -> Optional[type]:
        """Get a discovered adapter class by entry point name.

        Args:
            name: Entry point name (e.g. "telegram", "discord").

        Returns:
            The ChannelAdapter subclass, or None if not found.
        """
        return self._adapter_classes.get(name)

    def list_discovered_adapters(self) -> List[str]:
        """List names of adapter classes discovered via entry_points."""
        return list(self._adapter_classes.keys())

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, adapter: ChannelAdapter) -> None:
        """
        Add a channel adapter to the registry.

        If an adapter for the same channel_type already exists it will
        be replaced (the caller is responsible for disconnecting the old
        adapter first).

        Args:
            adapter: The ChannelAdapter instance to register.
        """
        channel = adapter.channel_type
        if channel in self._adapters:
            logger.warning(
                "Replacing existing adapter for channel '%s'", channel
            )
        self._adapters[channel] = adapter
        logger.info("Registered channel adapter: %s", channel)

    def unregister(self, channel_type: str) -> Optional[ChannelAdapter]:
        """
        Remove a channel adapter from the registry.

        Args:
            channel_type: The channel type to remove.

        Returns:
            The removed adapter, or None if not found.
        """
        adapter = self._adapters.pop(channel_type, None)
        if adapter:
            logger.info("Unregistered channel adapter: %s", channel_type)
        else:
            logger.debug(
                "Attempted to unregister unknown channel: %s", channel_type
            )
        return adapter

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, channel_type: str) -> Optional[ChannelAdapter]:
        """
        Get an adapter by its channel type.

        Args:
            channel_type: e.g. "telegram", "discord", "slack"

        Returns:
            The matching ChannelAdapter or None.
        """
        return self._adapters.get(channel_type)

    def list_channels(self) -> List[Dict[str, object]]:
        """
        Return a summary of all registered channels and their status.

        Returns:
            List of dicts with keys: channel_type, is_connected, enabled.
        """
        result = []
        for channel_type, adapter in self._adapters.items():
            config = adapter.config
            result.append({
                "channel_type": channel_type,
                "is_connected": adapter.is_connected,
                "enabled": config.enabled if config else True,
            })
        return result

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def set_inbound_router(self, router: InboundRouter) -> None:
        """
        Set the callback used to route inbound messages to the agent.

        Args:
            router: Async callable that processes an inbound ChannelMessage.
        """
        self._inbound_router = router

    async def route_message(self, message: ChannelMessage) -> None:
        """
        Route an inbound message to the registered handler.

        If no inbound router has been set, the message is logged and dropped.

        Args:
            message: The inbound ChannelMessage to route.
        """
        if message.direction != MessageDirection.INBOUND:
            logger.warning(
                "route_message called with non-inbound message (id=%s, dir=%s); ignoring",
                message.id,
                message.direction.value,
            )
            return

        if self._inbound_router is None:
            logger.warning(
                "No inbound router configured; dropping message id=%s from channel=%s",
                message.id,
                message.channel_type,
            )
            return

        try:
            await self._inbound_router(message)
        except Exception:
            logger.exception(
                "Error routing inbound message id=%s from channel=%s",
                message.id,
                message.channel_type,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def adapter_count(self) -> int:
        """Number of registered adapters."""
        return len(self._adapters)

    def __contains__(self, channel_type: str) -> bool:
        return channel_type in self._adapters
