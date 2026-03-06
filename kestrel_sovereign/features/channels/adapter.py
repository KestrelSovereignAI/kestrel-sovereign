"""
Abstract Channel Adapter interface.

Every messaging channel (Telegram, Discord, Slack, etc.) implements this
interface so the agent can send and receive messages through a uniform API.

To add a new channel:
    1. Subclass ``ChannelAdapter``
    2. Implement all abstract methods/properties
    3. Register the adapter with ``ChannelRegistry.register()``
"""

from abc import ABC, abstractmethod
from typing import Optional

from .models import ChannelConfig, DeliveryReceipt, MessageCallback


class ChannelAdapter(ABC):
    """
    Abstract base class for pluggable messaging channel adapters.

    Each adapter manages the lifecycle of a single channel type
    (connect, send, receive, disconnect) and exposes its connection
    status and channel identifier.
    """

    def __init__(self, config: Optional[ChannelConfig] = None):
        """
        Initialize the adapter with optional configuration.

        Args:
            config: Channel-specific configuration. Subclasses may
                    require this or allow it to be set later.
        """
        self._config = config

    @property
    def config(self) -> Optional[ChannelConfig]:
        """Return the current channel configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish a connection to the messaging service.

        Implementations should be idempotent (calling connect on an
        already-connected adapter is a no-op).
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Cleanly disconnect from the messaging service.

        Implementations should be idempotent (calling disconnect on
        an already-disconnected adapter is a no-op).
        """
        ...

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    @abstractmethod
    async def send_message(
        self,
        to: str,
        content: str,
        **kwargs,
    ) -> DeliveryReceipt:
        """
        Send a message through this channel.

        Args:
            to: Recipient identifier (channel-specific format).
            content: Text content of the message.
            **kwargs: Additional channel-specific options
                      (e.g. reply_to, parse_mode).

        Returns:
            A DeliveryReceipt indicating success/failure.
        """
        ...

    @abstractmethod
    async def on_message(self, callback: MessageCallback) -> None:
        """
        Register a callback to be invoked when an inbound message arrives.

        Multiple callbacks may be registered; the adapter must invoke
        all of them for each inbound message.

        Args:
            callback: Async callable that receives a ChannelMessage.
        """
        ...

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def channel_type(self) -> str:
        """
        Unique identifier for this channel type.

        Examples: ``"telegram"``, ``"discord"``, ``"slack"``.
        """
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the adapter currently has an active connection."""
        ...
