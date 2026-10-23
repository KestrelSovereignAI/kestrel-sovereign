# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.
"""Event handling components for the A2A server."""

from kestrel_sovereign._vendor.a2a.server.events.event_consumer import EventConsumer
from kestrel_sovereign._vendor.a2a.server.events.event_queue import Event, EventQueue, EventQueueLegacy
from kestrel_sovereign._vendor.a2a.server.events.in_memory_queue_manager import InMemoryQueueManager
from kestrel_sovereign._vendor.a2a.server.events.queue_manager import (
    NoTaskQueue,
    QueueManager,
    TaskQueueExists,
)


__all__ = [
    'Event',
    'EventConsumer',
    'EventQueue',
    'EventQueueLegacy',
    'InMemoryQueueManager',
    'NoTaskQueue',
    'QueueManager',
    'TaskQueueExists',
]
