# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.
"""Components for managing tasks within the A2A server."""

import logging

from kestrel_sovereign._vendor.a2a.server.tasks.base_push_notification_sender import (
    BasePushNotificationSender,
)
from kestrel_sovereign._vendor.a2a.server.tasks.inmemory_push_notification_config_store import (
    InMemoryPushNotificationConfigStore,
)
from kestrel_sovereign._vendor.a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from kestrel_sovereign._vendor.a2a.server.tasks.push_notification_config_store import (
    PushNotificationConfigStore,
)
from kestrel_sovereign._vendor.a2a.server.tasks.push_notification_sender import (
    PushNotificationEvent,
    PushNotificationSender,
)
from kestrel_sovereign._vendor.a2a.server.tasks.result_aggregator import ResultAggregator
from kestrel_sovereign._vendor.a2a.server.tasks.task_manager import TaskManager
from kestrel_sovereign._vendor.a2a.server.tasks.task_store import TaskStore
from kestrel_sovereign._vendor.a2a.server.tasks.task_updater import TaskUpdater


logger = logging.getLogger(__name__)

try:
    from kestrel_sovereign._vendor.a2a.server.tasks.database_task_store import (
        DatabaseTaskStore,
    )
except ImportError as e:
    _original_error = e
    # If the database task store is not available, we can still use in-memory stores.
    logger.debug(
        'DatabaseTaskStore not loaded. This is expected if database dependencies are not installed. Error: %s',
        e,
    )

    class DatabaseTaskStore:
        """Placeholder for DatabaseTaskStore when dependencies are not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'To use DatabaseTaskStore, its dependencies must be installed. '
                'You can install them with \'pip install "a2a-sdk[sql]"\''
            ) from _original_error


try:
    from kestrel_sovereign._vendor.a2a.server.tasks.database_push_notification_config_store import (
        DatabasePushNotificationConfigStore,
    )
except ImportError as e:
    _original_error = e
    # If the database push notification config store is not available, we can still use in-memory stores.
    logger.debug(
        'DatabasePushNotificationConfigStore not loaded. This is expected if database dependencies are not installed. Error: %s',
        e,
    )

    class DatabasePushNotificationConfigStore:
        """Placeholder for DatabasePushNotificationConfigStore when dependencies are not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'To use DatabasePushNotificationConfigStore, its dependencies must be installed. '
                'You can install them with \'pip install "a2a-sdk[sql]"\''
            ) from _original_error


__all__ = [
    'BasePushNotificationSender',
    'DatabasePushNotificationConfigStore',
    'DatabaseTaskStore',
    'InMemoryPushNotificationConfigStore',
    'InMemoryTaskStore',
    'PushNotificationConfigStore',
    'PushNotificationEvent',
    'PushNotificationSender',
    'ResultAggregator',
    'TaskManager',
    'TaskStore',
    'TaskUpdater',
]
