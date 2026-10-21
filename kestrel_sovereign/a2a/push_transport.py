"""Native push POSTs with pinned egress and original HTTP resource retirement."""

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import timedelta

import httpx
from a2a.server.tasks.push_notification_sender import PushNotificationEvent
from a2a.types import (
    Task,
    TaskArtifactUpdateEvent,
    TaskPushNotificationConfig,
    TaskStatusUpdateEvent,
)
from a2a.utils.constants import (
    A2A_JSON_MEDIA_TYPE,
    PROTOCOL_VERSION_1_0,
    VERSION_HEADER,
)
from a2a.utils.proto_utils import to_stream_response, validate_proto_required_fields
from google.protobuf.json_format import MessageToJson
from kestrel_sdk.security.outbound import assert_safe_url, pinned_httpx_async_transport

from kestrel_sovereign._async_ownership import await_owned_operation
from kestrel_sovereign.a2a.stores.unified.push_config_store import (
    PushConfigConflict,
    PushConfigStore,
    PushConfiguration,
    PushDeliveryAttempt,
)
from kestrel_sovereign.execution_custody import (
    bind_native_http_execution_admission,
    execution_cleanup_debts,
    execution_terminal_error,
    native_execution_response_owner,
    owned_native_http_client,
)


class PushDeliveryRevoked(PushConfigConflict):
    """The exact callback generation no longer admits an external request."""


def push_headers(config: TaskPushNotificationConfig) -> dict[str, str]:
    """Validate and build the standard callback headers without exposing secrets."""
    headers = {
        "Content-Type": A2A_JSON_MEDIA_TYPE,
        VERSION_HEADER: PROTOCOL_VERSION_1_0,
    }
    if config.token:
        headers["X-A2A-Notification-Token"] = config.token
    if config.HasField("authentication"):
        scheme, credentials = (
            config.authentication.scheme,
            config.authentication.credentials,
        )
        if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", scheme) or not credentials:
            raise PushConfigConflict(
                "Push authentication requires a scheme and credentials"
            )
        headers["Authorization"] = f"{scheme} {credentials}"
    # Refuse header injection and invalid HTTP header bytes before admitting
    # any network effect. Credentials are never interpolated into diagnostics.
    if any(
        any(ord(char) < 32 or ord(char) > 126 for char in value)
        for value in headers.values()
    ):
        raise PushConfigConflict(
            "Push authentication contains invalid HTTP header bytes"
        )
    return headers


async def validate_push_configuration(
    config: TaskPushNotificationConfig,
) -> TaskPushNotificationConfig:
    """Refuse unsafe or malformed callbacks before any registration write."""
    if not isinstance(config, TaskPushNotificationConfig):
        raise TypeError("Native push requires an official TaskPushNotificationConfig")
    config = TaskPushNotificationConfig.FromString(config.SerializeToString())
    validate_proto_required_fields(config)
    push_headers(config)
    await assert_safe_url(config.url, allowed_schemes=("https",))
    return config


class NativePushTransport:
    """Deliver one event; its durable worker owns retry and lifetime inventory.

    A successful return proves both an HTTP success and physical retirement.
    Failed delivery also retires its attempt when cleanup is confirmed. A
    native cleanup debt stays unretired and prevents Delete from reporting a
    false acknowledgement. No background task or implicit retry is created.
    """

    def __init__(self, store: PushConfigStore, *, timeout: timedelta):
        if not isinstance(store, PushConfigStore):
            raise TypeError("Native push requires its primary configuration ledger")
        if not isinstance(timeout, timedelta) or timeout <= timedelta():
            raise ValueError("Push delivery timeout must be a positive duration")
        self.store = store
        self.timeout = timeout.total_seconds()

    async def send(
        self,
        configuration: PushConfiguration,
        event: PushNotificationEvent,
        *,
        executor_id: str,
        before_dispatch: Callable[[], Awaitable[None]] | None = None,
    ) -> PushDeliveryAttempt:
        if not isinstance(
            event, (Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent)
        ):
            raise TypeError("Native push requires an official task event")
        validate_proto_required_fields(event)
        task_id = event.id if isinstance(event, Task) else event.task_id
        if task_id != configuration.config.task_id:
            raise PushConfigConflict("Push event belongs to another native task")
        headers = push_headers(configuration.config)
        payload = MessageToJson(to_stream_response(event)).encode("utf-8")
        attempt = self.store.prepare_delivery(configuration, executor_id=executor_id)
        admitted = None
        original = None
        try:
            admission = await await_owned_operation(
                self.store.admit_delivery(attempt),
                retirement_resource=attempt,
                retain_originals=True,
                name="native-push-admit-http-attempt",
            )
            admitted = admission.result
            error = execution_terminal_error(
                admission.cancellation,
                admission.error,
                retain_originals=True,
                prefer_cancellation=True,
            )
            if error is not None:
                raise error
            if not admitted:
                raise PushDeliveryRevoked("Push configuration is no longer active")
            validated = await assert_safe_url(
                configuration.config.url, allowed_schemes=("https",)
            )

            async def authorize():
                if not await self.store.delivery_is_current(attempt):
                    raise PushDeliveryRevoked(
                        "Push configuration was revoked before HTTP dispatch"
                    )
                if before_dispatch is not None:
                    await before_dispatch()

            async with owned_native_http_client(
                httpx.AsyncClient(
                    transport=pinned_httpx_async_transport(validated),
                    timeout=httpx.Timeout(self.timeout),
                    follow_redirects=False,
                    trust_env=False,
                )
            ) as http:
                with bind_native_http_execution_admission(http, authorize):
                    async with native_execution_response_owner(http):
                        # Bound the whole exchange, including a slow header
                        # producer; per-read socket deadlines alone do not.
                        async with asyncio.timeout(self.timeout):
                            response = await http.send(
                                http.build_request(
                                    "POST",
                                    validated.url,
                                    content=payload,
                                    headers=headers,
                                ),
                                stream=True,
                            )
                            response.raise_for_status()
                        # The callback body is not a protocol response. The
                        # native owner closes it without waiting for an
                        # unbounded body or interpreting it as an A2A task.
        except BaseException as error:  # noqa: BLE001 - original resource finalization
            # This is a resource-finalization boundary: preserve cancellation,
            # original execution control and failed physical cleanup alike.
            original = error
        if admitted is not False and not execution_cleanup_debts(original):
            outcome = await await_owned_operation(
                self.store.retire_delivery(attempt),
                retirement_resource=attempt,
                retain_originals=True,
                name="native-push-retire-http-attempt",
            )
            original = execution_terminal_error(
                original,
                outcome.error,
                outcome.cancellation,
                retain_originals=True,
            )
            if original is None and outcome.result is not True:
                original = PushConfigConflict(
                    "Push delivery lost its original retirement record"
                )
        if original is not None:
            raise original
        return attempt
