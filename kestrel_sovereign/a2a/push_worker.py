"""Owned native push execution, durable retries, and physical Delete retirement."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from uuid import uuid4

import httpx
from kestrel_sdk.security.exceptions import SSRFError

from kestrel_sovereign._async_ownership import (
    await_owned_operation,
    await_owned_task,
    gather_owned,
)
from kestrel_sovereign.a2a.push_outbox import NativePushOutbox, PushSubscription
from kestrel_sovereign.a2a.push_transport import (
    NativePushTransport,
    PushDeliveryRevoked,
)
from kestrel_sovereign.a2a.stores.unified.push_config_store import PushConfigConflict
from kestrel_sovereign.execution_custody import (
    execution_terminal_error,
    is_execution_control_error,
)
from kestrel_sovereign.signals.durable import DurableDelivery

logger = logging.getLogger(__name__)


class PushDeliveryLeaseLost(PushConfigConflict):
    """The original delivery token no longer authorizes external execution."""


class PushRetirementPending(PushConfigConflict):
    """Delete revoked admission but an original HTTP owner has not retired."""


@dataclass
class _DeliveryRun:
    subscription: PushSubscription
    revoked: bool = False


class NativePushWorker:
    """One host-owned runner using the existing durable lease/ACK contract.

    The host owns ``run`` through its regular resident-task lifetime. Every
    external request is a physically joined child; cancellation and Delete
    cannot acknowledge an HTTP request whose resource owners remain alive.
    Independent workers share durable claims, not an in-memory delivery queue.
    """

    def __init__(
        self,
        outbox: NativePushOutbox,
        transport: NativePushTransport,
        *,
        poll_interval: timedelta,
        renewal_interval: timedelta,
        retry_delay: timedelta,
        batch_size: int,
    ):
        if not isinstance(outbox, NativePushOutbox) or not isinstance(
            transport, NativePushTransport
        ):
            raise TypeError(
                "Native push requires its durable outbox and owned transport"
            )
        if transport.store is not outbox.configurations:
            raise ValueError(
                "Push runner and transport must share the exact configuration store"
            )
        for name, duration in (
            ("poll", poll_interval),
            ("renewal", renewal_interval),
            ("retry", retry_delay),
        ):
            if not isinstance(duration, timedelta) or duration <= timedelta():
                raise ValueError(f"Push {name} interval must be positive")
        if renewal_interval.total_seconds() >= outbox.lease_seconds:
            raise ValueError("Push renewal interval must be shorter than its lease")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Push batch size must be positive")
        self.outbox = outbox
        self.transport = transport
        self.poll_interval = poll_interval.total_seconds()
        self.renewal_interval = renewal_interval.total_seconds()
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.executor_id = "a2a.push:" + uuid4().hex
        self._active: dict[asyncio.Task, _DeliveryRun] = {}
        self._running = False
        self.started = asyncio.Event()

    @property
    def ready(self) -> bool:
        """Only the host's live managed runner may support an advertised card."""
        return self._running and self.started.is_set()

    async def run(self) -> None:
        """Run until the host cancels and physically joins this resident task."""
        if self._running:
            raise RuntimeError("Native push runner is already active")
        self.started.clear()
        self._running = True
        try:
            await self.outbox.forward_pending(limit=self.batch_size)
            self.started.set()
            while True:
                await self.run_once()
                await asyncio.sleep(self.poll_interval)
        finally:
            self._running = False

    def _lease_parameters(self, delivery: DurableDelivery) -> dict:
        if (
            delivery.agent_id != self.outbox.agent_id
            or delivery.lease_owner != self.executor_id
            or not delivery.lease_token
        ):
            raise PushDeliveryLeaseLost("Push execution has no original delivery lease")
        return {
            "agent_id": delivery.agent_id,
            "consumer_id": delivery.consumer_id,
            "delivery_id": delivery.delivery_id,
            "lease_token": delivery.lease_token,
        }

    async def _owned_transition(self, operation, delivery: DurableDelivery):
        """Join the real database transition before reporting cancellation.

        Canceling a monitor or HTTP sibling must not interrupt a lease COMMIT
        and manufacture uncertainty during an otherwise ordinary shutdown.
        Genuine commit failures retain their original control evidence.
        """
        outcome = await await_owned_operation(
            operation,
            retirement_resource=delivery,
            retain_originals=True,
            name="native-push-ledger-transition",
        )
        error = execution_terminal_error(
            outcome.cancellation,
            outcome.error,
            retain_originals=True,
            prefer_cancellation=True,
        )
        if error is not None:
            raise error
        return outcome.result

    async def _renew(self, delivery: DurableDelivery) -> None:
        if (
            await self._owned_transition(
                self.outbox.signals.renew_delivery_lease(
                    **self._lease_parameters(delivery)
                ),
                delivery,
            )
            is None
        ):
            raise PushDeliveryLeaseLost(
                "Push execution lost its original delivery lease"
            )

    async def _deliver(
        self,
        subscription: PushSubscription,
        delivery: DurableDelivery,
        run: _DeliveryRun,
    ) -> None:
        task = asyncio.current_task()
        self._active[task] = run
        try:
            payload = delivery.event.payload
            if (
                delivery.consumer_id != subscription.consumer_id
                or delivery.event.source != self.outbox.SOURCE
                or not isinstance(payload, dict)
                or set(payload) != {"task_id", "journal_sequence"}
                or payload["task_id"] != subscription.task_id
                or type(payload["journal_sequence"]) is not int
                or payload["journal_sequence"] <= 0
            ):
                raise PushConfigConflict(
                    "Push delivery differs from its original subscription"
                )
            if payload["journal_sequence"] > subscription.after_sequence:
                configuration = await self.outbox.configurations.get(
                    subscription.task_id,
                    subscription.config_id,
                    creator_id=subscription.creator_id,
                )
                if (
                    configuration is None
                    or configuration.generation != subscription.generation
                ):
                    raise PushDeliveryRevoked(
                        "Push subscription generation is no longer active"
                    )
                native = await self.outbox.configurations.task_store.get_for_creator(
                    subscription.task_id,
                    subscription.creator_id,
                    recipient_agent_id=self.outbox.configurations.scope.recipient_id,
                )
                if native is None:
                    raise PushConfigConflict(
                        "Active push configuration lost its canonical task"
                    )
                finished = asyncio.Event()

                async def post():
                    try:
                        await self.transport.send(
                            configuration,
                            native,
                            executor_id=self.executor_id,
                            before_dispatch=lambda: self._renew(delivery),
                        )
                    finally:
                        finished.set()

                async def monitor():
                    while not finished.is_set():
                        current = await self.outbox.configurations.get(
                            subscription.task_id,
                            subscription.config_id,
                            creator_id=subscription.creator_id,
                        )
                        if (
                            current is None
                            or current.generation != subscription.generation
                        ):
                            raise PushDeliveryRevoked(
                                "Push configuration was revoked during delivery"
                            )
                        await self._renew(delivery)
                        try:
                            await asyncio.wait_for(
                                finished.wait(), timeout=self.renewal_interval
                            )
                        except TimeoutError:
                            continue

                await gather_owned(
                    (post(), monitor()),
                    retain_originals=True,
                    retirement_resource=delivery,
                    name_prefix="native-push-request",
                )
            if not await self._owned_transition(
                self.outbox.signals.ack_delivery(**self._lease_parameters(delivery)),
                delivery,
            ):
                raise PushDeliveryLeaseLost(
                    "Push acknowledgement lost its original delivery lease"
                )
        finally:
            self._active.pop(task, None)

    async def run_once(self) -> int:
        """Forward a bounded journal batch and attempt one delivery per subscriber."""
        await self.outbox.forward_pending(limit=self.batch_size)
        count = 0
        for subscription in await self.outbox.subscriptions():
            delivery = await self.outbox.signals.claim_delivery(
                agent_id=self.outbox.agent_id,
                consumer_id=subscription.consumer_id,
                executor_id=self.executor_id,
            )
            if delivery is None:
                continue
            count += 1
            run = _DeliveryRun(subscription)
            outcome = await await_owned_operation(
                self._deliver(subscription, delivery, run),
                retirement_resource=delivery,
                retain_originals=True,
                name="native-push-delivery",
                cancel_on_caller_cancellation=True,
            )
            error = outcome.error
            # This worker's Delete deliberately cancels this exact child.
            # External cancellation and original custody control still win.
            if (
                run.revoked
                and isinstance(error, asyncio.CancelledError)
                and not is_execution_control_error(error)
            ):
                error = None
            error = execution_terminal_error(
                outcome.cancellation,
                error,
                retain_originals=True,
                prefer_cancellation=True,
            )
            if error is not None:
                if is_execution_control_error(error):
                    raise error
                if isinstance(error, (PushDeliveryRevoked, PushDeliveryLeaseLost)):
                    continue
                if not isinstance(error, (httpx.HTTPError, TimeoutError, SSRFError)):
                    raise error
                # Native HTTP errors may contain secret URLs/headers. Durable
                # retry diagnostics retain only the type, never their text.
                released = await self._owned_transition(
                    self.outbox.signals.nack_delivery(
                        **self._lease_parameters(delivery),
                        error=type(error).__name__,
                        retry_delay=self.retry_delay,
                    ),
                    delivery,
                )
                if released is None:
                    raise PushDeliveryLeaseLost(
                        "Push retry lost its original delivery lease"
                    ) from error
                logger.warning(
                    "Native push delivery will retry after %s", type(error).__name__
                )
        return count

    async def delete(self, task_id: str, config_id: str, *, creator_id: str) -> None:
        """Acknowledge Delete only after revocation and original HTTP retirement."""
        config = await self.outbox.revoke(task_id, config_id, creator_id=creator_id)
        if config is None:
            return
        original = None
        owners = [
            (task, run)
            for task, run in tuple(self._active.items())
            if (
                run.subscription.task_id,
                run.subscription.config_id,
                run.subscription.generation,
            )
            == (task_id, config_id, config.generation)
        ]
        for task, run in owners:
            run.revoked = True
            task.cancel()
        for task, _ in owners:
            outcome = await await_owned_task(task)
            error = outcome.error
            if isinstance(
                error, asyncio.CancelledError
            ) and not is_execution_control_error(error):
                error = None
            original = execution_terminal_error(
                original, error, outcome.cancellation, retain_originals=True
            )
        if original is not None:
            raise original
        if not await self.outbox.configurations.finish_delete(config):
            raise PushRetirementPending(
                "Push deletion is waiting for an original HTTP owner to retire"
            )
