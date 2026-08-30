"""The single scope-resolution seam for cooperative Stop."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    raise_owned_outcome,
)

from .receipt import StopOperationClaim, StopReceipt, StopReceiptConflict
from .types import StopDisposition, StopOutcome, StopRequest, StopScope

StopOperation = Callable[[StopRequest], Awaitable[StopDisposition]]
DEFAULT_STOP_TARGET_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class CooperativeStopTarget:
    """A snapshot of one agent's cooperative cancellation addresses."""

    target_id: str
    agent_id: str
    cancel: StopOperation
    turn_ids: frozenset[str] = field(default_factory=frozenset)
    tool_call_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_id, str)
            or not self.target_id.strip()
            or not isinstance(self.agent_id, str)
            or not self.agent_id.strip()
        ):
            raise ValueError("Stop targets require concrete target and agent identities")
        if not callable(self.cancel):
            raise TypeError("Stop target cancel operation must be callable")
        for field_name, addresses in (
            ("turn_ids", self.turn_ids),
            ("tool_call_ids", self.tool_call_ids),
        ):
            if not isinstance(addresses, frozenset) or any(
                not isinstance(address, str) or not address
                for address in addresses
            ):
                raise TypeError(
                    f"Stop target {field_name} must be a frozenset of concrete strings"
                )


class StopCleanupRegistry:
    """Application-lifetime owner for cleanup tails beyond one Stop request."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[StopOutcome]] = set()

    def retain(self, task: asyncio.Task[StopOutcome]) -> None:
        """Own and consume one detached target task until it truly terminates."""

        if not isinstance(task, asyncio.Task):
            raise TypeError("Stop cleanup registry only retains asyncio tasks")
        self._tasks.add(task)

        def consume(completed: asyncio.Task[StopOutcome]) -> None:
            self._tasks.discard(completed)
            try:
                completed.result()
            except BaseException:  # noqa: BLE001, S110 - exception consumption
                # The caller already received its typed timeout/cancellation.
                # This late outcome exists only to finish target cleanup.
                pass

        task.add_done_callback(consume)

    async def drain(self) -> None:
        """Join every retained cleanup tail before application teardown."""

        pending_cancellation: asyncio.CancelledError | None = None
        while self._tasks:
            task = next(iter(self._tasks))
            outcome = await await_owned_task(task, pending_cancellation)
            if pending_cancellation is None:
                pending_cancellation = outcome.cancellation
            # The callback normally removes completed work. Discard explicitly
            # as well so drain does not depend on callback scheduling order.
            self._tasks.discard(task)
        if pending_cancellation is not None:
            raise pending_cancellation


class CancellationAuthority:
    """Resolve Stop scopes and report every cooperative target independently."""

    def __init__(
        self,
        target_inventory: Callable[[], Iterable[CooperativeStopTarget]],
        *,
        cleanup_registry: StopCleanupRegistry,
        receipt_store: Any,
        target_timeout_seconds: float = DEFAULT_STOP_TARGET_TIMEOUT_SECONDS,
    ) -> None:
        if not callable(target_inventory):
            raise TypeError("target_inventory must be callable")
        if not isinstance(cleanup_registry, StopCleanupRegistry):
            raise TypeError("cleanup_registry must be a StopCleanupRegistry")
        if not callable(getattr(receipt_store, "load", None)) or not callable(
            getattr(receipt_store, "persist", None)
        ):
            raise TypeError("receipt_store must provide load and persist")
        self._target_inventory = target_inventory
        self._cleanup_registry = cleanup_registry
        self._receipt_store = receipt_store
        if (
            not isinstance(target_timeout_seconds, (int, float))
            or isinstance(target_timeout_seconds, bool)
            or not math.isfinite(target_timeout_seconds)
            or target_timeout_seconds <= 0
        ):
            raise ValueError("target_timeout_seconds must be positive and finite")
        self._target_timeout_seconds = float(target_timeout_seconds)

    async def stop(self, request: StopRequest) -> tuple[StopOutcome, ...]:
        request = self._validated_request(request)
        try:
            replay = await self._receipt_store.load(request)
        except StopReceiptConflict:
            return self._receipt_preflight_refusal(
                request,
                self._resolve(request),
                detail="Stop operation identity conflicts with durable evidence",
            )
        except Exception:  # noqa: BLE001 - durable evidence boundary
            return self._receipt_preflight_refusal(
                request,
                self._resolve(request),
                detail="Stop receipt storage is unavailable; cancellation not attempted",
            )
        if replay is not None:
            if not isinstance(replay, StopReceipt):
                return self._receipt_preflight_refusal(
                    request,
                    self._resolve(request),
                    detail="Stop receipt storage returned invalid evidence",
                )
            return replay.outcomes

        targets = self._resolve(request)
        owner = asyncio.create_task(
            self._claim_stop_and_persist(request, targets),
            name="cooperative-stop-operation",
        )
        outcome = await await_owned_task(owner)
        return raise_owned_outcome(outcome, operation="cooperative Stop receipt")

    async def _claim_stop_and_persist(
        self,
        request: StopRequest,
        targets: tuple[CooperativeStopTarget, ...],
    ) -> tuple[StopOutcome, ...]:
        """Own the durable claim through its effects and terminal receipt.

        Claiming and creating an effect owner cannot be two caller-owned
        awaits: cancellation in that gap would leave durable ``in progress``
        evidence with no task capable of completing it.  This task owns the
        entire claim-to-receipt transaction boundary.
        """

        claim_id: str | None = None
        claim_operation = getattr(self._receipt_store, "claim", None)
        if callable(claim_operation):
            try:
                claim = await claim_operation(request)
            except StopReceiptConflict:
                return self._receipt_preflight_refusal(
                    request,
                    targets,
                    detail="Stop operation identity conflicts with durable evidence",
                )
            except Exception:  # noqa: BLE001 - durable claim boundary
                return self._receipt_preflight_refusal(
                    request,
                    targets,
                    detail=(
                        "Stop receipt storage is unavailable; cancellation not attempted"
                    ),
                )
            if isinstance(claim, StopReceipt):
                return claim.outcomes
            if claim is None:
                return self._receipt_preflight_refusal(
                    request,
                    targets,
                    detail="An exact Stop operation is already in progress",
                )
            if not isinstance(claim, StopOperationClaim):
                return self._receipt_preflight_refusal(
                    request,
                    targets,
                    detail="Stop receipt storage returned an invalid operation claim",
                )
            claim_id = claim.claim_id

        return await self._stop_and_persist(
            request,
            targets,
            claim_id=claim_id,
        )

    async def _stop_and_persist(
        self,
        request: StopRequest,
        targets: tuple[CooperativeStopTarget, ...],
        *,
        claim_id: str | None,
    ) -> tuple[StopOutcome, ...]:
        """Own target effects through their durable receipt commit."""

        if not targets:
            if request.scope is StopScope.HOST:
                # An empty snapshot is not evidence that every host agent was
                # stopped. Represent the authority-level failure explicitly;
                # persisting an empty tuple would otherwise make an inventory
                # failure indistinguishable from a successful fan-out and the
                # endpoint could acknowledge Stop without reaching anything.
                outcomes: tuple[StopOutcome, ...] = (
                    StopOutcome(
                        scope=request.scope,
                        requested_target=None,
                        resolved_target=StopScope.HOST.value,
                        agent_id=StopScope.HOST.value,
                        disposition=StopDisposition.UNREACHABLE,
                        correlation_id=request.correlation_id,
                        detail="No cooperative Stop targets were discovered",
                    ),
                )
            else:
                outcomes = (
                    StopOutcome(
                        scope=request.scope,
                        requested_target=request.target,
                        resolved_target=request.target or StopScope.HOST.value,
                        agent_id=(
                            request.target_agent_id
                            or request.target
                            or "unresolved"
                        ),
                        disposition=StopDisposition.UNREACHABLE,
                        correlation_id=request.correlation_id,
                        detail="No cooperative Stop target resolved",
                    ),
                )
        else:
            outcomes = await self._stop_targets(request, targets)

        try:
            if claim_id is None:
                receipt = await self._receipt_store.persist(request, outcomes)
            else:
                receipt = await self._receipt_store.persist(
                    request,
                    outcomes,
                    claim_id=claim_id,
                )
            if not isinstance(receipt, StopReceipt):
                raise TypeError("Stop receipt storage returned invalid evidence")
            return receipt.outcomes
        except Exception:  # noqa: BLE001 - report only typed indeterminacy
            return tuple(
                replace(
                    outcome,
                    disposition=StopDisposition.REFUSED,
                    detail=(
                        "Cancellation may have completed, but its durable "
                        "Stop receipt could not be persisted"
                    ),
                )
                for outcome in outcomes
            )

    async def _stop_targets(
        self,
        request: StopRequest,
        targets: tuple[CooperativeStopTarget, ...],
    ) -> tuple[StopOutcome, ...]:
        async def stop_one(target: CooperativeStopTarget) -> StopOutcome:
            detail = None
            try:
                disposition = await target.cancel(request)
                if not isinstance(disposition, StopDisposition):
                    raise TypeError("Stop target returned an untyped disposition")
            except asyncio.CancelledError:
                disposition = StopDisposition.UNREACHABLE
                detail = "Cooperative Stop target was canceled"
            except Exception as error:  # noqa: BLE001 - target boundary
                disposition = StopDisposition.UNREACHABLE
                detail = f"Cooperative Stop target failed ({type(error).__name__})"
            return StopOutcome(
                scope=request.scope,
                requested_target=request.target,
                resolved_target=target.target_id,
                agent_id=target.agent_id,
                disposition=disposition,
                correlation_id=request.correlation_id,
                detail=detail,
            )

        tasks = {
            asyncio.create_task(
                stop_one(target),
                name=f"cooperative-stop:{target.target_id}",
            ): target
            for target in targets
        }
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self._target_timeout_seconds,
            )
        except BaseException:
            for task in tasks:
                task.cancel()
                self._detach_cleanup(task)
            raise

        completed = {tasks[task].target_id: task.result() for task in done}
        for task in pending:
            task.cancel()
            self._detach_cleanup(task)
        for task in pending:
            target = tasks[task]
            completed[target.target_id] = StopOutcome(
                scope=request.scope,
                requested_target=request.target,
                resolved_target=target.target_id,
                agent_id=target.agent_id,
                disposition=StopDisposition.UNREACHABLE,
                correlation_id=request.correlation_id,
                detail="Cooperative Stop target timed out",
            )
        return tuple(completed[target.target_id] for target in targets)

    @staticmethod
    def _receipt_preflight_refusal(
        request: StopRequest,
        targets: tuple[CooperativeStopTarget, ...],
        *,
        detail: str,
    ) -> tuple[StopOutcome, ...]:
        if not targets and request.scope is StopScope.HOST:
            return (
                StopOutcome(
                    scope=request.scope,
                    requested_target=None,
                    resolved_target=StopScope.HOST.value,
                    agent_id=StopScope.HOST.value,
                    disposition=StopDisposition.REFUSED,
                    correlation_id=request.correlation_id,
                    detail=detail,
                ),
            )
        if not targets:
            return (
                StopOutcome(
                    scope=request.scope,
                    requested_target=request.target,
                    resolved_target=request.target or StopScope.HOST.value,
                    agent_id=request.target_agent_id
                    or request.target
                    or "unresolved",
                    disposition=StopDisposition.REFUSED,
                    correlation_id=request.correlation_id,
                    detail=detail,
                ),
            )
        return tuple(
            StopOutcome(
                scope=request.scope,
                requested_target=request.target,
                resolved_target=target.target_id,
                agent_id=target.agent_id,
                disposition=StopDisposition.REFUSED,
                correlation_id=request.correlation_id,
                detail=detail,
            )
            for target in targets
        )

    @staticmethod
    def _validated_request(request: StopRequest) -> StopRequest:
        """Rebuild the request before inventory lookup or target side effects."""

        if not isinstance(request, StopRequest):
            raise TypeError("request must be a validated StopRequest")
        return StopRequest(
            scope=request.scope,
            actor_id=request.actor_id,
            target=request.target,
            target_agent_id=request.target_agent_id,
            reason=request.reason,
            cascade=request.cascade,
            correlation_id=request.correlation_id,
            turn_id=request.turn_id,
            span_id=request.span_id,
            trace_id=request.trace_id,
        )

    def _detach_cleanup(self, task: asyncio.Task[StopOutcome]) -> None:
        """Consume a late target outcome without extending Stop's deadline."""

        self._cleanup_registry.retain(task)

    def _resolve(self, request: StopRequest) -> tuple[CooperativeStopTarget, ...]:
        inventory = tuple(self._target_inventory())
        self._validate_inventory(inventory)
        if request.scope is StopScope.HOST:
            matches = inventory
        elif request.scope is StopScope.AGENT:
            matches = tuple(
                target
                for target in inventory
                if request.target in {target.target_id, target.agent_id}
            )
        elif request.scope is StopScope.TURN:
            matches = tuple(
                target
                for target in inventory
                if target.agent_id == request.target_agent_id
                and request.target in target.turn_ids
            )
        else:
            matches = tuple(
                target
                for target in inventory
                if target.agent_id == request.target_agent_id
                and request.target in target.tool_call_ids
            )
        return tuple(sorted(matches, key=lambda target: target.target_id))

    @staticmethod
    def _validate_inventory(
        inventory: tuple[CooperativeStopTarget, ...],
    ) -> None:
        if not all(isinstance(target, CooperativeStopTarget) for target in inventory):
            raise TypeError("target_inventory returned an untyped Stop target")
        target_ids = [target.target_id for target in inventory]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("target_inventory returned duplicate target identities")
        agent_ids = [target.agent_id for target in inventory]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("target_inventory returned duplicate agent identities")
        address_owners: dict[str, str] = {}
        for target in inventory:
            for address in {target.target_id, target.agent_id}:
                previous_owner = address_owners.setdefault(address, target.agent_id)
                if previous_owner != target.agent_id:
                    raise ValueError(
                        "target_inventory returned an ambiguous agent address"
                    )
