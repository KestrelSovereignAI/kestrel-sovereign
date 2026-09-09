"""The single scope-resolution seam for cooperative Stop."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from kestrel_sovereign.agent.invocation import validate_invocation_id
from kestrel_sovereign._async_ownership import (
    await_owned_task,
    raise_owned_outcome,
)

from .receipt import StopOperationClaim, StopReceipt, StopReceiptConflict
from .types import (
    AuthoritativeStopDescendant,
    StopDisposition,
    StopOutcome,
    StopRequest,
    StopScope,
)

StopOperation = Callable[[StopRequest], Awaitable[StopDisposition]]
DescendantResolver = Callable[
    [str], Awaitable[Iterable[AuthoritativeStopDescendant]]
]
UnloadedAgentStop = Callable[[str], Awaitable[StopDisposition]]
DEFAULT_STOP_TARGET_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class CooperativeStopTarget:
    """A snapshot of one agent's cooperative cancellation addresses."""

    target_id: str
    agent_id: str
    cancel: StopOperation
    turn_ids: frozenset[str] = field(default_factory=frozenset)
    tool_call_ids: frozenset[str] = field(default_factory=frozenset)
    turn_request_ids: Mapping[str, str] = field(default_factory=dict)
    turn_request_generations: Mapping[str, int] = field(default_factory=dict)
    resolves_public_turns_durably: bool = False

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
            if not isinstance(addresses, frozenset):
                raise TypeError(
                    f"Stop target {field_name} must be a frozenset of concrete strings"
                )
            try:
                for address in addresses:
                    validate_invocation_id(address)
            except ValueError as error:
                raise TypeError(
                    f"Stop target {field_name} must contain valid opaque work addresses"
                ) from error
        if not isinstance(self.turn_request_ids, Mapping):
            raise TypeError("Stop target turn_request_ids must be a mapping")
        turn_request_ids = dict(self.turn_request_ids)
        try:
            for turn_id, request_id in turn_request_ids.items():
                validate_invocation_id(turn_id)
                validate_invocation_id(request_id)
        except ValueError as error:
            raise TypeError(
                "Stop target turn_request_ids must map valid opaque turn "
                "addresses to valid opaque request addresses"
            ) from error
        object.__setattr__(
            self,
            "turn_request_ids",
            MappingProxyType(turn_request_ids),
        )
        if not isinstance(self.turn_request_generations, Mapping):
            raise TypeError("Stop target turn_request_generations must be a mapping")
        turn_request_generations = dict(self.turn_request_generations)
        if any(
            turn_id not in turn_request_ids
            or not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation <= 0
            for turn_id, generation in turn_request_generations.items()
        ):
            raise TypeError(
                "Stop target turn generations must bind known turns to "
                "positive integers"
            )
        object.__setattr__(
            self,
            "turn_request_generations",
            MappingProxyType(turn_request_generations),
        )
        if not isinstance(self.resolves_public_turns_durably, bool):
            raise TypeError("durable public-turn resolution flag must be boolean")


@dataclass(frozen=True, slots=True)
class _ResolvedStopAddress:
    """One ordered cascade address, whether loaded locally or not."""

    target_id: str
    agent_id: str
    target: CooperativeStopTarget | None


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
        descendant_resolver: DescendantResolver | None = None,
        unloaded_agent_stop: UnloadedAgentStop | None = None,
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
        if descendant_resolver is not None and not callable(descendant_resolver):
            raise TypeError("descendant_resolver must be callable when supplied")
        self._descendant_resolver = descendant_resolver
        if unloaded_agent_stop is not None and not callable(unloaded_agent_stop):
            raise TypeError("unloaded_agent_stop must be callable when supplied")
        self._unloaded_agent_stop = unloaded_agent_stop
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
                await self._resolve(request),
                detail="Stop operation identity conflicts with durable evidence",
            )
        except Exception:  # noqa: BLE001 - durable evidence boundary
            return self._receipt_preflight_refusal(
                request,
                await self._resolve(request),
                detail="Stop receipt storage is unavailable; cancellation not attempted",
            )
        if replay is not None:
            if not isinstance(replay, StopReceipt):
                return self._receipt_preflight_refusal(
                    request,
                    await self._resolve(request),
                    detail="Stop receipt storage returned invalid evidence",
                )
            return replay.outcomes

        targets = await self._resolve(request)
        owner = asyncio.create_task(
            self._claim_stop_and_persist(request, targets),
            name="cooperative-stop-operation",
        )
        outcome = await await_owned_task(owner)
        return raise_owned_outcome(outcome, operation="cooperative Stop receipt")

    async def _claim_stop_and_persist(
        self,
        request: StopRequest,
        targets: tuple[_ResolvedStopAddress, ...],
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
        targets: tuple[_ResolvedStopAddress, ...],
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
                        resolved_target=(
                            request.target_agent_id
                            if request.scope in {StopScope.TURN, StopScope.TOOL_CALL}
                            else request.target
                        )
                        or StopScope.HOST.value,
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
        targets: tuple[_ResolvedStopAddress, ...],
    ) -> tuple[StopOutcome, ...]:
        async def stop_one(
            target: CooperativeStopTarget,
            target_request: StopRequest,
            resolved_target: str,
        ) -> StopOutcome:
            detail = None
            try:
                disposition = await target.cancel(target_request)
                if not isinstance(disposition, StopDisposition):
                    raise TypeError("Stop target returned an untyped disposition")
            except asyncio.CancelledError:
                disposition = StopDisposition.UNREACHABLE
                detail = "Cooperative Stop target was canceled"
            except BaseExceptionGroup as error:
                # A cancellation-safe target may preserve cancellation beside
                # its own cleanup failure.  That group belongs to this target,
                # not to the fan-out owner task, so isolate it just like any
                # other target failure and still durably settle every sibling.
                disposition = StopDisposition.UNREACHABLE
                detail = f"Cooperative Stop target failed ({type(error).__name__})"
            except Exception as error:  # noqa: BLE001 - target boundary
                disposition = StopDisposition.UNREACHABLE
                detail = f"Cooperative Stop target failed ({type(error).__name__})"
            return StopOutcome(
                scope=request.scope,
                requested_target=request.target,
                resolved_target=resolved_target,
                agent_id=target.agent_id,
                disposition=disposition,
                correlation_id=request.correlation_id,
                detail=detail,
            )

        live_targets = tuple(
            (
                resolved,
                *self._request_for_target(
                    request,
                    resolved.target,
                    resolved_target=resolved.target_id,
                ),
            )
            for resolved in targets
            if resolved.target is not None
        )
        tasks = {
            asyncio.create_task(
                stop_one(resolved.target, target_request, resolved_target),
                name=f"cooperative-stop:{resolved.target_id}",
            ): (resolved, resolved_target)
            for resolved, target_request, resolved_target in live_targets
        }
        done: set[asyncio.Task[StopOutcome]] = set()
        pending: set[asyncio.Task[StopOutcome]] = set()
        if tasks:
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

        completed = {
            resolved.target_id: StopOutcome(
                scope=request.scope,
                requested_target=request.target,
                resolved_target=resolved.target_id,
                agent_id=resolved.agent_id,
                disposition=StopDisposition.UNREACHABLE,
                correlation_id=request.correlation_id,
                detail="Authoritative descendant has no cooperative Stop target",
            )
            for resolved in targets
            if resolved.target is None
        }
        completed.update({
            tasks[task][0].target_id: task.result()
            for task in done
        })
        for task in pending:
            task.cancel()
            self._detach_cleanup(task)
        for task in pending:
            target, resolved_target = tasks[task]
            completed[target.target_id] = StopOutcome(
                scope=request.scope,
                requested_target=request.target,
                resolved_target=resolved_target,
                agent_id=target.agent_id,
                disposition=StopDisposition.UNREACHABLE,
                correlation_id=request.correlation_id,
                detail="Cooperative Stop target timed out",
            )
        return tuple(completed[target.target_id] for target in targets)

    @staticmethod
    def _request_for_target(
        request: StopRequest,
        target: CooperativeStopTarget,
        *,
        resolved_target: str,
    ) -> tuple[StopRequest, str]:
        """Resolve a public turn address behind the single authority seam."""

        if (
            request.scope is not StopScope.TURN
            or request.target is None
            or not request.target_is_turn_id
        ):
            return request, resolved_target
        request_id = target.turn_request_ids.get(request.target)
        if request_id is None:
            return request, resolved_target
        return (
            StopRequest(
                scope=request.scope,
                actor_id=request.actor_id,
                target=request_id,
                target_agent_id=request.target_agent_id,
                reason=request.reason,
                cascade=request.cascade,
                correlation_id=request.correlation_id,
                target_is_turn_id=False,
                request_generation=target.turn_request_generations.get(
                    request.target
                ),
                turn_id=request.target,
            ),
            request_id,
        )

    @staticmethod
    def _receipt_preflight_refusal(
        request: StopRequest,
        targets: tuple[_ResolvedStopAddress, ...],
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
            target_is_turn_id=request.target_is_turn_id,
            request_generation=request.request_generation,
            turn_id=request.turn_id,
            span_id=request.span_id,
            trace_id=request.trace_id,
        )

    def _detach_cleanup(self, task: asyncio.Task[StopOutcome]) -> None:
        """Consume a late target outcome without extending Stop's deadline."""

        self._cleanup_registry.retain(task)

    async def _resolve(
        self,
        request: StopRequest,
    ) -> tuple[_ResolvedStopAddress, ...]:
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
                and (
                    (
                        request.target_is_turn_id
                        and (
                            target.resolves_public_turns_durably
                            or request.target in target.turn_request_ids
                        )
                    )
                    or (
                        not request.target_is_turn_id
                        and request.target in target.turn_ids
                    )
                )
            )
        else:
            matches = tuple(
                target
                for target in inventory
                if target.agent_id == request.target_agent_id
                and request.target in target.tool_call_ids
            )
        ordered = tuple(sorted(matches, key=lambda target: target.target_id))
        resolved = [
            _ResolvedStopAddress(target.target_id, target.agent_id, target)
            for target in ordered
        ]
        if (
            request.scope is not StopScope.AGENT
            or not request.cascade
            or self._descendant_resolver is None
            or not ordered
        ):
            return tuple(resolved)

        root = ordered[0]
        descendants = await self._descendant_resolver(root.agent_id)
        if isinstance(descendants, (str, bytes)):
            raise TypeError("descendant_resolver returned a scalar address")
        by_agent_id = {target.agent_id: target for target in inventory}

        seen_agent_ids = {root.agent_id}
        seen_names = {root.target_id.casefold()}
        for descendant in descendants:
            if not isinstance(descendant, AuthoritativeStopDescendant):
                raise TypeError(
                    "descendant_resolver returned an untyped descendant"
                )
            canonical_name = descendant.routing_name.casefold()
            if canonical_name in seen_names:
                raise ValueError(
                    "descendant_resolver returned a repeated descendant"
                )
            seen_names.add(canonical_name)
            if descendant.agent_id in seen_agent_ids:
                raise ValueError(
                    "descendant_resolver returned a cycle or duplicate agent"
                )
            seen_agent_ids.add(descendant.agent_id)
            target = by_agent_id.get(descendant.agent_id)
            if target is None and self._unloaded_agent_stop is not None:
                async def stop_unloaded(
                    _request: StopRequest,
                    *,
                    agent_id: str = descendant.agent_id,
                ) -> StopDisposition:
                    return await self._unloaded_agent_stop(agent_id)

                target = CooperativeStopTarget(
                    target_id=descendant.agent_id,
                    agent_id=descendant.agent_id,
                    cancel=stop_unloaded,
                )
            resolved.append(
                _ResolvedStopAddress(
                    # A routing name never becomes a Stop address. Bind the
                    # outcome and every local/remote operation to the signed DID.
                    target_id=descendant.agent_id,
                    agent_id=descendant.agent_id,
                    target=target,
                )
            )
        return tuple(resolved)

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
