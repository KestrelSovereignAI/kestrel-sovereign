"""The single scope-resolution seam for cooperative Stop."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from kestrel_sovereign.agent.invocation import validate_invocation_id

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
    turn_request_ids: Mapping[str, str] = field(default_factory=dict)
    turn_request_generations: Mapping[str, int] = field(default_factory=dict)

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


class CancellationAuthority:
    """Resolve Stop scopes and report every cooperative target independently."""

    def __init__(
        self,
        target_inventory: Callable[[], Iterable[CooperativeStopTarget]],
        *,
        cleanup_registry: StopCleanupRegistry,
        target_timeout_seconds: float = DEFAULT_STOP_TARGET_TIMEOUT_SECONDS,
    ) -> None:
        if not callable(target_inventory):
            raise TypeError("target_inventory must be callable")
        if not isinstance(cleanup_registry, StopCleanupRegistry):
            raise TypeError("cleanup_registry must be a StopCleanupRegistry")
        self._target_inventory = target_inventory
        self._cleanup_registry = cleanup_registry
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
        targets = self._resolve(request)
        if not targets:
            if request.scope is StopScope.HOST:
                # HOST fan-out has one outcome per resolved agent.  An empty
                # inventory is a successful empty fan-out, not a fabricated
                # agent with an empty DID.
                return ()
            return (
                StopOutcome(
                    scope=request.scope,
                    requested_target=request.target,
                    resolved_target=(
                        request.target_agent_id
                        if request.scope in {StopScope.TURN, StopScope.TOOL_CALL}
                        else request.target
                    )
                    or StopScope.HOST.value,
                    agent_id=request.target_agent_id or request.target or "unresolved",
                    disposition=StopDisposition.UNREACHABLE,
                    correlation_id=request.correlation_id,
                    detail="No cooperative Stop target resolved",
                ),
            )

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
            except Exception as error:  # noqa: BLE001 - target boundary
                # A host Stop is an andon cord, so one broken or remote target
                # cannot prevent later targets from observing it. Preserve one
                # truthful outcome per resolved target without exposing an
                # exception message that may contain provider or request data.
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

        resolved_targets = tuple(
            (target, *self._request_for_target(request, target))
            for target in targets
        )
        tasks = {
            asyncio.create_task(
                stop_one(target, target_request, resolved_target),
                name=f"cooperative-stop:{target.target_id}",
            ): (target, resolved_target)
            for target, target_request, resolved_target in resolved_targets
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

        completed = {
            tasks[task][0].target_id: task.result()
            for task in done
        }
        for task in pending:
            task.cancel()
            # Cooperative cancellation is advisory: a target may catch
            # CancelledError and wedge during cleanup.  Stop's deadline still
            # bounds the authority call, so ownership of that cleanup tail is
            # detached with exception consumption instead of awaited here.
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
    ) -> tuple[StopRequest, str]:
        """Resolve a public turn address behind the single authority seam."""

        if (
            request.scope is not StopScope.TURN
            or request.target is None
            or not request.target_is_turn_id
        ):
            return request, target.target_id
        request_id = target.turn_request_ids.get(request.target)
        if request_id is None:
            return request, target.target_id
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
            ),
            request_id,
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
                and (
                    request.target in target.turn_ids
                    or request.target in target.turn_request_ids
                )
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
