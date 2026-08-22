"""Readiness-gated private inference routing for :class:`LLMService`.

Provider packages own infrastructure.  Core accepts only a validated SDK
lease, keeps its route credentials in memory, and drains in-flight calls before
the provider is allowed to release capacity.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import openai
from kestrel_sdk.llm import (
    InferenceLease,
    InferenceLeaseState,
    InferencePrivacy,
)
from kestrel_sdk.llm.types import BackendType

from kestrel_sovereign.kestrel_config.constants import (
    CLIENT_CLOSE_TIMEOUT,
    HTTP_TIMEOUT_MEDIUM,
)

logger = logging.getLogger(__name__)

InferenceLeaseTouch = Callable[[str], Awaitable[InferenceLease]]

# The OpenAI client requires a non-empty value even for deliberately
# unauthenticated private-network endpoints. A provider-supplied Authorization
# header in ``default_headers`` overrides the SDK-generated bearer value.
_OPENAI_UNAUTHENTICATED_SENTINEL = "kestrel-private-route"
_MANAGED_REMOTE_FAILURE_MESSAGE = (
    "private inference route failed; no cloud fallback was attempted"
)


@dataclass(frozen=True)
class RemoteRouteSnapshot:
    """One in-flight reference to the active route."""

    lease_id: str
    model: str
    client: openai.AsyncOpenAI
    adapter: Any
    capabilities: frozenset[str]


class RemoteBackendMixin:
    """Mixin implementing the single remote-route state in ``LLMService``."""

    _managed_remote_failure_message = _MANAGED_REMOTE_FAILURE_MESSAGE

    @staticmethod
    async def _close_remote_client(
        client: openai.AsyncOpenAI,
        *,
        lease_id: str,
    ) -> None:
        """Close one route client without surfacing provider secret details."""

        try:
            await asyncio.wait_for(
                client.close(),
                timeout=CLIENT_CLOSE_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "Timed out closing private inference client for %s",
                lease_id,
            )
        # Client implementations may raise transport-specific exceptions.
        # The route has already been detached, so cleanup remains best-effort.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Private inference client close failed for %s (%s)",
                lease_id,
                type(exc).__name__,
            )

    def switch_backend(self, backend: BackendType, config: Any = None) -> None:
        """Switch ordinary backends; direct remote configuration is retired."""

        from .service import LLMServiceError

        if backend is BackendType.REMOTE_GPU:
            raise LLMServiceError(
                "Direct remote GPU configuration is retired; acquire a validated "
                "inference lease through an installed provider"
            )
        if config is not None:
            raise LLMServiceError("backend configuration is not accepted here")
        if self._remote_lease is not None:
            raise LLMServiceError(
                "release the active inference lease before switching backends"
            )
        self._backend = backend
        logger.info("LLMService switched to %s backend", backend.value)

    async def activate_inference_lease(
        self,
        lease: InferenceLease,
        *,
        capabilities: Sequence[str] = (),
        touch_lease: InferenceLeaseTouch,
    ) -> None:
        """Atomically activate a ready, OpenAI-compatible private route."""

        from .service import LLMServiceError

        if lease.state is not InferenceLeaseState.READY or lease.route is None:
            raise LLMServiceError("only a ready inference lease can be activated")
        if lease.route.protocol != "openai":
            raise LLMServiceError(
                f"unsupported inference route protocol {lease.route.protocol!r}"
            )
        if (
            lease.privacy is InferencePrivacy.AUTHENTICATED_ENDPOINT
            and lease.route.api_key is None
            and not lease.route.secret_headers
        ):
            raise LLMServiceError(
                "authenticated inference route did not provide credentials"
            )

        endpoint = lease.route.endpoint.get_secret_value().rstrip("/")
        api_key = (
            lease.route.api_key.get_secret_value()
            if lease.route.api_key is not None
            else _OPENAI_UNAUTHENTICATED_SENTINEL
        )
        headers = {
            name: value.get_secret_value()
            for name, value in lease.route.secret_headers.items()
        }
        if len({name.casefold() for name in headers}) != len(headers):
            raise LLMServiceError(
                "inference route contains duplicate case-insensitive headers"
            )

        # A READY status poll normally returns the same host-only route with a
        # fresher expiry/telemetry snapshot. Refresh that metadata in place so
        # routine polling never drains live calls or rebuilds the OpenAI client.
        # Signed URL or credential rotation still takes the drain-and-swap path
        # below because its route material differs.
        route_material = (
            lease.route.endpoint.get_secret_value(),
            (
                lease.route.api_key.get_secret_value()
                if lease.route.api_key is not None
                else None
            ),
            tuple(
                sorted(
                    (name, value.get_secret_value())
                    for name, value in lease.route.secret_headers.items()
                )
            ),
        )
        async with self._remote_route_condition:
            if self._remote_route_closed:
                # ``close()`` has already drained this route and will not run
                # again. Building a client here would leak it and would restore
                # a route the host has finished tearing down.
                raise LLMServiceError(
                    "LLM service is shutting down; the private inference route "
                    "cannot be activated"
                )
            active = self._remote_lease
            if active is not None and active.lease_id == lease.lease_id:
                active_route = active.route
                active_material = (
                    (
                        active_route.endpoint.get_secret_value(),
                        (
                            active_route.api_key.get_secret_value()
                            if active_route.api_key is not None
                            else None
                        ),
                        tuple(
                            sorted(
                                (name, value.get_secret_value())
                                for name, value in active_route.secret_headers.items()
                            )
                        ),
                    )
                    if active_route is not None
                    else None
                )
                if (
                    active_material == route_material
                    and self._remote_accepting
                    and self._remote_client is not None
                    and self._backend is BackendType.REMOTE_GPU
                ):
                    self._remote_lease = lease
                    self._remote_capabilities = frozenset(capabilities)
                    self._remote_touch_lease = touch_lease
                    return

        client = openai.AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
            default_headers=headers or None,
            timeout=HTTP_TIMEOUT_MEDIUM,
            max_retries=0,
        )

        old_client: openai.AsyncOpenAI | None = None
        activation_error: str | None = None
        async with self._remote_route_condition:
            if self._remote_route_closed:
                # ``close()`` won the race while this client was being built.
                # Fall through to the error path so the new client is closed
                # rather than leaked.
                activation_error = (
                    "LLM service is shutting down; the private inference route "
                    "cannot be activated"
                )
            elif self._remote_lease is not None:
                if self._remote_lease.lease_id != lease.lease_id:
                    activation_error = (
                        "another inference lease is already active; release it first"
                    )
                else:
                    # READY reconciliation can rotate signed URLs or auth
                    # material. Drain snapshots pinned to the prior client,
                    # then replace it atomically with the refreshed route.
                    self._remote_accepting = False
                    try:
                        async with asyncio.timeout(HTTP_TIMEOUT_MEDIUM):
                            while self._remote_inflight:
                                await self._remote_route_condition.wait()
                    except TimeoutError:
                        self._remote_accepting = True
                        activation_error = (
                            "timed out refreshing the active private inference route"
                        )
                    else:
                        old_client = self._remote_client
            if activation_error is None:
                self._remote_lease = lease
                self._remote_client = client
                self._remote_capabilities = frozenset(capabilities)
                self._remote_touch_lease = touch_lease
                self._remote_accepting = True
                self._backend = BackendType.REMOTE_GPU
                self._last_remote_error = None

        if activation_error is not None:
            await self._close_remote_client(client, lease_id=lease.lease_id)
            raise LLMServiceError(activation_error)
        if old_client is not None:
            await self._close_remote_client(old_client, lease_id=lease.lease_id)
        logger.info(
            "Activated private inference lease %s from provider %s",
            lease.lease_id,
            lease.provider_name,
        )

    async def deactivate_inference_lease(
        self,
        lease_id: str,
        *,
        require_active: bool = True,
    ) -> None:
        """Stop new calls, drain existing calls, then forget route secrets."""

        from .service import LLMServiceError

        client: openai.AsyncOpenAI | None = None
        async with self._remote_route_condition:
            active = self._remote_lease
            if active is None:
                if require_active:
                    raise LLMServiceError("inference lease is not active")
                return
            if active.lease_id != lease_id:
                if require_active:
                    raise LLMServiceError(
                        "another inference lease owns the active route"
                    )
                return

            self._remote_accepting = False
            self._backend = self._default_backend
            try:
                async with asyncio.timeout(HTTP_TIMEOUT_MEDIUM):
                    while self._remote_inflight:
                        await self._remote_route_condition.wait()
            except TimeoutError as exc:
                raise LLMServiceError(
                    "timed out draining the active private inference route; "
                    "provider release was not attempted"
                ) from exc

            client = self._remote_client
            self._remote_client = None
            self._remote_lease = None
            self._remote_capabilities = frozenset()
            self._remote_touch_lease = None
            self._remote_route_epoch += 1

        if client is not None:
            await self._close_remote_client(client, lease_id=lease_id)
        logger.info("Deactivated private inference lease %s", lease_id)

    @asynccontextmanager
    async def _remote_route_attempt(
        self,
        *,
        force_local_only: bool,
        model_override: str | None,
        required_capabilities: Sequence[str] = (),
    ) -> AsyncIterator[RemoteRouteSnapshot | None]:
        """Pin an active route for one call so release cannot race it."""

        from .service import LLMServiceError

        snapshot: RemoteRouteSnapshot | None = None
        selected_lease_id: str | None = None
        touch_lease: InferenceLeaseTouch | None = None
        selected_route_epoch: int | None = None
        async with self._remote_route_condition:
            lease = self._remote_lease
            client = self._remote_client
            route_selected = (
                lease is not None
                and not force_local_only
                and self._remote_first_allowed(model_override)
            )
            if route_selected:
                if (
                    self._backend is not BackendType.REMOTE_GPU
                    or not self._remote_accepting
                    or client is None
                ):
                    raise LLMServiceError(
                        "private inference route is draining or unavailable; no "
                        "cloud fallback was attempted"
                    )
                missing = set(required_capabilities).difference(
                    self._remote_capabilities
                )
                if missing:
                    names = ", ".join(sorted(missing))
                    raise LLMServiceError(
                        f"private inference lease lacks required capabilities: {names}"
                    )
                if datetime.now(UTC) >= lease.expires_at:
                    self._remote_accepting = False
                    raise LLMServiceError(
                        "private inference lease expired; reconcile or release it "
                        "before another LLM request"
                    )
                touch_lease = self._remote_touch_lease
                if touch_lease is None:
                    raise LLMServiceError(
                        "private inference route has no idle-deadline renewal "
                        "handler; no cloud fallback was attempted"
                    )
                selected_lease_id = lease.lease_id
                selected_route_epoch = self._remote_route_epoch

        if selected_lease_id is not None:
            assert touch_lease is not None
            assert selected_route_epoch is not None
            touched: InferenceLease | None = None
            try:
                touched = await touch_lease(selected_lease_id)
            except Exception as exc:  # noqa: BLE001 - provider boundary
                self._raise_managed_remote_failure(exc)
            if (
                not isinstance(touched, InferenceLease)
                or touched.lease_id != selected_lease_id
                or touched.state is not InferenceLeaseState.READY
            ):
                raise LLMServiceError(
                    "private inference lease is no longer ready; no cloud "
                    "fallback was attempted"
                )

            # Touch runs outside the route condition because it may refresh
            # credentials through ``activate_inference_lease``. Re-check every
            # invariant before pinning: an owner release may have won the race
            # after touch completed, in which case this call fails closed.
            async with self._remote_route_condition:
                lease = self._remote_lease
                client = self._remote_client
                if self._remote_route_epoch != selected_route_epoch:
                    # The route was torn down while this call was renewing.
                    # Comparing observable state is not enough: ``touch`` ends
                    # in ``activate_inference_lease`` for a READY lease, which
                    # rebuilds a route that looks identical to the one that was
                    # just deactivated - including after ``close()`` has already
                    # drained and returned. Refuse rather than resurrect it.
                    raise LLMServiceError(
                        "private inference route was released during renewal; no "
                        "cloud fallback was attempted"
                    )
                if (
                    lease is None
                    or lease.lease_id != selected_lease_id
                    or self._backend is not BackendType.REMOTE_GPU
                    or not self._remote_accepting
                    or client is None
                ):
                    raise LLMServiceError(
                        "private inference route is draining or unavailable; no "
                        "cloud fallback was attempted"
                    )
                if datetime.now(UTC) >= lease.expires_at:
                    self._remote_accepting = False
                    raise LLMServiceError(
                        "private inference lease expired during renewal; no cloud "
                        "fallback was attempted"
                    )
                self._remote_inflight += 1
                snapshot = RemoteRouteSnapshot(
                    lease_id=lease.lease_id,
                    model=lease.model,
                    client=client,
                    adapter=self._remote_adapter,
                    capabilities=self._remote_capabilities,
                )
        try:
            yield snapshot
        finally:
            if snapshot is not None:
                async with self._remote_route_condition:
                    self._remote_inflight -= 1
                    if self._remote_inflight == 0:
                        self._remote_route_condition.notify_all()

    def _raise_managed_remote_failure(self, exc: BaseException) -> None:
        """Record a safe category and fail closed without provider fallback."""

        from .service import LLMServiceError

        self._last_remote_error = type(exc).__name__
        raise LLMServiceError(self._managed_remote_failure_message) from exc

    def get_backend_status(self) -> dict[str, Any]:
        """Return credential- and endpoint-free route status."""

        lease = self._remote_lease
        return {
            "current_backend": self._backend.value,
            "default_backend": self._default_backend.value,
            "remote_active": bool(lease and self._remote_accepting),
            "remote_metadata": lease.to_public_dict() if lease else None,
            "last_remote_error": self._last_remote_error,
        }


__all__ = [
    "BackendType",
    "InferenceLeaseTouch",
    "RemoteBackendMixin",
    "RemoteRouteSnapshot",
]
