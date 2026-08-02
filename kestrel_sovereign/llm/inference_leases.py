"""Provider-neutral orchestration for owner-scoped inference leases.

The SDK defines the wire contract.  This module owns provider discovery,
quote selection, crash-recoverable orchestration, and the hand-off to
``LLMService``.  Infrastructure packages remain responsible for all billable
provider mutations and durable provider state.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from importlib import metadata as importlib_metadata
from typing import Any

from kestrel_sdk.llm import (
    INFERENCE_LEASE_PROVIDER_ENTRY_POINT_GROUP,
    InferenceLease,
    InferenceLeaseConstraintError,
    InferenceLeaseNotFoundError,
    InferenceLeaseProvider,
    InferenceLeaseProviderUnavailableError,
    InferenceLeaseProvisioningError,
    InferenceLeaseQuote,
    InferenceLeaseRequest,
    InferenceLeaseState,
    InferencePrivacy,
)

logger = logging.getLogger(__name__)

PersistLeaseState = Callable[[Mapping[str, Any] | None], Awaitable[None]]


class InferenceLeaseProviderDiscoveryError(InferenceLeaseProviderUnavailableError):
    """Installed provider entry points are ambiguous or invalid."""


@dataclass(frozen=True)
class InferenceLeaseRecord:
    """The non-secret state needed to resume one provider operation."""

    provider_name: str
    request: InferenceLeaseRequest
    quote: InferenceLeaseQuote
    lease: InferenceLease | None = None

    def to_persisted_dict(self) -> dict[str, Any]:
        """Return a credential-free record safe for agent metadata storage."""

        return {
            "schema_version": 1,
            "provider_name": self.provider_name,
            "request": self.request.to_public_dict(),
            "quote": self.quote.to_public_dict(),
            "lease": self.lease.to_public_dict() if self.lease else None,
        }

    @classmethod
    def from_persisted_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> InferenceLeaseRecord:
        """Rebuild a record without ever persisting a route or credentials."""

        if payload.get("schema_version") != 1:
            raise InferenceLeaseProvisioningError(
                "unsupported persisted inference lease state"
            )
        provider_name = payload.get("provider_name")
        request_payload = payload.get("request")
        quote_payload = payload.get("quote")
        lease_payload = payload.get("lease")
        if (
            not isinstance(provider_name, str)
            or not isinstance(request_payload, Mapping)
            or not isinstance(quote_payload, Mapping)
            or (lease_payload is not None and not isinstance(lease_payload, Mapping))
        ):
            raise InferenceLeaseProvisioningError(
                "persisted inference lease state is malformed"
            )

        try:
            request = _request_from_public_dict(request_payload, owner_id=owner_id)
            quote = _quote_from_public_dict(quote_payload)
            lease = (
                _lease_reference_from_public_dict(lease_payload, owner_id=owner_id)
                if lease_payload is not None
                else None
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InferenceLeaseProvisioningError(
                "persisted inference lease state is malformed"
            ) from exc

        if quote.provider_name != provider_name:
            raise InferenceLeaseProvisioningError(
                "persisted inference provider does not match its quote"
            )
        if lease is not None:
            lease.validate_for(request, quote)
        return cls(
            provider_name=provider_name,
            request=request,
            quote=quote,
            lease=lease,
        )


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TypeError("timestamp must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _request_from_public_dict(
    payload: Mapping[str, Any],
    *,
    owner_id: str,
) -> InferenceLeaseRequest:
    return InferenceLeaseRequest(
        request_id=payload["request_id"],
        owner_id=owner_id,
        model=payload["model"],
        runtime=payload["runtime"],
        max_hourly_cost_usd=Decimal(payload["max_hourly_cost_usd"]),
        max_total_cost_usd=Decimal(payload["max_total_cost_usd"]),
        privacy=InferencePrivacy(payload["privacy"]),
        capabilities=tuple(payload.get("capabilities") or ()),
        allowed_regions=tuple(payload.get("allowed_regions") or ()),
        expected_concurrency=payload["expected_concurrency"],
        expected_session_seconds=payload["expected_session_seconds"],
        idle_ttl_seconds=payload["idle_ttl_seconds"],
        ready_deadline_seconds=payload["ready_deadline_seconds"],
        requested_at=_parse_timestamp(payload["requested_at"]),
        metadata=payload.get("metadata") or {},
    )


def _quote_from_public_dict(payload: Mapping[str, Any]) -> InferenceLeaseQuote:
    return InferenceLeaseQuote(
        quote_id=payload["quote_id"],
        request_id=payload["request_id"],
        provider_name=payload["provider_name"],
        runtime=payload["runtime"],
        region=payload["region"],
        privacy=InferencePrivacy(payload["privacy"]),
        hourly_cost_usd=Decimal(payload["hourly_cost_usd"]),
        estimated_total_cost_usd=Decimal(payload["estimated_total_cost_usd"]),
        estimated_ready_seconds=payload["estimated_ready_seconds"],
        expires_at=_parse_timestamp(payload["expires_at"]),
        metadata=payload.get("metadata") or {},
    )


def _lease_reference_from_public_dict(
    payload: Mapping[str, Any],
    *,
    owner_id: str,
) -> InferenceLease:
    """Rebuild only non-ready leases; ready routes must be re-fetched."""

    state = InferenceLeaseState(payload["state"])
    if state is InferenceLeaseState.READY:
        # The public serializer intentionally omits the addressable route.  A
        # restart must ask the provider for fresh in-memory credentials before
        # activating anything.
        state = InferenceLeaseState.PENDING
    failure_payload = payload.get("failure")
    failure = None
    if failure_payload is not None:
        from kestrel_sdk.llm import InferenceLeaseFailure

        failure = InferenceLeaseFailure(
            code=failure_payload["code"],
            message=failure_payload["message"],
            retryable=failure_payload.get("retryable", False),
            metadata=failure_payload.get("metadata") or {},
        )
    if state is not InferenceLeaseState.FAILED:
        failure = None
    return InferenceLease(
        lease_id=payload["lease_id"],
        quote_id=payload["quote_id"],
        request_id=payload["request_id"],
        owner_id=owner_id,
        provider_name=payload["provider_name"],
        state=state,
        model=payload["model"],
        runtime=payload["runtime"],
        privacy=InferencePrivacy(payload["privacy"]),
        created_at=_parse_timestamp(payload["created_at"]),
        updated_at=_parse_timestamp(payload["updated_at"]),
        expires_at=_parse_timestamp(payload["expires_at"]),
        region=payload.get("region"),
        hourly_cost_usd=(
            Decimal(payload["hourly_cost_usd"])
            if payload.get("hourly_cost_usd") is not None
            else None
        ),
        estimated_total_cost_usd=(
            Decimal(payload["estimated_total_cost_usd"])
            if payload.get("estimated_total_cost_usd") is not None
            else None
        ),
        failure=failure,
        metadata=payload.get("metadata") or {},
    )


def _entry_points_for_group() -> Sequence[Any]:
    entry_points = importlib_metadata.entry_points()
    if hasattr(entry_points, "select"):
        return tuple(
            entry_points.select(group=INFERENCE_LEASE_PROVIDER_ENTRY_POINT_GROUP)
        )
    return tuple(entry_points.get(INFERENCE_LEASE_PROVIDER_ENTRY_POINT_GROUP, ()))


def discover_inference_lease_providers() -> dict[str, InferenceLeaseProvider]:
    """Discover a deterministic, validated provider map from entry points."""

    providers: dict[str, InferenceLeaseProvider] = {}
    claims: dict[str, list[str]] = {}
    loaded: list[tuple[str, str, InferenceLeaseProvider]] = []

    for entry_point in sorted(
        _entry_points_for_group(),
        key=lambda item: (item.name.lower(), str(item.value)),
    ):
        try:
            candidate = entry_point.load()
            if inspect.isclass(candidate):
                candidate = candidate()
        except Exception as exc:
            raise InferenceLeaseProviderDiscoveryError(
                f"inference provider entry point {entry_point.name!r} could not load"
            ) from exc
        if not isinstance(candidate, InferenceLeaseProvider):
            raise InferenceLeaseProviderDiscoveryError(
                f"inference provider entry point {entry_point.name!r} does not "
                "implement the SDK contract"
            )
        provider_name = candidate.provider_name.strip().lower()
        entry_name = entry_point.name.strip().lower()
        if provider_name != entry_name:
            raise InferenceLeaseProviderDiscoveryError(
                f"inference provider entry point {entry_name!r} reports a "
                "different provider name"
            )
        distribution = getattr(getattr(entry_point, "dist", None), "name", None)
        owner = str(distribution or entry_point.value)
        claims.setdefault(provider_name, []).append(owner)
        loaded.append((provider_name, owner, candidate))

    duplicates = {
        name: sorted(set(owners)) for name, owners in claims.items() if len(owners) > 1
    }
    if duplicates:
        details = "; ".join(
            f"{name}: {', '.join(owners)}"
            for name, owners in sorted(duplicates.items())
        )
        raise InferenceLeaseProviderDiscoveryError(
            f"duplicate inference provider entry points: {details}"
        )
    for provider_name, _owner, provider in loaded:
        providers[provider_name] = provider
    return providers


_ALLOWED_TRANSITIONS: dict[InferenceLeaseState, frozenset[InferenceLeaseState]] = {
    InferenceLeaseState.PENDING: frozenset(
        {
            InferenceLeaseState.PENDING,
            InferenceLeaseState.READY,
            InferenceLeaseState.FAILED,
            InferenceLeaseState.RELEASING,
            InferenceLeaseState.RELEASED,
            InferenceLeaseState.EXPIRED,
        }
    ),
    InferenceLeaseState.READY: frozenset(
        {
            InferenceLeaseState.READY,
            InferenceLeaseState.FAILED,
            InferenceLeaseState.RELEASING,
            InferenceLeaseState.RELEASED,
            InferenceLeaseState.EXPIRED,
        }
    ),
    InferenceLeaseState.RELEASING: frozenset(
        {
            InferenceLeaseState.RELEASING,
            InferenceLeaseState.RELEASED,
            InferenceLeaseState.EXPIRED,
            InferenceLeaseState.FAILED,
        }
    ),
    InferenceLeaseState.FAILED: frozenset({InferenceLeaseState.FAILED}),
    InferenceLeaseState.RELEASED: frozenset({InferenceLeaseState.RELEASED}),
    InferenceLeaseState.EXPIRED: frozenset({InferenceLeaseState.EXPIRED}),
}


class InferenceLeaseCoordinator:
    """Coordinate exactly one owner-scoped route for one ``LLMService``."""

    def __init__(
        self,
        *,
        llm_service: Any,
        owner_id: str,
        providers: Mapping[str, InferenceLeaseProvider] | None = None,
        persist_state: PersistLeaseState | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._owner_id = owner_id
        self._providers = dict(
            providers if providers is not None else discover_inference_lease_providers()
        )
        self._persist_state = persist_state
        self._record: InferenceLeaseRecord | None = None
        self._lock = asyncio.Lock()

    @property
    def current_record(self) -> InferenceLeaseRecord | None:
        return self._record

    async def _persist(self) -> None:
        if self._persist_state is None:
            raise InferenceLeaseProvisioningError(
                "durable inference lease storage is unavailable"
            )
        try:
            await self._persist_state(
                self._record.to_persisted_dict() if self._record else None
            )
        except InferenceLeaseProvisioningError:
            raise
        except Exception as exc:
            raise InferenceLeaseProvisioningError(
                "durable inference lease state could not be saved"
            ) from exc

    def _provider(self, name: str) -> InferenceLeaseProvider:
        provider = self._providers.get(name)
        if provider is None:
            raise InferenceLeaseProviderUnavailableError(
                f"inference provider {name!r} is not installed"
            )
        return provider

    @staticmethod
    def _validate_transition(
        previous: InferenceLease | None,
        current: InferenceLease,
    ) -> None:
        if previous is None:
            return
        if current.lease_id != previous.lease_id:
            raise InferenceLeaseConstraintError(
                "provider changed the inference lease identifier"
            )
        if current.state not in _ALLOWED_TRANSITIONS[previous.state]:
            raise InferenceLeaseConstraintError(
                f"invalid inference lease transition from {previous.state.value} "
                f"to {current.state.value}"
            )
        if current.updated_at < previous.updated_at:
            raise InferenceLeaseConstraintError(
                "provider returned stale inference lease state"
            )

    async def _apply_provider_lease(
        self,
        record: InferenceLeaseRecord,
        lease: InferenceLease,
    ) -> InferenceLease:
        lease.assert_owner(self._owner_id)
        lease.validate_for(record.request, record.quote)
        self._validate_transition(record.lease, lease)
        self._record = replace(record, lease=lease)
        await self._persist()

        if lease.state is InferenceLeaseState.READY:
            await self._llm_service.activate_inference_lease(
                lease,
                capabilities=record.request.capabilities,
            )
        elif lease.state in {
            InferenceLeaseState.FAILED,
            InferenceLeaseState.RELEASING,
            InferenceLeaseState.RELEASED,
            InferenceLeaseState.EXPIRED,
        }:
            await self._llm_service.deactivate_inference_lease(
                lease.lease_id,
                require_active=False,
            )
        return lease

    async def restore(self, payload: Mapping[str, Any] | None) -> InferenceLease | None:
        """Resume an interrupted acquire or reconcile a persisted lease."""

        if payload is None:
            return None
        async with self._lock:
            record = InferenceLeaseRecord.from_persisted_dict(
                payload,
                owner_id=self._owner_id,
            )
            self._record = record
            if record.lease is not None and record.lease.is_terminal:
                return record.lease
            provider = self._provider(record.provider_name)
            try:
                if record.lease is None:
                    lease = await provider.acquire(record.request, record.quote)
                else:
                    lease = await provider.status(
                        self._owner_id,
                        record.lease.lease_id,
                    )
            except Exception as exc:
                raise InferenceLeaseProvisioningError(
                    f"inference provider {record.provider_name!r} could not "
                    "reconcile persisted state"
                ) from exc
            return await self._apply_provider_lease(record, lease)

    async def _quote_provider(
        self,
        provider: InferenceLeaseProvider,
        request: InferenceLeaseRequest,
    ) -> InferenceLeaseQuote | None:
        try:
            if not provider.is_available():
                return None
            if not any(
                capability.satisfies(request) for capability in provider.capabilities()
            ):
                return None
            quote = await provider.quote(request)
            quote.validate_for(request)
            if quote.provider_name != provider.provider_name.strip().lower():
                raise InferenceLeaseConstraintError(
                    "quote provider does not match selected provider"
                )
            return quote
        except (InferenceLeaseConstraintError, InferenceLeaseProviderUnavailableError):
            return None
        # Third-party provider adapters may raise arbitrary client-library
        # exceptions. A bad quote is isolated so another provider can win.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Inference provider %s could not produce a valid quote (%s)",
                provider.provider_name,
                type(exc).__name__,
            )
            return None

    async def acquire(self, request: InferenceLeaseRequest) -> InferenceLease:
        """Select the cheapest valid quote, then acquire exactly that provider."""

        request_owner = request.owner_id
        if request_owner != self._owner_id:
            raise InferenceLeaseConstraintError(
                "inference request owner does not match this agent"
            )
        async with self._lock:
            if self._record is not None:
                previous = self._record
                if previous.lease is None:
                    if previous.request.request_id != request.request_id:
                        raise InferenceLeaseProvisioningError(
                            "an inference acquisition is already in progress"
                        )
                    provider = self._provider(previous.provider_name)
                    try:
                        lease = await provider.acquire(previous.request, previous.quote)
                    except Exception as exc:
                        raise InferenceLeaseProvisioningError(
                            f"inference provider {previous.provider_name!r} could not "
                            "resume acquisition"
                        ) from exc
                    return await self._apply_provider_lease(previous, lease)
                if not previous.lease.is_terminal:
                    raise InferenceLeaseProvisioningError(
                        "release the current inference lease before acquiring another"
                    )

            quotes = await asyncio.gather(
                *(
                    self._quote_provider(provider, request)
                    for _name, provider in sorted(self._providers.items())
                )
            )
            valid_quotes = [quote for quote in quotes if quote is not None]
            if not valid_quotes:
                raise InferenceLeaseProviderUnavailableError(
                    "no inference provider satisfies the requested privacy, "
                    "capability, region, readiness, and cost constraints"
                )
            quote = min(
                valid_quotes,
                key=lambda item: (
                    item.estimated_total_cost_usd,
                    item.estimated_ready_seconds,
                    item.hourly_cost_usd,
                    item.provider_name,
                ),
            )
            record = InferenceLeaseRecord(
                provider_name=quote.provider_name,
                request=request,
                quote=quote,
            )
            # Persist the idempotency request and selected quote before the
            # first billable mutation.  A crash can then resume acquire with
            # the exact same request identifier.
            self._record = record
            await self._persist()
            provider = self._provider(quote.provider_name)
            try:
                lease = await provider.acquire(request, quote)
            except Exception as exc:
                raise InferenceLeaseProvisioningError(
                    f"inference provider {quote.provider_name!r} could not acquire "
                    "capacity; retry with the same request_id"
                ) from exc
            return await self._apply_provider_lease(record, lease)

    def _record_for_lease(self, lease_id: str) -> InferenceLeaseRecord:
        record = self._record
        if record is None or record.lease is None or record.lease.lease_id != lease_id:
            raise InferenceLeaseNotFoundError(
                "inference lease was not found for this agent"
            )
        record.lease.assert_owner(self._owner_id)
        return record

    async def status(self, lease_id: str) -> InferenceLease:
        async with self._lock:
            record = self._record_for_lease(lease_id)
            assert record.lease is not None
            if record.lease.is_terminal:
                return record.lease
            provider = self._provider(record.provider_name)
            try:
                lease = await provider.status(self._owner_id, lease_id)
            except Exception as exc:
                raise InferenceLeaseProvisioningError(
                    f"inference provider {record.provider_name!r} could not report "
                    "lease status"
                ) from exc
            return await self._apply_provider_lease(record, lease)

    async def release(self, lease_id: str) -> InferenceLease:
        """Remove routing, drain calls, then request idempotent provider release."""

        async with self._lock:
            record = self._record_for_lease(lease_id)
            assert record.lease is not None
            if record.lease.state in {
                InferenceLeaseState.RELEASED,
                InferenceLeaseState.EXPIRED,
            }:
                return record.lease

            await self._llm_service.deactivate_inference_lease(
                lease_id,
                require_active=False,
            )
            releasing = replace(
                record.lease,
                state=InferenceLeaseState.RELEASING,
                route=None,
                failure=None,
                updated_at=datetime.now(UTC),
            )
            self._record = replace(record, lease=releasing)
            await self._persist()
            provider = self._provider(record.provider_name)
            try:
                lease = await provider.release(self._owner_id, lease_id)
            except Exception as exc:
                raise InferenceLeaseProvisioningError(
                    f"inference provider {record.provider_name!r} could not release "
                    "capacity; retry release with the same lease_id"
                ) from exc
            return await self._apply_provider_lease(self._record, lease)


__all__ = [
    "InferenceLeaseCoordinator",
    "InferenceLeaseProviderDiscoveryError",
    "InferenceLeaseRecord",
    "discover_inference_lease_providers",
]
