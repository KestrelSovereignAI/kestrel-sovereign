"""Owner-scoped tools for private, on-demand inference capacity."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import uuid4

from kestrel_sdk.llm import (
    InferenceLeaseError,
    InferenceLeaseRequest,
    InferencePrivacy,
)
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.llm.inference_leases import (
    InferenceLeaseCoordinator,
    discover_inference_lease_providers,
)

logger = logging.getLogger(__name__)

_STATE_KEY = "private_inference_lease_v1"
_DEFAULT_CAPABILITIES = ("chat", "streaming", "tools")


class InferenceLeaseFeature(Feature):
    """Request and manage one private inference route for this agent."""

    def __init__(self, agent: Any):
        super().__init__(agent)
        self._db: Any | None = None
        self._owner_id = ""
        self._coordinator: InferenceLeaseCoordinator | None = None
        self._reconciliation_error: str | None = None

    @property
    def tool_description(self) -> str:
        return (
            "Acquire bounded private inference capacity from installed provider "
            "packages, inspect readiness and cost without exposing its endpoint, "
            "and release it when finished. Model discovery and selection remain "
            "in ModelAgent."
        )

    async def initialize(self) -> None:
        llm_service = getattr(self.agent, "llm_service", None)
        owner_id = getattr(self.agent, "agent_id", None)
        if llm_service is None or not owner_id:
            raise RuntimeError(
                "InferenceLeaseFeature requires an agent-scoped LLMService and ID"
            )
        self._owner_id = str(owner_id)
        self._db = resolve_feature_database(self.agent)
        try:
            providers = discover_inference_lease_providers()
        except InferenceLeaseError as exc:
            # Provider packages are an optional extension boundary. Keep the
            # agent available when one installed entry point is malformed,
            # while failing every lease operation clearly and without falling
            # back to another cloud route.
            providers = {}
            self._reconciliation_error = str(exc)
            logger.error("Private inference provider discovery failed: %s", exc)
        self._coordinator = InferenceLeaseCoordinator(
            llm_service=llm_service,
            owner_id=self._owner_id,
            providers=providers,
            persist_state=self._persist_state if self._db is not None else None,
        )
        payload = await self._load_state()
        if payload is not None:
            try:
                await self._coordinator.restore(payload)
            except InferenceLeaseError as exc:
                self._reconciliation_error = str(exc)
                logger.error("Private inference lease reconciliation failed: %s", exc)

    def _require_coordinator(self) -> InferenceLeaseCoordinator:
        if self._coordinator is None:
            raise RuntimeError("InferenceLeaseFeature is not initialized")
        return self._coordinator

    def _assert_cloud_allowed(self) -> None:
        privacy_agent = getattr(self.agent, "privacy_agent", None)
        can_use_cloud = getattr(privacy_agent, "can_use_cloud", None)
        if not callable(can_use_cloud) or not can_use_cloud():
            raise InferenceLeaseError(
                "the active privacy policy does not permit cloud inference"
            )

    async def _load_state(self) -> dict[str, Any] | None:
        if self._db is None:
            return None
        rows = await self._db.fetchall(
            "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
            (self._owner_id, _STATE_KEY),
        )
        if not rows:
            return None
        try:
            payload = json.loads(rows[0][0])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Persisted private inference state is invalid") from exc
        if not isinstance(payload, dict):
            raise TypeError("Persisted private inference state is invalid")
        return payload

    async def _persist_state(self, payload: Any | None) -> None:
        if self._db is None:
            raise RuntimeError("Durable private inference storage is unavailable")
        if payload is None:
            await self._db.execute(
                "DELETE FROM agent_metadata WHERE agent_id = ? AND key = ?",
                (self._owner_id, _STATE_KEY),
            )
            return
        value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        await self._db.execute(
            """INSERT OR REPLACE INTO agent_metadata
               (agent_id, key, value, updated_at) VALUES (?, ?, ?, ?)""",
            (self._owner_id, _STATE_KEY, value, datetime.now(UTC)),
        )

    @tool(
        name="inference_lease_acquire",
        description=(
            "Acquire bounded, private on-demand inference capacity. This can "
            "start a billable GPU and therefore requires explicit approval. "
            "Returns only sanitized lifecycle, readiness, timing, and cost data; "
            "the endpoint and credentials remain host-only."
        ),
        category=ToolCategory.MODEL_MANAGEMENT,
    )
    async def acquire(
        self,
        model: str,
        max_hourly_cost_usd: str,
        max_total_cost_usd: str,
        runtime: str = "ollama",
        privacy: str = "authenticated_endpoint",
        expected_session_seconds: int = 900,
        idle_ttl_seconds: int = 300,
        ready_deadline_seconds: int = 900,
        expected_concurrency: int = 1,
        allowed_regions: list[str] | None = None,
        capabilities: list[str] | None = None,
        request_id: str | None = None,
    ) -> ToolResult:
        """Quote and acquire the cheapest satisfying private inference lease.

        Args:
            model: Exact model identifier the provider must serve.
            max_hourly_cost_usd: Hard hourly price ceiling as a decimal string.
            max_total_cost_usd: Hard session price ceiling as a decimal string.
            runtime: Serving runtime, normally ``ollama``.
            privacy: Maximum exposure: private_network, authenticated_endpoint,
                or public_endpoint.
            expected_session_seconds: Expected total lease duration.
            idle_ttl_seconds: Release threshold after inactivity.
            ready_deadline_seconds: Maximum acceptable cold-start time.
            expected_concurrency: Required simultaneous request capacity.
            allowed_regions: Optional allowed provider regions.
            capabilities: Required route abilities. Defaults to chat, streaming,
                and tools for a full agent route.
            request_id: Optional durable idempotency key. Reuse it after an
                interrupted acquire; otherwise one is generated before quoting.
        """

        try:
            self._assert_cloud_allowed()
            if self._db is None:
                return ToolResult.failed(
                    "Durable storage is required before acquiring billable capacity"
                )
            hourly = Decimal(max_hourly_cost_usd)
            total = Decimal(max_total_cost_usd)
            privacy_mode = InferencePrivacy(privacy)
            request = InferenceLeaseRequest(
                request_id=request_id or f"lease-{uuid4().hex}",
                owner_id=self._owner_id,
                model=model,
                runtime=runtime,
                max_hourly_cost_usd=hourly,
                max_total_cost_usd=total,
                privacy=privacy_mode,
                capabilities=tuple(capabilities or _DEFAULT_CAPABILITIES),
                allowed_regions=tuple(allowed_regions or ()),
                expected_concurrency=expected_concurrency,
                expected_session_seconds=expected_session_seconds,
                idle_ttl_seconds=idle_ttl_seconds,
                ready_deadline_seconds=ready_deadline_seconds,
            )
            lease = await self._require_coordinator().acquire(request)
        except (InvalidOperation, TypeError, ValueError) as exc:
            return ToolResult.failed(f"Invalid inference lease request: {exc}")
        except InferenceLeaseError as exc:
            return ToolResult.failed(str(exc))
        self._reconciliation_error = None
        return ToolResult.ok(
            confirmation=(f"Inference lease {lease.lease_id!r} is {lease.state.value}"),
            data={"lease": lease.to_public_dict()},
        )

    @tool(
        name="inference_lease_status",
        description=(
            "Poll this agent's private inference lease. A ready result means the "
            "host has activated the route; no endpoint or credential is exposed."
        ),
        category=ToolCategory.MODEL_MANAGEMENT,
    )
    async def status(self, lease_id: str) -> ToolResult:
        """Reconcile and return one owner-scoped lease.

        Args:
            lease_id: Lease identifier returned by inference_lease_acquire.
        """

        try:
            lease = await self._require_coordinator().status(lease_id)
        except InferenceLeaseError as exc:
            return ToolResult.failed(str(exc))
        self._reconciliation_error = None
        return ToolResult.ok(
            confirmation=(f"Inference lease {lease.lease_id!r} is {lease.state.value}"),
            data={"lease": lease.to_public_dict()},
        )

    @tool(
        name="inference_lease_release",
        description=(
            "Stop routing to this agent's private inference lease, drain active "
            "calls, and release its provider capacity to stop billing."
        ),
        category=ToolCategory.MODEL_MANAGEMENT,
    )
    async def release(self, lease_id: str) -> ToolResult:
        """Idempotently release one owner-scoped lease.

        Args:
            lease_id: Lease identifier returned by inference_lease_acquire.
        """

        try:
            lease = await self._require_coordinator().release(lease_id)
        except InferenceLeaseError as exc:
            return ToolResult.failed(str(exc))
        self._reconciliation_error = None
        return ToolResult.ok(
            confirmation=(f"Inference lease {lease.lease_id!r} is {lease.state.value}"),
            data={"lease": lease.to_public_dict()},
        )


__all__ = ["InferenceLeaseFeature"]
