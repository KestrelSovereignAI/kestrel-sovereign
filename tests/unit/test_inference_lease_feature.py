"""Agent feature and policy tests for private inference leases."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from kestrel_sdk.llm import (
    InferenceLease,
    InferenceLeaseState,
    InferencePrivacy,
    InferenceRoute,
)
from kestrel_sdk.tools.result import ToolResultStatus
from pydantic import SecretStr

from kestrel_sovereign.features import discover_feature_modules
from kestrel_sovereign.features.inference_lease.feature import (
    InferenceLeaseFeature,
)
from kestrel_sovereign.features.security.feature import (
    default_permission_for_feature,
    default_permission_for_tool,
)
from kestrel_sovereign.features.security.permissions import PermissionLevel
from kestrel_sovereign.llm.inference_leases import (
    InferenceLeaseProviderDiscoveryError,
)


class _Database:
    def __init__(self):
        self.values: dict[tuple[str, str], str] = {}

    async def fetchall(self, _query, params):
        value = self.values.get(tuple(params))
        return [] if value is None else [(value,)]

    async def execute(self, query, params):
        if query.lstrip().startswith("DELETE"):
            self.values.pop(tuple(params), None)
            return
        owner_id, key, value, _updated_at = params
        self.values[(owner_id, key)] = value


def _agent(*, db=None, cloud_allowed=True):
    storage = SimpleNamespace(database=db) if db is not None else object()
    return SimpleNamespace(
        agent_id="agent-1",
        llm_service=AsyncMock(),
        privacy_agent=SimpleNamespace(can_use_cloud=lambda: cloud_allowed),
        storage=storage,
    )


def _ready_lease() -> InferenceLease:
    now = datetime.now(UTC)
    return InferenceLease(
        lease_id="lease-1",
        quote_id="quote-1",
        request_id="request-1",
        owner_id="agent-1",
        provider_name="runpod",
        state=InferenceLeaseState.READY,
        model="qwen3:8b",
        runtime="ollama",
        privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(minutes=10),
        hourly_cost_usd=Decimal("0.50"),
        estimated_total_cost_usd=Decimal("0.20"),
        route=InferenceRoute(
            endpoint=SecretStr("https://private.example.test/v1"),
            model="qwen3:8b",
            api_key=SecretStr("route-secret"),
        ),
    )


def test_feature_is_discoverable_and_billable_tool_is_always_ask():
    assert (
        "kestrel_sovereign.features.inference_lease.feature"
        in discover_feature_modules()
    )
    assert (
        default_permission_for_feature("InferenceLeaseFeature") is PermissionLevel.ASK
    )
    assert (
        default_permission_for_tool(
            "InferenceLeaseFeature",
            "inference_lease_acquire",
        )
        is PermissionLevel.ALWAYS_ASK
    )
    assert (
        default_permission_for_tool(
            "InferenceLeaseFeature",
            "inference_lease_status",
        )
        is PermissionLevel.ALLOW
    )
    assert (
        default_permission_for_tool(
            "InferenceLeaseFeature",
            "inference_lease_release",
        )
        is PermissionLevel.ALLOW
    )


@pytest.mark.asyncio
async def test_acquire_refuses_cloud_when_privacy_policy_disallows_it():
    feature = InferenceLeaseFeature(_agent(cloud_allowed=False))
    feature._owner_id = "agent-1"
    feature._coordinator = AsyncMock()

    result = await feature.acquire("qwen3:8b", "1.00", "0.50")

    assert result.status is ToolResultStatus.ERROR
    assert "privacy policy" in result.error
    feature._coordinator.acquire.assert_not_awaited()


@pytest.mark.asyncio
async def test_acquire_refuses_billable_mutation_without_durable_storage():
    feature = InferenceLeaseFeature(_agent())
    feature._owner_id = "agent-1"
    feature._coordinator = AsyncMock()

    result = await feature.acquire("qwen3:8b", "1.00", "0.50")

    assert result.status is ToolResultStatus.ERROR
    assert "Durable storage" in result.error
    feature._coordinator.acquire.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_result_omits_endpoint_and_credentials():
    feature = InferenceLeaseFeature(_agent(db=_Database()))
    feature._owner_id = "agent-1"
    feature._db = _Database()
    feature._coordinator = AsyncMock()
    feature._coordinator.acquire.return_value = _ready_lease()

    result = await feature.acquire("qwen3:8b", "1.00", "0.50")

    assert result.status is ToolResultStatus.OK
    assert "private.example" not in repr(result.to_dict())
    assert "route-secret" not in repr(result.to_dict())
    request = feature._coordinator.acquire.await_args.args[0]
    assert request.owner_id == "agent-1"
    assert request.capabilities == ("chat", "streaming", "tools")


@pytest.mark.asyncio
async def test_persisted_state_is_scoped_by_owner_and_credential_free():
    db = _Database()
    first = InferenceLeaseFeature(_agent(db=db))
    first._owner_id = "agent-1"
    first._db = db
    payload = {
        "schema_version": 1,
        "provider_name": "runpod",
        "lease": _ready_lease().to_public_dict(),
    }

    await first._persist_state(payload)

    assert await first._load_state() == payload
    raw = db.values[("agent-1", "private_inference_lease_v1")]
    assert "private.example" not in raw
    assert "route-secret" not in raw
    second = InferenceLeaseFeature(_agent(db=db))
    second._owner_id = "agent-2"
    second._db = db
    assert await second._load_state() is None
    assert json.loads(raw)["provider_name"] == "runpod"


@pytest.mark.asyncio
async def test_initialize_reconciles_persisted_state(monkeypatch):
    db = _Database()
    payload = {"schema_version": 1, "provider_name": "runpod"}
    db.values[("agent-1", "private_inference_lease_v1")] = json.dumps(payload)
    coordinator = AsyncMock()
    coordinator_type = Mock(return_value=coordinator)
    monkeypatch.setattr(
        "kestrel_sovereign.features.inference_lease.feature."
        "discover_inference_lease_providers",
        lambda: {"runpod": object()},
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.inference_lease.feature.InferenceLeaseCoordinator",
        coordinator_type,
    )
    feature = InferenceLeaseFeature(_agent(db=db))

    await feature.initialize()

    coordinator.restore.assert_awaited_once_with(payload)
    assert coordinator_type.call_args.kwargs["owner_id"] == "agent-1"
    assert callable(coordinator_type.call_args.kwargs["persist_state"])


@pytest.mark.asyncio
async def test_initialize_isolates_broken_provider_discovery(monkeypatch):
    def fail_discovery():
        raise InferenceLeaseProviderDiscoveryError("duplicate provider claim")

    monkeypatch.setattr(
        "kestrel_sovereign.features.inference_lease.feature."
        "discover_inference_lease_providers",
        fail_discovery,
    )
    feature = InferenceLeaseFeature(_agent(db=_Database()))

    await feature.initialize()

    assert feature._coordinator is not None
    assert feature._coordinator._providers == {}
    assert feature._reconciliation_error == "duplicate provider claim"
