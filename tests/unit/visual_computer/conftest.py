"""Bounded adapter doubles with real SDK contracts and persistent SQLite."""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
import hashlib

from cryptography.fernet import Fernet
import pytest_asyncio
from kestrel_sdk.computer_use import (
    ActionReceipt, ComputerSession, DriverConnection, EnvironmentCapabilities,
    EnvironmentLease, EnvironmentQuote, EnvironmentRequest, ScreenFrame, ScreenObservation,
)
from kestrel_sdk.operator import OperatorContext
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.privacy import PrivacyConfig, PrivacyMode
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
from kestrel_sovereign.visual_computer.runtime import VisualAction, VisualComputerRuntime


class Adapter:
    driver_id = "test-driver"

    def __init__(self):
        self.effects = 0
        self.acquires = 0
        self.sessions = {}
        self.receipts = {}
        self.unknown_action = False
        self.unknown_acquire = False
        self.unknown_release = False
        self.before_effect = None
        self.last_connection = None

    async def capabilities(self):
        return EnvironmentCapabilities(provider_id="test-provider", location="local", surfaces=frozenset({"browser"}), operations=frozenset({"screenshot", "click", "type", "takeover"}))

    async def quote(self, request):
        return EnvironmentQuote(quote_id="quote", request_id=request.request_id, request_sha256=request.digest(), scope=request.scope,
            provider_id=request.provider_id, location="local", expires_at=request.expires_at, hourly_cost_usd=Decimal("0"), estimated_total_cost_usd=Decimal("0"), capabilities=await self.capabilities())

    async def acquire(self, request, quote):
        self.acquires += 1
        if self.unknown_acquire:
            self.unknown_acquire = False
            raise ResourceOutcomeUnknown("acquisition acknowledgement lost")
        return EnvironmentLease(lease_id="lease", request_id=request.request_id, quote_id=quote.quote_id, scope=request.scope,
            provider_id=request.provider_id, location="local", expires_at=request.expires_at, state="ready", provider_resource_id="physical-worker")

    async def inspect(self, scope, lease):
        return lease

    async def connect(self, scope, lease):
        return DriverConnection(lease_id=lease.lease_id, driver_id=self.driver_id, endpoint="private-endpoint", credential="private-secret", expires_at=lease.expires_at)

    async def open(self, request, connection):
        self.last_connection = connection
        session = ComputerSession(session_id=request.session_id, scope=request.scope, lease_id=request.lease_id, driver_id=self.driver_id,
            surface=request.surface, capabilities=request.required_capabilities, state="ready", controller="agent", control_epoch=0, expires_at=request.expires_at)
        self.sessions[session.session_id] = session
        return session

    async def status(self, scope, session_id):
        return self.sessions[session_id]

    async def observe(self, session):
        payload = b"private pixels"
        return ScreenObservation(frame=ScreenFrame(observation_id=f"frame-{session.control_epoch}-{self.effects}", session_id=session.session_id,
            control_epoch=session.control_epoch, observed_at=datetime.now(UTC), width=100, height=100, image_sha256=hashlib.sha256(payload).hexdigest(), media_type="image/png"), image=payload)

    async def act(self, session, request):
        if self.before_effect:
            await self.before_effect()
        self.effects += 1
        receipt = ActionReceipt(action_id=request.action_id, session_id=session.session_id, observation_id=request.observation_id,
            control_epoch=request.control_epoch, outcome="completed", recorded_at=datetime.now(UTC))
        self.receipts[request.action_id] = receipt
        if self.unknown_action:
            raise ResourceOutcomeUnknown("action acknowledgement lost")
        return receipt

    async def action_status(self, session, action_id):
        return self.receipts[action_id]

    async def control(self, session, request):
        self.sessions[session.session_id] = session.model_copy(update={"control_epoch": session.control_epoch+1,
            "controller": {"pause":"none", "takeover":"human", "resume":"agent"}[request.operation],
            "state": "ready" if request.operation == "resume" else "paused"})
        return self.sessions[session.session_id]

    async def release(self, scope, lease):
        if self.unknown_release:
            raise ResourceOutcomeUnknown("physical release acknowledgement lost")
        return lease.model_copy(update={"state":"released", "released_at": datetime.now(UTC)})


@pytest_asyncio.fixture
async def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    async def empty(db):
        pass
    path = tmp_path / "visual.db"
    db = await AsyncDatabase.sqlite(str(path), schema_initializer=empty)
    # This trusted test binding exercises service invariants only. Production
    # authentication is tested separately through the actual host resolver.
    from kestrel_sovereign.security.capability_authority import HostCapabilityAuthority
    from kestrel_sovereign.visual_computer.runtime import VisualAccessDenied
    authority = HostCapabilityAuthority(capability="visual-computer", label="visual computer", non_disclosing_actions=frozenset({VisualAction.STOP.value, VisualAction.RELEASE.value}), access_denied=VisualAccessDenied)
    repo = VisualCustodyRepository(db)
    adapter = Adapter()
    runtime = VisualComputerRuntime(repo, authority, providers={"test-provider":adapter}, drivers={"test-driver":adapter})
    privacy_gate = DurablePersistenceGate()
    holder = SimpleNamespace(privacy_mode=PrivacyMode.NORMAL, privacy_config=PrivacyConfig(), _get_durable_persistence_gate=lambda:privacy_gate)
    gate = AsyncReaderWriterLock()
    revoked = set()

    def context(*, principal="owner", agent="agent", seconds=120, actions=None):
        now = datetime.now(UTC)
        return OperatorContext(tenant_id="tenant", principal_id=principal, acting_agent_id=agent, correlation_id="test", issued_at=now,
            expires_at=now+timedelta(seconds=seconds), granted_capabilities={"visual-computer"}, permitted_boundary_ids={"visual-computer"},
            granted_actions=actions if actions is not None else {action.value for action in VisualAction})

    @contextmanager
    def bind(ctx):
        def current(context, action, link):
            if id(ctx) in revoked:
                raise PermissionError("current grant revoked")
        with authority.bind_authenticated(ctx, privacy_holder=holder, check_current=current, authority_gate=gate):
            yield ctx

    def request(ctx, **changes):
        now = datetime.now(UTC)
        values=dict(request_id="request", scope=runtime.scope(ctx), provider_id="test-provider", profile_id="offline", image="worker@sha256:"+"a"*64,
            surface="browser", required_capabilities=frozenset({"screenshot", "click", "type", "takeover"}), location="local", requested_at=now,
            expires_at=now+timedelta(minutes=10), idle_ttl_seconds=60, ready_timeout_seconds=10, max_hourly_cost_usd=Decimal("0"), max_total_cost_usd=Decimal("0"))
        values.update(changes)
        return EnvironmentRequest(**values)

    yield SimpleNamespace(runtime=runtime, repository=repo, db=db, adapter=adapter, context=context, bind=bind, request=request,
        holder=holder, gate=gate, privacy_gate=privacy_gate, revoked=revoked, path=path, empty=empty)
    await db.close()
