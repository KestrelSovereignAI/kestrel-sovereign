"""Lifecycle-owned visual capability bound to the original host caller."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import hashlib
import hmac
import json
from uuid import uuid4

from kestrel_sdk.operator import OperatorContext
from kestrel_sdk.resources import ResourceError, ResourceScope
from kestrel_sdk.storage.database.interface import TransactionError
from kestrel_sdk.security.encryption import get_master_key_bytes

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.auth import AuthMethod, CallerContext, current_caller_context, caller_context_scope
from kestrel_sovereign.config import load_section
from kestrel_sovereign.endpoints.agent_helpers import get_caller
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database, resolve_scoped_agent_did
from kestrel_sovereign.security.capability_authority import HostCapabilityAuthority
from kestrel_sovereign.security.host_authority import HostAuthorityError, require_sovereign_caller, stable_sovereign_secret
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate
from kestrel_sovereign.turn_scope import capture_turn_scope
from kestrel_sovereign.visual_computer.discovery import construct_adapters
from kestrel_sovereign.visual_computer.egress import ActiveCallEgress
from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
from kestrel_sovereign.visual_computer.runtime import VisualAccessDenied, VisualAction, VisualComputerRuntime, VisualPolicy


class VisualComputerFeature(Feature):
    """Authenticated visual sessions with independent explicit visual grants."""

    supports_subagent_dispatch = False
    promote_tools_on_startup = True

    def __init__(self, agent):
        super().__init__(agent)
        self._cfg = load_section("features").get("visual_computer", {})
        self._accepting = False
        self._generation = 0
        self._gate = AsyncReaderWriterLock()
        self.runtime = None
        self._router = None
        self._agent_id = None
        self._maintenance_task = None

    @property
    def tool_description(self):
        return "Explicitly granted visual sessions with observation-bound actions, Stop, takeover and handback"

    async def initialize(self):
        if self.runtime is not None:
            if (resolve_scoped_agent_did(self.agent) != self._agent_id
                    or resolve_feature_database(self.agent) is not self.runtime.repository.db):
                raise VisualAccessDenied("visual runtime owner changed")
            return
        if self._cfg.get("enabled") is not True:
            return  # Default capability is unavailable; no visual grant exists.
        require_durable_persistence_gate(self.agent)
        agent_id = resolve_scoped_agent_did(self.agent)
        db = resolve_feature_database(self.agent)
        key = get_master_key_bytes()
        if db is None or key is None:
            raise RuntimeError("visual computer requires original durable database and encryption key")
        grants = self._cfg.get("granted_actions", [])
        if not isinstance(grants, list) or not grants or not set(grants) <= {a.value for a in VisualAction}:
            raise ValueError("explicit visual granted_actions are required")
        providers = {name: dict(settings) for name, settings in self._cfg.get("providers", {}).items()}
        drivers = {name:dict(settings) for name,settings in self._cfg.get("drivers", {}).items()}
        egress = ActiveCallEgress()
        for name,settings in drivers.items():
            if name == "local-docker-browser":
                settings["egress_guard"] = egress.operation
        if not providers or not drivers:
            raise ValueError("visual provider and driver factories must be explicitly configured")
        for name, settings in providers.items():
            if name == "local-docker-browser":
                settings["credential_key"] = hmac.new(key, b"kestrel-visual-worker-v1", hashlib.sha256).digest()
        providers, drivers = construct_adapters(providers=providers, drivers=drivers)
        policy = VisualPolicy(permitted_locations=frozenset(self._cfg.get("permitted_locations", ["local"])),
            max_lifetime_seconds=self._cfg.get("max_lifetime_seconds", 1800),
            max_hourly_cost_usd=Decimal(str(self._cfg.get("max_hourly_cost_usd", "0"))),
            max_total_cost_usd=Decimal(str(self._cfg.get("max_total_cost_usd", "0"))),
            operation_seconds=self._cfg.get("operation_seconds", 30))
        authority = HostCapabilityAuthority(capability="visual-computer", label="visual computer",
            non_disclosing_actions=frozenset({VisualAction.STOP.value, VisualAction.RELEASE.value}),
            priority_actions=frozenset({VisualAction.STOP.value, VisualAction.RELEASE.value}),
            access_denied=VisualAccessDenied)
        self._agent_id = agent_id
        self.runtime = VisualComputerRuntime(VisualCustodyRepository(db), authority,
            providers=providers, drivers=drivers, policy=policy, check_destination=self._check_destination, egress=egress)
        from kestrel_sovereign.visual_computer.http import create_visual_computer_router
        self._router = create_visual_computer_router(self.runtime, resolve_context=self._resolve_context,
                                                    plan_operation=self._plan)

    def _check_destination(self, location):
        config = getattr(self.agent, "privacy_config", None)
        if location != "local" and getattr(config, "processing", None) != "cloud":
            raise VisualAccessDenied("privacy processing policy does not permit a remote visual environment")

    def get_tools(self):
        return super().get_tools() if self.runtime is not None else []

    def get_router(self):
        return self._router

    async def on_enable(self):
        if self.runtime is not None:
            async with self._gate:
                self._generation += 1
                self._accepting = True
            if self._maintenance_task is None or self._maintenance_task.done():
                self._maintenance_task = self._track_owned_background_task(
                    self._maintain(), name="visual-computer-cleanup", _resident_owner=True)

    async def _maintain(self):
        owner = ResourceScope(tenant_id=self._agent_id, principal_id="host-sovereign", agent_id=self._agent_id)
        while self._accepting:
            try:
                if await self.runtime.maintain_owned_sessions(owner):
                    logging.getLogger(__name__).warning("Visual physical cleanup requires original lease reconciliation")
            except (ResourceError, TransactionError, OSError):
                logging.getLogger(__name__).warning("Visual custody cleanup unavailable; original leases remain retained")
            await asyncio.sleep(5)

    async def _withdraw(self):
        self._accepting = False
        self._generation += 1
        await self._cancel_owned_background_tasks()
        if self.runtime is not None:
            owner = ResourceScope(tenant_id=self._agent_id, principal_id="host-sovereign", agent_id=self._agent_id)
            uncertain = await self.runtime.fence_owned_sessions(owner)
            if uncertain:
                recorder = getattr(self.agent, "record_feature_unavailable", None)
                if callable(recorder):
                    recorder(feature=self, feature_name=self.name, reason="visual input fence requires original control reconciliation")
        async with self._gate:
            pass

    async def on_disable(self):
        await self._withdraw()

    async def shutdown(self):
        await self._withdraw()
        return await super().shutdown()

    def _require_live(self, generation, request=None):
        if (self.runtime is None or not self._accepting or generation != self._generation
                or getattr(self, "enabled", True) is not True
                or self.agent.features.get(self.name) is not self
                or resolve_scoped_agent_did(self.agent) != self._agent_id
                or resolve_feature_database(self.agent) is not self.runtime.repository.db
                or self._cfg.get("enabled") is not True):
            raise VisualAccessDenied("visual runtime is no longer current")
        if request is not None:
            routed = getattr(request.state, "agent", None)
            if routed is not None and routed is not self.agent:
                raise VisualAccessDenied("visual runtime does not match route")
            manager = getattr(request.app.state, "agent_manager", None)
            if manager is not None:
                if not any(agent is self.agent for agent in manager.list_agents().values()):
                    raise VisualAccessDenied("visual runtime was withdrawn")
            elif getattr(request.app.state, "agent", None) is not self.agent:
                raise VisualAccessDenied("visual runtime was replaced")

    @asynccontextmanager
    async def _bind_context(self, caller, request=None):
        if (not isinstance(caller, CallerContext) or caller.auth_method is not AuthMethod.API_KEY
                or caller.is_sovereign is not True):
            raise VisualAccessDenied("visual access requires original sovereign authority")
        generation = self._generation
        self._require_live(generation, request)
        with caller_context_scope(caller):
            try:
                require_sovereign_caller("visual computer")
                if request is not None:
                    secret = stable_sovereign_secret("visual computer")
                    bearer = request.headers.get("authorization", "")
                    presented = (request.headers.get("x-api-key", ""), bearer[7:] if bearer.startswith("Bearer ") else "")
                    if not any(hmac.compare_digest(value.encode(), secret) for value in presented):
                        raise HostAuthorityError("explicit visual credential required")
            except HostAuthorityError:
                raise VisualAccessDenied("visual authority is unavailable") from None
            scope = capture_turn_scope(self.agent)
            now = self.runtime.authority.clock()
            context = OperatorContext(tenant_id=self._agent_id, principal_id="host-sovereign", acting_agent_id=self._agent_id,
                granted_actions=frozenset(self._cfg["granted_actions"]), granted_capabilities={"visual-computer"},
                permitted_boundary_ids={self.runtime.repository.storage_id}, correlation_id=uuid4().hex,
                # Retain one fixed original deadline for the configured call.
                # A hard 30-second cap silently cut off otherwise admitted
                # local vision calls; frame and resource expiry remain tighter
                # independent checks and never renew with a model retry.
                issued_at=now, expires_at=now+timedelta(seconds=min(
                    self.runtime.policy.operation_seconds, self.runtime.policy.max_lifetime_seconds)))
            def current(context, action, target):
                self._require_live(generation, request)
                if (target is not None or action.value not in self._cfg["granted_actions"]
                        or (request is not None and get_caller(request) is not caller)):
                    raise VisualAccessDenied("visual grant is no longer current")
                with scope.bind():
                    require_sovereign_caller("visual computer")
            with self.runtime.authority.bind_authenticated(context, privacy_holder=self.agent,
                    check_current=current, authority_gate=self._gate):
                yield context

    @asynccontextmanager
    async def _resolve_context(self, request):
        async with self._bind_context(get_caller(request), request) as context:
            yield context

    @tool(name="visual_computer_status", description="Inspect original-owned visual session custody")
    async def visual_computer_status(self, session_id: str) -> dict:
        """Inspect original-owned visual session and physical cleanup custody."""
        async with self._bind_context(current_caller_context()) as context:
            return await self.runtime.status(context, session_id)

    @tool(name="visual_computer_action", description="Submit an observation-bound visual action")
    async def visual_computer_action(self, request_json: str) -> dict:
        """Submit one exact SDK action with current observation, epoch, ID and deadline."""
        from kestrel_sdk.computer_use import ActionRequest
        request = ActionRequest.model_validate_json(request_json)
        async with self._bind_context(current_caller_context()) as context:
            return (await self.runtime.act(context, request)).model_dump(mode="json")

    @tool(name="visual_computer_action_status", description="Reconcile original accepted visual action without replay")
    async def visual_computer_action_status(self, session_id: str, action_id: str) -> dict:
        """Reconcile an accepted visual action by its original ID without replay."""
        async with self._bind_context(current_caller_context()) as context:
            return (await self.runtime.action_status(context, session_id, action_id)).model_dump(mode="json")

    @tool(name="visual_computer_stop", description="Stop visual input cooperatively")
    async def visual_computer_stop(self, session_id: str, expected_control_epoch: int, request_id: str) -> dict:
        """Stop visual input cooperatively; keep the physical lease in cleanup custody."""
        from kestrel_sdk.computer_use import ControlRequest
        request = ControlRequest(request_id=request_id, session_id=session_id,
            expected_control_epoch=expected_control_epoch, operation="pause")
        async with self._bind_context(current_caller_context()) as context:
            stopped = await self.runtime.control(context, session_id, request, action=VisualAction.STOP)
            return {"session_id":session_id, "stopped":stopped.controller == "none"}

    @tool(name="visual_computer_release", description="Request physical environment release")
    async def visual_computer_release(self, session_id: str) -> dict:
        """Request physical environment release; unknown outcomes retain cleanup custody."""
        async with self._bind_context(current_caller_context()) as context:
            return await self.runtime.release(context, session_id)

    @tool(name="visual_computer_admit", description="Admit a bounded visual environment under an explicit grant")
    async def visual_computer_admit(self, request_json: str) -> dict:
        """Admit an explicitly bounded environment; request JSON cannot choose an owner."""
        from kestrel_sdk.computer_use import EnvironmentRequest
        if len(request_json.encode()) > 64*1024:
            raise ValueError("visual request exceeds byte limit")
        value = json.loads(request_json)
        if not isinstance(value, dict) or "scope" in value:
            raise ValueError("visual ownership cannot be selected by input")
        async with self._bind_context(current_caller_context()) as context:
            value["scope"] = self.runtime.scope(context).model_dump(mode="json")
            request = EnvironmentRequest.model_validate_json(json.dumps(value))
            return (await self.runtime.admit(context, request)).model_dump(mode="json")

    def get_ui_contributions(self):
        from kestrel_sdk.features.ui import UIContributions
        if self.runtime is None:
            return None
        return UIContributions(static_dir=str(Path(__file__).parent / "static"),
            modules=["visual-computer.js"], capability="visual-computer")

    @tool(name="visual_plan", description="Plan one observation-bound visual action through the host inference router")
    async def visual_plan(self, session_id: str, prompt: str) -> dict:
        """Observe and plan once; submitting an action remains a separate authorized call."""
        async with self._bind_context(current_caller_context()) as context:
            return await self._plan(context, session_id, prompt)

    async def _plan(self, context, session_id, prompt):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > 16*1024:
            raise ValueError("visual planning requires a bounded nonempty prompt")
        service = getattr(self.agent, "llm_service", None)
        if service is None:
            raise RuntimeError("the host inference service is unavailable")
        model = self._cfg.get("visual_model")
        if model is not None and (not isinstance(model,str) or not model.strip() or len(model)>512):
            raise ValueError("configured visual model is invalid")
        from kestrel_sovereign.visual_computer.inference import VisualInference
        configured_options = self._cfg.get("visual_request_options")
        options = {}
        if configured_options is not None:
            if (not isinstance(configured_options,dict) or not model
                    or set(configured_options)-{"reasoning_effort","thinking_budget_tokens","raw"}):
                raise ValueError("visual inference options require a configured model and reasoning options")
            from kestrel_sdk.llm import RequestOptions
            options["request_options"] = RequestOptions(**configured_options)
        force_local = getattr(getattr(self.agent,"privacy_config",None),"processing",None) != "cloud"
        async with self.runtime.operation(context,VisualAction.OBSERVE):
            result = await VisualInference(self.runtime,service,**options).plan(context,session_id,prompt,
                model_override=model,force_local_only=force_local)
            return {key:value.model_dump(mode="json") if hasattr(value,"model_dump") else value
                    for key,value in result.items()}
