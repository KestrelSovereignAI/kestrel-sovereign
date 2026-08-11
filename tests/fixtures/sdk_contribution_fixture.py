"""Out-of-tree-style fixture: intentionally imports no Sovereign modules."""

from pathlib import Path

from fastapi import APIRouter

from kestrel_sdk.features import (
    FeaturePermissionDefaults,
    PermissionLevel,
    SetupStepClassification,
    SetupStepRegistration,
    UIContributions,
    WaitProviderRegistration,
    WorkflowRegistration,
)
from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.features.host_base import HostFeature
from kestrel_sdk.operator import (
    ServiceDescriptor,
    ServiceRegistration,
    ServiceScope,
)
from kestrel_sdk.signals import (
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Trust,
)
from kestrel_sdk.tools import Outcome, ToolCategory, WaitStatus


STATIC_DIR = Path(__file__).with_name("sdk_contribution_static")


class FixtureWaitProvider:
    signal = None

    def __init__(self, kind: str) -> None:
        self.kind = kind

    async def poll(self, handle: str) -> WaitStatus:
        return WaitStatus(outcome=Outcome.DONE, summary=handle)


def _source(name: str) -> SourceRegistration:
    async def handle(payload):
        return payload

    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handle,
        trust=Trust.TRUSTED,
        log_redaction=RedactionPolicy(summarize=lambda payload: ""),
    )


class _FixtureContributions:
    contribution_prefix = "fixture"

    def _init_contributions(self) -> None:
        self.contribution_calls = {
            "services": 0,
            "waits": 0,
            "workflows": 0,
            "permissions": 0,
            "setup": 0,
        }
        owner = self.contribution_owner
        self.service = object()
        self.wait_provider = FixtureWaitProvider(f"{self.contribution_prefix}-wait")
        self.actor = lambda value=None: value
        self.source = _source(f"{self.contribution_prefix}.signal")
        self.setup_step = lambda ctx: ctx.record(self.contribution_prefix)
        scope = (
            ServiceScope.AGENT
            if isinstance(self, Feature)
            else ServiceScope.HOST
        )
        agent_id = self.agent.did if scope is ServiceScope.AGENT else None
        self.service_registration = ServiceRegistration(
            descriptor=ServiceDescriptor(
                name=f"{self.contribution_prefix}-service",
                version="1.0.0",
                scope=scope,
            ),
            service=self.service,
            owner=owner,
            agent_id=agent_id,
        )
        self.wait_registration = WaitProviderRegistration(
            owner=owner,
            name=self.wait_provider.kind,
            provider=self.wait_provider,
        )
        self.workflow_registration = WorkflowRegistration(
            owner=owner,
            name=f"{self.contribution_prefix}-workflow",
            actor=self.actor,
            sources=(self.source,),
        )
        permission_overrides = (
            {"fixture_tool": PermissionLevel.ALLOW}
            if isinstance(self, Feature)
            else {}
        )
        self.permission_defaults = FeaturePermissionDefaults(
            feature_default=PermissionLevel.ASK,
            tool_overrides=permission_overrides,
        )
        self.setup_registration = SetupStepRegistration(
            owner=owner,
            name=f"{self.contribution_prefix}-setup",
            step=self.setup_step,
            classification=SetupStepClassification.OPTIONAL,
        )

    def get_service_registrations(self):
        self.contribution_calls["services"] += 1
        return (self.service_registration,)

    def get_wait_provider_registrations(self):
        self.contribution_calls["waits"] += 1
        return (self.wait_registration,)

    def get_workflow_registrations(self):
        self.contribution_calls["workflows"] += 1
        return (self.workflow_registration,)

    def get_feature_permission_defaults(self):
        self.contribution_calls["permissions"] += 1
        return self.permission_defaults

    def get_setup_step_registrations(self):
        self.contribution_calls["setup"] += 1
        return (self.setup_registration,)

    def get_router(self):
        router = APIRouter()

        @router.get(f"/api/{self.contribution_prefix}/fixture")
        async def fixture_route():
            return {"feature": self.contribution_prefix}

        return router

    def get_ui_contributions(self):
        return UIContributions(
            static_dir=str(STATIC_DIR),
            modules=["panel.js"],
        )


class SDKFixtureFeature(_FixtureContributions, Feature):
    contribution_prefix = "agent-fixture"
    tool_description = "SDK-only lifecycle contribution fixture"

    def __init__(self, agent):
        Feature.__init__(self, agent)
        self._init_contributions()
        self.initialized = False
        self.disabled = False

    async def initialize(self):
        self.initialized = True

    async def on_disable(self):
        self.disabled = True

    @tool(
        name="fixture_tool",
        description="Exercise the SDK-only fixture",
        category=ToolCategory.SYSTEM,
    )
    async def fixture_tool(self):
        return "fixture"


class SDKFixtureHostFeature(_FixtureContributions, HostFeature):
    name = "host-fixture"
    contribution_prefix = "host-fixture"

    def __init__(self):
        self._init_contributions()
        self.started = False
        self.stopped = False

    async def on_host_start(self, ctx):
        self.started = True

    async def on_host_stop(self, ctx):
        self.stopped = True


__all__ = ["SDKFixtureFeature", "SDKFixtureHostFeature"]
