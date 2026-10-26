"""Original registration identity, agent binding and cold publication fences."""

import asyncio

import pytest
from fastapi import FastAPI
from kestrel_sdk.operator import (
    OperatorAuthorizationError,
    ServiceRequirement,
    ServiceScope,
)
from kestrel_sovereign.host_features import (
    SovereignHostContext,
    start_host_features,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.operator import OperatorRuntimeRegistry, ServiceOperation
from tests.fixtures.sdk_contribution_fixture import SDKFixtureHostFeature
from tests.unit.test_operator_runtime import _service

REQUIREMENT = ServiceRequirement("operator.shell", "1.0.0", ServiceScope.HOST)


async def test_registry_exact_original_epoch_and_lexical_lifetime_block_same_object_aba():
    registry = OperatorRuntimeRegistry()
    service = object()
    registration = _service("owner", "1.0.0", service)
    original = registry.register("owner", services=(registration,))
    with pytest.raises(OperatorAuthorizationError):
        async with registry.host_service_operation(REQUIREMENT) as operation:
            assert (
                isinstance(operation, ServiceOperation) and operation.service is service
            )
            operation.check_current()
            registry.unregister(original)
            replacement = registry.register(
                "owner", services=(registration,)
            )  # SAME service and registration objects
            with pytest.raises(OperatorAuthorizationError):
                operation.check_current()
    with pytest.raises(OperatorAuthorizationError):
        operation.check_current()
    async with registry.host_service_operation(REQUIREMENT) as fresh:
        fresh.check_current()
    with pytest.raises(OperatorAuthorizationError):
        fresh.check_current()
    registry.unregister(replacement)
    for requirement in (
        REQUIREMENT,
        ServiceRequirement(
            "operator.shell", "1.0.0", ServiceScope.AGENT, agent_id="foreign-agent"
        ),
    ):
        with pytest.raises(OperatorAuthorizationError):
            async with registry.host_service_operation(requirement):
                pytest.fail("scope/absence admitted")


async def test_agent_binding_preserves_agent_registry_and_exact_retirement(tmp_path):
    registry = OperatorRuntimeRegistry()
    registration = _service("owner", "1.0.0", object())
    registry.register("owner", services=(registration,))
    agent = KestrelAgent(
        did="did:test:host-services",
        storage_path=str(tmp_path / "a.db"),
        sync_enabled=False,
        host_service_resolver=registry,
    )
    own = agent.operator_registry
    assert own is not registry
    with pytest.raises(OperatorAuthorizationError):
        async with agent.host_service_operation(REQUIREMENT) as operation:
            operation.check_current()
            assert agent.clear_host_service_resolver(OperatorRuntimeRegistry()) is False
            operation.check_current()
            agent.bind_host_service_resolver(None)
            agent.bind_host_service_resolver(registry)
            operation.check_current()  # ABA binding must not revive the frame.
    assert agent.operator_registry is own
    with pytest.raises(OperatorAuthorizationError):
        operation.check_current()
    gate = asyncio.Event()
    agent._host_context_publication_gate = gate
    with pytest.raises(OperatorAuthorizationError):
        async with agent.host_service_operation(REQUIREMENT):
            pytest.fail("unpublished host gate")
    gate.set()
    async with agent.host_service_operation(REQUIREMENT) as current:
        current.check_current()
    agent._host_services.retire()
    with pytest.raises(OperatorAuthorizationError):
        agent.bind_host_service_resolver(registry)


async def test_real_manager_cold_registration_rebinds_current_host_and_explicit_identity(
    tmp_path,
):
    manager = AgentManager(base_data_dir=tmp_path)
    first, second = OperatorRuntimeRegistry(), OperatorRuntimeRegistry()
    old, new = object(), object()
    first.register("old", services=(_service("old", "1.0.0", old),))
    second.register("new", services=(_service("new", "1.0.0", new),))
    manager.bind_host_service_resolver(first)
    cold = KestrelAgent(
        did="did:test:cold-host-service",
        storage_path=str(tmp_path / "cold.db"),
        sync_enabled=False,
        host_service_resolver=first,
    )
    own = cold.operator_registry
    # The cold object is absent from manager fan-out while host replacement commits.
    manager.bind_host_service_resolver(second)
    manager._register_agent("Cold", cold)
    assert manager.get_agent("missing") is None and manager.get_agent("Cold") is cold
    async with cold.host_service_operation(REQUIREMENT) as operation:
        assert operation.service is new
        assert own is cold.operator_registry
    # Removing the original exact route, even with a same-DID replacement,
    # cannot lend this operation to a foreign agent.
    with pytest.raises(OperatorAuthorizationError):
        async with cold.host_service_operation(REQUIREMENT) as original:
            manager._agents["Cold"] = KestrelAgent(
                did=cold.did,
                storage_path=str(tmp_path / "foreign.db"),
                sync_enabled=False,
            )
            original.check_current()
    assert manager.clear_host_service_resolver(first) is False
    assert manager.clear_host_service_resolver(second) is True


async def test_actual_host_registration_and_server_binding_revoke_before_shutdown(
    tmp_path,
):
    from kestrel_sovereign import server

    ctx = SovereignHostContext()
    feature = SDKFixtureHostFeature()
    assert await start_host_features([feature], ctx) == [feature]
    agent = KestrelAgent(
        did="did:test:server-host-service",
        storage_path=str(tmp_path / "server.db"),
        sync_enabled=False,
    )
    app = FastAPI()
    app.state.agent = agent
    app.state.agent_manager = None
    app.state.host_context = ctx
    app.state.host_features = [feature]
    server._bind_host_services(app, ctx.operator_registry)
    requirement = ServiceRequirement(
        feature.service_registration.descriptor.name, "1.0.0", ServiceScope.HOST
    )
    with pytest.raises(OperatorAuthorizationError):
        async with agent.host_service_operation(requirement) as operation:
            assert operation.service is feature.service
            await server._shutdown_host_features(app)
            operation.check_current()
    assert feature.stopped
    assert agent.operator_registry.resolve_compatible_service(requirement) is None
    with pytest.raises(OperatorAuthorizationError):
        async with agent.host_service_operation(requirement):
            pytest.fail("stopped host service")


async def test_failed_discovery_retirement_still_stops_independent_host_resources():
    from types import SimpleNamespace
    from kestrel_sovereign import server

    ctx = SovereignHostContext()
    feature = SDKFixtureHostFeature()
    await start_host_features([feature], ctx)
    original = RuntimeError("original discovery retirement failure")

    def failing_clear(expected):
        raise original

    app = FastAPI()
    app.state.agent_manager = None
    app.state.agent = SimpleNamespace(clear_host_service_resolver=failing_clear)
    app.state.host_context = ctx
    app.state.host_features = [feature]
    with pytest.raises(RuntimeError) as failed:
        await server._shutdown_host_features(app)
    assert failed.value is original
    assert feature.stopped
    assert (
        ctx.operator_registry.resolve_service(feature.service_registration.reference)
        is None
    )
