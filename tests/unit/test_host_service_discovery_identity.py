"""Opaque discovery comparison never substitutes for a fresh original lease."""

import asyncio
import pickle
from types import SimpleNamespace

import pytest
from kestrel_sdk.operator import OperatorAuthorizationError, ServiceRequirement, ServiceScope
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.operator import OperatorRuntimeRegistry, ServiceDiscoveryIdentity, ServiceOperation
from tests.unit.test_operator_runtime import _service

REQ = ServiceRequirement("operator.shell", "1.0.0", ServiceScope.HOST)


async def test_same_epoch_fresh_identity_matches_but_closed_checker_stays_closed():
    registry = OperatorRuntimeRegistry()
    registry.register("owner", services=(_service("owner", "1.0.0", object()),))
    async with registry.host_service_operation(REQ) as original:
        identity = original.discovery_identity
        original.check_current()
    async with registry.host_service_operation(REQ) as fresh:
        assert fresh.discovery_identity is not identity
        assert fresh.discovery_identity.matches(identity)
        fresh.check_current()
        with pytest.raises(OperatorAuthorizationError):
            original.check_current()
    assert repr(identity) == "<opaque host service discovery identity>"
    with pytest.raises(TypeError):
        pickle.dumps(identity)
    with pytest.raises(TypeError):
        ServiceDiscoveryIdentity()
    assert not identity.matches(object())
    assert not identity.matches(object.__new__(ServiceDiscoveryIdentity))
    # Existing two-field construction is descriptive data, not retained proof.
    counterfeit = ServiceOperation(original.service, lambda: None, identity)
    assert counterfeit is not original
    with pytest.raises(OperatorAuthorizationError):
        original.check_current()


async def test_same_service_and_registration_reregister_has_new_discovery_identity():
    registry = OperatorRuntimeRegistry()
    service = object()
    registration = _service("owner", "1.0.0", service)
    first = registry.register("owner", services=(registration,))
    async with registry.host_service_operation(REQ) as operation:
        identity = operation.discovery_identity
    registry.unregister(first)
    registry.register("owner", services=(registration,))
    async with registry.host_service_operation(REQ) as replacement:
        assert replacement.service is service
        replacement.check_current()
        assert not replacement.discovery_identity.matches(identity)


async def test_shared_set_seal_and_same_implementation_do_not_conflate_service_registrations():
    class HostileService:
        def __eq__(self, other):
            raise AssertionError("service equality must never run")
        def __repr__(self):
            raise AssertionError("service repr must never run")
    registry = OperatorRuntimeRegistry()
    service = HostileService()
    registry.register("owner", services=(_service("owner", "1.0.0", service),
        _service("owner", "1.0.0", service, name="operator.other")))
    async with registry.host_service_operation(REQ) as first:
        async with registry.host_service_operation(ServiceRequirement("operator.other", "1.0.0", ServiceScope.HOST)) as second:
            first.check_current(); second.check_current()
            assert first.service is second.service
            assert not first.discovery_identity.matches(second.discovery_identity)


async def test_binding_resolver_aba_changes_metadata_without_reviving_old_lease(tmp_path):
    registry = OperatorRuntimeRegistry()
    registry.register("owner", services=(_service("owner", "1.0.0", object()),))
    agent = KestrelAgent(did="did:test:discovery-identity", storage_path=str(tmp_path/"agent.db"),
                         sync_enabled=False, host_service_resolver=registry)
    assert agent.host_service_operation_api_version == 1 and agent.host_service_identity_api_version == 1
    async with agent.host_service_operation(REQ) as first:
        identity = first.discovery_identity
    agent.bind_host_service_resolver(None)
    agent.bind_host_service_resolver(registry)
    async with agent.host_service_operation(REQ) as current:
        current.check_current()
        assert current.service is first.service
        assert not current.discovery_identity.matches(identity)
    with pytest.raises(OperatorAuthorizationError):
        first.check_current()


async def test_active_operation_cannot_survive_original_binding_aba(tmp_path):
    registry = OperatorRuntimeRegistry()
    registry.register("owner", services=(_service("owner", "1.0.0", object()),))
    agent = KestrelAgent(
        did="did:test:active-discovery-aba",
        storage_path=str(tmp_path / "agent.db"),
        sync_enabled=False,
        host_service_resolver=registry,
    )
    with pytest.raises(OperatorAuthorizationError):
        async with agent.host_service_operation(REQ) as original:
            agent.bind_host_service_resolver(None)
            agent.bind_host_service_resolver(registry)
            async with agent.host_service_operation(REQ) as fresh:
                fresh.check_current()
                assert not fresh.discovery_identity.matches(original.discovery_identity)
                original.check_current()


@pytest.mark.parametrize("change", ["state", "generation"])
async def test_publication_identity_and_generation_each_change_discovery_identity(tmp_path, change):
    registry = OperatorRuntimeRegistry()
    registry.register("owner", services=(_service("owner", "1.0.0", object()),))
    agent = KestrelAgent(did="did:test:published-identity", storage_path=str(tmp_path/"agent.db"),
                         sync_enabled=False, host_service_resolver=registry)
    publication = SimpleNamespace(generation=1, service_resolver=registry)
    agent._host_context_publication_state = publication
    agent._host_context_publication_gate = asyncio.Event()
    agent._host_context_publication_gate.set()
    async with agent.host_service_operation(REQ) as old:
        identity = old.discovery_identity
    if change == "state":
        agent._host_context_publication_state = SimpleNamespace(generation=1, service_resolver=registry)
    else:
        publication.generation += 1
    async with agent.host_service_operation(REQ) as current:
        current.check_current()
        assert not current.discovery_identity.matches(identity)


async def test_publication_state_aba_with_new_generation_cannot_reuse_identity(tmp_path):
    registry = OperatorRuntimeRegistry()
    registry.register("owner", services=(_service("owner", "1.0.0", object()),))
    agent = KestrelAgent(
        did="did:test:publication-state-aba",
        storage_path=str(tmp_path / "agent.db"),
        sync_enabled=False,
        host_service_resolver=registry,
    )
    publication = SimpleNamespace(generation=1, service_resolver=registry)
    agent._host_context_publication_state = publication
    async with agent.host_service_operation(REQ) as original:
        identity = original.discovery_identity
    agent._host_context_publication_state = SimpleNamespace(
        generation=2, service_resolver=None
    )
    publication.generation += 1
    agent._host_context_publication_state = publication
    async with agent.host_service_operation(REQ) as fresh:
        fresh.check_current()
        assert not fresh.discovery_identity.matches(identity)


async def test_another_binding_to_same_agent_registry_cannot_reuse_identity(tmp_path):
    from kestrel_sovereign.operator.host_services import HostServiceBinding
    registry = OperatorRuntimeRegistry()
    registry.register("owner", services=(_service("owner", "1.0.0", object()),))
    agent = KestrelAgent(did="did:test:binding-identity", storage_path=str(tmp_path/"agent.db"),
                         sync_enabled=False, host_service_resolver=registry)
    async with agent.host_service_operation(REQ) as old:
        identity = old.discovery_identity
    agent._host_services = HostServiceBinding(agent, registry)
    async with agent.host_service_operation(REQ) as fresh:
        fresh.check_current()
        assert not fresh.discovery_identity.matches(identity)


async def test_old_two_field_registry_consumer_keeps_original_current_checks(tmp_path):
    from contextlib import asynccontextmanager
    class OriginalReadRegistry(OperatorRuntimeRegistry):
        @asynccontextmanager
        async def host_service_operation(self, requirement):
            async with super().host_service_operation(requirement) as original:
                yield ServiceOperation(original.service, original.check_current)
    registry = OriginalReadRegistry()
    registry.register("owner", services=(_service("owner", "1.0.0", object()),))
    agent = KestrelAgent(did="did:test:old-operation", storage_path=str(tmp_path/"agent.db"),
                         sync_enabled=False, host_service_resolver=registry)
    async with agent.host_service_operation(REQ) as original:
        original.check_current()
        assert original.discovery_identity is None
    with pytest.raises(OperatorAuthorizationError):
        original.check_current()


async def test_generation_metadata_forgery_does_not_invoke_foreign_equality():
    registry = OperatorRuntimeRegistry()
    registry.register("owner", services=(_service("owner", "1.0.0", object()),))
    async with registry.host_service_operation(REQ) as original:
        identity = original.discovery_identity
        class HostileEquality:
            def __eq__(self, other):
                raise AssertionError("foreign equality was invoked")
        forged = object.__new__(ServiceDiscoveryIdentity)
        parts = list(identity._parts)
        parts[5] = HostileEquality()
        object.__setattr__(forged, "_parts", tuple(parts))
        assert not identity.matches(forged)


async def test_issued_record_requires_exact_self_service_identity_checker_and_open_frame():
    from dataclasses import replace
    registry = OperatorRuntimeRegistry()
    registry.register("owner", services=(_service("owner", "1.0.0", object()),))
    async with registry.host_service_operation(REQ) as original:
        original.require_issued()
        for counterfeit in (
            ServiceOperation(original.service, lambda: None, original.discovery_identity),
            replace(original),
            replace(original, check_current=lambda: None),
            replace(original, service=object()),
            replace(original, discovery_identity=None),
        ):
            # Even copying the original opaque issuer cannot issue a different
            # record; it still names the exact original self/service/identity.
            object.__setattr__(counterfeit, "_issuer", original._issuer)
            with pytest.raises(OperatorAuthorizationError):
                counterfeit.require_issued()
        assert repr(original._issuer) == "<opaque original host operation issuance>"
        with pytest.raises((TypeError, pickle.PicklingError)):
            pickle.dumps(original)
        with pytest.raises(TypeError):
            pickle.dumps(original._issuer)
    with pytest.raises(OperatorAuthorizationError):
        original.require_issued()


async def test_issued_original_cannot_survive_registry_aba_with_same_service():
    registry = OperatorRuntimeRegistry()
    registration = _service("owner", "1.0.0", object())
    first = registry.register("owner", services=(registration,))
    with pytest.raises(OperatorAuthorizationError):
        async with registry.host_service_operation(REQ) as original:
            original.require_issued()
            registry.unregister(first)
            registry.register("owner", services=(registration,))
            async with registry.host_service_operation(REQ) as current:
                current.require_issued()
                assert current.service is original.service
                assert not current.discovery_identity.matches(original.discovery_identity)
                original.require_issued()


async def test_two_field_api1_stays_descriptive_and_cannot_claim_issuance():
    plain = ServiceOperation(object(), lambda: None)
    assert plain.check_current() is None
    with pytest.raises(OperatorAuthorizationError):
        plain.require_issued()
