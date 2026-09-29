"""Focused contract tests for Sovereign's generic operator runtime registry."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kestrel_sdk.features import WorkflowRegistration
from kestrel_sdk.operator import (
    ExecutionTargetDescriptor,
    ExecutionTargetReference,
    OperatorContext,
    ServiceDescriptor,
    ServiceReference,
    ServiceRegistration,
    ServiceRequirement,
    ServiceScope,
)
from kestrel_sovereign.operator import (
    ExecutionTargetRegistration,
    ExecutionTargetUnavailableError,
    OperatorRegistrationConflictError,
    OperatorRegistrationIdentityError,
    OperatorRegistrationSet,
    OperatorRuntimeRegistry,
)


NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def _registry() -> OperatorRuntimeRegistry:
    return OperatorRuntimeRegistry(clock=lambda: NOW)


def _context(**overrides: object) -> OperatorContext:
    values: dict[str, object] = {
        "principal_id": "principal-1",
        "tenant_id": "tenant-1",
        "granted_actions": set(),
        "granted_capabilities": {"shell.execute"},
        "permitted_boundary_ids": {"workspace-1"},
        "correlation_id": "request-1",
        "issued_at": NOW - timedelta(minutes=5),
        "expires_at": NOW + timedelta(minutes=5),
    }
    values.update(overrides)
    return OperatorContext(**values)  # type: ignore[arg-type]


def _service(
    owner: str,
    version: str,
    implementation: object,
    *,
    name: str = "operator.shell",
    scope: ServiceScope = ServiceScope.HOST,
    agent_id: str | None = None,
) -> ServiceRegistration:
    return ServiceRegistration(
        ServiceDescriptor(name, version, scope),
        implementation,
        owner,
        agent_id=agent_id,
    )


def _target(
    *,
    owner: str = "target-owner",
    tenant_id: str = "tenant-1",
    target_id: str = "target-1",
    boundary_id: str = "workspace-1",
    handle: object | None = None,
    capabilities: frozenset[str] = frozenset({"shell.execute"}),
    entitled_capabilities: frozenset[str] | None = None,
) -> ExecutionTargetRegistration:
    return ExecutionTargetRegistration(
        owner=owner,
        descriptor=ExecutionTargetDescriptor(
            target_id=target_id,
            target_kind="container",
            display_name="Build worker",
            tenant_id=tenant_id,
            boundary_id=boundary_id,
            capabilities=capabilities,
        ),
        handle=object() if handle is None else handle,
        entitled_capabilities=entitled_capabilities,
    )


def test_service_exact_and_highest_compatible_semver_resolution() -> None:
    registry = _registry()
    old, current, prerelease, next_major = object(), object(), object(), object()
    registrations = (
        _service("owner", "1.2.0", old),
        _service("owner", "1.9.4", current),
        _service("owner", "1.10.0-rc.1", prerelease),
        _service("owner", "2.0.0", next_major),
    )
    registry.register("owner", services=registrations)

    assert registry.resolve_service(registrations[0].reference) is old
    assert (
        registry.resolve_compatible_service(
            ServiceRequirement("operator.shell", "1.1.0", ServiceScope.HOST)
        )
        is current
    )
    assert (
        registry.resolve_compatible_service(
            ServiceRequirement("operator.shell", "3.0.0", ServiceScope.HOST)
        )
        is None
    )


def test_zero_major_compatibility_is_confined_to_required_minor() -> None:
    registry = _registry()
    compatible, breaking = object(), object()
    registry.register(
        "owner",
        services=(
            _service("owner", "0.2.9", compatible),
            _service("owner", "0.3.1", breaking),
        ),
    )

    assert (
        registry.resolve_compatible_service(
            ServiceRequirement("operator.shell", "0.2.1", ServiceScope.HOST)
        )
        is compatible
    )


def test_service_host_and_agent_namespaces_never_fall_back() -> None:
    registry = _registry()
    host, first_agent, second_agent = object(), object(), object()
    registry.register(
        "owner",
        services=(
            _service("owner", "1.0.0", host),
            _service(
                "owner",
                "1.0.0",
                first_agent,
                scope=ServiceScope.AGENT,
                agent_id="agent-1",
            ),
            _service(
                "owner",
                "1.0.0",
                second_agent,
                scope=ServiceScope.AGENT,
                agent_id="agent-2",
            ),
        ),
    )

    assert (
        registry.resolve_service(
            ServiceReference("operator.shell", "1.0.0", ServiceScope.HOST)
        )
        is host
    )
    assert (
        registry.resolve_service(
            ServiceReference(
                "operator.shell", "1.0.0", ServiceScope.AGENT, "agent-1"
            )
        )
        is first_agent
    )
    assert (
        registry.resolve_service(
            ServiceReference(
                "operator.shell", "1.0.0", ServiceScope.AGENT, "agent-3"
            )
        )
        is None
    )


def test_conflicting_batch_rolls_back_every_registry_kind() -> None:
    registry = _registry()
    active_workflow = WorkflowRegistration("active", "workflow", lambda: None)
    registry.register("active", workflows=(active_workflow,))
    prospective_service = _service("prospective", "1.0.0", object())
    conflicting_workflow = WorkflowRegistration(
        "prospective", "workflow", lambda: None
    )

    with pytest.raises(OperatorRegistrationConflictError):
        registry.register(
            "prospective",
            services=(prospective_service,),
            workflows=(conflicting_workflow,),
        )

    assert registry.resolve_service(prospective_service.reference) is None
    assert registry.resolve_workflow_actor("workflow") is active_workflow.actor


def test_duplicate_names_inside_batch_are_rejected_without_mutation() -> None:
    registry = _registry()
    first = WorkflowRegistration("owner", "workflow", lambda: None)
    second = WorkflowRegistration("owner", "workflow", lambda: None)

    with pytest.raises(OperatorRegistrationConflictError):
        registry.register("owner", workflows=(first, second))

    assert registry.resolve_workflow_actor("workflow") is None


def test_target_registration_reuses_strict_sdk_owner_identity() -> None:
    with pytest.raises((TypeError, ValueError)):
        _target(owner="not an owner")


def test_teardown_requires_exact_set_registration_and_implementation_identity() -> None:
    registry = _registry()
    implementation = object()
    service = _service("owner", "1.0.0", implementation)
    actor = lambda: None
    workflow = WorkflowRegistration("owner", "workflow", actor)
    handle = object()
    target = _target(owner="owner", handle=handle)
    active = registry.register(
        "owner",
        services=(service,),
        workflows=(workflow,),
        execution_targets=(target,),
    )
    forged_registration = _service("owner", "1.0.0", implementation)
    forged_set = OperatorRegistrationSet(
        "owner",
        services=(forged_registration,),
        workflows=(WorkflowRegistration("owner", "workflow", actor),),
        execution_targets=(_target(owner="owner", handle=handle),),
    )

    with pytest.raises(OperatorRegistrationIdentityError):
        registry.unregister(forged_set)

    assert registry.resolve_service(service.reference) is implementation
    assert registry.resolve_workflow_actor("workflow") is actor
    registry.unregister(active)
    assert registry.resolve_service(service.reference) is None
    assert registry.resolve_workflow_actor("workflow") is None
    with pytest.raises(OperatorRegistrationIdentityError):
        registry.unregister(active)


def test_quarantine_removes_exact_survivors_from_a_drifted_set() -> None:
    registry = _registry()
    service = _service("owner", "1.0.0", object())
    workflow = WorkflowRegistration("owner", "workflow", lambda: None)
    active = registry.register(
        "owner",
        services=(service,),
        workflows=(workflow,),
    )
    del registry._services[service.reference]

    assert registry.quarantine_registration_set(active) is True
    assert registry.resolve_service(service.reference) is None
    assert registry.resolve_workflow_actor(workflow.name) is None
    assert registry.quarantine_registration_set(active) is False


def test_quarantine_preserves_foreign_operator_replacements() -> None:
    registry = _registry()
    service = _service("owner", "1.0.0", object())
    workflow = WorkflowRegistration("owner", "workflow", lambda: None)
    active = registry.register(
        "owner",
        services=(service,),
        workflows=(workflow,),
    )
    foreign_actor = lambda: "foreign"
    registry._workflows[workflow.name] = WorkflowRegistration(
        "foreign-owner", workflow.name, foreign_actor
    )

    assert registry.quarantine_registration_set(active) is True
    assert registry.resolve_service(service.reference) is None
    assert registry.resolve_workflow_actor(workflow.name) is foreign_actor


def test_quarantine_removes_exact_survivors_when_set_ledger_is_missing() -> None:
    registry = _registry()
    service = _service("owner", "1.0.0", object())
    workflow = WorkflowRegistration("owner", "workflow", lambda: None)
    active = registry.register(
        "owner",
        services=(service,),
        workflows=(workflow,),
    )
    del registry._active_sets[id(active)]

    assert registry.quarantine_registration_set(active) is True
    assert registry.resolve_service(service.reference) is None
    assert registry.resolve_workflow_actor(workflow.name) is None
    assert registry.quarantine_registration_set(active) is False


def test_quarantine_rejects_forged_set_wrapping_public_registration() -> None:
    registry = _registry()
    workflow = WorkflowRegistration("victim", "workflow", lambda: None)
    active = registry.register("victim", workflows=(workflow,))
    public = registry.get_workflow_registration(workflow.name)
    assert public is workflow
    forged = OperatorRegistrationSet(owner="victim", workflows=(public,))

    assert registry.quarantine_registration_set(forged) is False
    assert registry.resolve_workflow_actor(workflow.name) is workflow.actor
    registry.unregister(active)


def test_quarantine_rejects_another_issued_sets_copied_seal() -> None:
    registry = _registry()
    attacker_workflow = WorkflowRegistration(
        "attacker", "attacker-workflow", lambda: "attacker"
    )
    victim_workflow = WorkflowRegistration(
        "victim", "victim-workflow", lambda: "victim"
    )
    attacker = registry.register("attacker", workflows=(attacker_workflow,))
    victim = registry.register("victim", workflows=(victim_workflow,))
    public_victim = registry.get_workflow_registration(victim_workflow.name)
    assert public_victim is victim_workflow
    forged = OperatorRegistrationSet(
        owner="victim",
        workflows=(public_victim,),
    )
    object.__setattr__(forged, "_registry_seal", attacker._registry_seal)

    assert registry.quarantine_registration_set(forged) is False
    assert registry.resolve_workflow_actor(victim_workflow.name) is (
        victim_workflow.actor
    )
    assert registry.resolve_workflow_actor(attacker_workflow.name) is (
        attacker_workflow.actor
    )
    registry.unregister(victim)
    registry.unregister(attacker)


def test_removing_middle_version_does_not_disturb_or_resurrect_other_sets() -> None:
    registry = _registry()
    first = _service("first", "1.0.0", object())
    middle = _service("middle", "1.1.0", object())
    last = _service("last", "1.2.0", object())
    first_set = registry.register("first", services=(first,))
    middle_set = registry.register("middle", services=(middle,))
    last_set = registry.register("last", services=(last,))

    registry.unregister(middle_set)

    assert registry.resolve_service(middle.reference) is None
    assert registry.resolve_service(first.reference) is first.service
    assert registry.resolve_service(last.reference) is last.service
    registry.unregister(last_set)
    assert (
        registry.resolve_compatible_service(
            ServiceRequirement("operator.shell", "1.0.0", ServiceScope.HOST)
        )
        is first.service
    )
    registry.unregister(first_set)


@pytest.mark.asyncio
async def test_target_resolution_returns_only_authorized_opaque_handle() -> None:
    registry = _registry()
    handle = {"command": "private", "env": {"TOKEN": "secret"}}
    target = _target(handle=handle)
    registry.register(target.owner, execution_targets=(target,))

    resolved = await registry.resolve_execution_target(
        ExecutionTargetReference("target-1", "workspace-1", "shell.execute"),
        _context(),
    )

    assert resolved is handle


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reference", "context"),
    [
        (
            ExecutionTargetReference("missing", "workspace-1", "shell.execute"),
            _context(),
        ),
        (
            ExecutionTargetReference("target-1", "workspace-1", "shell.execute"),
            _context(tenant_id="tenant-2"),
        ),
        (
            ExecutionTargetReference("target-1", "workspace-2", "shell.execute"),
            _context(permitted_boundary_ids={"workspace-2"}),
        ),
        (
            ExecutionTargetReference("target-1", "workspace-1", "secrets.read"),
            _context(granted_capabilities={"secrets.read"}),
        ),
        (
            ExecutionTargetReference("target-1", "workspace-1", "shell.execute"),
            _context(granted_capabilities=set()),
        ),
        (
            ExecutionTargetReference("target-1", "workspace-1", "shell.execute"),
            _context(
                issued_at=NOW - timedelta(minutes=10),
                expires_at=NOW,
            ),
        ),
    ],
)
async def test_target_denials_are_fail_closed_and_sanitized(
    reference: ExecutionTargetReference, context: OperatorContext
) -> None:
    registry = _registry()
    sensitive_handle = {
        "path": "/private/workspace",
        "command": "dangerous --flag",
        "credentials": "secret",
    }
    target = _target(handle=sensitive_handle)
    registry.register(target.owner, execution_targets=(target,))

    with pytest.raises(ExecutionTargetUnavailableError) as caught:
        await registry.resolve_execution_target(reference, context)

    message = str(caught.value)
    assert message == "execution target is unavailable"
    assert not any(value in message for value in sensitive_handle.values())


@pytest.mark.asyncio
async def test_unentitled_capability_is_denied_even_when_other_gates_allow() -> None:
    registry = _registry()
    target = _target(
        capabilities=frozenset({"shell.execute", "shell.admin"}),
        entitled_capabilities=frozenset({"shell.execute"}),
    )
    registry.register(target.owner, execution_targets=(target,))

    with pytest.raises(ExecutionTargetUnavailableError):
        await registry.resolve_execution_target(
            ExecutionTargetReference("target-1", "workspace-1", "shell.admin"),
            _context(granted_capabilities={"shell.admin"}),
        )


def test_browser_projection_is_closed_authorized_and_never_contains_handle() -> None:
    registry = _registry()
    sensitive_handle = {
        "path": "/private/workspace",
        "command": "build",
        "env": {"TOKEN": "secret"},
        "credentials": "secret",
        "configuration": {"runtime": "private"},
    }
    visible = _target(
        handle=sensitive_handle,
        capabilities=frozenset({"shell.execute", "shell.admin"}),
    )
    hidden_tenant = _target(
        tenant_id="tenant-2", target_id="other", handle=object()
    )
    registry.register(visible.owner, execution_targets=(visible,))
    registry.register(hidden_tenant.owner, execution_targets=(hidden_tenant,))

    projection = registry.project_execution_targets(_context())

    assert projection == (
        {
            "target_id": "target-1",
            "target_kind": "container",
            "display_name": "Build worker",
            "tenant_id": "tenant-1",
            "boundary_id": "workspace-1",
            "capabilities": ["shell.execute"],
        },
    )
    serialized = repr(projection)
    assert not any(key in serialized for key in sensitive_handle)
