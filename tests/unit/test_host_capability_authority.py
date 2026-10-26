"""Current host authority cannot be replaced by a serialized context or task copy."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import pytest
from kestrel_sdk.operator import OperatorContext

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.privacy import PrivacyMode, privacy_mode_to_config
from kestrel_sovereign.security.capability_authority import (
    HostCapabilityAccessDenied,
    HostCapabilityAuthority,
)
from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate


class Action(StrEnum):
    OBSERVE = "visual.observe"
    STOP = "visual.stop"


class Holder:
    def __init__(self):
        self.gate = DurablePersistenceGate()
        self.set_mode(PrivacyMode.NORMAL)

    def set_mode(self, mode):
        self.privacy_mode = mode
        self.privacy_config = privacy_mode_to_config(mode)

    def _get_durable_persistence_gate(self):
        return self.gate


class Admission:
    def __init__(self):
        self.now = datetime.now(UTC)
        self.holder = Holder()
        self.gate = AsyncReaderWriterLock()
        self.authority = HostCapabilityAuthority(
            capability="visual-computer", label="visual",
            non_disclosing_actions=frozenset({Action.STOP.value}),
            clock=lambda: self.now,
        )
        self.current = True
        self.targets = []
        self.context = OperatorContext(
            tenant_id="tenant", principal_id="sovereign", acting_agent_id="agent",
            granted_actions=frozenset(a.value for a in Action),
            granted_capabilities=frozenset({"visual-computer"}),
            permitted_boundary_ids=frozenset({"visual-computer"}),
            correlation_id="admission", issued_at=self.now - timedelta(seconds=1),
            expires_at=self.now + timedelta(minutes=1),
        )

    def check(self, context, action, target):
        if not self.current:
            raise PermissionError("private credential diagnostic")
        self.targets.append((context, action, target))

    def bind(self, context=None, check=None):
        return self.authority.bind_authenticated(
            context or self.context, privacy_holder=self.holder,
            check_current=check or self.check, authority_gate=self.gate,
        )

    def operation(self, action=Action.OBSERVE, context=None, **kwargs):
        return self.authority.operation(context or self.context, action,
                                        "visual-computer", **kwargs)


@pytest.mark.asyncio
async def test_exact_original_binding_and_lifetime_required():
    a = Admission()
    with pytest.raises(HostCapabilityAccessDenied):
        async with a.operation():
            pass
    with a.bind():
        with pytest.raises(HostCapabilityAccessDenied):
            async with a.operation(context=replace(a.context)):
                pass
        with pytest.raises(ValueError):
            with a.bind():
                pass
        async with a.operation() as check:
            check()
    with pytest.raises(HostCapabilityAccessDenied):
        check()


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"granted_actions": frozenset()},
    {"granted_capabilities": frozenset({"artifacts"})},
    {"permitted_boundary_ids": frozenset({"another-session"})},
])
async def test_safety_action_still_requires_exact_current_grants(changes):
    a = Admission()
    a.holder.set_mode(PrivacyMode.EPHEMERAL)
    context = replace(a.context, **changes)
    with a.bind(context), pytest.raises(HostCapabilityAccessDenied):
        async with a.operation(Action.STOP, context):
            pass
    assert not a.targets


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.ISOLATED, PrivacyMode.EPHEMERAL,
                                 PrivacyMode.ANONYMOUS, PrivacyMode.DEIDENTIFIED])
async def test_only_explicit_non_disclosing_safety_action_survives_raw_policy(mode):
    a = Admission()
    with a.bind():
        a.holder.set_mode(mode)
        with pytest.raises(HostCapabilityAccessDenied):
            async with a.operation():
                pass
        async with a.operation(Action.STOP) as check:
            check()
        a.current = False
        with pytest.raises(HostCapabilityAccessDenied) as caught:
            async with a.operation(Action.STOP):
                pass
        assert "credential" not in str(caught.value)


@pytest.mark.asyncio
async def test_rechecks_expiry_target_and_current_authority_inside_retained_operation():
    a = Admission()
    target = object()
    with a.bind():
        async with a.operation(target=target) as check:
            assert a.targets[-1] == (a.context, Action.OBSERVE, target)
            a.now = a.context.expires_at
            with pytest.raises(HostCapabilityAccessDenied):
                check()


@pytest.mark.asyncio
async def test_checker_must_raise_or_return_none_and_never_discloses_its_diagnostic():
    a = Admission()
    with a.bind(check=lambda *_: False), pytest.raises(HostCapabilityAccessDenied):
        async with a.operation():
            pass
    a.current = False
    with a.bind(), pytest.raises(HostCapabilityAccessDenied) as caught:
        async with a.operation():
            pass
    assert str(caught.value) == "visual authority is unavailable"


@pytest.mark.asyncio
async def test_same_task_nesting_cannot_give_copied_task_a_retained_lease():
    a = Admission()
    writer_entered = asyncio.Event()

    async def revoke():
        async with a.gate:
            a.current = False
            writer_entered.set()

    async def copied_context():
        async with a.operation():
            pytest.fail("copied task bypassed queued revocation")

    with a.bind():
        async with a.operation():
            writer = asyncio.create_task(revoke())
            await asyncio.sleep(0)
            # Same-task nesting must not deadlock behind its own queued writer.
            async with a.operation() as check:
                check()
            child = asyncio.create_task(copied_context())
            await asyncio.sleep(0)
            assert not child.done()
            assert not writer_entered.is_set()
        await asyncio.wait_for(writer, timeout=1)
        with pytest.raises(HostCapabilityAccessDenied):
            await asyncio.wait_for(child, timeout=1)


@pytest.mark.asyncio
async def test_cancellation_releases_current_operation_and_transition_gates():
    a = Admission()
    admitted = asyncio.Event()

    async def retained():
        async with a.operation():
            admitted.set()
            await asyncio.Event().wait()

    with a.bind():
        child = asyncio.create_task(retained())
        await asyncio.wait_for(admitted.wait(), timeout=1)
        child.cancel()
        with pytest.raises(asyncio.CancelledError):
            await child
        await asyncio.wait_for(a.gate.acquire(), timeout=1)
        a.gate.release()
        async with a.operation():
            pass
