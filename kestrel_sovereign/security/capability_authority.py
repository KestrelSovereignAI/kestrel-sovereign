"""Original host-context bindings shared by private capability services.

An SDK context is a description, not a bearer credential. Trusted host code
binds its exact identity for one lifetime and supplies a synchronous checker
of current authority. Privacy, authority and database leases retain their
existing order. Each capability explicitly identifies non-disclosing safety
or erasure actions; that exemption never bypasses owner/grant checks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from kestrel_sdk.operator import OperatorAuthorizationError, OperatorContext
from kestrel_sdk.resources import ResourceError

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.privacy import PrivacyConfig, PrivacyMode
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate


class HostCapabilityAccessDenied(ResourceError, OperatorAuthorizationError):
    """A private host capability has no current delivery authority."""


CurrentHostAuthority = Callable[[OperatorContext, StrEnum, object | None], None]


@dataclass
class _Binding:
    context: OperatorContext
    privacy_holder: object
    check_current: CurrentHostAuthority
    authority_gate: AsyncReaderWriterLock
    active: bool = True


class HostCapabilityAuthority:
    """Bind exact original contexts and retain current host policy leases."""

    def __init__(
        self, *, capability: str, label: str,
        non_disclosing_actions: frozenset[str] = frozenset(),
        access_denied: type[Exception] = HostCapabilityAccessDenied,
        clock: Callable[[], datetime] | None = None,
    ):
        if not capability or not label:
            raise ValueError("capability and label are required")
        self.capability = capability
        self.label = label
        self.non_disclosing_actions = frozenset(non_disclosing_actions)
        self.access_denied = access_denied
        self.clock = clock or (lambda: datetime.now(UTC))
        self._bindings: dict[int, _Binding] = {}
        self._operations: dict[asyncio.Task, _Binding] = {}

    @contextmanager
    def bind_authenticated(
        self, context: OperatorContext, *, privacy_holder: object,
        check_current: CurrentHostAuthority, authority_gate: AsyncReaderWriterLock,
    ) -> Iterator[OperatorContext]:
        if not isinstance(context, OperatorContext) or not callable(check_current):
            raise TypeError("host-resolved context and current authority checker required")
        if not isinstance(authority_gate, AsyncReaderWriterLock):
            raise TypeError("host authority transition gate required")
        context.require_fresh(self.clock())
        require_durable_persistence_gate(privacy_holder)
        if id(context) in self._bindings:
            raise ValueError(f"{self.label} context already has an active lifetime")
        binding = _Binding(context, privacy_holder, check_current, authority_gate)
        self._bindings[id(context)] = binding
        try:
            yield context
        finally:
            binding.active = False
            self._bindings.pop(id(context), None)

    def _check(self, binding: _Binding, context: OperatorContext,
               action: StrEnum, boundary_id: str, target: object | None) -> None:
        if (not binding.active or binding.context is not context
                or self._bindings.get(id(context)) is not binding):
            raise self.access_denied(f"{self.label} context is not active")
        try:
            context.require_fresh(self.clock())
            context.require_action(action.value)
            context.require_capability(self.capability)
            context.require_boundary(boundary_id)
            if binding.check_current(context, action, target) is not None:
                raise TypeError("host authority checker must return None or raise")
        except Exception:
            # Current-check diagnostics may contain credentials or private data.
            # An unavailable checker must never become an admission fallback.
            raise self.access_denied(f"{self.label} authority is unavailable") from None
        if action.value not in self.non_disclosing_actions:
            holder = binding.privacy_holder
            config = getattr(holder, "privacy_config", None)
            if (getattr(holder, "privacy_mode", None) not in (PrivacyMode.NORMAL, PrivacyMode.PUBLIC)
                    or not isinstance(config, PrivacyConfig)
                    or config.storage != "full" or config.assurance != "none"
                    or config.audit != "optional"):
                raise self.access_denied(f"privacy policy does not permit raw {self.capability}")

    @asynccontextmanager
    async def operation(self, context: OperatorContext, action: StrEnum,
                        boundary_id: str, *, target: object | None = None):
        binding = self._bindings.get(id(context))
        if binding is None or binding.context is not context:
            raise self.access_denied(f"{self.label} context was not resolved by this host")

        def check() -> None:
            self._check(binding, context, action, boundary_id, target)

        check()
        task = asyncio.current_task()
        existing = self._operations.get(task)
        if existing is not None:
            if existing is not binding:
                raise self.access_denied(f"nested {self.label} owner does not match")
            yield check
            return
        gate = require_durable_persistence_gate(binding.privacy_holder)
        async with gate.shared():
            async with binding.authority_gate.read():
                check()
                self._operations[task] = binding
                try:
                    yield check
                finally:
                    self._operations.pop(task, None)
