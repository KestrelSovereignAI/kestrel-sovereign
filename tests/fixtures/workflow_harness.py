"""Deterministic WorkflowHarness for Phase 1 tests.

The harness gives workflow/gate tests a real SQLite-backed WorkflowStore,
SignalLogStore, SourceRegistry, and SignalDispatcher without reaching for
networked actors. Non-deterministic actor cassette replay lands in a later
chunk; this foundation pins the local deterministic path first.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional

from kestrel_sdk.signals import RedactionPolicy, SignalMode, SourceRegistration

from kestrel_sovereign.features.workflows.runner import WorkflowRunner
from kestrel_sovereign.features.workflows.signing import sign_workflow_spec
from kestrel_sovereign.features.workflows.store import WorkflowStore
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.storage.db import SQLiteBackend


class HarnessAgent:
    def __init__(self, identity: AgentIdentity) -> None:
        self.identity = identity
        self.did = identity.legacy_did
        self.background_tasks: list[asyncio.Task] = []
        self.process_input_calls: list[str] = []

    async def process_input(self, prompt: str):
        self.process_input_calls.append(prompt)
        return "ok"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


class WorkflowHarness:
    """Async context manager for deterministic workflow runner tests."""

    def __init__(self, tmp_path: Path, *, did: str = "did:web:k.example") -> None:
        self.tmp_path = tmp_path
        self.identity = _identity(did)
        self.agent = HarnessAgent(self.identity)
        self.backend: Optional[SQLiteBackend] = None
        self.registry = SourceRegistry()
        self.signal_store: Optional[SignalLogStore] = None
        self.store: Optional[WorkflowStore] = None
        self.dispatcher: Optional[SignalDispatcher] = None
        self.runner: Optional[WorkflowRunner] = None

    async def __aenter__(self) -> "WorkflowHarness":
        backend = SQLiteBackend(str(self.tmp_path / "workflow-harness.db"))
        await backend.connect()
        self.backend = backend
        self.signal_store = SignalLogStore(backend)
        await self.signal_store.initialize()
        self.store = WorkflowStore(backend)
        await self.store.initialize()
        self.dispatcher = SignalDispatcher(
            agent=self.agent,
            registry=self.registry,
            lock_manager=OrderedLockManager(),
            store=self.signal_store,
        )
        self.runner = WorkflowRunner(
            store=self.store,
            dispatcher=self.dispatcher,
            registry=self.registry,
            agent_identity=self.identity,
            public_key_resolver=self.resolve_public_key,
            verification_methods_resolver=self.resolve_verification_methods,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pending = [task for task in self.agent.background_tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self.backend is not None:
            await self.backend.close()

    def resolve_public_key(self, did: str) -> bytes:
        if did != self.identity.legacy_did:
            raise KeyError(did)
        suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
        return suite.serialize_public_key(self.identity.legacy_keypair.public_key)

    def resolve_verification_methods(self, did: str) -> list:
        if did != self.identity.signing_did or not self.identity.is_hybrid:
            raise KeyError(did)
        return list(self.identity.new_verification_methods or [])

    def register_action(
        self,
        name: str,
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> None:
        self.registry.register(
            SourceRegistration(
                name=name,
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=handler,
                log_redaction=RedactionPolicy(summarize=lambda p: "<redacted>"),
            )
        )

    async def put_signed(self, spec):
        if self.store is None:
            raise RuntimeError("WorkflowHarness is not entered")
        signed = sign_workflow_spec(spec, self.identity)
        await self.store.put_definition(signed)
        return signed

    @property
    def parts(self):
        return SimpleNamespace(
            agent=self.agent,
            backend=self.backend,
            dispatcher=self.dispatcher,
            identity=self.identity,
            registry=self.registry,
            runner=self.runner,
            store=self.store,
        )


def _identity(did: str) -> AgentIdentity:
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    return AgentIdentity(
        legacy_did=did,
        legacy_keypair=suite.generate_keypair(),
        legacy_did_document={},
    )


__all__ = ["HarnessAgent", "WorkflowHarness"]
