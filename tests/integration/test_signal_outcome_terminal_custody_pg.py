"""Same fixed native signal journal contracts on isolated real PostgreSQL."""
from types import SimpleNamespace

import pytest

from kestrel_sovereign.signals import OrderedLockManager, SignalDispatcher, SignalLogStore, SourceRegistry
from tests.integration.test_execution_authority_postgres import native_pg  # noqa: F401
from tests.unit.test_signals_dispatcher import _FakeAgent
from tests.unit.test_signal_outcome_terminal_custody import (  # noqa: F401
    test_original_native_outcome_persists_after_work_cutoff_without_fresh_work,
    test_unadmitted_or_forged_native_journal_never_grants_storage,
    test_handler_cannot_use_journal_permission_or_replay_after_success,
    test_original_journal_refuses_substituted_identity,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.fixture
async def dispatcher_components(native_pg):
    backend, _, _ = native_pg
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=store)
    try:
        yield SimpleNamespace(backend=backend, store=store, registry=registry, agent=agent, dispatcher=dispatcher)
    finally:
        try:
            await dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            from kestrel_sovereign.execution_custody import is_execution_control_error
            assert is_execution_control_error(error)
