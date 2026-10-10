"""Integration-level tests for ``KestrelAgent.initialize()`` as a state machine.

These drive the REAL ``initialize()`` boundary (with the same storage /
memory / task-manager / feature doubles the existing ``TestInitialize`` suite
uses) and assert the #2522 boot-state-machine contract end to end:

* the public phase order IS the documented dependency sequence;
* a clean boot reaches ``READY`` and is idempotent;
* an injected failure at each phase boundary rolls back every resource the
  earlier phases opened (connection close / task-manager close / signal-source
  unregister / memory shutdown) and lands in the terminal ``FAILED`` state;
* a second ``initialize()`` after a failure is refused with
  ``AgentBootError`` — it never runs readiness on partial state;
* a boot cancelled mid-phase still unwinds every acquired resource;
* both the SQLite and the shared-pool PostgreSQL storage paths are exercised.
"""

import asyncio
import contextlib
import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.agent.boot import (
    AgentBootError,
    BootContext,
    BootPhase,
    BootPhaseState,
)
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.agent import custody as custody_module
from kestrel_sovereign.features.base import Feature as _SovereignFeature
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm import codex_app_server
from kestrel_sovereign.llm import service as llm_service_module
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from kestrel_sovereign.spawn.authority_registry import SpawnAuthorityRegistry
from kestrel_sovereign.spawn.mandate import SpawnMandate
from kestrel_sovereign.signals import DurableSignalStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import NodeSwapResult


# Phase method names in boot order — the injected-failure matrix patches one
# of these to raise so the earlier phases run their real bodies first.
PHASE_METHODS = [
    "_boot_phase_storage_privacy",
    "_boot_phase_host_authority_preflight",
    "_boot_phase_a2a_observability_signals",
    "_boot_phase_providers_payer_sync",
    "_boot_phase_identity_constitution_features",
    "_boot_phase_memory_bootstrap_context",
    "_boot_phase_periodic_services_readiness",
    "_boot_phase_host_authority_deadline",
]

PHASE_NAMES = [
    "storage_privacy",
    "host_authority_preflight",
    "a2a_observability_signals",
    "providers_payer_sync",
    "identity_constitution_features",
    "memory_bootstrap_context",
    "periodic_services_readiness",
    "host_authority_deadline",
]


@pytest.mark.asyncio
async def test_hosted_ephemeral_deadline_cancels_remainder_of_active_boot(tmp_path):
    """A valid preflight cannot leave later boot phases running past expiry."""

    agent = _make_agent(tmp_path)
    agent.storage = None
    agent._persisted_spawn_mandate = SpawnMandate(
        parent_did="did:test:parent",
        child_did=agent.agent_id,
        ttl_seconds=60,
        created_at=datetime.now(timezone.utc).isoformat(),
        parent_signature="verified-by-host",
    )
    agent._host_authority_preflight = AsyncMock()
    slow_phase_stopped = asyncio.Event()

    async def slow_phase(_ctx):
        try:
            await asyncio.Event().wait()
        finally:
            slow_phase_stopped.set()

    agent._boot_phases = lambda: [
        BootPhase(
            "host_authority_preflight",
            agent._boot_phase_host_authority_preflight,
        ),
        BootPhase("slow_active_boot", slow_phase),
        BootPhase(
            "host_authority_deadline",
            agent._boot_phase_host_authority_deadline,
        ),
    ]

    with patch(
        "kestrel_sovereign.kestrel_agent.remaining_spawn_ttl_seconds",
        return_value=0.05,
    ), pytest.raises(RuntimeError, match="expired during active agent boot"):
        await asyncio.wait_for(agent.initialize(), timeout=1)

    assert slow_phase_stopped.is_set()
    assert agent._boot_state is BootPhaseState.FAILED


@pytest.mark.asyncio
async def test_hosted_ephemeral_deadline_does_not_cancel_loader_after_boot(tmp_path):
    """Post-boot expiry is an admission failure, not caller cancellation."""

    agent = _make_agent(tmp_path)
    agent.storage = None
    agent._persisted_spawn_mandate = SpawnMandate(
        parent_did="did:test:parent",
        child_did=agent.agent_id,
        ttl_seconds=60,
        created_at=datetime.now(timezone.utc).isoformat(),
        parent_signature="verified-by-host",
    )
    agent._host_authority_preflight = AsyncMock()
    agent._boot_phases = lambda: [
        BootPhase(
            "host_authority_preflight",
            agent._boot_phase_host_authority_preflight,
        ),
        BootPhase(
            "host_authority_deadline",
            agent._boot_phase_host_authority_deadline,
        ),
    ]
    boot_completed = asyncio.Event()
    release_loader = asyncio.Event()

    async def loader():
        await agent.initialize()
        boot_completed.set()
        await release_loader.wait()
        return "loader survived post-boot expiry"

    with patch(
        "kestrel_sovereign.kestrel_agent.remaining_spawn_ttl_seconds",
        return_value=0.05,
    ):
        loader_task = asyncio.create_task(loader())
        await asyncio.wait_for(boot_completed.wait(), timeout=1)
        await asyncio.sleep(0.1)
        assert not loader_task.done()
        release_loader.set()
        assert await asyncio.wait_for(loader_task, timeout=1) == (
            "loader survived post-boot expiry"
        )


@pytest.mark.asyncio
async def test_signed_child_refuses_direct_boot_without_host_authority_verifier(
    tmp_path,
):
    """A durable signed child cannot become a standalone ungoverned root."""

    agent = _make_agent(tmp_path)
    agent.storage = None
    agent._persisted_spawn_mandate = SpawnMandate(
        parent_did="did:test:live-parent",
        child_did=agent.agent_id,
        ttl_seconds=3600,
        created_at=datetime.now(timezone.utc).isoformat(),
        parent_signature="requires-live-host-verification",
    )
    agent._host_authority_preflight = None

    with pytest.raises(RuntimeError, match="without a host authority verifier"):
        await agent._boot_phase_host_authority_preflight(BootContext())


@pytest.mark.asyncio
async def test_host_witness_refuses_direct_boot_after_local_receipt_loss(tmp_path):
    """Deleting child-owned lineage cannot promote a spawned DID to root."""

    storage_path = (
        tmp_path / "agent_data" / "WitnessedChild" / "kestrel_prime.db"
    )
    with patch(
        "kestrel_sovereign.llm.service.LLMService._load_from_disk_cache",
        return_value=False,
    ):
        agent = KestrelAgent(
            did="did:test:boot",
            storage_path=str(storage_path),
            db_backend="sqlite",
            sync_enabled=True,
        )
    agent.storage = None
    agent._persisted_spawn_mandate = None
    agent._host_authority_preflight = None
    mandate = SpawnMandate(
        parent_did="did:test:live-parent",
        child_did=agent.agent_id,
        ttl_seconds=3600,
        parent_signature="durable-host-witness",
    )
    # Every AgentManager places a child at <manager-base>/agent_data/<name>.
    # Direct boot must therefore recover the producing manager's witness rail,
    # not the private manager this child could create for its own descendants.
    SpawnAuthorityRegistry(tmp_path).record_active(
        child_name="WitnessedChild",
        child_did=agent.agent_id,
        mandate=mandate,
        config=LocalAgentConfig(data_dir="agent_data/WitnessedChild", port=8802),
    )

    with pytest.raises(RuntimeError, match="host spawn witness"):
        await agent._boot_phase_host_authority_preflight(BootContext())


@pytest.mark.asyncio
async def test_pending_spawn_authority_refuses_direct_boot_by_data_slot(tmp_path):
    """A pre-inception reservation still denies boot after the child DB appears."""

    child_name = "PendingDirectChild"
    storage_path = tmp_path / "agent_data" / child_name / "kestrel_prime.db"
    storage_path.parent.mkdir(parents=True)
    storage_path.touch()
    with patch(
        "kestrel_sovereign.llm.service.LLMService._load_from_disk_cache",
        return_value=False,
    ):
        agent = KestrelAgent(
            did="did:test:pending-direct-child",
            storage_path=str(storage_path),
            db_backend="sqlite",
            sync_enabled=True,
        )
    agent.storage = None
    agent._persisted_spawn_mandate = None
    agent._host_authority_preflight = None
    SpawnAuthorityRegistry(tmp_path).reserve_pending(
        child_name=child_name,
        parent_did="did:test:pending-direct-parent",
        mandate=SpawnMandate(parent_did="did:test:pending-direct-parent"),
        config=LocalAgentConfig(
            data_dir=f"agent_data/{child_name}",
            port=8802,
        ),
    )

    with pytest.raises(RuntimeError, match="pending spawn authority"):
        await agent._boot_phase_host_authority_preflight(BootContext())


@pytest.mark.asyncio
async def test_host_witness_refuses_replacement_did_direct_boot_by_data_slot(
    tmp_path,
):
    """Replacing the DID in an active host-owned slot cannot create a new root."""

    child_name = "ReplacedDirectChild"
    storage_path = tmp_path / "agent_data" / child_name / "kestrel_prime.db"
    storage_path.parent.mkdir(parents=True)
    storage_path.touch()
    with patch(
        "kestrel_sovereign.llm.service.LLMService._load_from_disk_cache",
        return_value=False,
    ):
        agent = KestrelAgent(
            did="did:test:replacement-direct-child",
            storage_path=str(storage_path),
            db_backend="sqlite",
            sync_enabled=True,
        )
    agent.storage = None
    agent._persisted_spawn_mandate = None
    agent._host_authority_preflight = None
    original_did = "did:test:original-direct-child"
    SpawnAuthorityRegistry(tmp_path).record_active(
        child_name=child_name,
        child_did=original_did,
        mandate=SpawnMandate(
            parent_did="did:test:replacement-direct-parent",
            child_did=original_did,
            parent_signature="durable-host-witness",
        ),
        config=LocalAgentConfig(
            data_dir=f"agent_data/{child_name}",
            port=8802,
        ),
    )

    with pytest.raises(RuntimeError, match="host spawn witness"):
        await agent._boot_phase_host_authority_preflight(BootContext())


def _durable_backend_double() -> MagicMock:
    """Provide the transactional backend contract used during signal boot."""
    backend = MagicMock()
    backend.backend_type = "sqlite"
    backend.execute_script = AsyncMock()
    backend.execute = AsyncMock(return_value=1)

    async def fetch_one(query, params=()):
        if "FROM sqlite_master" in query and "COLLATE NOCASE" in query:
            return (
                "index",
                DurableSignalStore.SOURCE_SEQUENCE_SCOPE_INDEX,
                DurableSignalStore.EVENTS,
            )
        return None

    backend.fetch_one = AsyncMock(side_effect=fetch_one)

    async def fetch_all(query, params=()):
        if query.strip() == "PRAGMA table_info(durable_signal_events)":
            # Signal boot's normal path now trusts the same catalog evidence
            # as a real freshly-created SQLite ledger. Keep this production-
            # wiring double honest instead of making it resemble a partially
            # migrated database whose history would need scanning.
            return [
                (0, "caller_identity", "TEXT", 0, None, 0),
                (1, "source_sequence", "BIGINT", 1, None, 0),
            ]
        if "FROM sqlite_master" in query and "type = 'trigger'" in query:
            return [
                (name, "durable_signal_events", ddl)
                for name, ddl in DurableSignalStore.SOURCE_SEQUENCE_GUARDS
            ] + [
                (name, "durable_signal_source_sequences", ddl)
                for name, ddl in DurableSignalStore.SOURCE_SEQUENCE_COUNTER_FENCES
            ]
        if query.startswith("PRAGMA index_list"):
            return [
                (
                    0,
                    DurableSignalStore.SOURCE_SEQUENCE_SCOPE_INDEX,
                    1,
                    "c",
                    0,
                )
            ]
        if query.startswith("PRAGMA index_xinfo"):
            return [
                (0, 0, "agent_id", 0, "BINARY", 1),
                (1, 1, "source", 0, "BINARY", 1),
                (2, 2, "source_sequence", 0, "BINARY", 1),
                (3, -1, None, 0, "BINARY", 0),
            ]
        return []

    backend.fetch_all = AsyncMock(side_effect=fetch_all)
    backend.fetch_val = AsyncMock(return_value=None)

    @contextlib.asynccontextmanager
    async def transaction(*, immediate: bool = False):
        yield

    backend.transaction = transaction
    return backend


@contextlib.contextmanager
def _boot_mocks(real_task_manager: bool = False):
    """Patch the heavy boot collaborators; yield handles for leak assertions.

    Mirrors the doubles the existing ``TestInitialize`` tests rely on: real
    ``initialize()`` runs, but storage / memory / task-manager are mocks and
    ambient remote-service configuration is cleared so the boot phases neither
    cold-restore from remote storage nor mint live provider credentials.
    ``close`` /
    ``shutdown`` are ``AsyncMock``s so a rollback's teardown calls are
    observable. Constructor-time LLM disk-cache isolation lives in
    ``_make_agent`` because construction precedes this context manager.

    With ``real_task_manager`` the agent builds its real ``TaskManager`` over
    real SQLite stores, and ``task_manager`` is ``None``.
    """
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage, patch(
        "kestrel_sovereign.kestrel_agent.discover_features", return_value=[]
    ) as discover_features, patch("kestrel_sovereign.kestrel_agent.verify_mandatory_feature_set"), patch(
        "kestrel_sovereign.kestrel_agent.MemorySystem"
    ) as MockMemorySystem, (
        contextlib.nullcontext()
        if real_task_manager
        else patch("kestrel_sovereign.kestrel_agent.TaskManager")
    ) as MockTaskManager, patch.dict(
        os.environ,
        {
            "GCS_BACKUP_BUCKET": "",
            "LIGHTHOUSE_API_KEY": "",
            "OPENROUTER_MANAGEMENT_API_KEY": "",
            "SOVEREIGN_IPFS_URL": "",
        },
    ):
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.add_node = AsyncMock()
        storage.compare_and_swap_node = AsyncMock(return_value=NodeSwapResult.SWAPPED)
        storage.db = MagicMock()
        storage.close = AsyncMock()
        storage._backend = _durable_backend_double()
        MockStorage.return_value = storage

        memory = AsyncMock()
        memory.initialize = AsyncMock()
        memory.retriever = MagicMock()
        memory.consolidator = MagicMock()
        memory.shutdown = AsyncMock()
        MockMemorySystem.return_value = memory

        task_manager = None
        if not real_task_manager:
            task_manager = AsyncMock()
            task_manager.initialize = AsyncMock()
            task_manager.register_agent = MagicMock()
            # unregister_agent is synchronous on the real TaskManager; make the
            # mock sync too so feature teardown doesn't leave an un-awaited
            # coroutine.
            task_manager.unregister_agent = MagicMock()
            task_manager.close = AsyncMock()
            MockTaskManager.return_value = task_manager

        yield SimpleNamespace(
            storage=storage,
            memory=memory,
            task_manager=task_manager,
            discover_features=discover_features,
        )


async def _cleanup(agent: KestrelAgent) -> None:
    """Best-effort teardown of anything a mocked boot left open.

    The mocked ``TaskManager.close`` is a no-op, so the real SQLite
    observability backend opened in phase 2 must be closed explicitly (same as
    the existing feature-init suite does).
    """
    with contextlib.suppress(Exception):
        await agent.shutdown()
    obs = getattr(agent, "observability_store", None)
    backend = getattr(obs, "backend", None)
    if backend is not None:
        with contextlib.suppress(Exception):
            await backend.close()


def _make_agent(tmp_path) -> KestrelAgent:
    # Keep the real default LLMService used by the phase-6 readiness check, but
    # prevent its constructor (which runs before _boot_mocks) from reading the
    # operator's process-wide on-disk model catalog. Model discovery itself is
    # lazy and is not called by these boot phases.
    with patch(
        "kestrel_sovereign.llm.service.LLMService._load_from_disk_cache",
        return_value=False,
    ) as load_disk_cache:
        agent = KestrelAgent(
            did="did:test:boot",
            storage_path=str(tmp_path / "boot.db"),
            db_backend="sqlite",
            sync_enabled=True,
        )
    load_disk_cache.assert_called_once_with()
    return agent


async def _wait_for_boot_phase_start(
    agent: KestrelAgent,
    boot_task: asyncio.Task,
    started: asyncio.Event,
    phase_name: str,
    *,
    guard_seconds: float = 10.0,
) -> None:
    """Wait for an instrumented phase with actionable deadlock diagnostics.

    The timeout is a deadlock guard over locally mocked boot work, not a
    performance budget over network I/O. If the phase is not reached, report
    the boot state and committed phase journal and cancel the boot task before
    failing so the test cannot leak an in-progress initializer.
    """
    started_task = asyncio.create_task(started.wait())
    try:
        done, _pending = await asyncio.wait(
            {started_task, boot_task},
            timeout=guard_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if started_task in done:
            return

        ctx = agent._boot_context
        state = agent._boot_state.value
        committed = list(ctx.committed_phases) if ctx is not None else []
        if boot_task in done:
            if boot_task.cancelled():
                outcome = "was cancelled"
            else:
                error = boot_task.exception()
                outcome = f"raised {error!r}" if error is not None else "completed"
            pytest.fail(
                f"boot {outcome} before entering phase {phase_name!r}; "
                f"state={state}, committed={committed}"
            )

        boot_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await boot_task
        pytest.fail(
            f"boot did not enter phase {phase_name!r} within the "
            f"{guard_seconds:g}s deadlock guard; "
            f"state={state}, committed={committed}"
        )
    finally:
        if not started_task.done():
            started_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await started_task


# ---------------------------------------------------------------------------
# Phase-order contract
# ---------------------------------------------------------------------------


def test_boot_phase_order_is_the_documented_dependency_sequence(tmp_path):
    agent = _make_agent(tmp_path)
    phases = agent._boot_phases()
    assert [p.name for p in phases] == PHASE_NAMES
    # The identity phase declares its durable retained resource explicitly.
    identity = next(p for p in phases if p.name == "identity_constitution_features")
    assert identity.retained  # non-empty: the durable identity graph node


# ---------------------------------------------------------------------------
# Clean boot — READY + idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_boot_reaches_ready(tmp_path):
    agent = _make_agent(tmp_path)
    started_when_reconciled = None

    async def capture_reconciliation_order():
        nonlocal started_when_reconciled
        started_when_reconciled = set(
            agent.dispatcher._started_durable_cognition_consumers
        )

    agent.reconcile_a2a_cognition_wakes = AsyncMock(
        side_effect=capture_reconciliation_order
    )
    try:
        with _boot_mocks():
            await agent.initialize()
        assert agent._boot_state is BootPhaseState.READY
        from kestrel_sovereign.signals.sources.workflow_rescue import SOURCE_NAMES

        # The Workflows built-in is registrable without Talon or any other
        # domain feature: core hosts its six provider-neutral source contracts.
        assert all(name in agent.signal_registry for name in SOURCE_NAMES)
        assert "a2a.peer_stop" in agent.signal_registry
        # Every wait wake the reconciler can build has a registered source:
        # without ``wait.replay`` a replay would be dropped as unknown and
        # locked away undelivered (#3390).
        assert "wait.complete" in agent.signal_registry
        assert "wait.replay" in agent.signal_registry
        from kestrel_sovereign.signals.sources.a2a import (
            DURABLE_COGNITION_CONSUMER_ID as A2A_COMPLETE_CONSUMER,
        )
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            DURABLE_COGNITION_CONSUMER_ID as A2A_SUBMITTED_CONSUMER,
        )

        assert {
            A2A_COMPLETE_CONSUMER,
            A2A_SUBMITTED_CONSUMER,
        } <= agent.dispatcher._started_durable_cognition_consumers
        agent.reconcile_a2a_cognition_wakes.assert_awaited_once_with()
        assert started_when_reconciled == set()
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# The serving record stays until every resource's release is confirmed (#3522)
#
# A server started without ``kestrel start`` is found by the guards only
# through its serving record, so the record must outlive every resource the
# agent may still hold. Each resource enters the agent's custody when it is
# acquired and leaves only when its release is reported by an owner in
# ``TRUTHFUL_CLOSE_OWNERS``. That list holds only ``task_manager`` (#3558) and
# ``llm_service`` (#3559) until the other owners' closes stop swallowing
# failures (#3560), so for now the record stays until the process exits.
# ---------------------------------------------------------------------------


def _guard_holder(home):
    """What the reanchor and ``update --no-restart`` guards read for the agent."""
    from kestrel_sovereign import cli

    return cli._agent_holder(home, "boot", LocalAgentConfig(data_dir=".", port=8801))


def _assert_guards_report_this_process(home) -> None:
    holder = _guard_holder(home)
    assert holder is not None, "the guards report the agent stopped"
    assert holder.verified
    assert f"PID {os.getpid()} serves it" in holder.evidence


class _CustodyFeature(_SovereignFeature):
    """A loaded feature whose teardown a test can make fail."""

    tool_name = "custody_feature"
    tool_description = "a feature whose teardown a test controls"
    #: A failing teardown reports ``RETAINED`` instead of raising.
    reports_failure = False

    def __init__(self, agent, fails=False):
        super().__init__(agent)
        self.fails = fails
        self.teardowns = 0

    async def initialize(self):
        return None

    async def shutdown(self):
        self.teardowns += 1
        if self.fails and not self.reports_failure:
            raise RuntimeError(f"{type(self).__name__} cleanup failed")
        await super().shutdown()
        if self.fails:
            return custody_module.ReleaseOutcome.RETAINED
        return None


class _FailingCustodyFeature(_CustodyFeature):
    tool_name = "failing_custody_feature"

    def __init__(self, agent):
        super().__init__(agent, fails=True)


class _RetainingCustodyFeature(_FailingCustodyFeature):
    tool_name = "retaining_custody_feature"
    reports_failure = True


#: Every owner a boot under ``_boot_holding_every_resource`` acquires a
#: resource from. A new kind of resource fails
#: ``test_these_tests_cover_every_owner_a_boot_acquires`` until it is added
#: here, and then joins every sweep below.
_BOOT_OWNERS = (
    "background_tasks",
    "feature",
    "heartbeat_runner",
    "llm_service",
    "memory_system",
    "resume_monitor",
    "salvage_worker",
    "signal_dispatcher",
    "storage",
    "sync_service",
    "task_manager",
)


@contextlib.contextmanager
def _boot_holding_every_resource(
    features=(_CustodyFeature,), real_task_manager: bool = False
):
    """``_boot_mocks`` plus the optional resources a boot can acquire.

    A sync worker, the heartbeat runner, and the given features, so that the
    custody sweeps below cover them too.
    """
    from kestrel_sovereign.heartbeat import HeartbeatConfig

    sync_service = MagicMock()
    sync_service.has_work = True
    sync_service.is_running = True
    sync_service.add_remote_target = MagicMock(return_value=True)
    sync_service.start = AsyncMock()
    sync_service.stop = AsyncMock()
    sync_service.force_snapshot = AsyncMock()
    with _boot_mocks(real_task_manager) as mocks, patch.dict(
        os.environ, {"GCS_BACKUP_BUCKET": "unit-test-bucket"}
    ), patch(
        "kestrel_sovereign.storage.sync.service.SyncService",
        return_value=sync_service,
    ), patch(
        "kestrel_sovereign.heartbeat.HeartbeatConfig.from_config",
        return_value=HeartbeatConfig(enabled=True),
    ), patch(
        "kestrel_sovereign.kestrel_agent.discover_features",
        side_effect=lambda agent, **_kw: [cls(agent) for cls in features],
    ):
        yield SimpleNamespace(**vars(mocks), sync_service=sync_service)


def _trust(monkeypatch, owners) -> None:
    """Treat ``owners`` as reporting a failed release truthfully."""
    monkeypatch.setattr(custody_module, "TRUTHFUL_CLOSE_OWNERS", frozenset(owners))


def _owners_held(agent) -> set:
    custody = agent._resource_custody()
    return {custody._held[name] for name in custody.held}


async def _stop(agent) -> None:
    await asyncio.wait_for(agent.shutdown(), timeout=30)


async def _fail_last_phase(_ctx):
    raise RuntimeError("injected after every resource was acquired")


@pytest.mark.asyncio
async def test_these_tests_cover_every_owner_a_boot_acquires(tmp_path):
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource():
            await agent.initialize()
            assert _owners_held(agent) == set(_BOOT_OWNERS)
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_booted_agent_records_that_this_process_serves_it(tmp_path):
    """Guards find the agent however it was launched (#3522).

    A server started without ``kestrel start`` has no PID file, so the agent
    records itself as boot begins. Only the task manager's and the LLM
    service's releases can be confirmed yet, so stopping the agent keeps the
    record, which goes stale when the process exits.
    """
    from kestrel_sovereign.multi_agent.liveness import serving_holder

    agent = _make_agent(tmp_path)
    assert serving_holder(tmp_path) is None
    try:
        with _boot_holding_every_resource():
            await agent.initialize()
            _assert_guards_report_this_process(tmp_path)

            await _stop(agent)

        # Every step ran and reported its resource released, but only the
        # task manager and the LLM service are trusted to report a failure,
        # so the rest stay held.
        assert _owners_held(agent) == set(_BOOT_OWNERS) - {
            "task_manager",
            "llm_service",
        }
        _assert_guards_report_this_process(tmp_path)
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_failed_boot_records_before_reading_and_keeps_its_record(tmp_path):
    from kestrel_sovereign.multi_agent.liveness import serving_holder

    agent = _make_agent(tmp_path)
    seen_during_boot = []

    async def fail(_ctx):
        seen_during_boot.append(serving_holder(tmp_path))
        raise RuntimeError("injected after storage")

    try:
        with _boot_mocks():
            with patch.object(agent, PHASE_METHODS[1], fail):
                with pytest.raises(RuntimeError, match="injected after storage"):
                    await agent.initialize()

        [holder] = seen_during_boot
        assert holder is not None, "recorded before boot reads anything"
        assert {"storage", "background_tasks"} <= set(
            agent._resource_custody().held
        )
        assert "serving record" in agent._boot_context.retained_resources
        _assert_guards_report_this_process(tmp_path)
    finally:
        await _cleanup(agent)


def _one_failing_phase(agent):
    """A boot of one phase: it records, acquires ``probe``, and fails."""

    async def body(ctx):
        agent._record_serving()

        async def release_probe():
            return custody_module.ReleaseOutcome.RELEASED

        ctx.on_rollback("probe", release_probe)
        raise RuntimeError("injected after acquiring the probe")

    return [BootPhase("probe", body)]


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted", [True, False], ids=["truthful", "unconverted"])
async def test_a_failed_boot_removes_its_record_only_once_its_rollback_is_confirmed(
    tmp_path, monkeypatch, trusted
):
    """The gate the boot failure path uses, without the rest of a boot."""
    _trust(monkeypatch, {"probe"} if trusted else ())
    agent = _make_agent(tmp_path)
    try:
        with patch.object(agent, "_boot_phases", lambda: _one_failing_phase(agent)):
            with pytest.raises(RuntimeError, match="acquiring the probe"):
                await agent.initialize()

        retained = agent._boot_context.retained_resources
        if trusted:
            assert agent._resource_custody().held == ()
            assert retained == []
            assert _guard_holder(tmp_path) is None
        else:
            assert agent._resource_custody().held == ("probe",)
            assert retained == ["probe", "serving record"]
            _assert_guards_report_this_process(tmp_path)
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_stopping_releases_the_record_once_every_owner_is_truthful(
    tmp_path, monkeypatch
):
    """The all-succeed case: every owner trusted, every release confirmed."""
    _trust(monkeypatch, _BOOT_OWNERS)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource():
            await agent.initialize()
            await _stop(agent)

        assert agent._resource_custody().held == ()
        assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_rollback_releases_the_record_once_every_owner_is_truthful(
    tmp_path, monkeypatch
):
    """The rollback releases what boot opened. The LLM service has no
    rollback step: stopping the agent after its failed boot closes it, and
    only then does the record go.
    """
    _trust(monkeypatch, _BOOT_OWNERS)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource():
            with patch.object(agent, PHASE_METHODS[-1], _fail_last_phase):
                with pytest.raises(RuntimeError, match="every resource"):
                    await agent.initialize()

            assert agent._resource_custody().held == ("llm_service",)
            assert "serving record" in agent._boot_context.retained_resources
            _assert_guards_report_this_process(tmp_path)

            await _stop(agent)

        assert agent._resource_custody().held == ()
        assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", _BOOT_OWNERS)
async def test_one_unconverted_owner_keeps_the_record_when_the_agent_stops(
    tmp_path, monkeypatch, owner
):
    """Its release step succeeds, but nothing trusts it to report a failure."""
    _trust(monkeypatch, set(_BOOT_OWNERS) - {owner})
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource():
            await agent.initialize()
            await _stop(agent)

            assert _owners_held(agent) == {owner}
            _assert_guards_report_this_process(tmp_path)
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", _BOOT_OWNERS)
async def test_one_unconverted_owner_keeps_the_record_through_a_rollback(
    tmp_path, monkeypatch, owner
):
    _trust(monkeypatch, set(_BOOT_OWNERS) - {owner})
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource():
            with patch.object(agent, PHASE_METHODS[-1], _fail_last_phase):
                with pytest.raises(RuntimeError, match="every resource"):
                    await agent.initialize()

            # The LLM service has no rollback step; stopping the agent closes it.
            assert _owners_held(agent) == {owner, "llm_service"}
            assert "serving record" in agent._boot_context.retained_resources
            _assert_guards_report_this_process(tmp_path)

            await _stop(agent)
            assert _owners_held(agent) == {owner}
            _assert_guards_report_this_process(tmp_path)
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# The real TaskManager closes truthfully (#3558)
#
# Its close raises when a store fails to close, and the agent keeps the
# manager so a later close retries that store. These boot the agent's real
# TaskManager over real SQLite stores. Every other owner is trusted here, and
# the task manager only by the real allow-list, so its close alone decides
# whether custody empties and the serving record goes.
# ---------------------------------------------------------------------------


def _trust_every_owner_but_the_task_manager(monkeypatch) -> None:
    _trust(
        monkeypatch,
        (set(_BOOT_OWNERS) - {"task_manager"}) | custody_module.TRUTHFUL_CLOSE_OWNERS,
    )


def _fail_store_close(manager: TaskManager):
    """Make the session store's close fail, leaving its connection open."""
    return patch.object(
        manager.session_service.backend,
        "close",
        side_effect=OSError("injected store close failure"),
    )


@pytest.mark.asyncio
async def test_a_failed_task_manager_close_keeps_custody_until_a_retry_closes_it(
    tmp_path, monkeypatch
):
    _trust_every_owner_but_the_task_manager(monkeypatch)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource(real_task_manager=True):
            await agent.initialize()
            manager = agent.task_manager
            assert isinstance(manager, TaskManager)
            store = manager.session_service.backend

            with _fail_store_close(manager):
                await _stop(agent)

            assert store.is_connected, "the store did not close"
            assert agent.task_manager is manager, "kept for a retry"
            assert _owners_held(agent) == {"task_manager"}
            _assert_guards_report_this_process(tmp_path)

            await _stop(agent)

            assert not store.is_connected
            assert agent._resource_custody().held == ()
            assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_failed_task_manager_rollback_keeps_the_manager_for_shutdown(
    tmp_path, monkeypatch
):
    _trust_every_owner_but_the_task_manager(monkeypatch)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource(real_task_manager=True):
            with contextlib.ExitStack() as failing:

                async def fail_with_a_store_that_cannot_close(_ctx):
                    failing.enter_context(_fail_store_close(agent.task_manager))
                    raise RuntimeError("injected after every resource was acquired")

                with patch.object(
                    agent, PHASE_METHODS[-1], fail_with_a_store_that_cannot_close
                ):
                    with pytest.raises(RuntimeError, match="every resource"):
                        await agent.initialize()

                manager = agent.task_manager
                assert isinstance(manager, TaskManager), "kept for shutdown"
                store = manager.session_service.backend
                assert store.is_connected, "the store did not close"

            # The LLM service has no rollback step; stopping the agent closes it.
            assert _owners_held(agent) == {"task_manager", "llm_service"}
            assert "serving record" in agent._boot_context.retained_resources
            _assert_guards_report_this_process(tmp_path)

            await _stop(agent)

            assert not store.is_connected
            assert agent._resource_custody().held == ()
            assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# The real LLMService closes truthfully (#3559)
#
# Its close raises when an adapter or client did not close, and keeps that
# handle so a later close retries it. These add a route to the agent's real
# LLMService whose real Codex adapter owns an app-server child process. Every
# other owner is trusted here, and the LLM service only by the real
# allow-list, so its close alone decides whether custody empties and the
# serving record goes.
# ---------------------------------------------------------------------------

#: An app-server that keeps running after its stdin closes, until killed.
_APP_SERVER_IGNORING_EOF = "import sys, time; sys.stdin.read(); time.sleep(600)"


def _trust_every_owner_but_the_llm_service(monkeypatch) -> None:
    _trust(
        monkeypatch,
        (set(_BOOT_OWNERS) - {"llm_service"}) | custody_module.TRUTHFUL_CLOSE_OWNERS,
    )


@contextlib.asynccontextmanager
async def _codex_route(agent):
    """Give the agent's LLM service a Codex route owning a live app-server."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _APP_SERVER_IGNORING_EOF,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        client = CodexAppServerClient(binary="codex-under-test")
        client._proc = proc
        adapter = CodexAdapter()
        adapter._client = client
        agent.llm_service.providers.append(
            {
                "name": "openai:plan",
                "vendor": "openai",
                "route": "plan",
                "adapter": adapter,
                "client": None,
                "model": "auto",
            }
        )
        yield SimpleNamespace(adapter=adapter, client=client, proc=proc)
    finally:
        if proc.returncode is None:
            # Past any kill a test patched onto the instance.
            asyncio.subprocess.Process.kill(proc)
            await proc.wait()


@pytest.mark.asyncio
async def test_a_failed_llm_adapter_close_keeps_custody_until_a_retry_closes_it(
    tmp_path, monkeypatch
):
    _trust_every_owner_but_the_llm_service(monkeypatch)
    monkeypatch.setattr(codex_app_server, "CODEX_APP_SERVER_EXIT_GRACE_S", 0.05)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource():
            await agent.initialize()
            async with _codex_route(agent) as route:
                real_kill = route.proc.kill

                def refuse_kill():
                    raise PermissionError("injected app-server kill failure")

                monkeypatch.setattr(route.proc, "kill", refuse_kill)
                await _stop(agent)

                assert route.proc.returncode is None, "the app-server still runs"
                assert route.adapter._client is route.client, "kept for a retry"
                assert route.client._proc is route.proc
                assert _owners_held(agent) == {"llm_service"}
                _assert_guards_report_this_process(tmp_path)

                monkeypatch.setattr(route.proc, "kill", real_kill)
                await _stop(agent)

                assert route.proc.returncode is not None
                assert route.adapter._client is None
                assert agent._resource_custody().held == ()
                assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_timed_out_llm_adapter_close_keeps_custody_until_a_retry_finishes_it(
    tmp_path, monkeypatch
):
    """The retry waits for the close the timeout left running, not a new one."""
    _trust_every_owner_but_the_llm_service(monkeypatch)
    monkeypatch.setattr(codex_app_server, "CODEX_APP_SERVER_EXIT_GRACE_S", 0.05)
    stops = []
    stop_may_proceed = asyncio.Event()
    real_stop = CodexAppServerClient._stop_process

    async def slow_stop(proc):
        stops.append(proc)
        await stop_may_proceed.wait()
        await real_stop(proc)

    monkeypatch.setattr(CodexAppServerClient, "_stop_process", staticmethod(slow_stop))
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource():
            await agent.initialize()
            async with _codex_route(agent) as route:
                monkeypatch.setattr(llm_service_module, "CLIENT_CLOSE_TIMEOUT", 0.05)
                await _stop(agent)

                assert route.proc.returncode is None, "the app-server still runs"
                assert route.adapter._client is route.client
                assert _owners_held(agent) == {"llm_service"}
                _assert_guards_report_this_process(tmp_path)

                stop_may_proceed.set()
                monkeypatch.setattr(llm_service_module, "CLIENT_CLOSE_TIMEOUT", 10.0)
                await _stop(agent)

                assert stops == [route.proc], "one close, waited for twice"
                assert route.proc.returncode is not None
                assert route.adapter._client is None
                assert agent._resource_custody().held == ()
                assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_failed_google_async_transport_close_keeps_custody_until_a_retry(
    tmp_path, monkeypatch
):
    """``Client.close()`` leaves the transport the Google adapter calls through.

    A real google-genai client: its synchronous transport closes, its
    asynchronous one fails to, and the agent's custody and serving record
    stay until a later shutdown closes it.
    """
    from google import genai

    from kestrel_sovereign.llm.google_adapter import GoogleAdapter

    _trust_every_owner_but_the_llm_service(monkeypatch)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource():
            await agent.initialize()
            client = genai.Client(api_key="test-key-never-sent")
            sync = client._api_client._httpx_client
            async_transport = client._api_client._async_httpx_client
            real_aclose = async_transport.aclose
            failures = [OSError("injected async transport close failure")]

            async def aclose():
                if failures:
                    raise failures.pop(0)
                await real_aclose()

            monkeypatch.setattr(async_transport, "aclose", aclose)
            agent.llm_service.providers.append(
                {
                    "name": "google:api",
                    "vendor": "google",
                    "route": "api",
                    "adapter": GoogleAdapter(),
                    "client": client,
                    "model": "gemini-test",
                }
            )

            await _stop(agent)

            assert sync.is_closed
            assert not async_transport.is_closed, "the async transport is open"
            assert _owners_held(agent) == {"llm_service"}
            _assert_guards_report_this_process(tmp_path)

            await _stop(agent)

            assert async_transport.is_closed
            assert agent._resource_custody().held == ()
            assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


_RELEASE_STEPS = {
    "memory_system": lambda mocks: mocks.memory.shutdown,
    "storage": lambda mocks: mocks.storage.close,
    "sync_service": lambda mocks: mocks.sync_service.stop,
    "task_manager": lambda mocks: mocks.task_manager.close,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", sorted(_RELEASE_STEPS))
async def test_a_truthful_owner_whose_release_raises_keeps_the_record(
    tmp_path, monkeypatch, owner
):
    _trust(monkeypatch, _BOOT_OWNERS)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource() as mocks:
            await agent.initialize()
            _RELEASE_STEPS[owner](mocks).side_effect = RuntimeError(
                "injected release failure"
            )
            await _stop(agent)

            assert _owners_held(agent) == {owner}
            _assert_guards_report_this_process(tmp_path)
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["stop", "rollback"])
@pytest.mark.parametrize("owner", sorted(_RELEASE_STEPS))
async def test_a_truthful_owner_whose_close_reports_retained_keeps_the_record(
    tmp_path, monkeypatch, owner, path
):
    """A close may report a failed release without raising; that is kept."""
    _trust(monkeypatch, _BOOT_OWNERS)
    retained = custody_module.ReleaseOutcome.RETAINED
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource() as mocks:
            if path == "rollback":
                _RELEASE_STEPS[owner](mocks).return_value = retained
                with patch.object(agent, PHASE_METHODS[-1], _fail_last_phase):
                    with pytest.raises(RuntimeError, match="every resource"):
                        await agent.initialize()
                assert _owners_held(agent) == {owner, "llm_service"}
            else:
                await agent.initialize()
                _RELEASE_STEPS[owner](mocks).return_value = retained
            await _stop(agent)

            assert _owners_held(agent) == {owner}
            _assert_guards_report_this_process(tmp_path)
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_truthful_feature_whose_teardown_reports_retained_keeps_the_record(
    tmp_path, monkeypatch
):
    _trust(monkeypatch, _BOOT_OWNERS)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource((_RetainingCustodyFeature,)):
            await agent.initialize()
            await _stop(agent)

            assert agent._resource_custody().held == (
                "feature:_RetainingCustodyFeature",
            )
            _assert_guards_report_this_process(tmp_path)
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_hangs", [True, False], ids=["abandoned", "flushed"])
async def test_an_abandoned_sync_snapshot_keeps_the_sync_service_held(
    tmp_path, monkeypatch, snapshot_hangs
):
    """Work an abandoned snapshot started may outlive the worker's stop."""
    from kestrel_sovereign import kestrel_agent

    _trust(monkeypatch, _BOOT_OWNERS)
    release = asyncio.Event()

    async def snapshot():
        if snapshot_hangs:
            await release.wait()

    agent = _make_agent(tmp_path)
    sync_service = MagicMock()
    sync_service.is_running = True
    sync_service.force_snapshot = snapshot
    sync_service.stop = AsyncMock()
    agent._sync_service = sync_service
    agent.storage = AsyncMock()
    custody = agent._resource_custody()
    custody.acquire("sync_service")
    custody.acquire("storage")
    try:
        with patch.object(kestrel_agent, "KESTREL_SHUTDOWN_TAIL_MIN_STEP_S", 0.05):
            await asyncio.wait_for(agent._run_durable_shutdown_tail(0.5), timeout=10)

        sync_service.stop.assert_awaited_once()
        assert custody.held == (("sync_service",) if snapshot_hangs else ())
    finally:
        release.set()
        await _cleanup(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["stop", "rollback"])
@pytest.mark.parametrize(
    "reported",
    [False, custody_module.ReleaseOutcome.RETAINED],
    ids=["fenced", "reports-retained"],
)
async def test_a_truthful_dispatcher_whose_release_is_unfinished_stays_held(
    tmp_path, monkeypatch, path, reported
):
    """``shutdown_durable_delivery()`` returns False while cognition fences it.

    A rollback also keeps the storage that unfinished release still uses open
    and held (#3522).
    """
    from kestrel_sovereign.signals import SignalDispatcher

    _trust(monkeypatch, _BOOT_OWNERS)
    release_cognition = asyncio.Event()

    async def wait_for_cognition(self):
        await release_cognition.wait()

    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource() as mocks:
            if path == "rollback":
                with patch.object(
                    SignalDispatcher,
                    "shutdown_durable_delivery",
                    AsyncMock(return_value=reported),
                ), patch.object(
                    SignalDispatcher,
                    "wait_for_durable_shutdown_release",
                    wait_for_cognition,
                ), patch.object(agent, PHASE_METHODS[-1], _fail_last_phase):
                    with pytest.raises(RuntimeError, match="every resource"):
                        await agent.initialize()
                assert agent.dispatcher is not None
                mocks.storage.close.assert_not_awaited()
                assert _owners_held(agent) == {
                    "signal_dispatcher",
                    "storage",
                    "llm_service",
                }
            else:
                await agent.initialize()
                with patch.object(
                    SignalDispatcher,
                    "shutdown_durable_delivery",
                    AsyncMock(return_value=reported),
                ):
                    await _stop(agent)
                assert _owners_held(agent) == {"signal_dispatcher"}
            _assert_guards_report_this_process(tmp_path)
    finally:
        release_cognition.set()
        await _cleanup(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reported",
    [custody_module.ReleaseOutcome.RETAINED, None],
    ids=["reports-retained", "returns-nothing"],
)
async def test_a_storage_preclose_that_reports_retained_keeps_storage_held(
    tmp_path, monkeypatch, reported
):
    _trust(monkeypatch, _BOOT_OWNERS)
    agent = _make_agent(tmp_path)
    storage = AsyncMock()
    storage.dispose_cached_sqla_factory = AsyncMock(return_value=reported)
    agent.storage = storage
    custody = agent._resource_custody()
    custody.acquire("storage")
    try:
        await asyncio.wait_for(agent._run_durable_shutdown_tail(0.5), timeout=10)

        storage.dispose_cached_sqla_factory.assert_awaited_once()
        storage.close.assert_awaited_once()
        assert custody.held == (("storage",) if reported is not None else ())
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("close_fails", [True, False], ids=["close-fails", "closed"])
async def test_a_standalone_hold_context_is_held_until_its_close_is_confirmed(
    tmp_path, monkeypatch, close_fails
):
    """``python -m kestrel_sovereign.main`` hands the agent its Hold context."""
    _trust(monkeypatch, {*_BOOT_OWNERS, custody_module.STANDALONE_HOLD_CONTEXT})
    close = AsyncMock(
        side_effect=RuntimeError("injected Hold close failure") if close_fails else None
    )
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource(), patch(
            "kestrel_sovereign.hold.close_bound_host_context", close
        ):
            await agent.initialize()
            agent._standalone_hold_context = SimpleNamespace()
            await _stop(agent)

            close.assert_awaited_once()
            if close_fails:
                assert agent._resource_custody().held == (
                    custody_module.STANDALONE_HOLD_CONTEXT,
                )
                _assert_guards_report_this_process(tmp_path)
            else:
                assert agent._resource_custody().held == ()
                assert _guard_holder(tmp_path) is None
    finally:
        agent._standalone_hold_context = None
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_truthful_feature_whose_teardown_times_out_keeps_the_record(
    tmp_path, monkeypatch
):
    from kestrel_sovereign import kestrel_agent

    class _Hangs(_CustodyFeature):
        tool_name = "hanging_custody_feature"

        async def shutdown(self):
            await asyncio.Event().wait()

    _trust(monkeypatch, _BOOT_OWNERS)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource((_Hangs,)):
            await agent.initialize()
            with patch.object(
                kestrel_agent, "KESTREL_FEATURE_SHUTDOWN_TIMEOUT_S", 0.2
            ):
                await _stop(agent)

            assert agent._resource_custody().held == ("feature:_Hangs",)
            _assert_guards_report_this_process(tmp_path)
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_cls",
    [_FailingCustodyFeature, _RetainingCustodyFeature],
    ids=["raises", "reports-retained"],
)
async def test_a_failing_feature_rollback_keeps_the_record_and_not_its_neighbour(
    tmp_path, monkeypatch, failing_cls
):
    """Both features are swept; only the failing one stays held.

    It also stays registered, so stopping the agent retries its teardown, and
    once that succeeds the record goes.
    """
    _trust(monkeypatch, _BOOT_OWNERS)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource((_CustodyFeature, failing_cls)):
            with patch.object(agent, PHASE_METHODS[-1], _fail_last_phase):
                with pytest.raises(RuntimeError, match="every resource"):
                    await agent.initialize()

            assert agent._resource_custody().held == (
                "llm_service",
                f"feature:{failing_cls.__name__}",
            )
            assert "features" in agent._boot_context.retained_resources
            [failing] = agent.features.values()
            assert type(failing) is failing_cls
            assert failing.teardowns == 1
            _assert_guards_report_this_process(tmp_path)

            failing.fails = False
            await _stop(agent)

            assert failing.teardowns == 2
            assert agent._resource_custody().held == ()
            assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


def _feature_failing_to_register(stage, failed_teardowns, reports_failure=False):
    """A feature class whose registration fails at ``stage``.

    Its first ``failed_teardowns`` teardowns fail, by raising or, with
    ``reports_failure``, by reporting ``RETAINED``. ``instances`` lists every
    instance made, so a test can reach the one a boot discovered.
    """

    class _FailsToRegister(_CustodyFeature):
        tool_name = "fails_to_register"
        instances: list = []

        def __init__(self, agent):
            super().__init__(agent)
            type(self).instances.append(self)

        async def initialize(self):
            if stage == "initialize":
                raise RuntimeError("injected registration failure")

        async def on_enable(self):
            if stage == "on_enable":
                raise RuntimeError("injected registration failure")
            await super().on_enable()

        async def shutdown(self):
            self.fails = self.teardowns < failed_teardowns
            return await super().shutdown()

    _FailsToRegister.reports_failure = reports_failure
    return _FailsToRegister


def _unreleased(agent) -> list:
    return [feature for _key, feature in agent._unreleased_feature_registry()]


# ``initialize`` fails before the feature reaches ``agent.features``;
# ``on_enable`` fails after it was added and dropped again.
_REGISTRATION_STAGES = ("initialize", "on_enable")


@pytest.mark.asyncio
@pytest.mark.parametrize("reports_failure", [False, True], ids=["raises", "reports-retained"])
@pytest.mark.parametrize("stage", _REGISTRATION_STAGES)
@pytest.mark.parametrize(
    "failed_teardowns",
    [1, 2],
    ids=["rollback-retry-succeeds", "stop-retry-succeeds"],
)
async def test_a_feature_whose_registration_cleanup_fails_is_retried(
    tmp_path, monkeypatch, stage, reports_failure, failed_teardowns
):
    """Its failed cleanup leaves the feature where rollback and stop find it.

    ``_register_feature`` tears down a feature whose registration failed. When
    that teardown fails too, the instance used to be dropped, so neither the
    boot rollback nor a later stop could reach it to retry, and once
    ``feature`` joins the truthful owners its custody could never end.
    """
    _trust(monkeypatch, _BOOT_OWNERS)
    failing_cls = _feature_failing_to_register(stage, failed_teardowns, reports_failure)
    agent = _make_agent(tmp_path)
    resource = f"feature:{failing_cls.__name__}"
    try:
        with _boot_holding_every_resource((_CustodyFeature, failing_cls)):
            with pytest.raises(RuntimeError, match="injected registration failure"):
                await agent.initialize()

            [failing] = failing_cls.instances
            # Once while registering, once more in the boot rollback.
            assert failing.teardowns == 2
            assert failing not in agent.features.values()
            if failed_teardowns == 1:
                assert _unreleased(agent) == []
                assert resource not in agent._resource_custody().held
            else:
                assert _unreleased(agent) == [failing]
                assert resource in agent._resource_custody().held
                assert "features" in agent._boot_context.retained_resources
            _assert_guards_report_this_process(tmp_path)

            await _stop(agent)

            # Stopping retries only a feature whose cleanup has not succeeded.
            assert failing.teardowns == 1 + failed_teardowns
            assert _unreleased(agent) == []
            assert agent._resource_custody().held == ()
            assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", _REGISTRATION_STAGES)
async def test_a_failed_registration_on_a_running_agent_is_retried_when_it_stops(
    tmp_path, monkeypatch, stage
):
    """No boot rollback runs here, so stopping the agent is the only retry."""
    _trust(monkeypatch, _BOOT_OWNERS)
    failing_cls = _feature_failing_to_register(stage, failed_teardowns=1)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource():
            await agent.initialize()
            failing = failing_cls(agent)

            with pytest.raises(RuntimeError, match="injected registration failure"):
                await agent._register_feature(failing)

            assert failing.teardowns == 1
            assert failing not in agent.features.values()
            assert _unreleased(agent) == [failing]
            assert f"feature:{failing_cls.__name__}" in agent._resource_custody().held

            await _stop(agent)

            assert failing.teardowns == 2
            assert _unreleased(agent) == []
            assert agent._resource_custody().held == ()
            assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_feature_whose_unload_fails_is_retried_when_the_agent_stops(
    tmp_path, monkeypatch
):
    """A runtime disable drops the feature, even when its teardown failed."""
    _trust(monkeypatch, _BOOT_OWNERS)
    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource((_FailingCustodyFeature,)):
            await agent.initialize()
            [failing] = agent.features.values()

            with pytest.raises(RuntimeError, match="cleanup failed"):
                await agent._disable_feature(failing.name)

            assert failing.teardowns == 1
            assert agent.features == {}
            assert _unreleased(agent) == [failing]

            failing.fails = False
            await _stop(agent)

            assert failing.teardowns == 2
            assert _unreleased(agent) == []
            assert agent._resource_custody().held == ()
            assert _guard_holder(tmp_path) is None
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_unloading_a_feature_that_is_not_loaded_leaves_its_namesake(tmp_path):
    """An unload drops only the instance it tears down."""
    agent = _make_agent(tmp_path)
    loaded = _CustodyFeature(agent)
    stray = _CustodyFeature(agent)
    agent.features = {loaded.name: loaded}

    await agent._unregister_feature_runtime(stray)

    assert agent.features == {loaded.name: loaded}
    assert stray.teardowns == 1


@pytest.mark.asyncio
async def test_a_truthful_dispatcher_fenced_by_live_cognition_stays_held(
    tmp_path, monkeypatch
):
    """A fenced dispatcher is not released, so neither is the record.

    Its release returns while cognition still owns a delivery lease; the
    continuation releases it, and storage, once that settles, and then the
    record goes.
    """
    from kestrel_sovereign.signals import SignalDispatcher

    _trust(monkeypatch, _BOOT_OWNERS)
    release_cognition = asyncio.Event()
    stop = SignalDispatcher.shutdown_durable_delivery
    calls = 0

    async def fenced_stop(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Live cognition still owns its lease.
            self._durable_shutdown_owner_fenced = True
            return False
        return await stop(self)

    async def wait_for_cognition(self):
        await release_cognition.wait()
        self._durable_shutdown_owner_fenced = False

    agent = _make_agent(tmp_path)
    try:
        with _boot_holding_every_resource(), patch.object(
            SignalDispatcher, "shutdown_durable_delivery", fenced_stop
        ), patch.object(
            SignalDispatcher,
            "wait_for_durable_shutdown_release",
            wait_for_cognition,
        ):
            await agent.initialize()
            await _stop(agent)

            assert {"signal_dispatcher", "storage"} <= set(
                agent._resource_custody().held
            )
            _assert_guards_report_this_process(tmp_path)

            release_cognition.set()
            await asyncio.wait_for(agent.wait_for_shutdown_completion(), timeout=30)
            assert agent._resource_custody().held == ()
            assert _guard_holder(tmp_path) is None
    finally:
        release_cognition.set()
        await _cleanup(agent)


async def _fail_after_the_dispatcher_starts(_ctx):
    raise RuntimeError("injected after dispatcher startup")


@pytest.mark.asyncio
async def test_a_boot_rollback_that_keeps_the_dispatcher_keeps_the_serving_record(
    tmp_path, monkeypatch
):
    """Durable dispatcher teardown fails on both attempts during rollback.

    The rollback keeps the dispatcher and its storage open for a later stop,
    so even with every owner trusted the record stays and the guards report
    the agent running.
    """
    from kestrel_sovereign.signals import SignalDispatcher

    _trust(monkeypatch, _BOOT_OWNERS)
    agent = _make_agent(tmp_path)
    try:
        with _boot_mocks():
            with patch.object(
                agent, PHASE_METHODS[3], _fail_after_the_dispatcher_starts
            ), patch.object(
                SignalDispatcher,
                "shutdown_durable_delivery",
                AsyncMock(side_effect=RuntimeError("injected teardown failure")),
            ):
                with pytest.raises(RuntimeError, match="after dispatcher startup"):
                    await agent.initialize()

            assert agent._boot_state is BootPhaseState.FAILED
            assert agent.dispatcher is not None
            assert agent._raw_storage is not None
            held = set(agent._resource_custody().held)
            assert {"storage", "signal_dispatcher"} <= held, held
            retained = agent._boot_context.retained_resources
            assert {"storage", "signal_dispatcher", "serving record"} <= set(
                retained
            )
            _assert_guards_report_this_process(tmp_path)
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_boot_rollback_keeps_storage_open_beneath_a_fenced_dispatcher(
    tmp_path, monkeypatch
):
    """Rollback keeps a dispatcher fenced by live cognition, and its storage.

    The dispatcher's release returns ``False`` while cognition still owns a
    delivery (#3522). Rollback keeps its handle, so it does not close the
    storage that owner still uses, and the guards keep reporting the agent
    served. Once cognition settles the continuation closes storage, and a
    later stop releases the record.
    """
    from kestrel_sovereign.signals import SignalDispatcher

    _trust(monkeypatch, _BOOT_OWNERS)
    release_cognition = asyncio.Event()
    stop = SignalDispatcher.shutdown_durable_delivery
    calls = 0

    async def fenced_stop(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Live cognition still owns its lease.
            self._durable_shutdown_owner_fenced = True
            return False
        return await stop(self)

    async def wait_for_cognition(self):
        await release_cognition.wait()
        self._durable_shutdown_owner_fenced = False

    agent = _make_agent(tmp_path)
    try:
        with _boot_mocks() as mocks, patch.object(
            agent, PHASE_METHODS[3], _fail_after_the_dispatcher_starts
        ), patch.object(
            SignalDispatcher, "shutdown_durable_delivery", fenced_stop
        ), patch.object(
            SignalDispatcher,
            "wait_for_durable_shutdown_release",
            wait_for_cognition,
        ):
            with pytest.raises(RuntimeError, match="after dispatcher startup"):
                await agent.initialize()

            assert agent._boot_state is BootPhaseState.FAILED
            assert agent.dispatcher is not None
            assert agent._raw_storage is mocks.storage
            mocks.storage.close.assert_not_awaited()
            assert {"storage", "signal_dispatcher"} <= set(
                agent._resource_custody().held
            )
            _assert_guards_report_this_process(tmp_path)

            release_cognition.set()
            await asyncio.wait_for(agent.wait_for_shutdown_completion(), timeout=30)
            mocks.storage.close.assert_awaited_once()

            await _stop(agent)
            mocks.storage.close.assert_awaited_once()
            assert agent._resource_custody().held == ()
            assert _guard_holder(tmp_path) is None
    finally:
        release_cognition.set()
        await _cleanup(agent)


def test_a_symlinked_store_records_in_the_registered_data_dir(tmp_path):
    """A guard looks in the data directory, not where its database points."""
    data_dir = tmp_path / "agent_data" / "emma"
    data_dir.mkdir(parents=True)
    volume = tmp_path / "volume"
    volume.mkdir()
    (volume / "emma.db").touch()
    (data_dir / "kestrel_prime.db").symlink_to(volume / "emma.db")
    agent = _make_agent(tmp_path)
    agent.storage_path = str(data_dir / "kestrel_prime.db")

    assert agent._serving_data_dir() == data_dir.resolve()


def test_an_agent_without_an_on_disk_store_records_nothing(tmp_path):
    agent = _make_agent(tmp_path)
    agent.storage_path = ":memory:"

    assert agent._serving_data_dir() is None
    agent._record_serving()
    assert agent._serving_record is None


def test_a_record_that_cannot_be_removed_is_kept_for_a_retry(tmp_path):
    from kestrel_sovereign.multi_agent.liveness import ServingRecord

    agent = _make_agent(tmp_path)
    agent._record_serving()
    record = agent._serving_record
    assert record is not None

    with patch.object(ServingRecord, "release", side_effect=OSError("read-only")):
        assert agent._release_serving_record() is False
    assert agent._serving_record is record
    _assert_guards_report_this_process(tmp_path)

    assert agent._release_serving_record() is True
    assert agent._serving_record is None
    assert _guard_holder(tmp_path) is None


@pytest.mark.asyncio
async def test_boot_records_its_embedding_profile_after_loading_the_config(tmp_path):
    """Offline tools compare their resolution with the recorded profile (#3420).

    Recorded before the persisted embedding config is applied, it would name a
    profile the agent does not search.
    """
    agent = _make_agent(tmp_path)
    calls = []

    def spy(name):
        original = getattr(agent, name)

        async def recorded(*args, **kwargs):
            calls.append(name)
            return await original(*args, **kwargs)

        return recorded

    order = [
        "_load_embedding_route",
        "_load_route_embedding_models",
        "record_active_embedding_profile",
    ]
    for name in order:
        setattr(agent, name, spy(name))
    try:
        with _boot_mocks():
            await agent.initialize()
        assert agent._boot_state is BootPhaseState.READY
        assert calls == order
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_second_initialize_when_ready_is_a_noop(tmp_path):
    agent = _make_agent(tmp_path)
    try:
        with _boot_mocks():
            await agent.initialize()
        assert agent._boot_state is BootPhaseState.READY
        # Called again with NO mocks in scope: it must short-circuit on READY
        # before touching AsyncStorage, so this neither raises nor re-runs.
        await agent.initialize()
        assert agent._boot_state is BootPhaseState.READY
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_host_authority_preflight_refuses_before_feature_discovery(tmp_path):
    """A bad configured receipt cannot start even one feature worker."""

    from kestrel_sovereign.spawn.mandate import SpawnMandate

    agent = _make_agent(tmp_path)
    mandate = SpawnMandate(
        parent_did="did:test:parent",
        child_did=agent.did,
        parent_signature="00",
    )
    observed = []

    def refuse_unverified_receipt(candidate):
        assert candidate is agent
        observed.append(agent._persisted_spawn_mandate)
        raise RuntimeError("invalid persisted authority")

    agent._host_authority_preflight = refuse_unverified_receipt
    try:
        with _boot_mocks() as mocks, patch(
            "kestrel_sovereign.spawn.mandate_reload.read_spawn_mandate",
            new=AsyncMock(return_value=mandate),
        ):
            with pytest.raises(RuntimeError, match="invalid persisted authority"):
                await agent.initialize()

        assert observed == [mandate]
        mocks.discover_features.assert_not_called()
        assert agent._boot_state is BootPhaseState.FAILED
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_resume_callback_reconciles_sidecars_when_audit_persistence_fails(tmp_path):
    """Volatile handoff expiry cannot depend on `system.resumed` persistence."""
    from kestrel_sdk.signals import Status
    from kestrel_sovereign.signals.dispatcher import _TransientDurableHandoff

    agent = _make_agent(tmp_path)
    try:
        with _boot_mocks():
            await agent.initialize()

        dispatcher = agent.dispatcher
        now = datetime.now(timezone.utc)
        dispatcher._transient_durable_handoffs["expired-resume-handoff"] = (
            _TransientDurableHandoff(
                payload={"raw": "must-not-survive-resume"},
                consumer_id="workflow-wait",
                created_at=now - timedelta(minutes=2),
                retention_until=now + timedelta(days=1),
                expires_at=now - timedelta(seconds=1),
                initial_lease_token="live-only-capability",
            )
        )
        dispatcher._durable_store.persist_signal = AsyncMock(
            side_effect=RuntimeError("forced resumed-signal persistence failure")
        )
        original_dispatch_signal = dispatcher.dispatch_signal
        results = []

        async def capture_failed_dispatch(*args, **kwargs):
            result = await original_dispatch_signal(*args, **kwargs)
            results.append(result)
            return result

        dispatcher.dispatch_signal = capture_failed_dispatch

        # `dispatch_signal` encodes this persistence error as Status.FAILED;
        # it does not raise to the resume callback. The sidecar must already
        # be reconciled by the direct callback path.
        await agent.resume_monitor._on_resume(3600.0)

        assert "expired-resume-handoff" not in dispatcher._transient_durable_handoffs
        dispatcher._durable_store.persist_signal.assert_awaited_once()
        assert [result.status for result in results] == [Status.FAILED]
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# Injected failure at each phase boundary — rollback + terminal FAILED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_index", list(range(len(PHASE_METHODS))))
async def test_injected_phase_failure_rolls_back_and_fails_terminally(
    tmp_path, fail_index
):
    agent = _make_agent(tmp_path)
    boom = AsyncMock(side_effect=RuntimeError(f"injected@{PHASE_NAMES[fail_index]}"))
    try:
        with _boot_mocks() as mocks:
            with patch.object(agent, PHASE_METHODS[fail_index], boom):
                with pytest.raises(RuntimeError, match="injected@"):
                    await agent.initialize()

            # Terminal state, regardless of which phase failed.
            assert agent._boot_state is BootPhaseState.FAILED

            # Storage (phase 1) committed for every failure at index >= 1, so
            # its connection must have been closed and the handle dropped.
            if fail_index >= 1:
                mocks.storage.close.assert_awaited()
                assert agent._raw_storage is None
                assert agent.storage is None
                assert agent.privacy_agent is None
            else:
                # Storage phase itself failed before opening anything.
                assert agent._raw_storage is None

            # A2A task manager (phase 3) + core signal sources.
            if fail_index >= 3:
                mocks.task_manager.close.assert_awaited()
                assert agent.task_manager is None
                # Core signal sources were unregistered on rollback.
                assert "a2a.task_complete" not in agent.signal_registry
                assert "a2a.peer_stop" not in agent.signal_registry

            # Memory system (phase 6).
            if fail_index >= 6:
                mocks.memory.shutdown.assert_awaited()
                assert getattr(agent, "memory_system", None) is None

            # A retry over the rolled-back partial state is refused — readiness
            # can never run on it.
            with pytest.raises(AgentBootError, match="previously failed"):
                await agent.initialize()
            assert agent._boot_state is BootPhaseState.FAILED
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_failed_boot_never_started_periodic_services(tmp_path):
    """A failure before the readiness phase leaves no heartbeat/resume runner."""
    agent = _make_agent(tmp_path)
    boom = AsyncMock(side_effect=RuntimeError("injected@memory"))
    try:
        with _boot_mocks():
            with patch.object(
                agent, "_boot_phase_memory_bootstrap_context", boom
            ):
                with pytest.raises(RuntimeError):
                    await agent.initialize()
        # Phase 6 never ran → no periodic services exist.
        assert getattr(agent, "heartbeat_runner", None) is None
        assert getattr(agent, "resume_monitor", None) is None
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_later_boot_failure_tears_down_durable_dispatcher_before_storage(tmp_path):
    """Phase-2 owner liveness cannot outlive a phase-3 boot failure."""
    agent = _make_agent(tmp_path)
    captured = {}

    async def fail_after_dispatcher(_ctx: BootContext) -> None:
        captured["dispatcher"] = agent.dispatcher
        raise RuntimeError("injected after dispatcher initialization")

    try:
        with _boot_mocks() as mocks:
            with patch.object(
                agent, "_boot_phase_providers_payer_sync", fail_after_dispatcher
            ):
                with pytest.raises(RuntimeError, match="after dispatcher"):
                    await agent.initialize()

            dispatcher = captured["dispatcher"]
            assert agent._boot_state is BootPhaseState.FAILED
            assert agent.dispatcher is None
            assert dispatcher._runtime_owner_heartbeat_timer is None
            assert dispatcher._durable_runtime_owner_registered is False
            # The owner release happens before the phase-1 storage close in
            # LIFO rollback order, rather than leaving timer work on a closed
            # database backend.
            release_calls = [
                call
                for call in mocks.storage._backend.execute.await_args_list
                if "stopped_at" in str(call)
            ]
            assert release_calls
            mocks.storage.close.assert_awaited()
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_boot_rollback_stops_owner_registered_at_sqlite_commit_cancellation(
    tmp_path,
):
    """Boot teardown releases an owner whose registration await never returned.

    ``aiosqlite`` can complete ``commit`` on its worker before cancellation is
    raised back into the caller.  Exercise that concrete boundary, then drive
    the real boot rollback seam (rather than only dispatcher shutdown) to
    prove an ambiguous registration cannot leave a live runtime owner behind.
    """
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )
    from kestrel_sovereign.storage.db import SQLiteBackend

    agent = _make_agent(tmp_path)
    backend = SQLiteBackend(str(tmp_path / "boot-owner-commit-boundary.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=SourceRegistry(),
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    # Keep schema setup outside the armed registration transaction.
    await dispatcher._durable_store.initialize()
    connection = backend._connection
    assert connection is not None
    original_commit = connection.commit
    original_register = dispatcher._durable_store.register_runtime_owner
    registration_in_flight = False

    async def cancel_after_committed_owner_registration():
        await original_commit()
        if registration_in_flight:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)

    async def register_with_armed_commit(*args, **kwargs):
        nonlocal registration_in_flight
        registration_in_flight = True
        try:
            return await original_register(*args, **kwargs)
        finally:
            registration_in_flight = False

    try:
        connection.commit = cancel_after_committed_owner_registration
        dispatcher._durable_store.register_runtime_owner = register_with_armed_commit
        with pytest.raises(asyncio.CancelledError):
            await dispatcher.initialize_durable_delivery()

        assert dispatcher._durable_initialized is False
        assert dispatcher._durable_runtime_owner_registered is False
        assert dispatcher._durable_runtime_owner_registration_started is True

        # This is the callback registered before durable initialization's first
        # await. It must release the ambiguous owner before boot storage closes.
        connection.commit = original_commit
        dispatcher._durable_store.register_runtime_owner = original_register
        agent.dispatcher = dispatcher
        await agent._boot_teardown_dispatcher()

        assert agent.dispatcher is None
        owner = await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners "
            "WHERE agent_id = ? AND owner_id = ?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner is not None and owner[0] is not None
        assert dispatcher._durable_runtime_owner_registration_started is False
    finally:
        connection.commit = original_commit
        dispatcher._durable_store.register_runtime_owner = original_register
        if not dispatcher._durable_shutdown:
            await dispatcher.shutdown_durable_delivery()
        await backend.close()


# ---------------------------------------------------------------------------
# Failure DURING an initializer's own await — the resource the initializer
# opened before raising is still torn down (#2522 P1). Teardown is registered
# BEFORE ``initialize()``'s first await, not after it returns, so a
# TaskManager whose 3rd store fails (leaking the first two) or a storage whose
# migration fails mid-connection does not leak.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["storage", "task_manager", "memory"])
async def test_mid_initialize_failure_still_tears_down_the_resource(
    tmp_path, resource
):
    agent = _make_agent(tmp_path)
    try:
        with _boot_mocks() as mocks:
            # The resource's OWN initialize() raises partway (it may have opened
            # a connection / started a worker before raising).
            getattr(mocks, resource).initialize.side_effect = RuntimeError(
                f"injected mid-{resource}-initialize"
            )
            with pytest.raises(RuntimeError, match="injected mid-"):
                await agent.initialize()

            assert agent._boot_state is BootPhaseState.FAILED
            # The teardown registered before the failing await ran on rollback.
            if resource == "storage":
                mocks.storage.close.assert_awaited()
                assert agent._raw_storage is None
            elif resource == "task_manager":
                # Storage (phase 1) committed and the task manager's teardown
                # fired even though its OWN initialize is what raised.
                mocks.storage.close.assert_awaited()
                mocks.task_manager.close.assert_awaited()
                assert agent.task_manager is None
            else:  # memory
                mocks.storage.close.assert_awaited()
                mocks.task_manager.close.assert_awaited()
                mocks.memory.shutdown.assert_awaited()
                assert getattr(agent, "memory_system", None) is None
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# Feature-owned signal sources are unregistered when a LATER phase fails
# (#2522 P2). A feature that successfully registers a dispatcher source and is
# then rolled back must not leave its feature-bound handler in the registry.
# ---------------------------------------------------------------------------


def _fake_source_registration(name: str):
    from kestrel_sdk.signals import (
        RedactionPolicy,
        SignalMode,
        SourceRegistration,
        Trust,
    )

    async def handler(payload):
        return None

    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        trust=Trust.TRUSTED,
        log_redaction=RedactionPolicy(summarize=lambda p: ""),
    )


@pytest.mark.asyncio
async def test_feature_owned_signal_sources_absent_after_rollback(tmp_path):
    from types import SimpleNamespace as _NS

    from kestrel_sovereign.features.base import Feature as _SovereignFeature

    class _SourceRegisteringFeature(_SovereignFeature):
        FAKE_SOURCE = "fake.feature_source"

        tool_name = "fake_source_feature"
        tool_description = "fake source-registering feature"

        def __init__(self, agent):
            super().__init__(agent)
            self.shutdown_calls = 0

        async def initialize(self):
            from kestrel_sovereign.signals import RegistrationPolicy

            # Registered AS THIS FEATURE, so base shutdown releases it.
            self._register_signal_sources(
                _fake_source_registration(self.FAKE_SOURCE),
                RegistrationPolicy.OPTIONAL,
            )

        async def shutdown(self):
            self.shutdown_calls += 1
            await super().shutdown()

        def get_agent_card(self):
            return _NS(name=self.name, skills=[])

    agent = _make_agent(tmp_path)
    feature_ref = {}

    def _discover(a, **_kw):
        feature_ref["feature"] = _SourceRegisteringFeature(a)
        return [feature_ref["feature"]]

    boom = AsyncMock(side_effect=RuntimeError("injected@memory"))
    try:
        with _boot_mocks():
            with patch(
                "kestrel_sovereign.kestrel_agent.discover_features",
                side_effect=_discover,
            ):
                with patch.object(
                    agent, "_boot_phase_memory_bootstrap_context", boom
                ):
                    with pytest.raises(RuntimeError, match="injected@memory"):
                        await agent.initialize()

            assert agent._boot_state is BootPhaseState.FAILED
            feature = feature_ref["feature"]
            # The feature registered its source in phase 4, then phase 5 failed →
            # boot rollback shut the feature down → its source is unregistered.
            assert feature.shutdown_calls >= 1
            assert (
                _SourceRegisteringFeature.FAKE_SOURCE not in agent.signal_registry
            )
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_feature_hooks_and_wait_providers_absent_after_rollback(tmp_path):
    """A LATER phase failure rolls back a feature's hooks AND wait providers —
    not just its ``shutdown()`` (#2522).

    A feature registers a hook (via ``get_hooks()``, auto-registered in
    ``_register_feature``) and a ``Waitable`` provider (in
    ``post_all_features_loaded``). The old boot rollback called only
    ``feature.shutdown()``, so the hook and the ``task:``/``talon:``-style wait
    provider were left registered on a dead agent. Rollback now runs the
    canonical per-feature teardown, so the hook registry AND the wait-provider
    registry are empty afterwards.
    """
    from types import SimpleNamespace as _NS

    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
    from kestrel_sovereign.features.base import Feature as _SovereignFeature

    class _NoopHook(Hook):
        def __init__(self) -> None:
            super().__init__(
                name="fake_feature_hook", events=[HookEvent.SESSION_START]
            )

        async def execute(self, input):  # noqa: A002 - SDK signature
            return HookOutput()

    class _FakeWaitable:
        kind = "fakewait"
        signal = None

        async def poll(self, handle):  # pragma: no cover - never polled
            raise NotImplementedError

    class _HookWaitFeature(_SovereignFeature):
        FAKE_SOURCE = "fake.hookwait_source"

        tool_name = "hook_wait_feature"
        tool_description = "feature registering a hook + wait provider"

        def __init__(self, agent):
            super().__init__(agent)
            self._hook = _NoopHook()

        async def initialize(self):
            from kestrel_sovereign.signals import RegistrationPolicy

            self._register_signal_sources(
                _fake_source_registration(self.FAKE_SOURCE),
                RegistrationPolicy.OPTIONAL,
            )

        def get_hooks(self):
            return [self._hook]

        async def post_all_features_loaded(self, agent):
            # Same canonical call the real task:/talon: features use — records
            # ownership so shutdown()/rollback unregisters it.
            self._register_wait_provider(
                agent.wait_registry, _FakeWaitable(), replace=True
            )

        def get_agent_card(self):
            return _NS(name=self.name, skills=[])

    agent = _make_agent(tmp_path)
    holder: dict = {}

    def _discover(a, **_kw):
        holder["feature"] = _HookWaitFeature(a)
        return [holder["feature"]]

    boom = AsyncMock(side_effect=RuntimeError("injected@memory"))
    try:
        with _boot_mocks():
            with patch(
                "kestrel_sovereign.kestrel_agent.discover_features",
                side_effect=_discover,
            ):
                with patch.object(
                    agent, "_boot_phase_memory_bootstrap_context", boom
                ):
                    with pytest.raises(RuntimeError, match="injected@memory"):
                        await agent.initialize()

            assert agent._boot_state is BootPhaseState.FAILED
            feature = holder["feature"]

            # The hook registered in phase 4 is gone — every event bucket empty.
            assert all(
                not hooks for hooks in agent.hooks_manager._hooks.values()
            ), "feature hook left registered after boot rollback"
            assert feature._hook not in agent.hooks_manager.get_hooks(
                HookEvent.SESSION_START
            )

            # The wait provider registered in post_all_features_loaded is gone.
            assert agent.wait_registry.kinds() == []
            assert agent.wait_registry.get("fakewait") is None

            # And its dispatcher signal source too (base shutdown path).
            assert _HookWaitFeature.FAKE_SOURCE not in agent.signal_registry
            # The feature itself was dropped from the agent.
            assert feature.name not in agent.features
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_feature_source_unregistered_when_its_own_init_fails_after_registering(
    tmp_path,
):
    """A feature that registers a source and THEN raises in initialize() is not
    left in the registry — ``_register_feature`` shuts the failing feature down
    even though it never entered ``self.features`` (#2522 P1 + P2)."""
    from kestrel_sovereign.features.base import Feature as _SovereignFeature

    class _FailAfterRegisterFeature(_SovereignFeature):
        FAKE_SOURCE = "fake.fail_after_register"

        tool_name = "fail_after_register_feature"
        tool_description = "feature that fails after registering a source"

        async def initialize(self):
            from kestrel_sovereign.signals import RegistrationPolicy

            self._register_signal_sources(
                _fake_source_registration(self.FAKE_SOURCE),
                RegistrationPolicy.OPTIONAL,
            )
            raise RuntimeError("feature init boom after registering source")

    agent = _make_agent(tmp_path)

    def _discover(a, **_kw):
        return [_FailAfterRegisterFeature(a)]

    try:
        with _boot_mocks():
            with patch(
                "kestrel_sovereign.kestrel_agent.discover_features",
                side_effect=_discover,
            ):
                with pytest.raises(RuntimeError, match="feature init boom"):
                    await agent.initialize()

            assert agent._boot_state is BootPhaseState.FAILED
            # Non-mandatory feature: its init error propagates and rolls the boot
            # back, but its source must not survive in the registry.
            assert _FailAfterRegisterFeature.FAKE_SOURCE not in agent.signal_registry
            # It never entered self.features (init failed before assignment).
            assert "_FailAfterRegisterFeature" not in {
                type(f).__name__ for f in getattr(agent, "features", {}).values()
            }
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# Cancellation mid-boot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boot_cancelled_mid_phase_unwinds_all_resources(tmp_path):
    """Cancellation rolls back local resources, including an active sync worker."""
    agent = _make_agent(tmp_path)
    started = asyncio.Event()
    sync_service = MagicMock()
    sync_service.has_work = True
    sync_service.is_running = True
    sync_service.add_remote_target = MagicMock(return_value=True)
    sync_service.start = AsyncMock()
    sync_service.stop = AsyncMock()

    async def hang(ctx: BootContext) -> None:
        started.set()
        await asyncio.Event().wait()  # owns nothing new; awaits cancellation

    try:
        with _boot_mocks() as mocks:
            # Exercise sync rollback without constructing a real GCS client or
            # allowing cleanup to upload a snapshot. _boot_mocks deliberately
            # blanks every ambient remote target for all other boot tests.
            with patch.dict(
                os.environ, {"GCS_BACKUP_BUCKET": "unit-test-bucket"}
            ), patch(
                "kestrel_sovereign.storage.sync.service.SyncService",
                return_value=sync_service,
            ), patch.object(
                agent, "_boot_phase_periodic_services_readiness", hang
            ):
                # Hang in the final phase, AFTER storage/a2a/sync/memory
                # committed. The guard below observes that named phase and
                # reports the phase journal if deterministic boot work wedges.
                task = asyncio.create_task(agent.initialize())
                await _wait_for_boot_phase_start(
                    agent, task, started, "periodic_services_readiness"
                )
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            assert agent._boot_state is BootPhaseState.FAILED
            # Every acquired resource released despite cancellation mid-boot.
            mocks.storage.close.assert_awaited()
            mocks.task_manager.close.assert_awaited()
            mocks.memory.shutdown.assert_awaited()
            sync_service.add_remote_target.assert_called_once()
            sync_service.start.assert_awaited_once()
            sync_service.stop.assert_awaited_once()
            assert agent._raw_storage is None
            assert agent.task_manager is None
            assert agent._sync_service is None

            # Retry refused after a cancelled/partial boot.
            with pytest.raises(AgentBootError):
                await agent.initialize()
    finally:
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# Re-entrancy guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_initialize_is_refused(tmp_path):
    agent = _make_agent(tmp_path)
    entered = asyncio.Event()

    async def hang_storage(ctx: BootContext) -> None:
        entered.set()
        await asyncio.Event().wait()  # hold the boot IN_PROGRESS

    first = None
    try:
        with patch.object(agent, "_boot_phase_storage_privacy", hang_storage):
            first = asyncio.create_task(agent.initialize())
            await _wait_for_boot_phase_start(
                agent, first, entered, "storage_privacy"
            )
            assert agent._boot_state is BootPhaseState.IN_PROGRESS
            # A second concurrent call while the first is IN_PROGRESS is refused.
            with pytest.raises(AgentBootError, match="already in progress"):
                await agent.initialize()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        assert agent._boot_state is BootPhaseState.FAILED
    finally:
        if first is not None and not first.done():
            first.cancel()
            with contextlib.suppress(Exception):
                await first
        await _cleanup(agent)


# ---------------------------------------------------------------------------
# Shared-pool PostgreSQL storage path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_advisory", [False, True])
async def test_storage_phase_uses_shared_postgres_pool(tmp_path, shared_advisory):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async

    credentials = await create_kestrel_identity_async(
        str(tmp_path),
        identity_method="did:pkh",
        agent_name="Shared pool semantic authority test",
    )
    pool = MagicMock()
    host_advisory_backend = MagicMock() if shared_advisory else None
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=str(tmp_path / "kestrel_prime.db"),
        db_backend="postgres",
        pg_pool=pool,
        database_url="postgresql://scheduler-test/kestrel",
        shared_postgres_advisory_backend=host_advisory_backend,
        llm_service=MagicMock(),
    )
    assert agent.identity is not None
    ctx = BootContext()
    # ``storage.db`` is a REAL database, not a MagicMock: this agent has an
    # on-disk inception whose birth record lives in a different database, so
    # the phase now reconciles it (#2871) and a duck-typed double would fail on
    # the first await. The runtime database is deliberately a second file so
    # the reconciliation path is the one production takes.
    runtime_db = await AsyncDatabase.sqlite(str(tmp_path / "runtime.db"))
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage, patch(
        "kestrel_sovereign.storage.db.postgres.PostgresBackend"
    ) as MockPGBackend:
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.db = runtime_db
        storage.close = AsyncMock()
        MockStorage.return_value = storage
        pg_backend = MagicMock()
        MockPGBackend.from_pool.return_value = pg_backend

        try:
            await agent._boot_phase_storage_privacy(ctx)
        finally:
            await runtime_db.close()

        # The shared pool was adopted (not a fresh DSN connection).
        MockPGBackend.from_pool.assert_called_once_with(
            pool,
            advisory_dsn=(
                None if shared_advisory else "postgresql://scheduler-test/kestrel"
            ),
            advisory_backend=host_advisory_backend,
        )
        _, kwargs = MockStorage.call_args
        assert kwargs.get("backend") is pg_backend
        capability = kwargs.get("_assertion_tenant_capability")
        assert capability is not None and capability.tenant_id == agent.did
    assert agent._raw_storage is storage
    # Storage teardown was registered for reverse-order rollback.
    assert "storage" in ctx.rollback_labels


@pytest.mark.asyncio
async def test_storage_phase_keeps_pool_recipe_when_database_url_is_ambient(
    monkeypatch,
):
    """An ambient URL must not replace a custom pool's advisory connector."""

    async def custom_connector(*_args, **_kwargs):
        return None

    ssl_context = object()

    class CustomConnectorPool:
        _connect_args = ()
        _connect_kwargs = {"ssl": ssl_context}
        _connect = staticmethod(custom_connector)
        _connection_class = object
        _record_class = object

        def get_max_size(self):
            return 3

    pool = CustomConnectorPool()
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://ambient/kestrel")
    agent = KestrelAgent(
        did="did:test:pg-ambient-url",
        db_backend="postgres",
        pg_pool=pool,
        llm_service=MagicMock(),
    )
    ctx = BootContext()
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage:
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.db = MagicMock()
        storage.close = AsyncMock()
        MockStorage.return_value = storage

        await agent._boot_phase_storage_privacy(ctx)

    backend = MockStorage.call_args.kwargs["backend"]
    assert agent._database_url == "postgresql://ambient/kestrel"
    assert backend._advisory_dsn is None
    assert backend._advisory_connect_args == ()
    assert backend._advisory_connect_kwargs["connect"] is custom_connector
    assert backend._advisory_connect_kwargs["ssl"] is ssl_context


@pytest.mark.asyncio
async def test_storage_phase_supports_pool_only_postgres_embedding():
    """A pool-only embedder retains an independent scheduler gate recipe."""

    class PoolOnlyAsyncpgDouble:
        _connect_args = ("postgresql://pool-only/kestrel",)
        _connect_kwargs = {"server_settings": {"application_name": "host"}}
        _connect = staticmethod(lambda *_args, **_kwargs: None)
        _connection_class = object
        _record_class = object

        def get_max_size(self):
            return 3

    pool = PoolOnlyAsyncpgDouble()
    agent = KestrelAgent(
        did="did:test:pg-pool-only",
        db_backend="postgres",
        pg_pool=pool,
        llm_service=MagicMock(),
    )
    ctx = BootContext()
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage:
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.db = MagicMock()
        storage.close = AsyncMock()
        MockStorage.return_value = storage

        await agent._boot_phase_storage_privacy(ctx)

    backend = MockStorage.call_args.kwargs["backend"]
    assert backend._pool is pool
    assert backend._advisory_connect_args == ("postgresql://pool-only/kestrel",)
    assert backend._advisory_connect_kwargs["server_settings"] == {
        "application_name": "host"
    }
    assert backend._advisory_max_pool_size == 3


@pytest.mark.asyncio
async def test_storage_phase_uses_sqlite_by_default(tmp_path):
    agent = _make_agent(tmp_path)
    ctx = BootContext()
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage:
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.db = MagicMock()
        storage.close = AsyncMock()
        MockStorage.return_value = storage

        await agent._boot_phase_storage_privacy(ctx)

        # SQLite path: first positional arg is the storage path, no backend kw.
        args, kwargs = MockStorage.call_args
        assert args and args[0] == str(tmp_path / "boot.db")
        assert "backend" not in kwargs
    assert "storage" in ctx.rollback_labels


# ---------------------------------------------------------------------------
# A feature's duplicate signal source must not abort boot (issue #2951)
# ---------------------------------------------------------------------------


def _feature_contributing(agent, source_name: str):
    """A fixture feature whose workflow contributes *source_name*.

    The contract deliberately differs from core's registration of the same
    name — which is the real shape: two `fleet_stalled_sweep` registrations
    bound to different callbacks are a genuine mismatch, not a duplicate.
    """
    import dataclasses

    from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature

    feature = SDKFixtureFeature(agent)
    colliding = dataclasses.replace(feature.source, name=source_name)
    feature.workflow_registration = type(feature.workflow_registration)(
        owner=feature.contribution_owner,
        name=feature.workflow_registration.name,
        actor=feature.actor,
        sources=(colliding,),
    )
    return feature


@pytest.mark.asyncio
async def test_one_features_duplicate_source_does_not_abort_boot(tmp_path):
    """The regression: one stale feature took every agent on the host down.

    `kestrel-feature-talon` 0.2.0 contributed a source core had just reclaimed,
    and boot failed for Meridian, Claw, Nellie and Emma alike. The agent must
    now boot, without that feature, and say why.
    """
    from kestrel_sovereign.signals.sources.workflow_rescue import SOURCE_NAMES

    agent = _make_agent(tmp_path)
    feature = _feature_contributing(agent, sorted(SOURCE_NAMES)[0])
    try:
        with _boot_mocks(), patch(
            "kestrel_sovereign.kestrel_agent.discover_features",
            return_value=[feature],
        ):
            await agent.initialize()

        # The agent boots.
        assert agent._boot_state is BootPhaseState.READY
        # The offending feature does not load...
        assert feature not in agent.features.values()
        # ...and the reason is REPORTED, not merely logged.
        reasons = [r.reason for r in agent.rejected_feature_contributions]
        assert len(reasons) == 1
        assert sorted(SOURCE_NAMES)[0] in reasons[0]
        # Core's own registration is untouched.
        assert all(name in agent.signal_registry for name in SOURCE_NAMES)
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_a_rejected_feature_is_absent_from_the_mandatory_readiness_check(tmp_path):
    """A rejected MANDATORY feature must still fail boot.

    Degrading a mandatory feature to "did not load" would turn an identity gap
    into a capability gap — the inversion #2871 warns against. The postcondition
    that catches it is `verify_mandatory_feature_set`, which can only do so if
    the rejected feature is genuinely ABSENT from the mapping it is handed.

    That absence is what this asserts. The raising behaviour of the
    postcondition itself is covered by its own tests; this pins the half that
    this change could break.
    """
    from kestrel_sovereign.signals.sources.workflow_rescue import SOURCE_NAMES

    agent = _make_agent(tmp_path)
    feature = _feature_contributing(agent, sorted(SOURCE_NAMES)[0])
    try:
        with _boot_mocks(), patch(
            "kestrel_sovereign.kestrel_agent.discover_features",
            return_value=[feature],
        ), patch(
            "kestrel_sovereign.kestrel_agent.verify_mandatory_feature_set"
        ) as verify:
            await agent.initialize()

        # It ran at agent readiness...
        stages = [c.kwargs.get("stage") for c in verify.call_args_list]
        assert "agent readiness" in stages
        # ...over a feature set that does NOT contain the rejected feature, so a
        # mandatory one would be counted as missing.
        checked = verify.call_args_list[stages.index("agent readiness")].args[0]
        assert feature not in (
            checked.values() if hasattr(checked, "values") else checked
        )
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_after_post_load", [False, True])
async def test_post_all_features_loaded_barrier_flag(tmp_path, fail_after_post_load):
    """#2474: the scheduler defers every tick until this barrier completes.

    The flag is False while any feature is still in post-load wiring, True by
    the time ready hooks run, and a later boot failure's rollback clears it so
    a torn-down agent never reports its features as loaded.
    """
    from types import SimpleNamespace as _NS

    from kestrel_sovereign.features.base import Feature as _SovereignFeature

    observed: dict = {}

    class _BarrierProbeFeature(_SovereignFeature):
        tool_name = "barrier_probe_feature"
        tool_description = "records the feature-load barrier as seen by hooks"

        async def initialize(self):
            return None

        async def post_all_features_loaded(self, agent):
            observed["post_load"] = agent._post_all_features_loaded_complete

        async def on_agent_ready(self, agent):
            observed["ready"] = agent._post_all_features_loaded_complete

        def get_agent_card(self):
            return _NS(name=self.name, skills=[])

    agent = _make_agent(tmp_path)
    assert agent._post_all_features_loaded_complete is False
    boom = AsyncMock(side_effect=RuntimeError("injected@memory"))
    try:
        with _boot_mocks(), patch(
            "kestrel_sovereign.kestrel_agent.discover_features",
            side_effect=lambda a, **_kw: [_BarrierProbeFeature(a)],
        ):
            if fail_after_post_load:
                with patch.object(
                    agent, "_boot_phase_memory_bootstrap_context", boom
                ):
                    with pytest.raises(RuntimeError, match="injected@memory"):
                        await agent.initialize()
            else:
                await agent.initialize()

        assert observed["post_load"] is False
        if fail_after_post_load:
            assert agent._boot_state is BootPhaseState.FAILED
            assert agent._post_all_features_loaded_complete is False
        else:
            assert observed["ready"] is True
            assert agent._post_all_features_loaded_complete is True
    finally:
        await _cleanup(agent)


@pytest.mark.asyncio
async def test_storage_phase_fails_boot_on_a_malformed_continuation_check(tmp_path, monkeypatch):
    """#3527: the continuation check reads its env when it runs, because the
    server imports it before loading .env; a malformed value must still fail
    the boot rather than a turn."""
    monkeypatch.setenv("KESTREL_CONTINUATION_CHECK", "sometimes")
    agent = _make_agent(tmp_path)
    ctx = BootContext()
    with patch("kestrel_sovereign.kestrel_agent.AsyncStorage") as MockStorage:
        storage = AsyncMock()
        storage.initialize = AsyncMock()
        storage.get_node = AsyncMock(return_value=None)
        storage.db = MagicMock()
        storage.close = AsyncMock()
        MockStorage.return_value = storage

        with pytest.raises(ValueError, match="KESTREL_CONTINUATION_CHECK"):
            await agent._boot_phase_storage_privacy(ctx)
