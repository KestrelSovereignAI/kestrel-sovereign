"""Real boot/publication and SQLite queue components, not fleet acceptance.

Only the small phase body is supplied here; native initialize/READY, resident
ownership, SQL writes and physical retirement are the production components.
Full configured Frinz recovery remains a separate integration gate.
"""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.agent.boot import BootPhase, BootPhaseState
from kestrel_sovereign.agent.runtime_start import MaintenanceBootError
from kestrel_sovereign.features.delivery.feature import DeliveryFeature
from kestrel_sovereign.features.delivery.queue import DeliveryQueue
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features import isolated_runtime
from kestrel_sovereign.features.isolated_runtime import ProxyFeature
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_agent_boot_phases import _make_agent


async def close_agent(agent):
    await agent._shutdown_background_tasks()
    await agent.llm_service.close()
    await agent.llm_service.wait_for_close_completion()


@pytest.mark.asyncio
async def test_maintenance_retirement_joins_without_new_status_publication(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "held-scheduler.db"))

    async def effect(*_args):
        raise AssertionError("held scheduler dispatched business work")

    runner = SchedulerRunner(db, "did:test:held-scheduler", effect)
    await runner._ensure_tables()
    try:
        before = await db.fetchval("SELECT total_changes()")
        await runner.stop(publish_stopped=False)
        assert await db.fetchval("SELECT total_changes()") == before
        assert runner._task is None and runner._worker_task is None
        assert runner._runtime_status_publish_task is None
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("distribution,version", [
    ("kestrel-channel-telegram", "0.1.9"),
    ("kestrel-channel-telegram", "unknown"),
    ("unreviewed-isolated-service", "1.0.0"),
])
async def test_maintenance_refuses_unsupported_child_before_launch(tmp_path, monkeypatch, distribution, version):
    agent = _make_agent(tmp_path)
    agent.defer_agent_readiness_to_host(maintenance=True)
    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature", entry_point="telegram.feature:TelegramFeature",
        distribution=distribution, runtime="isolated-venv", service="kestrel-telegram-service",
    )
    proxy = ProxyFeature(agent, runtime)
    monkeypatch.setattr(isolated_runtime, "_feature_distribution_version", lambda *_args: version)
    try:
        with pytest.raises(MaintenanceBootError, match="maintenance"):
            proxy._assert_child_start_allowed()
    finally:
        await close_agent(agent)


@pytest.mark.asyncio
async def test_maintenance_does_not_resolve_or_claim_a_hosted_route(tmp_path):
    agent = _make_agent(tmp_path)
    agent.defer_agent_readiness_to_host(maintenance=True)

    async def serving_route_resolver(_proxy):
        raise AssertionError("maintenance attempted serving route acquisition")

    agent.hosted_telegram_route_attestation_resolver = serving_route_resolver
    proxy = ProxyFeature(agent, InstalledFeatureRuntime(
        class_name="TelegramFeature", entry_point="telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram", runtime="isolated-venv", service="kestrel-telegram-service",
    ))
    try:
        await proxy._resolve_hosted_telegram_startup_attestation()
        assert proxy._hosted_telegram_route_claim is None
        assert proxy._hosted_telegram_startup_attested is False
    finally:
        await close_agent(agent)


@pytest.mark.asyncio
async def test_maintenance_refuses_bin_override_without_child_distribution_proof(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    agent.defer_agent_readiness_to_host(maintenance=True)
    monkeypatch.setenv("KESTREL_FEATURE_TELEGRAMFEATURE_BIN", "/bin/true")
    monkeypatch.setattr(isolated_runtime, "_feature_distribution_version", lambda *_args: "0.1.10")
    proxy = ProxyFeature(agent, InstalledFeatureRuntime(
        class_name="TelegramFeature", entry_point="telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram", runtime="isolated-venv", service="kestrel-telegram-service",
    ))
    try:
        with pytest.raises(MaintenanceBootError, match="maintenance"):
            proxy._assert_child_start_allowed()
    finally:
        await close_agent(agent)


@pytest.mark.asyncio
async def test_maintenance_child_uses_reserved_native_sdk_handshake_not_route_attestation(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    agent.defer_agent_readiness_to_host(maintenance=True)
    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature", entry_point="telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram", runtime="isolated-venv", service="kestrel-telegram-service",
    )
    proxy = ProxyFeature(agent, runtime)
    monkeypatch.setattr(isolated_runtime, "_feature_distribution_version", lambda *_args: "0.1.10")
    client = None
    try:
        client = proxy._build_client(config={"_kestrel_host_runtime_capabilities": ["forged-owner"]})
        assert client.config["_kestrel_host_runtime_capabilities"] == [
            "channel-inbound-acknowledgement-v1", "kestrel-maintenance-boot-v1",
        ]
        assert client.process is None
    finally:
        if client is not None:
            await client.stop()
        await close_agent(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("maintenance", [False, True])
async def test_original_ready_publication_respects_permanent_maintenance_hold(tmp_path, maintenance):
    agent = _make_agent(tmp_path)
    db = await AsyncDatabase.sqlite(str(tmp_path / "runtime-effects.db"))
    await db.execute("CREATE TABLE effects (value INTEGER)")
    entered = asyncio.Event()
    worker = None

    async def effect():
        await db.execute("INSERT INTO effects VALUES (1)")
        entered.set()
        await asyncio.Event().wait()

    async def phase(_ctx):
        nonlocal worker
        worker = agent._track_runtime_task(effect(), name="maintenance-test-resident")
        await asyncio.sleep(0)
        assert not entered.is_set()

    agent._boot_phases = lambda: [BootPhase("native-publication-probe", phase)]
    try:
        agent.defer_agent_readiness_to_host(maintenance=maintenance)
        await agent.initialize()
        assert agent._boot_state is BootPhaseState.READY
        assert agent.maintenance_boot is maintenance
        if maintenance:
            await asyncio.sleep(0)
            assert not agent._runtime_publication_ready.is_set()
            assert await db.fetchval("SELECT count(*) FROM effects") == 0
            assert not entered.is_set()
            with pytest.raises(RuntimeError, match="maintenance"):
                await agent.complete_deferred_agent_readiness()
            # A constitutional transition is not permission to start workers.
            agent._safe_mode = False
            await agent.initialize()  # native READY idempotence
            assert not agent._runtime_publication_ready.is_set()
        else:
            await asyncio.wait_for(entered.wait(), 3)
            assert agent._runtime_publication_ready.is_set()
            assert await db.fetchval("SELECT count(*) FROM effects") == 1
    finally:
        await close_agent(agent)
        assert worker is None or worker.done()
        assert not agent._background_tasks
        await db.close()


@pytest.mark.asyncio
async def test_maintenance_requires_preboot_claim_and_cannot_be_downgraded(tmp_path):
    agent = _make_agent(tmp_path)
    try:
        agent.defer_agent_readiness_to_host(maintenance=True)
        agent.defer_agent_readiness_to_host()  # an ordinary manager cannot release it
        assert agent.maintenance_boot is True
        with pytest.raises(AttributeError):
            agent.maintenance_boot = False
        agent._boot_phases = lambda: []
        await agent.initialize()
        with pytest.raises(RuntimeError, match="before.*initialize"):
            agent.defer_agent_readiness_to_host(maintenance=True)
    finally:
        await close_agent(agent)


@pytest.mark.asyncio
async def test_maintenance_blocks_public_turns_and_both_ready_hook_surfaces(tmp_path):
    agent = _make_agent(tmp_path)
    effects = []

    async def ready(_agent):
        effects.append("ready")

    feature = SimpleNamespace(on_agent_ready=ready)
    try:
        agent.defer_agent_readiness_to_host(maintenance=True)
        agent.features["LateFeature"] = feature
        agent._safe_mode = False  # process policy is independent of SafeMode
        with pytest.raises(MaintenanceBootError):
            await agent.process_input("must not run")
        with pytest.raises(MaintenanceBootError):
            async for _ in agent.process_input_streaming("must not run"):
                pytest.fail("maintenance generation yielded business output")
        with pytest.raises(MaintenanceBootError):
            await agent._notify_agent_ready_hooks()
        await agent._notify_feature_runtime_ready(feature)
        with pytest.raises(MaintenanceBootError):
            await agent._start_a2a_cognition_runtime()
        assert effects == []
    finally:
        await close_agent(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
async def test_maintenance_policy_rejects_non_boolean_configuration(tmp_path, invalid):
    agent = _make_agent(tmp_path)
    try:
        with pytest.raises(TypeError, match="boolean"):
            agent.defer_agent_readiness_to_host(maintenance=invalid)
        assert agent.maintenance_boot is False
    finally:
        await close_agent(agent)


@pytest.mark.asyncio
async def test_unconfigured_empty_registry_is_not_maintenance_repair_proof(tmp_path):
    agent = _make_agent(tmp_path)
    try:
        agent.defer_agent_readiness_to_host(maintenance=True)
        agent._boot_phases = lambda: []  # component probe, not full-boot acceptance
        await agent.initialize()
        registry = agent._ensure_feature_contribution_runtime()
        assert registry.validate_active_integrity() is True
        assert agent.verify_feature_lifecycle_integrity() is False, "empty registry certified maintenance repair"
    finally:
        await close_agent(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("inventory", [set(), [], {"not a feature"}, {1}])
async def test_invalid_maintenance_inventory_does_not_latch_policy(tmp_path, inventory):
    agent = _make_agent(tmp_path)
    try:
        with pytest.raises(ValueError, match="expected_features"):
            agent.defer_agent_readiness_to_host(maintenance=True, expected_features=inventory)
        assert agent.maintenance_boot is False
    finally:
        await close_agent(agent)


@pytest.mark.asyncio
async def test_maintenance_inventory_cannot_be_replaced_or_used_without_hold(tmp_path):
    agent = _make_agent(tmp_path)
    try:
        with pytest.raises(ValueError, match="maintenance"):
            agent.defer_agent_readiness_to_host(expected_features={"MemoryFeature"})
        agent.defer_agent_readiness_to_host(maintenance=True, expected_features={"MemoryFeature"})
        with pytest.raises(RuntimeError, match="cannot be replaced"):
            agent.defer_agent_readiness_to_host(maintenance=True, expected_features={"DeliveryFeature"})
    finally:
        await close_agent(agent)


@pytest.mark.asyncio
async def test_prepared_delivery_queue_does_not_reclaim_legacy_rows(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "delivery-maintenance.db"))
    queue = DeliveryQueue(db, "did:test:held-delivery")
    try:
        await queue.start(polling=False)
        entry = await queue.enqueue("webhook", "https://invalid.example", {"text": "fixture"})
        # Explicit legacy-row fixture, not a claim that a native effect ran.
        await db.execute("UPDATE delivery_queue SET status='in_flight' WHERE id=?", (entry,))
        await queue.start(polling=False)
        assert await db.fetchval("SELECT status FROM delivery_queue WHERE id=?", (entry,)) == "in_flight"
        assert queue._task is None and not queue._running
    finally:
        await queue.stop()
        await db.close()


@pytest.mark.asyncio
async def test_actual_feature_startup_holds_delivery_and_scheduler_side_effects(tmp_path):
    agent = _make_agent(tmp_path)
    db = await AsyncDatabase.sqlite(str(tmp_path / "feature-maintenance.db"))
    agent.storage = agent._raw_storage = SimpleNamespace(db=db)
    delivery, scheduler = DeliveryFeature(agent), SchedulerFeature(agent)
    try:
        agent.defer_agent_readiness_to_host(maintenance=True)
        await delivery.initialize()
        await scheduler.initialize()
        await scheduler.post_all_features_loaded(agent)
        assert delivery._queue is not None
        assert delivery._queue._task is None
        assert scheduler._runner is not None
        assert scheduler._runner._task is None
        assert await db.fetchval("SELECT count(*) FROM scheduled_tasks") == 0
    finally:
        try:
            if hasattr(delivery, "_queue"):
                await delivery.shutdown()
        finally:
            try:
                if hasattr(scheduler, "_runner"):
                    await scheduler.shutdown()
            finally:
                try:
                    await close_agent(agent)
                finally:
                    await db.close()
