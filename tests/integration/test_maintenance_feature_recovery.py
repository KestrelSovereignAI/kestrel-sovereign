"""Real incepted identity -> full Core boot -> persisted latch -> restart/exit.

SQLite plus the actual selected bundled/installed Workflow features. Only the
external semantic genesis auditor is a deterministic fixture. No READY,
registry, lifecycle verifier, native receipt, clock or lease is substituted.
This does not claim the complete Frinz inventory or PostgreSQL fleet custody.
"""

import pytest
from kestrel_sdk.payer_policy import PayerKind, PayerPolicy, PayerSpec

from kestrel_sovereign.agent.boot import BootPhaseState
from kestrel_sovereign.agent.constitution import SafeModeCause
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES


async def shutdown_fixture_agent(agent):
    """Use the stable line's public original shutdown/completion boundaries."""
    try:
        await agent.shutdown()
    finally:
        await agent.wait_for_shutdown_completion()


def no_provider_policy():
    host = PayerPolicy.host_env_default()
    return PayerPolicy(**{
        field: PayerSpec(vendor=getattr(host, field).vendor, kind=PayerKind.NONE)
        for field in ("llm", "storage", "compute", "tools", "comms")
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("with_workflows", [False, True])
async def test_real_full_boot_can_prove_and_exit_persisted_feature_restriction(tmp_path, with_workflows):
    if with_workflows:
        pytest.importorskip("kestrel_feature_workflows", reason="optional paired external-feature integration")
    async def genesis_auditor(_prompt):
        return {
            "risk_level": 1,
            "reasoning": "Deterministic external-auditor fixture, not semantic model acceptance.",
        }

    credentials = await create_kestrel_identity_async(
        output_dir=str(tmp_path / "maintenance-identity"),
        agent_name="Native maintenance recovery",
        identity_method="did:pkh",
        genesis_auditor=genesis_auditor,
        genesis_audit_provenance="test:external-model-fixture",
    )
    expected = set(MANDATORY_FEATURES) | {
        "MemoryFeature", "DeliveryFeature", "SchedulerFeature",
    }
    if with_workflows:
        expected.add("WorkflowsFeature")

    def fresh():
        policy = no_provider_policy()
        service = LLMService(agent_data_dir=tmp_path / "provider-free-service")
        agent = KestrelAgent(
            did=credentials.agent_did,
            storage_path=credentials.db_path,
            llm_service=service,
            payer_policy=policy,
            allowed_features=expected,
            sync_enabled=False,
        )
        inventory = set(expected)
        agent.defer_agent_readiness_to_host(maintenance=True, expected_features=inventory)
        inventory.add("CallerMutationMustNotReplaceFrozenInventory")
        return agent

    first = fresh()
    try:
        await first.initialize()
        assert first._boot_state is BootPhaseState.READY
        assert expected <= set(first.features), "configured recovery feature is absent"
        assert first.verify_feature_lifecycle_integrity() is True
        before = await first.storage.get_node(credentials.agent_did)
        assert before.properties["genesis_audit"]["status"] == "passed"
        assert await first.enter_safe_mode(
            "Native persisted feature-recovery fixture",
            cause=SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value,
        ) is True
        persisted = await first._constitution_state_store.load(credentials.agent_did)
        assert persisted.safe_mode and persisted.safe_mode_cause == "feature_lifecycle_uncertain"
    finally:
        await shutdown_fixture_agent(first)

    restarted = fresh()
    try:
        await restarted.initialize()
        assert restarted._boot_state is BootPhaseState.READY
        assert restarted._safe_mode is True
        assert restarted._safe_mode_cause == "feature_lifecycle_uncertain"
        assert expected <= set(restarted.features)
        assert restarted.verify_feature_lifecycle_integrity() is True
        assert not restarted._runtime_publication_ready.is_set()
        missing = restarted.features.pop("DeliveryFeature")
        try:
            assert restarted.verify_feature_lifecycle_integrity() is False
            await restarted.exit_safe_mode(authorization="Negative missing-inventory fixture")
            assert restarted._safe_mode is True
            assert (await restarted._constitution_state_store.load(credentials.agent_did)).safe_mode is True
        finally:
            restarted.features["DeliveryFeature"] = missing
        assert restarted.verify_feature_lifecycle_integrity() is True
        await restarted.exit_safe_mode(authorization="Local fixture sovereign explicitly authorizes native exit")
        durable = await restarted._constitution_state_store.load(credentials.agent_did)
        assert durable.safe_mode is False
        assert restarted._safe_mode is False
        after = await restarted.storage.get_node(credentials.agent_did)
        assert after.node_id == before.node_id == credentials.agent_did
        assert after.properties["constitution_hash"] == before.properties["constitution_hash"]
        assert after.properties["genesis_audit"] == before.properties["genesis_audit"]
        events = await restarted._constitution_state_store.list_events(credentials.agent_did)
        assert any(event["event_type"] == "safe_mode_exited" for event in events)
        assert restarted.maintenance_boot is True
        assert not restarted._runtime_publication_ready.is_set()
        assert restarted.features["DeliveryFeature"]._queue._task is None
        assert restarted.features["SchedulerFeature"]._runner._task is None
        if with_workflows:
            assert restarted.features["WorkflowsFeature"]._recovery_task is None
    finally:
        await shutdown_fixture_agent(restarted)
