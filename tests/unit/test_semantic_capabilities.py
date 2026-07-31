"""Per-agent stable/draft semantic capability runtime contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kestrel_sovereign.agent.sleep import SleepMixin
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.knowledge.capabilities import (
    SemanticCapabilityConfigurationError,
    SemanticRuntimeCapabilities,
    semantic_capabilities_from_config,
)
from kestrel_sovereign.knowledge.corpus import GovernedCorpusPolicy
from kestrel_sovereign.knowledge.maintenance import SemanticMaintenanceStatus
from kestrel_sovereign.knowledge.rdf_codec import UnsupportedRdfCapabilityError
from kestrel_sovereign.knowledge.assertion import (
    EpistemicState,
    OntologyRef,
    Visibility,
)
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

EXPERIMENTAL = {
    "mode": "experimental",
    "rdf12": {
        "capability": "rdf-profile:rdf12-cr-20260407-experimental",
        "version": "0.1.0",
    },
    "sparql12": {
        "capability": "query-profile:sparql12-20260605-experimental",
        "version": "0.1.0",
    },
    "shacl12": {
        "capability": "validation-profile:shacl12-core-20260602-experimental",
        "version": "0.1.0",
    },
    "shape_set": {
        "identifier": "kestrel-assertion-shapes-shacl12-experimental",
        "version": "0.1.0",
    },
}


def test_semantic_capabilities_default_to_stable_without_draft_pins() -> None:
    stable = SemanticRuntimeCapabilities.stable()

    assert stable.to_mapping() == {"mode": "stable"}
    assert stable.allow_experimental is False
    assert stable.validation_capability == "validation-profile:shacl-core-20170720"


def test_experimental_semantic_capabilities_require_all_exact_local_pins() -> None:
    selected = semantic_capabilities_from_config(EXPERIMENTAL)

    assert selected.allow_experimental is True
    assert selected.capability_versions()["rdf12_version"] == "0.1.0"
    assert selected.capability_versions()["sparql12_version"] == "0.1.0"
    assert selected.capability_versions()["validation_capability"] == (
        "validation-profile:shacl12-core-20260602-experimental"
    )

    partial = {key: value for key, value in EXPERIMENTAL.items() if key != "sparql12"}
    with pytest.raises(SemanticCapabilityConfigurationError, match="require all exact"):
        semantic_capabilities_from_config(partial)

    mismatched = {
        **EXPERIMENTAL,
        "rdf12": {**EXPERIMENTAL["rdf12"], "version": "9.9.9"},
    }
    with pytest.raises(SemanticCapabilityConfigurationError, match="does not match"):
        semantic_capabilities_from_config(mismatched)


@pytest.mark.asyncio
async def test_sleep_runtime_passes_agent_selected_draft_contract_and_reports_it() -> (
    None
):
    selected = semantic_capabilities_from_config(EXPERIMENTAL)

    class Storage:
        captured = None

        async def run_semantic_maintenance(self, profile, **kwargs):
            self.captured = kwargs["semantic_capabilities"]
            return SimpleNamespace(
                status=SemanticMaintenanceStatus.NO_OP,
                reason=None,
                source_generation=0,
                checkpoint_generation=0,
                assertions_inferred=0,
                assertions_retracted=0,
                to_mapping=lambda: {
                    "status": "no_op",
                    "reason": None,
                    "source_generation": 0,
                    "checkpoint_generation": 0,
                    "backlog_assertions": 0,
                    "backlog_reports": 0,
                    "capability_versions": {
                        "semantic_maintenance": "v3",
                        **selected.capability_versions(),
                    },
                },
            )

    class Agent(SleepMixin):
        semantic_inference_profile = None
        semantic_inference_configured = False
        semantic_maintenance_configured = True
        semantic_inference_limits = None
        semantic_maintenance_limits = None
        semantic_capabilities = selected

        def __init__(self) -> None:
            self.storage = Storage()

    agent = Agent()
    report = await agent.sleep(
        skip_consolidation=True,
        skip_export=True,
        skip_reflection=True,
    )

    assert agent.storage.captured is selected
    active = report.semantic_maintenance_diagnostics()["active_capabilities"]
    assert "semantic_capability_mode=experimental" in active
    assert "rdf12_version=0.1.0" in active
    assert "sparql12_version=0.1.0" in active
    assert (
        "validation_capability=validation-profile:shacl12-core-20260602-experimental"
        in active
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selection", "expected_mode"),
    [
        (SemanticRuntimeCapabilities.stable(), "stable"),
        (semantic_capabilities_from_config(EXPERIMENTAL), "experimental"),
    ],
)
async def test_sleep_uses_capabilities_through_privacy_storage_facade(
    tmp_path,
    selection,
    expected_mode: str,
) -> None:
    """The production privacy façade must retain the agent's exact selection."""
    identity_dir = tmp_path / expected_mode
    credentials = await create_kestrel_identity_async(
        str(identity_dir),
        identity_method="did:pkh",
        agent_name=f"Semantic {expected_mode} capability test",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    capability = _resolve_authenticated_agent_assertion_capability(
        credentials.agent_did,
        identity,
    )
    raw_storage = AsyncStorage(
        ":memory:",
        agent_id=credentials.agent_did,
        _assertion_tenant_capability=capability,
        semantic_capabilities=selection,
    )
    await raw_storage.initialize()
    try:
        runtime = raw_storage.semantic_rdf_capability_report()
        if expected_mode == "experimental":
            assert runtime.rdf12 is not None
            assert runtime.sparql12 is not None
            assert selection.rdf_runtime_matches(runtime)
        facade = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)

        class Backend:
            def execute_readonly(self, query, *, timeout_seconds):
                return ()

            def cancel_readonly(self, *, profile):
                return None

        if expected_mode == "experimental":
            adapter = facade.semantic_sparql12_read_adapter(
                Backend(), lambda row, ownership: row
            )
            assert type(adapter).__name__ == "Sparql12AssertionReadAdapter"
        else:
            with pytest.raises(UnsupportedRdfCapabilityError, match="explicitly selected"):
                facade.semantic_sparql12_read_adapter(
                    Backend(), lambda row, ownership: row
                )

        class Agent(SleepMixin):
            semantic_inference_profile = None
            semantic_inference_configured = False
            semantic_maintenance_configured = False
            semantic_capabilities_configured = True
            semantic_inference_limits = None
            semantic_maintenance_limits = None
            semantic_capabilities = selection

            def __init__(self) -> None:
                self.storage = facade

        report = await Agent().sleep(
            skip_consolidation=True,
            skip_export=True,
            skip_reflection=True,
        )

        assert report.semantic_maintenance is not None
        assert report.semantic_maintenance["status"] == "no_op"
        active = report.semantic_maintenance_diagnostics()["active_capabilities"]
        assert f"semantic_capability_mode={expected_mode}" in active
        if expected_mode == "experimental":
            assert "rdf12_version=0.1.0" in active
            assert "sparql12_version=0.1.0" in active
        else:
            assert not any(item.startswith("rdf12_") for item in active)
            assert not any(item.startswith("sparql12_") for item in active)
    finally:
        await raw_storage.close()


@pytest.mark.asyncio
async def test_experimental_sleep_readiness_and_corpus_share_one_durable_runtime(
    tmp_path,
) -> None:
    """An experimental sleep cannot become stable-only at the corpus boundary."""
    selection = semantic_capabilities_from_config(EXPERIMENTAL)
    identity_dir = tmp_path / "experimental-corpus"
    credentials = await create_kestrel_identity_async(
        str(identity_dir),
        identity_method="did:pkh",
        agent_name="Experimental corpus capability test",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    capability = _resolve_authenticated_agent_assertion_capability(
        credentials.agent_did, identity
    )
    raw_storage = AsyncStorage(
        ":memory:",
        agent_id=credentials.agent_did,
        _assertion_tenant_capability=capability,
        semantic_capabilities=selection,
    )
    await raw_storage.initialize()
    try:
        storage = PrivacyEnforcingStorage(raw_storage, PrivacyMode.NORMAL)

        class Agent(SleepMixin):
            semantic_inference_profile = None
            semantic_inference_configured = False
            semantic_maintenance_configured = False
            semantic_capabilities_configured = True
            semantic_inference_limits = None
            semantic_maintenance_limits = None
            semantic_capabilities = selection

            def __init__(self) -> None:
                self.storage = storage

        before = await storage.semantic_maintenance_training_readiness(
            None, semantic_capabilities=selection
        )
        assert before.ready is False
        assert before.reason == "semantic_maintenance_state_missing"
        before_checkpoint = await raw_storage.db.fetchone(
            "SELECT checkpoint_generation FROM semantic_maintenance_state "
            "WHERE tenant_id = ?",
            (credentials.agent_did,),
        )
        assert before_checkpoint is None

        sleep = await Agent().sleep(
            skip_consolidation=True, skip_export=True, skip_reflection=True
        )
        assert sleep.semantic_maintenance["status"] == "no_op"
        readiness = await storage.semantic_maintenance_training_readiness(
            None, semantic_capabilities=selection
        )
        assert readiness.ready is True
        checkpoint = await raw_storage.db.fetchone(
            "SELECT checkpoint_generation, status FROM semantic_maintenance_state "
            "WHERE tenant_id = ?",
            (credentials.agent_did,),
        )
        assert checkpoint == (0, "no_op")
        capability_versions = await storage.semantic_maintenance_capability_versions(
            None, semantic_capabilities=selection
        )
        assert capability_versions["semantic_capability_mode"] == "experimental"
        assert capability_versions["rdf12_runtime"] == "rdf12-cr-20260407@0.1.0"
        assert (
            capability_versions["sparql12_runtime"]
            == "sparql12-20260605-experimental@0.1.0"
        )
        policy = GovernedCorpusPolicy(
            policy_id="experimental-self-corpus",
            policy_version="1",
            accepted_epistemic_states=(EpistemicState.ASSERTED,),
            accepted_visibility=(Visibility.PRIVATE,),
            accepted_privacy_classifications=("normal",),
            accepted_consent_references=("policy:training-v1",),
            accepted_grounding_classes=("operator-attested",),
            accepted_source_kinds=("operator-note",),
            accepted_ontology_pins=(
                OntologyRef(
                    "https://example.test/ontology", "1.0.0", "sha256:test", "test"
                ),
            ),
            accepted_semantic_capability_versions=tuple(capability_versions.items()),
        )
        snapshot = await storage.governed_assertion_corpus_snapshot(
            policy=policy,
            inference_profile=None,
            semantic_capabilities=selection,
        )
        assert snapshot.capability_versions == capability_versions
    finally:
        await raw_storage.close()
