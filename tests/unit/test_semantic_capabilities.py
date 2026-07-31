"""Per-agent stable/draft semantic capability runtime contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kestrel_sovereign.agent.sleep import SleepMixin
from kestrel_sovereign.knowledge.capabilities import (
    SemanticCapabilityConfigurationError,
    SemanticRuntimeCapabilities,
    semantic_capabilities_from_config,
)
from kestrel_sovereign.knowledge.maintenance import SemanticMaintenanceStatus

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
