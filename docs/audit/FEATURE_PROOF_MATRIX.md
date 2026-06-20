---
type: Audit Ledger
title: Feature Proof Matrix
description: This is the first pass at mapping each discoverable feature module to
  direct proof.
resource: /docs/audit/FEATURE_PROOF_MATRIX.md
tags:
- docs
- audit
- audit-ledger
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: documentation
canonical: false
generated: false
privacy: public
---

# Feature Proof Matrix

This is the first pass at mapping each discoverable feature module to direct proof.

Status meanings:

- `Direct`: module has at least one focused unit or integration test aimed at the feature itself
- `Indirect`: behavior is exercised through broader endpoint/e2e/system tests, but there is no focused feature-level proof yet
- `Gap`: no meaningful direct proof located in the current suite

| Feature Module | Primary Source | Current Proof | Status |
|---|---|---|---|
| `audit_anchor` | `kestrel_sovereign/features/audit_anchor.py` | `tests/unit/test_audit_anchor_feature.py` | Direct |
| `bootstrap` | `kestrel_sovereign/features/bootstrap/` | `tests/unit/test_bootstrap_feature.py`, `tests/unit/test_bootstrap_service.py`, `tests/integration/test_bootstrap_flow.py` | Direct |
| `bridge` | `kestrel_sovereign/features/bridge/` | `tests/unit/test_bridge_feature.py` | Direct |
| `channels` | `kestrel_sovereign/features/channels.py` | `tests/unit/test_channels_feature.py` | Direct |
| `code_edit` | `kestrel_sovereign/features/code_edit/` | `tests/unit/test_code_edit_feature.py` | Direct |
| `compute` | `kestrel_sovereign/features/compute.py` | `tests/unit/test_compute_feature.py`, `tests/integration/test_compute_security_integration.py` | Direct |
| `consent` | `kestrel_sovereign/features/consent.py` | `tests/unit/test_consent_feature.py`, `tests/unit/test_consent_timeout.py` | Direct |
| `constitution` | `kestrel_sovereign/features/constitution.py` | `tests/unit/test_constitution_audit.py`, `tests/integration/test_constitution_adversarial.py`, `tests/integration/test_constitution_real_llm.py` | Direct |
| `context` | `kestrel_sovereign/features/context.py` | `tests/unit/test_context_builder.py`, `tests/unit/test_context_management.py`, `tests/integration/test_context_e2e.py` | Direct |
| `council` | `kestrel-feature-council` optional package via `kestrel_sovereign.features` entry point | `tests/unit/test_extracted_feature_boundary_contracts.py`; package tests live in `kestrel-feature-council` | Boundary |
| `delivery` | `kestrel_sovereign/features/delivery.py` | `tests/unit/test_delivery_feature.py` | Direct |
| `deploy` | `kestrel_sovereign/features/deploy/` | `tests/unit/test_deploy_feature.py`, `tests/unit/test_deploy_models.py`, `tests/integration/test_deploy_e2e.py` | Direct |
| `gcp_compute` | `kestrel_sovereign/features/gcp_compute/` | `tests/unit/test_gcp_compute_feature_contracts.py`, `tests/integration/test_gcp_compute_e2e.py`, `tests/unit/test_cloud_launcher_contracts.py` | Direct |
| `github` | `kestrel-feature-github` optional package via `kestrel_sovereign.features` entry point | `tests/unit/test_extracted_feature_boundary_contracts.py`; package tests live in `kestrel-feature-github` | Boundary |
| `health` | `kestrel_sovereign/features/health/` | `tests/unit/test_health_feature.py`, `tests/unit/test_heartbeat.py` | Direct |
| `identity` | `kestrel_sovereign/features/identity/` | `tests/unit/test_identity_package.py`, `tests/unit/test_identity_constitution_endpoint_contracts.py`, `tests/integration/test_identity_export_import.py` | Direct |
| `keys` | `kestrel_sovereign/features/keys.py` | `tests/unit/test_key_resolution_security.py`, `tests/integration/test_key_management.py` | Direct |
| `mcp` | `kestrel-feature-mcp` optional package via `kestrel_sovereign.features` entry point | `tests/unit/test_extracted_feature_boundary_contracts.py`, `tests/integration/test_dynamic_features.py`, `tests/integration/test_orchestration_e2e.py` | Boundary |
| `memory` | `kestrel_sovereign/features/memory.py` | `tests/unit/test_memory_manager.py`, `tests/unit/test_memory_system.py`, `tests/integration/test_memory_feature_e2e.py` | Direct |
| `memory_agency` | `kestrel_sovereign/features/memory_agency.py` | `tests/unit/test_memory_agency_feature.py` | Direct |
| `model` | `kestrel_sovereign/features/model/feature.py` | `tests/unit/test_model_selection_contracts.py`, `tests/unit/test_model_set_openrouter.py`, `tests/integration/test_model_set_routing_e2e.py` | Direct |
| `observability` | `kestrel-feature-observability` optional package via `kestrel_sovereign.features` entry point | `tests/unit/test_extracted_feature_boundary_contracts.py`; package tests live in `kestrel-feature-observability` | Boundary |
| `peers` | `kestrel_sovereign/features/peers/` | `tests/unit/test_peers_feature.py` | Direct |
| `reflection` | `kestrel-feature-reflection` optional package via `kestrel_sovereign.features` entry point | `tests/unit/test_extracted_feature_boundary_contracts.py`; package tests live in `kestrel-feature-reflection` | Boundary |
| `runpod` | `kestrel_sovereign/features/runpod/` | `tests/unit/test_runpod_model_contracts.py`, `tests/unit/test_runpod_logs.py`, `tests/integration/test_runpod_feature.py` | Direct |
| `save` | `kestrel_sovereign/features/save.py` | `tests/unit/test_saved_items.py`, `tests/unit/test_commands_conversations_endpoint_contracts.py`, `tests/integration/test_sovereignty_guarantees.py` | Direct |
| `scheduler` | `kestrel_sovereign/features/scheduler.py` | `tests/unit/test_scheduler_feature.py` | Direct |
| `security` | `kestrel_sovereign/features/security.py` | `tests/unit/test_security_feature.py`, `tests/unit/test_security_endpoint_contracts.py`, `tests/integration/test_compute_security_integration.py` | Direct |
| `sovereignty` | `kestrel_sovereign/features/sovereignty.py` | `tests/unit/test_sovereignty_endpoint_contracts.py`, `tests/unit/test_sovereignty_sanitization.py`, `tests/integration/test_sovereignty_e2e.py` | Direct |
| `state_of_mind` | `kestrel_sovereign/features/state_of_mind.py` | `tests/unit/test_state_of_mind_feature.py` | Direct |
| `strategic_memory` | `kestrel_sovereign/features/strategic_memory.py` | `tests/unit/test_strategic_memory_async_contracts.py` | Direct |
| `tasks` | `kestrel_sovereign/features/tasks/` | `tests/unit/test_workflow_executor.py`, `tests/unit/test_a2a_task_manager.py`, `tests/integration/test_orchestration_e2e.py` | Direct |
| `vastai` | `kestrel_sovereign/features/vastai/` | `tests/unit/test_vastai_feature.py`, `tests/unit/test_cloud_launcher_contracts.py`, `tests/integration/test_vastai_e2e.py` | Direct |
| `visual_identity` | `kestrel_sovereign/features/visual_identity/` | `tests/unit/test_visual_identity_feature.py` | Direct |
| `voice` | `kestrel-feature-voice` optional package via `kestrel_sovereign.features` entry point | `tests/unit/test_extracted_feature_boundary_contracts.py`; package tests live in `kestrel-feature-voice` | Boundary |
| `wallet` | `kestrel-feature-wallet` optional package via `kestrel_sovereign.features` entry point | `tests/unit/test_extracted_feature_boundary_contracts.py`; package tests live in `kestrel-feature-wallet` | Boundary |
| `web_search` | `kestrel_sovereign/features/web_search/` | `tests/unit/test_web_search_feature.py` | Direct |
| `webhooks` | `kestrel_sovereign/features/webhooks.py` | `tests/unit/test_webhooks_feature.py` | Direct |
| `wellness` | `kestrel_sovereign/features/wellness.py` | `tests/unit/test_wellness_feature.py`, `tests/unit/test_wellness_telemetry_guard.py` | Direct |

## Highest-priority remaining feature-level proof gaps

- No module is currently in `Gap` status in this first-pass feature proof matrix.
- Remaining work is deeper than simple feature presence: tighten core/extracted boundary proof for `mcp`, direct proof quality for `reflection`, and deployment/cloud workflows, and keep driving endpoint/adversarial coverage under the umbrella audit.

This matrix should stay aligned with `FEATURE_MODULE_MATRIX.md` and should be used to drive the next round of gap-closing work.
