---
type: Audit Ledger
title: Review Lanes
description: Review lane assignments and expected report format for documentation
  audit work.
resource: /docs/audit/documentation-2026-05/LANES.md
tags:
- audit
- documentation
- may-2026
timestamp: 2026-05-30 00:00:00+00:00
status: snapshot
owner: documentation-audit
canonical: false
generated: false
privacy: public
---


# Review Lanes

Use these lanes for subagents or human reviewers. Lanes are system-oriented because the main risk is cross-document disagreement about current architecture.

| Lane | Focus | Initial docs |
|---|---|---|
| Package Boundaries | Core vs feature packages vs provider packages vs standalone tools | `README.md`, `KESTREL_FEATURES.md`, `pyproject.toml`, `kestrel_sovereign/data/feature_registry.toml`, `docs/guides/BUILDING_FEATURES.md` |
| Context | Prompt assembly, history forms, pruning, token budgets, context diagnostics | `docs/architecture/CONTEXT_SYSTEM_DESIGN.md`, `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md`, `docs/architecture/LLM_SERVICE_ARCHITECTURE.md`, `docs/generated/FEATURES_developer.md` |
| Memory Retrieval Storage | Memory graph, saved items, vector search, export/import, encryption, privacy effects | `docs/architecture/MEMORY_SYSTEM.md`, `docs/architecture/MEMORY_OWNERSHIP.md`, `docs/architecture/storage/STORAGE_ARCHITECTURE.md`, `docs/SOVEREIGNTY.md`, `docs/user-documentation/SOVEREIGNTY_USER_GUIDE.md` |
| LLM Routing | Model preference source, provider packages, capabilities, streaming/honesty | `docs/architecture/LLM_SERVICE_ARCHITECTURE.md`, `docs/architecture/LLM_PROVIDER_CAPABILITIES.md`, `docs/architecture/llm/PROVIDER_PLUGINS.md`, `docs/architecture/llm/HONESTY_LAYER.md` |
| Signals Workflows Talon | Wake sources, workflow package ownership, Talon boundary | `docs/architecture/SIGNAL_DISPATCHER.md`, `docs/architecture/SIGNAL_SOURCES_GUIDE.md`, `docs/architecture/WORKFLOWS_*`, `README.md`, `KESTREL_FEATURES.md` |
| Cloud Training Deployment | Cloud Run, cloud provider packages, LoRA/training docs, deployment commands | `docs/deployment/README.md`, `docs/architecture/TRAINING_PROVIDER_ARCHITECTURE.md`, `docs/architecture/PLAN_RUNPOD_INTEGRATION.md`, `docs/architecture/RUNPOD_LORA_TRAINING.md`, `docs/architecture/VASTAI_TRAINING.md` |
| User Public Docs | User guides, demos, launch copy, optional-package claims | `docs/user-documentation/`, `docs/use_cases/`, `docs/demos/`, `docs/concepts/`, `docs/design/launch/` |
| Index Diagrams Hygiene | Navigation, diagrams, old meta docs, internal/public separation | `docs/README.md`, `docs/architecture/README.md`, `docs/diagrams/`, `docs/archive/`, `docs/business/`, `docs/outreach/`, `docs/legal/`, `docs/planning/`, `docs/plans/`, `docs/strategy/`, `docs/vision/` |

## Report Filename Convention

Write lane reports under `reports/` using:

```text
<lane-slug>_report.md
```

Examples:

- `package_boundaries_report.md`
- `context_report.md`
- `memory_retrieval_storage_report.md`

