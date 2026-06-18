---
type: Audit Ledger
title: Canonical Sources
description: Source-of-truth map for the May 2026 documentation audit.
resource: /docs/audit/documentation-2026-05/CANONICAL_SOURCES.md
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


# Canonical Sources

This file records which documentation surfaces are allowed to act as source-of-truth during the May 2026 audit.

| Subject | Canonical source | Derived or secondary sources | Notes |
|---|---|---|---|
| Audit ledger | `docs/audit/DOCUMENTATION_AUDIT_5_2026.md` | `docs/audit/documentation-2026-05/*` | The ledger is the executive view; this folder holds working artifacts. |
| Documentation navigation | `docs/README.md` | directory READMEs | Should link to canonical docs, not duplicate details. |
| Public project narrative | `README.md` | launch/user docs | Should describe package boundaries at a high level only. |
| Feature inventory | `KESTREL_FEATURES.md` | `docs/generated/FEATURES_*.md` | Must be fixed before generated docs are regenerated. |
| Runtime feature catalog | `kestrel_sovereign/data/feature_registry.toml` | Feature Store UI/API docs | Needs clarified field semantics. |
| Generated feature docs | `scripts/generate_feature_docs.py` plus `KESTREL_FEATURES.md` | `docs/generated/*.md` | Generated docs are not edited directly. |
| OKF migration | `docs/audit/OKF_MIGRATION_PLAN.md` plus `scripts/docs_okf.py` | `docs/**/index.md`, `docs/**/log.md` | OKF indexes/logs are generated from frontmatter and should not be hand-edited. |
| Demo evidence | `demos/*/demo.cjs` plus `demos/*/eye.toml` | `docs/generated/DEMO_EVIDENCE.md`, `docs/demos/DEMO_SCRIPT.md` | Demo evidence is generated from executable demos and visual-review configs. |
| Package metadata | `pyproject.toml` | README install sections | Source for runtime deps, optional deps, and entry-point groups. |
| Architecture index | `docs/architecture/README.md` | individual architecture docs | Index should only mark a doc Active when current code/package ownership is known. |
| Context architecture | To be selected by context lane | `CONTEXT_SYSTEM_DESIGN.md`, `CONTEXT_C_DURABLE_SALVAGE.md` | Current docs need reconciliation. |
| Memory/retrieval/storage architecture | To be selected by memory lane | memory/storage/user sovereignty docs | Current docs need reconciliation. |
| LLM routing architecture | `docs/architecture/LLM_SERVICE_ARCHITECTURE.md` after re-audit | LLM provider docs | Must align with SDK and `kestrel-llms`. |
| Signals architecture | `docs/architecture/SIGNAL_DISPATCHER.md` after re-audit | `SIGNAL_SOURCES_GUIDE.md`, workflow docs | Workflow ownership must be clarified. |
| Talon | `AGENTS.md` plus `kestrel-talon` docs/package, after verification | README/generated docs | Must distinguish standalone package, control surface, runtime preferences, and policy. |
| Talon docs gates | `.kestreltalon/quality.yaml` | `.github/workflows/docs-okf.yml` | Talon quality checks mirror CI for OKF/generated-doc/demo-evidence freshness. |
| Cloud Run deployment | `docs/deployment/README.md` | README deployment section | External cloud provider packages should be linked, not documented as in-core. |
