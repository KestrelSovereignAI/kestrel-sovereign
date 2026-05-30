# Report Index

Reports captured during the May 2026 documentation audit.

## Pattern And Cleanup

- [`pattern_review_report.md`](reports/pattern_review_report.md) - prior audit/index/batch pattern to reuse.
- [`cleanup_candidates_report.md`](reports/cleanup_candidates_report.md) - stale artifacts, safe archives, and must-not-archive docs.

## System Lanes

- [`package_boundaries_report.md`](reports/package_boundaries_report.md) - core vs external feature packages vs provider packages vs standalone tools.
- [`context_report.md`](reports/context_report.md) - prompt assembly, history forms, route caps, durable salvage, context diagnostics.
- [`memory_retrieval_storage_report.md`](reports/memory_retrieval_storage_report.md) - memory scoring, encrypted search, saved-item vector search, storage backends, sovereignty exports.
- [`llm_routing_report.md`](reports/llm_routing_report.md) - `kestrel.toml [llm]`, provider routes, `kestrel-llms`, model preference, old `llm_config.toml` claims.
- [`signals_workflows_talon_report.md`](reports/signals_workflows_talon_report.md) - wake sources, workflow extraction/status, Talon boundary, mesh-to-A2A drift.
- [`cloud_training_deployment_report.md`](reports/cloud_training_deployment_report.md) - Cloud Run, external cloud providers, training adapters, RunPod/Vast/LoRA historical docs.
- [`user_public_docs_report.md`](reports/user_public_docs_report.md) - user docs, generated audience docs, launch claims, optional-package language.
- [`index_diagrams_hygiene_report.md`](reports/index_diagrams_hygiene_report.md) - docs indexes, diagrams, archive index, public/internal hygiene.

## Consolidated First Slices

1. Package truth: fix `KESTREL_FEATURES.md`, `feature_registry.toml`, package-boundary wording, and generated docs.
2. Main entry points: update `README.md`, `docs/README.md`, and `docs/architecture/README.md`.
3. Current-state architecture: context, memory/storage, LLM, signals/Talon, cloud/training.
4. Public/user docs: user guides, generated audience docs, launch/demo claims.
5. Hygiene: diagrams, archive metadata, code-review notes, internal business/outreach/legal classification.

