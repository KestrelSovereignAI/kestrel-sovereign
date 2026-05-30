# Cloud Training Deployment Lane Report

Source: subagent lane review, read-only, 2026-05-30.

## Scope Reviewed

Reviewed Cloud Run deployment, GPU provider/training docs, LoRA research notes, package metadata, current provider code, feature registry, generated feature docs, and recent git history.

## Canonical Doc Recommendation

Make `docs/deployment/README.md` canonical for Cloud Run operations. Make `docs/architecture/TRAINING_PROVIDER_ARCHITECTURE.md` canonical only for the in-core training protocol/factory/adapters, after updating paths and provider ownership. Treat RunPod/VastAI LoRA docs as historical or experimental runbooks, not source-of-truth architecture.

## Stale Or Conflicting Claims

- `docs/architecture/TRAINING_PROVIDER_ARCHITECTURE.md` uses old imports like `from features.training`, file paths like `features/training/`, and `features.runpod.runpod_manager`; current package paths are `kestrel_sovereign.features.training.*`, and RunPod/VastAI/GCP managers are lazy-loaded from external `kestrel-cloud-*` packages.
- `docs/architecture/TRAINING_PROVIDER_ARCHITECTURE.md` says default priority is `vertex_ai > replicate > gcp_compute > vastai`; current code priority is `local_mps > runpod > vertex_ai > replicate > gcp_compute > vastai`.
- `docs/architecture/TRAINING_PROVIDER_ARCHITECTURE.md` links `docs/FLUX2_TRAINING_CONFIG.md`, but the file is actually `docs/research/FLUX2_TRAINING_CONFIG.md`.
- `docs/architecture/PLAN_RUNPOD_INTEGRATION.md` claims `RunPodManager`, `BrainRouter`, and RunPod feature code live in `features/runpod/`; current repo has no source files there, and `RunPodManager` comes from `kestrel_cloud_runpod`.
- `docs/architecture/RUNPOD_LORA_TRAINING.md` calls itself "Source of Truth" for LoRA/selfie operations while also admitting it predates the split. It references removed/inaccurate paths such as `features/visual_identity/feature.py`, `kestrel/endpoints/selfie.py`, `kestrel/services/lora_training_service.py`, and `features/runpod/runpod_manager.py`.
- `docs/architecture/VASTAI_TRAINING.md` says general VastAI compute is active in `features/vastai/` with `manage_vastai`; no such source directory exists in this repo, and the training adapter now imports `kestrel_cloud_vastai.manager`.
- `docs/generated/FEATURES_user.md` and `docs/generated/FEATURES_investor.md` overstate in-core multi-cloud GPU capability by presenting GCP Compute/RunPod/Vast.ai as platform capabilities without the external-package caveat.

## Code/Package Evidence

- `pyproject.toml` says GCP Compute, Vast.ai, and RunPod support are in `kestrel-cloud-gcp`, `kestrel-cloud-vastai`, and `kestrel-cloud-runpod`; only CloudRunProvider stays in core.
- `kestrel_sovereign/features/training/factory.py` defines current provider priority and capabilities, including `local_mps`.
- `kestrel_sovereign/features/training/adapters/runpod_adapter.py`, `vastai_adapter.py`, and `gcp_compute_adapter.py` lazy-import managers from external `kestrel_cloud_*` packages.
- `kestrel_sovereign/cli_runpod.py` confirms `kestrel runpod` is a wrapper around external `kestrel-cloud-runpod`, not in-core RunPod ownership.
- `kestrel_sovereign/features/deploy/manager.py` discovers external cloud providers through the `kestrel_sovereign.cloud_providers` entry point group.
- `kestrel_sovereign/features/deploy/providers/cloudrun.py`, `kestrel_sovereign/cli_deploy.py`, `deploy_config.toml`, and `.github/workflows/deploy.yml` support the current `kestrel deploy` Cloud Run path.
- Recent history includes feature extraction work reinforcing the drift.

## Docs To Update

- `docs/architecture/TRAINING_PROVIDER_ARCHITECTURE.md`
- `docs/architecture/README.md`
- `README.md` feature-stability wording for training/cloud adapters
- `KESTREL_FEATURES.md`, if generated docs need clearer external-provider language
- `docs/audit/FEATURE_PROOF_MATRIX.md`, which still lists `visual_identity` as direct despite registry/test evidence of extraction.

## Docs To Archive Or Mark Historical

- `docs/architecture/PLAN_RUNPOD_INTEGRATION.md` - mark historical design/PRD unless rewritten for `kestrel-cloud-runpod`.
- `docs/architecture/RUNPOD_LORA_TRAINING.md` - mark historical Q1 2026 runbook; remove "Source of Truth" wording.
- `docs/architecture/VASTAI_TRAINING.md` - mark historical/deprioritized training backend; remove active in-core VastAI compute claim.
- `docs/research/LoRA/` - keep as research, clearly non-operational.

## Generated Docs To Regenerate

Regenerate after canonical inventory/provider wording is fixed:

- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_developer.md`
- `docs/generated/FEATURES_investor.md`

## Open Questions

- Should in-core training adapters remain documented as supported bridges, or should all cloud training operational docs move to `kestrel-cloud-*` repos?
- Is `kestrel runpod` still intended as a core CLI surface, or should it eventually move beside `kestrel-cloud-runpod`?
- Should `runpod_config.toml` and `vastai_config.toml` stay in this repo as examples, or move to external package templates?

## Suggested First PR Slice

Update `TRAINING_PROVIDER_ARCHITECTURE.md` and `architecture/README.md` first: fix package paths/imports, current provider priority, external manager ownership, broken FLUX2 link, and add a short "what core owns vs external packages own" table.

