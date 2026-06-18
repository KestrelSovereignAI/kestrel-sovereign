---
type: Audit Report
title: Package Boundaries Report
description: 'Lane report from the May 2026 documentation audit: Package Boundaries
  Report.'
resource: /docs/audit/documentation-2026-05/reports/package_boundaries_report.md
tags:
- audit
- documentation
- may-2026
- report
timestamp: 2026-05-30 00:00:00+00:00
status: snapshot
owner: documentation-audit
canonical: false
generated: false
privacy: public
---


# Package Boundaries Lane Report

Source: subagent lane review, read-only, 2026-05-30.

## Scope Reviewed

Reviewed package boundary docs and evidence: `README.md`, `KESTREL_FEATURES.md`, `pyproject.toml`, `kestrel_sovereign/data/feature_registry.toml`, `docs/guides/BUILDING_FEATURES.md`, `docs/architecture/core/MODULAR_RUNTIME.md`, generated feature docs, discovery code, registry code, and boundary tests.

## Canonical Doc Recommendation

Make `docs/architecture/core/MODULAR_RUNTIME.md` the canonical boundary contract, with `KESTREL_FEATURES.md` as the canonical audited inventory. `README.md` should summarize only from those two. `feature_registry.toml` should be treated as runtime/catalog metadata, not prose truth, until its `core` and `package` semantics are corrected.

## Stale Or Conflicting Claims

- `README.md` says `pip install kestrel-sovereign` includes voice, but `feature_registry.toml` marks voice as `kestrel-feature-voice` with `core = false`, and there are no source files under `kestrel_sovereign/features/voice`.
- `README.md` says "everything else is a feature" and groups cloud providers with feature packages. This conflicts with `docs/architecture/core/MODULAR_RUNTIME.md`, which distinguishes feature packages from cloud/voice/storage provider packages.
- `README.md` lists `packages/` and top-level `features/`, but neither directory exists in the workspace.
- `README.md` still describes RunPod, Vast.ai, and GCP Compute as experimental in-repo feature surfaces, while `pyproject.toml` says those live in extracted `kestrel-cloud-*` packages.
- `feature_registry.toml` calls the file a catalog of "all feature packages," but it includes core modules, external feature packages, and voice provider packages.
- `feature_registry.toml` lists `kestrel-voice-elevenlabs`, `kestrel-voice-deepgram`, and `kestrel-voice-openai` entries as registry feature entries even though they are provider packages, not `Feature` lifecycle packages.
- Several registry entries set `core = true` while `package` is an external `kestrel-feature-*`, including web search, channels, privacy, scheduler, talon, webhooks, wellness, consent, delivery, response audit, audit anchor, and state of mind.
- Talon is confused across names: `feature_registry.toml` says `kestrel-feature-talon`, `pyproject.toml` says `kestrel-talon`, and current code has an in-core thin coordinator wrapper.
- `docs/guides/BUILDING_FEATURES.md` teaches feature authors to import from `kestrel_sovereign.features.base`; newer package-boundary docs and `pyproject.toml` point to SDK-owned contracts.
- Generated docs inherit stale claims: `FEATURES_user.md` lists GCP Compute, RunPod, Vast.ai as agent capabilities; `FEATURES_user.md` presents voice as built in; `FEATURES_investor.md` says native voice is platform-integrated; `FEATURES_developer.md` lists Talon and voice together in the core-style inventory.

## Code/Package Evidence

- Discovery code explicitly loads local core modules first, then entry-point packages: `kestrel_sovereign/features/__init__.py`.
- Current discovery returns 33 local discoverable feature classes, including `TalonCoordinatorFeature`, `WebSearchFeature`, `ChannelFeature`, etc.; voice is not discovered locally.
- Empty placeholder dirs exist for extracted packages (`voice`, `mcp`, `github`, `wallet`, `observability`, `reflection`, `council`, `runpod`, `vastai`, `gcp_compute`, `code_edit`) but contain no source files.
- Provider entry-point groups are declared in `pyproject.toml`.
- `kestrel-llms[all]==0.1.8` is a direct dependency in `pyproject.toml`.
- `kestrel-talon` is not a direct dependency in `pyproject.toml` or `uv.lock`; only comments and the in-core coordinator reference it.
- Talon coordinator code describes itself as thin dispatch to external `kestrel-talon`: `kestrel_sovereign/features/talon/coordinator.py`.
- Boundary tests enforce voice/wallet/github/mcp/etc. as external package boundaries: `tests/unit/test_extracted_feature_boundary_contracts.py`.

## Docs To Update

- `README.md`
- `KESTREL_FEATURES.md`
- `kestrel_sovereign/data/feature_registry.toml`
- `docs/guides/BUILDING_FEATURES.md`
- `docs/audit/FEATURE_PROOF_MATRIX.md`
- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_developer.md`
- `docs/generated/FEATURES_investor.md`

## Docs To Archive Or Mark Historical

- Any RunPod/Vast/GCP Compute feature-era docs that still describe old in-core feature paths should be marked historical or moved under archive after the cloud lane confirms specifics.
- `docs/audit/FEATURE_PROOF_MATRIX.md` should be updated rather than archived, but its stale rows for deleted in-core cloud feature paths need correction.

## Generated Docs To Regenerate

- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_developer.md`
- `docs/generated/FEATURES_investor.md`
- Potentially `docs/generated/README.md` if generation metadata changes.

## Open Questions

- Should `core = true` mean "ships inside `kestrel-sovereign`" only, or "first-party maintained/available by default"?
- Should core entries with future external package names use a new field such as `future_package` or `extracted_package`, leaving `package = "kestrel-sovereign"` while source lives here?
- Is Talon supposed to be documented as an in-core coordinator feature plus external standalone CLI, or should it disappear from the core feature inventory entirely?
- Is `kestrel-talon` intentionally not a dependency despite instructions saying it is installed as one?
- Should provider packages appear in `feature_registry.toml` at all, or should they move to provider-specific registries?

## Suggested First PR Slice

Normalize the source-of-truth boundary without touching behavior: define `core = true` as "discoverable from local `kestrel_sovereign/features`," fix `feature_registry.toml` package names for true core entries, remove provider packages from feature-package rows or label them separately, update `README.md` install/add-on wording, and regenerate `docs/generated/FEATURES_*`.

