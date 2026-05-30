# Index Diagrams Hygiene Lane Report

Source: subagent lane review, read-only, 2026-05-30.

## Scope Reviewed

Reviewed index/hygiene surfaces: `docs/README.md`, `docs/architecture/README.md`, `docs/diagrams/**`, `docs/archive/**`, `docs/business/**`, `docs/outreach/**`, `docs/legal/**`, `docs/planning/**`, `docs/plans/**`, `docs/strategy/**`, `docs/vision/**`, and `docs/code_reviews/**`.

## Canonical Doc Recommendation

Make `README.md` the public entry point, `docs/README.md` the curated public docs index, and `docs/architecture/README.md` the technical architecture index. Treat `KESTREL_FEATURES.md` plus `kestrel_sovereign/data/feature_registry.toml` as feature truth. Treat diagrams as presentation/historical unless each file gains a status/date/owner banner and is reconciled with current code/package boundaries.

## Stale Or Conflicting Claims

- `docs/diagrams/README.md` links `01-kestrel-kestrel-overview.md`, but that file does not exist.
- `docs/diagrams/README.md` lists only DA-01 through DA-08 in the top-level data architecture table, while `docs/diagrams/data-architecture/README.md` has DA-09 through DA-11.
- Diagram status metadata is broadly missing: almost every file under `docs/diagrams` lacks `Status`, `Last Updated`/date, and `Owner`.
- `docs/diagrams/06-llm-management.md` and `docs/diagrams/ARCHITECTURE_DIAGRAMS.md` still present `BrainRouter`, RunPod GPU routing, and older GPU/provider diagrams as first-class current architecture, despite local note text saying to defer to `LLMService`.
- `docs/diagrams/04-feature-framework.md` shows in-tree `mcp/`, `creative/`, and "12 features"; current feature discovery is local plus installed entry points, and feature inventory is much larger.
- `docs/diagrams/data-architecture/README.md` lists old absolute paths such as `/storage/async_storage.py` and `/kestrel/server.py`; actual paths are under `kestrel_sovereign/`.
- `docs/diagrams/data-architecture/DA-07-cryostasis.md` conflicts internally: slide 2 says trigger `< $5.00`, slide 4 says balance drops to `$0.02`.
- `docs/README.md` does not link `business/`, `outreach/`, or `legal/`, but still calls out their pre-release sensitivity. Their own indexes expose named advisors, family relationships, NDA status, contact info, and confidential/investor material as ordinary docs.
- `docs/code_reviews/README.md` keeps PR review notes active without PR state, owner, or expiry; the May audit flags this as cleanup.
- `docs/archive` has no archive index/README. `docs/archive/meta/DOCUMENTATION_INVENTORY_2025.md` is archived but still contains "Active" classifications from 2025 that could mislead if found directly.

## Code/Package Evidence

- `pyproject.toml` depends on `kestrel-sovereign-sdk>=0.17.0,<1` and `kestrel-llms[all]`, and explicitly says extracted `kestrel-feature-*`, `kestrel-cloud-*`, `kestrel-storage-*`, `kestrel-voice-*`, and `kestrel-talon` packages are installed independently.
- `pyproject.toml` exposes entry-point groups `kestrel_sovereign.features`, `kestrel_sovereign.cloud_providers`, `kestrel_sovereign.voice_providers`, and `kestrel_sovereign.storage_providers`.
- `kestrel_sovereign/features/__init__.py` discovers local feature classes and installed feature entry points.
- `kestrel_sovereign/storage/tiered_manager.py` returns `Decimal("5.00")` as the cryostasis fallback trigger.
- `kestrel_sovereign/llm/service.py` is current LLM service code; diagrams still lead with `BrainRouter`.
- `kestrel_sovereign/privacy.py` and `tests/unit/test_privacy_preset_consistency.py` confirm the five privacy presets and independent flags, so `docs/diagrams/03-privacy-modes.md` is broadly aligned.

## Docs To Update

- `docs/README.md` - clarify public/internal boundaries; replace soft pre-release warning with explicit internal/private docs policy.
- `docs/diagrams/README.md` - remove broken `01` link, include DA-09 through DA-11, add status legend.
- `docs/diagrams/ARCHITECTURE_DIAGRAMS.md`
- `docs/diagrams/04-feature-framework.md`
- `docs/diagrams/06-llm-management.md`
- `docs/diagrams/data-architecture/README.md`
- `docs/business/README.md`, `docs/outreach/README.md`, `docs/legal/README.md` - mark internal/private or remove from public-facing navigation.
- `docs/code_reviews/README.md` - classify active vs historical PR notes.

## Docs To Archive Or Mark Historical

- Most or all of `docs/outreach` should be private/internal or archived before public release.
- Sensitive investor/contact/entity docs in `docs/business` and `docs/legal` should be private/internal unless redacted.
- `docs/planning/PROJECT_ROADMAP.md` appears Nov 2025-current; mark historical unless revalidated.
- `docs/plans/TODO_IPFS_GATEWAY_STANDARDIZATION.md` says complete; archive or convert to completion note.
- `docs/code_reviews/claude-pr-*.md` should move to archive after PR state confirmation.

## Generated Docs To Regenerate

- `docs/generated/FEATURES_developer.md`
- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_investor.md`
- `docs/audit/REPO_MAP.md`, if it remains active.

## Open Questions

- Should `docs/business`, `docs/outreach`, and `docs/legal` remain in-repo at all for public release, or move to a private/internal docs store?
- Are the diagram decks intended to be current architecture docs, sales/presentation artifacts, or historical snapshots?
- Who owns diagram metadata and freshness reviews?
- Which `docs/code_reviews/claude-pr-*.md` notes still correspond to open PRs or unresolved follow-up issues?

## Suggested First PR Slice

Do a low-risk hygiene PR: fix `docs/diagrams/README.md` broken/stale links, add a standard status/date/owner banner template to diagram indexes, mark the diagram collection "presentation snapshot pending revalidation," add `docs/archive/README.md`, and update `docs/README.md` to explicitly separate public docs from internal business/outreach/legal materials.

