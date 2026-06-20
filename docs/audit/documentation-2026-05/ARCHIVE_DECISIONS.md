---
type: Audit Ledger
title: Archive Decisions
description: Cleanup and archive decisions from the May 2026 documentation audit.
resource: /docs/audit/documentation-2026-05/ARCHIVE_DECISIONS.md
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


# Archive Decisions

This file records cleanup decisions made during the May 2026 documentation audit.

## Already Archived Before This Audit

- `docs/archive/meta/DOCUMENTATION_INVENTORY_2025.md` - prior documentation inventory. Keep as historical reference; superseded for active audit work by `docs/audit/DOCUMENTATION_AUDIT_5_2026.md` and this workspace.
- `docs/archive/plans/*` - historical planning artifacts. Leave archived unless a current doc still links to them as active guidance.
- `docs/archive/code_reviews/*` and `docs/archive/reviews/*` - historical reviews. Leave archived.

## Active Audit Artifacts To Keep

- `docs/audit/DOCUMENTATION_AUDIT_5_2026.md`
- `docs/audit/documentation-2026-05/`
- `docs/audit/README.md`
- `docs/audit/REPO_MAP.md` if it remains part of the generated repo-map workflow.

## Cleanup Candidates To Review

- `docs/code_reviews/*.md` - decide whether these are active review records or should move under `docs/archive/code_reviews/`.
- `docs/audit/issues/*` - keep active issue bodies that still correspond to open/current audit programs; archive completed or superseded issue batches under `docs/archive/meta/superseded_by_2026_05/` only after confirming they are no longer referenced.
- `docs/plans/*` and `docs/planning/*` - classify as current roadmap, historical plan, or internal planning.
- `docs/business/`, `docs/outreach/`, `docs/legal/`, `docs/strategy/`, `docs/vision/` - classify for public release hygiene.

## Decisions Made In This Pass

- Created `docs/audit/documentation-2026-05/` as the current audit workspace instead of adding more loose files.
- Recorded subagent cleanup findings in `reports/pattern_review_report.md` and `reports/cleanup_candidates_report.md`.

## Must Not Archive Without Further Review

- `docs/archive/KESTREL_FEATURES_legacy.md` - tests reference this as the archived legacy catalog.
- `docs/audit/FEATURE_PROOF_MATRIX.md` - current proof matrix; tests reference it.
- `docs/audit/SEAM_CAMPAIGNS.md` - active cross-feature seam tracker.
- `docs/audit/SYNC_ASYNC_AUDIT.md` - active sync/async control document.
- `docs/audit/AUTH_SURFACE_MATRIX.md` - current auth classification matrix.
- `docs/audit/issues/00-umbrella-whole-of-vision-audit.md`
- `docs/audit/issues/01-foundation-domain.md`
- `docs/audit/issues/02-runtime-domain.md`
- `docs/audit/issues/03-security-domain.md`
- `docs/audit/issues/04-platform-domain.md`
- `docs/audit/issues/05-audit-matrix.md`
- `docs/audit/issues/06-seam-campaigns.md`
- `docs/development/DEVCONTAINER_QUICKSTART.md`

## Open Cleanup Policy Notes

- `docs/archive/` is gitignored for new files in this repository, while some existing archive files such as `docs/archive/KESTREL_FEATURES_legacy.md` are already tracked. Do not assume newly added archive indexes or moved files will be tracked unless the ignore policy is changed intentionally.
- Left `docs/archive/meta/DOCUMENTATION_INVENTORY_2025.md` in place as historical context.
