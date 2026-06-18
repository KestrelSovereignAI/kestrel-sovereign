---
type: Audit Ledger
title: Stale Artifact Review
description: Ledger of stale artifacts and revalidation candidates for the May 2026
  documentation audit.
resource: /docs/audit/documentation-2026-05/STALE_ARTIFACTS.md
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


# Stale Artifact Review

This file tracks prior audit/index/review artifacts that should not silently compete with the May 2026 documentation audit.

## Keep As Historical

- `docs/archive/meta/DOCUMENTATION_INVENTORY_2025.md` - prior corpus inventory. Already archived.
- `docs/archive/plans/*` - prior planning artifacts. Keep archived.
- `docs/archive/code_reviews/*` - prior code reviews. Keep archived.
- `docs/archive/reviews/*` - prior reviews. Keep archived.

## Keep But Mark Generated Or Snapshot

- `docs/audit/REPO_MAP.md` - useful generated repo map. Keep active only if regenerated as part of the current workflow or clearly labeled as a snapshot.

## Needs Status Banner Or Re-Audit

- `docs/audit/API_ENDPOINT_MATRIX.md`
- `docs/audit/API_SURFACE_RECONCILIATION.md`
- `docs/audit/AUTH_SURFACE_MATRIX.md`
- `docs/audit/FEATURE_AUDIT_MATRIX.md`
- `docs/audit/FEATURE_INVENTORY_RECONCILIATION.md`
- `docs/audit/FEATURE_MODULE_MATRIX.md`
- `docs/audit/SEAM_CAMPAIGNS.md`
- `docs/audit/SYNC_ASYNC_AUDIT.md`

These may still be useful, but they likely predate recent package extraction, context, memory, signal, and LLM changes. Treat them as inputs, not current truth, until revalidated.

## Candidate Issue-Batch History

- `docs/audit/issues/413-execution-order.md`
- `docs/audit/issues/413-prd.json`
- `docs/audit/issues/codex-provider-execution-order.md`
- `docs/audit/issues/codex-provider-prd.json`

These are reusable pattern examples. Move to archive only after confirming they are no longer active Talon/batch inputs.

## Candidate Review Cleanup

- `docs/code_reviews/README.md`
- `docs/code_reviews/claude-pr-*.md`

Classify as active review records or move under `docs/archive/code_reviews/`.

The old `docs/code_reviews/README.md` pointed at these files, which are absent from the current tree:

- `docs/code_reviews/ROUND_3_PLAN.md`
- `docs/code_reviews/CODE_REVIEW_01_03_2026.md`
- `docs/code_reviews/CODE_REVIEW_01_03_2026_STATUS.md`
- `docs/code_reviews/SPRINT_1_COMPLETION.md`

The index was updated in this audit to list the files that actually exist.
