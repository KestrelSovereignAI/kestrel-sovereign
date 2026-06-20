---
type: Audit Ledger
title: Documentation Inventory - May 2026
description: Working inventory seed for the May 2026 documentation audit.
resource: /docs/audit/documentation-2026-05/DOCUMENTATION_INVENTORY.md
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


# Documentation Inventory - May 2026

Status: OKF corpus inventory

This inventory records the OKF migration baseline. It is not a document-by-document freshness review; lane reviewers still own content accuracy, stale claims, and public-release hygiene.

## Counts

- Current `docs/` markdown files: 261.
- Current OKF corpus: 255 OKF documents validate with `scripts/docs_okf.py validate --all docs`.
- Reserved OKF generated views: 6 files (`docs/audit/index.md`, `docs/audit/log.md`, `docs/generated/index.md`, `docs/generated/log.md`, `docs/architecture/index.md`, `docs/architecture/log.md`).
- 0 non-reserved markdown files are missing OKF frontmatter.
- Previous inventory: `docs/archive/meta/DOCUMENTATION_INVENTORY_2025.md`.
- Current audit ledger: `docs/audit/DOCUMENTATION_AUDIT_5_2026.md`.
- OKF migration plan: `docs/audit/OKF_MIGRATION_PLAN.md`.
- Generated OKF indexes/logs: `docs/audit/index.md`, `docs/audit/log.md`, `docs/generated/index.md`, `docs/generated/log.md`, `docs/architecture/index.md`, `docs/architecture/log.md`.

## Directory Map

| Directory | Current interpretation | Audit action |
|---|---|---|
| `docs/architecture/` | Architecture specs and PRDs | Add/revalidate status banners and package ownership. |
| `docs/audit/` | Active and historical audit materials | Keep current May 2026 audit visible; mark older matrices by status; use `docs/audit/index.md` for OKF migration status. |
| `docs/audit/issues/` | GitHub issue bodies and Talon batch inputs | Keep active batches; archive superseded batches after verification. |
| `docs/generated/` | Generated feature docs and demo evidence | Regenerate feature docs after `KESTREL_FEATURES.md` is corrected; regenerate demo evidence after demo/eye config changes. |
| `docs/guides/` | Contributor guides | Reconcile with feature/package extraction and SDK contracts. |
| `docs/deployment/` | Operator runbooks | Confirm Cloud Run CLI behavior and external-provider boundaries. |
| `docs/user-documentation/` | End-user explanations and guides | Add optional-package notes where needed. |
| `docs/use_cases/` | Use-case narratives | Check for stale promises. |
| `docs/demos/` | Demo scripts | Verify current UI/CLI behavior with `kestrel-flight` demos and `kestrel-eye` review configs. |
| `docs/diagrams/` | Mermaid/data architecture diagrams | Add status/date/owner banners. |
| `docs/research/` | Research notes | Keep separate from operator guidance. |
| `docs/design/` | Brand/design assets and launch drafts | Check public messaging and optional-package claims. |
| `docs/development/` | Developer notes and experiments | Classify active vs historical. |
| `docs/plans/` | Current or future plans | Classify current roadmap vs archive. |
| `docs/planning/` | Roadmaps/backlogs | Classify current vs historical. |
| `docs/business/` | Business planning | Public-release hygiene review. |
| `docs/outreach/` | Outreach materials | Public-release hygiene review. |
| `docs/legal/` | Legal docs/templates | Public-release hygiene review. |
| `docs/strategy/` | Strategy docs | Public-release hygiene review. |
| `docs/vision/` | Vision docs | Public-release hygiene review. |
| `docs/archive/` | Historical docs | Do not treat as current guidance. |

## Next Inventory Step

Use generated OKF indexes to track classification and timestamps, then use lane reports to fill in current status and action for documents whose content needs revalidation.

Commands:

```bash
uv run python scripts/docs_okf.py inventory docs/audit/documentation-2026-05 --format markdown
uv run python scripts/docs_okf.py validate --all docs
uv run python scripts/docs_okf.py index --check
uv run python scripts/docs_okf.py log --check
```
