---
type: Review Lane
title: User Public Docs
description: Review prompt for the User Public Docs lane of the May 2026 documentation
  audit.
resource: /docs/audit/documentation-2026-05/lanes/user_public_docs.md
tags:
- audit
- documentation
- may-2026
- review-lane
timestamp: 2026-05-30 00:00:00+00:00
status: snapshot
owner: documentation-audit
canonical: false
generated: false
privacy: public
---


# Lane Brief: User Public Docs

Goal: audit user-facing docs, use cases, demos, and launch copy for stale availability claims and unclear optional-package requirements.

Start with:

- `docs/user-documentation/`
- `docs/use_cases/`
- `docs/demos/`
- `docs/concepts/`
- `docs/design/launch/`
- `README.md`
- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_investor.md`

Check for:

- claims that optional packages are included by default
- UI or CLI flows that no longer match current behavior
- memory/privacy/export claims that disagree with architecture docs
- launch/demo copy promising moved or aspirational features
- public docs that expose internal implementation detail unnecessarily

Report to: `reports/user_public_docs_report.md`

