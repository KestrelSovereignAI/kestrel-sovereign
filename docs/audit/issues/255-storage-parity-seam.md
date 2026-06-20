---
type: Issue Body
title: 255 Storage Parity Seam
description: 'Part of #255.'
resource: /docs/audit/issues/255-storage-parity-seam.md
tags:
- docs
- audit
- issue-body
timestamp: '2026-06-18T00:00:00Z'
status: snapshot
owner: documentation
canonical: false
generated: false
privacy: public
---

# 255 Storage Parity Seam

## Parent

Part of #255.

## Problem

SQLite-local endpoint tests cover much of the storage behavior, but the seam campaign requires parity across SQLite/PostgreSQL for conversations, sync, and tasking.

## Goal

Create a backend-switching seam suite that proves storage, sync, and tasking semantics are consistent across supported database backends.

## Required scenarios

- conversation/session list and transcript queries on SQLite and PostgreSQL
- task listing/filtering/status serialization on SQLite and PostgreSQL
- sync/webhook processing does not block the event loop and persists the same semantic records
- encrypted metadata/query behavior remains equivalent across backends

## Invariants

- API payload shape is backend-independent
- ordering, pagination, session boundary, and encryption flags match across backends
- async handlers do not call blocking backend work directly
- backend-specific failures are explicit and do not masquerade as empty data

## Proof expectations

- shared fixture that runs the same contract against both backends when PostgreSQL is available
- skip/fail behavior must distinguish unavailable infrastructure from broken parity
- update `docs/audit/SEAM_CAMPAIGNS.md` when proven
