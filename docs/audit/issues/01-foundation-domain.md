---
type: Issue Body
title: 01 Foundation Domain
description: 'Kestrel’s core trust claims start here: constitutional integrity, DID
  identity, portable exports, continuity verification, encrypted storage, memory integrity,
  and sovereignty g...'
resource: /docs/audit/issues/01-foundation-domain.md
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

# 01 Foundation Domain

## Problem

Kestrel’s core trust claims start here: constitutional integrity, DID identity, portable exports, continuity verification, encrypted storage, memory integrity, and sovereignty guarantees. These areas underpin nearly every other feature, but they also span multiple modules and persistence layers, which makes drift and false confidence likely unless we test invariants across the full path.

## Goal

Audit and red-team the foundation layer so constitutional and sovereignty guarantees are proven end to end, not just asserted in isolated modules.

## In Scope

- constitution verification, safe mode, audit anchoring
- DID inception, signing, verification, identity packages
- export/import, migration, continuity verification, graceful degradation
- storage backends, encryption, privacy wrapper interactions
- memory system, retrieval, consolidation, graph integrity
- sync, tiered storage, decentralized storage adapters, sovereignty receipts

## Source-of-Truth Areas

- `kestrel_sovereign/agent/constitution.py`
- `kestrel_sovereign/inception_service.py`
- `kestrel_sovereign/identity/`
- `kestrel_sovereign/storage/`
- `kestrel_sovereign/features/sovereignty/`
- `kestrel_sovereign/filecoin_adapter.py`

## Required Proof

- unit tests for invariants and explicit failure behavior
- integration tests for export/import, encryption, continuity, and dual-backend parity
- adversarial tests for constitution bypass and tamper scenarios
- real-resource tests for Filecoin/Lighthouse paths where applicable

## High-Risk Seams

- constitution hash verification vs stored anchor state
- export/import under rotated or layered keys
- privacy mode rules applied before persistence or export
- SQLite/PostgreSQL divergence in storage and sync semantics
- continuity guarantees across substrate migration with degraded capabilities

## Exit Criteria

- every foundation claim in `KESTREL_FEATURES.md` is mapped to explicit invariants
- storage and sovereignty claims have dual-backend coverage where applicable
- tamper, replay, malformed package, and key-rotation cases are covered
- no duplicate source of truth remains for identity or sovereignty-critical behavior
