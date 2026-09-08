---
type: Architecture Spec
title: Delivery Enqueue Idempotency
description: Owner-scoped durable replay semantics for DeliveryFeature queue actions.
resource: /docs/architecture/delivery_idempotency.md
tags:
- docs
- architecture
- architecture-spec
- delivery
timestamp: '2026-09-08T00:00:00Z'
status: active
owner: architecture
canonical: false
generated: false
privacy: public
---

# Delivery enqueue idempotency

Programmatic callers may pass an optional owner-scoped `idempotency_key` to
`DeliveryFeature.enqueue_message`. A safe replay returns the canonical queue
entry ID; a changed request raises `DeliveryIdempotencyConflict`, and a replay
of a dead-lettered entry raises `DeliveryIdempotencyTerminal` until the entry is
explicitly retried. Whether `max_retries` was omitted is part of request
identity. Keyed payloads must be JSON-serializable without string fallbacks.

The raw key is not stored. `delivery_idempotency` retains its SHA-256 digest and
payload digest under the delivery owner's DID. This minimizes accidental raw-key
disclosure but does not make a guessable key confidential. The
`delivery_queue_schema_v3` migration lock serializes creation and upgrades of
the ledger, its retention index, and SQLite's explicitly marked compensation
trigger. Queue rows retain the historical `content_hash` representation for
readers from a rolling deployment and store the semantic JSON identity in
`canonical_content_hash`. Current readers consult both identities. Keyed
PostgreSQL enqueues also take a transaction-scoped lock on owner, recipient,
and canonical content so distinct replay keys cannot race past short-window
deduplication.

`delivery_purge` expires successful replay claims with their delivered queue
rows. Its age threshold is therefore also the completed-delivery replay-safety
window: reuse after that retention period creates a new delivery. Dead-letter
claims remain with their dead-letter record and move to the new canonical queue
ID on explicit retry. Retry locks that record, records a resumable candidate ID,
preserves its stored payload representation and retry policy, and deletes the
dead letter only after the live row and replay mapping exist. This prevents both
payload loss in joined SQLite transactions and duplicate live rows when
operators race. Purge removes delivered rows before their replay claims, so a
partially committed joined operation leaves a safe stale claim rather than two
deliveries. Ordinary ledger deletion never deletes a live queue row; only a
marker written by failed enqueue compensation invokes the SQLite cleanup
trigger.
