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
`canonical_content_hash`. Current readers consult both identities. Every
PostgreSQL enqueue takes a transaction-scoped lock on owner, recipient, and
canonical content, while SQLite uses its immediate writer transaction, so keyed
and plain requests cannot race past short-window deduplication. A keyed request
adopts a short-window row only when its delivery channel and effective retry
policy also match; otherwise its durable replay claim would point at different
delivery semantics. Schema initialization backfills at most 500 missing
canonical identities for the current owner, and each enqueue checks the indexed
null set so rows written later by an older rolling-deployment process are
reconciled before deduplication without an unbounded startup migration.

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
trigger. A stale replay claim records its prior queue ID while a replacement is
being created, so failed joined-transaction compensation restores that claim
instead of deleting its fail-closed conflict history. The move into dead letter
uses the same recoverable ordering: it writes
the tombstone before deleting the live row, while queue processing and keyed
replay treat any temporary dual-row state as terminal until the transition is
resumed. Explicit retry checks the tombstone first, removes any residual live
original, and only then consumes the tombstone and exposes the single retry row.
While retry is resumable, both `original_id` and `retry_entry_id` remain
tombstoned for processing, deduplication, replay, listing, and retention.
Legacy dead-letter rows added before retry policy persistence keep a nullable
policy marker and fall back to the active queue configuration; new rows always
persist their original policy.
