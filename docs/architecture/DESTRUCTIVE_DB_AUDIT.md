---
type: Architecture Spec
title: Destructive DB Audit Trail
description: 'Kestrel records destructive conversation-history operations in a separate SQLite database with append-only guarantees for constitutional compliance.'
resource: /docs/architecture/DESTRUCTIVE_DB_AUDIT.md
tags:
- docs
- architecture
- architecture-spec
- storage
- security
- audit
timestamp: '2026-06-22T00:00:00Z'
status: active
owner: architecture
canonical: false
generated: false
privacy: public
---

# Destructive DB Audit Trail

Kestrel records destructive conversation-history operations in a separate
SQLite database, `kestrel_audit.db`, next to the agent's main
`kestrel_prime.db`. The audited data lives in `destructive_audit_log`, which is
append-only: SQLite triggers reject `UPDATE` and `DELETE` against the table.

Each record is written before the destructive operation executes. If the
pre-operation audit write fails, the purge fails closed and the target rows are
left intact. The record includes timestamp, target `agent_id`, caller/process
identity, operation type, row count, target scope, reason, and a deterministic
SHA-256 hash of the rows selected for destruction.

`AuditAnchorFeature` includes `destructive_audit_log` entries in the existing
audit-anchor payload alongside `security_audit_log` entries, so the isolated
audit trail participates in the existing asynchronous anchoring flow.

## Destructive Operations

The following conversation-history operations are classified as destructive
because they issue hard SQL deletes:

- `purge_message`
- `purge_conversation_session`
- `purge_all`
- `purge_all_since`
- `purge_trash_older_than`

User-facing `clear_history`, `delete_message`, and `delete_conversation_session`
are soft-delete operations. They stamp `deleted_at` and remain recoverable from
Trash until one of the hard-delete purge operations above removes the rows.
