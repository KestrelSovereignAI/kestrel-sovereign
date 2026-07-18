---
type: Architecture Spec
title: Agent Identity Contract
description: 'Every Kestrel agent has a single canonical identity: its **DID** (`self.did`).'
resource: /docs/architecture/AGENT_IDENTITY_CONTRACT.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Agent Identity Contract

## Summary

Every Kestrel agent has a single canonical identity: its **DID** (`self.did`).

## Rules

1. **`self.did` is the source of truth.** It is set once at construction and never reassigned.
2. **`self.agent_id` is a read-only property** that returns `self.did`. It exists for backward compatibility with storage interfaces and feature packages that accept an `agent_id` parameter.
3. **Do not independently set `agent_id`.** Any code that previously wrote `self.agent_id = ...` on a `KestrelAgent` instance must be removed; the property will raise `AttributeError` on assignment.
4. **No `getattr` fallback chains.** Code like `getattr(self, 'agent_id', '') or getattr(self, 'did', '')` is banned. Use `self.did` (or `self.agent.did` from a feature) directly.

## For Feature Authors

When your feature needs to identify its parent agent:

```python
# In Feature.initialize():
agent_id = self.agent.did      # preferred
agent_id = self.agent.agent_id # also works (returns did)
```

When passing identity to storage layers, use the `agent_id` parameter name — the value is always the DID:

```python
store = AsyncConversationStore(db=db, agent_id=self.agent.did)
```

## For Storage Authors

Storage classes accept `agent_id: str` as a constructor parameter. Callers always pass the agent's DID. The parameter is named `agent_id` (not `did`) because storage is identity-scheme-agnostic — it just needs a unique key.

## Identity package replace contract

`IdentityImporter.import_package(..., merge_mode="replace")` is an exact,
atomic replacement of the database state that the importer owns. It is not a
broad purge of every row associated with the DID.

The destructive inventory is:

| Package component | Database inventory replaced |
| --- | --- |
| Episodes | `memory_episodes` rows for the target DID |
| Saved items | `saved_items` rows for the target DID |
| Temporal patterns | `temporal_patterns` rows for the target DID |
| Reflection insights | `reflection_insights` rows for the target DID |
| Relationships | Target-DID edges to `user` graph nodes; orphaned component nodes are reclaimed |
| Skills | Target-DID `has_skill` edges to `skill` graph nodes; orphaned component nodes are reclaimed |
| Wallet | `wallet_state` and `wallet_transactions` rows for the target DID |

Graph nodes shared through another edge are preserved after the target DID's
component edge is removed. The target's `agent` node, constitution/governance
nodes, lifecycle/sovereignty evidence, and existing `migration_record` nodes
are never part of replace cleanup. Migration history is append-only: a
successful import adds one new `migration_record` and keeps earlier evidence.

Required package rows are validated before cleanup. Cleanup, every component
insert, wallet history, and the new migration record then run in one
`AsyncDatabase.transaction()` on both SQLite and PostgreSQL. A required delete,
insert, or audit failure rolls back the whole unit, returns `success=False`, an
empty migration id, and no imported-row statistics. Helpers must not commit or
turn an integrity failure into a warning. Warnings are reserved for intentional
policy/merge outcomes, such as an explicitly allowed unsigned package, a
`skip_existing` result, or a wallet-history row already present during merge.

Some signed-package fields are verification or transport inputs rather than
members of this database replacement inventory: source DID metadata,
constitution text, personality/system-prompt calibration, tool preferences,
legal-entity data, voice configuration, and the source migration-history
array. The importer must not overwrite target-owned runtime configuration,
keys, governance files, or external artifacts ad hoc. A future restorer for
one of those fields must stage and validate its non-database changes before
the database transaction, publish them only after the database commit, and
provide an explicit compensation/rollback path; until then, database import
success makes no claim that those target-owned artifacts were replaced.

## History

See [#500](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/500) for the consolidation rationale.
