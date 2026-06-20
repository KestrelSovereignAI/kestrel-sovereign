---
type: Architecture Spec
title: Subagent Isolation Audit
description: '**Issue:** [#569 - Subagent isolation audit -- explicit opt-in for shared
  state in feature dispatch](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/569)
  **Phase:...'
resource: /docs/architecture/subagent_isolation_audit.md
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

# Subagent Isolation Audit

**Issue:** [#569 - Subagent isolation audit -- explicit opt-in for shared state in feature dispatch](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/569)
**Phase:** 1 (Audit Only -- No Code Changes)
**Date:** 2026-04-05
**Scope:** Audit all 40 `feature.py` files in `kestrel_sovereign/features/*/` for shared agent state access patterns.

---

## Methodology

For each feature, the following `self.agent.*` access patterns were audited:

- **Storage Read**: Reads from `self.agent.storage` (includes `.db`, `.get_conversation_history()`, graph queries, file retrieval)
- **Storage Write**: Writes to `self.agent.storage` (includes `.db.execute()` for INSERT/UPDATE/CREATE TABLE, `.add_node()`, `.store_file()`)
- **Memory Read**: Reads from `self.agent.memory_system` (consolidator, retriever)
- **Memory Write**: Writes to `self.agent.memory_system` (memory modification, decay protection changes)
- **History Write**: Writes to conversation history (via storage or privacy agent)
- **Other Features**: Accesses `self.agent.features` to reach sibling features
- **LLM Service**: Accesses `self.agent.llm_service` for generation or model management
- **Hooks Manager**: Accesses `self.agent.hooks_manager` for dynamic hook registration

## Classification Legend

| Classification | Meaning |
|---|---|
| **READ_ONLY** | Only reads from agent state, never writes |
| **WRITES_STORAGE** | Creates tables or writes rows to agent storage DB |
| **WRITES_MEMORY** | Modifies memory metadata (decay_protected, pins, etc.) |
| **WRITES_HISTORY** | Writes to conversation_history table |
| **ACCESSES_OTHER_FEATURES** | Reaches into sibling features via `self.agent.features` |
| **EXTERNAL_ONLY** | Does not access agent storage at all (uses external APIs, config files, or own managers) |
| **LLM_CALLER** | Calls `self.agent.llm_service.generate()` |

---

## Audit Results

| Feature | Storage Read | Storage Write | Memory Read | Memory Write | History Write | Other Features | LLM | Hooks | Classification |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|----------------|
| audit_anchor | Y | Y | - | - | - | Y | - | - | WRITES_STORAGE, ACCESSES_OTHER_FEATURES |
| bootstrap | Y | Y | - | - | - | - | - | - | WRITES_STORAGE |
| bridge | Y | Y | - | - | - | Y | - | - | WRITES_STORAGE, ACCESSES_OTHER_FEATURES |
| channels | Y | Y | - | - | - | - | - | - | WRITES_STORAGE |
| code_edit | - | - | - | - | - | Y | - | - | EXTERNAL_ONLY, ACCESSES_OTHER_FEATURES |
| compute | - | - | - | - | - | Y | - | Y | EXTERNAL_ONLY, ACCESSES_OTHER_FEATURES |
| consent | Y | Y | - | - | - | - | Y | - | WRITES_STORAGE, LLM_CALLER |
| context | - | - | - | - | - | - | Y | - | READ_ONLY, LLM_CALLER |
| council | - | - | - | - | - | - | - | - | EXTERNAL_ONLY |
| delivery | Y | Y | - | - | - | - | - | - | WRITES_STORAGE |
| deploy | - | - | - | - | - | - | - | - | EXTERNAL_ONLY |
| gcp_compute | - | - | - | - | - | - | Y | - | EXTERNAL_ONLY, LLM_CALLER |
| github | - | - | - | - | - | - | - | - | EXTERNAL_ONLY |
| heartbeat | Y | Y | - | - | - | - | - | - | WRITES_STORAGE |
| identity | Y | Y | - | - | - | - | - | - | WRITES_STORAGE |
| keys | Y | Y | - | - | - | Y | - | - | WRITES_STORAGE, ACCESSES_OTHER_FEATURES |
| mcp | - | - | - | - | - | - | - | - | EXTERNAL_ONLY |
| memory | Y | - | Y | - | - | - | - | - | READ_ONLY |
| memory_agency | Y | Y | - | Y | - | - | - | - | WRITES_STORAGE, WRITES_MEMORY |
| model | - | - | - | - | - | Y | Y | - | LLM_CALLER, ACCESSES_OTHER_FEATURES |
| observability | - | - | - | - | - | - | - | Y | READ_ONLY |
| peers | - | - | - | - | - | - | - | - | EXTERNAL_ONLY |
| privacy | Y | - | - | - | Y | - | - | - | WRITES_HISTORY |
| reflection | Y | Y | - | - | - | Y | Y | - | WRITES_STORAGE, ACCESSES_OTHER_FEATURES, LLM_CALLER |
| response_audit | - | - | - | - | - | - | - | Y | EXTERNAL_ONLY |
| runpod | - | - | - | - | - | - | Y | - | EXTERNAL_ONLY, LLM_CALLER |
| save | Y | Y | - | - | - | - | - | - | WRITES_STORAGE |
| scheduler | Y | Y | - | - | - | Y | - | - | WRITES_STORAGE, ACCESSES_OTHER_FEATURES |
| security | - | - | - | - | - | Y | - | Y | ACCESSES_OTHER_FEATURES |
| sovereignty | Y | Y | - | - | - | Y | - | - | WRITES_STORAGE, ACCESSES_OTHER_FEATURES |
| spawn | - | - | - | - | - | - | - | - | EXTERNAL_ONLY |
| strategic_memory | - | - | - | - | - | Y | - | - | EXTERNAL_ONLY, ACCESSES_OTHER_FEATURES |
| tasks | - | - | - | - | - | - | - | - | EXTERNAL_ONLY |
| vastai | - | - | - | - | - | - | Y | - | EXTERNAL_ONLY, LLM_CALLER |
| visual_identity | Y | Y | - | - | - | - | - | - | WRITES_STORAGE |
| voice | Y | Y | - | - | - | - | - | - | WRITES_STORAGE |
| wallet | - | - | - | - | - | - | - | - | EXTERNAL_ONLY |
| web_search | - | - | - | - | - | - | - | - | EXTERNAL_ONLY |
| webhooks | Y | Y | - | - | - | - | - | - | WRITES_STORAGE |
| wellness | Y | Y | - | - | - | - | - | - | WRITES_STORAGE |

---

## Summary Statistics

| Category | Count | Features |
|----------|-------|----------|
| **WRITES_STORAGE** | 20 | audit_anchor, bootstrap, bridge, channels, consent, delivery, heartbeat, identity, keys, memory_agency, reflection, save, scheduler, sovereignty, visual_identity, voice, webhooks, wellness, privacy (history), context (via context_manager) |
| **READ_ONLY** (storage) | 2 | memory, observability |
| **EXTERNAL_ONLY** | 12 | code_edit, council, deploy, github, mcp, peers, spawn, tasks, wallet, web_search, vastai, runpod |
| **ACCESSES_OTHER_FEATURES** | 10 | audit_anchor, bridge, code_edit, compute, keys, model, reflection, scheduler, security, sovereignty, strategic_memory |
| **LLM_CALLER** | 7 | consent, context, gcp_compute, model, reflection, runpod, vastai |
| **WRITES_MEMORY** | 1 | memory_agency |
| **WRITES_HISTORY** | 1 | privacy |
| **HOOKS_MANAGER** | 4 | compute, observability, response_audit, security |

---

## Detailed Cross-Feature Access Patterns

These are the most concerning patterns for isolation because one feature reaches into another feature's internal state.

### Features that access `self.agent.features` (sibling features)

| Feature | What it accesses | Why |
|---------|-----------------|-----|
| **audit_anchor** | Iterates `self.agent.features` to find `SecurityFeature.permission_store` | Reads security audit log from SecurityFeature's separate DB |
| **bridge** | Iterates `self.agent.features` to enumerate all tools | Builds capabilities list for bridge protocol |
| **code_edit** | `self.agent.features.get("security")` | Requests approval via SecurityFeature's approval_queue |
| **compute** | `self.agent.features.get('SecurityFeature')` | Requests approval via SecurityFeature's approval_queue |
| **keys** | `self.agent.features.get("security")` or `self.agent.get_feature("security")` | Requests approval for key rotation |
| **model** | `self.agent.features.get("ConsentFeature")` | Records agent consent before model change |
| **reflection** | `self.agent.get_feature("wallet")`, `self.agent.get_feature("github")` | Gets wallet for economic gate, GitHub for ticket creation |
| **scheduler** | Iterates all `self.agent.features` to find tools by name | Executes scheduled tasks by looking up feature tools |
| **security** | Iterates all `self.agent.features` to register tool permissions | Registers all tools with default ASK permission |
| **sovereignty** | Iterates `self.agent.features` to find `AuditAnchorFeature` | Includes audit anchor status in sovereignty export |
| **strategic_memory** | Iterates `self.agent._features` to find `TalonCoordinatorFeature` | Dispatches issues to Talon coordinator |

### Features that create their own DB tables in shared storage

These features use `self.agent.storage.db` to create tables, sharing the same SQLite database:

| Feature | Tables Created |
|---------|---------------|
| audit_anchor | `audit_anchors` |
| bridge | `bridge_sessions`, `bridge_log` |
| channels | `channel_messages`, `channel_config` |
| consent | `consent_log` |
| delivery | (via DeliveryQueue) |
| heartbeat | `heartbeat_log` |
| memory_agency | `memory_pins` |
| reflection | (via ReflectionDatabaseHelper) |
| save | (via SavedItemsStore) |
| scheduler | `scheduled_tasks`, `task_execution_log` |
| visual_identity | (via storage layer) |
| voice | (via storage.store_file) |
| webhooks | `webhook_config`, `webhook_log` |
| wellness | `wellness_checkpoints` |

### Features that use `self.agent.llm_service`

| Feature | Usage | Writes back? |
|---------|-------|-------------|
| consent | `self.agent.llm_service.generate()` -- generates consent reflection | No |
| context | Uses `self.llm_service` (cached from agent) for summarization/compaction | No |
| gcp_compute | `self.llm_service.switch_backend()` -- **mutates** LLM routing | **YES** |
| model | `self.llm_service.set_model_preference()` -- **mutates** model config | **YES** |
| reflection | Uses `llm_service` for interaction analysis | No |
| runpod | `self.llm_service.switch_backend()` -- **mutates** LLM routing | **YES** |
| vastai | `self.llm_service.switch_backend()` -- **mutates** LLM routing | **YES** |

### Features that dynamically register hooks

| Feature | Hook Type | Method |
|---------|-----------|--------|
| compute | `ComputeSecurityHook`, `ComputeDebugHook` | `get_hooks()` (lifecycle) |
| observability | `ObservabilityHook` | `get_hooks()` (lifecycle) |
| response_audit | `ResponseAuditHook` | `get_hooks()` + dynamic `hooks_manager.register()` |
| security | `SecurityHook` | `get_hooks()` (lifecycle) |

---

## Risk Assessment

### High Risk (shared mutable state, no isolation boundary)

1. **All 20 features write to the same SQLite database** via `self.agent.storage.db`. There is no table-level namespace isolation -- any feature could accidentally read/write another feature's tables.

2. **LLM routing mutation** (`gcp_compute`, `runpod`, `vastai`, `model`) -- these features mutate the shared `llm_service` state. If two features call `switch_backend()` concurrently, the result is undefined.

3. **Cross-feature access** via `self.agent.features` -- 10 features reach into sibling features. The most common pattern is accessing `SecurityFeature.approval_queue`, but some (like `scheduler`) iterate all features to invoke tools.

### Medium Risk (read-only access to shared state)

4. **Memory reads** (`memory` feature) -- reads conversation history and memory system but does not write. Safe as long as the memory system handles concurrent reads.

5. **Hooks registration** -- 4 features register hooks via `get_hooks()` lifecycle method. This is the intended pattern and is safe.

### Low Risk (no shared state access)

6. **12 EXTERNAL_ONLY features** -- these do not touch agent storage at all. They use their own managers, external APIs, or config files.

---

## Recommendations for Phase 2

1. **Table namespacing**: Each feature should own a declared set of tables. A feature attempting to access an undeclared table should be flagged.

2. **Explicit state contract**: Each feature should declare what it reads and writes via a class-level manifest (e.g., `READS = ["storage.db"]`, `WRITES = ["storage.db:consent_log"]`).

3. **LLM service isolation**: `switch_backend()` and `set_model_preference()` mutations should go through a coordination layer, not be called directly by individual features.

4. **Cross-feature access gateway**: Replace direct `self.agent.features[name]` access with a typed request/response protocol. Features should declare dependencies explicitly.

5. **Privacy agent consultation**: Only `privacy` feature should write to conversation history. Other features that need to store data should use their own tables.
