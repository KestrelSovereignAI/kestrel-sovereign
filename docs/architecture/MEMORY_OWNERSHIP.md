# Memory System Ownership Map

> Created for Issue #501. Defines which classes belong to core, feature, or storage layers
> to enable clean extraction in Epic #466 Wave 4.

## Layer Assignment

### Core (kernel needs these to function)

| Class | File | Role | Notes |
|-------|------|------|-------|
| MemorySystem | `storage/memory_system.py` | **Facade** for all memory components | Single entry point for enrichment, retrieval, consolidation |
| MemoryRetriever | `storage/memory_retriever.py` | Weighted retrieval (semantic/emotional/importance/recency/access/certainty) | Owned by MemorySystem |
| MemoryConsolidator | `storage/memory_consolidator.py` | Episode creation, pattern detection, archival | Owned by MemorySystem |
| MemoryMetadata | `storage/memory_models.py` | Data model for message metadata | Shared data contract |
| MemoryEpisode | `storage/memory_models.py` | Data model for consolidated episodes | Shared data contract |
| TemporalPattern | `storage/memory_models.py` | Data model for detected patterns | Shared data contract |
| ConversationManager | `agent/conversation_manager.py` | History retrieval, filtering, compaction, marking | Conversation operations |
| MemoryManager | `agent/memory_manager.py` | Stash operations, episode triggers, hierarchical compaction | Context parking |
| ContextManager | `agent/context_manager.py` | Orchestrator dispatching to ConversationManager + MemoryManager | Top-level orchestrator |
| ContextBuilder | `agent/context_builder.py` | Assembles context window for LLM | Context assembly |

### Memory Feature Package (extractable)

| Class | File | Role | Notes |
|-------|------|------|-------|
| MemoryFeature | `features/memory/feature.py` | User-facing search/recall tools | Wraps MemorySystem for tool API |
| MemoryAgencyFeature | `features/memory_agency/feature.py` | Agent self-management: pin/release/facts | Distinct from MemoryFeature |

### Separate Concern (NOT memory package)

| Class | File | Role | Why Separate |
|-------|------|------|-------------|
| StrategicMemoryFeature | `features/strategic_memory/feature.py` | YAML-based strategic context (vision, milestones, blockers) | Not conversation memory. File-based, no DB. Different concern entirely. |
| AsyncRAGStore | `storage/async_rag_store.py` | Document chunking and hybrid search | Document search, not conversation memory. Different tables, different data. |
| MemoryService (A2A) | `a2a/stores/unified/memory_service.py` | Inter-agent searchable memory (FTS) | A2A protocol store. Uses `a2a_memory` table, not conversation_history. |
| SQLiteMemoryService | `a2a/stores/memory_service.py` | SQLite wrapper for A2A MemoryService | Backward-compat wrapper, deprecated path |
| PostgresMemoryService | `a2a/stores/postgres.py` | PostgreSQL wrapper for A2A MemoryService | Deprecated, scheduled for removal |

### Observability (read-only, stays in wellness)

| Class | File | Role | Notes |
|-------|------|------|-------|
| MemoryHealthCalculator | `features/wellness/metrics.py` | Memory health scoring | Read-only observer, no memory ownership |
| MemoryChecker | `kestrel-feature-reflection` optional package | Constitution/conversation/KG/RAG health checks | Health checks, not memory operations |

## Call Chain (after consolidation)

```
User request
  → KestrelAgent.generate_response()
    → ContextManager.build_context()
      → ConversationManager.get_conversation_history()
      → MemoryManager.retrieve_memories()
        → MemorySystem.retrieve()
          → MemoryRetriever.retrieve()
      → MemoryManager.create_episode_if_needed()
        → MemorySystem.consolidator.create_session_episode()
    → ContextBuilder.build()

Tool calls (LLM-initiated):
  → MemoryFeature.recall_emotional()
    → MemorySystem.retriever.retrieve()
  → MemoryFeature.search_memory()
    → storage.get_conversation_history() (client-side search)
  → MemoryAgencyFeature.memory_pin()
    → Direct DB (memory_pins table)
```

## Contracts

### Core → Storage Contract

MemorySystem is the single facade. All memory operations go through it:
- `enrich_metadata(content, role)` → enriched metadata dict
- `retrieve(query, limit, emotional_context)` → scored messages
- `consolidate()` → consolidation report
- `get_episodes(limit)` → episode list
- `.retriever` → MemoryRetriever (for direct access when needed)
- `.consolidator` → MemoryConsolidator (for episode/archival operations)

### Feature → Core Contract

Features access memory through:
1. `agent.memory_system` → MemorySystem facade
2. `agent.storage` → AsyncStorage (for raw conversation history)
3. `agent.storage.db` → AsyncDatabase (for direct queries, e.g., pin tracking)

### Why 3 Memory Features Are Separate

1. **MemoryFeature**: User-facing search/recall tools. Answers "do you remember X?"
2. **MemoryAgencyFeature**: Agent self-management. Controls what the agent chooses to preserve (pin/release). Different authorization model.
3. **StrategicMemoryFeature**: Not memory at all. YAML-based strategic context (vision, milestones, GitHub integration). Should be renamed to avoid confusion.

## Changes Made (Issue #501)

1. **Eliminated duplicate MemoryConsolidator**: Agent now uses `memory_system.consolidator` instead of creating a separate one. The standalone consolidator lacked `graph_store`, causing episodes not to appear in the Knowledge Graph.
2. **Merged duplicate search methods**: `full_history_search` and `search_memory` in MemoryFeature were identical (the former just delegated to the latter). PR #633 fully merged them — `full_history_search` was removed and `search_memory` gained an optional `session_id` parameter for session-scoped queries.
3. **MemoryFeature uses MemorySystem facade**: Instead of reaching through to raw retriever/consolidator.
