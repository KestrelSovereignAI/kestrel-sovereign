---
type: Architecture Spec
title: Human-Like Memory System
description: '**Last Updated:** 2026-05-31 **Status:** Historical companion; canonical
  implementation details live in [`../MEMORY_SYSTEM.md`](../MEMORY_SYSTEM.md) **Commit:**
  `0b83115`'
resource: /docs/architecture/storage/HUMAN_MEMORY_SYSTEM.md
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

# Human-Like Memory System

**Last Updated:** 2026-05-31
**Status:** Historical companion; canonical implementation details live in
[`../MEMORY_SYSTEM.md`](../MEMORY_SYSTEM.md)
**Commit:** `0b83115`

> This page preserves the original human-memory design narrative. For current
> backend/storage truth, retrieval weights, and embedding status, use
> [`../MEMORY_SYSTEM.md`](../MEMORY_SYSTEM.md) and
> [`STORAGE_ARCHITECTURE.md`](STORAGE_ARCHITECTURE.md). In particular,
> `MemoryRetriever` now has a six-factor score including certainty, and
> embedding generation is still Ollama-backed while provider-standard
> embedding functions are being added.

---

## Vision

Transform Kestrel's memory from "semantic search" to "human-like recall."

**Goal:** Make AI conversations feel like talking to a friend who *actually remembers*, not a search engine.

Key capabilities:
- **Emotional tagging** - memories have feelings attached
- **Temporal patterns** - "you always get quiet on Sundays"
- **Forgetting curve** - unimportant things fade
- **Associative links** - one memory triggers related ones
- **Narrative consolidation** - episodes become stories

---

## Architecture Overview

The memory system integrates with the existing storage layer as infrastructure, not a Feature agent.

```
┌─────────────────────────────────────────────────────────────┐
│                 Existing Storage Layer                       │
├─────────────────────────────────────────────────────────────┤
│  AsyncConversationStore     AsyncGraphStore    AsyncRAGStore │
│  └─ metadata JSON ──────────┴─ concept nodes ──┴─ search()   │
│         │                          │                 │       │
│         └──────────────────────────┴─────────────────┘       │
│                           │                                  │
│                    ┌──────▼──────┐                           │
│                    │MemorySystem │  (orchestrates all)       │
│                    └─────────────┘                           │
│                           │                                  │
│    ┌──────────────────────┼──────────────────────┐          │
│    │                      │                      │           │
│    ▼                      ▼                      ▼           │
│ EmotionalTagger    TemporalAnalyzer    AssociativeLinker    │
│                                                              │
│    ┌──────────────────────┼──────────────────────┐          │
│    │                      │                                  │
│    ▼                      ▼                                  │
│ MemoryRetriever    MemoryConsolidator                       │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- All components in `storage/` (infrastructure layer, not `features/`)
- Uses existing `AsyncGraphStore` for concept associations
- Stores emotional/importance/decay/certainty fields in `conversation_history.metadata` JSON; current schema also includes `conversation_history.embedding_vec` as storage groundwork for vector semantic retrieval
- All methods async with `agent_id` scoping

---

## Components

### 1. EmotionalTagger

Analyzes messages for emotional content and importance.

**Location:** `storage/emotional_tagger.py`

**Capabilities:**
- **Sentiment analysis:** Calculates emotional valence (-1.0 to +1.0) and intensity (0.0 to 1.0)
- **Emotion categories:** Detects joy, sadness, anger, fear, surprise, nostalgia, love, anxiety, hope
- **Importance detection:** Identifies life events, personal disclosures, explicit memory markers
- **Temporal context:** Records time of day and day of week

**Importance Signals:**
| Signal | Pattern Examples | Boost |
|--------|-----------------|-------|
| Personal disclosure | "I've never told anyone", "between you and me" | +0.3 |
| Life event | "got promoted", "passed away", "moving to" | +0.4 |
| Explicit marker | "remember this", "this is important" | +0.3 |
| Emphasis | Multiple `!`, ALL CAPS | +0.1 |

**Usage:**
```python
from storage import EmotionalTagger

tagger = EmotionalTagger()
metadata = await tagger.analyze("I just got promoted at work!", "user")

# metadata.emotional_valence = 0.8 (positive)
# metadata.emotional_intensity = 0.7
# metadata.emotional_categories = ["joy"]
# metadata.importance = 0.9
# metadata.importance_reasons = ["life_event"]
```

---

### 2. TemporalAnalyzer

Detects patterns in when and how users communicate.

**Location:** `storage/temporal_analyzer.py`

**Pattern Types:**
- **Time preference:** "Most active late at night"
- **Day preference:** "Reflective on Sundays"
- **Emotion × time correlation:** "Deeper conversations at night"

**Usage:**
```python
from storage import TemporalAnalyzer

analyzer = TemporalAnalyzer(db)
patterns = await analyzer.detect_patterns(messages, agent_id, min_observations=5)

# Returns list of TemporalPattern objects with:
# - pattern_type: "time_preference" | "day_preference" | "emotion_time_correlation"
# - description: Human-readable pattern description
# - confidence: 0.0 to 1.0
# - observations: Count of supporting data points
```

---

### 3. AssociativeLinker

Builds concept associations in the knowledge graph.

**Location:** `storage/associative_linker.py`

**Concept Categories:**
| Category | Examples |
|----------|----------|
| People | mom, dad, wife, husband, friend, boss |
| Places | home, work, office, school, city names |
| Times | morning, evening, christmas, birthday |
| Emotions | happy, sad, angry, scared, anxious |
| Activities | cooking, reading, working, traveling |

**How it works:**
1. Extract concepts from message text
2. Create/update concept nodes in AsyncGraphStore
3. Link messages to concepts
4. Strengthen associations between co-occurring concepts

**Usage:**
```python
from storage import AssociativeLinker

linker = AssociativeLinker(graph_store)
concepts = await linker.extract_and_link(message_id, content, agent_id)

# Get related concepts
associated = await linker.get_associated_concepts("mom", agent_id)
# Returns: ["sunday", "phone", "brooklyn", ...]
```

---

### 4. MemoryRetriever

Retrieves memories using human-like weighting.

**Location:** `storage/memory_retriever.py`

**Scoring Algorithm:**
| Factor | Weight | Description |
|--------|--------|-------------|
| Semantic | 25% | Keyword/concept overlap; vector/cosine storage groundwork exists but is not the current retriever score |
| Emotional | 20% | Mood-congruent recall |
| Importance | 20% | From metadata tagging |
| Recency | 15% | Ebbinghaus decay curve |
| Access | 10% | Rehearsal effect (frequently accessed = stronger) |
| Certainty | 10% | Epistemic certainty from metadata |

**Ebbinghaus Forgetting Curve:**
- Base half-life: 30 days
- Importance extends half-life: 1.0x to 3.0x multiplier
- Access count provides additional boost

```python
# Memory strength = 0.5 ^ (days_old / effective_half_life)
effective_half_life = 30 * (1.0 + importance * 2.0)  # 30 to 90 days
```

**Mood-Congruent Recall:**
Memories matching the current emotional context score higher. If you're sad, sad memories surface more easily (just like human memory).

**Usage:**
```python
from storage import MemoryRetriever, MemoryMetadata

retriever = MemoryRetriever(conversation_store, linker)

# Current emotional context (e.g., user seems sad)
context = MemoryMetadata(emotional_valence=-0.5)

results = await retriever.retrieve(
    query="my mom",
    agent_id=agent_id,
    emotional_context=context,
    limit=10
)
# Returns messages sorted by human-like relevance
```

---

### 5. MemoryConsolidator

Nightly memory maintenance (like sleep consolidation in humans).

**Location:** `storage/memory_consolidator.py`

**Operations:**
1. **Episode creation:** Groups related messages into narrative episodes
2. **Pattern detection:** Finds temporal patterns across conversation history
3. **Archiving:** Marks fully decayed memories (doesn't delete)

**Episode Creation:**
- Groups messages by day
- Identifies high-emotion clusters
- Creates narrative episodes with:
  - Title: "A difficult conversation", "A joyful moment"
  - Summary: Message count and emotional arc
  - Emotional arc: "difficulty → resolution", "generally positive"

**Archiving Criteria:**
Messages are archived (not deleted) when:
- Strength < 10% (DECAY_ARCHIVE_THRESHOLD)
- Not decay_protected
- No recent access

**Usage:**
```python
from storage import MemoryConsolidator

consolidator = MemoryConsolidator(db, agent_id)
report = await consolidator.run_consolidation()

# report = {
#     "episodes_created": 3,
#     "patterns_found": 2,
#     "messages_archived": 15,
#     "total_messages_processed": 500,
#     "timestamp": "2025-12-08T22:30:00Z"
# }
```

---

### 6. MemorySystem Facade

Unified interface for all memory components.

**Location:** `storage/memory_system.py`

**Provides:**
- Single initialization point
- Coordinated message processing
- Unified retrieval interface
- Consolidation runner

**Usage:**
```python
from storage import MemorySystem

# Initialize (creates all components)
memory = MemorySystem(
    db=async_database,
    conversation_store=conversation_store,
    graph_store=graph_store,
    agent_id=agent_id
)
await memory.initialize()

# Process incoming message (tags + links concepts)
enriched_metadata = await memory.process_message(
    content="I just talked to mom about moving to Brooklyn",
    role="user",
    metadata=existing_metadata
)

# Retrieve with human-like weighting
results = await memory.retrieve(
    query="mom",
    limit=10,
    emotional_context=current_emotion
)

# Run nightly consolidation
report = await memory.consolidate()
```

---

## Database Schema

### Extended Metadata (conversation_history.metadata JSON)

The human-memory enrichment fields live in the `conversation_history.metadata`
JSON column:

```python
metadata = {
    # Existing fields (preserved)
    "enc": True,
    "new_session": True,
    "audit_failure": False,

    # NEW: Emotional layer
    "emotional_valence": 0.7,       # -1.0 to +1.0
    "emotional_intensity": 0.5,     # 0.0 to 1.0
    "emotional_categories": ["joy", "nostalgia"],

    # NEW: Importance layer
    "importance": 0.8,              # 0.0 to 1.0
    "importance_reasons": ["life_event", "personal_disclosure"],

    # NEW: Temporal layer
    "time_of_day": "late_night",    # morning/afternoon/evening/late_night
    "day_of_week": "sunday",

    # NEW: Decay layer
    "access_count": 3,
    "last_accessed": "2025-12-08T22:30:00Z",
    "decay_protected": False,
    "archived": False
}
```

### New Tables

**temporal_patterns:**
```sql
CREATE TABLE IF NOT EXISTS temporal_patterns (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL,      -- time_preference, day_preference, emotion_time_correlation
    description TEXT NOT NULL,        -- "Most active late at night"
    trigger_conditions TEXT,          -- JSON: {"time_of_day": "late_night"}
    confidence REAL DEFAULT 0.0,
    observations INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_temporal_patterns_agent ON temporal_patterns(agent_id);
```

**memory_episodes:**
```sql
CREATE TABLE IF NOT EXISTS memory_episodes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    title TEXT NOT NULL,              -- "A difficult conversation"
    summary TEXT,
    timespan_start TIMESTAMP,
    timespan_end TIMESTAMP,
    key_message_ids TEXT,             -- JSON array of message IDs
    emotional_arc TEXT,               -- "difficulty → resolution"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_memory_episodes_agent ON memory_episodes(agent_id);
```

---

## Agent Memory Tools (MemoryFeature)

The MemoryFeature provides tools for active memory search. When the agent needs to recall past conversations, it uses these tools.

**Location:** `features/memory/feature.py`

### Available Tools

| Tool | Description | When to Use |
|------|-------------|-------------|
| `search_memory` | Encryption-aware keyword search; optionally session-scoped | "Do you remember", "what did we discuss" — primary recall tool |
| `recall_emotional` | Human-like weighted retrieval | Recalling with emotional context (sad memories surface when sad) |
| `recall_recent` | Get recent N messages | "What did we just discuss?" |
| `search_documents` | Search RAG document chunks | Finding info from uploaded files/knowledge |
| `search_case_law` | Search audit decisions | Finding precedent for governance |
| `get_episodes` | Get consolidated memory episodes | High-level narrative recall |
| `memory_status` | System health check | Debugging/diagnostics |
| `memory_consolidate` | Run the consolidation pipeline | Scheduled nightly; manually as fallback |

### Tool Details

#### search_memory
```python
# Search all conversation history
await feature.search_memory(query="Wyoming", limit=20)

# Search within a single session
await feature.search_memory(query="Wyoming", limit=20, session_id="<uuid>")

# Returns: {"success": True, "results": [...], "count": N, "session_id": ...}
```
Decrypts each message client-side before matching, so it works correctly
with per-agent encryption. Previously this was split into a duplicated
`full_history_search` tool that just delegated back here; PR #633 merged
them and added the `session_id` parameter for session-scoped queries.

#### recall_emotional
```python
await feature.recall_emotional(
    query="my trip to Wyoming",
    mood="positive",  # positive, negative, neutral
    limit=10
)
# Uses human-like weighting:
# - 30% semantic relevance
# - 25% emotional congruence (mood-matching)
# - 20% importance
# - 15% recency (with decay)
# - 10% access frequency
```

#### recall_recent
```python
await feature.recall_recent(limit=20)
# Returns most recent N messages
```

### Data Flow

```
User: "Do you remember when we talked about Wyoming?"
                    │
                    ▼
        ┌───────────────────────┐
        │  Agent recognizes     │
        │  memory recall needed │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │ Calls recall_emotional│
        │ or full_history_search│
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │ MemoryRetriever       │
        │ scores all history    │
        │ using human weights   │
        └───────────┬───────────┘
                    │
                    ▼
        Returns matching memories
        with scores and metadata
```

### Important: Encryption Consideration

When conversation encryption is enabled (default), `search_memory` performs a database-level search against encrypted content, which will NOT find matches. The agent should:

1. Try `search_memory` first (fast)
2. If no results, use `full_history_search` (slower but decrypts)
3. Or use `recall_emotional` which always decrypts

---

## Integration with KestrelAgent

To integrate the memory system with KestrelAgent:

```python
# In kestrel_agent.py

class KestrelAgent:
    async def initialize(self):
        # ... existing initialization ...

        # Initialize memory system
        self.memory = MemorySystem(
            db=self.storage.db,
            conversation_store=self.storage.conversation,
            graph_store=self.storage.graph,
            agent_id=self.agent_id
        )
        await self.memory.initialize()

    async def add_conversation(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add conversation with emotional tagging."""
        meta = dict(metadata) if metadata else {}

        # Enrich user messages with emotional/importance analysis
        if role == "user":
            enriched = await self.memory.process_message(content, role, meta)
            meta.update(enriched)

        await self.storage.conversation.add_conversation(role, content, meta)

    async def get_context(self, query: str) -> List[Dict]:
        """Get relevant context using human-like retrieval."""
        # Analyze current query for emotional context
        current_emotion = await self.memory.tagger.analyze(query, "user")

        return await self.memory.retrieve(
            query=query,
            emotional_context=current_emotion,
            limit=10
        )
```

---

## The Payoff

**Before (semantic search):**
```
User: "I'm feeling down about my mom"
Agent: "You mentioned your mom on March 15th. She lives in Brooklyn."
```

**After (human-like memory):**
```
User: "I'm feeling down about my mom"
Agent: "I remember you telling me about her Sunday calls, and how much
       those meant to you after she passed. Is it one of those days
       where you just miss hearing her voice?"
```

**That's the difference between a search engine and a friend.**

---

## Files

| File | Purpose |
|------|---------|
| `storage/memory_models.py` | MemoryMetadata, TemporalPattern, MemoryEpisode dataclasses |
| `storage/emotional_tagger.py` | Sentiment analysis and importance detection |
| `storage/temporal_analyzer.py` | Time-based pattern detection |
| `storage/associative_linker.py` | Concept extraction and graph linking |
| `storage/memory_retriever.py` | Weighted retrieval with decay |
| `storage/memory_consolidator.py` | Episode creation and archiving |
| `storage/memory_system.py` | Unified facade |
| `tests/unit/test_memory_system.py` | 28 unit tests |

---

## References

- **Related vision:** `docs/development/DEEP_CONV.md` - "True Memory That Feels Human"
- **Storage architecture:** `docs/architecture/storage/STORAGE_ARCHITECTURE.md`
- **Ebbinghaus forgetting curve:** https://en.wikipedia.org/wiki/Forgetting_curve

---

*Last Updated: January 25, 2026*
