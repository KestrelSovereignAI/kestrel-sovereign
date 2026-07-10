---
type: Audit Report
title: Memory System Deep Dive — Round 2
description: Live-path and static audit of Kestrel's semantic, lexical, emotional, importance, rehearsal, episode, and persistence layers.
resource: /docs/audit/MEMORY_SYSTEM_ROUND_2_2026-07-10.md
tags:
- audit
- memory
- retrieval
- dogfooding
timestamp: '2026-07-10T00:00:00Z'
status: active
owner: architecture
canonical: false
generated: false
privacy: public
---

# Memory System Deep Dive — Round 2

## Scope and method

This pass audited the current `main` after the first memory campaign. It used an
isolated worktree (`codex/memory-audit-r2`) and an isolated test agent (`kite-r2`)
on port 8782. The live agent used Ollama `qwen2.5:0.5b` for chat and
`qwen3-embedding:0.6b` (1024 dimensions) for conversation embeddings. The
same corpus was then restarted with conversation embeddings disabled to exercise
the lexical path. No production agent or production database was used.

The audit traced the complete path:

`HTTP invoke → context retrieval → candidate generation → semantic/emotional/importance scoring → token insertion → rehearsal → persistence → post-response tagging/routing → consolidation/episodes`.

## Findings

### 1. Mixed vector/legacy corpora both miss exact matches and inject unrelated rows

Severity: high.

With embeddings enabled, `!memory recall zirconium neutral 10 0.1` failed to
return the exact old, unembedded `zirconium` row. Because some other rows had
embeddings, vector kNN returned a non-empty set and the retriever never ran its
full-corpus lexical candidate source. It instead returned ten unrelated rows.
The control query `absentneedle` also returned ten unrelated rows.

The relevance floor is not calibrated to real embedding distributions. For the
query `axolotl`, raw Qwen cosine values were 0.6076 for the exact fact but also
0.3494–0.5792 for unrelated onboarding, gardening, and operator-notice rows.
The retriever multiplies cosine by 0.7, while the shipped floor is 0.1, so nearly
every vector candidate is eligible. Importance, recency, emotion, and certainty
then make irrelevant rows look strong (observed totals approximately 0.46–0.64).

Required correction: always merge lexical exact/token candidates with vector
candidates, including mixed and degraded embedding states; calibrate vector
eligibility so salience cannot turn the embedding model's positive similarity
floor into context noise; retain component telemetry and profile-aware behavior.

### 2. Lexical candidate generation and final relevance scoring disagree on tokens

Severity: high.

With embeddings disabled, `!memory recall axolotl neutral 10 0.1` returned the
two matching rows, while `!memory recall axolotl? neutral 10 0.1` returned zero.
`AsyncConversationStore` uses a punctuation-aware tokenizer to find candidates,
but `MemoryRetriever._score_semantic` uses lowercase whitespace splitting. The
candidate is found and then rejected because `axolotl? != axolotl`.

Required correction: one shared lexical projection/tokenizer must own candidate
matching and lexical relevance scoring, including stopwords, negation, wrappers,
and punctuation.

### 3. Episode keyword fallback requires the whole natural-language query verbatim

Severity: medium-high.

An unembedded episode with summary “We discussed the ancient zirconium compass”
was returned for `!memory episodes 10 zirconium` but not for
`!memory episodes 10 what did we discuss about zirconium`. The fallback uses one
`LIKE '%<entire query>%'` predicate. This is the only episode recall path on
PostgreSQL today and the recovery path for legacy/null embeddings on SQLite.

Required correction: tokenize and rank episode fallback matches, preserve exact
phrase preference, and merge them with vector results without starving either
source.

### 4. Bootstrap conversation turns bypass canonical memory ingestion

Severity: high.

The first live turn said “Please remember this exact fact: my cobalt axolotl is
named Quasar-17.” It persisted but never ran emotional tagging, importance
classification, concept extraction, or schema routing. Recall treated it as a
legacy row with default importance 0.5. A normal post-bootstrap message correctly
received `emotional_tag_version=heuristic-v2`, importance 1.0, concepts, and
schema-routing metadata.

Required correction: every persisted bootstrap user/assistant exchange must
enter the same canonical post-response memory pipeline, subject to the same
privacy gate and using canonical row identity.

### 5. Post-response enrichment scans and decrypts the entire corpus twice per turn

Severity: medium-high.

Phase 1 calls `get_full_history_with_ids()` merely to locate the just-persisted
user/assistant pair. The background temporal phase calls it again and slices the
last 50 rows in Python. At 100,010 plaintext rows, an otherwise small local turn
took 1.79 seconds; the explicit full-corpus lexical recall alone took about
0.68–0.72 seconds. Encrypted corpora pay decryption/canonicalization per row.

Required correction: thread persisted row IDs when possible and provide a
bounded recent-history query for temporal analysis. No request-path operation
that needs two rows or fifty rows should materialize the full history.

### 6. SQLite's advertised atomic metadata merge loses concurrent writes

Severity: high.

`update_message_metadata()` claims optimistic locking, but its SQLite path is a
plain `SELECT`, Python dict merge, and unconditional full-JSON `UPDATE`. An
adversarial interleaving reproduced the loss: access count 2 was atomically
incremented to 3, then a stale derived-metadata write restored it to 2.

Required correction: use one SQL JSON merge statement on SQLite, matching the
PostgreSQL contract, and regression-test interleavings with rehearsal and
post-response derived metadata.

### 7. Emotional attribution and confidence are applied asymmetrically

Severity: medium-high.

The assistant sentence “I'm so sad that your sister died yesterday” was tagged
as the assistant's emotion and assigned importance 0.95 (`life_event` plus
`high_emotion`). This duplicates and can outrank the user's actual life event.
Separately, opposite-valence scoring returned exactly 0.3 at confidence 1.0,
0.4, and 0.01: confidence attenuates same-valence reward but not opposite-valence
penalty.

Required correction: make importance/subject attribution role-aware so an
assistant's acknowledgement of the user's event is not a second assistant life
event, and scale both congruent rewards and incongruent penalties by reliable
emotional evidence.

### 8. Memory documentation and request logging contain avoidable slop

Severity: medium (logging), low (documentation).

The normal request path logs `[SESSION-DEBUG]` at INFO and includes the first 50
characters of stored conversation content. This is unnecessary content leakage
in routine logs. Memory docs also disagree with implementation: module and
manager prose still advertise old 30%/25% weights, the example retrieval config
omits `memory_min_relevance`, and the semantic scorer docstring repeats a line.

Required correction: remove content-bearing INFO diagnostics (retain bounded,
content-free DEBUG telemetry) and align the documented scoring/config contract
with the actual single source of truth.

## Performance observation

The full-corpus lexical fallback is bounded in memory, not bounded in work. On
the isolated plaintext SQLite corpus it took approximately 0.16 seconds at
20,010 rows and 0.68–0.72 seconds at 100,010 rows. The salience source also uses
a temporary B-tree for JSON-expression ordering (`EXPLAIN QUERY PLAN`) and scans
the agent's active rows. These paths need explicit latency budgets, corpus-size
telemetry, and an indexed/searchable design before million-row or encrypted
agents are routine. The correctness fixes should not hide this remaining scaling
work behind broad exception fallbacks.

## Proposed PR grouping

1. **Retrieval correctness:** findings 1–3. They share candidate generation,
   tokenization, eligibility, and vector/lexical merge contracts.
2. **Ingestion and persistence integrity:** findings 4–6. They share canonical
   row identity, post-response lifecycle, bounded reads, and metadata mutation.
3. **Emotional calibration:** finding 7. Keep behavioral scoring changes isolated
   for focused regression review.
4. **Logging and documentation hygiene:** finding 8. Low-risk cleanup with no
   scoring behavior mixed into it.

Each implementation PR must include focused tests, a live re-verification against
`kite-r2`, Claude Code Opus review of the worktree, and green repository CI before
merge.
