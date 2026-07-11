---
type: Audit Report
title: Memory Retrieval Quality Benchmark — July 2026
description: Empirical precision, recall, abstention, contradiction, mood, importance, and long-horizon evaluation of conversation-memory retrieval.
resource: /docs/audit/MEMORY_RETRIEVAL_QUALITY_2026-07-11.md
tags:
- audit
- memory
- retrieval
- benchmark
timestamp: '2026-07-11T00:00:00Z'
status: active
owner: architecture
canonical: false
generated: false
privacy: public
---

# Memory Retrieval Quality Benchmark — July 2026

## Method

The versioned suite contains 24 conversational memories and 18 natural-language
queries: 15 answerable paraphrases and 3 deliberately unanswerable questions.
It includes explicitly superseded facts, positive/negative emotional memories,
high-importance distractors, uncertain claims, and relevant memories up to 900
days old. The runner uses Kestrel's production embedding query contract and
`MemoryRetriever` component scorer.

Run it with:

```bash
uv run python scripts/benchmark_memory_quality.py --model <ollama-model>
```

Metrics are macro recall/precision at 5, mean reciprocal rank, top-1 accuracy,
abstention accuracy, forbidden stale-hit rate, and mean returned results.

## Baseline failures

At the shipped `min_relevance=0.1`, salience dominated relevance and none of the
models abstained on unknown facts:

| Model / representation | Recall@5 | Precision@5 | MRR | Top-1 | Abstention |
|---|---:|---:|---:|---:|---:|
| Qwen3 0.6B, instructed query | 1.000 | 0.307 | 0.802 | 0.667 | 0.000 |
| Qwen3 8B, instructed query | 1.000 | 0.264 | 0.728 | 0.533 | 0.000 |
| Nomic, raw text on both sides | 0.667 | 0.133 | 0.483 | 0.400 | 0.000 |

Concrete failures included “favorite planet” retrieving “favorite breakfast,”
recent setup text outranking the cobalt-axolotl fact, and an old MongoDB
consideration appearing with the final PostgreSQL decision.

The Nomic result was a contract violation. Its upstream model card states that
retrieval documents must use `search_document:` and questions must use
`search_query:`. Kestrel supplied neither. See the
[Nomic Embed model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#usage).

## Corrections and result

- enforce the canonical 60% lexical acceptance threshold during final scoring;
- exclude rows explicitly marked `superseded_by` from general recall;
- raise default semantic eligibility from 0.1 to 0.2;
- calibrate Qwen3's raw-cosine projection floor to 0.27 and Nomic's to 0.35;
- apply Nomic's asymmetric document/query prefixes and version that document
  representation in the embedding profile, forcing safe reindexing;
- retain only candidates within 75% of the query's strongest semantic signal
  before human-like salience reranking;
- suppress prior question-shaped user turns that merely echo the current
  retrieval probe, including nonce-suffixed negative probes; and
- add the missing `(agent_id, lexical_index_id)` history index used by orphan
  cleanup after lexical backfill.

| Model | Recall@5 | Precision@5 | MRR | Top-1 | Abstention | Forbidden hits |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 0.933 | 0.933 | 0.933 | 0.933 | 1.000 | 0.000 |
| Qwen3 8B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Nomic instructed | 1.000 | 0.967 | 1.000 | 1.000 | 0.333 | 0.000 |

The 0.6B miss is the unusual-pet paraphrase, where the small model ranks grief
about a dog above the cobalt axolotl. This is a model-capacity limitation, not a
threshold that can be repaired without losing other valid paraphrases. Qwen3 8B
is the measured high-quality local choice.

## Live-path evidence

The corrected code was also driven through the HTTP invoke API against an
isolated Kite agent containing 100,023 conversation rows and 720,378 blind
token rows. An exact axolotl fact remained recallable, while the adversarial
query `absentneedle-7f3c91` returned zero memories instead of retrieving an
older question that mentioned the probe stem.

That run exposed a separate scale defect: post-backfill orphan cleanup used an
index ordered `(agent_id, lexical_index_version, lexical_index_id)`, but its
correlated lookup constrained only `agent_id` and `lexical_index_id`. SQLite
therefore scanned the 100,000-row per-agent history for every token row and
blocked the invoke path for minutes. The dedicated two-column index changes the
query plan to an exact covering lookup. After restart, status completed in
57–61 ms, the warm exact recall in about 0.3 seconds, and the negative probe in
about 0.24 seconds; requests remained responsive during a repeated backfill.

Existing Nomic corpora intentionally enter a degraded lexical-fallback window
because their legacy raw-document vectors no longer match the versioned
profile. Operators must run `kestrel embeddings reindex` for each agent after
upgrading; incompatible old and new representations are never mixed.

## Residual risk

Nomic still cannot reliably distinguish an unsupported personal question from
a semantically adjacent fact: “favorite planet” resembles “favorite breakfast,”
and “birthday” resembles dated obligations. Its answerable ranking is now
correct, but fixed cosine thresholds cannot separate those score bands without
dropping valid grief/book paraphrases. A second-stage entailment/answerability
gate needs its own latency and privacy design; do not hide that limitation by
overfitting this small suite.
