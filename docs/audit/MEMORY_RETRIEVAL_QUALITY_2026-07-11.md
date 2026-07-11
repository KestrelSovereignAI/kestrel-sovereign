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

## Round-one residual risk

Nomic still cannot reliably distinguish an unsupported personal question from
a semantically adjacent fact: “favorite planet” resembles “favorite breakfast,”
and “birthday” resembles dated obligations. Its answerable ranking is now
correct, but fixed cosine thresholds cannot separate those score bands without
dropping valid grief/book paraphrases. A second-stage entailment/answerability
gate therefore needed an explicit latency and privacy design; a threshold-only
patch would merely overfit this small suite.

## Follow-up: second-stage answerability (#2378)

Suite v2 expands the abstention set from 3 to 12 unsupported personal
attributes. Correctly instructed Nomic without a judge scored 1.000 recall and
0.500 abstention. A 3B Llama judge was empirically inadequate: it preserved
recall but did not improve the expanded abstention score. This is evidence that
"an LLM call" alone is not a quality guarantee.

A local Gemma 31B evidence judge produced 1.000 recall, precision, MRR, top-1,
and abstention across all 27 queries. On the measured host its 21 non-empty
candidate calls averaged 4.46 seconds warm and peaked at 7.20 seconds. The
production gate makes one batched call per recall and enforces a 12-second
timeout. Failures retain exact lexical evidence only; lexical-only retrieval
bypasses the judge entirely.

Suite v2 also exposed one Qwen3 8B miss that v1 did not: the unsupported phone
number query retrieved the generic welcome message, yielding 0.917 abstention
without a judge. The answerability stage therefore applies to all configured
bi-encoder models by default, not only Nomic; a provider may opt out only by
declaring `embedding_answerability_gate = false` for a separately validated
evidence layer.

Run the measured configuration with:

```bash
uv run python scripts/benchmark_memory_quality.py \
  --model nomic-embed-text:latest --judge-model gemma4:31b
```

Candidate content follows the same privacy boundary as the agent's normal LLM
turn: local-only modes force a local judge, while cloud routing is possible only
when the active privacy policy permits it. No answerability payload is stored by
the memory layer.

The benchmark adapter requests Ollama JSON mode with temperature zero. The
production LLM service relies on the strict JSON instruction and may use the
active model's normal sampling defaults, so these quality and parse-success
figures are a model-qualification result, not a guarantee for every production
route. Set `retrieval.memory_answerability_model` to the qualified model, then
repeat live-path verification. The global
`retrieval.memory_answerability_gate = false` kill switch removes the added
call and preserves normal semantic recall if judge latency, cost, or failure
rate is unacceptable.

The remaining risk is judge capability: the 3B result proves that small models
can return syntactically valid but incorrect evidence decisions. Deployments
using Nomic should qualify their chosen chat model with this suite; unavailable,
timed-out, or malformed judges fail closed rather than leaking a topical guess.
