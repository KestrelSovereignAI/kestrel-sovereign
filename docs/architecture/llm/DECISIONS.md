---
type: Architecture Spec
title: Decisions — Typed Choice / Score / Noul as an LLM Modality
description: '**Canonical spec for the decisions modality** (#3424): typed questions
  evaluated against a state, returning probability distributions code can branch
  on. Peer of chat and embeddings under vendor / route / model.'
resource: /docs/architecture/llm/DECISIONS.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-10-02T00:00:00Z'
status: draft
owner: architecture
canonical: true
generated: false
privacy: public
---

# Decisions — Typed Choice / Score / Noul as an LLM Modality

> **Status:** draft spec for epic [#3424](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3424), slice [#3425](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3425). No code implements this yet. Once code lands, the code wins and this doc is a bug — update it in the same change.
>
> Read [LLM_SERVICE_ARCHITECTURE.md](../LLM_SERVICE_ARCHITECTURE.md) first. This spec adds a modality to that architecture; it does not change vendor / route / model.

## 1. What a decision is

A **decision request** is a `state` plus a set of named, independent **questions**. A **decision model** answers every question with a probability distribution. No prose is generated and nothing is parsed.

| Type | The question asks | The answer carries |
|---|---|---|
| `choice` | Which of these named options fits? | `choice` (argmax) and `probabilities` over the options |
| `score` | Where does the state sit on this ordered rubric (levels 0..N-1)? | `score` (Σ i·pᵢ) and `probabilities` over the levels |
| `noul` | Is this statement true of the state? | `p_true` in [0, 1] |

Questions are evaluated in isolation against the same state. No question can see another question's answer. Any composition happens in the caller's code.

The wire shape was introduced by TypeSafe's Jev (`POST /v1/systemone`). Today the same shape is served by:
- hosted vendors (TypeSafe, OpenRouter, Vercel, Cloudflare Workers AI);
- local runtimes (Ollama ≥ 0.35; llama.cpp and SGLang on master);
- open-weight models (Nimble, Tev1, Clef).

Kestrel treats `/v1/systemone` as the de facto contract and TypeSafe's OpenAPI document as its reference. No vendor is special in code.

### Why a modality, not a prompt pattern

Kestrel already makes many decisions of this shape. Each one either asks a chat model for JSON or a marker string and then parses it, or uses a regex. The epic's call-site census lists them. A decision model:
- answers in tens to hundreds of milliseconds;
- batches many questions over one state in a single call;
- returns distributions, so thresholds become tunable policy instead of prompt wording.

## 2. Contract

### 2.1 Types live in the SDK

The request and response types live in `kestrel_sdk.llm.decisions`. This lets features and external adapters build and read them without importing core. The types are frozen dataclasses, matching `ProviderCapabilities`. The contract version bump is `SDK_LLM_CONTRACT_VERSION` 6 → 7.

```python
@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    options: Mapping[str, str | None]      # option id -> description (None: id describes itself)

@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str
    levels: Sequence[str]                  # lowest first; level i is index i

@dataclass(frozen=True)
class NoulQuestion:
    instructions: str
    true_means: str | None = None
    false_means: str | None = None

Question = ChoiceQuestion | ScoreQuestion | NoulQuestion

@dataclass(frozen=True)
class DecisionRequest:
    state: str | Mapping[str, Any] | Sequence[Any]   # JSON-serialisable
    questions: Mapping[str, Question]                # 1..N, ids unique

@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]               # exactly the requested option ids, sums to 1

@dataclass(frozen=True)
class ScoreAnswer:
    score: float                                     # Σ i·p_i, recomputed by Kestrel
    probabilities: Sequence[float]                   # index = level, sums to 1

@dataclass(frozen=True)
class NoulAnswer:
    p_true: float

@dataclass(frozen=True)
class DecisionResult:
    answers: Mapping[str, ChoiceAnswer | ScoreAnswer | NoulAnswer]
    vendor: str
    route: str
    model: str                                       # the model that actually answered
    input_tokens: int | None
    duration_ms: int
```

**Naming.** The question types keep their wire names (`choice`, `score`, `noul`). Keeping them avoids a translation table between our types and every route's dialect.

**Typed options and levels.** Kestrel's types are deliberately narrower than TypeSafe's. TypeSafe allows `criteria` values that are strings, objects or arrays, and `instructions` may be null. Kestrel keeps option descriptions as `str | None` and `instructions` as a required non-empty `str`. That is the intersection every known route accepts, so a request built from these types is valid everywhere. When a richer description is needed, the caller puts that structure in `state` and refers to it by path, e.g. ``"Does `memory.content` answer `question`?"``.

### 2.2 Kestrel owns the answer's meaning

Adapters return raw vendor responses. A single normaliser in `kestrel_sovereign/llm/decisions/` turns them into `DecisionResult`. It enforces:

- **Exact coverage.** Every requested question has exactly one answer of the matching type. A choice answer's `probabilities` cover exactly the requested option ids, and a score answer covers exactly `len(levels)` levels.
  - A missing question, an extra key, or a type mismatch is a `DecisionProtocolError`.
  - The result is refused whole, never partially accepted. A partial answer set silently drops the questions that matter most, so the protocol error is the honest outcome.
- **Numeric sanity.** Every probability is finite and in [0, 1]. Distributions sum to 1 within `1e-3` and are then renormalised. A larger deviation is a `DecisionProtocolError`.
- **Argmax consistency.** `ChoiceAnswer.choice` is recomputed as the argmax of `probabilities`. The vendor's `choice` field is checked against it and logged on mismatch, but never trusted over the distribution.
- **Score recomputation.** `ScoreAnswer.score = Σ i·pᵢ` over Kestrel's level indices. The vendor's `score` and `legend` are ignored.

### 2.3 Kestrel owns confidence; vendor `confidence` is dropped

Vendors disagree about what `confidence` means:
- Jev uses `(K·p_max − 1)/(K − 1)`.
- Ollama uses `1 − H(p)/ln K`.
- The others don't document theirs.

A field that means different things depending on which route answered is a proxy. A caller thresholding on it would silently change behaviour on a route switch. So `DecisionResult` **does not carry the vendor's `confidence`**.

Callers threshold on `probabilities` / `p_true`. When a caller wants a concentration measure, it calls the one SDK function:

```python
def concentration(probabilities: Sequence[float]) -> float:
    """1 − H(p)/ln K. 0 = uniform, 1 = one-hot. Not a probability of being right."""
```

This is the entropy form because it uses the whole distribution, not just its peak. `noul` has no concentration: `p_true` is the whole answer.

### 2.4 Calibration is per (caller, model)

A threshold tuned on one model's distributions does not transfer to another model. Thresholds are therefore keyed by the model that answered:

```toml
[decisions.thresholds.memory_answerability]
uncalibrated = "default"         # "default" | "refuse"
default = 0.5
[decisions.thresholds.memory_answerability.models]
"ollama:local/<model-id>"       = 0.62    # example as of 2026-10; set from the eval harness
"openrouter:api/<model-id>"     = 0.55
```

- `DecisionResult.model` and `.route` let the caller look up the right threshold through `decision_threshold(caller, result)`. That lookup lives in core, so every caller reads thresholds the same way.
- When there is no entry for the answering model, the caller's declared `uncalibrated` policy applies:
  - `default`: use the declared default, and mark telemetry `calibrated=false`;
  - `refuse`: treat the call as not completed.
- Model ids appear here only as config keys, which is an allowed location under the no-hardcoded-IDs rule. Entries are written from eval-harness output (§9), not by hand-tuning.

## 3. Placement

### 3.1 One front door: `LLMService.decide`

```python
async def decide(
    self,
    request: DecisionRequest,
    *,
    caller: str,                          # stable id, e.g. "memory_answerability"; keys thresholds + telemetry
    timeout_seconds: float,               # required: every caller owns its deadline
    model_override: str | None = None,    # same selector grammar as generate()
    force_local_only: bool | None = None, # None -> self._current_force_local_only()
    session_id: str | None = None,
) -> DecisionResult: ...
```

`decide` sits beside `generate` and `get_embedding_service`, and it is the only path to a decision model. Callers already hold an `LLMService` (the answerability gate receives one; features reach `self.agent.llm_service`), so no new object needs to be threaded through.

There is no separate long-lived `DecisionService` instance. Embeddings bind to one route because vectors must stay in one space. Decisions are stateless per call, so they can be **routed per call** (§5). The implementation lives in `kestrel_sovereign/llm/decisions/`: normaliser, fit check, resolver and thresholds. `LLMService.decide` is its only public entry point.

### 3.2 Adapter surface (SDK)

`LLMAdapter` gains two optional methods, with defaults that mean "not supported". This is the same shape as `aembed`.

| Method | Default | Returns |
|---|---|---|
| `adecide(client, model, request, *, timeout)` | raises `DecisionsNotSupported` | the vendor's raw JSON response as a dict |
| `list_decision_models(client)` | `[]` | `List[DecisionModelInfo]` |

```python
@dataclass(frozen=True)
class DecisionModelInfo:
    id: str
    vendor: str
    route: str
    context_limit: int | None        # effective serving limit, not the base model's
    max_questions: int | None        # per request; None = no published cap
    max_options: int | None          # per choice/score question
    max_request_bytes: int | None    # serialised request cap, when the runtime enforces one
    parallel_questions: bool | None  # True: questions share one pass; False: evaluated serially
    created_at: str | None
```

**The adapter is the dialect seam.** It translates a `DecisionRequest` into its route's wire format and returns the raw response. All divergence lives in the adapter that owns the route:
- Ollama: string-or-null criteria; required `instructions`; 2–26 options and 1–64 questions.
- OpenRouter: extra `provider`, `session_id`, `trace` and `user` request fields; `usage.cost` in the response.
- Vercel: `boolean` in place of `noul`.

Framework code never branches on vendor.

`ProviderCapabilities` gains `supports_decisions: bool = False`. Discovery folds the resolved state into the route's `capabilities` dict (§4), exactly as it does for `supports_embeddings`.

### 3.3 Model category

`ModelCategory` gains `DECISION`. Without it, an installed Ollama decision model (`/api/tags` lists it like any other model) would appear in the chat dropdown as a chat model. `list_models` implementations classify decision-capable models as `DECISION`, and the existing `category != "chat"` filter keeps them out of chat.

## 4. Discovery

Discovery is capability-driven per route, with no pinned lists. It mirrors embedding discovery (`discover_embedding_models` → `reconcile_embedding_capabilities`):

- `discover_decision_models(vendor=, route=, use_cache=)` runs per route, single-flight, cached per instance and invalidated alongside the chat catalog.
- `reconcile_decision_capabilities()` runs inside `discover_all_models` on both the cache-hit and cache-miss paths, like its embedding twin. It sets `capabilities["supports_decisions"] = True` and records the discovered `DecisionModelInfo` list on the route. It only ever turns capability **on**; a failed discovery leaves the previous state alone and logs.

How each route discovers its models:

| Route | Discovery | Limits come from |
|---|---|---|
| OpenRouter | `GET {base_url}/models?output_modalities=decisions` (modality `text->decisions`) | `context_length` per model. Question and option caps are not published; use `None`. |
| Ollama | `/api/tags`, then `/api/show` per model, keeping models whose `capabilities` include `"decision"` | `context_limit` comes from the **`num_ctx` parameter**, never from `model_info.*.context_length`, which is the base model's limit. For example, Nimble's library page shows 256K, but the registry params set `num_ctx` to 8194. Caps (1–64 questions, 2–26 options, 64 KiB request body without images) are adapter-declared from the Ollama API docs. `parallel_questions=False` as of 0.35. |
| Any other route | the adapter's `list_decision_models` | the adapter |

**Older Ollama without the `"decision"` capability.** If an Ollama build serves `/v1/systemone` but does not report `"decision"` in `/api/show`, its models are not discovered, and the spec does not guess from names. The operator either upgrades or pins the model (§5.2). A pin is verified with a canary decision at set time, the same way `aset_embedding_route` canaries a cloud route. A model with no discoverable `num_ctx` has `context_limit=None` and cannot pass the fit check unless the pin also sets `decision_context_limit`.

## 5. Routing and model selection

### 5.1 Config

```toml
[llm]
decision_route = "auto"        # "auto" | "<vendor>[:<route>]" | "none"

[llm.vendors.ollama.routes.local]
decision_model = "auto"        # "auto" | "<model-id>" (pin)
decision_hints = []            # substring patterns, never full ids
# decision_context_limit = 8192   # only with a pinned model whose limit is not discoverable
```

### 5.2 Resolution, per call

1. If `decision_route == "none"` → `DecisionUnavailable(DISABLED)`.
2. **Candidate routes.**
   - An explicit `decision_route` is terminal: only that route.
   - `"auto"`: every route with `supports_decisions`, in `route_priority` order.
   - Then apply privacy (§6): when local-only, drop every route that is not `is_local`.
3. **Candidate model per route.**
   - A pinned `decision_model` is that model.
   - Otherwise the discovered models are filtered by `decision_hints`.
   - Exactly one survivor → that model.
   - Several survivors → `DecisionUnavailable(AMBIGUOUS_MODEL)` naming them. Kestrel does not pick one arbitrarily, because thresholds are per model (§2.4) and an arbitrary pick would silently change calibration.
4. **Fit check** (§7) of the request against the model's limits. The first candidate that fits answers.
5. No candidate fits → `DecisionUnavailable(NO_FITTING_MODEL)`, carrying each rejected candidate and its reason.

A caller's `model_override` uses the existing selector grammar (`vendor`, `vendor:route`, `vendor/model`, `vendor:route/model`). It narrows step 2 or 3 and is terminal. It never widens past the privacy filter.

Within a single `decide` call, Kestrel **never retries the request on a different route** after a dispatch failure. A transport or protocol failure on the selected candidate is raised to the caller. Silently re-asking on another model would change which calibration applies mid-call. Fallthrough happens only during resolution (steps 2–5), before anything is sent.

## 6. Privacy modes

Privacy modes **route** decisions; they do not disable them.

- `force_local_only` defaults to `self._current_force_local_only()`, the same provider the embedding resolver uses. It returns `True` (fail closed) when the bound callable raises.
- When local-only, only `is_local` routes are candidates. If none can answer, the caller gets `DecisionUnavailable(NO_LOCAL_ROUTE)`. There is **never** a silent cloud fallback.
- What is sent follows the same rule as chat: under local-only, state never leaves the host.

## 7. Context fit, caps and batching

- **Fit before dispatch.** The request is serialised in its route's dialect, and its size is estimated with the token heuristics core uses for context budgeting. The estimate must leave headroom for the model's own prompt framing. If `context_limit` is unknown, or the estimate exceeds it, or the serialised request exceeds `max_request_bytes`, that candidate is skipped (§5.2). Kestrel **never truncates `state`**: a decision about half a document is a different decision.
- **Per-question cost on serial routes.** Some backends re-send the full state for each question (`parallel_questions=False`). The fit check is per question there, but wall time scales with the number of questions. `DecisionModelInfo.parallel_questions` is surfaced so latency-sensitive callers can choose a route or narrow their question set.
- **Splitting by question cap.** When a request exceeds a route's `max_questions`, Kestrel splits it into consecutive chunks on that same route and model, then merges the answers. This is safe by contract, because questions are independent and cannot see each other. It is not a fallback: the same model answers every question. A choice or score question with more options than `max_options` is a fit failure for that candidate and is never split.

## 8. Failure semantics, timeouts and accounting

**Exceptions.** Every outcome other than a complete, normalised `DecisionResult` is an exception:

| Exception | Meaning |
|---|---|
| `DecisionUnavailable(reason, candidates)` | Nothing was sent. Reasons: `DISABLED`, `NO_ROUTE`, `NO_LOCAL_ROUTE`, `AMBIGUOUS_MODEL`, `NO_FITTING_MODEL`. |
| `DecisionTimeout` | `timeout_seconds` elapsed. Enforced by `asyncio.timeout` in `decide`, and also passed to the adapter's HTTP client. The Ollama route must set an explicit HTTP timeout, because the chat `AsyncClient` has none. |
| `DecisionTransportError` | Network or HTTP failure, including a 404 for `/v1/systemone` on a runtime too old to serve it. |
| `DecisionProtocolError` | The response violated §2.2. |

**Fail-closed is the caller's policy, not the service's.** For example, the answerability gate maps any exception to `completed=False` and its existing lexical-evidence path. Each caller documents its policy at its call site.

**Usage accounting.** Every dispatched call, successful or not, records through the same sinks chat uses:
- `_track_model_usage(model, provider, tokens=input_tokens)`;
- `_log_llm_call(...)` with `metadata={"modality": "decision", "caller": caller, "question_count": n, "calibrated": bool}`, and with the route's reported cost when present (OpenRouter `usage.cost`);
- the Prometheus `LLM_CALLS` / `LLM_DURATION` series, labelled with modality.

The same content-redaction rules apply to the `llm_calls` row. State and question text are prompt content.

## 9. Eval harness

Thresholds and model choices are set from measurement.

- **Samples.** Each caller has a labelled sample set: `state`, `questions`, and the expected answers.
  - Samples committed to the repo are synthetic or public (`tests/evals/decisions/<caller>/*.jsonl`).
  - Operator-local samples drawn from real memory or turns live under the agent data directory and are never committed.
- **Runner.** `kestrel decisions eval --caller <id> [--route <selector>]` runs every sample against each candidate model. It reports:
  - accuracy;
  - Brier score and expected calibration error;
  - latency p50 / p95;
  - a proposed threshold per model.
- **Recording.** An accepted proposal is written into `[decisions.thresholds.<caller>.models]` (§2.4). The run's sample-set hash and date go into a comment beside the entry, so a threshold can always be traced to the evidence that set it.

## 10. Feature (SDK) surface

Features call `self.agent.llm_service.decide(...)` with SDK types, the same duck-typed access they already use for `generate`. Feature code needs no import from core: the request, answer and exception types and `concentration()` all live in `kestrel_sdk.llm.decisions`.

Isolated (out-of-process) features have no LLM RPC today. Decisions do not add one; that is a separate gap that also covers `generate`.

## 11. Explicitly out of scope

- **Serving `/v1/systemone` inbound.** Kestrel already serves `/v1/models` and `/v1/chat/completions`. Exposing decisions to other agents would be a later decision.
- **Images in `state`.** The contract extension is reserved. When a route we use accepts images (Clef, OpenAI's Decisions API, Ollama main), `DecisionRequest` gains an `images` field and `DecisionModelInfo` gains `accepts_images`. Requests carrying images route only to models that accept them. Until then there is no field, and there is no silent image drop.
- **A role table** mapping callers to models. Callers pass `model_override` from their own config, as the answerability gate does today.
- **Runtime setters and endpoints** (`/api/decision/models`, `/api/decision/settings`). Configuration is file-based until a caller needs a runtime switch. Their shape will mirror `/api/embedding/*`.

## 12. Rollout

Slices are tracked on #3424. Each one lands with tests that fail without it.

1. This spec.
2. **SDK contract plus core.**
   - SDK types, adapter methods, `supports_decisions`, `ModelCategory.DECISION` and the contract version bump. The SDK ships first and core bumps its pin.
   - Core: `LLMService.decide`, the normaliser, resolution, fit check, accounting and thresholds.
   - Routes: OpenRouter and Ollama `adecide` / `list_decision_models`. Two routes ship together so the dialect seam is exercised by real divergence.
3. **Eval harness.**
4. **First caller:** the memory answerability gate, measured against the current gate on recorded retrievals.
5. Response audit, as a `score` question.
6. Reflection capture pre-gate, and batched sleep attestation.
7. Per-heuristic migrations (tagger, schema router, continuation intent, turn classifier, injection scoring, GitHub-app should-respond), each with labelled samples.
8. Images in state.
9. Decision-driven model routing, after the per-turn routing freeze.

## Related

- [LLM_SERVICE_ARCHITECTURE.md](../LLM_SERVICE_ARCHITECTURE.md): vendor / route / model, discovery, routing, no hardcoded model IDs.
- [PROVIDER_PLUGINS.md](PROVIDER_PLUGINS.md): the adapter contract that `adecide` / `list_decision_models` extend.
- [MEMORY_SYSTEM.md](../MEMORY_SYSTEM.md): second-stage answerability, the first caller.
- TypeSafe System One API reference: <https://docs.typesafe.ai/api.md> (OpenAPI: <https://api.typesafe.ai/openapi.json>).
- Ollama decision models: <https://docs.ollama.com/api/systemone>.
- OpenRouter Decisions API: <https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request.md>.
