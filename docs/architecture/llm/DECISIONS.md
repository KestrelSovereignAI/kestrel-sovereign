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

> **Status:** draft spec for epic [#3424](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3424), slice [#3425](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3425). Nothing implements this yet. Once code lands, the code is authoritative and any disagreement in this doc is a bug — update the doc in the same change.
>
> Read [LLM_SERVICE_ARCHITECTURE.md](../LLM_SERVICE_ARCHITECTURE.md) first. This spec adds a modality to that architecture. It does not change vendor / route / model.

## 1. What a decision is

A **decision request** is a `state` plus a set of named, independent **questions**. A **decision model** answers every question with a probability distribution. It generates no prose, so there is nothing to parse.

| Type | The question asks | The answer carries |
|---|---|---|
| `choice` | Which of these named options fits? | `choice` (the argmax) and `probabilities` over the options |
| `score` | Where does the state sit on this ordered rubric (levels 0..N-1)? | `score` (Σ i·pᵢ) and `probabilities` over the levels |
| `noul` | Is this statement true of the state? | `p_true` in [0, 1] |

Each question is evaluated in isolation against the same state. No question sees another question's answer. Any composition happens in the caller's code.

TypeSafe's Jev introduced the wire shape (`POST /v1/systemone`). The same shape is now served by hosted vendors (TypeSafe, OpenRouter, Vercel, Cloudflare Workers AI), local runtimes (Ollama ≥ 0.35; llama.cpp and SGLang on master) and open-weight models (Nimble, Tev1, Clef). Kestrel treats `/v1/systemone` as the de facto contract and uses TypeSafe's OpenAPI document as its reference. No vendor is special in code.

### Why a modality, not a prompt pattern

Kestrel already makes many decisions of this shape. Today each one either asks a chat model for JSON or a marker string and parses the reply, or applies a regex; the epic's call-site census lists them. A decision model does better on three counts:
- it answers in tens to hundreds of milliseconds;
- it batches many questions over one state;
- it returns distributions, so thresholds become tunable policy instead of prompt wording.

## 2. Contract

### 2.1 Types live in the SDK

The request and response types live in `kestrel_sdk.llm.decisions`. Features and external adapters can build and read them without importing core. They are frozen dataclasses, like `ProviderCapabilities`. Adding them bumps `SDK_LLM_CONTRACT_VERSION` from 6 to 7.

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
    questions: Mapping[str, Question]

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
    thresholds: Mapping[str, float]                  # per question id, resolved for this model (§2.4)
    calibrated: bool | None                          # False: "default" policy applied; None: caller declares no thresholds
    input_tokens: int | None
    duration_ms: int
```

**Naming.** The question types keep their wire names (`choice`, `score`, `noul`), so there is no translation table between our types and every route's dialect.

**A deliberately narrow contract.** TypeSafe allows `criteria` values to be strings, objects or arrays, and allows `instructions` to be null. Kestrel narrows both:
- option descriptions are `str | None`;
- `instructions` is a required, non-empty `str`.

That is the intersection every known route accepts, so a valid request is valid everywhere. When a richer description is needed, put the structure in `state` and refer to it by path, e.g. ``"Does `memory.content` answer `question`?"``.

### 2.2 Request validation

`decide` validates the request **before** routing or serialising for any route. A violation raises `DecisionRequestInvalid` naming the rule, and nothing is sent.

| Rule | Bound | Why |
|---|---|---|
| Question count | 1 – `MAX_QUESTIONS` (64) | 64 is the smallest published route cap (Ollama). A request valid here is valid on every known route, and §7 never splits a request. |
| Question / option ids | `^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$`, unique within their map | Ids appear as JSON keys in every dialect and in telemetry. |
| `choice` options | 2 – `MAX_OPTIONS` (26) | One option is not a decision. 26 is the smallest published cap (Ollama's letter scoring). |
| `score` levels | 2 – `MAX_OPTIONS` (26) | Same reasons. |
| `instructions`, descriptions | non-empty after strip; descriptions may be `None` | A blank prompt is not a question. |
| `instructions` | ≤ `MAX_INSTRUCTIONS_CHARS` (4,096) | A question is one narrow judgement, not a document. |
| Option / level / `true_means` / `false_means` text | ≤ `MAX_DESCRIPTION_CHARS` (1,024) each | Same reason. |
| `state` | a strict JSON value tree; see below | Every route must read the same state that Kestrel measured. |
| Whole request | canonical JSON (state and questions) ≤ `MAX_REQUEST_BYTES` (1 MiB) | Bounds every later step. See the measurement procedure below. |

**A strict JSON value tree for `state`.** A value is accepted only if it is built from:
- `dict` with **`str` keys only**, `list` or `tuple`;
- `str`, `bool` or `None`;
- `int`, or a **finite** `float`.

Validation rejects, with `DecisionRequestInvalid`:
- a non-`str` key. Python's encoder would silently turn `1` into `"1"`, and that can collide with an existing `"1"` key;
- `NaN` or `±Infinity`;
- any other type;
- a container that contains itself (a cycle);
- nesting deeper than `MAX_STATE_DEPTH` (64).

These rules ensure every server parses the same value Kestrel measured.

**How the whole-request size is measured.** Validation measures the request **once**, in a single bounded pass, and stops early:
1. Check the text-field limits. They are plain length checks.
2. Walk the `state` tree with an explicit stack. The walk enforces the tree rules above, tracking the containers on the current path to detect cycles. As it goes, it adds up each node's UTF-8 encoded length, including keys and punctuation.
3. Abort with `DecisionRequestInvalid(MAX_REQUEST_BYTES)` as soon as the running total, including the questions, passes the cap.
4. Only a request that passes is encoded, as canonical JSON with `allow_nan=False`.

**Validation produces an immutable snapshot, before the first `await`.** The same walk builds a private, deep-frozen copy of the request:
- every `dict` becomes a fresh read-only mapping;
- every `list` becomes a `tuple`;
- every question's options and levels are copied the same way.

The snapshot's canonical JSON bytes are computed once from that copy. All later steps use **only the snapshot**: sizing, the fit check, privacy routing, and the adapter's dialect serialisation (`adecide` receives the snapshot, never the caller's object). A caller that mutates its own `state` while `decide` awaits discovery, a canary or the network cannot change what is sent. What goes out is exactly what was validated and measured.

The early stop means an oversized `state` costs at most about the cap in encoding work; it is never fully serialised. Later steps use this measured size:
- The per-route fit check (§7) uses it for its token estimate and does not re-tokenise.
- Each adapter's dialect serialisation is at most a constant factor of it, because dialects rename keys and do not expand content.

These bounds, the validation function and the snapshot type are module constants and code in `kestrel_sdk.llm.decisions`. They belong in the SDK because adapters receive the snapshot type and features build requests against the same limits. Core calls the SDK validator; it does not keep its own copy. They are policy, not configuration, in the same way loop-policy constants are. A future route with stricter caps tightens §7's per-candidate fit check; it does not change these constants.

### 2.3 Kestrel owns the answer's meaning

Adapters return the raw vendor response. One normaliser in `kestrel_sovereign/llm/decisions/` converts it into `DecisionResult`, and it enforces four rules:

- **Exact coverage.** Every requested question gets exactly one answer of the matching type. A `choice` answer's `probabilities` cover exactly the requested option ids, and a `score` answer covers exactly `len(levels)` levels. A missing question, an extra key, or a type mismatch raises `DecisionProtocolError`. The whole result is refused, never partially accepted, because a partial answer set would silently drop exactly the questions that matter.
- **Numeric sanity.** Every probability must be finite and in [0, 1]. A distribution that sums to 1 within `1e-3` is renormalised. Anything further off raises `DecisionProtocolError`.
- **Argmax consistency.** `ChoiceAnswer.choice` is recomputed as the argmax of `probabilities`. If the vendor's `choice` disagrees, the mismatch is logged and the vendor value is discarded.
- **Score recomputation.** `ScoreAnswer.score = Σ i·pᵢ` over Kestrel's own level indices. The vendor's `score` and `legend` are ignored.

### 2.4 Kestrel owns confidence; vendor `confidence` is dropped

Vendors disagree on what `confidence` means. Jev computes `(K·p_max − 1)/(K − 1)`, Ollama computes `1 − H(p)/ln K`, and other vendors don't document theirs. A field whose meaning depends on which route answered is a proxy: a caller that thresholds on it silently changes behaviour when the route changes. **`DecisionResult` therefore does not carry the vendor's `confidence` at all.**

Callers threshold on `probabilities` or `p_true`. A caller that wants a concentration measure uses the single SDK function:

```python
def concentration(probabilities: Sequence[float]) -> float:
    """1 − H(p)/ln K. 0 = uniform, 1 = one-hot. Not a probability of being right."""
```

This is the entropy form because it uses the whole distribution, not just its peak. Validation (§2.2) guarantees K ≥ 2. A `noul` answer has no concentration; `p_true` is the whole answer.

### 2.5 Calibration is per (caller, model), and the service resolves it

A threshold tuned on one model's distributions does not transfer to another model. Thresholds are therefore keyed by caller, by the model that answers, and by question id:

```toml
[decisions.thresholds.memory_answerability]
uncalibrated = "default"         # "default" | "refuse"
default = { answers = 0.5 }      # per question id
[decisions.thresholds.memory_answerability.models."ollama:local/<model-id>"]
answers = 0.62                   # example as of 2026-10; written from the eval harness (§9)
```

The service resolves calibration, not the caller. This is the only way telemetry can report the truth.

**A model is calibrated for a request only if its entry has a threshold for every question id in that request.** Coverage is all-or-nothing: an entry that covers `answers` but not `relevant` counts as uncalibrated for a request that asks both. Thresholds from two calibrations are never mixed.

- Under **`refuse`**, a model that is uncalibrated for the request is rejected for this call during resolution (§5.2), before anything is sent.
- Under **`default`**, an uncalibrated model can still answer. Every threshold on the result then comes from `default`, and `calibrated=False` is set on the result and in telemetry. The `default` table itself must cover every question id the request asks; a gap there is a configuration error, raised as `DecisionRequestInvalid` before routing.
- **Callers with no thresholds table.** A caller that uses distributions directly (for example, ranking by `p_true`) declares no `[decisions.thresholds.<caller>]` table. Its results carry `thresholds={}` and `calibrated=None`, meaning not applicable.

`DecisionResult.thresholds` is the only threshold source a caller reads, so every caller applies calibration the same way. Model ids appear here only as config keys, which the no-hardcoded-IDs rule permits.

## 3. Placement

### 3.1 One front door: `LLMService.decide`

```python
async def decide(
    self,
    request: DecisionRequest,
    *,
    caller: str,                          # stable id, e.g. "memory_answerability"; keys thresholds + telemetry
    timeout_seconds: float,               # required: every caller owns its deadline
    model_override: str | None = None,    # §5.3 grammar
    local_only: bool = False,             # may only TIGHTEN privacy (§6)
    session_id: str | None = None,
) -> DecisionResult: ...
```

`decide` sits beside `generate` and `get_embedding_service`, and it is the only path to a decision model. Callers already hold an `LLMService`: the answerability gate is handed one, and features reach `self.agent.llm_service`. No new object has to be threaded through.

There is no long-lived `DecisionService` instance. Embeddings bind to one route because vectors must stay in one space. Decisions are stateless per call and are **resolved per call** (§5). The implementation (validation, normaliser, resolver, fit check, thresholds and accounting) lives in `kestrel_sovereign/llm/decisions/`, and `LLMService.decide` is its only public entry point.

### 3.2 Adapter surface (SDK)

`LLMAdapter` gains two optional methods, following the same pattern as `aembed`: the default means "not supported".

| Method | Default | Returns |
|---|---|---|
| `adecide(client, model, request, *, timeout)` | raises `DecisionsNotSupported` | the vendor's raw JSON response, as a dict. `request` is the validated, immutable snapshot (§2.2). |
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

**The adapter is the dialect seam.** It translates a `DecisionRequest` into its route's wire format and returns the raw response. Every divergence lives in the adapter that owns the route, and framework code never branches on vendor. The divergences known today:
- **Ollama:** criteria descriptions must be a string or null; `instructions` is required; 2–26 options; 1–64 questions; a 64 KiB request body.
- **OpenRouter:** extra `provider`, `session_id`, `trace` and `user` request fields; `usage.cost` in the response.
- **Vercel:** `boolean` in place of `noul`.

`ProviderCapabilities` gains `supports_decisions: bool = False`, which an adapter sets when its route *can* serve decisions. Which models a route currently serves is discovery state (§4); the static flag does not carry it.

### 3.3 Decision capability is per route; chat listings exclude decision-only models

Decision models are a separate discovery facet, as embedding models are. `discover_decision_models` runs per route, with no per-vendor collapse, and its result lives on the route (§4). It does **not** add a category to the vendor's shared chat catalog. A model that serves chat on one route and decisions on another keeps its chat category.

Chat listings must still not offer a model that cannot chat. Today an installed Ollama decision model appears in `/api/tags` exactly like any other model. An adapter's `list_models` therefore drops a model when the runtime reports `decision` capability **and does not report chat (`completion`) capability** for it. This uses the runtime's own capability report; the model name is never used to decide.

## 4. Discovery

Discovery is capability-driven and per route, with no pinned lists. It follows the embedding discovery pair (`discover_embedding_models` → `reconcile_embedding_capabilities`), with one deliberate difference: decision state can turn **off**.

- `discover_decision_models(vendor=, route=, use_cache=)` runs single-flight and is cached per instance. It is invalidated together with the chat catalog.
- `reconcile_decision_capabilities()` runs inside `discover_all_models`, on both the cache-hit and cache-miss paths. On a **successful** discovery it **replaces** the route's decision model list with what the route reports now, which may be empty. A model that has been uninstalled or withdrawn stops being a candidate at the next refresh. On a **failed** discovery the last successful list is kept, the route is marked `decision_discovery_stale_since=<time>`, and a WARNING is logged. It is a WARNING rather than an ERROR because a local daemon that is down, or too old to serve decisions, is a normal host state, and every catalog refresh re-tries. A stale list still produces candidates, because availability is not a safety property. A dispatch to a model that has since disappeared fails as `DecisionTransportError` (§8) and is never retried elsewhere.

How each route discovers its models:

| Route | Discovery | Where limits come from |
|---|---|---|
| OpenRouter | `GET {base_url}/models?output_modalities=decisions` (modality `text->decisions`) | `context_length` per model. Question and option caps are unpublished, so they are `None`; §2.2's bounds still apply. |
| Ollama | `/api/tags`, then `/api/show` for each model; keep models whose `capabilities` include `"decision"` | `context_limit` comes from the **`num_ctx` parameter**, never from `model_info.*.context_length`, which is the base model's limit. For example, the library page lists Nimble at 256K, but the registry params set `num_ctx` to 8194. The caps (1–64 questions, 2–26 options, 64 KiB body without images) come from the Ollama API docs and are declared by the adapter. `parallel_questions=False` as of 0.35. |
| Any other route | the adapter's `list_decision_models` | the adapter |

### 4.1 Pins are verified at reconcile time

A route may pin `decision_model` (§5.1). This covers Ollama builds that serve `/v1/systemone` but don't report `"decision"` in `/api/show`, and any route whose discovery is unavailable. The spec never infers a decision model from its name.

- A pin is verified with a **canary decision**: one `noul` question over a fixed synthetic state, sent through the route's `adecide`.
- The canary runs whenever the discovery cache is refreshed: at boot, on a cache miss, and on an explicit refresh. On the cache-hit path, reconciliation reuses the cached canary outcome and makes no network call.
- Each canary has its own deadline, `decision_canary_timeout_seconds` (default 10). Canaries for different routes run concurrently, so one hung route cannot stall discovery past that deadline.
- How each outcome changes the pin:

| Outcome | Pin state afterwards |
|---|---|
| Passes | `verified`; the pin becomes a candidate. |
| Protocol failure (the model answered with a malformed result), or a definitive not-found (HTTP 404 for the endpoint or the model) | `unverified`, recorded with the reason. Logged at ERROR. |
| Timeout, cancellation or another transport failure | **unchanged**: a verified pin stays verified, and a never-verified pin stays unverified. The pin is marked `canary_stale_since=<time>`. A transient outage never flips a verified pin off; a later real dispatch failure surfaces through §8 instead. |

- No unverified pin is ever dispatched to.
- Canary calls are recorded like any other decision call (§8), under `caller="kestrel.canary"`.
- A model with no discoverable `num_ctx` has `context_limit=None` and cannot pass the fit check unless the pin also sets `decision_context_limit`.

## 5. Routing and model selection

### 5.1 Config

```toml
[llm]
decision_route = "auto"        # "auto" | "<vendor>[:<route>]" | "none"
decision_canary_timeout_seconds = 10   # per-pin canary deadline (§4.1)

[llm.vendors.ollama.routes.local]
decision_model = "auto"        # "auto" | "<model-id>" (pin; verified per §4.1)
decision_hints = []            # substring patterns, never full ids
# decision_context_limit = 8192   # only with a pinned model whose limit is not discoverable
```

### 5.2 Resolution, per call

The steps run in order. No decision request is sent until step 4, and **no network contact of any kind**, discovery and canaries included, reaches a route that the privacy filter has removed.

1. **Disabled.** If `decision_route == "none"`, raise `DecisionUnavailable(DISABLED)`.
2. **Candidate routes**, filtered **before** any discovery:
   1. Start from configured routes whose adapter declares `supports_decisions` (§3.2), in `route_priority` order. A route whose adapter has no decision surface at all is not a candidate. An explicit `decision_route` is terminal, so only that route remains.
   2. Apply `model_override` (§5.3).
   3. Apply privacy (§6): under effective local-only, drop every route that is not `is_local`.
   4. **Then** read decision state for the remaining routes only. A remaining route whose decision discovery has never run (a cold cache) is discovered now, together with its pin canary (§4.1). This is the only discovery `decide` ever triggers, and it is scoped to these routes. It is the same rule the chat path follows, where local-only turns skip discovery that would contact the cloud.
   5. A remaining route with no pin whose discovery found no decision models stays in the list. Step 3 rejects it with `NO_MODELS`, so the outcome is always attributable to a route.
3. **Walk the routes in order.** For each route, resolve its model and stop at the first one that passes every check. A route that fails a check is skipped, and its rejection reason is recorded.
   - **Pick the model.** A route with no pin and no discovered decision models is rejected with `NO_MODELS`. A verified pin is the route's model. An unverified pin rejects the route with `UNVERIFIED_PIN`, and the route's discovered models are not consulted instead, because the operator's pin is authoritative for that route. With no pin, the discovered models are filtered by `decision_hints`. If exactly one survives, that is the model. If several survive, reject the route with `AMBIGUOUS_MODEL` and name the survivors. Kestrel never picks among them arbitrarily, because thresholds are per model and an arbitrary pick would silently change calibration.
   - **Calibration** (§2.5). Under `refuse`, reject the route with `NOT_CALIBRATED` if its model is uncalibrated for this request.
   - **Fit check** (§7). Reject the route with `NO_FIT` if the request exceeds the model's limits.
4. **Dispatch** to the first route that passed. If none passed, raise `DecisionUnavailable(NO_CANDIDATE)` with each route's rejection reason. One ambiguous or ill-fitting route never blocks a usable route earlier or later in the order.

Within one `decide` call, Kestrel **never re-sends the request to another candidate** after a dispatch failure. A transport or protocol failure from the selected candidate is raised to the caller, because silently re-asking a different model would change which calibration applies. Moving on to another route happens only during resolution (step 3), before anything is sent.

### 5.3 `model_override` grammar

The `generate` path's `resolve_provider_routing` treats a selector without `/` as a model id. Decisions route by a different rule, so they define their own grammar. It is the explicit form of the selectors `mandate._resolve_model_selector` already accepts:

| Form | Meaning |
|---|---|
| `<vendor>` | routes of that vendor only |
| `<vendor>:<route>` | that route only |
| `<vendor>/<model>` | that model, on any route of that vendor |
| `<vendor>:<route>/<model>` | that model on that route |

A bare model id is **not** accepted, because one id can exist on several vendors with different calibration. The `cheap` alias is not accepted either, because it names a chat model.

An override is terminal: it narrows the candidates and never widens them. When the override names a model, it replaces model selection on every route it leaves:
- **Discovered models.** On a route with no pin, the named model is used if that route's discovery reports it. If discovery found other decision models but not this one, the route is rejected with `NOT_SERVED`. If discovery found no decision models at all, `NO_MODELS` takes precedence, as in §5.2. `decision_hints` do not apply, because an exact model needs no narrowing.
- **Pins.** On a route with a verified pin, the named model must equal the pin; otherwise the route is rejected with `PIN_CONFLICT`. An unverified pin still rejects the route with `UNVERIFIED_PIN`. The operator's pin is never bypassed and never silently ignored. If the override named a single route (`<vendor>:<route>/<model>`), that rejection is the whole outcome: `NO_CANDIDATE`, with the one route's reason. If an explicit `decision_route` is set and the override names a different route or vendor, `decide` raises `DecisionUnavailable(SELECTOR_CONFLICT)`. The operator's routing decision and the caller's request disagree, and neither silently wins.

## 6. Privacy modes

Privacy modes **route** decisions to local models. They never disable decisions.

- The effective restriction is **`local_only OR self._current_force_local_only()`**. The live privacy provider is the same one the embedding resolver reads, and it fails closed (`True`) when the bound callable raises. A caller can tighten privacy with `local_only=True`, but it cannot loosen it: there is no parameter that turns the live restriction off.
- Under effective local-only, only `is_local` routes are candidates. If no route survives the privacy filter, the caller gets `DecisionUnavailable(NO_LOCAL_ROUTE)`. If local routes survive but every one is rejected (for example `NO_MODELS` or `NO_FIT`), the caller gets `NO_CANDIDATE` with those per-route reasons. **There is never a silent cloud fallback.**
- The effective restriction is evaluated once, at the start of `decide`, together with the invocation context (§8). It applies to the whole call.
- **Background discovery is not a decision call.** Catalog discovery at boot and on refresh (§4) is not triggered by `decide`. It carries no request content, and pin canaries send only a fixed synthetic state. Inside a `decide` call, discovery is scoped to the routes that survive the privacy filter (§5.2, step 2).

## 7. Context fit, caps and batching

- **Fit before dispatch.** The request's size, measured once during validation (§2.2), is converted to a token estimate with the heuristics core uses for context budgeting. The estimate leaves headroom for the model's own prompt framing. `max_request_bytes` is checked against the measured canonical size plus a fixed envelope allowance (`ENVELOPE_BYTES`, 1 KiB) for the fields a route adds around it (`model`, `keep_alive`, OpenRouter's `provider`/`user`). Dialects rename keys and add envelope fields; they do not expand content. A candidate is skipped (§5.2) if any of these holds:
  - its `context_limit` is unknown;
  - the estimate exceeds `context_limit`;
  - the serialised request exceeds `max_request_bytes`;
  - the request exceeds the candidate's `max_questions`;
  - any question exceeds the candidate's `max_options`.
- **Never truncate `state`.** A decision about half a document is a different decision.
- **One `decide` is one dispatch.** Kestrel never splits a request across several dispatches. §2.2 caps every request at 64 questions, so a valid request fits every known route's question cap. A caller with more independent questions makes more `decide` calls and owns their deadlines; no single call hides a fan-out.
- **Serial routes.** Some backends re-send the full state for every question (`parallel_questions=False`). The fit check is per question on those routes, but wall time grows with question count. `DecisionModelInfo.parallel_questions` is exposed so latency-sensitive callers can pick a route or narrow their question set.

## 8. Failure semantics, timeouts, cancellation and accounting

### 8.1 Exceptions

Anything other than a complete, normalised `DecisionResult` is raised as an exception.

| Exception | Meaning |
|---|---|
| `DecisionRequestInvalid(rule)` | The request failed validation (§2.2). Nothing was sent. |
| `DecisionUnavailable(reason, rejections)` | Nothing was sent. `reason` is one of `DISABLED`, `SELECTOR_CONFLICT`, `NO_ROUTE` (no configured route remains after the override), `NO_LOCAL_ROUTE` (no route survived the privacy filter) or `NO_CANDIDATE` (routes remained, but each was rejected). `rejections` lists each route with its reason: `NO_MODELS`, `AMBIGUOUS_MODEL`, `NOT_CALIBRATED`, `NO_FIT`, `UNVERIFIED_PIN`, `NOT_SERVED` or `PIN_CONFLICT`. `NO_ROUTE` means no configured route survived the override. If routes survived but every one was rejected, including the case where every route discovered zero models, the reason is `NO_CANDIDATE`. |
| `DecisionTimeout` | `timeout_seconds` elapsed. The deadline covers the whole call: resolution, the fit check, dispatch and normalisation. It is enforced with `asyncio.timeout` in `decide` and is also passed to the adapter's HTTP client. The Ollama route must set an explicit HTTP timeout, because the chat `AsyncClient` has none. |
| `DecisionTransportError` | A network or HTTP failure. This includes a 404 for `/v1/systemone` on a runtime too old to serve it, and a model that has disappeared since discovery. |
| `DecisionProtocolError` | The response violated §2.3. |

Fail-closed behaviour is **the caller's policy, not the service's**. For example, the answerability gate maps every exception to `completed=False` and its existing lexical-evidence path. Each caller documents its policy at its call site.

### 8.2 Cancellation and ambiguous delivery

- **Cancellation propagates.** `decide` does not swallow `CancelledError`. A Stop or caller cancellation aborts the in-flight HTTP request and re-raises.
- **No retry, ever.** Not after a timeout, a cancellation or a transport error. The request may already have been processed, and possibly billed, by the vendor. Re-sending would charge twice and could answer from a different model.
- **Accounting still happens.** A dispatched call that ends in a timeout, cancellation or transport error is still recorded (§8.3). Its usage is recorded as unknown (`usage_available=False`) when the route did not report it.
- **Recording survives cancellation.** The record is written by a separate task protected with `asyncio.shield`. `decide` awaits that task, for at most `DECISION_RECORD_TIMEOUT` (2 s), in its `finally` path, and then re-raises the original exception (`CancelledError` included). Cancelling `decide` does not cancel the shielded write. If the write overruns its bound, it is handed to the service's supervised background tasks to finish, and a WARNING notes the late record. The caller's exception is never replaced by a recording error, and recording never re-raises into the caller.

### 8.3 Accounting and redaction

**Freeze the context first.** At entry, `decide` freezes the invocation context, using the same `LLMInvocationContext` snapshot `generate` uses. It passes that snapshot explicitly to every telemetry write, including the writes on timeout, cancellation and protocol-error paths. Telemetry never re-resolves ambient context after an await.

**Content is never logged.** Decision telemetry never records `state`, question text, option descriptions, or vendor error bodies that echo the request. The `llm_calls` row's prompt and response columns are written as null. Only content-free fields are kept:
- caller, vendor, route and model;
- question count and question types;
- serialised request size in bytes;
- duration and success;
- error class (not message);
- tokens and cost when reported;
- `calibrated`.

This is stricter than chat telemetry, and that is deliberate: decision `state` is often private memory or turn content passed by a background caller, with no user-visible turn to attribute it to.

**One modality-aware recorder.** Today `_log_llm_call` does two jobs: it writes the content-free sinks, and it increments the chat-only Prometheus series (`LLM_CALLS`, `LLM_DURATION`, `LLM_TOKENS`). Routing decisions through it unchanged would count every decision as a chat call. Slice 2 therefore extracts the recording into a single recorder that takes `modality: Literal["chat", "embedding", "decision"]`. Chat calls go through it with `modality="chat"` and produce exactly the rows, series and metering they produce today. Every dispatched decision, successful or not, is recorded through it with `modality="decision"`:

| Sink | What is written for a decision |
|---|---|
| `model_usage` | input tokens against (model, provider), as for chat |
| `llm_calls` | the frozen context and only the content-free fields above. `metadata` carries `{"modality": "decision", "caller", "question_count", "calibrated", "usage_available"}`, plus `provider_reported_cost_usd` when the route reports a cost. |
| Prometheus | **dedicated** series, selected by modality: `kestrel_llm_decision_calls_total{provider, model, caller, success}`, `kestrel_llm_decision_duration_seconds{provider, model, caller}` and `kestrel_llm_decision_tokens_total{model}`. They are defined in `kestrel_sdk.metrics` next to `LLM_CALLS`. Decisions never touch the chat series, so their labels and meaning do not change. |
| Metering callback | invoked under the same conditions as chat: `success`, usage available, and a token breakdown present. Decisions are billable tokens, so a callback written against the original signature still receives every decision as an ordinary billing event. `modality` is added to the existing opt-in set, `_metering_callback_optional_kwargs` (alongside `cost` and the cache-token fields), and is passed only to callbacks that declare it. Slice 2 includes a test that drives a decision through a callback with the original signature and asserts it is metered without a `TypeError`. |

Embeddings currently bypass all of these sinks, which is tracked as [#3426](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3426). They join the same recorder with `modality="embedding"` rather than getting a copy of it.

## 9. Eval harness

Thresholds and model choices come from measurement.

- **Samples.** Each caller has a labelled sample set: a `state`, its `questions`, and the expected answers. Committed samples are synthetic or public and live under `tests/evals/decisions/<caller>/*.jsonl`. Operator-local samples drawn from real memory or turns live under the agent data directory and are never committed.
- **Runner.** `kestrel decisions eval --caller <id> [--route <selector>]` runs the samples against each candidate model. For each model it reports accuracy, Brier score, expected calibration error, latency p50/p95, and a proposed threshold for every question id.
- **Recording.** An accepted proposal is written into `[decisions.thresholds.<caller>.models."<route>/<model>"]` (§2.5), with the run's sample-set hash and date in an adjacent comment, so every threshold can be traced back to the evidence behind it.

## 10. Feature (SDK) surface

Features call `self.agent.llm_service.decide(...)` with SDK types, through the same duck-typed access they already use for `generate`. The request, answer and exception types, `concentration()`, and the validation constants all live in `kestrel_sdk.llm.decisions`. Feature code needs no import from core.

Isolated (out-of-process) features have no LLM RPC today. Decisions do not add one; that gap also covers `generate` and is a separate piece of work.

## 11. Explicitly out of scope

- **Serving `/v1/systemone` inbound.** Kestrel already serves `/v1/models` and `/v1/chat/completions`. Exposing decisions to other agents is a later decision.
- **Images in `state`.** The contract extension is reserved. When a route we use accepts images (Clef, OpenAI's Decisions API, Ollama main), `DecisionRequest` gains an `images` field and `DecisionModelInfo` gains `accepts_images`; a request carrying images then routes only to models that accept them. Until then there is no `images` field, so an image cannot be silently dropped.
- **A role table** mapping callers to models. Callers pass `model_override` from their own config, as the answerability gate does today.
- **Runtime setters and endpoints** (`/api/decision/models`, `/api/decision/settings`). Configuration is file-based, and pins are verified at reconcile time (§4.1). When a caller needs a runtime switch, the endpoints will mirror `/api/embedding/*` and run the same canary at set time.

## 12. Rollout

Slices are tracked on #3424. Each one lands with tests that fail without it.

1. This spec.
2. **SDK contract plus core.**
   - SDK: types, validation constants, adapter methods, `supports_decisions`, the decision metrics series, and the contract version bump. The SDK ships first and core bumps its pin.
   - Core: `LLMService.decide`, the normaliser, resolver, fit check, thresholds, accounting, and the chat-listing exclusion (§3.3).
   - Routes: OpenRouter and Ollama `adecide` / `list_decision_models`. Two routes ship together so the dialect seam is exercised by real divergence.
3. **Eval harness.**
4. **First caller:** the memory answerability gate, measured against the current gate on recorded retrievals. The migration translates the gate's two routing inputs explicitly:
   - **Privacy.** The gate's injected `force_local_only_provider` becomes `local_only=self._force_local_only()`. Under §6, `decide` ORs that value with the live provider. Today the injected provider *replaces* the live one, so the only possible behaviour change is that the gate becomes stricter.
   - **Model selection.** `[retrieval] memory_answerability_model` uses `generate`'s selector grammar, which allows a bare model id. It is replaced by `memory_answerability_decision_model` in §5.3's grammar. A config that still sets the old key fails validation at load, with an error naming the new key. The old value is never silently reinterpreted under the new grammar.
5. Response audit, as a `score` question.
6. Reflection capture pre-gate, and batched sleep attestation.
7. Per-heuristic migrations, each with labelled samples: tagger, schema router, continuation intent, turn classifier, injection scoring, and GitHub-app should-respond.
8. Images in state.
9. Decision-driven model routing, after the per-turn routing freeze.

## Related

- [LLM_SERVICE_ARCHITECTURE.md](../LLM_SERVICE_ARCHITECTURE.md) — vendor / route / model, discovery, routing, no hardcoded model IDs.
- [PROVIDER_PLUGINS.md](PROVIDER_PLUGINS.md) — the adapter contract that `adecide` / `list_decision_models` extend.
- [MEMORY_SYSTEM.md](../MEMORY_SYSTEM.md) — second-stage answerability, the first caller.
- TypeSafe System One API reference: <https://docs.typesafe.ai/api.md> (OpenAPI: <https://api.typesafe.ai/openapi.json>).
- Ollama decision models: <https://docs.ollama.com/api/systemone>.
- OpenRouter Decisions API: <https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request.md>.
