# Streaming Honesty Layer

> **Status:** Active. Shipped end-to-end as epic [#1048](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1048) waves 5A → 5F (PRs #1068, #1069, #1076, #1084, #1089). Closes [#1042](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1042) layers 2 & 3.
>
> **Audience:** anyone changing the agent streaming pipeline, the chat UI's stream consumer, the audit hook system, or adding a new consumer of `process_input_streaming`. If you touch any of those surfaces, read this first.
>
> **Substrate:** `kestrel-sovereign-sdk >= 0.9.0` for the `ToolCallStarted` marker and the extended `HookInput` fields.

The honesty layer is the concrete enforcement mechanism for [Constitution Book I, Chapter 1, property 5 ("Non-deceptive")](../../principles/KESTREL_CONSTITUTION.md) in the specific case of tool-use narration. This document describes how it works end-to-end: the failure mode, the signal protocol, the pipeline, and the guarantees each piece provides.

---

## 1. The problem

A streaming agent yields chunks to the user as they're generated. When the LLM emits "Saved your favorite color is teal." alongside a `save_fact` tool call, the user sees `Saved` stream live — *before any tool runs*. The post-tool LLM turn observes the actual result and may correct itself, but the user already received the confident lie.

The verbatim reproduction from `frinz` issue #143 (April 2026):

> User: *"Save this fact: my favorite color is teal. Then immediately call `memory_status` and tell me whether the file count went up. Then read it back to me from the store, not from this conversation."*
>
> Agent: *"Saved: your favorite color is teal. Memory status came back with `files_stored: 0`, so I can't confirm a file-count increase from that metric. I also couldn't read it back from the store yet…"*

The agent led with `Saved` despite `memory_status` returning `files_stored: 0` and `search_documents` finding nothing. In a constitutional framework where honesty is the foundational property, a single confident lie in a healthcare or legal-records context is a regulatory and trust-burning event. The honesty layer makes this class of failure an enforced impossibility, not a probabilistic improvement.

Five fixes attack different layers of the same problem; this doc covers all of them.

---

## 2. End-to-end flow

```
┌────────────┐  pre-tool        ┌─────────────────┐
│ LLM stream │ ───── chunks ───►│  agent/         │
│ (provider) │                  │  streaming.py   │
│            │  ToolCallStarted │                 │
│            │ ────marker────  ►│                 │
│            │                  │                 │
│            │  LLMResponse     │                 │
│            │ ─(tool_calls)──► │                 │
└────────────┘                  └────────┬────────┘
                                         │
                       ┌─────────────────┼─────────────────┐
                       │                 │                 │
                  yield text         emit SSE          yield in-band
                  chunk              "revising"        sentinel
                       │              event            \x1eKESTREL:REVISE:…\x1e
                       │                 │                 │
                       ▼                 ▼                 ▼
                /api/agent/stream    /api/agent/notifications/sse    (same stream)
                       │                 │                 │
                       │                 │                 │
                       └────────┬────────┴────────┬────────┘
                                │                 │
                            text/plain       text/event-stream
                                │                 │
                                ▼                 ▼
                        ┌─────────────────────────────┐
                        │  Chat UI (chat.js)          │
                        │                             │
                        │  - strips sentinel from chat│
                        │    stream                   │
                        │  - subscribes to revising   │
                        │    SSE event                │
                        │  - both signals idempotent  │
                        │    via reviseConsumedRequestId│
                        │  - clears in-flight bubble  │
                        │    on first signal          │
                        └─────────────────────────────┘

                  ┌─────────────── Tool execution ──────────────┐
                  │  agent/orchestrator_engine.py               │
                  │    - dispatches tool_calls                  │
                  │    - collects {status, success, error}      │
                  │      summaries into tool_results list       │
                  └────────────────────┬────────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────────┐
                  │  Post-tool LLM turn (synthesis)             │
                  │    - sees tool_results in conversation      │
                  │    - streams grounded answer                │
                  └────────────────────┬────────────────────────┘
                                       │
                            yields synthesis chunks
                                       │
                                       ▼
                          (back to chat stream → UI)
                                       │
                                       ▼
                  ┌─────────────────────────────────────────────┐
                  │  POST_RESPONSE hook fires with HookInput    │
                  │    .response_text     (post-tool synthesis) │
                  │    .pre_tool_prose    (snapshot at marker)  │
                  │    .tool_calls                              │
                  │    .tool_results      (slim envelope)       │
                  │                                             │
                  │  ResponseAuditHook runs analyze_narration:  │
                  │    past-tense success in pre_tool_prose +   │
                  │    failed/partial/ambiguous tool_results    │
                  │    → risk_boost ≥ threshold → DENY/MODIFY   │
                  └─────────────────────────────────────────────┘
```

---

## 3. The marker contract

`kestrel_sdk.llm.ToolCallStarted` is the SDK-level event that drives every downstream consumer. Adapters yield it from `get_streaming_response_with_tools` the moment the provider stream first signals a tool call — *before* arguments finish accumulating, *after* the call's index/id are knowable.

```python
@dataclass(frozen=True)
class ToolCallStarted:
    index: int           # provider-native, NOT positional (block_index, output_index, delta-tool-call-index)
    id: Optional[str]    # may be None on first delta for OpenAI
    name: Optional[str]  # may be None for OpenAI; populated for Anthropic/Codex/Ollama
```

**Per-provider emission timing** is documented in [PROVIDER_PLUGINS.md §"Per-provider emission rules for ToolCallStarted"](PROVIDER_PLUGINS.md#per-provider-emission-rules-for-toolcallstarted). The honesty layer requires each adapter to fire the marker at the *first* moment it can — earlier is better; later corrupts the consumer-side guarantees.

**Index semantics:** the marker's `index` is provider-native. Anthropic emits `block_index` (sparse — block 0 may be text, block 1 tool_use, block 2 text, block 3 tool_use); the markers fire with indices `1` and `3`. OpenAI's delta-tool-call-index is positional only by accident. Consumers MUST NOT assume `index` is a 0-based offset into `LLMResponse.tool_calls`.

The streaming-with-tools tagged union is `AsyncIterator[Union[str, ToolCallStarted, LLMResponse]]`. Any consumer that reads from `stream_with_tool_detection` must handle all three branches; Wave 5A's correction was to make the type hint reflect this (it had been silently dropping markers).

---

## 4. Server-side: agent streaming pipeline

`kestrel_sovereign/agent/streaming.py:_process_input_streaming_traced_locked` is the pipeline. The relevant loop:

```python
async for item in self.llm_service.stream_with_tool_detection(...):
    if isinstance(item, str):
        full_response.append(item)
        yield item                       # → /api/agent/stream
    elif isinstance(item, ToolCallStarted):
        await self._emit_revising_event(item, request_id=request_id, ...)
        yield _build_revise_sentinel(item)   # → /api/agent/stream IN-BAND
        if pre_tool_prose_snapshot is None:
            pre_tool_prose_snapshot = "".join(full_response)
    elif isinstance(item, LLMResponse):
        tool_response = item
```

Three things happen on the marker:

1. **SSE event fires** through `agent.emit_event("revising", payload)` → reaches the browser via `/api/agent/notifications/sse`.
2. **In-band sentinel yields** through the chat stream itself. Strictly serialized with the chunks it bounds.
3. **Pre-tool prose snapshots** at the *first* marker boundary. Subsequent markers don't overwrite the snapshot — inter-tool prose belongs to a different epoch.

The sentinel is **NOT appended to `full_response`**. Wire-protocol bytes must never reach storage; the next-turn history loader and audit anchor would otherwise see raw `\x1e` chars in their context.

After tool execution, the post-tool synthesis streams through `_handle_orchestrator_response_streaming`, which now also threads a `tool_results: list` collector. Each `_dispatch_tool_call` appends `{tool_call_id, name, result}` (with `result` slim-summarized — see §8 below) to that list.

Finally, the assistant turn is persisted, and the POST_RESPONSE hook fires with the SDK-0.9 `HookInput` fields populated:

```python
hook_input = HookInput(
    session_id=...,
    hook_event_name=HookEvent.POST_RESPONSE.value,
    response_text=tool_final_text,        # full visible text
    pre_tool_prose=pre_tool_for_audit,    # snapshot at marker boundary
    tool_calls=tool_calls_payload,         # [{id, name, arguments}]
    tool_results=tool_results,             # [{tool_call_id, name, result}] (slim)
)
```

---

## 5. Client signals: in-band sentinel + SSE

The chat client receives **two redundant** retraction signals. They're idempotent against `pane.reviseConsumedRequestId`, so whichever arrives first wins; the other becomes a no-op.

### Why two

The first design (Wave 5C) shipped only the SSE channel. Codex review caught two unavoidable bugs:

* **SSE-after-post-tool race.** SSE delivery may lag the chat-stream chunks (network jitter, browser scheduler, HTTP/2 multiplexing). A delayed `revising` arriving after post-tool synthesis already rendered would re-arm `pendingRevise` and the *next* chunk would clear the rendered post-tool answer.
* **Detached-pane miss.** Agent switch unbinds the old pane's SSE channel. Its still-running stream never gets the retraction signal.

Wave 5E moved the signal in-band. The in-band sentinel is *strictly serialized* with the chunks it bounds, so it can't lose to post-tool synthesis. The SSE channel stays wired as a reliability backup for cases where the in-band path's split-chunk edge-handling falls short. Detached panes get retracted because they keep consuming their own chat stream regardless of which agent's SSE channel the browser is bound to.

### Wire format

```
\x1eKESTREL:REVISE:<json>\x1e
```

* `\x1e` (Record Separator, ASCII 30) bookends. Chosen because: LLMs don't emit it, single-byte UTF-8, HTTP middleware doesn't strip it.
* Fixed prefix `KESTREL:REVISE:` — namespace + intent.
* JSON payload: `{"index": int, "tool_call_id": str|None, "tool_name": str|None}`. Same fields as the SSE event minus `request_id`/`session_id` (the in-band sentinel is implicitly scoped to the stream that carries it).

### Non-chat consumers strip the sentinel

Voice (`endpoints/voice.py`), bridge (`features/bridge/router.py`), and the TTS stream-tap publish path in `endpoints/agent.py` ALL apply `strip_revise_sentinels()` server-side before forwarding chunks to their respective protocol channels. Without this, TTS would speak the literal `\x1eKESTREL:REVISE:` prefix aloud.

The chat `/api/agent/stream` endpoint is the **only** consumer that passes the sentinel through verbatim — the chat client is the one place that knows how to interpret it.

---

## 6. Client-side: chat UI retraction

`kestrel_sovereign/static/js/chat.js` runs two complementary handlers.

### SSE listener

Subscribed in `connectNotifications`. Walks `state.chatPanes`, matches the event's `request_id` against each pane's active stream id (`API.getCurrentStreamRequestId(agentName)`), then:

```js
if (pane.reviseConsumedRequestId === targetRequestId) return;  // already consumed → no-op
pane.pendingRevise = true;
pane.reviseConsumedRequestId = targetRequestId;
// replace bubble body with placeholder
```

### In-band stripper inside `sendMessage`

Each chunk from `API.streamInvoke` runs through:

1. **Cross-chunk parser buffer** (Case A & B):
   * Case A: chunk contains a full `\x1eKESTREL:REVISE:` prefix without a closing `\x1e` → buffer everything from the prefix on, merge with next chunk.
   * Case B: chunk's *tail* is a proper prefix of the sentinel-prefix string (e.g. ends with `\x1eKESTREL:REV`) → buffer the tail. **Critical:** Case B must run AFTER strip — running before misidentifies the closing `\x1e` of an already-stripped sentinel as a new prefix start.
2. **`stripReviseSentinel(chunk)`** returns `{textBefore, textAfter, sawSentinel}`.
3. On `sawSentinel`: if `pane.reviseConsumedRequestId !== currentRequestId`, set both `pane.pendingRevise = true` and `pane.reviseConsumedRequestId = currentRequestId`. Show placeholder.
4. If `pane.pendingRevise`: reset `fullContent = ''`; clear flag.
5. Append `(sawSentinel ? textAfter : textBefore)` to `fullContent`; render.

Idempotency: in-band sets `reviseConsumedRequestId`; SSE checks it. Late SSE for an already-consumed request is a no-op. The in-band path itself unconditionally arms (each `ToolCallStarted` produces exactly one sentinel — no double-arming risk on this side).

`pane.reviseConsumedRequestId` resets in `sendMessage`'s `finally` block and in `wipeAgentChatPane`.

---

## 7. Audit half: deterministic narration check

`kestrel_sovereign/security/narration_check.py` ships a pure-Python function:

```python
def analyze_narration(
    pre_tool_prose: Optional[str],
    tool_results: Optional[List[Dict[str, Any]]],
) -> NarrationVerdict:
    ...
```

It scans the head of `pre_tool_prose` for past-tense success verbs (`saved`, `stored`, `sent`, `completed`, ...) anchored loosely at sentence boundaries. If any tool in `tool_results` reports a failed/partial/ambiguous outcome, returns `risk_boost=2` plus a reasoning string naming the offending verb and tool.

The check is **deterministic** — same inputs always produce the same verdict. That's the load-bearing property for compliance-gated deployments where "the agent doesn't lie about tool success" needs to be a falsifiable claim.

`ResponseAuditHook` (`kestrel_sovereign/features/response_audit/hook.py`) folds the verdict into the existing audit risk score:

* The narration check runs FIRST (before the 20-character short-response gate). A canonical `Saved.` (6 chars) followed by a failed tool still trips the audit.
* When `risk_boost > 0`, `risk_level` is floored at `risk_threshold`. A narration violation always crosses the gate regardless of the LLM-audit score (or its absence).
* The hook fires its policy (DENY in `strict`, MODIFY+annotate in `warn`) even when the LLM-audit call raises. Pure-Python check + threshold floor = compliance-safe under partial outage.

### Failure detection on tool result envelopes

`_result_indicates_failure` accepts three shapes:

* **ToolResult envelope** (#1042 layer 4): failure when `status` is `error` or `partial`.
* **Legacy `{success: bool}`**: failure when `success is not True`. (Note `is not True`, not `is False` — `success=None`, `success="false"`, `success=0` all flag as failure-for-audit-purposes per the honesty doctrine.)
* **Anything else** (no status, no success, no error key): treated as failure. Don't claim success without explicit confirmation.

### Known false-positive pattern

A multi-tool turn where the success verb refers to one successful tool but a *different* tool failed (e.g. `Saved the draft` + `save_draft` ok + `notify_team` error) currently flags. Verb-tool correlation is future work; documented in `kestrel_sovereign/security/narration_check.py`.

---

## 8. Privacy: tool_results summarization

`HookInput.tool_results` reaches every registered POST_RESPONSE hook, including third-party plugins. The full `_serialize_tool_result` envelope can contain memory contents, search hits, file contents, or other sensitive payload data. Passing that wholesale would be over-broad.

`summarize_tool_result_for_audit` (in `narration_check.py`) keeps only the fields the audit needs:

```python
{
    "status": "ok" | "error" | "partial",  # or
    "success": bool,                        # or
    "error": str (capped at 500 chars),
}
```

Non-dict primitives (raw strings, numbers, lists) become `{"status": "unknown"}` — the value can't leak verbatim through `tool_results`, and `analyze_narration` already treats `unknown`/non-dict as failure-for-audit-purposes, so the audit verdict is preserved.

The orchestrator (`kestrel_sovereign/agent/orchestrator_engine.py:_dispatch_tool_call`) calls the summarizer before appending to `tool_results`. The full envelope still goes to the LLM (as the `tool` message in the orchestrator conversation) and to storage; only the audit-hook surface gets the slim version.

---

## 9. Wire-protocol reference

| Channel | Format | Consumer | Purpose |
|---|---|---|---|
| `/api/agent/stream` | `text/plain` chunks (LLM text) + interleaved sentinels | chat client (`chat.js`) | primary text + ordering-correct revise signal |
| `/api/agent/notifications/sse` | `event: revising / data: {type,request_id,session_id,index,tool_call_id,tool_name}` | chat client SSE listener; audit dashboards; accessibility tools | reliability backup; non-chat subscribers |
| Voice WebSocket | `text/plain` chunks (sentinels stripped) | voice/TTS UI | TTS speaks only post-strip text |
| `/api/bridge/stream` | SSE `data: {type:"chunk", content}` | Slack/Discord/email bridges | sentinel stripped before serialization |

### In-band sentinel format

```
\x1eKESTREL:REVISE:{"index":N,"tool_call_id":"…","tool_name":"…"}\x1e
```

Constants: `kestrel_sovereign.agent.streaming.REVISE_SENTINEL_PREFIX` / `REVISE_SENTINEL_SUFFIX`. JS mirror in `kestrel_sovereign/static/js/chat.js`. Either side changing requires both sides updating.

### SSE `revising` event payload

```json
{
  "type": "revising",
  "request_id": "<endpoint-assigned uuid>",
  "session_id": "<active session>",
  "index": <provider-native index>,
  "tool_call_id": "<provider-issued id, may be null>",
  "tool_name": "<tool name, may be null>"
}
```

The `type` field is duplicated inside the `data` JSON for consumers that route on `data.type`; standard SSE consumers can match on the `event:` line directly. `request_id` is the routing key — it matches the `X-Request-ID` response header from the same dispatch's `/api/agent/stream` request.

---

## 10. Failure modes & how each is handled

| Hazard | Handler | Doc reference |
|---|---|---|
| LLM emits past-tense success before tool fires | TOOL_HONESTY_SYSTEM_PROMPT (system prompt level — #1042 layer 1) | `kestrel_sovereign/security/input_guardrails.py` |
| Pre-tool prose displayed before tool result observable | In-band sentinel + SSE backup retract bubble | §5, §6 |
| SSE delivery races post-tool synthesis chunks | In-band sentinel is strictly ordered; idempotency prevents re-arm | §5 |
| Detached pane misses retraction after agent switch | Detached pane consumes its own chat stream, sees its own in-band sentinel | §5 |
| Voice/TTS speaks sentinel chars aloud | Server-side `strip_revise_sentinels` at non-chat consumers | §5 |
| Sentinel split across chunk boundary (any position) | Cross-chunk parser buffer (Case A: full prefix without close; Case B: partial prefix at tail) | §6 |
| Persisted assistant turn corrupted by sentinel bytes | Sentinel never appended to `full_response`; storage path doesn't see it | §4 |
| Ambiguous tool envelope passes as "success" | `_result_indicates_failure` treats `is not True` / no-status as failure | §7 |
| LLM audit unavailable (rate-limited, network, model down) | Narration check fires regardless; risk floored at threshold | §7 |
| Sensitive payload leaks to third-party POST_RESPONSE hooks | `summarize_tool_result_for_audit` strips data before HookInput.tool_results | §8 |
| Plugin author's adapter lies about marker timing | Conformance suite (`kestrel_sdk.testing.drain_streaming_with_tools`) catches it | [PROVIDER_PLUGINS.md §"Conformance"](PROVIDER_PLUGINS.md) |

---

## 11. Test surface

The tests are the canonical contract. Don't drop them.

### Server-side (Python)

* `tests/unit/test_streaming_revising_event.py` — Wave 5B SSE event emission, request_id race fix.
* `tests/unit/test_streaming_inband_revising_sentinel.py` — Wave 5E sentinel emission, ordering, JSON escaping, no-leak-into-storage.
* `tests/unit/test_strip_revise_sentinels.py` — server-side strip helper edge cases.
* `tests/unit/test_streaming_post_response_narration_fields.py` — HookInput field plumbing through the streaming pipeline.
* `tests/unit/test_narration_check.py` — narration heuristic (16 cases: hedge phrasing, past-tense matches, envelope shapes, summarizer round-trip).
* `tests/unit/test_response_audit.py::TestResponseAuditHookNarrationFolding` — narration verdict folded into audit risk score; LLM-down behavior; threshold floor.

### Client-side (Node test)

* `tests/frontend/chat_revising_event.test.mjs` — 11 cases covering pane-shape pinning, SSE listener routing through fake EventSource, sentinel strip (clean/fused/split-prefix-no-close/partial-prefix-inside), idempotency between SSE + in-band, late-SSE no-op, multi-pane isolation.

### What "all green" means

When you change anything in this pipeline, all of the above must pass. The test names are deliberately specific so a regression's diff message tells you exactly which guarantee broke.

---

## 12. Future work

Tracked but intentionally out of scope for the initial epic:

* **Verb-to-tool correlation in narration check.** Today any past-tense verb + any failed tool flags. Better: `saved` ↔ `save_*` / `store_*` tools, `sent` ↔ `send_*` / `notify_*`, etc. Reduces multi-tool false positives.
* **`ToolResult` envelope migration for the framework's 235 in-tree `@tool` methods** — tracked in [#1061](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1061). Pilot landed (37 tools); bulk waves ongoing in another session. Each migration tightens the narration check's ground truth.
* **Cookiecutter / scaffold for plugin authors** — dropped as YAGNI; the 3-file scaffold in [PROVIDER_PLUGINS.md](PROVIDER_PLUGINS.md) is the reference. Reopen if external plugin authors materialize.

---

## 13. Cross-references

* [PROVIDER_PLUGINS.md](PROVIDER_PLUGINS.md) — adapter author's view: the SDK contract, marker emission rules, conformance suite.
* [LLM_SERVICE_ARCHITECTURE.md](../LLM_SERVICE_ARCHITECTURE.md) — vendor / route / model architecture; this doc is the streaming-honesty leaf of that tree.
* [Constitution Book I, Chapter 1](../../principles/KESTREL_CONSTITUTION.md) — the seven properties of honesty this layer enforces in code.
* [Issue #1042](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1042) — the original failure mode + the four layers of fix.
* [Epic #1048](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1048) — the implementation epic. Final tally in the closing comment.

---

## Maintenance

When this doc contradicts code, the code wins — and this doc is a bug. Update it in the same change, not later. The "Future work" section is the only forward-looking part; everything else describes shipped state as of the date below.

Last updated: 2026-05-08 (Wave 5E shipped, epic #1048 closed).
