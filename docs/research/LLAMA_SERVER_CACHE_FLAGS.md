# llama-server cache flags

Recommended `llama-server` startup flags when Kestrel talks to it via the
`llama_cpp:local` route. Pairs with issue #704 (client sends `cache_prompt:
true`) and issue #703 (stable prompt prefix) to deliver real turn-2+ TTFT
reductions on a multi-turn conversation.

## Recommended command

```bash
llama-server \
  --model /path/to/your/model.gguf \
  --ctx-size 131072 \
  --port 8001 \
  --kv-unified \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --reasoning-format deepseek \
  \
  --cache-reuse 256                       # see below
  --slot-save-path "$LLAMA_SLOT_DIR"      # see below; e.g. ~/llama-slots
  --parallel 1                            # bump if running multiple
                                          # concurrent Kestrel agents
                                          # against the same server
```

## Why each new flag matters

### `--cache-reuse 256`

llama-server's default behavior is exact-prefix-match caching: the slot's
KV is reused only when the incoming request's prefix is byte-identical up
to some position. `--cache-reuse N` relaxes this so that when the prefix
diverges by a short run (N tokens or fewer), llama-server can re-shift
the existing KV instead of reprefilling the entire suffix from the
divergence point.

Typical win: the first turn after Kestrel runs a history-compression
pass. Compression replaces older raw turns with a synthetic summary
message; without `--cache-reuse`, that single insertion invalidates the
entire KV tail. With `--cache-reuse 256`, llama-server detects the small
middle divergence and only reprocesses the surrounding window.

`256` is a reasonable default — larger values cost more CPU per request
in the hope of recovering more cache; smaller values save CPU but
invalidate more aggressively.

### `--slot-save-path <dir>`

Persists slot KV state to disk so an `llama-server` restart doesn't cost
a cold prefill. Particularly valuable when working in an IDE that
restarts the Kestrel host process frequently: without this flag, every
IDE reload causes a fresh 20-second+ prefill on a long conversation.

The directory must exist and be writable. Disk usage grows with
conversation length and with the number of parallel slots.

### `--parallel N`

Default is 1 slot. Bump this if you run multiple Kestrel agents (or any
concurrent clients) against the same llama-server — each gets its own
independent slot and independent KV cache. Otherwise slot contention
starves one conversation's cache to feed the other.

## How Kestrel cooperates

- **Client side** (issue #704): when the active route is `llama_cpp`,
  `OpenAIAdapter.get_response(...)` sends `extra_body={"cache_prompt":
  True}` on the chat completion request. This is a llama-server-specific
  hint to be aggressive about retaining this slot's KV state for prefix
  matching across requests.
- **Prompt shape** (issue #703): Kestrel's system prompt is stable
  across turns, and historical user messages are consistently wrapped in
  `<user_input>` tags when loaded. Together these make the token-level
  prefix between turn N and turn N+1 byte-identical for the system + the
  N-2-and-earlier history — which is exactly what llama-server's cache
  matches against.

## Verifying it works

```bash
uv run python scripts/bench_prompt_cache_providers.py --only llama_cpp --turns 3
```

Expect: turn 1 wall-clock substantially larger than turn 2+ mean. On a
131k-context Kimi model with a non-trivial conversation, turn-1 cold
TTFT of 20+ seconds should drop to 3–5 seconds on turn 2+. If turn 2+
is roughly the same as turn 1, something is defeating the cache — check
that the system prompt is byte-stable (issue #703 merged) and that the
startup flags above are applied.
