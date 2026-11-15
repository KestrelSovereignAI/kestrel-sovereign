---
type: Research Note
title: 'Plan allowance, step 0: what the anthropic:plan route and the Claude CLI report'
description: Step 0 of issue 4097. Where the anthropic:plan route receives a provider
  response and what it keeps, which rate-limit header names already appear in the
  host logs and llm.usage records, and the form of `claude -p "/usage"` output.
  Names and formats only, no account values.
resource: /docs/research/4097-plan-allowance-step0.md
tags:
- docs
- research
- research-note
- llm
- anthropic-plan
timestamp: '2026-11-15T00:00:00Z'
status: snapshot
owner: architecture
canonical: false
generated: false
privacy: public
---

# Plan allowance, step 0: what the anthropic:plan route and the Claude CLI report

**Issue:** #4097, step 0 only. **Date:** 2026-11-15. **Code read at:** `265d90661`.

This note records readings. It implements none of the issue's Directions 1 to 3.
Per the order note on the issue, the design (an allowance endpoint, and a reserve
that refuses work) goes to architecture for a ruling before any code.

It records names and formats, not this account's values. Every number this
account reported is a placeholder (`<P>`, `<N>`, `<D>`). No request ids,
credential tags, emails or account identifiers are reproduced. No credential,
token, keychain item or auth file was read, and no HTTP call was made to the
provider.

Each finding is marked:

- **measured**: seen in this run, at the cited `file:line`, in a log record, or
  in a command's output;
- **inferred**: reasoned from measured facts, with the reasoning stated.

## Answers in brief

1. **Response and headers.** Both plan-route call paths open one SDK stream, at
   `kestrel_sovereign/llm/anthropic_adapter.py:113-114`. The SDK stream exposes
   the HTTP response, so its headers are reachable there. A failed call's
   exception also carries the response. Kestrel keeps no response header on
   success. On a failure it reads only `retry-after-ms` / `retry-after`, to size
   a wait, and does not store either. The `retry_at=` value is Kestrel's own
   clock plus that advised wait (`kestrel_sovereign/llm/retry.py:663-671`). It is
   not a reset time the provider sent.
2. **Header names in records: none.** There is no rate-limit header name in any
   of the six host logs or in any `llm.usage` record. The plan endpoint's
   rate-limit header names are therefore **not established**. Section 2 names
   what would establish them.
3. **`claude -p "/usage"`** ran once, exited 0 in about 3 s and needed no
   terminal. It prints plain text with **two weekly meters**: all models, and
   one model family. Each is an integer percent used with a minute-precision
   local reset time. The output is not machine-readable.

The dispatch note's branch ("if the headers carry utilization and a reset time
per window") cannot be decided from this run, because the header names are not
established. The order note already sends the design to architecture either way.

## 1. Where the anthropic:plan route receives a response

### 1.1 Route, adapter and client

- **measured.** The plan route's adapter is `ClaudeMaxAdapter`
  (`kestrel_sovereign/llm/claude_max_adapter.py:30`), a subclass of
  `AnthropicAdapter`. It overrides authentication shape, discovery and pricing
  metadata, and inherits all message handling.
- **measured.** `kestrel_sovereign/llm/provider_registry.py:442-466` builds the
  plan client as `anthropic.AsyncAnthropic(auth_token=..., max_retries=0)`. The
  SDK's own retries are off, so `with_retry` in
  `kestrel_sovereign/llm/retry.py:517` is the only retry layer.
- **measured.** The SDK version is `anthropic` 1.15.0, pinned at `uv.lock:266-267`.
  The running fleet host's interpreter (from its process command line) belongs
  to a virtual environment with 1.15.0 installed.

### 1.2 The one seam both call paths share

- **measured.** `_anthropic_stream_with_retry`
  (`kestrel_sovereign/llm/anthropic_adapter.py:96`) opens the request with
  `client.messages.stream(**api_params)` and `await cm.__aenter__()`
  (`anthropic_adapter.py:113-114`).
- **measured.** The non-incremental path, `get_response`
  (`anthropic_adapter.py:1391`), calls `_anthropic_final_message`
  (`anthropic_adapter.py:229`). That opens the same stream
  (`anthropic_adapter.py:254`) and returns `stream.get_final_message()`, the
  SDK's parsed `Message`.
- **measured.** The streaming path, `get_streaming_response_with_tools`
  (`anthropic_adapter.py:1588`), opens the same stream at
  `anthropic_adapter.py:1700`. Plain streaming (`anthropic_adapter.py:1570`)
  goes through `_plain_stream` (`kestrel_sovereign/llm/adapter.py:464`).

### 1.3 Are headers available there?

- **measured (SDK source, 1.15.0).** `AsyncMessageStream` has a `response`
  property returning the `httpx2.Response`
  (`anthropic/lib/streaming/_messages.py:224`). The SDK itself reads
  `request-id` from `self.response.headers` (same file, line 229). So
  `stream.response.headers` is available as soon as the stream opens, on both
  call paths.
- **measured.** Kestrel already reaches `stream.response` at that point:
  `_report_body_progress` (`anthropic_adapter.py:185-203`, called at lines 163
  and 255) replaces the response's body stream for progress reporting. It does
  not read `.headers`.
- **measured (SDK source).** On an HTTP error, `APIStatusError.__init__` keeps
  `self.response`, the `httpx2.Response` (`anthropic/_exceptions.py:81-86`).
  `RateLimitError` is one of these. So a 429's response headers are reachable
  from the exception.

### 1.4 What is kept today

- **measured.** On success, `get_response` returns
  `LLMResponse(raw=response, ...)` where `response` is the parsed `Message`,
  a body with no headers (`anthropic_adapter.py:1539-1548`). The streaming
  terminal response has `raw=None` (`anthropic_adapter.py:1910-1919`). Only
  token counts and `stop_reason` are taken from either.
- **measured.** The per-call record is built in `_finalize_invocation`
  (`kestrel_sovereign/llm/service.py:4566`) and written by `_log_llm_call`
  (`service.py:4962`). The `llm.usage` log line (`service.py:5208-5228`) has
  these keys: `provider`, `model`, `duration_ms`, `success`, `input_tokens`,
  `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`,
  `tools`, `structured_output`, `cost`, `usage_available`, `stop_reason`,
  `session_id`, `correlation_id`. Non-chat calls also have `modality` and
  `caller`.
- **measured.** The durable observability row gets the same arguments plus a
  metadata dict. Its possible keys are `path`, `force_local_only`,
  `inference_lease_id`, `correlation_id`, `provider_reported_cost_usd`,
  `usage_available`, `stop_reason` and `content_redacted`
  (`service.py:4650-4676`, `service.py:5047-5053`). None is a header.
- **measured.** The only response headers Kestrel reads from a plan response are
  `retry-after-ms` and then `retry-after`, on an exception
  (`retry.py:313-352`, header reads at lines 330 and 338). They size the retry
  wait. `AdvisedWaitExceedsRetryBudget` keeps the provider response object
  (`retry.py:188`) for as long as the exception lives. Nothing writes a header
  to a log or a record.
- **inferred.** A successful plan call keeps no header at all. A failed one
  keeps the advised wait only as the `retry_at` it computes, and keeps the
  headers only inside a transient exception object.

### 1.5 The code that produced `retry_at=` in the issue body

- **measured.** The line in the issue body is the warning at
  `retry.py:666-671`:
  `"LLM retry declined: advised wait %.0fs exceeds the %.0fs this call could
  still wait (status=%s, retry_at=%s): %s: %s"`.
- **measured.** `retry_at = datetime.now(UTC) + timedelta(seconds=min(advised,
  ADVISED_WAIT_HORIZON_SECONDS))` (`retry.py:663-665`). Here `advised` is
  `retry_after_seconds(e)` (`retry.py:624`), which reads `retry-after-ms`
  first, then `retry-after`, then an exception `retry_after` attribute
  (`retry.py:330`, `338`, `350`). The horizon is seven days (`retry.py:50`).
- **measured (log).** The record is in `logs/host.log.2` at 21:57:43 host time on
  2026-11-13. Convert that timestamp to UTC with the host's offset and add the
  advised wait the issue body quotes. The result is the `retry_at` the issue
  body quotes, to the second. So `retry_at` is the host clock plus the
  provider's advised wait.
- **inferred.** The provider sent a relative wait. Kestrel turned it into an
  absolute time. The log does not say which of `retry-after-ms` and
  `retry-after` carried it, because the wait is printed with `%.0f`.
- **inferred.** The "840s this call could still wait" is the throttle budget at
  its first attempt: `THROTTLE_MAX_DELAY` 120 s times (`THROTTLE_MAX_RETRIES`
  8 minus 1), with nothing waited yet (`retry.py:41-42`, `retry.py:616-619`).
  The plan-limit budget (`retry.py:83-84`) would allow 64 s and the default
  budget (`retry.py:30-32`) 240 s. A 429 is a throttle, not a plan-limit error,
  because `is_plan_limit_error` (`retry.py:363`) matches non-429 statuses only.
- **measured (log).** The decline follows directly on the transport's
  `HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 429 Too
  Many Requests"` line, with no `LLM retry n/m` line between them. After it
  come the adapter's error line and the call's `llm.usage` record
  (`provider: anthropic:plan`, `success: false`). The provider's error body is
  `{'type': 'error', 'error': {'type': 'rate_limit_error', 'message': "This
  request would exceed your account's rate limit. Please try again later."},
  'request_id': '<request id>'}`. The body names no window, no utilization
  and no reset time.

## 2. Rate-limit header names already in logs or llm.usage records

### 2.1 What was read, and how

Read-only. A throwaway script outside the repository scanned each file byte by
byte. It printed only header-like names, `llm.usage` keys, counts, and line
shapes with every digit run, hex run, request id and email replaced by a
placeholder. No log was changed or copied.

| File | Span (host local time) |
|---|---|
| `logs/host.log` | 2026-11-15 10:38 to 14:10 |
| `logs/host.log.1` | 2026-11-14 16:17 to 2026-11-15 10:38 |
| `logs/host.log.2` | 2026-11-13 19:42 to 2026-11-14 16:17 |
| `logs/host.log.3` | 2026-11-12 19:06 to 2026-11-13 19:42 |
| `logs/host.log.4` | 2026-11-12 00:58 to 19:06 |
| `logs/host.log.5` | 2026-11-10 19:33 to 2026-11-12 00:58 |

The fleet host's project directory holds these files (paths above are relative
to it). They have no gaps between them. The same scan also covered the older,
manually kept copies in that directory (`host.log.before-*`,
`host_startup.log`, `kestrel.log`, `server.log`) and `~/.kestrel/logs/host.log`.
None of those has an `llm.usage` record.

Names searched, case-insensitive: `anthropic-ratelimit-*`, `x-ratelimit-*`,
`retry-after`, `retry-after-ms`, `anthropic-*limit*`, any token containing
`unified`, and any token containing `rate-limit`, `rate_limit` or `ratelimit`.

### 2.2 Results

- **measured.** No response header name appears in any file. The hits for the
  broad patterns are all something else:
  - `unified` is the module path `kestrel_sovereign.a2a.stores.unified...` and
    a configuration-migration message;
  - `rate-limit` and `rate-limited` are prose in two log messages, a wake
    parked until "the model provider's advised rate-limit reset" and the
    peer-stop description;
  - `RateLimitError` and `rate_limit_error` appear twice each in
    `host.log.2`: the 21:57 decline line and its traceback, both for the same
    429.
- **measured.** Every `llm.usage` record in the six files has exactly the keys
  listed in section 1.4 and no others. That covers all three providers in
  them, `anthropic:plan`, `openrouter:api` and `ollama:local`. No record has a
  header field.
- **measured.** The six files hold exactly one HTTP 429 from
  `api.anthropic.com`: the 2026-11-13 21:57 decline. The transport logs each
  response at INFO as a status line only (`"HTTP/1.1 <status> <reason>"`), with
  no headers.
- **measured.** Every other `LLM retry` line in the six files is
  `APIConnectionError` with `status=None`, `budget=default`,
  `retry_after=None`. No throttle retry was logged.

### 2.3 Not examined

- The durable `llm_calls` observability rows in the agent databases. Step 0's
  scope named the logs and `llm.usage` records. **inferred:** section 1.4 shows
  the row is written from the same arguments, so it has no header column.
- Trace stores (Phoenix). Not in step 0's scope.

### 2.4 Header names: not established

The plan endpoint's rate-limit header names, and whether they include
utilization, are **not established** by this run. Each of these would establish
them, and each is outside step 0:

- **A code change at the seam in section 1.3.** It would log the names, and per
  Direction 1 the values, of `stream.response.headers` after
  `anthropic_adapter.py:114`, and of `exc.response.headers` on a 429. This is
  Direction 1 itself, which waits for the architecture ruling.
- **One direct HTTP request to the plan endpoint** that prints the response
  header names. This needs the plan credential and a call to the provider, both
  excluded from step 0.

## 3. `claude -p "/usage"`

### 3.1 The run

- **measured.** It ran once, on 2026-11-15, with Claude Code `2.1.322`. The
  command was `claude -p "/usage" </dev/null`, from a scratch directory. Stdin
  was closed because `claude -p` blocks on inherited stdin (see this
  repository's `AGENTS.md`). Stdout and stderr went to separate private files,
  which were read only through a masking filter.
- **measured.** Exit status 0, about 3 s, no terminal or TTY required. Stdout
  was 21 lines of UTF-8 plain text: no ANSI escapes, and the only non-ASCII
  characters are `·` separators. Stderr was one line. It was not retried.

### 3.2 Form of stdout

Numbers are placeholders. The two meter lines are verbatim apart from the
placeholders.

```text
You are currently using your subscription to power your Claude Code usage

Current week (all models): <P>% used · resets <Mon> <D> at <h>:<mm><am|pm> (<IANA time zone>)
Current week (<model family>): <P>% used · resets <Mon> <D> at <h>:<mm><am|pm> (<IANA time zone>)

What's contributing to your usage?
Approximate · sessions on this machine only · shares overlap

Last <N>h · <N> requests · <N> sessions
  <P>%  <advice> · run /clear
  <P>%  <advice> · run /context
  <P>%  <advice>
Last <N>d · <N> requests · <N> sessions
  <P>%  <advice> · run /clear
  <P>%  <advice> · run /context
  <P>%  <advice>
  <P>%  <advice>
  <P>%  <advice>
  Top skills: <skill> <P>%
  Top subagents: <name> <P>%, <name> <P>%
```

### 3.3 Meters, fields and units

- **measured.** There are two meters, both for the current week:
  - `all models`;
  - one model family. The family shown is the one in the `model` field of this
    host's `anthropic:plan` `llm.usage` records.
- **measured.** Each meter has two fields:
  - **utilization**, an integer percent **used** (not remaining);
  - **reset**, a month abbreviation, a day, and a 12-hour clock time to the
    minute in the host's IANA time zone. There is no year, no seconds and no
    UTC offset.
- **measured.** This output has no shorter-window meter, such as a per-session
  or five-hour meter. Whether one appears under other conditions is **not
  established**.
- **measured.** The "What's contributing" section labels itself "Approximate ·
  sessions on this machine only · shares overlap". **inferred:** it is a local
  estimate, not the account's meter, and not a source for an allowance figure.
- **measured.** Stderr was a warning about the sign-in's expiry window: two
  dates and a day count, which this note does not reproduce. It advised
  `claude auth login` or `claude setup-token`. It contained no credential.

### 3.4 Machine-readable?

- **measured.** No. There is no JSON and no `key=value`: these are sentences
  with `·` separators.
- **inferred.** A line pattern could parse the two meter lines. But the wording
  is not a published contract and may change between CLI versions. The reset
  time would also need the host's time zone and a year inferred from the
  current date.
- **Not tried** (one run only): whether `--output-format json` changes the
  form of a slash command's output.

### 3.5 Not established about `/usage`

- Whether it makes a network request or a model turn, which endpoint it reads,
  whether it counts against the allowance, and how current its percentages are.
  The CLI's documentation, or a network trace of one run, would establish these.
  Neither is in step 0's scope.
- Which of the two meters reached its limit at 21:57 on 2026-11-13.
  **inferred:** the refusal was a weekly window. The advised wait was more than
  two days and well inside the seven-day horizon (`retry.py:50`).

## 4. Facts for the architecture ruling

No recommendation is made here. These are the facts the design question
depends on.

- **Header source.** There is one seam for success and failure on both call
  paths (section 1.3). It is discarded today, and its header names are unknown.
  Reading it needs no new request, but needs a code change to see what it
  carries.
- **CLI source.** `claude -p "/usage"` already gives two weekly meters, a
  percent used and a reset time each, from the plan account. It is human text,
  with minute-precision local reset times. Its cost per call and its freshness
  are unknown. The dispatch note's question stands: who runs a host process
  that polls it, how often, and at what cost.
- **What Kestrel knows today at a refusal.** Only `retry_at`, computed from the
  advised wait on the 429 (section 1.5). It learns this when the provider
  refuses, not before.
