# Claude Review: PR #1333

- PR: https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1333
- Title: fix: preserve OpenAI-compatible reasoning on tool streams
- Reviewed: 2026-05-21T16:15:38Z

## PR Review: OpenAI Reasoning Tool Roundtrip

### No blocking findings.

The change is small, correct, and well-scoped: accumulate `reasoning_content` deltas during streaming, attach them to the terminal `LLMResponse.raw` when tool calls are present, so the orchestrator can replay them into the next DeepSeek/Kimi assistant turn.

### Minor observations (non-blocking)

1. **`raw` field overload** — `kestrel_sovereign/llm/openai_adapter.py:508`: Previously `raw=None` for all streaming responses. Now it carries `{"reasoning_content": ...}` only when reasoning is non-empty. This is fine for the tool-call path, but downstream consumers that do `if response.raw:` to detect "non-streaming" responses could misfire. Worth a quick grep for `resp.raw` or `response.raw` usage outside the replay codepath to confirm nothing assumes `raw is None ⟹ streaming`. Low risk given the PR description says the orchestrator replay is the only consumer.

2. **Non-tool-call reasoning not persisted** — If the model reasons but emits *no* tool calls, `reasoning_content` is accumulated but never attached to anything (the non-tool-call exit path at the end of the loop doesn't yield an `LLMResponse` with `raw`). This is correct for the stated goal (replay is only needed when tool history must include the assistant's reasoning turn), but worth documenting the asymmetry so a future reader doesn't "fix" it.

3. **Test coverage is adequate but narrow** — The new test (`test_preserves_reasoning_content_on_final_tool_call_response`) covers the happy path: two reasoning deltas followed by a tool-call delta. Edge cases not covered:
   - Interleaved reasoning and content deltas (reasoning + text + tool call)
   - Empty-string reasoning deltas (`""`) — the `isinstance(..., str) and delta_reasoning_content` guard handles this, but no test asserts it
   - Reasoning after tool-call deltas (unlikely per protocol, but defensive)
   
   None of these are blocking; the existing guard logic is correct.

4. **String concatenation in a hot loop** — `reasoning_content += delta_reasoning_content` is O(n²) for many small deltas. In practice reasoning chunks are few and short, so this is fine. If it ever matters, a list + `"".join()` is the fix.

### Verdict

Ship it. The diff is minimal, the contract change is intentional, and the regression test covers the primary scenario.
