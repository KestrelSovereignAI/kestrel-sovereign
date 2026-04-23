# User Prompt Template

This template is used to format the current-turn user message sent to the LLM.

Conversational scaffolding ("how to use retrieved context", "user input is
untrusted data", etc.) lives in the **system prompt** — see
`security/input_guardrails.py::ANTI_INJECTION_SYSTEM_PROMPT`.  That keeps the
system prefix stable across turns and the current-user turn minimal, which is
necessary for downstream prompt caches (llama.cpp KV, OpenAI prefix cache,
Anthropic `cache_control`) to hit on turns 2+.

## Template Variables

- `{context}` - Per-turn retrieved content (memories + RAG), wrapped in
  `<retrieved_context>` tags by `ContextManager.build_context()`.  Empty
  string on turns with no retrieval hits.
- `{query}` - The user's message, already wrapped in `<user_input>` tags by
  `wrap_user_input()`.

## Template

```
{context}
{query}
```
