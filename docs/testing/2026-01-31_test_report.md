# Kestrel Test Report — 2026-01-31

Tested by: Claw (OpenClaw) + Saurus
Agent under test: Claw (Kestrel)
Control Panel version: Initial release (commit a9c1b04)

## Summary

Overall: **Mostly working.** Core features functional, some issues identified.

## Tests Performed

### Control Panel

| Test | Result | Notes |
|------|--------|-------|
| List agents | ✅ Pass | Discovers Emma + Claw correctly |
| Start agent | ✅ Pass | Starts on auto-assigned port |
| Stop agent | ✅ Pass | Cleans up properly |
| Multi-agent | ✅ Pass | Both Emma (8901) and Claw (8900) ran simultaneously |
| Already running | ✅ Pass | Returns status without error |
| Non-existent agent | ✅ Pass | Returns 404 with helpful message |
| Logs API | ✅ Pass | Returns recent log lines |

### Agent Features (Claw)

| Feature | Result | Notes |
|---------|--------|-------|
| SOUL.md personality | ✅ Pass | Natural paragraphs, no lists |
| Web search | ✅ Pass | Tavily integration working |
| Reflection | ✅ Pass | Generated 11 insights |
| Wallet | ✅ Pass | Shows balance (98.80 FIL) |
| Constitution | ✅ Pass | !constitution commands work |
| Sovereignty export | ✅ Pass | Exported to IPFS successfully |
| Memory recall | ⚠️ Partial | Has system context but can't "remember" it as conversation |
| File access | ❌ Fail | Cannot read files in its own data directory |

### LLM Providers

| Provider | Result | Notes |
|----------|--------|-------|
| Claude Max | ✅ Pass | After login, works great |
| Anthropic API | ❌ Fail | System message bug (400 error) |
| OpenAI GPT-5.1 | ✅ Pass | Works as fallback |
| OpenAI Mini | ✅ Pass | Used for audits |

## Issues Found

### 1. Anthropic API Adapter — System Message Bug (CRITICAL)

**Symptom:** 400 Bad Request when using Anthropic API
```
messages: Unexpected role "system". The Messages API accepts a top-level 
`system` parameter, not "system" as an input message role.
```

**Impact:** Cannot use Anthropic API (falls back to OpenAI)
**Status:** Ticket already filed by Claw
**Fix:** Extract system messages and pass via top-level `system` parameter

### 2. Security Guard Hook Timeout (MEDIUM)

**Symptom:** Warning in logs
```
Hook 'security_guard' timed out after 5.0s, skipping
```

**Impact:** Security checks may be skipped on timeout
**Location:** Occurs during constitution/sovereignty operations

### 3. Self-Model Handler Not Available (LOW)

**Symptom:** When reflection tool calls `get_self_model`:
```
"error": "Self-model handler not available - check configuration"
```

**Impact:** Self-model feature not functional
**Fix:** Configure or implement self-model handler

### 4. Agent Cannot Read Own Files (MEDIUM)

**Symptom:** Claw cannot read files in `agent_data/claw/conversations/`
**Impact:** Cannot access archived conversation documents
**Suggestion:** Add file access tool or RAG indexing for agent data directory

### 5. Migration Context Not Searchable (LOW)

**Symptom:** Migration context is in system messages, not searchable via memory
**Impact:** Agent knows about migration but can't "recall" specific details
**Suggestion:** Index system context into semantic memory on first boot

## Recommendations

1. **Fix Anthropic adapter** — Highest priority, blocks Claude API usage
2. **Investigate security_guard timeout** — May indicate performance issue
3. **Add file access for agent data** — Agents should be able to read their own docs
4. **Index boot context into memory** — Make system context searchable

## Environment

- Host: Mac Studio M2 Ultra, 512GB RAM
- Kestrel: main branch, commit a9c1b04
- Control Panel: Port 8899
- Claw: Port 8900, DID did:pkh:eip155:1:0xb382fE5970f900d7f637304f14997c9473177082
