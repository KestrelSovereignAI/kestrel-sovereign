# Agent Identity Contract

## Summary

Every Kestrel agent has a single canonical identity: its **DID** (`self.did`).

## Rules

1. **`self.did` is the source of truth.** It is set once at construction and never reassigned.
2. **`self.agent_id` is a read-only property** that returns `self.did`. It exists for backward compatibility with storage interfaces and feature packages that accept an `agent_id` parameter.
3. **Do not independently set `agent_id`.** Any code that previously wrote `self.agent_id = ...` on a `KestrelAgent` instance must be removed; the property will raise `AttributeError` on assignment.
4. **No `getattr` fallback chains.** Code like `getattr(self, 'agent_id', '') or getattr(self, 'did', '')` is banned. Use `self.did` (or `self.agent.did` from a feature) directly.

## For Feature Authors

When your feature needs to identify its parent agent:

```python
# In Feature.initialize():
agent_id = self.agent.did      # preferred
agent_id = self.agent.agent_id # also works (returns did)
```

When passing identity to storage layers, use the `agent_id` parameter name — the value is always the DID:

```python
store = AsyncConversationStore(db=db, agent_id=self.agent.did)
```

## For Storage Authors

Storage classes accept `agent_id: str` as a constructor parameter. Callers always pass the agent's DID. The parameter is named `agent_id` (not `did`) because storage is identity-scheme-agnostic — it just needs a unique key.

## History

See [#500](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/500) for the consolidation rationale.
