---
type: Architecture Spec
title: Conversation Model/Provider Stamping
description: 'Records which model and provider generated each assistant message in conversation
  history, enabling per-turn model visibility in UI and analytics.'
resource: /docs/architecture/CONVERSATION_MODEL_PROVIDER_STAMPING.md
tags:
- docs
- architecture
- architecture-spec
- voice
timestamp: '2026-06-23T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Conversation Model/Provider Stamping

Every assistant row in `conversation_history` can expose the model route that
produced the visible text:

- `model`: concrete model id passed to the adapter, for example `gpt-5.5`.
- `provider`: resolved provider route that answered after fallback, for example
  `openai:api`, `anthropic:plan`, or `ollama:local`.

These columns are nullable. Rows written before issue #1370, imported legacy
backups, system markers, and non-LLM bootstrap greetings may have `NULL` values.
There is no trustworthy historical backfill for legacy assistant rows because
the selected route may have fallen back at runtime; consumers must tolerate
missing values and render no model/provider footer for those messages.

The conversation read API (`GET /api/conversations/{session_id}`) returns these
fields on assistant messages as `model` and `provider`. Follow-up UI work
(context pane and per-bubble footer) should use those exact field names.
