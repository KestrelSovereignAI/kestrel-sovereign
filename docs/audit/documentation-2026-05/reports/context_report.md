---
type: Audit Report
title: Context Report
description: 'Lane report from the May 2026 documentation audit: Context Report.'
resource: /docs/audit/documentation-2026-05/reports/context_report.md
tags:
- audit
- documentation
- may-2026
- report
timestamp: 2026-05-30 00:00:00+00:00
status: snapshot
owner: documentation-audit
canonical: false
generated: false
privacy: public
---


# Context Lane Report

Source: subagent lane review, read-only, 2026-05-30.

## Scope Reviewed

- Context lane docs and generated feature docs.
- Code paths for prompt assembly, token budgeting, context status diagnostics, route caps, retrieval insertion, canonical/raw vs rendered transport history, and durable salvage.
- Primary files under `docs/architecture/`, `kestrel_sovereign/agent/`, and `kestrel_sovereign/endpoints/agent.py`.

## Canonical Doc Recommendation

- Create or promote one active canonical context spec, ideally `docs/architecture/CONTEXT_SYSTEM.md`.
- Treat `docs/architecture/CONTEXT_SYSTEM_DESIGN.md` as historical design unless it is rewritten to current-state truth.
- Keep `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md` as the salvage design/implementation note, but update status and substrate claims.

## Stale Or Conflicting Claims

- `docs/architecture/CONTEXT_SYSTEM_DESIGN.md` says "No code in this branch," describes static/adaptive budget as current, says `/api/agent/context-status` is history-only, and says tool schema tokens are unmeasured. Current code has whole-window measurement, route-cap-aware model identity, tool-schema token estimates, elastic budget in live context assembly, and diagnostic breakdowns.
- `docs/architecture/CONTEXT_SYSTEM_DESIGN.md` says C remains design-first/unimplemented. Code now has `kestrel_sovereign/agent/salvage.py`, salvage worker lifecycle, and feature-flagged sync salvage in `ContextManager.build_context`.
- `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md` says async summarization rides on `SignalDispatcher`. Current implementation appears to use `SalvageWorker` background asyncio tasks plus janitor state in `kestrel_sovereign/agent/salvage.py`.
- `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md` says popup unconditionally surfaces `silently-pruned path still active` because C has not shipped. Endpoint now computes this from `not is_durable_salvage_enabled()`.
- `docs/generated/FEATURES_developer.md` is stale. It reports 34 core modules and includes old inventory such as `voice`, while the current inventory differs and includes newer entries such as `cli` and `skills`.
- Context docs understate or omit the canonical/raw vs rendered transport split: user `content` is canonical raw/clean form, while `rendered_content` plus `metadata.sent_form` is the byte-stable LLM transport form.

## Code/Package Evidence

- Elastic budgeting and degraded mode: `kestrel_sovereign/agent/token_budget.py`.
- Whole-window measurement and tool-schema token estimation: `kestrel_sovereign/agent/context_builder.py`.
- Live context assembly uses elastic budget, system prompt caps, relevance gates, dynamic user context outside system prompt, lumpy anchor/prune, and feature-flagged salvage: `kestrel_sovereign/agent/context_manager.py`.
- Context diagnostics endpoint returns whole-window `breakdown`, `silently_pruned_path_active`, salvage counts, cheap/full modes: `kestrel_sovereign/endpoints/agent.py`.
- Route-level per-turn context caps: `kestrel_sovereign/agent/token_counter.py` and `kestrel_sovereign/llm/model_catalog.py`.
- Canonical/raw vs rendered transport tests: `tests/unit/test_conversation_sent_form.py` and `tests/unit/test_canonical_transport_split.py`.

## Docs To Update

- `docs/architecture/CONTEXT_SYSTEM_DESIGN.md`
- `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md`
- `docs/architecture/README.md`
- `docs/architecture/LLM_SERVICE_ARCHITECTURE.md`
- `KESTREL_FEATURES.md`, only if the canonical inventory needs a brief context diagnostics note.

## Docs To Archive Or Mark Historical

- Mark `docs/architecture/CONTEXT_SYSTEM_DESIGN.md` historical if a new canonical current-state context doc is created.
- In `docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md`, mark the original "No code / SignalDispatcher" portions as historical design notes or replace them with current implementation status.

## Generated Docs To Regenerate

- `docs/generated/FEATURES_developer.md`
- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_investor.md`
- Potentially `docs/audit/REPO_MAP.md` after doc edits.

## Open Questions

- Is durable salvage considered shipped when the feature flag is off by default, or only available?
- Should `measure_context_breakdown()` be changed to use the same elastic budget, lumpy anchor, and relevance/trivial-turn gates as `ContextManager.build_context`, or should docs explicitly state the remaining approximation?
- Should salvage async work intentionally remain `SalvageWorker`, or should code eventually move to `SignalDispatcher` as the C design says?
- Where should the canonical/raw vs rendered transport split live canonically: context doc, storage doc, or both?

## Suggested First PR Slice

Documentation-only: add a current-state context spec or rewrite the top/status/current-behavior sections of `CONTEXT_SYSTEM_DESIGN.md`, update `CONTEXT_C_DURABLE_SALVAGE.md` status/substrate claims, and regenerate the generated feature docs.

