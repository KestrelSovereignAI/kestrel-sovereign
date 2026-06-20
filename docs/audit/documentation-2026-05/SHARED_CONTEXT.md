---
type: Audit Ledger
title: Shared Context
description: Shared context for May 2026 documentation audit reviewers.
resource: /docs/audit/documentation-2026-05/SHARED_CONTEXT.md
tags:
- audit
- documentation
- may-2026
timestamp: 2026-05-30 00:00:00+00:00
status: snapshot
owner: documentation-audit
canonical: false
generated: false
privacy: public
---


# Shared Context

Date: 2026-05-30

The May 2026 documentation audit is driven by recent changes in package ownership and runtime architecture. Reviewers should ground findings in current code, package metadata, and recent commit history rather than trusting older architecture claims.

## Current High-Risk Drift Areas

- Feature/package extraction: `kestrel-feature-*`, `kestrel-cloud-*`, `kestrel-storage-*`, `kestrel-voice-*`, `kestrel-talon`, `kestrel-llms`, and `kestrel-sovereign-sdk`.
- Feature inventory drift: `KESTREL_FEATURES.md`, generated feature docs, and `kestrel_sovereign/data/feature_registry.toml`.
- Context: cache-stable history pruning, canonical conversation history vs rendered provider transport, token budgets, retrieval insertion, feature-prompt caps.
- Memory/retrieval/storage: vector backends, pgvector kNN, saved-item search, encryption backfill, external-ref asset restoration, privacy mode effects.
- LLM routing: provider capabilities, `kestrel.toml [llm]`, retired `llm_config.toml`, Codex/Anthropic/OpenAI transport behavior.
- Signals/workflows: `SignalDispatcher`, wake sources, constitutional injection, workflow extraction.
- Talon: standalone `kestrel-talon` package, in-agent control surface, runtime preferences vs operator policy.
- Cloud/training: Cloud Run in this repo vs external cloud/provider packages and research-era LoRA docs.
- Public release hygiene: business, outreach, legal, planning, strategy, vision, archive material.

## Primary Sources To Check

- `README.md`
- `docs/README.md`
- `docs/architecture/README.md`
- `KESTREL_FEATURES.md`
- `kestrel_sovereign/data/feature_registry.toml`
- `pyproject.toml`
- `docs/generated/README.md`
- `docs/audit/DOCUMENTATION_AUDIT_5_2026.md`
- `docs/archive/meta/DOCUMENTATION_INVENTORY_2025.md`
- `docs/audit/issues/`
- Current code paths for the reviewed lane.

## Required Report Shape

Each lane report should include:

- Scope reviewed.
- Canonical doc recommendation.
- Stale or conflicting claims with file paths.
- Code/package evidence with file paths.
- Docs to update.
- Docs to archive or mark historical.
- Generated docs to regenerate.
- Open questions.
- Suggested first PR slice.

