## Parent

Part of #255.

## Problem

Model mandate and fallback behavior crosses UI endpoint rewriting, API model endpoints, command selection, runtime provider routing, discovery cache, and generated audience docs. Component tests exist, but the seam campaign is only partial.

## Goal

Prove a selected or discovered model route remains consistent across UI, API, command, and runtime paths without hardcoded current-model drift.

## Required scenarios

- UI selects a model/provider and runtime generation uses that same route
- command-driven model selection and API-driven model selection converge on `llm_service.get_model_preference()`
- discovered provider models update the local cache and generated docs without hardcoded model names
- fallback routing does not override explicit user preference unless the selected provider is unavailable

## Invariants

- `kestrel.toml` is the pre-start source of truth and discovery only updates the cache/runtime view
- no endpoint duplicates fallback logic already owned by the LLM service
- generated audience docs resolve `auto` through discovered/cache-backed selection
- route rewrites preserve host-level auth endpoints and per-agent model endpoints

## Proof expectations

- unit seam tests for UI/API/command/runtime preference convergence
- fixture-backed discovery-cache tests for generated docs and startup behavior
- update `docs/audit/SEAM_CAMPAIGNS.md` when proven
