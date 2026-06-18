---
type: Issue Body
title: 255 Cloud Local Drift Seam
description: 'Part of #255.'
resource: /docs/audit/issues/255-cloud-local-drift-seam.md
tags:
- docs
- audit
- issue-body
timestamp: '2026-06-18T00:00:00Z'
status: snapshot
owner: documentation
canonical: false
generated: false
privacy: public
---

# 255 Cloud Local Drift Seam

## Parent

Part of #255.

## Problem

Cloud/local provider behavior crosses model discovery, local cache, startup config, Ollama, RunPod, Vast.ai, GCP, Vertex, and Cloud Run. Existing proof is partial and historical hardcoded model drift has already caused regressions.

## Goal

Prove cloud/local provider selection and deployment defaults do not drift from discovery/cache truth or block runtime async paths.

## Required scenarios

- startup uses config/cache before live discovery is available
- discovery refresh updates cache and model ranking without hardcoded latest-model names
- local Ollama and cloud RunPod/Vast/GCP/Vertex adapters fail closed when credentials/runtime are missing
- deployment defaults do not contradict runtime provider assumptions

## Invariants

- config is the pre-start source of truth; discovery updates runtime/cache afterward
- no hardcoded provider "latest" model names are required for correctness
- missing cloud credentials produce legible unavailable status, not half-configured providers
- blocking cloud/local adapter calls are isolated from the event loop

## Proof expectations

- provider contract tests for discovery/cache/startup fallback
- mocked cloud adapter tests for unavailable and misconfigured states
- update `docs/audit/SEAM_CAMPAIGNS.md` when proven
