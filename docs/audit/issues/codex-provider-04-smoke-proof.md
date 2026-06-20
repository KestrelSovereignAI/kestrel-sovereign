---
type: Issue Body
title: Add smoke proof and docs for sovereign `codex_provider`
description: 'Parent: Codex provider adapter and model-selection issues'
resource: /docs/audit/issues/codex-provider-04-smoke-proof.md
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

# Add smoke proof and docs for sovereign `codex_provider`

Parent: Codex provider adapter and model-selection issues

## Problem

If `codex_provider` lands without proof, it will be easy for it to drift into a demo-only path.

## Goal

Add a small but real proof path and documentation showing how to use Codex as a sovereign provider.

## Scope

- add focused tests or smoke proof for provider init and basic invoke path
- document config examples
- document expected failure modes and prerequisites

## Acceptance criteria

- there is a documented smoke path for Codex provider
- there is at least one automated proof covering initialization and basic response flow
- docs clearly state prerequisites and limitations

## References

- `llm_config.toml.example`
- `README.md`
- sovereign LLM provider docs

## Talon note

This issue is where the implementation stops being “we think this works” and becomes “we can prove this path exists.”
