---
type: Issue Body
title: 255 Privacy Active Session Seam
description: 'Part of #255.'
resource: /docs/audit/issues/255-privacy-active-session-seam.md
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

# 255 Privacy Active Session Seam

## Parent

Part of #255.

## Problem

Privacy-mode transitions are covered at the API, command-handler, and model-restore level, but the remaining seam risk is active runtime state: an in-flight stream/session can overlap a transition into or out of local-only privacy mode.

## Goal

Prove privacy transitions are atomic enough for active sessions and streams: storage, model routing, voice side effects, and isolated-session behavior must not observe mixed privacy state.

## Required scenarios

- stream or active request is in progress while privacy mode changes to `isolated`
- privacy mode changes back to `normal` while the previous model route is restored
- command path and API path converge on the same async transition semantics under active-session conditions
- isolated-session save cannot persist data through the wrong storage/privacy adapter

## Invariants

- no coroutine leaks or background consent calls are left unawaited
- local-only modes cannot route to cloud providers during or after transition
- cloud-allowed modes restore the prior resolved cloud route exactly once
- storage persistence follows the active privacy config, not stale request-local state

## Proof expectations

- fast unit seam tests with mocked streaming/storage/model boundaries
- at least one integration-style test covering the API path during a simulated active stream
- update `docs/audit/SEAM_CAMPAIGNS.md` when proven
