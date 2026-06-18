---
type: Issue Body
title: 03 Security Domain
description: Security-sensitive behavior is distributed across auth, permissions,
  privacy modes, key storage, hooks, compute controls, and public webhook entry points.
  That distribution is n...
resource: /docs/audit/issues/03-security-domain.md
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

# 03 Security Domain

## Problem

Security-sensitive behavior is distributed across auth, permissions, privacy modes, key storage, hooks, compute controls, and public webhook entry points. That distribution is necessary, but it creates bypass risk if any layer assumes another one already enforced a rule.

## Goal

Audit and red-team all security-critical paths so privacy, permission, and authentication guarantees hold under malicious inputs and cross-feature routing.

## In Scope

- API key auth, bearer auth, query-param SSE auth, localhost bootstrap auth
- hierarchical permissions and approval queue semantics
- privacy modes, PII scrubbing, anonymization, cloud-LLM restrictions
- encrypted key storage, rotation, layered resolution
- security hooks, compute gating, webhook validation
- bypass attempts through tools, commands, MCP, A2A, and web endpoints

## Source-of-Truth Areas

- `server.py`
- `endpoints/security.py`
- `kestrel_sovereign/privacy.py`
- `kestrel_sovereign/storage/privacy_wrapper.py`
- `kestrel_sovereign/security/`
- `kestrel_sovereign/features/security/`
- `kestrel_sovereign/features/privacy/`
- `kestrel_sovereign/features/webhooks/`

## Required Proof

- unit tests for permission resolution, privacy flags, and key handling
- integration tests for authenticated and unauthenticated access paths
- adversarial suites for jailbreak, auth bypass, approval bypass, prompt injection, and delegation bypass
- e2e tests for security-sensitive UI flows

## High-Risk Seams

- SSE and browser bootstrap auth semantics vs standard API auth
- privacy mode transitions during active sessions
- anonymization claims vs actual persisted content
- approval decisions bypassed through alternate entry points
- webhook/public endpoint validation and audit logging

## Exit Criteria

- every auth surface has explicit deny-path tests
- every privacy claim is tested against actual persistence and provider routing
- every permissioned feature has bypass-focused seam tests
- no security behavior depends on undocumented implicit ordering between layers
