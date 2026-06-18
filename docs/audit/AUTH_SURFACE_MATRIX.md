---
type: Audit Ledger
title: Auth Surface Matrix
description: The live auth model is enforced centrally in [`server.py`](../../server.py),
  with endpoint semantics layered on top.
resource: /docs/audit/AUTH_SURFACE_MATRIX.md
tags:
- docs
- audit
- audit-ledger
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: documentation
canonical: false
generated: false
privacy: public
---

# Auth Surface Matrix

The live auth model is enforced centrally in [`server.py`](../../server.py), with endpoint semantics layered on top.

| Class | Description | Representative routes |
|---|---|---|
| `Public` | No API key or session required | `/health`, `/health/detailed`, `/webhooks/stripe/crypto` |
| `Public-Localhost` | Public only from localhost-like callers when bootstrap is enabled | `/api/auth/key` |
| `OAuth public entrypoints` | Browser sign-in flow entrypoints | `/auth/login`, `/auth/callback`, `/auth/logout` |
| `APIKeyOrSession` | Protected routes that accept either API key auth or OAuth session auth | most `/agent/*` and `/api/*` routes |
| `APIKeyOrSession+SSEQuery` | Protected routes that also permit `?api_key=` because `EventSource` cannot send custom headers | `/agent/stream`, `/agent/notifications/sse` |
| `OAuthSessionSemantic` | Route may pass middleware via API key but still only returns success data with a real browser session | `/auth/me` |
| `Browser-Conditional` | Root page behavior depends on UI mode and OAuth-required mode | `/` |

## Key findings

- The auth model is not just public versus protected.
- Most protected routes are effectively `API key OR OAuth session`.
- `/auth/me` is not middleware-public; it is semantically session-backed.
- `/api/auth/key` is intentionally narrow and must remain localhost-scoped.
