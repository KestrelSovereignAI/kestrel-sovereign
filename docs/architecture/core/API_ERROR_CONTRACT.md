---
type: Architecture
title: API Error Contract
description: Canonical HTTP error envelopes, correlation IDs, and frontend normalization.
status: active
privacy: public
---

# API error contract

Kestrel HTTP APIs use one canonical JSON envelope for errors:

```json
{
  "error": {
    "code": "input_required",
    "message": "Input not provided.",
    "details": [
      {"location": ["body", "input"], "message": "Field required", "code": "missing"}
    ],
    "correlation_id": "support-safe-request-id"
  },
  "detail": "Input not provided."
}
```

`error.code` is a stable machine-readable identifier. `error.message` and
optional `error.details` are safe for user display. The top-level `detail`
field remains during migration for FastAPI and older-client compatibility.
Every error also returns the same support identifier in the
`X-Correlation-ID` response header.

The production middleware stack establishes correlation context before
authentication, so authentication and CSRF failures use the same envelope and
header contract as route exceptions. For configured cross-origin console
clients, CORS exposes `X-Correlation-ID` so browser JavaScript can read and
display the support reference.

The global handlers in `kestrel_sovereign.api_errors` translate framework
HTTP exceptions, request-validation failures, and unexpected exceptions into
this envelope. Validation responses retain locations, messages, and error
codes but never echo the rejected `input` or validator context. Unexpected
exceptions are logged with their correlation ID and return only a generic 500
message.

Endpoints that need a domain-specific code should raise `ApiHTTPException`.
Unmigrated `HTTPException` endpoints remain compatible and receive a fallback
code such as `http_404`. The agent invoke/stream setup endpoints are the first
representative migrated slice; other endpoints can adopt stable codes as they
are changed rather than through a mechanical rewrite.

The Sovereign Console parses every failed request—including stream setup—with
`parseResponseError()` and throws `ApiError`. `ApiError.body` preserves the
parsed payload for structured consumers such as upgrade/tier gates, while its
`message`, `details`, and `correlationId` are normalized for user display.
Callers must render those display fields through `textContent` or an equivalent
text-safe DOM API; raw bodies must never be inserted into HTML.
