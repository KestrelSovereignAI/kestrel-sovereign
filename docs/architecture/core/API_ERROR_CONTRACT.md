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

Callers may supply `X-Correlation-ID`, but Kestrel preserves it only when it is
1–128 characters from the support-safe set `A-Z a-z 0-9 . _ : -`. Invalid
values are replaced with a newly generated opaque ID before they reach logs,
JSON, or response headers. HTTP statuses that forbid response bodies (such as
204 and 304) carry only the correlation header and any original protocol
headers; they cannot carry the JSON envelope.

The production middleware stack establishes correlation context before
authentication, so authentication and CSRF failures use the same envelope and
header contract as route exceptions. Rejected CORS preflights are normalized at
the CORS boundary because they do not enter that inner request context. For
configured cross-origin console clients, CORS permits the console's `PATCH`,
CSRF, and destructive-confirmation request surface and exposes
`X-Correlation-ID`, `X-Request-ID`, and `X-Session-ID` so browser JavaScript can
preserve support references, cancellation, and session continuity.

The global handlers in `kestrel_sovereign.api_errors` translate framework
HTTP exceptions, request-validation failures, and unexpected exceptions into
this envelope. Validation responses retain locations, messages, and error
codes but never echo the rejected `input` or validator context. Unexpected
exceptions are logged with their correlation ID and return only a generic 500
message.

Endpoints that need a domain-specific code should raise `ApiHTTPException` or
return `api_error_response()` when a concrete response is required. Plain
`HTTPException` endpoints remain compatible and receive a fallback code such
as `http_404`.

Two wire-level exceptions are intentional and covered by tests:

- `/health` and `/health/detailed` retain their stable readiness/diagnostic
  schemas for load balancers and operator health tooling instead of changing
  shape based on the API error contract.
- `/api/github/{path}` preserves the status and JSON body of an HTTP response
  actually received from GitHub because it is a transparent, repo-scoped
  proxy. Kestrel-authored scope/configuration failures and upstream transport
  or decoding failures still use the canonical envelope and never expose raw
  exception text.

The Sovereign Console parses every failed request—including stream setup—with
`parseResponseError()` and throws `ApiError`. `ApiError.body` preserves the
parsed payload for structured consumers such as upgrade/tier gates, while its
`code`, `message`, `details`, and `correlationId` are normalized for consumers
and user display. Detail collections are bounded, and object-valued locations
are discarded rather than stringified.
Callers must render those display fields through `textContent` or an equivalent
text-safe DOM API; raw bodies must never be inserted into HTML. The shared toast
and panel-error renderers follow that rule by default. The separately named
`Toast.showTrustedHtml()` path is reserved for console-owned, pre-sanitized rich
fragments such as the upgrade CTA.
