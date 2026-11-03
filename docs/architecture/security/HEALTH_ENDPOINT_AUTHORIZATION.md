---
type: Security Guide
title: Health Endpoint Authorization
description: Public readiness and authenticated operator-diagnostic exposure boundary.
status: active
owner: security
canonical: true
privacy: public
---

# Health endpoint authorization

Kestrel separates platform-probe readiness from operator diagnostics. Health
state is useful to load balancers, but provider routes, active models, disk
capacity, agent names, constitutional state, and backend failure messages are
deployment secrets rather than liveness data.

## Public readiness contract

`GET /health` requires no credential so Cloud Run, container orchestrators, and
external load balancers can decide whether to route traffic. Its JSON schema is
intentionally limited to:

| Field | Meaning |
|---|---|
| `status` | Aggregate `ok`, `degraded`, or `unhealthy` readiness |
| `agent_initialized` | Whether at least one runtime initialized |

HTTP 200 means ready. HTTP 503 means degraded, unhealthy, or constitutionally
restricted. The public response never distinguishes mandatory-feature,
identity, safe-mode, audit-pending, or startup-exception causes. It never
contains agent names, failure records, provider/model/reachability data, disk
values, paths, hostnames, or exception strings.

This small response is a compatibility contract for deployment probes. Adding
a diagnostic field to `/health` is a security-surface change, not a harmless
observability enhancement.

## Authenticated operator contract

`GET /health/detailed` passes through Kestrel's normal global authentication
middleware. A valid `X-API-Key`, bearer API key, bearer JWT, or signed OAuth
session can reach it; an unauthenticated request receives HTTP 401 before any
health feature or diagnostic check runs.

Authenticated output may include per-check database, LLM, memory, disk, context,
and bootstrap results, plus controlled multi-agent and safe-mode records. These
details can contain operational failure messages and therefore must not be
copied into public routes, middleware errors, or load-balancer logs.

## Readiness and diagnostic status answer different questions

`/health` and the HealthFeature tick (also served by `/health/detailed`) can
disagree, and both can be right. A tick that reads `degraded` while `/health`
returns `ok` is the expected outcome of a standing warning, not a fault in
either surface.

- `/health` asks whether the host may receive traffic. It reads only the
  readiness latches: a startup error, mandatory-feature or identity failures,
  constitutional Safe Mode, scheduler worker availability, and invocation
  ownership. It never runs the per-check diagnostics.
- The tick runs the standard checks
  (`kestrel_sovereign/features/health/checks.py:run_standard_checks`) and
  derives its status with `derive_overall_status`: any `warn` or non-critical
  `fail` gives `degraded`; a `fail` of `database`, `llm_service` or
  `scheduler_liveness` gives `unhealthy`. A warning names a condition an
  operator should act on, such as a schedule the scheduler disabled or a
  route in use whose last call failed, while the agent keeps serving.

A tick that is `degraded` on every tick therefore names a standing condition.
Read its warning checks rather than treating the status as noise. The checks
only warn about what affects this agent (#3873): a provider configured but
selected by no route in use is reported as `standby` by `llm_service` and
`model_discovery` without degrading the tick, and a scheduler-paused schedule
that a later registration under the same idempotency key replaced is reported
as superseded history rather than a standing disablement.

`llm_service` never sends a completion to judge a route. The routes in use
are the ones the next turn would try: an active private inference lease the
next call would take (`remote_gpu`, which has no cloud fallback), else the
routes routing selects. Each is judged on how the last real call on it ended,
including a lease refusing a call before any adapter ran; with no call since
startup it is reported `unprobed`. A route disabled this session by a
permanent auth failure is not standby: routing skips it until its key is
rotated and the service restarted, so it is reported `disabled`, with the
recorded reason and its last call, and warns.

## Regression history

[Issue #137](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/137)
classified unauthenticated provider, model, identity, command, and file-surface
disclosure as HIGH severity. [PR #158](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/158)
removed those diagnostic APIs from the public allowlist while retaining the
minimal health probe.

`/health/detailed` later reintroduced the same information class through a new
public endpoint, and `/health` accumulated reachability and failure records.
[Issue #2611](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2611)
restored the original boundary: public aggregate readiness, authenticated
diagnostics. Tests exercise remote unauthenticated, API-key, OAuth-session,
degraded, multi-agent, and safe-mode cases so future health additions cannot
silently reopen the disclosure.
