## Problem

`KESTREL_FEATURES.md` makes a broad set of product, architectural, security, privacy, and deployment claims, but the current test suite is organized primarily by implementation history. We have substantial coverage already, including unit, integration, e2e, real-LLM, load, and adversarial tests, but we do not yet have a single audit matrix proving every catalog claim and every high-risk seam.

That leaves room for drift:

- feature claims without explicit proof
- duplicate logic paths with inconsistent behavior
- cross-feature regressions that pass isolated tests
- security, privacy, and sovereignty guarantees that are asserted but not systematically red-teamed

This audit program is the control layer for closing that gap.

## Goal

Create and execute a whole-of-vision, test-first audit and red-team program for every feature listed in `KESTREL_FEATURES.md`, with no workaround fixes, no hidden fallbacks, and no duplicate sources of truth.

## Scope

- build a canonical audit matrix from `KESTREL_FEATURES.md`
- map every claim to source-of-truth code, user-visible surfaces, invariants, and proof requirements
- create domain-level audit streams and track them via GitHub issues
- identify proof gaps in unit, integration, e2e, adversarial, dual-backend, load, and real-resource tests
- fix root causes only, consolidating duplicated logic where needed
- run whole-system seam campaigns across UI, API, services, storage, providers, and background workers

## Non-goals

- cosmetic cleanup without proof impact
- adding fallback paths to mask failures
- preserving duplicate implementations for convenience
- closing tickets on partial confidence or “good enough” manual testing

## Audit Principles

- Test first. Every material change starts by expressing the expected invariant in a failing or tightened test.
- One source of truth. If audit work reveals multiple implementations of the same behavior, consolidate.
- Fail fast and clearly. No silent recovery for high-stakes paths like auth, privacy, encryption, sovereignty, or constitutional enforcement.
- Attack seams, not just components. Most serious regressions happen in transitions between systems.
- Track proof explicitly. A feature claim is not done until the required proof layers are green.

## Audit Matrix Requirements

For every feature claim, record:

- claim being made
- source-of-truth file(s)
- user-visible surfaces: command, tool, endpoint, UI, config, background worker, hook
- invariants: must always, must never
- failure and abuse cases
- required proof layers
- current proof status and gaps

## Domain Streams

- Foundation: constitution, identity, continuity, sovereignty, storage, memory
- Runtime: agent loop, tools, commands, context, bootstrap, lifecycle, A2A, observability
- Security: privacy, auth, permissions, keys, hooks, webhooks, adversarial bypass attempts
- Platform: LLM providers, mandate, streaming, APIs, UI, CLI, deployment, config

## Exit Criteria

- every section in `KESTREL_FEATURES.md` is represented in the audit matrix
- every high-risk feature or seam has an owner issue
- every claim has explicit proof or an explicit unresolved issue
- SQLite/PostgreSQL parity is proven where claimed
- adversarial coverage exists for constitutional, auth, privacy, permission, and delegation seams
- no audit ticket is closed with workaround-only changes

## Tracking

This umbrella issue owns the domain issues below and acts as the executive ledger for the audit program.

- #261 Audit domain: foundation, identity, storage, memory, sovereignty
- #259 Audit domain: runtime, context, tools, commands, A2A
- #258 Audit domain: security, privacy, auth, permissions, keys
- #257 Audit domain: LLM platform, APIs, UI, CLI, deployment
- #256 Build canonical audit matrix from `KESTREL_FEATURES.md`
- #255 Run cross-feature seam and red-team campaigns
