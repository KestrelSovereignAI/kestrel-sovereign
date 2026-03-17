## Problem

Kestrel’s platform layer spans provider adapters, model routing, API contracts, frontend behavior, CLI, config loading, and deployment targets. This is where source-of-truth drift is most likely because the same behavior is often surfaced through multiple interfaces.

## Goal

Audit the platform surfaces so provider capabilities, model mandate behavior, APIs, frontend clients, CLI commands, and deployment contracts are all proven against the same canonical logic.

## In Scope

- LLM provider adapters, registry, retry, streaming, structured output, vision, usage tracking
- model discovery, catalog, metadata, and mandate routing
- REST, SSE, and OpenAI-compatible API contracts
- frontend API client and static SPA behavior
- CLI workflows and config loading
- Docker targets, Cloud Run scripts, and cloud provider config contracts

## Source-of-Truth Areas

- `kestrel_sovereign/llm/`
- `endpoints/`
- `static/js/`
- `kestrel_cli.py`
- `kestrel_sovereign/config.py`
- `docker/`
- `scripts/cloudrun/`

## Required Proof

- unit tests for mandate logic, provider capability handling, config resolution
- integration tests for provider routing, API contracts, and frontend-client behavior
- e2e tests for critical UI workflows
- dual-backend and real-provider tests where the catalog claims them
- infrastructure tests for deployment scripts and container assumptions

## High-Risk Seams

- model mandate behavior diverging across UI, command, API, and runtime
- provider capability metadata vs actual adapter behavior
- frontend API client auth/session logic vs backend contract
- OpenAI-compatible API drift from advertised schema
- config fallback behavior hiding deprecated or conflicting settings

## Exit Criteria

- model preference routing is proven as a single source of truth across all surfaces
- API and frontend contracts are covered by both contract tests and user-flow tests
- deployment scripts and Docker targets are validated against their documented claims
