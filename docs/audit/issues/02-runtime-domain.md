## Problem

The runtime layer is where user intent, LLM responses, tools, commands, context, and background tasking meet. It is also where “small” inconsistencies become user-visible regressions. Kestrel already has substantial runtime tests, but the coverage is not yet organized around runtime invariants and whole-flow seam failures.

## Goal

Audit the agent runtime as a coherent system, proving that context assembly, tool loops, commands, lifecycle transitions, and A2A orchestration behave consistently under normal and failure conditions.

## In Scope

- `KestrelAgent` request processing and tool loop
- streaming and cancellation behavior
- command routing and feature delegation
- context assembly, token budgeting, conversation and memory retrieval
- bootstrap lifecycle, sleep/backup/retirement flows
- A2A tasking, task state transitions, notifications, observability

## Source-of-Truth Areas

- `kestrel_sovereign/kestrel_agent.py`
- `kestrel_sovereign/agent/`
- `kestrel_sovereign/command_handler.py`
- `kestrel_sovereign/kestrel_agent_tools.py`
- `kestrel_sovereign/a2a/`
- `kestrel_sovereign/bootstrap/`

## Required Proof

- unit tests for loop invariants, state transitions, and cancellation
- integration tests for command/tool/context/A2A flows
- e2e tests covering UI and API entry points into the same runtime behavior
- adversarial tests for tool-loop abuse, delegation abuse, and state desync

## High-Risk Seams

- command path vs API path vs UI path divergence
- tool-call iterations, recursion limits, and cancellation
- context budget allocation under long sessions and privacy constraints
- A2A task state consistency across storage and SSE notifications
- bootstrap state transitions vs persisted SOUL and runtime identity

## Exit Criteria

- each runtime surface has a clear canonical path and matching tests
- user-visible runtime behavior is consistent across commands, API, and UI
- cancellation, retries, and task transitions fail clearly instead of silently drifting
