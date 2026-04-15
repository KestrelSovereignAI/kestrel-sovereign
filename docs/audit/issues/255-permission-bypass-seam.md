## Parent

Part of #255.

## Problem

Permission checks can be bypassed when privileged behavior is exposed through multiple entrypoints. MCP, A2A, tools, commands, and compute policies need one adversarial campaign that proves they share the same security decision.

## Goal

Red-team privileged operations through every protocol/entrypoint and prove they cannot bypass approval or audit requirements.

## Required scenarios

- command-triggered privileged action vs tool-triggered privileged action
- MCP gateway/tool invocation attempts a protected filesystem/process/network operation
- A2A task invokes an operation requiring approval
- negative cases for malformed permission scope, stale approval, and replayed approval

## Invariants

- approval decisions are enforced before side effects
- equivalent privileged operations receive equivalent security treatment regardless of entrypoint
- denials are auditable and do not produce partial side effects
- replay/stale approval attempts fail closed

## Proof expectations

- direct adversarial tests for command/tool/MCP/A2A paths
- fixture that asserts shared approval/audit boundary, not copied policy logic
- update `docs/audit/SEAM_CAMPAIGNS.md` when proven
