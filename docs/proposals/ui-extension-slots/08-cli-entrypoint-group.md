# 08 — CLI extension via entry-point group

**Type:** Feature (CLI)
**Depends on:** none (independent quick win)
**Risk:** Low

## Problem

The `kestrel` CLI is argparse-based and every subcommand group is a **hardcoded
import**: `from kestrel_sovereign.cli_x import add_x_subparser; add_x_subparser(...)`
in [cli.py:1449-1784](../../../kestrel_sovereign/cli.py), plus a hardcoded dispatch
dict at [cli.py:1826](../../../kestrel_sovereign/cli.py). A feature cannot add a
`kestrel <feature> ...` subcommand without editing core.

## Goal

Discover CLI extensions via a new entry-point group, mirroring the feature discovery
the backend already uses (`kestrel_sovereign.features`), so a feature package can
contribute a subcommand group with **zero** core edits.

## Design

- New entry-point group: `kestrel_sovereign.cli`.
- Each entry point resolves to a callable matching the **existing convention**
  exactly — `add_<name>_subparser(subparsers)` — and registers its own dispatch
  handler into a registry the dispatcher drains, replacing the hardcoded dict for
  extension commands.
- `build_parser()` ([cli.py:1436](../../../kestrel_sovereign/cli.py)) discovers and
  invokes registered extensions in a loop after the core groups.
- Reuse `discover_entry_point_classes`-style discovery
  ([entrypoints.py](../../../kestrel_sovereign/entrypoints.py)) adapted for callables.

## Critical constraint (must be in the ticket and the docs)

The CLI runs **host-side, out-of-process from the live agent.** A feature CLI command
**cannot** touch in-process feature state. It operates against host config or the
agent's HTTP API — exactly like `cli_features` does today. So:

- Feature CLI commands are **thin clients over the feature's router** (the real
  primitive). Document this so authors don't try to call in-process tools from the
  CLI and get confused.
- This is why the router (tickets 02/05 backend) is built first conceptually; the CLI
  is a convenience wrapper, though it can ship independently since it only needs the
  HTTP API.

## Tasks

1. Add `kestrel_sovereign.cli` entry-point discovery to `build_parser()`.
2. Replace the hardcoded dispatch dict lookup with: core dict first, then the
   extension registry.
3. Name collision handling: a feature group name colliding with a core command is
   rejected with a clear error (core wins), logged at startup.
4. Reference example: a feature ships `kestrel <feature> status` that hits its own
   router endpoint.
5. Tests: discovery, dispatch, collision rejection, missing-feature no-op.

## Acceptance criteria

- An installed feature adds a working `kestrel <feature> ...` subcommand with no edit
  to `cli.py`.
- Core commands are unaffected; a malicious/buggy extension that throws at
  registration is isolated (logged, skipped) and does not break the whole CLI.
- Docs state the out-of-process constraint and the "thin client over router" pattern.
