# First slice for #416: add `MODULAR_RUNTIME.md` and a contract test scaffold

Parent issue: #416
Parent epic: #413

## Problem

Issue #416 is directionally right but still broad enough that an autonomous coding agent could overreach.

We need a minimal first landing that:

- creates the canonical modular-runtime doc
- names the kernel/module contract concepts
- adds a small test scaffold
- does not attempt major runtime refactors yet

## Goal

Land the smallest useful version of the modular-runtime boundary as codebase truth.

## Scope

- add `docs/architecture/core/MODULAR_RUNTIME.md`
- define:
  - runtime kernel responsibilities
  - module responsibilities
  - stable module id
  - provided/required capabilities
  - owned state
  - startup hooks
  - router contribution
  - export/import fragments
- add `tests/unit/test_module_contracts.py`
- keep the test lightweight and structural for now

## Non-goals

- refactoring feature discovery
- changing startup behavior
- moving routes
- moving legal/runtime state ownership

## Acceptance criteria

- `MODULAR_RUNTIME.md` exists and is specific enough to guide later work
- `test_module_contracts.py` exists and passes
- no production runtime behavior changes are required to complete this slice

## Talon note

Do not “help” by implementing descriptors or refactoring startup in this issue. This slice is successful only if it stays small and establishes vocabulary cleanly.
