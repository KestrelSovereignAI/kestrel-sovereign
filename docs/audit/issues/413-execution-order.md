# Epic 413 Execution Order

Parent epic: [#413](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/413)

## Recommended order

1. [#419](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/419) First slice for `#416`: add `MODULAR_RUNTIME.md` and a contract test scaffold
2. [#416](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/416) Define the modular runtime kernel boundary for `kestrel-sovereign`
3. [#414](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/414) Introduce stable module descriptors above feature discovery
4. [#415](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/415) Extract feature-specific startup wiring out of `KestrelAgent.initialize()`
5. [#418](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/418) Split portable state fragments from runtime behavior using legal personhood as a proving ground
6. [#417](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/417) Move toward capability-mounted routers and enforce modular seams with tests

## Why this order

- `#419` creates the smallest safe first landing so Talon does not overreach.
- `#416` then expands the kernel-boundary truth once the first artifact exists.
- `#414` creates the metadata layer needed for safer extraction work.
- `#415` reduces the core-runtime behavior sink once the descriptor path exists.
- `#418` uses legal continuity as the first serious proving ground for portable module-owned state.
- `#417` lands router modularization and seam enforcement after the runtime contracts are clearer.

## Talon batch hint

From the repo root, a Talon batch run can use:

```bash
kestrel-talon batch --prd docs/audit/issues/413-prd.json --max-iterations 20
```

That PRD is intentionally ordered to match the dependency chain above.
