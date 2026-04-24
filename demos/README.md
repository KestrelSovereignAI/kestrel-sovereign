# Kestrel Sovereign — Demos

Every demo lives in its own subdirectory. One shared helpers module; no cross-tree imports.

## Layout

```
demos/
├── shared/
│   └── demo_helpers.cjs     # auth, panel navigation, spawn API helpers
├── run.sh                   # canonical entry point — isolates demo runs
├── spawn/                   # Issue #354 — narrated spawn lifecycle
├── technical/               # Track A — DID, Constitution, Memory, Privacy, Sovereignty, Permissions, Memory Hygiene
├── falconer/                # Falconer product demo — Claws → Talon mesh dispatch
├── metrics/                 # Observability dashboard — KPI cards, timeline, duration, distribution
├── feature-store/           # Feature Store — browse features, search, drill into skills
└── tasks/                   # Tasks & Activity — background task queue + real-time activity log
```

Each demo directory contains:

| File | Purpose |
|---|---|
| `demo.cjs` | Playwright-scripted demo |
| `config.cjs` | Playwright config (video on, slowMo, 1440x900 viewport) |
| `eye.toml` | kestrel-eye review config (not every demo has one yet) |
| `narration.md` | Static narration doc (beat-by-beat story) |
| `presenter.md` | Speaker notes for live presentation |
| `demo-output/` | Generated: screenshots, videos, `narration.md` transcript (gitignored) |

## Running a demo

**Always use `demos/run.sh`.** It spins up an isolated demo agent on its own port so the demo never touches your live `localhost:8888` data.

```bash
demos/run.sh technical
demos/run.sh spawn
demos/run.sh falconer

# Different port if 8900 is busy:
DEMO_PORT=9001 demos/run.sh technical
```

The runner:
1. Creates a fresh DB at `agent_data/demo/` via `scripts/setup_demo_agent.py`
2. Starts a dedicated uvicorn on `DEMO_PORT` (default 8900) against that DB
3. Runs the Playwright demo against the isolated server
4. Stops the isolated server on exit (even on failure)

**Never point Playwright at your live server directly.** The raw command `cd demos/<name> && npx playwright test --config=config.cjs` will hit whatever `KESTREL_URL` resolves to — if that's your working instance, Act 3's `clearConversationHistory` and Act 6's permission toggles will mutate real data.

Outputs land in `demos/<name>/demo-output/` — screenshots, `narration.md`, and (via Playwright) per-test video under `demo-output/playwright/`.

## Vision review (kestrel-eye)

Screenshots are reviewed by [kestrel-eye](https://github.com/KestrelSovereignAI/kestrel-eye) using a cheap vision model.

```bash
kestrel-eye review --config demos/<name>/eye.toml
kestrel-eye run    --config demos/<name>/eye.toml --loop
```

## Writing a new demo

1. `mkdir demos/<name>` — one directory per demo.
2. Copy one of the existing `config.cjs` as a starting template; set `testMatch: 'demo.cjs'` and `outputDir` to `./demo-output/playwright`.
3. Write `demo.cjs`. Import helpers from `../shared/demo_helpers.cjs`.
4. Write `narration.md` (the story) and (optionally) `eye.toml` (the expectations).
5. Add `presenter.md` if the demo has a live-delivery script.

Do **not** write demo files into `tests/e2e/`. That directory is for `test_*.spec.cjs` only.
