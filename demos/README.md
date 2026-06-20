# Kestrel Sovereign — Demos

Every demo lives in its own subdirectory. One shared helpers module; no cross-tree imports.

## Stories vs. vignettes

* **Stories** (`spawn`, `technical`, `falconer`) — multi-act narratives, often
  tens of screenshots. Built for keynote walkthroughs.
* **Vignettes** — short, single-feature demos (~30s, 5–8 screenshots, one
  happy path). Built to ride along with the feature PR that adds them and
  double as e2e regression evidence.

The convention going forward: **a feature PR ships its vignette under
`demos/<feature-name>/`**. Copy `demos/TEMPLATE/` to start a new one.

## Layout

```
demos/
├── shared/
│   └── demo_helpers.cjs     # auth, panel navigation, spawn API helpers
├── run.sh                   # canonical entry point — isolates demo runs
├── TEMPLATE/                # copy this to start a new vignette
├── spawn/                   # Issue #354 — narrated spawn lifecycle (story)
├── technical/               # Track A — DID, Constitution, Memory, Privacy, Sovereignty, Permissions (story)
├── falconer/                # Falconer product demo — Claws → Talon mesh dispatch (story)
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

**Always use `kestrel demo run`.** It spins up an isolated demo agent on its own port so the demo never touches your live `localhost:8888` data.

```bash
kestrel demo run technical
kestrel demo run spawn
kestrel demo run falconer

# Different port if 8900 is busy:
kestrel demo run technical --port 9001
```

The runner:
1. Creates a fresh DB at `agent_data/demo/` via `scripts/setup_demo_agent.py`
2. Starts a dedicated uvicorn on `DEMO_PORT` (default 8900) against that DB
3. Runs the Playwright demo against the isolated server
4. Stops the isolated server on exit (even on failure)

**Never point Playwright at your live server directly.** The raw command `cd demos/<name> && npx playwright test --config=config.cjs` will hit whatever `KESTREL_URL` resolves to — if that's your working instance, Act 3's `clearConversationHistory` and Act 6's permission toggles will mutate real data.

Outputs land in `demos/<name>/demo-output/` — screenshots, `narration.md`, and (via Playwright) per-test video under `demo-output/playwright/`.

## Documentation evidence

The generated docs evidence index lives at [`docs/generated/DEMO_EVIDENCE.md`](../docs/generated/DEMO_EVIDENCE.md). Regenerate it after adding, renaming, or removing a demo:

```bash
uv run python scripts/generate_demo_evidence_docs.py
uv run python scripts/generate_demo_evidence_docs.py --check
```

When a user-facing doc describes a UI workflow, link it to the matching `demo.cjs`, `eye.toml`, and generated evidence row so Kestrel Flight and Kestrel Eye can keep the prose honest.

## Vision review (kestrel-eye)

Screenshots are reviewed by [kestrel-eye](https://github.com/KestrelSovereignAI/kestrel-eye) using a cheap vision model.

```bash
kestrel-eye review --config demos/<name>/eye.toml
kestrel-eye run    --config demos/<name>/eye.toml --loop
```

## Writing a new vignette

```bash
cp -R demos/TEMPLATE demos/<feature-name>
cd demos/<feature-name>
# Edit demo.cjs (beats), narration.md (the story), eye.toml (expectations)
kestrel demo run <feature-name>
```

See `demos/TEMPLATE/README.md` for the full convention.

Do **not** write demo files into `tests/e2e/`. That directory is for
`test_*.spec.cjs` only.
