# Demo Template — Copy This For A New Vignette

A **vignette** is a short, single-feature demo (~30s, 5–8 screenshots, one happy
path). It doubles as an e2e regression check and as documentation for the
feature. Compare with the *story* demos (`spawn`, `technical`, `falconer`)
which are longer, multi-act narratives.

## Convention

Every feature PR should ship a vignette under `demos/<feature-name>/`. If the
PR adds a new UI surface, the demo proves it works end-to-end against an
isolated agent. If a future PR breaks the surface, the demo notices.

## How to spin one up

```bash
cp -R demos/TEMPLATE demos/<feature-name>
cd demos/<feature-name>
# 1. Edit demo.cjs — replace the placeholder beats with your feature's flow
# 2. Edit narration.md — beat-by-beat story (consumed by kestrel-eye)
# 3. Edit eye.toml — list expected screenshots and what each should show
# 4. Run it:
kestrel demo run <feature-name>
```

The runner spins an isolated agent on `agent_data/demo/` (port 8900 by
default) — no risk to your live `localhost:8888` multi_agent. See
[`../README.md`](../README.md) for the full safety story.

## What each file is for

| File | Purpose |
|---|---|
| `demo.cjs` | Playwright script — drives the UI, takes screenshots, narrates beats |
| `config.cjs` | Playwright config (uses `buildDemoConfig` from `@kestrel/flight`) |
| `narration.md` | Hand-written story (the *intent*); the demo also writes a generated transcript to `demo-output/narration.md` at runtime |
| `eye.toml` | kestrel-eye review config — vision model checks each screenshot matches `expected` |
| `demo-output/` | Generated, gitignored — screenshots, video, transcript |

## Vignette philosophy (vs. story demos)

* **One feature.** A vignette covers `trash` *or* `voice` *or* `privacy-modes`
  — not all three.
* **5–8 screenshots.** Enough to prove the feature works; not a tour.
* **Narrate the *why*, not the *what*.** Identifiers tell you what; the
  narration tells you what the feature exists to *do*.
* **No flaky chains.** A vignette should not depend on slow LLM calls if it
  can avoid them. Drive UI affordances directly when possible.
* **Graceful failure.** A vignette is a demo, not a test — if a beat can't
  find an element, narrate that and continue.
