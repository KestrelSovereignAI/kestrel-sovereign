# <Feature Name> Vignette — Narration

> Replace this whole file. Describe the feature's *purpose* and walk a
> presenter through the beats. The Playwright `demo.cjs` writes a generated
> transcript to `demo-output/narration.md` at runtime — that file is the
> tool's view; this file is the *human's* view.

## Why this feature exists

One paragraph. What was wrong before; what changed; what the user can now do
that they could not before.

## The beats

### Beat 1 — Starting view
What the user sees on entry. Why the panel matters.

### Beat 2 — Trigger
The action that exercises the feature. The intent, not the click target.

### Beat 3 — Result
What changed. Why that matters.

### Beat 4 — Bookend
Closing shot. The surface is clean; the user can move on.

## Running the vignette

```bash
demos/run.sh <feature-name>
```

Outputs land in `demos/<feature-name>/demo-output/`.
