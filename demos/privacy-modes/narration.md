# Privacy Modes Vignette — Narration

## Why this feature exists

Privacy is **not a checkbox**.  Kestrel ships five modes that trade storage,
LLM routing, and shareability differently.  Each is a contract the agent
honors for the duration of a session — and each has user-visible side
effects (the wrapper drops writes, the LLM router refuses cloud calls,
exports become available).

The five modes (defined in `kestrel_sovereign/privacy.py`):

| Mode | Storage | LLM | Shareable |
|---|---|---|---|
| EPHEMERAL | nothing | local only | no |
| ISOLATED | session-only | local only | no |
| ANONYMOUS | scrubbed (PII removed) | cloud allowed | no |
| NORMAL | persistent | cloud allowed | no |
| PUBLIC | persistent | cloud allowed | yes |

The privacy badge in the chat header is the user's constant signal of which
contract is in force.

## The beats

### Beat 1 — The selector
Click the badge.  All five modes visible side by side, ordered from
strictest (EPHEMERAL) to most open (PUBLIC).

### Beats 2–6 — Walk the spectrum
Toggle each mode in turn.  The badge updates to reflect the new contract.

* **Beat 2 — EPHEMERAL** — nothing stored, local LLM only.
* **Beat 3 — ISOLATED** — temporary session storage, local LLM only.
* **Beat 4 — ANONYMOUS** — scrubbed storage, cloud LLM allowed.
* **Beat 5 — NORMAL** — standard persistent storage.
* **Beat 6 — PUBLIC** — shareable and exportable.

### Beat 7 — Bookend
Restore NORMAL — a demo agent should never be left in EPHEMERAL or PUBLIC
after a tour.

## Running the vignette

```bash
kestrel demo run privacy-modes
kestrel-eye review --config demos/privacy-modes/eye.toml
```
