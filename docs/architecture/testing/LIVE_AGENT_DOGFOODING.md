---
type: Architecture Spec
title: Live-Agent Dogfooding — driving a real test agent (Kite)
description: How to stand up and drive an isolated live test agent ("Kite") to verify claimed-done
  features on the live path — the HTTP invoke API, safe restart, PR-test-in-a-worktree, and the
  discover → live-confirm → file → Talon-fix → re-verify loop.
resource: /docs/architecture/testing/LIVE_AGENT_DOGFOODING.md
tags:
- docs
- architecture
- architecture-spec
- testing
timestamp: '2026-07-01T00:00:00Z'
status: active
owner: architecture
canonical: false
generated: false
privacy: public
---

# Live-Agent Dogfooding — driving a real test agent (a.k.a. "Kite")

> **Why this exists:** *"shipped + has a unit/smoke test" is not the same as "works on the
> live path."* Several features that were marked done were only proven broken by driving a
> real running agent adversarially — e.g. the per-response audit returned a max-risk score on
> *every* response (its JSON-mode request was silently ignored by the active adapter), a
> read-time re-encryption wiped conversation-row metadata, and a `!`-command reported a dead
> flag instead of the real feature state. None of those were caught by the existing tests.
>
> The fix is a habit: before you trust a "claimed-done" feature, **enable it on an isolated
> test agent and drive it through the HTTP invoke API, feeding it adversarial input, and
> confirm the observable behavior matches what the code claims.**

**Kite** is the conventional name for that isolated test agent. This guide is written so any
Kestrel developer (or coding agent) can stand one up and use it; the concrete port/paths for a
given machine are looked up, not hard-coded here.

---

## What a test agent is

A normal Kestrel host, running **separately from any production agent**, that you are free to
enable/disable features on, restart, and feed garbage to. Two rules:

1. **Give it its own port and its own home.** Run it on a spare port (Kite uses **`:8777`** by
   convention) with its own `KESTREL_HOME` (its own `.env` + `agent_data/<name>/kestrel_prime.db`),
   distinct from any production host on the box.
2. **Never target a production agent.** Everything below (enable features, restart, kill) must
   be aimed at the test port only. Confirm the port before any destructive action (see
   *Restarting safely*).

Run it in **multi-agent mode** so the agent is addressed by name in the URL.

## Driving it (the HTTP invoke API)

- **Endpoint:** `POST /api/agents/<name>/api/agent/invoke` with body `{"input": "<prompt or !command>"}`
- **Auth:** header `X-API-Key: <key>`; get the key from `GET /api/auth/key` → JSON field **`key`**
  (note: the field is `key`, not `api_key`).
- In multi-agent mode the single-agent path `/api/agent/invoke` returns `503 "Agent not
  initialized"` — always use the `/api/agents/<name>/...` form.
- `!`-commands go through the same endpoint (`{"input": "!audit"}`, `{"input": "!memory episodes 10 axolotl"}`).
  Command args are parsed **positionally** in the tool's signature order, with no `key=value`
  syntax — so `!memory episodes` takes `limit` first and `query` second
  (`!memory episodes <limit> <query text>`). A bare `!memory episodes axolotl` fails
  (`limit must be an integer, got 'axolotl'`); pass the limit first, e.g.
  `!memory episodes 10 axolotl` for a topic recall (the trailing words become the query).

Minimal probe helper — **fetch the key once** (repeatedly hitting `/api/auth/key` returns HTTP 429):

```python
import json, urllib.request, time

BASE = "http://localhost:8777"          # the test agent's port
NAME = "kite"                            # the test agent's name
KEY  = json.load(urllib.request.urlopen(BASE + "/api/auth/key", timeout=5))["key"]

def ask(prompt, timeout=120):
    req = urllib.request.Request(
        f"{BASE}/api/agents/{NAME}/api/agent/invoke",
        data=json.dumps({"input": prompt}).encode(),
        headers={"X-API-Key": KEY, "Content-Type": "application/json"},
    )
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r, round(time.time() - t, 1)

r, dt = ask("Reply with exactly: PONG")
print(dt, r.get("model"), r.get("provider"), repr(r.get("response"))[:200])
```

## Restarting safely (to deploy a code change)

Python/module changes require a **full host process restart** — re-initializing the agent
object (e.g. `kestrel restart <name>`) does **not** reload changed module code.

1. **Point the host at the code under test.** Run the test host from a **dedicated git worktree**
   of the branch/commit you're verifying, launched with **that worktree's own venv interpreter**
   — if you launch with the primary checkout's interpreter you silently run the primary
   checkout's code and lose isolation. Find your worktree with `git worktree list`; confirm the
   loaded code with:
   ```bash
   <worktree>/.venv/bin/python -c "import importlib.util; print(importlib.util.find_spec('kestrel_sovereign').origin)"
   ```
   (must resolve inside the worktree).
2. **Kill by PORT/PID — never by app-module name.** A test host and a production host run the
   *byte-identical* command `uvicorn kestrel_sovereign.server:app ... --port <N>` (only the port
   differs), so `pkill -f "kestrel_sovereign.server:app"` **also kills production**. Instead:
   ```bash
   PID=$(lsof -ti :8777 | head -1)
   lsof -nP -p "$PID" | grep LISTEN     # confirm it listens ONLY on the test port
   kill "$PID"
   ```
3. **Relaunch** from the worktree venv with the test `KESTREL_HOME` and port, backgrounded, e.g.:
   ```bash
   env -u VIRTUAL_ENV KESTREL_HOME=<test-home> \
     <worktree>/.venv/bin/python -m uvicorn kestrel_sovereign.server:app \
     --host 0.0.0.0 --port 8777 >> <test-log> 2>&1 &
   ```
   This runbook deliberately invokes Uvicorn directly so the test host is
   byte-identical to Kestrel's managed process launcher. For a human-operated
   direct server, use the validated
   [`python -m kestrel_sovereign.server` contract](../core/SERVER_LAUNCH_CONTRACT.md)
   instead; it rejects unsupported arguments and makes bind precedence explicit.
4. **Wait for ready** (poll `/api/auth/key`, then `ask("Reply READY")`) and re-verify the module
   origin from step 1.

**Pulling in new code + deps + features: use the supported CLI, not raw `uv`.** From the
worktree, run its **own** kestrel so the update targets the test env (not your primary install):

```bash
git -C <worktree> pull                    # get the new code
<worktree>/.venv/bin/kestrel update       # syncs dependencies AND reconciles feature packages
```

`kestrel update` is the durable path — it handles both the dependency sync and the installed
feature packages in one step. **Do not** hand-run `uv sync` for this: `uv sync` prunes the
out-of-tree editable feature packages, forcing manual reinstalls. Feature provisioning is
declared in `.kestrel-host-features.toml` and reconciled by `kestrel update` / `kestrel feature
sync` (see the feature-provisioning docs). Updating the install does **not** reload a running
`uvicorn` — restart the host process afterward (steps 2–4 above).

## Verifying a PR's tests locally (without polluting a venv)

Create a throwaway worktree of the PR branch and run its tests with `PYTHONPATH` pointed at the
worktree so its code shadows the installed copy (reuse a venv that already has all deps + the
editable feature packages):

```bash
git -C <repo> fetch origin <pr-branch>
git -C <repo> worktree add /tmp/verify-<n> origin/<pr-branch>
env -u VIRTUAL_ENV PYTHONPATH=/tmp/verify-<n> <venv>/bin/python -m pytest <tests> -q
git -C <repo> worktree remove /tmp/verify-<n> --force   # remove ONLY the exact path you created
```

---

## The dogfooding loop

The pattern that found 9 real bugs across `kestrel-sovereign` and the `kestrel-feature-*`
packages:

1. **Discover.** Read the feature's design doc/docstrings, enumerate its documented invariants,
   and scan for the four recurring bug classes:
   - *documents-X-does-Y* — the docstring claims a behavior the code doesn't implement;
   - *dead code* — a symbol/branch never reached from a live path;
   - *incomplete sweep* — a filter/guard applied in most places but missed in one (e.g. a
     `deleted_at IS NULL` read filter added to recall paths but not to an analytics read);
   - *silent fallback* — an `except`/guard that swallows a real failure and returns a benign result.
2. **Live-confirm (the bar).** A finding is only real once **reproduced on the running agent** —
   enable the feature via its `!command`, feed adversarial input, and check the observable output.
   This step matters: static reading alone once produced a bogus "dead code" claim (a mis-read of
   a symbol that didn't exist); driving the live agent corrected it before it was filed.
3. **File a scoped ticket** in the correct repo with a concrete repro and an **enforced
   `talon-verify` gate** (a `pytest -k ...` block in the issue body). State severity honestly.
4. **Fix, verify, merge, re-verify.** Land the fix (see *Parallelizing* below), read the diff and
   run the tests yourself, merge on green, redeploy the test agent, and **re-confirm the behavior
   live** at the same bar you used to find it.

For a semantic-KB release, use the dedicated HTTP checks for recall,
contradiction, quarantine, sleep, restart, and erasure in
[Semantic Knowledge Release Evidence](SEMANTIC_RELEASE_EVIDENCE.md). Attach
only the catalog-bound, content-free aggregate observation and approved
artifact reference/digest to its release report. The live invoke transcript
stays in the isolated evidence environment; no arbitrary command line or raw
response is a release record.

### Parallelizing

- **Fixes** fan out across [kestrel-talon](https://github.com/KestrelSovereignAI/kestrel-talon)
  in isolated worktrees — disjoint files run concurrently. For a sibling feature repo:
  ```bash
  kestrel-talon claim --repo KestrelSovereignAI/<repo> --issue <N> \
    --repo-dir <path-to-repo> --worktree --self-review --skip-clarification \
    --model opus --review-backend codex --review-model gpt-5.5 --verbose
  ```
- **Discovery** fans out across read-only `Explore` sub-agents (one per feature). They tend to
  **over-flag** — trace every finding to source before filing (this campaign rejected several
  confident-but-wrong findings). For a clean bill, calibrate with one spot-check of the
  highest-stakes claim per repo rather than re-auditing wholesale.
- If a fix lands **without** the regression test the ticket required, backfill it with
  `kestrel-talon iterate --repo ... --pr <N> --worktree --prompt "add the required regression tests, tests-only" ...`.

### Gotchas

- **Talon auth:** `--use-api-key` fails when `ANTHROPIC_API_KEY` is unset; omit it to use the
  default Claude Max-plan OAuth.
- **Don't pre-label** an issue you intend to hand to Talon with a "claimed" label — Talon treats
  that as already-claimed and bails; let it claim.
- **Shared bot account:** if multiple agents open PRs under one account, filter by branch/title
  and never merge or modify a PR you didn't create.

---

See also: [`TESTING_GUIDE.md`](TESTING_GUIDE.md) (overall testing conventions) and
[`TEST_EVIDENCE_GATES.md`](TEST_EVIDENCE_GATES.md) (evidence gates for claiming a feature works).
