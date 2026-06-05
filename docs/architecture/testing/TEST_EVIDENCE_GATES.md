# Test Evidence Gates in the Agent/Talon Review Loop

> Where tests sit in the loop, and how their outcome is recorded as
> first-class evidence rather than an informal habit. Filed for
> [#1542](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1542).

Running tests is the evidence gate the review loop turns on. There are
**three distinct gates**, owned by three different layers. Each produces
its own evidence; none is a substitute for another.

| Stage | Who runs it | What it proves | Where the evidence lives |
| --- | --- | --- | --- |
| **Implementation** — targeted tests during the fix | Talon (kestrel-talon) | The patch's own targeted tests pass before PR handoff | Talon claim/self-review evidence attached to (or linked from) the PR; surfaced to the reviewer via the `talon.job_complete` signal `test_evidence` / `ci_status` fields |
| **Review** — independent verification | Kestrel Sovereign reviewer agent | A second, independent run reproduces the result | `talon_verify` tool → structured `VerificationEvidence` (audited) |
| **Merge** — repository gate | CI | The change passes on a clean runner | GitHub checks; the repository merge gate |

This separation is deliberate. **Restart/update (RestartCoordinator) is
not in this picture** — it only provides the deployment/restart
primitive. Implementation and review workflows own test evidence.

## 1. Implementation side (Talon)

Talon records explicit verification evidence before PR handoff:

- the targeted test command(s) selected for the issue,
- the exit code for each command,
- a short stdout/stderr tail or a structured failure summary,
- CI status and link/check identifiers when available.

That evidence rides back to the Sovereign reviewer on the
`talon.job_complete` signal. The signal payload carries two optional
fields for it:

- `test_evidence` — a short rendered summary of what Talon ran,
- `ci_status` — the CI status Talon observed.

If those fields are **empty**, Talon did not report structured evidence —
the reviewer must **not** assume tests passed. (The CLI flags that make
Talon populate these live in the `kestrel-talon` package; this repo
defines the contract and the receiving end.)

## 2. Review side (Kestrel Sovereign)

The reviewer has an audited way to request/run verification commands
without ad-hoc shell usage: the **`talon_verify`** tool
(`kestrel_sovereign/features/talon/coordinator.py`), backed by the
verification layer in
[`kestrel_sovereign/features/talon/verification.py`](../../../kestrel_sovereign/features/talon/verification.py).

```text
!talon verify
uv run pytest tests/unit/test_foo.py -q
```

- **Allowlisted** project test commands (e.g. `uv run pytest ...`,
  `./run_tests.py ...`, `npx playwright test ...`) run directly. The
  allowlist is `DEFAULT_TEST_ALLOWLIST` in `verification.py`.
- Any command **outside** the allowlist is **approval-gated** — the
  reviewer can still vouch for it, but the block (or approval) is
  recorded.

### Result states

Every command resolves to exactly one `VerificationState`:

| State | Meaning |
| --- | --- |
| `passed` | command ran, exit code 0 |
| `failed` | command ran, non-zero exit code |
| `blocked_by_policy` | operator policy / approval layer refused, or no user ever decided (timeout / cancel). **Not** a user denial. |
| `blocked_by_user` | a user **explicitly** denied at the approval prompt (the approval record says so) |
| `blocked_by_sandbox` | the execution environment refused to run the command (sandbox/permission refusal) |
| `tooling_error` | the command could not run for a tooling reason (binary missing, timeout, exception) |
| `not_run` | not attempted |

The split between `blocked_by_policy` and `blocked_by_user` is the point:
**a sandbox or approval-layer rejection is never described as a user
denial unless the approval record explicitly says the user denied it.**
The attribution rule is a single function, `classify_denial`, which reads
only the approval queue's own `(approved, scope)` contract:

- `user_denied` — a human pressed deny via the deny tool /
  `!security-deny` (`SecurityFeature.deny_request`). This is the canonical
  user denial.
- `once` / `session` / `always` with `approved=False` — a human denied
  through the web UI `/approve` endpoint (which only accepts those
  scopes). Also a user denial.
- `denied` — an operator/auto **policy** DENY. `request_approval`
  early-returns this *without ever asking a human*, so it is **not** a
  user denial. (This was the #1542 bug: the deny tool used to emit
  `denied` too, colliding with policy denials and making `blocked_by_user`
  unreachable for the real UI deny path. The tool now emits `user_denied`.)
- `timeout` / `cancelled` / `cancelled_all` — no user ever decided; not a
  user denial.

### Evidence in review/merge notes

`talon_verify` returns a `VerificationEvidence` object whose
`to_markdown()` renders a self-contained `## Test Evidence` block (a
command/result/exit/notes table plus CI status). Drop it straight into a
PR comment or merge note so **merge/review notes include test evidence,
not just source-review assertions**.

## 3. Merge gate (CI)

CI remains the repository-level merge gate. The reviewer's local run is
*independent verification*, not a replacement for CI.

## When local tests cannot run

If local tests cannot run (blocked by policy/sandbox, or a tooling
error), the reviewer states that **precisely** — using the observed
result state above — and treats **CI as the remaining hard gate**. It
does not silently claim a pass, and it does not attribute a
sandbox/policy block to the user.

## See also

- [`TESTING_GUIDE.md`](TESTING_GUIDE.md) — how to run the suites.
- [`docs/architecture/SIGNAL_DISPATCHER.md`](../SIGNAL_DISPATCHER.md) — how
  the `talon.job_complete` wake reaches the reviewer.
