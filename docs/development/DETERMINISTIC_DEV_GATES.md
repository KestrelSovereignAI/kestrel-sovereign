---
type: Design
title: Deterministic development gates
description: 'Closes the recurring class of agent-authored defect where a fix touches
  one of N sites sharing an invariant. Establishes that detection is grep and AST,
  judgement is the model, and the ORDER is code in both cases.'
resource: /docs/development/DETERMINISTIC_DEV_GATES.md
tags:
- docs
- development
- design
- talon
- workflows
timestamp: '2026-08-25T00:00:00Z'
status: proposed
owner: architecture
---

# Deterministic development gates

> Detection is `grep` and AST. Judgement is the model. **The order is code
> either way.**

## 1. The defect this exists to stop

One question does not get asked: **what else shares the invariant I am
changing?**

It produces three symptoms that read as separate problems and are not:

- *"I fixed the instance, not the class."*
- *"Each review round found what I broke in the last one."*
- *"I changed a shared function without asking who else called it."*

### Measured, over one ticket's review rounds

| Round | Invariant | Sites | Changed | Caught by |
|---|---|---|---|---|
| #3107 r2–r3 | literal `action="tool_execution"` | 5 | 1 | review, twice |
| #3107 r4–r5 | dispatch-envelope labelling | 2 hook events + 1 registration | 1 each | review, twice |
| #3112 | `_make_inline_tool_executor` — **two independent copies** | 2 | 1 | review, after a fix |
| #3107 r7 | refusal decision set | 15 distinct values live | 5 hardcoded | review |
| #3107 r8 | `fold_searchable` callers | 2 | function changed for 1 | review |

Every row is `grep -c` away. None required judgement to **detect** — only to
resolve. All were found by review, which is the most expensive place to find
them and the furthest from where they were made.

### Why the habit does not hold

There is no forcing function. A failing test pushes on the instance; nothing
pushes on the class. Fixing the instance produces green, and **green terminates
the search**.

This is not a knowledge problem. During the ticket above, the rule *"grep who
else calls this before changing it"* was written into durable memory, recited to
another agent, and then broken four more times by the agent that wrote it. The
knowledge was never missing. The **sequence** was.

## 2. The load-bearing claim

The model's failure is **ordering, not capability**.

That matters because it rules out the obvious fix. "Be more careful", "add it to
the prompt", "remember to check" — all target capability, and capability was
never the constraint. The constraint is that skipping the question is free.

So:

- Where detection is mechanical, **make it mechanical.** A model is a poor
  detector and today it is both detector and decider.
- Where judgement is genuinely needed, **keep the model — but make its turn a
  stage in a sequence it does not control.**

A stage that is "a model reads this and judges" is fine. A model deciding
*whether and when* that stage happens is the defect.

### Order matters even when every step is present

Two sequences both "include mutation testing":

```
BAD   fix → write test matching the fix → mutate → green
GOOD  derive co-change set → fix → mutate each site → read survivors
```

The first confirms the patch. The second finds the door nobody looked at. An
ordering encoded in a workflow cannot drift; one carried in working memory
drifts at every turn boundary.

## 3. Where it runs

**Talon-local, inside the iteration.** `.kestreltalon/quality.yaml` →
`kestreltalon/processor.py:2138` (discover) → `:2310` (run), per iteration,
before the iteration is declared complete. That is before the fix is finished
and before anything is pushed, which is when the class is still cheap to see.

**CI is the backstop only.** The changed-files ruff F811 gate proved the shape,
and it caught its first real defect *after* a push and a red check — later and
more expensive than the same question asked in the loop. CI keeps what must be
enforced against a branch nobody ran locally.

## 4. The three gates

### 4.1 Co-change surfacing

For every symbol and string literal the diff modifies, report the occurrences it
does **not** touch.

```
fold_searchable          modified 1 of 2 call sites   (permissions.py:1346 unchanged)
action="tool_execution"  modified 1 of 5 occurrences  (approval_queue.py:310,327,382,415)
```

Advisory, not blocking — most of the time some sites are legitimately different
and the check cannot know which. The point is that the remainder is on screen
**before** the commit rather than in review two hours later.

**Implemented** as `scripts/check_co_change.py` (#3124). It reads modified
function bodies and both added *and removed* string literals — a rename at one
of N sites leaves the siblings under the old name, which never appears on the
new side at all. Unchanged sites are ranked by proximity (same file, then same
directory) because `git grep` returns path order, which buried the four sibling
sites *below* thirty unrelated hits when the round-2 case was reproduced.

Validated against that case: changing `action="tool_execution"` at one of the
five `approval_queue.py` sites reports the other four first, followed by
`hooks.py:107,118,137` — the r2/r3 and r4/r5 findings, which together cost six
review rounds, in the first eight lines of output.

It is **not yet a `quality.yaml` check**: it exits 0 by design, and
`failure_summary()` builds from failed checks only, so its report would be
captured and discarded. Only its unit tests run per iteration today. The
surfacing invocation lands as a check when kestrel-talon#230 adds an advisory
state; until then it is a manual `uv run python scripts/check_co_change.py`.

### 4.2 Caller coverage on signature change

A changed function signature lists its callers; each must appear in the diff or
be acknowledged. Would have caught the `request_approval(audit_action=...)`
widening that broke a test double, and the two-executors case directly.

### 4.3 Mutation-on-diff

For each changed hunk, apply a mechanical mutant, run only the tests the diff
touches, report survivors.

This is the one that actually worked — it caught **nine** test-side failures in
one ticket, every one invisible to reading and to review. It caught them
manually. Automating it removes the only step that reliably worked and depended
on somebody remembering to do it.

Note the second-order finding it produced: a surviving mutant twice revealed
that the *reasoning about which fix did the work* was wrong, not merely that a
test was missing.

## 5. The sequence

```
1. compute the co-change set for this diff          DETERMINISTIC
2. classify each unchanged site:                    MODEL — forced, only after 1
   same invariant, or legitimately different?
3. mutation-on-diff across the changed sites        DETERMINISTIC
4. read the survivors and decide                    MODEL — forced, only after 3
```

Steps 2 and 4 are exactly the judgement that keeps being skipped. They stop
being skippable not because the model improved, but because the sequence is code
and the model is a stage inside it.

## 6. What each repo owns

| Repo | Owns | Status |
|---|---|---|
| **workflows** | the sequence — durable, ordered, resumable stages | load-bearing; do first |
| **kestrel-talon** | an advisory check state | kestrel-talon#230 |
| **kestrel-sovereign** | the three check scripts + `quality.yaml` entries | §4.1 shipped (#3124); §4.2 and §4.3 open |

### The harness already exists

`kestreltalon/quality.py` has `QualityCheck` / `discover_checks()` /
`run_checks()`, config-driven, running per iteration. **No new engine mechanism
is needed** for the checks themselves.

The gap is what we put in it: `kestrel-sovereign/.kestreltalon/quality.yaml` has
ten checks and **every one is documentation**. A deterministic-check harness that
runs every iteration, used only to keep docs honest.

### The one genuine engine change

`QualityResult.failure_summary()` (quality.py:339–351) builds its sections from
**failed** checks only. A passing check's stdout is captured and discarded. So a
*surfacing* gate is invisible to the agent that needs to read it.

The engine has silent-pass and hard-fail. This class needs a third: **advisory**
— output surfaced, iteration not blocked. Without it the choice is failing on
every legitimate co-occurrence, which trains suppression, or nothing at all. A
gate people routinely suppress is worse than no gate: it teaches that the whole
category is noise.

## 7. What is honestly NOT mechanisable

Stated so this is not oversold. Each of these still needs a model, and each is a
**stage**, not a hope:

- **A comment asserting a property the code does not hold.** Hit three times in
  one day; the third was load-bearing on a trust boundary rather than on
  documentation.
- **A rule keyed on a proxy.** Recognising that a hook *event* stands in for a
  dispatch *name*, or that a session id stands in for turn ownership, needs
  semantics. The tell is repetition: three rounds finding the same defect one
  door over means the rule is keyed on a proxy.
- **Whether an unchanged co-occurrence is legitimately different.** This is why
  §4.1 surfaces rather than fails.

## 8. Provenance

Derived from measured failures across `#3105`, `#3107` (eight review rounds),
`#3112` and `#3114` on 2026-08-24/25, in both human-directed and agent-authored
work. The pattern was raised by the Sovereign after reading a day of commit
messages in which the same admission recurred in escalating forms.

Related: `#3124`, `kestrel-talon#230`, `#3122` (full-diff review has no tool),
`#3114` (enumeration that grows), `#3118` (positional command parsing).
