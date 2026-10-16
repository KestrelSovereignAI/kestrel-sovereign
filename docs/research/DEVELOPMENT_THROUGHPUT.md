---
type: Research Note
title: 'Development Throughput: Scion, CI, and Parallel Lanes'
description: Whether Anthropic's Scion source-control system points to a transition
  for Kestrel, what actually limits how fast changes merge across core, Frinz and
  the feature repositories, and the experiments run against it.
resource: /docs/research/DEVELOPMENT_THROUGHPUT.md
tags:
- docs
- research
- research-note
- ci
- developer-experience
timestamp: '2026-10-16T00:00:00Z'
status: active
owner: engineering
canonical: false
generated: false
privacy: public
---

# Development Throughput: Scion, CI, and Parallel Lanes

**Started:** 2026-10-16. **Status:** active. The experiment log below is
updated as results come in.

**Question.** CI, merging and sequential development feel like our biggest
bottleneck. Anthropic replaced git in its monorepo with an in-house system,
Scion, for a similar-sounding reason. Should we follow, and if not, what
should we change?

**Short answer.** No transition to Scion. It is not available, and it solves
a problem we do not have. Git costs us under a minute per CI run. What limits
us is that almost all work runs in **one lane**, and each change spends
**about 25–27 minutes in CI**. Both can be fixed without leaving git or
GitHub.

---

## 1. What Scion is

Source: [Anthropic Engineering, "Redesigning source control" (2026-10-13)](https://www.anthropic.com/engineering/redesigning-source-control).

- **Why they built it.** Merges into Anthropic's monorepo went from about
  2,500 a day in May 2026 to more than 30,000 a day in September. The main
  branch's history is almost 100 GB, and CI performs on the order of 25
  million checkouts a day. Hundreds of git mirrors stopped keeping up.
- **What it changes.** The server does what a git client does locally:
  commits, merges, rebases, history and blame. A *changeset* stores the files
  changed relative to its first parent, with content-addressed pointers into
  blob storage, not a pointer to a whole tree. Clients keep a bounded cache
  and use sparse or virtualized checkouts. A node daemon, a regional cache and
  the authoritative server each hit their cache more than 99% of the time.
- **Results.** CI checkouts take seconds instead of minutes, CI got back 25%
  of its CPU capacity, and the infrastructure is 200x smaller than the git
  mirrors it replaced.
- **How they merge.** "We haven't been able to effectively replicate the
  industry standard of a global repository lock on the merge queue, so
  instead, we've invested in post-merge and pre-deploy automated testing."
  Concurrent changes can conflict, "but Claude can typically fix these
  conflicts faster than humans can even identify them."
- **How they review.** Engineers review the *intent* of each change rather
  than every line. Their review tool, Antique, supports semantic approval
  rules, so a comment-only change and a sensitive logic change get different
  scrutiny.
- **Who it is for.** Anthropic calls it "by no means a suitable replacement
  for git for general use", meant only for "truly enormous monorepos". It
  will not be open-sourced. They suggest git forges adopt server-side
  operations instead.

### Why it does not apply to us

| | Anthropic | kestrel-sovereign |
|---|---|---|
| Merges per day | 30,000+ | 6.6 |
| History size | ~100 GB | 431 MB packed |
| Checkout cost in CI | minutes, before Scion | under 1 minute (`lint-and-imports`, full history, finishes in 0.8 min) |
| Repository shape | one monorepo | about 40 repositories: core, SDK, Talon, Frinz and the feature, channel and provider packages |

### Ideas that do transfer

1. **No global merge lock; test heavily after merge and before deploy.**
   Our branch protection already does not require a branch to be up to date
   before merging (`strict: false`), so merges are lock-free. But agents
   still wait for the slowest tier of CI, which is not a required check,
   before they merge.
2. **Conflicts are cheap when an agent resolves them.** Only 8% of our PRs
   needed a merge from the base branch. Running more work in parallel
   would raise that, and the cost of resolving it is low.
3. **Review depth should follow risk.** A docs-only change and a
   storage-migration change should not get the same review rounds.

## 2. Baseline measurements

Window: 2026-09-16 to 2026-10-16. Produced by
[`scripts/ci/throughput_report.py`](../../scripts/ci/throughput_report.py),
which reads GitHub through the `gh` CLI. Stage boundaries come from issue
labels (`agent-ready`, `agent-claimed`) and PR timestamps. Re-run it after
any change below to compare against this baseline.

### kestrel-sovereign (198 merged PRs)

| Stage | Median | p75 | p90 |
|---|---|---|---|
| Ticket ready → claimed (queue) | 0.0 h | 2.2 h | 14.6 h |
| Claimed → PR opened (implement and self-review) | 32 min | 44 min | 60 min |
| PR opened → merged (CI and merge) | 31 min | 66 min | 152 min |
| Gap between consecutive merges | 73 min | 131 min | 392 min |

- **Throughput:** 6.6 merged PRs a day.
- **Work in flight** (claim to merge, sampled every 10 minutes): **mean 1.5,
  p90 3, max 6**. The host's Talon `worker_concurrency` setting is 1 (the
  allowed range is 1–8). Most of what runs beyond that one lane is
  interactive sessions.
- **Lead time** from claim to merge: median 1.2 h, p90 3.7 h. A few tickets
  sit open for days (blocked, clarifying, or held for review).
- **Merge conflicts:** 8% of PRs carry a merge-from-base commit.
- **Superseded CI:** 132 of 339 `pull_request` runs between 10-04 and 10-16
  were cancelled by a newer push to the same PR.

### Core CI, last 15 successful `pull_request` runs

| Job | Median | p90 | Required check |
|---|---|---|---|
| integration-tests | 27.0 min | 29.9 min | no |
| unit-tests | 25.6 min | 30.4 min | yes |
| console-smoke | 1.5 min | 1.6 min | no |
| lint-and-imports | 0.8 min | 0.9 min | yes |
| clean-install (sync, wheel) | ~1 min | ~1.2 min | yes |
| Wait for a runner | 0.1 min | 0.1 min | n/a |

- **No queueing for runners.** The repository is public, so standard Linux
  runners cost nothing.
- **The unit tier ran 25,224 tests in 24 min 37 s on 2 xdist workers** (CI
  log line `created: 2/2 workers`).
- **It is growing fast:** about 5 minutes at the start of September (#2810),
  13 minutes on 09-23 (#3330), 25 minutes now.
- **Integration runs serially** because its tests share one PostgreSQL
  database. Only 151 of its 1,249 tests actually share state (#2794).

### Other repositories

| Repository | Merged PRs/day | PR opened → merged (median) | CI |
|---|---|---|---|
| frinz | 0.5 | 19 min | 10.3 min |
| kestrel-talon | 0.7 | 18 min | 2.5 min Linux; 11.1 min Windows |
| kestrel-feature-talon | 0.6 | 6 min | 2.4 min |
| SDK, workflows, reflection, features, telegram, eye, flight, claws, story-archive, observability, frinz-catalog | ≤ 0.1 each | varies | 1–3 min |

None of these repositories is limited by how many PRs it can merge. What
costs time across them is the **release cascade**:

1. Release core or the SDK.
2. Raise each dependent package's version floor.
3. Publish each package to PyPI.
4. Run `kestrel update` on the host.

Each step is a sequential PR in another repository. The SDK 0.39 cascade
(10-02) also showed that a package whose published metadata still caps the
old version can downgrade the SDK underneath core during sync.

## 3. What the numbers mean

Throughput is roughly *lanes ÷ time per ticket*.

- **Time per ticket.** The median ticket takes about an hour: half
  implementation and self-review, half CI and merge.
- **Faster CI alone.** Halving CI cuts that hour by about a quarter, so
  throughput rises by about a third.
- **More lanes.** Going from one lane to three, with the queue kept full,
  could roughly triple throughput.
- **Both together** compound. More lanes is the bigger lever, and faster CI
  is what keeps more lanes from making each ticket slower.

### The host constraint

Talon jobs, local test runs and the production agents all run on the same
development host:

- **Hardware:** Apple M3 Ultra, 24 performance + 8 efficiency cores, 512 GB.
- **Load:** about 8 with the production host idle.
- **The 2026-08-04 incident:** every Claude Code session died at once while
  the host was running all of these together:
  - a full unit suite at `-n 4`;
  - two model reviews;
  - a Docker PostgreSQL;
  - a probe host.

  Since then the rule has been one heavy job at a time. The host is expected
  to stay in the testing loop, so more lanes must fit on it. The experiments
  below measure how many fit.

## 4. Experiments

| ID | Hypothesis | Method | Status | Result |
|---|---|---|---|---|
| E1 | The CI runner has more vCPUs than the 2 xdist workers use | Print `nproc` and psutil's core counts on the runner; dispatch the unit tier at `-n logical` and `-n 4` | planned | |
| E2 | Splitting the unit tier across N runners brings it under 10 minutes | Matrix of N shards behind an aggregator job that keeps the required check name `unit-tests`; dispatch on an experiment branch | planned | |
| E3 | A small set of files dominates the unit tier's 25 minutes | `--durations` and per-file totals from one run | planned | |
| E4 | The host runs the full unit suite far faster than CI | Time `tests/unit` at `-n 8`, `-n 16` and `-n 24` on the host | planned | |
| E5 | The host can run two Talon-sized test loads at once without new flakes | Two concurrent unit suites at `-n 8` each; compare wall time, failure sets and memory pressure with a single run | planned | |
| E6 | Parallelizing the integration tier brings it under 10 minutes | Per-worker databases for the 151 shared tests (#2794) | proposed (ticket) | |
| E7 | Auto-merge plus a pre-deploy integration gate removes the post-PR wait without breaking main | Turn on repository auto-merge; agents arm `gh pr merge --auto --match-head-commit <sha>`; run integration as a gate on the deploy SHA | needs a policy decision | |
| E8 | Testing feature repositories against core `main` nightly catches cascade breakage before a release | Scheduled matrix workflow that installs core `main` with each feature package's `main` | proposed | |

Each experiment will be written up below with its command, raw numbers and
conclusion as it completes.

## 5. Current recommendations

These are provisional until the experiments report.

1. **Widen the lane.** Raise Talon's `worker_concurrency` to 2, then 3,
   within whatever E4 and E5 show the host can carry. Narrow each run's local
   quality check to the tests for that ticket, and leave the full suite to CI.
   Talon already gives each worktree its own database and warns about
   overlapping files between parallel branches.
2. **Shard CI.** Run E1 and E2 for the unit tier, and #2794 for integration.
   Keep the required check names stable through aggregator jobs, because a
   renamed required check blocks every PR until branch protection is edited.
3. **Make merging fire-and-forget.** Enable auto-merge so an agent arms a PR
   once instead of polling CI (E7).
4. **Automate the cross-repository cascade.** Add nightly compatibility CI
   (E8), and open the version-floor bump PRs automatically. Do not
   consolidate into one monorepo:
   - the open-source split (#462) keeps public and private code apart;
   - core must never import feature packages;
   - the cascade's cost is latency and manual steps, which automation removes.

## Sources

- [Anthropic Engineering — Redesigning source control](https://www.anthropic.com/engineering/redesigning-source-control)
- [TechSpot coverage](https://www.techspot.com/news/114209-anthropic-ditched-git-own-version-control-system-because.html)
- Issues: #2794 (integration parallelism), #2795 (runner sizing), #2810 and #3330 (unit-tier budget), #3276 (load-sensitive test tail), #3664 (idle queue)
