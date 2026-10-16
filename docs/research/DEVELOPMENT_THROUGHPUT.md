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

| ID | Hypothesis | Status | Result |
|---|---|---|---|
| E1 | The CI runner has more vCPUs than the 2 xdist workers use | **confirmed; `-n logical` shipped in #3670** | 4 vCPUs (2 cores × 2 threads). `-n auto` counts physical cores and starts 2 workers. Unit tier: 21.7 min at 2 workers, **14.1 at 4**, 10.2 at 8. But 8 failed two near-limit tests on its second run. |
| E2 | Splitting the unit tier across 4 runners brings it under 10 minutes | **confirmed** | Split by measured time at 4 workers each: jobs take 4.2–6.5 min, so **~6.5 min wall time**. Exposed three order-dependent tests that must be fixed first. |
| E3 | A small set of tests dominates the unit tier | **confirmed** | 547 tests (2%) take ≥1 s each and are **52%** of test time; 49 tests take ≥5 s and are 18%. |
| E4 | The host runs the full unit suite far faster than CI | **confirmed** | 5 min 24 s at `-n 8`; **3 min 53 s at `-n 16`**, all 24,787 tests green. |
| E5 | The host can run two full test loads at once | **partly** | Two suites at `-n 8`: 7.3 and 7.9 min, one or two load-timeout failures each. Three suites at `-n 6`: 12.5–14.1 min, four to six failures each. |
| E6 | Parallelizing the integration tier brings it under 10 minutes | proposed (#2794) | |
| E7 | Auto-merge plus a pre-deploy integration gate removes the post-PR wait without breaking main | needs a policy decision | |
| E8 | Testing feature repositories against core `main` nightly catches cascade breakage before a release | proposed | |

### E1 — CI worker count

**Method.** On the experiment branch `experiment/ci-throughput`, the
unit-tier job was copied with different `-n` values and dispatched with
`workflow_dispatch`. A step printed `nproc`, `lscpu` and psutil's core
counts. Each comparison ran in the same workflow run as an unchanged copy
of the baseline job.

**Runner.** AMD EPYC 7763, 4 vCPUs (2 cores × 2 threads), 15 GB.
`psutil.cpu_count(logical=False)` is 2, and that is what xdist's `auto`
uses.

| Workers | Unit tier | Failures not caused by the experiment |
|---|---|---|
| 2 (`-n auto`, today) | 21 min 44 s | 1: `test_agent_cancellation` timeout |
| 4 (`-n logical`) | 14 min 08 s | 0 |
| 8 (`-n 8`) | 10 min 14 s | 0 |

**What it means.** Twice as many workers as vCPUs still helps. The suite
spends most of its time waiting (timeouts, subprocesses, sleeps, database
round trips) rather than computing.

| Workers | Summed test time | Wall time | Speedup over serial |
|---|---|---|---|
| 2 | 41.5 min | 21.7 min | 1.9× |
| 4 | 49.3 min | 14.1 min | 3.5× |
| 8 | 62.3 min | 10.2 min | 6.1× |

Each test slows a little as workers contend for the CPU, but far less than
the parallelism gained.

**The second run at 8 workers failed.** That run was the CI of #3670. It
took 15 min 40 s on a slower runner, and two tests that sit near their
limits failed:

- `test_paging_survives_tombstones_filling_whole_pages` crossed the 60 s
  per-test timeout. It takes 27.7 s at 2 workers and 41 s at 4.
- The 50 ms graph-index benchmark measured 55 ms.

Eight workers on 4 vCPUs oversubscribe the CPU, and the slowest tests have
no margin for that. One clean run and one failing run is too flaky for a
merge gate. So #3670 shipped `-n logical` (4 workers, one per vCPU); its own CI ran the unit tier in 15 min 01 s, all green. Going past the vCPU count waits until the slow tests get faster (E3,
#3276).

### E2 — Sharding the unit tier

**Round 1** split the test files into four groups of equal total file
size, each run at `-n auto` (2 workers):

| Shard | Tests | Test time | Job time |
|---|---|---|---|
| 0 | 6,317 | 5 min 24 s | 6 min |
| 1 | 6,372 | 6 min 29 s | 7 min |
| 2 | 6,244 | 8 min 55 s | 10 min |
| 3 | 6,186 | 5 min 10 s | 5 min |

- **Balance.** File size is a poor stand-in for run time; the slowest shard
  set the wall time.
- **Order-dependent tests.** Splitting exposed three tests that depend on
  test order. `test_server_lifespan_wires_and_closes_host_features[False/True]`
  and `test_server_retains_control_context_after_optional_mount_failure` (in
  `tests/unit/test_host_features.py`) need `KESTREL_API_KEY` in the
  environment but never set it. In a full run they pass only because
  earlier tests assign `os.environ["KESTREL_API_KEY"]` directly and never
  restore it: `test_bridge_router_mounted.py`,
  `test_bridge_stream_error_boundary.py`,
  `test_feature_route_lifecycle_gate.py` and `test_auth_decision_table.py`.
  Under xdist this is already a latent flake. It must be fixed before any
  sharding.

**Round 2** split by measured per-file time (from the E1 JUnit output) and
ran each shard at `-n logical` (4 workers):

| Shard | Tests | Test time | Job time |
|---|---|---|---|
| 0 | 5,208 | 3 min 57 s | 4.8 min |
| 1 | 6,749 | 5 min 10 s | 6.5 min |
| 2 | 5,928 | 3 min 06 s | 4.2 min |
| 3 | 7,235 | 5 min 08 s | 6.4 min |

- **Wall time:** about 6.5 minutes, against 21.7 for today's single job.
  It uses about 22 runner-minutes, which cost nothing on this public
  repository.
- **Failures:** the same three `test_host_features` tests failed again, a
  second independent reproduction. Nothing else failed.
- **Balance:** still uneven, because the per-file times came from one run.
- **Next combination to try:** four shards at `-n 8` each should land near
  4 minutes. It carries the same oversubscription risk that E1's second
  8-worker run hit.

### E3 — Where the unit tier's time goes

From the CI JUnit output of the 4-worker run (25,195 tests, 49.3 min summed
test time):

- **Slow tests:** 547 tests (2%) take at least 1 s each, 25.4 min in total
  (52%). 49 tests take at least 5 s each, 8.8 min in total (18%). The
  slowest takes 49 s.
- **Concentration by file:** the top 10 files hold 35% of test time, the
  top 25 hold 51%, the top 100 hold 79%.
- **Heaviest files:**
  - `test_session_training_lifecycle.py` (3.4 min);
  - `test_generation_gate.py` (2.9);
  - `test_privacy_mode_durability.py` (2.5);
  - `test_pytest_coverage_shutdown.py` (1.6 min for 2 tests);
  - `test_memory_attestation_window.py` (1.5).

These slow tests are also where most of the load-sensitive failures in E5
come from. Making them coordinate on events rather than durations (#3276)
would speed the suite up and make it steadier.

### E4 — The full unit suite on the host

Each run was a full `tests/unit` run in its own worktree and virtual
environment, with the production host and agents running. PostgreSQL cases
were skipped: 439 skipped locally against 67 in CI.

| Workers | Wall time | Failures | Peak load average |
|---|---|---|---|
| 8 | 5 min 24 s | 0 (excluding one docs-generation miss fixed on `main`) | 17 |
| 16 | 3 min 53 s | 0 | 24 |

Memory was at least 95% free throughout.

### E5 — Concurrent lanes on the host

Each lane was a separate worktree running the full unit suite at once.

| Lanes × workers | Wall time per lane | Failures per lane | Peak load average |
|---|---|---|---|
| 1 × 8 | 5 min 24 s | 0 | 17 |
| 2 × 8 | 7 min 16 s, 7 min 51 s | 2, 1 | 22 |
| 3 × 6 | 14 min 07 s, 12 min 29 s, 12 min 43 s | 4–6 each | 28 |

- **Which tests failed.** Every failure was a timing test: subprocess-pytest
  timeouts, watchdog and stall bounds, drain and shutdown tails. None was a
  logic failure.
- **Where the limit is.** Memory stayed at least 97% free throughout. The
  limit is the suite's sensitivity to load: past about 16 workers in total,
  timing tests start failing.
- **Implications.** Two full suites at once are tolerable. Three are not:
  aggregate throughput barely beats running them one after another, and the
  flake rate makes results unreadable.

So more Talon lanes fit on the host only if each lane runs the tests for its
own ticket. The full suite belongs either in CI or in a single queued host
job at `-n 16`, which takes about 4 minutes.

## 5. Platform alternatives

Researched 2026-10-16 against current vendor documentation. The full
source list is at the end.

**Verdict: stay on GitHub.** What slows us is parallelism inside CI, and
that can be fixed on GitHub at no cost. No other forge fixes something
GitHub cannot, and every move would mean rewriting:
- Talon's GitHub backend;
- the GitHub feature package and GitHub App;
- PyPI trusted publishing;
- release signing.

| Option | What it would fix | Cost | Migration | Main risk |
|---|---|---|---|---|
| More xdist workers and sharding on GitHub's standard runners | Unit tier to ~10 min with no sharding (E1), less with it | $0 on public repositories (Free plan: 20 concurrent jobs; Team: 60) | Low | Bursts of agent PRs queueing for runners |
| Self-hosted runner on the host | Unit tier ~4 min (E4) | $0 today | Medium | **Fork PRs execute code on the host** if used for the public repository; the host becomes a single point of failure |
| Buildkite with an agent on the host, GitHub as the forge | Same speed as above; fork builds are off by default | Free for up to 5 users and 10 concurrent self-hosted jobs | Medium | Reports commit statuses, not check runs, so Talon's CI watcher would need to read both |
| Third-party runners (Ubicloud, Namespace, Blacksmith, WarpBuild, Depot, RunsOn) | Faster hosted runners with no host risk | About $0.001–0.003 per vCPU-minute, not free for public repositories | Low | Vendor churn; a cost where today there is none |
| GitHub merge queue, Graphite, Mergify, Aviator, Trunk | A main branch that is never broken | $0 (GitHub's own queue on public repositories) to ~$40 per user per month | Low | Adds a CI pass per merge; buys safety, not throughput |
| GitLab | Merge trains | Premium at $29 per user per month for merge trains | High | Rewrites CI, Talon backend, publishing |
| Forgejo / Codeberg | Nothing we need | Hosting time | High | PyPI trusted publishing does not support it; no merge queue; Codeberg is for free-software projects |

### Facts that bear on the choice

- **Self-hosted runner fee.** GitHub's proposed per-minute fee for
  self-hosted runners would apply only to private repositories. It was
  postponed and had no new date as of mid-2026. Self-hosted runners on
  public repositories remain free.
- **Security on public repositories.** GitHub's own guidance is that
  self-hosted runners should almost never serve public repositories. A fork
  PR can edit the workflow file, so an `if:` condition is not a security
  boundary. The real mitigations are:
  - require approval to run workflows from outside contributors;
  - ephemeral one-job runners in throwaway VMs with no secrets and no host
    mounts;
  - a dedicated runner group.

  Tart (now `openai/tart`) and Apple's `container` 1.0 can both run Linux
  VMs on the host.
- **New on GitHub in October 2026.**
  - Stacked pull requests are generally available, and a stack enters the
    merge queue as one unit.
  - An asynchronous merge API is generally available.
  - Both fit agent workflows.
- **What Actions costs us now.** From the billing API: $39 in August, $34
  in September, $2 so far in October. The macOS cost from May (#1463) is
  gone, so cost is not a reason to move.
- **Reliability.** Actions is GitHub's most incident-prone service. The
  2026-08-06 outage lasted 10 h 42 min, and some trigger events were lost
  for good. A gate we run ourselves would be a hedge against this.

## 6. Current recommendations

Updated with the results above.

1. **Run the unit tier at `-n logical`** (E1, #3670). It is free, needs one
   line changed, and cuts about 7 minutes from every PR. `-n 8` would cut
   about 11 minutes, but it waits until the slowest tests get faster.
2. **Fix the order-dependent `test_host_features` tests** (E2, #3671) and
   the direct `os.environ` writes that hide them. They are a latent flake
   today and block sharding.
3. **Shard both tiers once that is fixed** (#3672). Four unit shards split
   by measured time came to about 6.5 minutes (E2). Parallelize integration
   too (#2794). Keep the required check names stable through aggregator
   jobs.
4. **Widen the lane to two Talon workers, not three** (E5). Each lane runs
   only its ticket's tests locally, and the full suite runs in CI.
   Reconsider three lanes after the slow, load-sensitive tests (E3, #3276)
   are reduced.
5. **Consider the host as a CI gate, starting with private repositories.**
   A self-hosted runner, or Buildkite with fork builds off, would bring the
   unit tier to about 4 minutes. For the public core repository this needs
   ephemeral VMs and outside-contributor approval, with GitHub-hosted
   sharding kept as the fallback.
6. **Make merging fire-and-forget.** Enable auto-merge, or use the new
   asynchronous merge API (E7).
7. **Automate the cross-repository cascade** (E8). Do not consolidate into
   one monorepo.

## Sources

- [Anthropic Engineering — Redesigning source control](https://www.anthropic.com/engineering/redesigning-source-control)
- [TechSpot coverage](https://www.techspot.com/news/114209-anthropic-ditched-git-own-version-control-system-because.html)
- [GitHub — 2026 pricing changes for Actions](https://github.com/resources/insights/2026-pricing-changes-for-github-actions); [Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions); [runner pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing)
- [GitHub — secure use of self-hosted runners](https://docs.github.com/en/actions/reference/security/secure-use); [Actions limits](https://docs.github.com/en/actions/reference/limits)
- [GitHub changelog — stacked pull requests GA](https://github.blog/changelog/2026-10-06-stacked-pull-requests-generally-available); [async merge API GA](https://github.blog/changelog/2026-10-01-github-async-merge-api-generally-available); [merge queue GA](https://github.blog/changelog/2023-07-12-pull-request-merge-queue-is-now-generally-available)
- [GitLab merge trains](https://docs.gitlab.com/ci/pipelines/merge_trains); [GitLab pricing](https://about.gitlab.com/pricing)
- [Forgejo Actions vs GitHub Actions](https://forgejo.org/docs/next/user/actions/github-actions); [PyPI trusted publishers](https://docs.pypi.org/trusted-publishers/adding-a-publisher)
- [Buildkite pricing](https://buildkite.com/pricing); [Buildkite GitHub integration](https://buildkite.com/docs/pipelines/source-control/github)
- [Ubicloud](https://www.ubicloud.com/docs/about/pricing), [Namespace](https://namespace.so/pricing), [Blacksmith](https://www.blacksmith.sh/pricing), [WarpBuild](https://www.warpbuild.com/pricing), [Depot](https://depot.dev/pricing), [RunsOn](https://runs-on.com/pricing) pricing pages
- [Actions outage, 2026-08-06](https://blog.incidenthub.cloud/github-actions-pages-outage-aug-6-2026)
- Issues: #2794 (integration parallelism), #2795 (runner sizing), #2810 and #3330 (unit-tier budget), #3276 (load-sensitive test tail), #3664 (idle queue)
