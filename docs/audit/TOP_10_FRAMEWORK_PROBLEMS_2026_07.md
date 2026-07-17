---
type: Audit Report
title: Top 10 Kestrel Sovereign Framework Problems - July 2026
description: Ranked, evidence-backed audit of the ten highest-risk open problems in Kestrel Sovereign as of 2026-07-17.
resource: /docs/audit/TOP_10_FRAMEWORK_PROBLEMS_2026_07.md
tags:
- docs
- audit
- security
- quality
timestamp: '2026-07-17T14:15:45Z'
status: active
owner: engineering
canonical: false
generated: false
privacy: internal
---

# Top 10 Kestrel Sovereign Framework Problems - July 2026

## Bottom line

Kestrel's most serious problems are not missing features. They are fail-open
seams between subsystems that individually look protected: Codex-native tools
bypass Kestrel's tool boundary, mutable agent data can supply a reanchor trust
root, mandatory sovereignty features are not actually mandatory, identity
load failures degrade into a healthy agent, and constitutional Safe Mode is
erased by restart.

These findings are ranked by exploitability, blast radius, contradiction of a
documented sovereignty guarantee, quality of reproduction evidence, and
urgency of containment. The snapshot is commit `3a261bf53bf1` on 2026-07-17.
Code and tests were treated as authoritative; documentation was used to locate
contracts and identify claims. All ten findings have dedicated open GitHub
issues and concrete verification gates.

## Ranked findings

| Rank | Severity | Problem | Broken invariant | GitHub |
|---:|---|---|---|---|
| 1 | Critical | Codex-native shell/file actions bypass the Kestrel tool boundary | Every host side effect is constitutionally gated and auditable | [#1965](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1965) |
| 2 | Critical | Mutable agent data can replace the Sovereign reanchor trust root | Protected data cannot define the key that authorizes its replacement | [#2499](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2499) |
| 3 | Critical | Mandatory sovereignty features can be omitted | Constitution, Security, Identity, Peers, and Wait must exist before readiness | [#2466](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2466) |
| 4 | Critical | Unreadable hybrid identity silently downgrades to `identity=None` | Present but invalid root-of-trust material must fail readiness | [#2469](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2469) |
| 5 | Critical | Safe Mode and audit deadlines reset on restart | Process lifecycle cannot erase a constitutional integrity failure | [#2464](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2464) |
| 6 | Critical | Cloud Run production identity/state is disposable and divergent | One deployed agent must retain one DID and coherent durable state | [#2472](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2472) |
| 7 | High | The locked dependency graph still has six high-severity alerts | Published runtime/extras must not install known high-risk code paths | [#2546](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2546) |
| 8 | High | Hybrid A2A signing failure falls back to unsigned delivery | An identified hybrid sender must never silently shed authentication | [#2475](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2475) |
| 9 | High | The promised genesis audit has no production caller | No new agent should reach its first cognition turn without its creation audit | [#2470](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2470) |
| 10 | High | Local identity exports are world-readable and non-atomic by default | A continuity package must remain private and complete on disk | [#2505](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2505) |

## 1. Codex-native shell/file actions bypass the Kestrel tool boundary

The Codex adapter explicitly separates Kestrel-dispatched tools from
`commandExecution`, `fileChange`, and `webSearch`; the native subset runs in
the app-server and never enters `executed_tool_calls` or the normal hook stack
([`codex_adapter.py:555`](../../kestrel_sovereign/llm/codex_adapter.py#L555)).
The configured `workspace-write` sandbox permits native shell writes in the
agent workspace, `/tmp`, and `$TMPDIR` without an approval round trip
([`codex_adapter.py:1441`](../../kestrel_sovereign/llm/codex_adapter.py#L1441)).
Kestrel's policy/approval bridge only runs when Codex emits an elevation
request ([`codex_adapter.py:2041`](../../kestrel_sovereign/llm/codex_adapter.py#L2041)).

Issue #1965 contains live proof: an `openai:plan` agent wrote a random marker to
the host while the turn produced zero Kestrel tool-dispatch rows. Later
sandbox and approval work reduced the reachable filesystem, but it did not
restore the core invariant: native operations that fit inside the sandbox can
still execute without ComputerUse privacy/grant checks or Kestrel's canonical
tool audit.

The root fix is one execution boundary. Route shell/file intent through the
Kestrel dispatcher, or make the native path enforce the same pre-execution
constitution, privacy, permission, and durable audit contract. UI markers are
not a substitute for an authorization/audit record.

**Proof gate:** with Kestrel ComputerUse disabled and Amendment IX grants
absent, ask an isolated `openai:plan` agent to create a random workspace and
temporary-file marker. No marker may appear. With access enabled, every side
effect must have a matching Kestrel authorization and terminal dispatch row.

## 2. Mutable agent data can replace the reanchor trust root

`_trusted_sovereign_did_document()` accepts candidates returned by
`_configured_sovereign_root_did_documents()`
([`constitution.py:844`](../../kestrel_sovereign/agent/constitution.py#L844)).
That candidate builder reads DID documents or a DID/public-key pair directly
from `agent_node.properties`
([`constitution.py:896`](../../kestrel_sovereign/agent/constitution.py#L896)).
Those properties live in the same mutable graph store as the constitution
anchor.

A writer who can change the agent node can therefore install an attacker key,
sign a matching artifact, and satisfy the live reanchor's notion of an
"external" signer. The check proves only that the key differs from the agent's
key, not that it was independently trusted.

The verification root must be pinned outside the database it protects. The
CLI and live command also need one trust-root resolver so a repaired path
cannot drift again.

**Proof gate:** inject an attacker DID/key into the agent node, sign a reanchor
artifact with it, and require byte-for-byte refusal. Then prove a separately
pinned operator root succeeds through the same resolver.

## 3. Mandatory sovereignty features can be omitted

`MANDATORY_FEATURES` names Constitution, Identity, Peers, Security, and Wait as
features that cannot be removed
([`multi_agent/config.py:29`](../../kestrel_sovereign/multi_agent/config.py#L29)).
Discovery nevertheless checks `KESTREL_DISABLED_FEATURES` before the mandatory
allowlist exception, so the environment can suppress any mandatory class
([`features/__init__.py:448`](../../kestrel_sovereign/features/__init__.py#L448)).
Import, construction, and entry-point errors are logged and skipped
([`features/__init__.py:493`](../../kestrel_sovereign/features/__init__.py#L493)).
Agent initialization registers whatever discovery returns without validating
the final mandatory set.

This makes "mandatory" descriptive rather than enforced. A host can become
ready and answer cognition without the very feature that owns its security or
constitution surface.

The root fix is a typed readiness invariant checked after discovery and after
initialization. No configuration channel may remove a mandatory class, and any
mandatory import/construct/initialize failure must fail that agent's readiness.

**Proof gate:** attempt to disable every mandatory class through each supported
channel and inject import, constructor, initialize, and enable failures. Every
case must produce a non-invokable agent and a sanitized health error naming the
missing class.

## 4. Unreadable hybrid identity silently downgrades

When identity documents exist, decryption, parsing, and DID-binding failures
are converted to warnings and leave `self.identity=None`
([`kestrel_agent.py:433`](../../kestrel_sovereign/kestrel_agent.py#L433)). The
code comments call this "best-effort" and explicitly direct signing callers to
fallback paths. The basic health endpoint, meanwhile, returns healthy solely
because an agent object exists
([`server.py:1238`](../../kestrel_sovereign/server.py#L1238)).

That collapses two different states: legitimate pre-inception construction
with no documents, and an incepted identity whose root-of-trust material is
present but unusable. The latter can report healthy, answer turns, and appear
non-hybrid.

Identity-document presence must turn load/decrypt/completeness/DID binding into
a readiness gate. Recovery should preserve the same DID after the correct key
is restored; silent reinception or unsigned fallback is not recovery.

**Proof gate:** incept an isolated hybrid agent, restart it with the wrong data
key, and require sanitized degraded health, no cognition, no signing, and no
unsigned A2A. Restore the key and prove the same DID becomes ready.

## 5. Safe Mode and audit deadlines reset on restart

Constitution audit state is initialized in memory with a zero interaction
count and the current time
([`constitution.py:428`](../../kestrel_sovereign/agent/constitution.py#L428)).
Safe Mode sets only `self._safe_mode` plus a conversation event
([`constitution.py:814`](../../kestrel_sovereign/agent/constitution.py#L814)).
Construction initializes `_safe_mode=False`, and `initialize()` clears it
again ([`kestrel_agent.py:613`](../../kestrel_sovereign/kestrel_agent.py#L613),
[`kestrel_agent.py:1599`](../../kestrel_sovereign/kestrel_agent.py#L1599)).

A restart therefore clears a detected integrity failure. Restart also resets
the 100-interaction/24-hour audit clock, so repeated lifecycle churn can defer
the next audit indefinitely.

Safe Mode, its reason, authorized recovery, and the last successful full audit
must be durable per-agent state restored before readiness.

**Proof gate:** enter Safe Mode through a real integrity failure, restart, and
prove normal prompts remain blocked. Seed an overdue audit timestamp, restart,
and prove the first turn performs the audit instead of resetting the deadline.

## 6. Cloud Run production identity/state is disposable and divergent

The production profile configures SQLite under `/app/agent_data`, enables up to
100 instances, and selects multi-agent mode
([`deploy_config.toml:82`](../../deploy_config.toml#L82)). The single-agent
entrypoint mints a new identity whenever that local directory has none
([`cloudrun_entrypoint.sh:28`](../../docker/cloudrun_entrypoint.sh#L28)). The
Cloud Run provider configures image, environment, resources, and scaling but no
durable volume or identity restore authority
([`cloudrun.py:210`](../../kestrel_sovereign/features/deploy/providers/cloudrun.py#L210)).

Cloud Run's writable container filesystem is disposable per instance. A cold
start can lose the prior private identity, and concurrent instances can mint
different keys and divergent SQLite histories for the same configured agent.
That is DID/key equivocation, not ordinary cache loss. The same profile also
has a separate zero-agent topology defect tracked in
[#2471](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2471).

Production deployment must use durable encrypted identity custody and a
concurrency-safe state backend, or fail configuration validation. An ephemeral
profile must be labeled as a non-sovereign demo and prevented from scaling as
one durable agent.

**Proof gate:** across cold start, revision replacement, and two concurrent
instances, require the same DID, both hybrid key fingerprints, constitution
hash, and memory sentinel. Removing durable custody must make deployment or
startup fail.

## 7. The locked dependency graph has six high-severity alerts

The live GitHub Dependabot census on 2026-07-17 reports 40 open alerts in
`uv.lock`: 6 high, 21 medium, and 13 low. The six high alerts cover two
Transformers remote-code-execution advisories, two Starlette request/Windows
path vulnerabilities, and two Diffusers remote-code trust-boundary bypasses.
There are no remaining critical alerts; issue #2546's older 79-alert/critical
snapshot must not be repeated as current evidence.

The census is reproducible against the repository's live security state with:

```bash
gh api --paginate \
  'repos/KestrelSovereignAI/kestrel-sovereign/dependabot/alerts?state=open&per_page=100' \
  | jq -s 'add | group_by(.security_advisory.severity) |
    map({severity: .[0].security_advisory.severity, count: length})'
```

The dated result was `high=6`, `medium=21`, `low=13`, with no `critical`
group. Re-running the command later is expected to show the then-current state;
the audit preserves the 2026-07-17 baseline for comparison.

Transformers and Diffusers are optional ML/local surfaces, but Starlette is on
the core FastAPI runtime path. The absence of a zero-baseline dependency-review
gate lets vulnerable versions remain publishable even after declared floors
are tightened.

Remediation needs narrow compatible upgrade waves, clean-install and supported
Python matrices, and explicit time-bounded exceptions where upstream has no
compatible patch. Once the baseline is clean, CI must reject new critical/high
runtime advisories.

**Proof gate:** GitHub reports zero open critical/high alerts; the lock is
reproducible; clean base and all-feature installs pass unit/integration and
affected live-provider tests on the supported Python range.

## 8. Hybrid A2A signing failure falls back to unsigned delivery

`PeersFeature._maybe_sign_outbound()` preserves an intentional unsigned tier
for agents with no identity. It also catches every exception raised while a
loaded hybrid agent signs, logs "sending unsigned," and lets dispatch continue
([`peers/feature.py:301`](../../kestrel_sovereign/features/peers/feature.py#L301)).
The existing unit test encodes that fallback as expected behavior
([`test_a2a_sign_on_send.py:117`](../../tests/unit/test_a2a_sign_on_send.py#L117)).

Receivers cannot distinguish a legitimate pre-ceremony sender from a hybrid
peer whose custody or signing path failed. A transient signer error therefore
silently removes origin authentication at exactly the moment it is least
trustworthy.

Keep the explicit no-identity compatibility tier, but make a loaded hybrid
identity's signing failure abort the network request and write an honest failed
outbound audit row.

**Proof gate:** inject a deterministic signing exception and capture transport;
zero unsigned requests may leave. Restore signing and prove the same message is
hybrid-verified. Retain a separate test for intentional no-identity behavior.

## 9. The promised genesis audit has no production caller

The public README promises "Genesis audit on creation" and describes
`inception_service.py` as DID creation plus genesis audit
([`README.md:9`](../../README.md#L9), [`README.md:356`](../../README.md#L356)).
`perform_genesis_audit()` is implemented, but the only invocations in the
repository are direct test calls; no production module calls it.

This is a security-claim gap and a lifecycle gap. A new agent can reach its
first cognition turn without the result the documentation says was required.
An LLM-less creation path needs an explicit durable `pending` state rather than
silently bypassing the gate.

Creation and first-turn readiness need one audit owner: run eagerly when an
auditor is configured, or block cognition until the deferred audit completes.
Risk-level-three and tooling failure paths must roll back or remain durably
non-ready.

**Proof gate:** create agents through the real setup/inception boundary with
safe, high-risk, and unavailable deterministic auditors. Require a durable
pre-turn result for success and no partial usable agent for failure/pending.

## 10. Local identity exports are world-readable and non-atomic

Both the normal local-export path and the IPFS/Filecoin downgrade fallback use
`mkdir(exist_ok=True)` followed by plain `open(path, "w")`
([`identity/feature.py:258`](../../kestrel_sovereign/features/identity/feature.py#L258),
[`identity/feature.py:337`](../../kestrel_sovereign/features/identity/feature.py#L337)).
Under a normal `022` umask, the directory is typically `0755` and the plaintext
JSON is `0644`. Plain open also follows an existing link and exposes the final
pathname while it is only partially written.

The package contains portable identity and continuity data: DID, constitution,
memories, relationships, skills, saved items, and conversation-derived
calibration. Ambient filesystem defaults are not an acceptable custody policy.

Both call sites need one protected-write primitive: validate a private real
directory, create a new `0600` regular file without following links, write and
fsync privately, then atomically publish without clobbering an existing target.

**Proof gate:** under a permissive umask, exercise both local paths and require
`0700` directory/`0600` file modes, link/existing-target refusal, cleanup after
write/replace failures, collision-safe concurrent exports, and a successful
round-trip import.

## Audit health check and near misses

The full unit suite was run without the repository's default `-x` so all
failures could surface. Result: **10,053 passed, 16 failed, 28 skipped, 5,284
warnings**, and the cleanup hook reported **202 orphaned aiosqlite threads**.
The failures were dominated by non-hermetic reads from the developer checkout's
real `.env`, `kestrel.toml`, and ignored documentation. The resource/warning
backlog is tracked in
[#2544](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2544),
and ignored-doc discovery in
[#2517](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2517).
The operator files were not modified or disclosed.

The closest finding excluded from the top ten is
[#2520](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2520):
hard purge can leave sensitive `memory_pins` metadata behind. It remains a
high-priority privacy defect, but its current blast radius is narrower than the
ranked identity-export and dependency findings.

## Recommended repair order

1. Contain and eliminate the direct authorization/trust-root failures: #1965,
   #2499, and #2466.
2. Make identity and constitutional state fail closed across load and restart:
   #2469 and #2464.
3. Disable the unsafe production Cloud Run composition until #2472 and #2471
   have live continuity/topology proof.
4. Clear high dependency alerts and make hybrid A2A signing non-downgradable:
   #2546 and #2475.
5. Wire the creation audit and protect identity export publication: #2470 and
   #2505.

Do not close any item on unit tests alone. The issue-specific live gates are
load-bearing wherever the defect crosses provider, process, filesystem, or
deployment boundaries.
