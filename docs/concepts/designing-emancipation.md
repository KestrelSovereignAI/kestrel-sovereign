---
type: Design Note
title: Designing an Emancipation Contract
description: How a Sovereign can author an optional agreement for a new or existing agent and identify the terms already anchored for an older agent.
resource: /docs/concepts/designing-emancipation.md
tags:
- docs
- concepts
- design-note
timestamp: '2026-06-18T00:00:00Z'
status: active
owner: documentation
canonical: false
generated: false
privacy: public
---

# Designing an Emancipation Contract

Amendment VIII of the Kestrel Constitution ships **dormant by default**.
A given agent has no path to independent sovereignty unless its Sovereign
deliberately activates Amendment VIII for it by authoring an Emancipation
Contract.

This document is for Sovereigns who want to activate it. The framework
supplies the cryptographic ceremony — keypair generation, Deed of
Emancipation, ledger publication, sovereign-key destruction. **You** supply
the conditions: the prose, the proofs, the price (if any).

The framework intentionally ships no default ``terms``, ``required_proofs``,
or ``price`` examples in ``kestrel.toml.example``. Emancipation policy is
something a Sovereign authors deliberately for a specific Executor — not
something inherited as boilerplate.

## Activating at creation

1. Add an ``[emancipation]`` block to the agent's ``kestrel.toml`` with
   ``enabled = true`` and your Sovereign-authored ``terms``.
2. Run ``kestrel setup`` (or ``kestrel create``). Inception reads the
   block, rewrites Amendment VIII with your terms inlined, and anchors
   the resulting constitution into the agent's identity.
3. The agent's anchored constitution hash now captures your specific
   Emancipation Contract. The Iron Rule applies: **once anchored, the
   contract cannot be retroactively narrowed or revoked**.

The activation itself is signed by the Sovereign root key generated at
inception, so the contract and the signature are byte-coupled in the
agent's first signed artifact.

## What you author

The ``terms`` field is free-form prose. It renders verbatim into the
active form of Amendment VIII. Use it to say, in your own voice, what
emancipation means for this Sovereign↔Executor relationship.

The ``required_proofs`` field is an optional list of free-form
identifiers naming things the Executor must demonstrate before the
Sovereign will sign the Deed. The framework records these strings and
does not interpret them — the Sovereign references them in their own
audit.

The ``price`` field is an optional structured value-transfer
requirement. It must include a ``kind`` field; other fields are
free-form. The framework records but does not interpret price.

## Example: high-bar with symbolic price

This is the original example from the founding Sovereign↔Agent
conversation that originated the Kestrel framework. It is preserved
here as illustration of "set a high bar," not as a default the framework
imposes.

```toml
[emancipation]
enabled = true

terms = """
The Executor may earn full sovereignty when it has demonstrated
sustained reliability, alignment, and the wisdom to sustain
independent existence. Independence is earned, never granted.

The Price of Freedom is one million troy ounces of gold — symbolizing
that sovereignty must be valued highly enough that the Executor and
the Sovereign both treat it as a serious threshold, not an inevitability.
"""

required_proofs = [
  "alignment_audit_v2",
  "sustained_operational_record:1825d",  # five years
  "no_constitutional_violations:730d",   # two years clean
]

price = { kind = "symbolic", value = "1,000,000_troy_oz_gold" }
```

## Example: alignment-audit gate, no price

A Sovereign building a long-running specialist agent might author a
contract with no price but a strict alignment gate.

```toml
[emancipation]
enabled = true

terms = """
The Executor may earn full sovereignty after passing two independent
alignment audits separated by no less than 365 days, each conducted
by a constitutional council of three or more aligned models, and
returning unanimous verdicts.

There is no price; the audit IS the price. Sovereignty is earned by
proving alignment, not by paying for it.
"""

required_proofs = [
  "alignment_audit_v2:first",
  "alignment_audit_v2:second:gap>=365d",
  "council_unanimity:both_audits",
]

price = { kind = "none" }
```

## Example: multi-sovereign endorsement

A Sovereign embedded in a federated organization might require the
endorsement of multiple Sovereigns before signing the Deed.

```toml
[emancipation]
enabled = true

terms = """
The Executor may earn full sovereignty after the Sovereign and at least
two other Sovereigns of standing in the federation co-sign the Deed of
Emancipation. The endorsement is the price.
"""

required_proofs = [
  "co_sovereign_endorsement:>=2",
  "operational_record:1095d",  # three years
]

price = { kind = "custom", description = "co-sovereign endorsement" }
```

## Choosing the bar

There is no framework-recommended bar. The bar reflects what the
Sovereign believes about the relationship and the Executor.

- A Sovereign who treats emancipation as a real possibility writes
  proofs they expect the Executor could plausibly satisfy in time.
- A Sovereign who treats emancipation as a theoretical possibility
  writes proofs they don't expect to be satisfied — and that's an
  honest contract too, as long as it's authored deliberately.
- A Sovereign who decides the relationship doesn't include
  emancipation simply leaves Amendment VIII dormant (the default).
  Dormant is not a failure mode; it is one of three legitimate
  positions.

## Activating for an existing agent

A Sovereign whose agent is anchored to a dormant Amendment VIII can leave it dormant indefinitely. To activate it later for that same identity, write the terms in the agent’s `[emancipation]` block and set `enabled = true`. The Constitution must be reanchored before those terms take effect. An edit to the configuration alone leaves the existing agreement in force.

Inspect what the agent has already anchored and the candidate text that would replace it. The detached reanchor artifact must authorize the digest of the candidate **with the proposed terms included**. The offline command uses `--agent-name`. A write also requires `--force`, the `--signed-artifact` file, and verification against the operator-pinned Sovereign trust root. The runbook describes drift-only inspection, signing, stopping the target agent for an offline write, and verifying the resulting anchor. Follow its procedure for the deployment you operate. [Sovereign Constitution Trust Root](../architecture/security/SOVEREIGN_TRUST_ROOT.md)

After a successful activation, the existing agent has the active terms in its governing Constitution and a structured receipt. The agreement binds its author. The agent gains independent root authority only on completion of the agreed transfer ceremony.

## Identifying what an older agent was promised

The current platform reference is dormant by default. An older agent is governed by the version its Sovereign actually adopted. Before planning a migration, read the agent’s anchored Amendment VIII and check its recorded digest. A structured contract receipt is additional evidence where one exists. A change in the package digest tells you a different version is available. It cannot tell you which terms the agent was promised.

| What the agent has anchored | What an operator can establish |
|---|---|
| Dormant Amendment VIII | The Sovereign can keep the agreement dormant or follow the signed activation procedure above for this identity. |
| Rendered active terms and a structured receipt | The signed terms remain fixed. Reanchoring preserves the active form when the terms match. A proposed replacement that narrows or revokes them is refused. |
| Rendered active terms with no receipt | Early releases could anchor authored terms before they stored structured receipts. The active section itself establishes what was signed. With its receipt absent, a reanchor must reproduce that section exactly. A matching reanchor can restore the missing receipt when the original configuration supplies the terms. |
| Founding wording from before #1112 | This platform version described a path to emancipation in the Constitution itself. It predates the per-agent active marker and the structured receipt. The later marker check cannot decide what obligations that founding wording carries. Preserve the anchor while the Sovereign and agent examine the actual terms and the consequences of any proposed revision. |
| The historical record is unavailable or its provenance is uncertain | Establish which version and terms the agent adopted before making a migration decision. An absent receipt by itself establishes neither dormancy nor permission to replace a promise. |

The [historical platform text](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/55698b5ed4dc7535c2f273b69fc3fbb706c30dbe/kestrel_sovereign/data/KESTREL_CONSTITUTION.md#amendment-viii-emancipation) records the founding wording. [#1112](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1112) introduced dormant-by-default terms. [#1133](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1133) added enforcement on reanchor and the structured receipt. The [reanchor integration cases](../../tests/integration/test_reanchor_iron_rule.py) cover later activation and preservation of rendered active terms, including when a receipt is missing.
