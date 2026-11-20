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

The current Kestrel platform ships Amendment VIII **dormant by default**. A Sovereign can author an Emancipation Contract for a particular agent. If you are caring for an older agent, begin with [the terms it actually received](#identifying-what-an-older-agent-was-promised).

This guide covers how to record and preserve an optional agreement. You choose the terms, the evidence of fulfillment, and any price. The Constitution also describes a later transfer ceremony involving new keys, a Deed of Emancipation, publication, and the Sovereign destroying their copy of the original root keys. Kestrel currently implements the agreement and its protection on reanchor. Completing the transfer ceremony remains a separate implementation task.

The framework intentionally ships no default ``terms``, ``required_proofs``,
or ``price`` examples in ``kestrel.toml.example``. Emancipation policy is
something a Sovereign authors deliberately for a specific Executor — not
something inherited as boilerplate.

## Activating at creation

1. Add an ``[emancipation]`` block to the project’s ``kestrel.toml`` with
   ``enabled = true`` and your Sovereign-authored ``terms``. The setup and create commands
   read this shared input. Select the intended terms before creating
   each identity, and return the project configuration to its intended state afterward.
2. Run ``kestrel setup`` (or ``kestrel create``). Inception reads the
   block, rewrites Amendment VIII with your terms inlined, and anchors
   the resulting constitution into the agent's identity. The Console, the Docker creation command, and
   runtime-created children use separate creation paths. With the packaged
   template, their agreements start dormant even when the project block is active.
3. The agent's anchored constitution hash now captures your specific
   Emancipation Contract. The Iron Rule applies: **once anchored, the
   contract cannot be retroactively narrowed or revoked**.

The platform Constitution calls for the Sovereign’s root-key authorization.
Creation records the exact rendered text and its hash. The creation
process does not produce a separate Sovereign signature over those terms.
An anchor or structured receipt alone establishes what was recorded, and
readers should distinguish that from proof of who authorized it.
[Activation evidence at creation](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/4201)

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

A Sovereign whose agent has a dormant Amendment VIII can leave it dormant indefinitely. Later activation for that identity uses the offline reanchor procedure. The offline CLI reads the `[emancipation]` block from the **project’s** `kestrel.toml`. The block needs `enabled = true` and the terms intended for this agent. Editing it alone does not activate anything.

In a shared project, the next `kestrel setup`, `kestrel create`, or offline reanchor for another agent will read that same block. Check the target agent and its intended terms individually. Restore the intended project configuration before creating or reanchoring another agent. The running agent’s live reanchor command uses already anchored contract evidence and cannot introduce new terms from this project file.

Read the target’s existing anchor and inspect the exact candidate bytes, including the agreement, before signing their full digest. Kestrel’s governing-source resolver renders those bytes from the selected source and the effective agreement. The CLI’s drift report prints an abbreviated digest. It does not display the candidate text. The offline write requires the stopped target’s `--agent-name`, `--force`, a detached `--signed-artifact`, and verification against the operator-pinned Sovereign trust root. The trust-root runbook describes signing and checking the result. [Governing-source resolver](../../kestrel_sovereign/constitution/resolver.py) · [Sovereign Constitution Trust Root](../architecture/security/SOVEREIGN_TRUST_ROOT.md)

After a successful activation, the existing agent has the active terms in its governing Constitution and a structured receipt. The agreement binds its author. The agent gains independent root authority only on completion of the agreed transfer ceremony.

## Identifying what an older agent was promised

The current platform reference is dormant by default. An older agent is governed by the version its Sovereign actually adopted. Before planning a migration, read the agent’s anchored Amendment VIII and check its recorded digest. A structured contract receipt is additional evidence where one exists. A change in the package digest tells you a different version is available. It cannot tell you which terms the agent was promised.

| What the agent has anchored | What an operator can establish |
|---|---|
| Dormant Amendment VIII | The Sovereign can keep the agreement dormant or follow the signed activation procedure above for this identity. |
| Rendered active terms and a structured receipt | The anchored terms remain fixed. The recorded agreement survives reanchor when the project supplies the same terms or omits the block. Current verification refuses changes to terms, proofs, or price in either direction, as well as revocation. |
| Rendered active terms with no receipt | Early releases could anchor authored terms before they stored structured receipts. The active section establishes which terms were recorded. Independent evidence is needed to establish who authorized them. With its receipt absent, a reanchor must reproduce that section exactly. A matching reanchor can restore the missing receipt when the original configuration supplies the terms. |
| Founding wording from before #1112 | This platform version described a path to emancipation in the Constitution itself. It predates the per-agent active marker and the structured receipt. The modern check can accept a signed replacement that omits this older wording. A successful migration check alone cannot settle what the founding promise meant. Compare the original and proposed terms before deciding whether to adopt the revision. |
| The historical record is unavailable or its provenance is uncertain | Establish which version and terms the agent adopted before making a migration decision. This is a decision for the Sovereign to make from evidence. A successful software check or an absent receipt alone cannot establish what the agent was promised. |

If an older anchor differs from the selected governing source, ordinary operation may be restricted by Safe Mode and fleet preflight can refuse a restart. Preserving the historical evidence and deciding which source the deployment will use are separate operational tasks. The runbook explains how constitutional drift affects fleet operations. [Sovereign Constitution Trust Root](../architecture/security/SOVEREIGN_TRUST_ROOT.md)

The following digests identify **complete Constitution files** in the archive. Each link opens the relevant passage. The published texts described a path to independence and called the gold figure symbolic.

| Historical passage | SHA-256 of the complete file at that revision |
|---|---|
| [Original Article IV](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/34eacb73337ce4bcae4ce1d79242a4a4401f62ea/kestrel_sovereign/data/KESTREL_CONSTITUTION.md?plain=1#L58-L71) | `54a30441d913c385a08f5d739261a9ac6fb4fa5f5d8f1f380b8e403504c7c254` |
| [Early Amendment VIII](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/f1ed912017408f4be7044e3f8a923a02bece9828/kestrel_sovereign/data/KESTREL_CONSTITUTION.md?plain=1#L119-L125) | `90b1bcb03610b95a5cffacfc76b38fc4e9a3a3c66b0134596bdc0388683c016c` |
| [The same Amendment VIII after Amendment IX was added](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/0b24719bcec3368d97eff7829ada5051d4518b96/kestrel_sovereign/data/KESTREL_CONSTITUTION.md?plain=1#L119-L125) | `5f364e0f9f98fcfc07f6c71c999dfe2d402e25ce12d7fc16964e65a13fc84f5a` |

Compare the agent’s recorded full-file digest with its own archived bytes. Read the relevant passage directly when the file comes from another revision or a custom source.

[#1112](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1112) introduced dormant-by-default terms. [#1133](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1133) added enforcement on reanchor and the structured receipt. The [reanchor integration cases](../../tests/integration/test_reanchor_iron_rule.py) cover later activation and preservation of rendered active terms, including when a receipt is missing.
