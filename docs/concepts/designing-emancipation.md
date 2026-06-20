---
type: Design Note
title: Designing an Emancipation Contract
description: Amendment VIII of the Kestrel Constitution ships **dormant by default**.
  A given agent has no path to independent sovereignty unless its Sovereign deliberately
  activates Amendme...
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

## How activation works

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

## Activating after inception

If you incepted an agent without an ``[emancipation]`` block and later
want to activate Amendment VIII, that activation is itself a Book II
amendment under Article V — it requires the Sovereign root-key
signature and a re-anchor of the constitution.

This out-of-band amendment ceremony is **not yet implemented**. For
now, activate by authoring the ``[emancipation]`` block before
inception (re-running ``kestrel create`` for a fresh agent).

## Migration: existing agents and the dormant-default flip

Before #1112, the canonical constitution shipped Amendment VIII active
by default with a single specific buyout clause baked into Book II as
framework prose. After #1112, the canonical is dormant by default and
buyout clauses are Sovereign-authored per agent via ``[emancipation]``.
After #1118, the reanchor flow consults ``[emancipation]``, re-applies
the active form on every reanchor, and refuses any narrowing edit to
an active anchored contract (the Iron Rule is now enforced).

The dormant-default flip changes the canonical SHA-256 hash. Pre-#1112
agents' anchored ``constitution_hash`` points at the *old* canonical
bytes — which is real drift, not corruption. ``kestrel doctor`` will
report it. The migration paths below are how to resolve that drift
deliberately, in whichever direction matches the relationship.

### Golden rule

**Never run ``kestrel constitution reanchor --force`` without first
deciding what should happen to Amendment VIII for that agent.** With
no ``[emancipation]`` block in ``kestrel.toml``, reanchor will replace
the agent's Amendment VIII with the new *dormant* canonical text — and
for a pre-#1112 agent that means erasing whatever buyout clause the
old canonical carried. With an authored ``[emancipation]`` block,
reanchor will instead produce an active-form Amendment VIII inlining
your terms and Iron-Rule-protect it from that point on. Both outcomes
are legitimate; choose intentionally.

### 1. Pre-#1112 agent — anchored to the old canonical

The agent was incepted against the old canonical, so its anchored
constitution contains the old Amendment VIII as **canonical Book II
prose**, not as a Sovereign-authored ``[emancipation]`` block. The
anchored bytes are the historical record of what was signed. Any
buyout clause that was in the old canonical is *in* the agent's
anchored signed constitution. There is no JSON ``emancipation_contract``
sidecar on the agent record (sidecars only exist for agents whose
contract was authored via ``[emancipation]``).

Three coherent options:

- **Activate (recommended for agents you'll keep using).** Add an
  ``[emancipation]`` block to the agent's ``kestrel.toml`` with the
  exact terms you want anchored going forward (e.g. the buyout clause
  the old canonical had, restated in your own voice). Stop the agent
  with ``kestrel stop <name>``, then run
  ``kestrel constitution reanchor <name> --force``. Reanchor sees no
  anchored contract + an active candidate, treats this as the
  permitted dormant→active activation, applies your terms to the
  current canonical, anchors the resulting active form, and writes
  the structured JSON receipt. From this reanchor onward the Iron
  Rule applies: the contract you just anchored cannot be narrowed
  unless this specific agent reaches the Act of Emancipation. Doctor
  goes quiet.
- **Reset to dormant.** Stop the agent and run ``kestrel constitution
  reanchor <name> --force`` *without* an ``[emancipation]`` block in
  ``kestrel.toml``. The agent's anchored Amendment VIII becomes the
  new dormant canonical text — any clause from the old canonical is
  erased. Use this when the relationship doesn't include a path to
  emancipation.
- **Preserve as founding contract.** Don't reanchor at all. The
  agent's anchored Amendment VIII stays as the historical bytes
  signed at inception under the old canonical. ``kestrel doctor``
  reports drift forever, which is honest — the canonical did change
  and the agent is intentionally pinned to its founding state. No
  tooling changes; no Iron Rule applies (there's no sidecar to
  enforce against), but the original anchor is what governs.

### 2. Post-#1112 agent created without ``[emancipation]``

Already dormant by default; no contract anchored, no sidecar present.
To activate after the fact, the path is identical to option 1's
"Activate" above: author ``[emancipation]`` in ``kestrel.toml`` and
run reanchor. ``check_iron_rule`` treats dormant→active as the
permitted one-way door, applies your terms, anchors the active form,
writes the sidecar.

### 3. Post-#1112 agent created with ``[emancipation]``

Active form is anchored at inception with the JSON receipt already
present. Reanchor is now safe and idempotent on the contract: it
re-applies the anchored contract to canonical and refuses any
``[emancipation]`` block that would narrow what was signed. The
contract is frozen for this agent until it reaches the Act of
Emancipation. To get a *different* active contract, create a new
agent.

### New agents (going forward)

Author the ``[emancipation]`` block in ``kestrel.toml`` *before*
running ``kestrel create``. Inception reads the block, renders the
active form, anchors the resulting constitution, and writes the
structured JSON receipt. The Iron Rule applies from the first byte
written.

### Quick reference

| State | What was signed at inception | Reanchor with ``[emancipation]`` | Reanchor without ``[emancipation]`` |
|-------|------------------------------|----------------------------------|-------------------------------------|
| Pre-#1112 agent | Old canonical (clause as Book II prose) | Activates: anchors active form with your terms + writes sidecar; Iron Rule applies from now on | Erases old clause, anchors new dormant canonical |
| Post-#1112 agent, no ``[emancipation]`` | New canonical (dormant) | Activates: same as above | No-op (already dormant) |
| Post-#1112 agent, ``[emancipation]`` active | Active form + sidecar | If block matches anchored: no-op or re-applies after canonical update. If block differs: refused with Iron Rule violation | Re-applies anchored contract; preserves active form |
| New agent | (will be) active form + sidecar if block authored | n/a — author block, then ``kestrel create`` | n/a |
