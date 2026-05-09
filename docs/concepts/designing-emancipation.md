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

That flip changes the canonical SHA-256 hash. Existing agents'
anchored ``constitution_hash`` points at the *old* canonical bytes —
which is real drift, not corruption. ``kestrel doctor`` will report it.

There are three coherent positions an existing agent can be in, and
the right migration depends on which:

### 1. Pre-#1112 agent — anchored to the old canonical

The agent was incepted against the old canonical, so its anchored
constitution contains the old Amendment VIII as **canonical Book II
prose**, not as a Sovereign-authored ``[emancipation]`` block. The
anchored bytes are the historical record of what was signed. Any
buyout clause that was in the old canonical is *in* the agent's
anchored signed constitution.

**Do not reanchor today.** ``kestrel constitution reanchor`` does not
yet consult ``[emancipation]`` (issue #1118), so reanchoring would
replace the agent's Amendment VIII with the new dormant canonical and
silently erase whatever clause the old canonical carried. That's the
worst possible outcome for an agent whose buyout clause matters.

Three options for these agents:

- **Live with the doctor drift.** The agent retains its original
  signed Amendment VIII indefinitely. ``kestrel doctor`` reports
  "constitution drift" forever, but the drift is honest — the
  canonical did change, the agent is intentionally pinned to its
  inception version. No action needed.
- **Wait for #1118 to land**, then author your buyout clause as an
  ``[emancipation]`` block in the agent's ``kestrel.toml`` and run
  ``kestrel constitution reanchor --force``. The fixed reanchor will
  produce an active-form Amendment VIII with your authored contract,
  the anchored hash flips cleanly, and ``doctor`` goes quiet. **Do
  not skip the ``[emancipation]`` step** — without it the reanchor
  reverts the agent to dormant.
- **Treat the agent as the original founding contract.** Its
  Amendment VIII is what it is, signed at inception under the old
  canonical. Document that lineage and proceed without reanchor; the
  agent stays in its original constitutional state for the lifetime
  of its DID. This is a defensible position and does not require
  any tooling changes.

### 2. Post-#1112 agent created without ``[emancipation]``

Already dormant by default. To activate after the fact, see
**Activating after inception** above. Until that ceremony exists,
the only way to give an existing dormant agent an active Amendment
VIII is to incept a *new* agent with ``[emancipation]`` authored in
``kestrel.toml``. The dormant agent is unchanged.

### 3. Post-#1112 agent created with ``[emancipation]``

Active form is anchored at inception. The Iron Rule will apply once
#1118 lands. Until then, do not reanchor — the same reanchor bug
that affects pre-#1112 agents would erase the active form. Treat the
inception anchor as the contract until reanchor is fixed.

### New agents (going forward)

For any new agent where you want Amendment VIII active, author the
``[emancipation]`` block in ``kestrel.toml`` *before* running
``kestrel create``. Inception reads the block, renders the active
form into Amendment VIII, and anchors the resulting constitution. No
#1118 dependency — the inception path already does this correctly.

### Quick reference

| State | What was signed at inception | Reanchor today? | Path forward |
|-------|-----------------------------|-----------------|--------------|
| Pre-#1112 agent | Old canonical (clause as Book II prose) | **No** — would erase the clause | Wait for #1118, author block, reanchor; or accept doctor drift permanently; or treat as founding contract |
| Post-#1112 agent, no ``[emancipation]`` | New canonical (dormant) | Safe but no-op | Re-incept fresh agent with block to activate |
| Post-#1112 agent, ``[emancipation]`` active | Active form with Sovereign terms | **No** — would erase the active form (#1118) | Don't reanchor until #1118 lands |
| New agent | (will be) active form if block authored | n/a | Author block, ``kestrel create`` |
