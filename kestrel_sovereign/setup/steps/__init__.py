"""Wizard steps. Each step is a callable ``run(ctx: SetupContext) -> None``.

Steps must be idempotent: running a step twice with the same answers
yields the same files. The orchestrator at
:mod:`kestrel_sovereign.setup.wizard` runs them in declared order.

There are two kinds of steps:

  - **Default**, listed in :data:`ORDERED` — run on every ``kestrel setup``.
  - **Optional**, listed in :data:`BY_NAME` only — run only when the user
    explicitly asks (``kestrel setup talon``). These exist for opt-in
    integrations the user does not want set up automatically.
"""

from kestrel_sovereign.setup.steps import (
    agent,
    emancipation,
    integrations,
    keys,
    llm,
    talon,
    verify,
)

#: Stable ordering used when the user runs ``kestrel setup`` with no step.
#:
#: ``integrations`` lives between ``llm`` and ``agent`` so the user can
#: opt into Tavily/voice/cloud credentials immediately after picking
#: their LLM provider, but still before the agent is incepted (which
#: is the slow step). Declining everything in ``integrations`` is a
#: no-op — the wizard flows straight into ``agent``.
#:
#: ``emancipation`` runs immediately before ``agent`` because Amendment
#: VIII activation must be persisted to ``kestrel.toml`` before
#: inception, so the contract is folded into the agent's anchored
#: constitution hash. The default in every flow except interactive is
#: dormant (no block written, Amendment VIII renders dormant).
ORDERED = (
    ("keys", keys.run),
    ("llm", llm.run),
    ("integrations", integrations.run),
    ("emancipation", emancipation.run),
    ("agent", agent.run),
    ("verify", verify.run),
)

#: Optional, opt-in steps. Not in ORDERED — only fire when named.
OPTIONAL = (
    ("talon", talon.run),
)

#: Lookup for ``kestrel setup <step>``. Includes both default and optional.
BY_NAME = {name: fn for name, fn in (*ORDERED, *OPTIONAL)}

__all__ = [
    "ORDERED", "OPTIONAL", "BY_NAME",
    "agent", "emancipation", "integrations", "keys", "llm", "talon", "verify",
]
