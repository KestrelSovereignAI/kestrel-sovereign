"""Wizard steps. Each step is a callable ``run(ctx: SetupContext) -> None``.

Steps must be idempotent: running a step twice with the same answers
yields the same files. The orchestrator at
:mod:`kestrel_sovereign.setup.wizard` runs them in declared order.
"""

from kestrel_sovereign.setup.steps import agent, keys, llm, verify

#: Stable ordering used when the user runs ``kestrel setup`` with no step.
ORDERED = (
    ("keys", keys.run),
    ("llm", llm.run),
    ("agent", agent.run),
    ("verify", verify.run),
)

#: Lookup for ``kestrel setup <step>``. Names match the ORDERED keys.
BY_NAME = {name: fn for name, fn in ORDERED}

__all__ = ["ORDERED", "BY_NAME", "agent", "keys", "llm", "verify"]
