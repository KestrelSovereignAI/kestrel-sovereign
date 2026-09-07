"""One description of a webhook name collision and its remedy (#3239).

The dispatch router refuses a collided name where it is consumed (#3216);
the host reports it where it is created (boot, onboarding, registration, a
runtime enable) and the owning agents read it through their webhook tools.
Every one of those surfaces printed its own remedy, and two of them said
"use the agent-prefixed address" for a collision the prefixed form refuses
too. The remedy is a function of the owners alone, so it lives here once.
"""

from __future__ import annotations

from typing import Optional, Sequence


def collision_facts(owners: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split the owner labels into the agents that own the name through
    more than one of their own receivers and the agents that own it once.

    ``owners`` carries one label per owning receiver, so an agent whose two
    receivers own the name appears twice.
    """
    counts: dict[str, int] = {}
    for label in owners:
        counts[label] = counts.get(label, 0) + 1
    duplicated = sorted(label for label, n in counts.items() if n > 1)
    single = sorted(label for label, n in counts.items() if n == 1)
    return duplicated, single


def describe_collision(
    name: str,
    owners: Sequence[str],
    *,
    own_endpoint: Optional[str] = None,
) -> str:
    """The sentence every surface prints for one collided name.

    Mirrors the refusal it explains: between agents the agent-prefixed
    address resolves the collision; for an agent that owns the name through
    two of its own receivers the agent-prefixed form is refused too
    (``receiver.py``'s scoped refusal) and only an unregister resolves it.
    A mixed case says both. ``own_endpoint`` is the reader's own
    agent-prefixed address, when the surface knows it.
    """
    duplicated, single = collision_facts(owners)
    n = len(owners)
    listed = ", ".join(owners)
    if duplicated and not single:
        who = ", ".join(duplicated)
        return (
            f"'{name}' is owned by {n} enabled receivers of one agent ({who}): "
            f"both the unprefixed /webhooks/{name} form and the agent-prefixed "
            f"form are refused; unregister one of them."
        )
    address = own_endpoint or f"/api/agents/<agent>/webhooks/{name}"
    if duplicated:
        return (
            f"'{name}' is owned by {n} enabled receivers on this host (agents: "
            f"{listed}): the unprefixed /webhooks/{name} form is refused for "
            f"every owner; for {', '.join(duplicated)} the agent-prefixed form "
            f"is refused too (two of its own receivers) and one of them must be "
            f"unregistered; other senders use {address}."
        )
    return (
        f"'{name}' is owned by {n} enabled receivers on this host (agents: "
        f"{listed}): the unprefixed /webhooks/{name} form is refused for every "
        f"owner; point each sender at {address}."
    )
