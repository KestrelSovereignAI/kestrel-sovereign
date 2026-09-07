"""One description of a webhook name collision and its remedy (#3239).

The dispatch router refuses a collided name where it is consumed (#3216);
the host reports it where it is created (boot, onboarding, registration, a
runtime enable, a rollback) and the owning agents read it through their
webhook tools. Every one of those surfaces once printed its own remedy,
derived from the owner labels at print time, and each round of review
found one of them handing out an address the router refuses. The remedy
is a function of the owners alone, so it lives here once — and it prints
an address ONLY for an agent whose agent-prefixed form actually
dispatches: an agent that owns the name through exactly one receiver.
"""

from __future__ import annotations

from typing import Optional, Sequence


def collision_facts(owners: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split the owner labels into the agents that own the name through
    more than one of their own receivers (the agent-prefixed form is
    refused for them too) and the agents that own it exactly once (the
    agent-prefixed form dispatches to them).

    ``owners`` carries one label per owning receiver, so an agent whose two
    receivers own the name appears twice.
    """
    counts: dict[str, int] = {}
    for label in owners:
        counts[label] = counts.get(label, 0) + 1
    duplicated = sorted(label for label, n in counts.items() if n > 1)
    single = sorted(label for label, n in counts.items() if n == 1)
    return duplicated, single


def prefixed_form_dispatches(owners: Sequence[str], label: Optional[str]) -> bool:
    """Whether ``/api/agents/{label}/webhooks/{name}`` dispatches: the agent
    owns the name through exactly one enabled receiver."""
    return label is not None and list(owners).count(label) == 1


def describe_collision(
    name: str,
    owners: Sequence[str],
    *,
    own_label: Optional[str] = None,
    own_endpoint: Optional[str] = None,
) -> str:
    """The sentence every surface prints for one collided name.

    Mirrors the refusal it explains. For every agent that owns the name
    once, the agent-prefixed address resolves the collision; for every agent
    that owns it through two of its own receivers the agent-prefixed form is
    refused too and only an unregister resolves it. The reader's own address
    (``own_endpoint``, for ``own_label``) is printed only when it dispatches
    to the reader; otherwise the generic form, and only when some agent's
    prefixed form dispatches at all.
    """
    duplicated, single = collision_facts(owners)
    n = len(owners)
    listed = ", ".join(owners)
    if duplicated and not single:
        if len(duplicated) == 1:
            return (
                f"'{name}' is owned by {n} enabled receivers of one agent "
                f"({duplicated[0]}): both the unprefixed /webhooks/{name} form "
                f"and the agent-prefixed form are refused; unregister one of "
                f"them."
            )
        return (
            f"'{name}' is owned by {n} enabled receivers on this host (agents: "
            f"{listed}), each of {', '.join(duplicated)} through two of its own "
            f"receivers: the unprefixed /webhooks/{name} form and every "
            f"agent-prefixed form are refused; each must unregister one of them, "
            f"after which the agent-prefixed form is the address."
        )
    address = (
        own_endpoint
        if own_endpoint and prefixed_form_dispatches(owners, own_label)
        else f"/api/agents/<agent>/webhooks/{name}"
    )
    if duplicated:
        return (
            f"'{name}' is owned by {n} enabled receivers on this host (agents: "
            f"{listed}): the unprefixed /webhooks/{name} form is refused for "
            f"every owner; for {', '.join(duplicated)} the agent-prefixed form "
            f"is refused too (two of its own receivers) and one of them must be "
            f"unregistered; senders of {', '.join(single)} use {address}."
        )
    return (
        f"'{name}' is owned by {n} enabled receivers on this host (agents: "
        f"{listed}): the unprefixed /webhooks/{name} form is refused for every "
        f"owner; point each sender at {address}."
    )
