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
    counts = {label: list(owners).count(label) for label in duplicated}

    def _through(label: str) -> str:
        return f"{counts[label]} of its own receivers"

    if duplicated and not single:
        if len(duplicated) == 1:
            return (
                f"'{name}' is owned by {n} enabled receivers of one agent "
                f"({duplicated[0]}): both the unprefixed /webhooks/{name} form "
                f"and the agent-prefixed form are refused; unregister all but "
                f"one of them."
            )
        return (
            f"'{name}' is owned by {n} enabled receivers on this host (agents: "
            f"{listed}), "
            + ", ".join(f"{label} through {_through(label)}" for label in duplicated)
            + f": the unprefixed /webhooks/{name} form and every agent-prefixed "
            f"form are refused; each must unregister all but one of them, after "
            f"which the agent-prefixed form is the address."
        )
    own_address_dispatches = bool(own_endpoint) and prefixed_form_dispatches(
        owners, own_label
    )
    generic = f"/api/agents/<agent>/webhooks/{name}"
    if duplicated:
        # This sentence names the single owners whose senders can use the
        # prefixed form. A concrete address may stand there only when the
        # reader is the ONLY such agent: naming other agents next to one
        # agent's address would send their senders into that receiver.
        address = (
            own_endpoint
            if own_address_dispatches and single == [own_label]
            else generic
        )
        return (
            f"'{name}' is owned by {n} enabled receivers on this host (agents: "
            f"{listed}): the unprefixed /webhooks/{name} form is refused for "
            f"every owner; for "
            + ", ".join(f"{label} ({_through(label)})" for label in duplicated)
            + " the agent-prefixed form is refused too and all but one of those "
            f"receivers must be unregistered; senders of {', '.join(single)} use "
            f"{address}."
        )
    # Addressed to the reader's own senders ("each sender" of the reader), so
    # the reader's own dispatching address is right however many agents own
    # the name once.
    address = own_endpoint if own_address_dispatches else generic
    return (
        f"'{name}' is owned by {n} enabled receivers on this host (agents: "
        f"{listed}): the unprefixed /webhooks/{name} form is refused for every "
        f"owner; point each sender at {address}."
    )
