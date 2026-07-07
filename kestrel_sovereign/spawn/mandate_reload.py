"""Reload-time reconstruction and enforcement of a spawned child's mandate (#2137).

A spawn mandate's restrictions are recorded on the child's ``spawned_by``
delegation edge at inception. Every agent boot path funnels through
``KestrelAgent.initialize()``, which calls :func:`read_spawn_mandate` +
:func:`register_restriction_hook` so a child's ``restricted_tools`` are
hard-denied whenever it runs — fresh spawn, reload, host restart, single-agent
server, or CLI — not only in the process that first spawned it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from kestrel_sovereign.spawn.mandate import SpawnMandate
from kestrel_sovereign.spawn.mandate_hook import MandateRestrictionHook

logger = logging.getLogger(__name__)


async def read_spawn_mandate(storage: Any, agent_did: str) -> Optional[SpawnMandate]:
    """Reconstruct a child's mandate from its persisted ``spawned_by`` edge.

    Returns ``None`` for a root (non-spawned) agent or a spawn with no recorded
    enforceable constraints. The reconstruction is intentionally UNSIGNED: the
    edge is the durable record and the mandate is used only to *re-apply
    restrictions* (which only ever tighten — fail-safe), not to re-verify the
    parent signature. ``features_allowed`` is deliberately omitted — feature-load
    narrowing is enforced at load (#1946), and leaving it unset avoids the
    constitution re-validation's features-subset check misfiring on reload
    against the child's own already-narrowed feature set.
    """
    try:
        edges = await storage.get_edges_from(agent_did)
    except Exception as e:  # noqa: BLE001 — a read failure must not crash boot
        logger.warning(
            "spawn-mandate reload: could not read delegation edges for %s: %s",
            agent_did[:30], e,
        )
        return None

    spawned = [e for e in edges if getattr(e, "label", None) == "spawned_by"]
    if not spawned:
        return None
    props = spawned[0].properties or {}
    constraints = props.get("additional_constraints") or {}
    if not constraints:
        return None

    kwargs = {
        "parent_did": spawned[0].target_id,
        "purpose": props.get("purpose", ""),
        "ttl_seconds": props.get("ttl_seconds", 3600),
        "max_child_depth": props.get("max_child_depth", 0),
        "additional_constraints": constraints,
        "child_did": agent_did,
    }
    if props.get("created_at"):
        kwargs["created_at"] = props["created_at"]
    return SpawnMandate(**kwargs)


def register_restriction_hook(hooks_manager: Any, mandate: Any) -> int:
    """Register the runtime restricted_tools deny hook for a mandate.

    Returns the number of distinct restricted tools enforced (0 = no-op, e.g. a
    mandate with no ``restricted_tools`` or a missing hooks manager).
    """
    constraints = getattr(mandate, "additional_constraints", None) or {}
    restricted = constraints.get("restricted_tools") or []
    if not restricted or hooks_manager is None:
        return 0
    hooks_manager.register(MandateRestrictionHook(restricted))
    count = len(set(restricted))
    logger.info("Registered MandateRestrictionHook for %d restricted tool(s) (#2137).", count)
    return count
