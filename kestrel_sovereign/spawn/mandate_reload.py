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
    enforceable ``additional_constraints``. The reconstruction is intentionally
    UNSIGNED: the edge is the durable record and the mandate is used only to
    *re-apply restrictions* (which only ever tighten — fail-safe), not to
    re-verify the parent signature.

    Fail CLOSED: a delegation-edge read error is NOT swallowed — it propagates so
    boot fails rather than silently continuing without a spawned child's
    restrictions (the read runs against already-initialised local storage, so a
    failure here is genuinely exceptional).

    Scope: ``features_allowed`` is deliberately NOT reconstructed onto this
    mandate object. Enforcing a spawned child's feature allowlist runs at feature
    discovery (see :func:`read_spawn_features_allowed`, applied in
    ``KestrelAgent.initialize`` — #2226); keeping it off the mandate object also
    avoids the constitution re-validation's features-subset check misfiring on
    reload against the child's own already-narrowed feature set.
    """
    # No try/except: propagate read errors (fail closed) — see docstring.
    edges = await storage.get_edges_from(agent_did)

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


async def read_spawn_features_allowed(storage: Any, agent_did: str) -> Optional[list]:
    """Read a spawned child's persisted ``features_allowed`` ceiling (#2226).

    Returns a non-empty ``list`` of canonical feature class names when a ceiling
    is recorded, else ``None`` (no ceiling → load all).

    An EMPTY or absent ``features_allowed`` returns ``None``, by design:
    ``SpawnMandate.features_allowed`` defaults to ``[]`` and ``_do_spawn`` treats
    ``[]`` as "inherit the parent ceiling / unspecified" — a genuine zero-feature
    grant is not expressible in the current data model. Crucially this keeps
    backward compatibility: children spawned before this change persisted an
    empty ``features_allowed`` for the inherited case, and must NOT be
    reinterpreted as "only mandatory features" on a post-upgrade restart. Real
    ceilings (explicit, or the inherited set now written by ``_do_spawn``) are
    always non-empty.

    Applied at feature discovery in ``KestrelAgent.initialize`` so the ceiling is
    enforced on EVERY boot path — single-agent server, CLI, direct
    ``KestrelAgent`` — not only the ``AgentManager`` path that threads it through
    ``LocalAgentConfig``.

    Fail CLOSED: like :func:`read_spawn_mandate`, an edge read error propagates
    (boot fails) rather than silently loading a child without its feature ceiling.
    """
    edges = await storage.get_edges_from(agent_did)
    spawned = [e for e in edges if getattr(e, "label", None) == "spawned_by"]
    if not spawned:
        return None
    raw = (spawned[0].properties or {}).get("features_allowed")
    if not raw:
        return None
    names = [str(f) for f in raw if str(f).strip()]
    return names or None


def register_restriction_hook(hooks_manager: Any, mandate: Any) -> int:
    """Register the runtime tool-constraint hook for a mandate.

    Returns the number of distinct restricted tools enforced (0 = no-op, e.g. a
    mandate with no tool constraints or a missing hooks manager).
    """
    constraints = getattr(mandate, "additional_constraints", None) or {}
    restricted = constraints.get("restricted_tools") or []
    restricted_args = constraints.get("restricted_tool_args") or {}
    allowed = constraints.get("allowed_tools") if "allowed_tools" in constraints else None
    if (not restricted and not restricted_args and allowed is None) or hooks_manager is None:
        return 0
    hooks_manager.register(
        MandateRestrictionHook(
            restricted,
            allowed_tools=allowed,
            restricted_tool_args=restricted_args,
        )
    )
    count = len(set(restricted)) + len(restricted_args) + int(allowed is not None)
    logger.info(
        "Registered MandateRestrictionHook for %d tool constraint(s) (#2137, #2321).",
        count,
    )
    return count
