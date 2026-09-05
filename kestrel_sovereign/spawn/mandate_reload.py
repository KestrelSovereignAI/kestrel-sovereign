"""Reload-time reconstruction and enforcement of a spawned child's mandate.

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

from kestrel_sovereign.spawn.mandate import (
    SpawnMandate,
    validate_spawn_max_child_depth,
)
from kestrel_sovereign.spawn.mandate_hook import MandateRestrictionHook

logger = logging.getLogger(__name__)


async def _read_spawned_by_edge(storage: Any, agent_did: str) -> Any:
    """Return the one durable lineage edge for ``agent_did``.

    A child can have at most one authority parent.  Silently choosing the first
    of several edges would make database iteration order decide who may govern
    the child, so ambiguous lineage fails closed on every reload path.
    """

    edges = await storage.get_edges_from(agent_did)
    spawned = [edge for edge in edges if getattr(edge, "label", None) == "spawned_by"]
    if not spawned:
        return None
    if len(spawned) != 1:
        raise ValueError(
            f"Agent {agent_did!r} has {len(spawned)} spawned_by parents; "
            "refusing ambiguous delegation authority"
        )
    edge = spawned[0]
    if getattr(edge, "source_id", None) != agent_did:
        raise ValueError("spawned_by edge source does not match the child DID")
    parent_did = getattr(edge, "target_id", None)
    if not isinstance(parent_did, str) or not parent_did or parent_did == agent_did:
        raise ValueError("spawned_by edge has an invalid parent DID")
    properties = getattr(edge, "properties", None)
    if properties is not None and not isinstance(properties, dict):
        raise TypeError("spawned_by edge properties must be a mapping")
    return edge


async def read_spawn_mandate(storage: Any, agent_did: str) -> Optional[SpawnMandate]:
    """Reconstruct a child's mandate from its persisted ``spawned_by`` edge.

    Returns ``None`` only for a root (non-spawned) agent. Legacy edges may
    reconstruct an unsigned projection for restriction enforcement and
    attribution. Governance consumers must independently require and verify
    ``parent_signature`` before treating the projection as authority.

    Fail CLOSED: a delegation-edge read error is NOT swallowed — it propagates so
    boot fails rather than silently continuing without a spawned child's
    restrictions (the read runs against already-initialised local storage, so a
    failure here is genuinely exceptional).

    ``features_allowed`` is reconstructed on the returned projection and read
    by :func:`read_spawn_features_allowed` during feature discovery. The
    constitution check runs before this reload projection is attached to the
    child, so it still cannot misread the child's already-narrowed set as a new
    grant.
    """
    # No try/except: propagate read/validation errors (fail closed).
    edge = await _read_spawned_by_edge(storage, agent_did)
    if edge is None:
        return None
    props = edge.properties or {}

    constraints = props.get("additional_constraints")
    if constraints is None:
        constraints = {}
    if not isinstance(constraints, dict):
        raise TypeError("spawned_by additional_constraints must be a mapping")
    features_allowed = props.get("features_allowed")
    if features_allowed is None:
        features_allowed = []
    if not isinstance(features_allowed, list) or any(
        not isinstance(feature, str) or not feature.strip()
        for feature in features_allowed
    ):
        raise TypeError("spawned_by features_allowed must be a list of names")

    created_at = props.get("created_at", "")
    if not isinstance(created_at, str):
        raise TypeError("spawned_by created_at must be a string")
    purpose = props.get("purpose", "")
    constitution_hash = props.get("constitution_hash", "")
    budget_allocation = props.get("budget_allocation", 0.0)
    parent_signature = props.get("parent_signature")
    ttl_seconds = props.get("ttl_seconds", 3600)
    max_child_depth = props.get("max_child_depth", 0)
    if not isinstance(purpose, str):
        raise TypeError("spawned_by purpose must be a string")
    if not isinstance(constitution_hash, str):
        raise TypeError("spawned_by constitution_hash must be a string")
    if not isinstance(budget_allocation, (int, float)) or isinstance(
        budget_allocation, bool
    ):
        raise TypeError("spawned_by budget_allocation must be numeric")
    if parent_signature is not None and not isinstance(parent_signature, str):
        raise TypeError("spawned_by parent_signature must be a string")
    if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool):
        raise TypeError("spawned_by ttl_seconds must be an integer")
    # ``ttl_seconds <= 0`` is the documented persistent-child sentinel.  Keep
    # its exact value on the authority projection; only delegation depth is a
    # non-negative bound.
    try:
        validate_spawn_max_child_depth(max_child_depth)
    except (TypeError, ValueError) as error:
        raise type(error)(str(error).replace("spawn mandate", "spawned_by")) from None

    kwargs = {
        "parent_did": edge.target_id,
        "purpose": purpose,
        "ttl_seconds": ttl_seconds,
        "max_child_depth": max_child_depth,
        "additional_constraints": constraints,
        "features_allowed": list(features_allowed),
        "child_did": agent_did,
        "constitution_hash": constitution_hash,
        # Preserve the original JSON number representation. ``1`` and ``1.0``
        # compare numerically but serialize differently in the signed payload.
        "budget_allocation": budget_allocation,
        "parent_signature": parent_signature,
        # Legacy parent-only edges predate mandate timestamps.  Preserve that
        # absence instead of manufacturing a fresh validity window at boot.
        "created_at": created_at,
    }
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
    mandate = await read_spawn_mandate(storage, agent_did)
    if mandate is None:
        return None
    raw = mandate.features_allowed
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
