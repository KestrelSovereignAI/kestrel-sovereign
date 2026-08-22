"""``kestrel embeddings`` — operator visibility into stamped profiles (#1477).

Two subcommands:

``audit``
    Read-only. For each embedded table (``conversation_history``,
    ``saved_items``, ``document_chunks``) prints
    ``{profile_id: row_count}`` plus NULL count, joined to the
    ``embedding_profiles`` registry for human-readable
    provider/model/dim. Use to answer "what profiles do I have rows
    from?" and "do I have mixed-profile data that would benefit from
    a reindex?".

``reindex``
    Migrate stored vectors to the currently RESOLVED embedding
    profile (#2289). Sweeps ``conversation_history``, ``saved_items``
    and ``document_chunks`` for rows whose ``embedding_profile_id``
    differs from the target (or is NULL), re-embeds the source text in
    batches, and rewrites ``embedding_vec`` + ``embedding_profile_id``.
    Idempotent and resumable — an interrupted run loses nothing and a
    re-run continues. Prints a dry-run report by default; pass
    ``--yes`` to actually rewrite rows. Refuses when no embedding
    provider resolves, when ``embedding_route = "none"``, or when the
    resolved embedding dimension doesn't match the vector column width
    (with instructions for the required re-migration).

Both subcommands open the same production :class:`AsyncDatabase` the
agent/server open — ``KESTREL_DATABASE_URL`` for Postgres, otherwise
``KESTREL_DB_PATH/kestrel_prime.db`` (or the ``--agent-name`` /
``--data-dir`` selected agent's ``kestrel_prime.db``). In a
multi-agent checkout with more than one agent they refuse to guess and
require ``--agent-name`` / ``--data-dir`` — and, since #2327, so does a
checkout whose ``multi_agent.toml`` defines agents while ``KESTREL_DB_PATH``
is also set (rather than let the env var silently win against the wrong
database). A resolved SQLite file that doesn't exist is a hard error: these
verbs operate on an existing corpus and never create an empty one. The audit
subcommand never
touches the LLM stack, so it works without credentials. The reindex
subcommand needs the LLM stack to resolve the target embedding profile,
applying the agent's persisted runtime ``embedding_route`` (#2263)
first so it re-embeds to the profile the live agent actually resolves.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Tables that carry an ``embedding_profile_id`` column (#1477).
_EMBEDDED_TABLES: Tuple[str, ...] = (
    "conversation_history",
    "saved_items",
    "document_chunks",
)


def _add_db_target_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared ``--agent-name`` / ``--data-dir`` DB selectors.

    Both subcommands resolve the same production database the agent/server
    open (see :func:`_resolve_db_target`); these flags disambiguate which
    agent to operate on in a multi-agent checkout.
    """
    parser.add_argument(
        "--agent-name",
        default=None,
        help="Agent whose database to open (resolved via multi_agent.toml "
             "or agent_data/<name>/kestrel_prime.db). Required in a "
             "multi-agent checkout unless --data-dir is given.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Explicit agent data directory containing kestrel_prime.db. "
             "Overrides --agent-name and the KESTREL_DB_PATH default.",
    )


def _project_root() -> str:
    """The Kestrel project/home dir (``KESTREL_HOME`` → marker walk → default).

    Agent discovery must anchor here, not on ``os.getcwd()``: the rest of the
    CLI resolves the home through :func:`kestrel_sovereign.paths.project_dir`,
    so an operator running ``kestrel embeddings reindex`` from outside the
    project root (with ``KESTREL_HOME`` set) must find the same
    ``multi_agent.toml`` / ``agent_data`` the agent/server would — otherwise
    the multi-agent guard is silently bypassed (#2327).
    """
    try:
        from kestrel_sovereign.paths import project_dir

        return str(project_dir())
    except Exception:  # pragma: no cover - defensive
        return os.getcwd()


def _resolve_data_dir(data_dir: Any, root: str) -> str:
    """Resolve a roster ``data_dir`` (possibly relative) against ``root``."""
    text = str(data_dir)
    if os.path.isabs(text):
        return text
    return os.path.normpath(os.path.join(root, text))


def _agent_data_dir_for(agent_name: str) -> Optional[str]:
    """Resolve an agent name to its data directory (``multi_agent.toml`` first)."""
    root = _project_root()
    try:
        from kestrel_sovereign.multi_agent.config import MultiAgentConfig

        cfg = MultiAgentConfig.load(
            config_path=os.path.join(root, "multi_agent.toml"),
        )
        agent = cfg.agents.get(agent_name)
        data_dir = getattr(agent, "data_dir", None) if agent is not None else None
        if data_dir:
            return _resolve_data_dir(data_dir, root)
    except Exception:  # pragma: no cover - defensive
        pass
    candidate = os.path.join(root, "agent_data", agent_name)
    if os.path.isdir(candidate):
        return candidate
    return None


def _discover_local_agents() -> Dict[str, str]:
    """Return ``{agent_name: data_dir}`` for agents in this checkout.

    Prefers a ``multi_agent.toml`` roster; otherwise scans
    ``agent_data/*/kestrel_prime.db``. Anchored on
    :func:`kestrel_sovereign.paths.project_dir` (``KESTREL_HOME``-aware), not
    ``os.getcwd()``, so the multi-agent guard holds even when the command is
    run from outside the project root (#2327). Used to refuse to guess which
    DB to open when the checkout hosts more than one agent (#2289).
    """
    root = _project_root()
    agents: Dict[str, str] = {}
    try:
        if os.path.exists(os.path.join(root, "multi_agent.toml")):
            from kestrel_sovereign.multi_agent.config import MultiAgentConfig

            cfg = MultiAgentConfig.load(
                config_path=os.path.join(root, "multi_agent.toml"),
            )
            for name, agent in cfg.agents.items():
                data_dir = getattr(agent, "data_dir", None)
                if data_dir:
                    agents[name] = _resolve_data_dir(data_dir, root)
    except Exception:  # pragma: no cover - defensive
        agents = {}
    if agents:
        return agents
    base = os.path.join(root, "agent_data")
    if os.path.isdir(base):
        for entry in sorted(os.listdir(base)):
            if os.path.exists(os.path.join(base, entry, "kestrel_prime.db")):
                agents[entry] = os.path.join(base, entry)
    return agents


def _resolve_db_target(
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve which database to open the way the agent/server does.

    Returns ``(error, postgres_url, sqlite_path)``. On success exactly one
    of ``postgres_url`` / ``sqlite_path`` is set; ``error`` is set (and the
    others None) when the target can't be resolved unambiguously.

    Resolution order, mirroring ``server.py``'s single-agent boot and the
    multi-agent data-dir convention:

    1. ``KESTREL_DATABASE_URL`` (Postgres) — the same env server.py reads.
    2. Legacy ``DATABASE_URL`` (postgres:// or sqlite://) — back-compat.
    3. ``--data-dir`` → ``<dir>/kestrel_prime.db``.
    4. ``--agent-name`` → the agent's data dir + ``kestrel_prime.db``.
    5. ``KESTREL_DB_PATH`` → ``<dir>/kestrel_prime.db`` (server default).
    6. A single discovered agent → its ``kestrel_prime.db``.
    7. Multiple discovered agents → refuse; require ``--agent-name`` /
       ``--data-dir`` rather than guessing.
    8. Nothing discovered → ``<cwd>/kestrel_prime.db`` (server fallback).
    """
    backend = os.environ.get("KESTREL_DB_BACKEND", "").lower()
    kestrel_db_url = os.environ.get("KESTREL_DATABASE_URL")
    if kestrel_db_url and (
        backend == "postgres"
        or kestrel_db_url.startswith(("postgresql://", "postgres://"))
    ):
        return (None, kestrel_db_url, None)

    db_url = os.environ.get("DATABASE_URL")
    if db_url and db_url.startswith(("postgresql://", "postgres://")):
        return (None, db_url, None)
    if db_url and db_url.startswith("sqlite://"):
        # sqlite:///abs/path → ``/abs/path``; sqlite:///rel/path → ``rel/path``.
        path = (
            db_url[len("sqlite:///"):]
            if db_url.startswith("sqlite:///")
            else db_url[len("sqlite://"):]
        )
        return (None, None, path)

    data_dir = getattr(args, "data_dir", None)
    if data_dir:
        return (None, None, os.path.join(data_dir, "kestrel_prime.db"))

    agent_name = getattr(args, "agent_name", None)
    if agent_name:
        resolved = _agent_data_dir_for(agent_name)
        if resolved is None:
            return (
                f"agent '{agent_name}' not found — expected a data dir at "
                f"agent_data/{agent_name}/ or an entry in multi_agent.toml.",
                None,
                None,
            )
        return (None, None, os.path.join(resolved, "kestrel_prime.db"))

    db_path_dir = os.environ.get("KESTREL_DB_PATH")
    if db_path_dir:
        # KESTREL_DB_PATH must NOT silently override a multi-agent roster
        # (#2327). Production hosts set KESTREL_DB_PATH in .env; in a home
        # whose multi_agent.toml defines agents, honouring it blindly opens
        # the wrong DB (or creates an empty one) and reports a confident
        # false success against the wrong database. Refuse and require an
        # explicit selector rather than guessing which corpus is meant.
        roster = _discover_local_agents()
        if roster:
            names = ", ".join(sorted(roster))
            return (
                f"KESTREL_DB_PATH is set ({db_path_dir}) but this checkout "
                f"defines agent(s): {names}. Refusing to let KESTREL_DB_PATH "
                "silently choose the database — pass --agent-name <name> or "
                "--data-dir <dir> to select which agent's corpus to operate "
                "on.",
                None,
                None,
            )
        return (None, None, os.path.join(db_path_dir, "kestrel_prime.db"))

    agents = _discover_local_agents()
    if len(agents) == 1:
        (only_dir,) = agents.values()
        return (None, None, os.path.join(only_dir, "kestrel_prime.db"))
    if len(agents) > 1:
        names = ", ".join(sorted(agents))
        return (
            "multiple agents found in this checkout; specify which to open "
            "with --agent-name <name> or --data-dir <dir>. "
            f"Known agents: {names}.",
            None,
            None,
        )

    return (None, None, os.path.join(_project_root(), "kestrel_prime.db"))


async def _load_persisted_embedding_route(
    db: "Any", agent_id: Optional[str]
) -> Tuple[bool, Optional[str]]:
    """Read the runtime ``embedding_route`` knob persisted in agent_metadata.

    Mirrors ``ModelPreferenceMixin._load_embedding_route`` (#2263): a value
    persisted there via the API/UI overrides the config default and is what
    the live agent actually resolves. Returns ``(found, route)`` where
    ``route`` may be ``None`` (explicit auto/follow-chat). When ``agent_id``
    is None a single stored row is used (the common single-agent DB); an
    ambiguous multi-row case reports not-found and leaves the config default.
    """
    try:
        if agent_id:
            rows = await db.fetchall(
                "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
                (agent_id, "embedding_route"),
            )
        else:
            rows = await db.fetchall(
                "SELECT value FROM agent_metadata WHERE key = ?",
                ("embedding_route",),
            )
        if not rows or len(rows) != 1:
            return (False, None)
        return (True, json.loads(rows[0][0]))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("could not read persisted embedding_route: %s", exc)
        return (False, None)


async def _load_persisted_route_embedding_models(
    db: "Any", agent_id: Optional[str]
) -> Dict[str, Dict[str, Any]]:
    """Read the runtime per-route ``embedding_model`` pins from agent_metadata.

    Mirrors ``ModelPreferenceMixin._load_route_embedding_models`` (#2337): the
    embeddings UI persists ``{"<vendor>:<route>": {"model": str, "dim": int|None}}``
    here, and re-applying it re-advertises embedding support for the pinned route
    the same way the live agent does at boot. Returns the override map (possibly
    empty). When ``agent_id`` is None a single stored row is used (the common
    single-agent DB); an ambiguous multi-row case returns empty and leaves the
    config/discovery default.
    """
    try:
        if agent_id:
            rows = await db.fetchall(
                "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
                (agent_id, "embedding_model_overrides"),
            )
        else:
            rows = await db.fetchall(
                "SELECT value FROM agent_metadata WHERE key = ?",
                ("embedding_model_overrides",),
            )
        if not rows or len(rows) != 1:
            return {}
        overrides = json.loads(rows[0][0])
        return overrides if isinstance(overrides, dict) else {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("could not read persisted embedding_model_overrides: %s", exc)
        return {}


async def _apply_persisted_embedding_config(
    llm_service: "Any", db: "Any", agent_id: Optional[str]
) -> Optional[str]:
    """Apply the live agent's persisted embedding resolution order to a service.

    A freshly-constructed CLI-context ``LLMService`` validates a persisted
    ``embedding_route`` against **static config only** — it neither loads the
    runtime per-route model pins (#2337) nor runs embedding-model discovery
    (#2338) that the serving process applies at boot. Net effect (#2361): a route
    the server resolves fine (e.g. ``openrouter:api`` pinned via the UI) is
    rejected here as "does not advertise embedding support". This reproduces the
    boot path (``ModelPreferenceMixin._load_route_embedding_models`` →
    ``_load_embedding_route``) so the CLI and server agree about the same
    persisted state:

    1. Re-apply persisted per-route ``embedding_model`` pins — each pin
       re-advertises embedding support for that exact route (#2337).
    2. Fold live embedding discovery into route capabilities (#2338).
    3. Apply the persisted ``embedding_route`` knob (#2263).

    Returns ``None`` on success, or an operator-facing error string when the
    persisted route still can't be applied (caller refuses rather than reindex
    into the wrong profile).
    """
    # (1) Re-apply persisted per-route embedding_model pins (#2337). Each pin
    # writes embedding capability onto its exact route, so a route whose support
    # is only advertised via a UI-set model pin is recognised before validation.
    overrides = await _load_persisted_route_embedding_models(db, agent_id)
    for route, spec in overrides.items():
        if not isinstance(spec, dict):
            continue
        model = spec.get("model")
        if not model:
            continue
        try:
            llm_service.set_route_embedding_model(
                route, model, spec.get("dim"), persist=False
            )
            print(f"# using persisted embedding_model pin: {route} -> {model}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "skipping persisted embedding_model pin for %s: %s", route, exc
            )

    # (2) Fold live embedding discovery into route capabilities BEFORE the sync
    # validator runs (#2338), mirroring the settings endpoint and the boot path.
    # A route discovered dynamically (e.g. OpenRouter with no TOML pin) advertises
    # embedding support only after reconciliation; without this the persisted
    # route would be rejected as "does not advertise embedding support".
    if hasattr(llm_service, "reconcile_embedding_capabilities"):
        try:
            await llm_service.reconcile_embedding_capabilities(use_cache=True)
        except Exception as exc:  # pragma: no cover - discovery must not fail CLI
            logger.debug("embedding capability reconcile skipped: %s", exc)

    # (3) Apply the persisted embedding_route (#2263). If it still can't be
    # applied, REFUSE — continuing would silently reindex production rows into
    # whatever route was already active (the wrong embedding profile).
    found, route = await _load_persisted_embedding_route(db, agent_id)
    if found:
        try:
            llm_service.set_embedding_route(route, persist=False)
            if route:
                print(f"# using persisted embedding_route: {route}")
        except Exception as exc:
            return (
                f"the persisted embedding_route {route!r} is no longer valid "
                f"({exc}). Fix the embedding route (via the settings API/UI or "
                "config) before reindexing."
            )
    return None


def add_embeddings_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``embeddings`` subcommand under the top-level CLI parser."""
    parser = subparsers.add_parser(
        "embeddings",
        help="Inspect / manage embedding_profile_id stamps (#1477)",
    )
    embed_sub = parser.add_subparsers(dest="embeddings_command")

    audit_p = embed_sub.add_parser(
        "audit",
        help="Report {profile_id: row_count} per embedded table.",
    )
    audit_p.add_argument(
        "--table",
        choices=_EMBEDDED_TABLES,
        default=None,
        help="Limit the audit to one table (default: all).",
    )
    audit_p.add_argument(
        "--agent-id",
        default=None,
        help="Filter to one agent's rows (default: all agents).",
    )
    _add_db_target_args(audit_p)

    reindex_p = embed_sub.add_parser(
        "reindex",
        help="Re-embed stored vectors to the resolved embedding profile.",
    )
    reindex_p.add_argument(
        "--table",
        choices=_EMBEDDED_TABLES,
        default=None,
        help="Limit the reindex to one table (default: all three).",
    )
    reindex_p.add_argument(
        "--agent-id",
        default=None,
        help="Restrict every table to one agent's owned rows.",
    )
    _add_db_target_args(reindex_p)
    reindex_p.add_argument(
        "--batch",
        type=int,
        default=100,
        help="Rows re-embedded per batch/commit (default: 100).",
    )
    reindex_p.add_argument(
        "--rate-limit",
        type=float,
        default=0.0,
        dest="rate_limit",
        help="Seconds to sleep between batches — throttle for "
             "rate-limited cloud embedding providers (default: 0).",
    )
    reindex_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report per-table stale counts + an estimate without "
             "modifying any rows. This is also the default when --yes "
             "is not passed.",
    )
    reindex_p.add_argument(
        "--yes",
        action="store_true",
        help="Actually re-embed and rewrite rows. Without it the "
             "command only reports the dry-run scope.",
    )

    parser.set_defaults(_handler=run)


async def _audit(db: "Any", *, table: Optional[str], agent_id: Optional[str]) -> int:
    """Print profile-id row counts for the configured tables.

    Returns 0 on success, 1 if any table is missing the
    ``embedding_profile_id`` column (suggests the migration hasn't
    landed yet).
    """
    tables = (table,) if table else _EMBEDDED_TABLES
    overall_ok = True

    # Pull the human-readable registry into a lookup keyed by id.
    profile_lookup: Dict[str, Dict[str, Any]] = {}
    try:
        rows = await db.fetchall(
            "SELECT id, provider, model, dim FROM embedding_profiles", (),
        )
        for row in rows:
            profile_lookup[row[0]] = {
                "provider": row[1],
                "model": row[2],
                "dim": row[3],
            }
    except Exception as exc:
        logger.warning(
            "embedding_profiles registry not readable (table missing?): %s",
            exc,
        )

    print("# embedding_profile_id audit")
    if agent_id:
        print(f"# agent_id filter: {agent_id}")

    for tname in tables:
        # Build the WHERE clause. Empty agent filter means "all agents";
        # document chunks use their independent ownership ledger.
        agent_clause = ""
        agent_params: Tuple[Any, ...] = ()
        if agent_id and tname == "document_chunks":
            agent_clause = (
                " WHERE EXISTS ("
                "SELECT 1 FROM document_chunk_owners owners "
                "WHERE owners.chunk_id = document_chunks.chunk_id "
                "AND owners.agent_id = ?)"
                " AND EXISTS ("
                "SELECT 1 FROM file_owners file_owner "
                "WHERE file_owner.content_hash = document_chunks.file_hash "
                "AND file_owner.agent_id = ?)"
            )
            agent_params = (agent_id, agent_id)
        elif agent_id:
            agent_clause = " WHERE agent_id = ?"
            agent_params = (agent_id,)
        try:
            rows = await db.fetchall(
                f"SELECT embedding_profile_id, COUNT(*) FROM {tname}"
                f"{agent_clause} GROUP BY embedding_profile_id",
                agent_params,
            )
        except Exception as exc:
            print(f"\n{tname}: ERROR — {exc}")
            overall_ok = False
            continue

        print(f"\n{tname}:")
        if not rows:
            print("  (no rows)")
            continue
        total = sum(int(r[1] or 0) for r in rows)
        # Sort: NULL first (most-actionable), then by count desc.
        sorted_rows = sorted(
            rows,
            key=lambda r: (r[0] is not None, -int(r[1] or 0)),
        )
        for pid, count in sorted_rows:
            label = pid or "NULL"
            profile = profile_lookup.get(pid) if pid else None
            descriptor = ""
            if profile:
                descriptor = (
                    f"  ({profile['provider']}/{profile['model']}"
                    f" dim={profile['dim']})"
                )
            elif pid:
                descriptor = "  (unknown — not in embedding_profiles registry)"
            pct = (100.0 * int(count or 0) / total) if total else 0.0
            print(f"  {label:<14} {int(count or 0):>8}  ({pct:5.1f}%){descriptor}")
        print(f"  {'TOTAL':<14} {total:>8}")

    return 0 if overall_ok else 1


def _resolve_target(
    llm_service: Optional[Any] = None,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Resolve the target embedding profile from the active provider.

    Returns ``(error_message, embedding_service, target_profile_id)``.
    ``error_message`` is non-None (and the other two None) when
    reindexing must refuse: no provider resolves, ``embedding_route =
    "none"``, or the service can't describe itself. Factored out so
    unit tests can drive the refusal paths without an LLM.
    """
    try:
        from kestrel_sovereign.llm.service import LLMService
    except Exception as exc:  # pragma: no cover - defensive
        return (f"could not import LLMService: {exc}", None, None)

    try:
        service_owner = llm_service if llm_service is not None else LLMService()
    except Exception as exc:
        return (f"could not initialize LLM service: {exc}", None, None)

    # Explicit "none" means the operator has disabled embeddings — never
    # silently re-embed anyway.
    try:
        route = service_owner.get_embedding_route()
    except Exception:
        route = None
    if route and str(route).strip().lower() == "none":
        return (
            'embedding_route is set to "none" — embeddings are disabled. '
            "Set an embedding route/provider before reindexing.",
            None,
            None,
        )

    embedding_service = service_owner.get_embedding_service()
    if embedding_service is None:
        return (
            "no embedding-capable provider resolves for the active "
            "configuration; semantic search is on keyword fallback. "
            "Configure an embedding route/provider before reindexing.",
            None,
            None,
        )
    target = embedding_service.current_profile_id()
    if target is None:
        return (
            "the resolved embedding service can't describe itself "
            "(missing model/dim metadata); cannot derive a target profile.",
            None,
            None,
        )
    return (None, embedding_service, target)


def _resolve_column_dim() -> Optional[int]:
    """The deployment's vector-column width (``resolve_embedding_dim``)."""
    try:
        from kestrel_sovereign.storage.sqla.conversation_message import (
            resolve_embedding_dim,
        )

        return int(resolve_embedding_dim())
    except Exception:  # pragma: no cover - defensive
        return None


async def _reindex(
    db: "Any",
    *,
    table: Optional[str],
    agent_id: Optional[str],
    batch: int,
    rate_limit: float,
    dry_run: bool,
    apply: bool,
    llm_service: Optional[Any] = None,
    embedding_service: Optional[Any] = None,
    target_profile_id: Optional[str] = None,
    target_dim: Optional[int] = None,
) -> int:
    """Re-embed stored vectors to the resolved embedding profile (#2289).

    ``embedding_service`` / ``target_profile_id`` / ``target_dim`` may
    be injected (tests); otherwise they're resolved from the active
    LLM provider. Without ``apply`` (i.e. no ``--yes``) this only
    prints the dry-run scope.
    """
    from kestrel_sovereign.storage.embedding_reindex import (
        REINDEX_TABLES,
        EmbeddingReindexer,
    )

    # Resolve the target profile unless the caller injected one.
    if target_profile_id is None or embedding_service is None:
        # Apply the agent's persisted runtime embedding config (#2263/#2337/#2338)
        # BEFORE resolving the target, so the CLI re-embeds to the profile the
        # live agent actually resolves — not the config/default route. Mirrors
        # ModelPreferenceMixin._load_route_embedding_models → _load_embedding_route
        # on the agent boot path: re-apply per-route model pins, run embedding
        # discovery, then set the persisted route. Without this a UI-created route
        # the server resolves fine (e.g. openrouter:api pinned at runtime) is
        # rejected here as "does not advertise embedding support" (#2361).
        # Only done when we construct the LLMService (production path); an
        # injected service is presumed already configured (tests).
        if llm_service is None:
            try:
                from kestrel_sovereign.llm.service import LLMService

                llm_service = LLMService()
            except Exception as exc:
                print(
                    f"ERROR: could not initialize LLM service: {exc}",
                    file=sys.stderr,
                )
                return 2
            err = await _apply_persisted_embedding_config(llm_service, db, agent_id)
            if err is not None:
                print(f"ERROR: {err}", file=sys.stderr)
                return 2
        err, embedding_service, target_profile_id = _resolve_target(llm_service)
        if err is not None:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2

    if target_dim is None:
        target_dim = getattr(embedding_service, "embedding_dim", None)

    column_dim = _resolve_column_dim()

    # Dimension guard: writing target-dim vectors into a column sized to
    # a different width would strand recall. Refuse with the migration
    # steps rather than corrupt the column (#2289).
    if target_dim and column_dim and int(target_dim) != int(column_dim):
        print(
            "ERROR: resolved embedding dimension "
            f"({target_dim}) does not match the vector-column width "
            f"({column_dim}). Re-embedding at the new dimension requires "
            "a column migration first:\n"
            f"  1. Set KESTREL_EMBEDDING_DIM={target_dim} in the "
            "environment.\n"
            "  2. Drop the embedding_vec column on conversation_history, "
            "saved_items and document_chunks so the Phase-2 migration "
            f"recreates it at dim {target_dim} on next start.\n"
            "  3. Restart the agent, then re-run "
            "`kestrel embeddings reindex --yes`.",
            file=sys.stderr,
        )
        return 2

    tables = (table,) if table else REINDEX_TABLES
    reindexer = EmbeddingReindexer(
        db,
        embedding_service,
        target_profile_id,
        column_dim=column_dim,
        batch_size=batch,
        rate_limit_s=rate_limit,
    )

    # --- Dry-run report (always shown first) --------------------------------
    counts = await reindexer.count_all_stale(agent_id=agent_id, tables=tables)
    total_stale = sum(counts.values())

    print("# embeddings reindex")
    print(f"target_profile: {target_profile_id}")
    print(f"embedding_dim:  {target_dim if target_dim is not None else '(unknown)'}")
    print(f"agent_id:       {agent_id or '(all)'}")
    print(f"batch:          {batch}")
    if rate_limit:
        print(f"rate_limit:     {rate_limit}s between batches")
    print("stale rows (profile != target):")
    for tname in tables:
        print(f"  {tname:<22} {counts.get(tname, 0):>8}")
    print(f"  {'TOTAL':<22} {total_stale:>8}")

    if not apply:
        if total_stale:
            print(
                "\nDry-run only. Re-run with --yes to re-embed the "
                f"{total_stale} row(s) above.",
            )
        else:
            print("\nNothing to do — all rows already on the target profile.")
        return 0

    if not total_stale:
        print("\nNothing to do — all rows already on the target profile.")
        return 0

    # --- Apply --------------------------------------------------------------
    print("\nRe-embedding...")
    grand_reembedded = 0
    for tname in tables:
        stats = await reindexer.reindex_table(tname, agent_id=agent_id)
        grand_reembedded += stats.reembedded
        extras = []
        if stats.skipped_empty:
            extras.append(f"{stats.skipped_empty} empty")
        if stats.skipped_dim_mismatch:
            extras.append(f"{stats.skipped_dim_mismatch} dim-mismatch")
        if stats.failed:
            extras.append(f"{stats.failed} failed")
        suffix = f" ({', '.join(extras)})" if extras else ""
        print(
            f"  {tname:<22} re-embedded {stats.reembedded}/"
            f"{stats.scanned} scanned{suffix}"
        )
    print(f"\nDone. Re-embedded {grand_reembedded} row(s) to {target_profile_id}.")
    return 0


def run(args: argparse.Namespace) -> int:
    """Dispatch the subcommand chosen on the CLI."""
    if not getattr(args, "embeddings_command", None):
        print("usage: kestrel embeddings {audit|reindex} ...", file=sys.stderr)
        return 2

    # ``reindex`` constructs an LLMService and re-embeds stored content, so it
    # needs provider credentials *and* ``KESTREL_DATA_KEY`` to decrypt what it
    # is re-embedding — both of which it used to get as a side effect of that
    # constructor (#2896). Loading here, before ``_resolve_db_target``, also
    # makes this module's own claim true: it documents resolving the database
    # "the same way the agent/server does", and ``server.py`` resolves it
    # *after* loading the home's environment.
    from kestrel_sovereign.paths import load_project_env, project_dir

    load_project_env(project_dir())

    async def _runner() -> int:
        try:
            from kestrel_sovereign.storage.async_database import AsyncDatabase
        except Exception as exc:
            print(f"ERROR: could not import AsyncDatabase: {exc}", file=sys.stderr)
            return 2

        # Resolve the production DB the same way the agent/server does
        # (KESTREL_DATABASE_URL for Postgres, else KESTREL_DB_PATH/
        # kestrel_prime.db or the selected agent's data dir) rather than
        # guessing an unused ``default/memory.db`` (#2289).
        err, pg_url, sqlite_path = _resolve_db_target(args)
        if err is not None:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2
        # reindex/audit are maintenance verbs over an EXISTING corpus. If the
        # resolved SQLite file doesn't exist, refuse rather than let the driver
        # create a brand-new empty DB and truthfully count its zero rows —
        # that is the "confident false success against the wrong database" this
        # command must never produce (#2327).
        if not pg_url and (not sqlite_path or not os.path.exists(sqlite_path)):
            print(
                f"ERROR: no database found at {sqlite_path!r} — reindex/audit "
                "operate on an existing agent corpus and will not create a new "
                "empty database. Point --data-dir / --agent-name (or "
                "KESTREL_DB_PATH) at an existing kestrel_prime.db.",
                file=sys.stderr,
            )
            return 2
        try:
            if pg_url:
                db = await AsyncDatabase.postgres(pg_url)
            else:
                db = await AsyncDatabase.sqlite(sqlite_path)
        except Exception as exc:
            print(f"ERROR: could not connect to DB: {exc}", file=sys.stderr)
            return 2
        try:
            if args.embeddings_command == "audit":
                return await _audit(db, table=args.table, agent_id=args.agent_id)
            if args.embeddings_command == "reindex":
                return await _reindex(
                    db,
                    table=args.table,
                    agent_id=args.agent_id,
                    batch=args.batch,
                    rate_limit=args.rate_limit,
                    dry_run=args.dry_run,
                    apply=args.yes,
                )
            print(
                f"unknown subcommand: {args.embeddings_command}",
                file=sys.stderr,
            )
            return 2
        finally:
            close = getattr(db, "close", None)
            if close:
                try:
                    await close()
                except Exception:
                    pass

    return asyncio.run(_runner())
