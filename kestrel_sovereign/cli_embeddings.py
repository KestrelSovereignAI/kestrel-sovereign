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
    Backfills ``embedding_profile_id`` (and re-embeds rows) so all
    rows for an agent share one profile. Implementation note:
    ``--dry-run`` works today and reports the row counts that would
    be touched. Full re-embedding requires per-table re-encryption
    handling for ``conversation_history`` and is intentionally
    deferred — operators with NULL/mixed rows can still audit + see
    the scope of the work.

Both subcommands open an :class:`AsyncDatabase` from
``DATABASE_URL`` or fall back to the default SQLite path in the
current working directory. The audit subcommand never touches the
LLM stack, so it works without credentials. The reindex subcommand
needs the LLM stack to derive the target profile when
``--target-profile auto`` is used.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tables that carry an ``embedding_profile_id`` column (#1477).
_EMBEDDED_TABLES: Tuple[str, ...] = (
    "conversation_history",
    "saved_items",
    "document_chunks",
)


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

    reindex_p = embed_sub.add_parser(
        "reindex",
        help="Backfill / re-stamp embedding_profile_id for an agent.",
    )
    reindex_p.add_argument(
        "--table",
        choices=_EMBEDDED_TABLES,
        required=True,
        help="Which table to reindex.",
    )
    reindex_p.add_argument(
        "--agent-id",
        required=False,
        help="Restrict to one agent's rows. Required for "
             "conversation_history and saved_items; document_chunks "
             "is global so this is ignored.",
    )
    reindex_p.add_argument(
        "--target-profile",
        default="auto",
        help="Profile id to backfill into, or ``auto`` to derive from "
             "the active LLM provider's embedding service.",
    )
    reindex_p.add_argument(
        "--batch",
        type=int,
        default=500,
        help="Rows per batch when re-embedding (default: 500).",
    )
    reindex_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without modifying any rows. Required for "
             "now — full row re-embedding is not yet implemented.",
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
        # Build the WHERE clause. Empty agent filter means "all agents".
        # document_chunks has no agent_id — drop the filter there.
        agent_clause = ""
        agent_params: Tuple[Any, ...] = ()
        if agent_id and tname != "document_chunks":
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


async def _reindex(
    db: "Any",
    *,
    table: str,
    agent_id: Optional[str],
    target_profile: str,
    batch: int,
    dry_run: bool,
) -> int:
    """Backfill / re-stamp ``embedding_profile_id``.

    Today: only ``--dry-run`` is implemented. Full re-embedding
    requires per-table re-encryption handling for
    ``conversation_history`` (the content column is encrypted) and is
    a follow-up. Operators with NULL or mixed rows still get the
    audit + dry-run scope so they can plan the migration.
    """
    if not dry_run:
        print(
            "ERROR: full reindex is not yet implemented — re-embedding "
            "rows requires per-table re-encryption handling that's "
            "tracked as a follow-up. Re-run with --dry-run to see the "
            "scope of rows that would be touched.",
            file=sys.stderr,
        )
        return 2

    if table != "document_chunks" and not agent_id:
        print(
            f"ERROR: --agent-id is required for table {table}.",
            file=sys.stderr,
        )
        return 2

    # Derive target profile id if requested.
    resolved_target: Optional[str] = None
    if target_profile == "auto":
        try:
            from kestrel_sovereign.llm.embedding_service import (
                get_provider_embedding_service,
            )

            service = get_provider_embedding_service()
            if service is None:
                print(
                    "ERROR: --target-profile auto needs a configured "
                    "embedding-capable provider; none found.",
                    file=sys.stderr,
                )
                return 2
            resolved_target = service.current_profile_id()
            if resolved_target is None:
                print(
                    "ERROR: active embedding service can't describe "
                    "itself (missing model/dim metadata).",
                    file=sys.stderr,
                )
                return 2
        except Exception as exc:
            print(f"ERROR: could not resolve target profile: {exc}", file=sys.stderr)
            return 2
    else:
        resolved_target = target_profile

    where_parts: List[str] = ["(embedding_profile_id IS NULL OR embedding_profile_id != ?)"]
    params: List[Any] = [resolved_target]
    if agent_id and table != "document_chunks":
        where_parts.append("agent_id = ?")
        params.append(agent_id)

    sql = (
        f"SELECT COUNT(*) FROM {table} WHERE "
        + " AND ".join(where_parts)
        + " AND embedding_vec IS NOT NULL"
    )
    try:
        rows = await db.fetchall(sql, tuple(params))
    except Exception as exc:
        print(f"ERROR: count query failed: {exc}", file=sys.stderr)
        return 2

    affected = int(rows[0][0]) if rows else 0
    print(f"# embeddings reindex --dry-run")
    print(f"table:         {table}")
    print(f"agent_id:      {agent_id or '(all)'}")
    print(f"target_profile: {resolved_target}")
    print(f"batch:         {batch}")
    print(f"rows that would be re-embedded: {affected}")
    return 0


def run(args: argparse.Namespace) -> int:
    """Dispatch the subcommand chosen on the CLI."""
    if not getattr(args, "embeddings_command", None):
        print("usage: kestrel embeddings {audit|reindex} ...", file=sys.stderr)
        return 2

    async def _runner() -> int:
        try:
            from kestrel_sovereign.storage.async_database import AsyncDatabase
        except Exception as exc:
            print(f"ERROR: could not import AsyncDatabase: {exc}", file=sys.stderr)
            return 2

        # ``DATABASE_URL`` overrides; otherwise look for a SQLite
        # file in the current working directory. Matches the
        # default-path convention other CLI subcommands use.
        db_url = os.environ.get("DATABASE_URL")
        try:
            if db_url and db_url.startswith(("postgresql://", "postgres://")):
                db = await AsyncDatabase.postgres(db_url)
            elif db_url and db_url.startswith("sqlite://"):
                # sqlite:///abs/path → ``/abs/path``; sqlite:///rel/path → ``rel/path``
                path = db_url[len("sqlite:///") :] if db_url.startswith("sqlite:///") else db_url[len("sqlite://") :]
                db = await AsyncDatabase.sqlite(path)
            else:
                default_path = os.path.join(
                    os.getcwd(), "agent_data", "default", "memory.db"
                )
                db = await AsyncDatabase.sqlite(default_path)
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
                    target_profile=args.target_profile,
                    batch=args.batch,
                    dry_run=args.dry_run,
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
