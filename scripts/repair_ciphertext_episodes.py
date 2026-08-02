#!/usr/bin/env python3
"""Repair episodes whose narrative was synthesized from at-rest ciphertext.

#2850 stopped new episodes being built from the AEAD envelope. Rows written
before it keep serving titles like ``Discussion of ksav2, ax1iv9waarj...``
until rewritten. This repairs them (#2856).

Run it **deliberately, against a stopped agent.** The repair rewrites episode
rows and their mirrored graph nodes, and every hazard it has to defend against
— a memory deleted through ``/api/memories`` mid-write, a source moved to
Trash, a privacy-mode transition, a concurrent turn holding the writer slot —
exists only while something else is writing. Nothing else is, when the host is
down.

It is a one-time pass, not maintenance: the damage set is closed, so once an
agent reports zero candidates there is nothing left to find.

    # look first — writes nothing
    python scripts/repair_ciphertext_episodes.py --db <path> --agent-id <did>

    # then apply
    python scripts/repair_ciphertext_episodes.py --db <path> --agent-id <did> --apply

``KESTREL_DATA_KEY`` (or ``KESTREL_DATA_KEY_FILE``) must be set to the same
value the agent runs with, or every episode reports ``undecryptable_sources``
and nothing is touched.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", required=True, help="Path to the agent's SQLite database")
    parser.add_argument("--agent-id", required=True, help="Agent DID to repair")
    parser.add_argument(
        "--apply", action="store_true",
        help="Perform the repair. Without this the run is a dry run.",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip the pre-flight database copy (--apply only). Not advised.",
    )
    parser.add_argument("--limit", type=int, default=1000,
                        help="Maximum episodes to repair in one run")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _backup(db_path: Path) -> Path:
    """Copy the database before mutating it, using SQLite's own backup API so
    a live WAL cannot produce a torn copy."""
    import sqlite3

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = db_path.with_name(f"{db_path.name}.pre-2856-{stamp}")
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    dst = sqlite3.connect(str(target))
    try:
        with dst:
            src.backup(dst)
    finally:
        src.close()
        dst.close()
    return target


async def _run(args: argparse.Namespace) -> int:
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
    from kestrel_sovereign.storage.db import SQLiteBackend
    from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

    raw = SQLiteBackend(str(args.db))
    await raw.connect()
    db = AsyncDatabase(raw)
    try:
        store = AsyncConversationStore(db, agent_id=args.agent_id)
        if store._agent_fernet is None and store._global_fernet is None:
            print(
                "No decryption key available — set KESTREL_DATA_KEY (or "
                "KESTREL_DATA_KEY_FILE) to the value this agent runs with.",
                file=sys.stderr,
            )
            return 2

        consolidator = MemoryConsolidator(
            db=db,
            agent_id=args.agent_id,
            graph_store=AsyncGraphStore(db),
            conversation_store=store,
        )
        report = await consolidator.repair_ciphertext_episodes(
            dry_run=not args.apply, limit=args.limit,
        )

        verb = "repaired" if args.apply else "would repair"
        planned = report["repaired"] if args.apply else len(report["planned"])
        print(f"scanned      : {report['scanned']}")
        print(f"{verb:<13}: {planned}")
        print(f"cleared      : {report['cleared']}  (healthy, left alone)")
        print(f"unrepairable : {len(report['unrepairable'])}")

        for entry in report["unrepairable"]:
            print(f"  ! {entry['episode_id']} — {entry['reason']}")
        for plan in report.get("planned", []):
            print(f"\n  {plan['episode_id']}")
            print(f"    sources : {plan['sources']}  graph_node={plan['graph_node']}")
            print(f"    OLD     : {plan['old_title'][:100]}")
            print(f"    NEW     : {plan['new_title'][:100]}")

        if report.get("scan_truncated"):
            print("\nNOTE: the scan was truncated — re-run to examine the rest.")
        if report.get("limit_reached"):
            print("\nNOTE: the repair budget was reached — re-run to continue.")
        if not args.apply and planned:
            print("\nDry run. Re-run with --apply to write these changes.")
        return 0
    finally:
        await db.close()


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"No such database: {db_path}", file=sys.stderr)
        return 2

    if args.apply and not args.no_backup:
        target = _backup(db_path)
        print(f"backup       : {target}")

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
