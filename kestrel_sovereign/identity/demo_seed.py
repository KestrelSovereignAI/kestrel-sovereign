"""Restore the pinned signing identity and initial SQLite state of a Cloud Run demo."""

from __future__ import annotations

import argparse
import lzma
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from kestrel_sovereign.identity.custody_bundle import (
    IdentityCustodyError,
    restore_identity_bundle_from_env,
)

MAX_DATABASE_BYTES = 8 * 1024 * 1024


def restore_demo_seed(agent_dir: Path, source: Path, *, expected_did: str) -> None:
    """Verify both pinned inputs before admitting a demo to the startup roster."""
    if agent_dir.is_symlink() or (
        agent_dir.exists()
        and any(
            path.name != "SOUL.md" or path.is_symlink() or not path.is_file()
            for path in agent_dir.iterdir()
        )
    ):
        raise IdentityCustodyError("Demo seed refuses to replace existing agent state")

    agent_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".demo-restore-", dir=agent_dir.parent) as temp:
        staged_agent = Path(temp) / "agent"
        restore_identity_bundle_from_env(staged_agent, expected_did=expected_did)

        database = staged_agent / "kestrel_prime.db"
        with lzma.open(source, "rb") as compressed:
            contents = compressed.read(MAX_DATABASE_BYTES + 1)
        if len(contents) > MAX_DATABASE_BYTES:
            raise IdentityCustodyError("Demo seed database exceeds the restore limit")
        database.write_bytes(contents)
        database.chmod(0o600)

        database_uri = database.as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(database_uri, uri=True)) as db:
            integrity = db.execute("PRAGMA quick_check").fetchall()
            demos = db.execute(
                """
                SELECT node_id, json_extract(properties, '$.is_demo'),
                       json_extract(properties, '$.is_test_instance')
                FROM graph_nodes WHERE node_type = 'agent'
                """
            ).fetchall()
        if integrity != [("ok",)] or demos != [(expected_did, 1, 1)]:
            raise IdentityCustodyError("Demo seed database does not bind to the pinned test identity")

        if not agent_dir.exists():
            os.replace(staged_agent, agent_dir)
            return
        soul = agent_dir / "SOUL.md"
        if soul.is_file():
            shutil.copy2(soul, staged_agent / soul.name)
        previous = Path(temp) / "previous"
        os.replace(agent_dir, previous)
        try:
            os.replace(staged_agent, agent_dir)
        except OSError:
            os.replace(previous, agent_dir)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--database-seed", type=Path, required=True)
    parser.add_argument("--expected-did", required=True)
    args = parser.parse_args()
    restore_demo_seed(args.agent_dir, args.database_seed, expected_did=args.expected_did)


if __name__ == "__main__":
    main()
