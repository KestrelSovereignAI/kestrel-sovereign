"""Restore the pinned signing identity and initial SQLite state of a Cloud Run demo."""

from __future__ import annotations

import argparse
import lzma
import os
import re
import sqlite3
import tempfile
import tomllib
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from kestrel_sovereign.identity.custody_bundle import (
    IdentityCustodyError,
    restore_identity_bundle,
)
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.security.path_identity import paths_overlap_by_filesystem_identity

MAX_DATABASE_BYTES = 8 * 1024 * 1024


def restore_demo_seed(
    agent_dir: Path, source: Path, bundle: bytes, *,
    expected_did: str, soul_source: Path | None = None,
) -> None:
    """Verify both pinned inputs before admitting a demo to the startup roster."""
    agent_dir = agent_dir.absolute()
    if agent_dir.is_symlink() or (
        agent_dir.exists()
        and any(
            path.name != "SOUL.md" or path.is_symlink() or not path.is_file()
            for path in agent_dir.iterdir()
        )
    ):
        raise IdentityCustodyError("Demo seed refuses to replace existing agent state")
    if soul_source is not None and (agent_dir / "SOUL.md").exists():
        raise IdentityCustodyError("Demo seed refuses to replace the existing SOUL")

    agent_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".demo-restore-", dir=agent_dir.parent) as temp:
        staged_agent = Path(temp) / "agent"
        restore_identity_bundle(bundle, staged_agent, expected_did=expected_did)

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
        if soul_source is not None:
            (staged_agent / "SOUL.md").write_bytes(soul_source.read_bytes())

        if not agent_dir.exists():
            os.replace(staged_agent, agent_dir)
            return
        # A baked SOUL may leave the target directory on an overlay lower
        # layer. Move the validated upper-layer artifacts, never that directory.
        transferred = []
        try:
            for artifact in staged_agent.iterdir():
                destination = agent_dir / artifact.name
                os.replace(artifact, destination)
                transferred.append(destination)
        except OSError:
            for destination in reversed(transferred):
                os.replace(destination, staged_agent / destination.name)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--agent-data-dir", type=Path, required=True)
    parser.add_argument("--host-control-dir", type=Path, required=True)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    config = MultiAgentConfig.load(
        args.config, auto_discover_fallback=False, runtime_base=Path.cwd()
    )
    roster = {name.strip() for name in os.environ.get("KESTREL_AGENTS", "").split(",")}
    published = {
        name.strip() for name in os.environ.get("KESTREL_A2A_PUBLISHED_AGENTS", "").split(",")
    }
    with args.config.open("rb") as source:
        seeds = tomllib.load(source).get("demo_seeds", {})
    if (
        not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) for name in roster)
        or roster != set(config.agents)
        or args.registration not in seeds
        or published != set(seeds)
        or not published <= roster
        or not all(
            isinstance(agent, LocalAgentConfig)
            and agent.resolve_data_dir(Path.cwd()) == args.agent_data_dir / name
            for name, agent in config.agents.items()
        )
        or any(
            not config.agents[name].autostart
            or paths_overlap_by_filesystem_identity(args.host_control_dir, args.agent_data_dir / name)
            for name in seeds
        )
    ):
        raise IdentityCustodyError("Demo seed must bind to the configured discovery registration's data directory")
    for name, seed in seeds.items():
        restore_demo_seed(
            args.agent_data_dir / name, Path(seed["database"]),
            Path(seed["identity"]).read_bytes(), expected_did=seed["did"],
            soul_source=Path(seed["soul"]) if "soul" in seed else None,
        )
    print("demo-" + uuid4().hex)


if __name__ == "__main__":
    main()
