"""Restore the pinned signing identity and initial SQLite state of a Cloud Run demo."""

from __future__ import annotations

import argparse
import json
import lzma
import os
import re
import sqlite3
import stat
import tempfile
import tomllib
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from kestrel_sovereign.a2a.native_sender import verification_document_fingerprint
from kestrel_sovereign.identity.custody_bundle import (
    IdentityCustodyError,
    restore_identity_bundle,
)
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.security.path_identity import paths_overlap_by_filesystem_identity
from kestrel_sovereign.storage.db.interface import ConnectionError

MAX_DATABASE_BYTES = 8 * 1024 * 1024
DEMO_SEED_STATE_TABLE = "a2a_demo_seed_lifetime"


def demo_signing_key_fingerprint(identity) -> str:
    return verification_document_fingerprint({
        "id": identity.signing_did,
        "verificationMethod": identity.new_verification_methods,
    })


@dataclass(frozen=True)
class DemoSeed:
    directory: Path
    database: Path
    bundle: bytes
    did: str
    database_sha256: str
    signing_key_fingerprint: str
    soul_source: Path | None = None


@contextmanager
def _stage_demo_seed(seed: DemoSeed, tenant: str) -> Iterator[Path]:
    agent_dir = seed.directory.absolute()
    if agent_dir.is_symlink() or (
        agent_dir.exists()
        and any(
            path.name != "SOUL.md" or path.is_symlink() or not path.is_file()
            for path in agent_dir.iterdir()
        )
    ):
        raise IdentityCustodyError("Demo seed refuses to replace existing agent state")
    if seed.soul_source is not None and (agent_dir / "SOUL.md").exists():
        raise IdentityCustodyError("Demo seed refuses to replace the existing SOUL")

    agent_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".demo-restore-", dir=agent_dir.parent) as temp:
        staged_agent = Path(temp) / "agent"
        restore_identity_bundle(seed.bundle, staged_agent, expected_did=seed.did)

        database = staged_agent / "kestrel_prime.db"
        with seed.database.open("rb") as source:
            compressed = source.read(MAX_DATABASE_BYTES + 1)
        if len(compressed) > MAX_DATABASE_BYTES or sha256(compressed).hexdigest() != seed.database_sha256:
            raise IdentityCustodyError("Demo seed database differs from its pinned snapshot")
        with lzma.open(BytesIO(compressed), "rb") as source:
            contents = source.read(MAX_DATABASE_BYTES + 1)
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
        if integrity != [("ok",)] or demos != [(seed.did, 1, 1)]:
            raise IdentityCustodyError("Demo seed database does not bind to the pinned test identity")
        identity = load_agent_identity(None, storage_dir=staged_agent)
        fingerprint = demo_signing_key_fingerprint(identity)
        if fingerprint != seed.signing_key_fingerprint:
            raise IdentityCustodyError("Demo seed identity differs from its pinned signing keys")
        with closing(sqlite3.connect(database)) as db, db:
            db.execute(
                f"""CREATE TABLE {DEMO_SEED_STATE_TABLE} (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    tenant TEXT NOT NULL,
                    signing_keys TEXT NOT NULL
                )"""
            )
            db.execute(
                f"INSERT INTO {DEMO_SEED_STATE_TABLE} VALUES (1, ?, ?)",
                (tenant, fingerprint),
            )
        if seed.soul_source is not None:
            (staged_agent / "SOUL.md").write_bytes(seed.soul_source.read_bytes())
        yield staged_agent


def restore_demo_seeds(
    seeds: Iterable[DemoSeed], *, tenant: str, roster_update: tuple[Path, Path] | None = None,
) -> None:
    """Validate the whole roster before installing any demo into its runtime directory."""
    with ExitStack() as staging:
        prepared = [
            (seed.directory.absolute(), staging.enter_context(_stage_demo_seed(seed, tenant)))
            for seed in seeds
        ]
        staged_roster = None
        if roster_update is not None:
            template, runtime_config = roster_update
            runtime_config = runtime_config.resolve()
            if not (runtime_config.exists() and template.samefile(runtime_config)):
                directory = staging.enter_context(
                    tempfile.TemporaryDirectory(prefix=".demo-roster-", dir=runtime_config.parent),
                )
                staged_roster = Path(directory) / runtime_config.name
                staged_roster.write_bytes(template.read_bytes())
                if runtime_config.exists():
                    previous = runtime_config.stat()
                    current = staged_roster.stat()
                    if (current.st_uid, current.st_gid) != (previous.st_uid, previous.st_gid):
                        os.chown(staged_roster, previous.st_uid, previous.st_gid)
                    staged_roster.chmod(stat.S_IMODE(previous.st_mode))
        with ExitStack() as rollback:
            for agent_dir, staged_agent in prepared:
                if not agent_dir.exists():
                    os.replace(staged_agent, agent_dir)
                    rollback.callback(os.replace, agent_dir, staged_agent)
                    continue
                # A baked SOUL can leave an overlay directory on a lower layer.
                # Move only the validated upper-layer artifacts in that case.
                for artifact in staged_agent.iterdir():
                    destination = agent_dir / artifact.name
                    os.replace(artifact, destination)
                    rollback.callback(os.replace, destination, artifact)
            if staged_roster is not None:
                os.replace(staged_roster, runtime_config)
            rollback.pop_all()


def read_demo_seed_manifest(path: Path) -> dict:
    with path.open("rb") as source:
        return tomllib.load(source)


async def read_seeded_signing_key_fingerprint(
    backend, tenant: str, *, database_identity: tuple[int, int],
) -> str:
    """Only a database restored by this host lifetime can publish a seeded route."""
    if backend.backend_type != "sqlite":
        raise ValueError("A seeded demo requires the database restored for its boot tenant")
    try:
        backend.assert_connected_file_still_valid(expected_identity=database_identity)
    except ConnectionError as error:
        raise ValueError("A seeded demo requires its original connected SQLite file") from error
    if not await backend.fetch_one(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (DEMO_SEED_STATE_TABLE,),
    ):
        raise ValueError("A seeded demo requires the database restored for its boot tenant")
    record = await backend.fetch_one(f"SELECT tenant, signing_keys FROM {DEMO_SEED_STATE_TABLE}")
    if record is None or record[0] != tenant:
        raise ValueError("A seeded demo requires the database restored for its boot tenant")
    return record[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--agent-data-dir", type=Path, required=True)
    parser.add_argument("--host-control-dir", type=Path, required=True)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    if os.environ.get("KESTREL_DB_BACKEND") != "sqlite":
        raise IdentityCustodyError("Pinned demo seeds require SQLite storage")
    config = MultiAgentConfig.load(
        args.config, auto_discover_fallback=False, runtime_base=Path.cwd()
    )
    roster = {name.strip() for name in os.environ.get("KESTREL_AGENTS", "").split(",")}
    seeds = read_demo_seed_manifest(args.seeds)
    if (
        not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) for name in roster)
        or roster != set(config.agents)
        or args.registration not in seeds
        or not set(seeds) <= roster
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
    protected = [args.agent_data_dir, args.host_control_dir, args.seeds]
    protected.extend(
        Path(seed[item]) for seed in seeds.values()
        for item in ("database", "identity", "soul") if item in seed
    )
    if key_file := os.environ.get("KESTREL_DEMO_SEED_DATA_KEY_FILE"):
        protected.append(Path(key_file))
    if any(paths_overlap_by_filesystem_identity(args.runtime_config, path) for path in protected):
        raise IdentityCustodyError("Runtime roster overlaps with demo or host custody")
    tenant = "demo-" + uuid4().hex
    restore_demo_seeds((
        DemoSeed(
            directory=args.agent_data_dir / name,
            database=Path(seed["database"]),
            bundle=Path(seed["identity"]).read_bytes(),
            did=seed["did"],
            database_sha256=seed["database_sha256"],
            signing_key_fingerprint=seed["signing_key_fingerprint"],
            soul_source=Path(seed["soul"]) if "soul" in seed else None,
        )
        for name, seed in seeds.items()
    ), tenant=tenant, roster_update=(args.config, args.runtime_config))
    database_identities = {}
    for name in seeds:
        status = (args.agent_data_dir / name / "kestrel_prime.db").lstat()
        database_identities[name] = (status.st_dev, status.st_ino)
    print(f"{tenant}\t{json.dumps(database_identities, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
