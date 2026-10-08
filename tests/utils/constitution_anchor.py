"""Seed a project whose agents are anchored to given constitution hashes.

The constitution adoption gate (#3517) reads each agent's anchor the way
``kestrel doctor`` does: stock ``sqlite3`` over the agent's ``kestrel_prime.db``,
scoped by the ownership witness a real write lays down beside the row. These
helpers write that shape and nothing more, so a test can say "Emma is anchored
to X" without running inception.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

from kestrel_sovereign.multi_agent.config import (
    MULTI_AGENT_CONFIG_FILENAME,
    HostConfig,
    LocalAgentConfig,
    MultiAgentConfig,
)

#: Where the packaged constitution sits inside a source checkout.
PACKAGED_CONSTITUTION_RELPATH = Path("kestrel_sovereign/data/KESTREL_CONSTITUTION.md")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def agent_did(name: str) -> str:
    return f"did:test:{name}"


def write_agent_anchor(db_path: Path, *, name: str, constitution_hash: str) -> None:
    """An agent database whose agent node is anchored to ``constitution_hash``.

    Replaces any database already at ``db_path``.
    """
    did = agent_did(name)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.executescript(
            """
            CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT,
                properties TEXT
            );
            CREATE TABLE graph_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT
            );
            CREATE TABLE graph_node_owners (
                node_id TEXT NOT NULL,
                agent_id TEXT NOT NULL
            );
            CREATE TABLE graph_edge_owners (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                label TEXT NOT NULL,
                agent_id TEXT NOT NULL
            );
            CREATE TABLE schema_backfills (
                name TEXT PRIMARY KEY,
                completed_at TIMESTAMP
            );
            """
        )
        conn.execute(
            "INSERT INTO graph_nodes(node_id, node_type, label, properties) "
            "VALUES (?, 'agent', ?, ?)",
            (did, name, json.dumps({"name": name, "constitution_hash": constitution_hash})),
        )
        conn.execute("INSERT INTO schema_backfills VALUES ('ownership_2649', NULL)")
        conn.execute(
            "INSERT INTO graph_node_owners(node_id, agent_id) VALUES (?, ?)",
            (did, did),
        )
        conn.execute(
            "INSERT INTO graph_edges(source_id, target_id, label, properties) "
            "VALUES (?, ?, 'governed_by', NULL)",
            (did, constitution_hash),
        )
        conn.execute(
            "INSERT INTO graph_edge_owners(source_id, target_id, label, agent_id) "
            "VALUES (?, ?, 'governed_by', ?)",
            (did, constitution_hash, did),
        )
        conn.commit()


def seed_anchored_agents(project_dir: Path, anchors: dict[str, str]) -> None:
    """Register local agents, each anchored to its hash in ``anchors``."""
    agents = {}
    for port, (name, constitution_hash) in enumerate(anchors.items(), start=8801):
        data_dir = Path("agent_data") / name.lower()
        agents[name] = LocalAgentConfig(data_dir=data_dir, port=port, autostart=True)
        write_agent_anchor(
            project_dir / data_dir / "kestrel_prime.db",
            name=name,
            constitution_hash=constitution_hash,
        )
    project_dir.mkdir(parents=True, exist_ok=True)
    MultiAgentConfig(host=HostConfig(), agents=agents).save(
        project_dir / MULTI_AGENT_CONFIG_FILENAME
    )


def git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` with a hermetic identity; return stdout."""
    result = subprocess.run(
        [
            "git",
            "-c", "user.name=Kestrel Test",
            "-c", "user.email=test@kestrel.invalid",
            "-c", "commit.gpgsign=false",
            "-c", "core.autocrlf=false",
            *args,
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def commit_constitution(repo: Path, content: bytes, message: str) -> str:
    """Commit ``content`` as the checkout's packaged constitution."""
    path = repo / PACKAGED_CONSTITUTION_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    git(repo, "add", str(PACKAGED_CONSTITUTION_RELPATH))
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


def origin_and_clone(tmp_path: Path, content: bytes) -> tuple[Path, Path]:
    """An ``origin`` repository and a clone of it tracking ``origin/main``.

    Both start with ``content`` as the packaged constitution. Commit to
    ``origin`` to model an upstream change the clone has not pulled yet.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    commit_constitution(origin, content, "initial constitution")
    clone = tmp_path / "checkout"
    git(tmp_path, "clone", "-q", str(origin), str(clone))
    return origin, clone
