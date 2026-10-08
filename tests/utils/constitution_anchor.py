"""Seed a project whose agents are anchored to given constitution hashes.

The constitution adoption gate (#3517) reads each agent's anchor the way
``kestrel doctor`` does: stock ``sqlite3`` over the agent's ``kestrel_prime.db``,
scoped by the ownership witness a real write lays down beside the row. These
helpers write that shape and nothing more, so a test can say "Emma is anchored
to X" without running inception.

The gate judges installed code in a fresh interpreter, which no in-process
patch reaches. :func:`link_installed_package` builds the ``kestrel_sovereign``
that interpreter imports, with its own packaged constitution and resolver.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import kestrel_sovereign
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


#: Appended to an installed resolver to model a release that renders the
#: governing constitution differently without changing its text.
_RENDERING_CHANGE = """

_released_resolve_governing_constitution_bytes = resolve_governing_constitution_bytes


def resolve_governing_constitution_bytes(*args, **kwargs):
    return _released_resolve_governing_constitution_bytes(*args, **kwargs) + {suffix!r}
"""


@dataclass(frozen=True)
class InstalledPackage:
    """A ``kestrel_sovereign`` on disk for a fresh interpreter to import.

    ``root`` goes on the child's ``PYTHONPATH``. Every module links to the
    package this test process imported, except the resolver, which is a copy
    the test may change, and the packaged constitution, which links to
    ``constitution``: the one file both the in-process and fresh views read.
    """

    root: Path
    constitution: Path

    @property
    def resolver(self) -> Path:
        return self.root / "kestrel_sovereign" / "constitution" / "resolver.py"

    def change_rendering(self, suffix: bytes) -> None:
        """Install a resolver whose governing bytes end with ``suffix``."""
        with self.resolver.open("a", encoding="utf-8") as handle:
            handle.write(_RENDERING_CHANGE.format(suffix=suffix))


def _link_tree(source: Path, target: Path, *, except_: frozenset[str]) -> None:
    target.mkdir(parents=True)
    for entry in source.iterdir():
        if entry.name != "__pycache__" and entry.name not in except_:
            (target / entry.name).symlink_to(entry)


def link_installed_package(root: Path, constitution: Path) -> InstalledPackage:
    """Build the installed package a fresh interpreter imports from ``root``.

    Point the child at it with ``PYTHONPATH=root`` and
    ``PYTHONDONTWRITEBYTECODE=1``: the linked modules' bytecode caches belong
    to the real package. POSIX only (symlinks).
    """
    real = Path(kestrel_sovereign.__file__).resolve().parent
    package = root / "kestrel_sovereign"
    _link_tree(real, package, except_=frozenset({"constitution", "data"}))
    _link_tree(
        real / "constitution",
        package / "constitution",
        except_=frozenset({"resolver.py"}),
    )
    shutil.copyfile(
        real / "constitution" / "resolver.py",
        package / "constitution" / "resolver.py",
    )
    _link_tree(
        real / "data",
        package / "data",
        except_=frozenset({PACKAGED_CONSTITUTION_RELPATH.name}),
    )
    (package / "data" / PACKAGED_CONSTITUTION_RELPATH.name).symlink_to(constitution)
    return InstalledPackage(root=root, constitution=constitution)
