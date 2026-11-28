"""#4153: the read-only node-type census (rv-4135-1, "census before enforcement")."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import textwrap
from pathlib import Path

import pytest

from kestrel_sovereign.storage import node_type_census as census_module
from kestrel_sovereign.storage import node_type_registry as registry
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode
from kestrel_sovereign.storage.node_type_census import (
    CORE_OWNER,
    SITE_GRAPH_NODE,
    SITE_SQL,
    WriteSite,
    run_census,
    scan_write_sites,
)

ALICE = "did:example:alice"
BOB = "did:example:bob"

#: The ruled types: rv-4135-1's, with ``decision`` replaced by the
#: ``strategy_decision`` rv-4244-1 split from it. Their seeds follow the
#: rulings, not the census. ``decision`` is out of this set, so its seed is
#: checked against the census like any other (rv-4244-1 §5): this exemption
#: is what let its wrong v1 seed pass.
RULED = {
    "fact", "strategy_pattern", "strategy_blocker", "strategy_decision",
    "learned_fact", "note", "excerpt", "stash",
}

#: Ruled types whose census shows more than one owner, each with the issue
#: that settles who owns it. Empty: every ruled type the census sees has one.
RULED_MULTI_OWNER: dict[str, int] = {}


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


@pytest.fixture
def source_tree(tmp_path) -> Path:
    """A package with one feature writer, two core writers and a dynamic site."""
    root = tmp_path / "src" / "kestrel_sovereign"
    _write(root, "__init__.py", "")
    _write(root, "features/__init__.py", "")
    _write(root, "features/base.py", "class Feature:\n    pass\n")
    _write(
        root,
        "features/demo/feature.py",
        """
        from kestrel_sovereign.features.base import Feature
        from kestrel_sovereign.storage.async_graph_store import GraphNode
        from .constants import DEMO_TYPE

        class DemoFeature(Feature):
            async def save(self, store):
                await store.add_node(GraphNode(node_id="n", node_type=DEMO_TYPE, label="l", properties={}))

            def describe(self):
                return GraphNode(node_id="n", node_type="demo_read_only", label="l", properties={})
        """,
    )
    _write(root, "features/demo/constants.py", 'DEMO_TYPE = "demo_note"\n')
    _write(
        root,
        "storage/core_writer.py",
        '''
        """Docstring mentioning INSERT INTO graph_nodes is not a write."""
        from kestrel_sovereign.storage.async_graph_store import GraphNode

        class Base:
            NODE_TYPE = "demo_core"

        class Writer(Base):
            async def write(self, db, store):
                await store.compare_and_swap_node("x", None, GraphNode("x", self.NODE_TYPE, "l", {}))
                await db.execute(
                    "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
                    "VALUES (?, 'demo_sql', ?, ?)",
                    ("a", "b", "c"),
                )

            async def rewrite(self, store, node):
                await store.add_node(GraphNode(node.node_id, node.node_type, node.label, {}))
        ''',
    )
    return root


def test_scan_lists_writer_call_sites(source_tree):
    sites = {(site.node_type, site.function): site for site in scan_write_sites(source_tree)}

    assert set(sites) == {
        ("demo_note", "DemoFeature.save"),
        ("demo_core", "Writer.write"),
        ("demo_sql", "Writer.write"),
        (None, "Writer.rewrite"),
    }
    note = sites[("demo_note", "DemoFeature.save")]
    assert (note.owning_feature, note.kind, note.path) == (
        "DemoFeature", SITE_GRAPH_NODE, "kestrel_sovereign/features/demo/feature.py",
    )
    assert note.module == "kestrel_sovereign.features.demo.feature"
    assert sites[("demo_core", "Writer.write")].owning_feature == CORE_OWNER
    assert sites[("demo_sql", "Writer.write")].kind == SITE_SQL
    assert sites[(None, "Writer.rewrite")].expression == "node.node_type"


@pytest.fixture
def handed_on_tree(tmp_path) -> Path:
    """Writers of nodes they did not build, beside the storage facades that forward them."""
    root = tmp_path / "src" / "kestrel_sovereign"
    _write(root, "__init__.py", "")
    _write(
        root,
        "storage/rewriters.py",
        """
        from dataclasses import replace

        from kestrel_sovereign.storage.async_graph_store import GraphNode

        async def read_modify_write(store, node_id):
            node = await store.get_node(node_id)
            node.properties["seen"] = True
            await store.add_node(node)

        async def pass_through(store, received):
            await store.compare_and_swap_node(received.node_id, None, received)

        async def by_keyword(store, received):
            await store.compare_and_swap_node(node_id="x", expected=None, new_node=received)

        async def through_getattr(store, received):
            swap = getattr(store, "compare_and_swap_node", None)
            await swap("x", None, received)

        async def from_a_builder(store, build):
            await store.adopt_shared_content_node(build())

        async def built_by_name(store):
            node = GraphNode("n", "demo_named", "l", {})
            await store.add_node(node)

        async def built_or_read(store, existing):
            node = GraphNode("n", "demo_built_or_read", "l", {})
            if existing:
                node = existing
            await store.add_node(node)

        async def closure_over_a_built_node(store):
            node = GraphNode("n", "demo_closure", "l", {})

            async def write():
                await store.add_node(node)

            await write()

        async def from_fields(store, fields):
            await store.add_node(GraphNode(**fields))
        """,
    )
    _write(
        root,
        "storage/async_storage.py",
        """
        from dataclasses import replace

        class AsyncStorage:
            async def add_node(self, node, *, capability=None):
                await self.graph.add_node(node)

            async def compare_and_swap_node(self, node_id, expected, new_node):
                new_node = replace(new_node, label="changed")
                return await self.graph.compare_and_swap_node(node_id, expected, new_node)

            async def restore(self, node):
                await self.graph.add_node(node)
        """,
    )
    _write(
        root,
        "storage/node_type_census.py",
        """
        async def not_a_writer(store, node):
            await store.add_node(node)
        """,
    )
    return root


def test_scan_lists_writes_of_nodes_it_did_not_build(handed_on_tree):
    """A rewrite, a pass-through or an alias is a dynamic site; a forwarder is not (#4153 review)."""
    sites = scan_write_sites(handed_on_tree)
    found = {(site.function, site.node_type, site.expression) for site in sites}

    assert found == {
        ("read_modify_write", None, "node"),
        ("pass_through", None, "received"),
        ("by_keyword", None, "received"),
        ("through_getattr", None, "received"),
        ("from_a_builder", None, "build()"),
        ("built_by_name", "demo_named", "'demo_named'"),
        ("built_or_read", "demo_built_or_read", "'demo_built_or_read'"),
        ("built_or_read", None, "node"),
        ("closure_over_a_built_node", "demo_closure", "'demo_closure'"),
        ("from_fields", None, "GraphNode(**fields)"),
        # A forwarder that rebinds its node, and a facade method no forwarder
        # entry names, are writers like any other.
        ("AsyncStorage.compare_and_swap_node", None, "new_node"),
        ("AsyncStorage.restore", None, "node"),
    }
    assert all(site.kind == SITE_GRAPH_NODE for site in sites)
    # The census module is not read, and a forwarder passing on its own node is not listed.
    assert census_module.CENSUS_MODULE not in {site.module for site in sites}
    assert ("AsyncStorage.add_node", None, "node") not in found


def test_forwarders_name_real_facades_passing_on_their_node(tmp_path, monkeypatch):
    """Each excluded forwarder exists and writes only the node it was handed.

    The facades are copied into a scratch tree, scanned once with the exclusion
    lifted and once with it, so a renamed or changed forwarder fails here
    rather than silently leaving a writer out of the census.
    """
    real_root = census_module.default_package_root()
    root = tmp_path / "src" / real_root.name
    modules = {module for module, _ in census_module.GRAPH_WRITE_FORWARDERS}
    for module in modules:
        relative = Path(*module.split(".")[1:]).with_suffix(".py")
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text((real_root / relative).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(census_module, "GRAPH_WRITE_FORWARDERS", frozenset())
    unexcluded = scan_write_sites(root)
    monkeypatch.undo()
    excluded = scan_write_sites(root)

    for module, function in census_module.GRAPH_WRITE_FORWARDERS:
        forwarded = [s for s in unexcluded if (s.module, s.function) == (module, function)]
        assert forwarded, (module, function)
        assert {s.expression for s in forwarded} <= {"node", "new_node"}, forwarded
    assert not [
        s for s in excluded if (s.module, s.function) in census_module.GRAPH_WRITE_FORWARDERS
    ]
    assert set(unexcluded) - set(excluded) == {
        s for s in unexcluded if (s.module, s.function) in census_module.GRAPH_WRITE_FORWARDERS
    }


async def _fixture_db(path: Path) -> None:
    database = await AsyncDatabase.sqlite(str(path))
    try:
        for agent, counts in ((ALICE, {"demo_note": 3, "demo_core": 1}), (BOB, {"demo_note": 2})):
            store = AsyncGraphStore(database, agent_id=agent)
            for node_type, count in counts.items():
                for index in range(count):
                    await store.add_node(
                        GraphNode(
                            node_id=f"{agent}:{node_type}:{index}",
                            node_type=node_type,
                            label=node_type,
                            properties={},
                        )
                    )
        await database.execute(
            "INSERT INTO graph_nodes (node_id, node_type, label, properties) VALUES (?, ?, ?, ?)",
            ("legacy:1", "demo_unowned", "legacy", "{}"),
        )
    finally:
        await database.close()


async def test_census_counts_by_type_and_agent(tmp_path, source_tree):
    db_path = tmp_path / "kestrel_prime.db"
    await _fixture_db(db_path)
    database = await AsyncDatabase.sqlite(str(db_path))
    try:
        report = await run_census(database, write_sites=scan_write_sites(source_tree))
    finally:
        await database.close()

    note = report.row("demo_note")
    assert note.total_rows == 5
    assert dict(note.rows_by_agent) == {ALICE: 3, BOB: 2}
    assert note.unowned_rows == 0
    assert [(s.path, s.function) for s in note.write_sites] == [
        ("kestrel_sovereign/features/demo/feature.py", "DemoFeature.save")
    ]
    assert note.census_owner == "DemoFeature"

    core = report.row("demo_core")
    assert (core.total_rows, dict(core.rows_by_agent)) == (1, {ALICE: 1})
    assert core.census_owner == CORE_OWNER

    unowned = report.row("demo_unowned")
    assert (unowned.total_rows, dict(unowned.rows_by_agent), unowned.unowned_rows) == (1, {}, 1)
    assert unowned.write_sites == ()
    assert unowned.owning_feature is None
    assert "demo_unowned" in report.unowned

    sql = report.row("demo_sql")
    assert sql.total_rows == 0 and sql.census_owner == CORE_OWNER
    assert [site.function for site in report.dynamic_sites] == ["Writer.rewrite"]

    # Registered types appear with their entry; the agent root rows do too.
    assert report.registry_present
    assert report.row("todo_item").registered.owning_feature == "TodoFeature"
    payload = report.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    rendered = report.render()
    assert "demo_note: 5 row(s)" in rendered
    assert f"{ALICE}=3, {BOB}=2" in rendered
    assert "owner: UNOWNED" in rendered


def _state(db_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    digests = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(db_path.parent.glob(db_path.name + "*"))
    }
    # Immutable, so this probe neither creates WAL sidecars nor reads through
    # them: nothing writes the database while a test holds it.
    connection = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )]
        counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }
    finally:
        connection.close()
    return digests, counts


def test_census_is_read_only(tmp_path, capsys):
    import asyncio

    db_path = tmp_path / "kestrel_prime.db"
    asyncio.run(_fixture_db(db_path))
    before = _state(db_path)

    assert census_module.main(["--sqlite", str(db_path), "--json"]) == 0

    assert _state(db_path) == before
    report = json.loads(capsys.readouterr().out)
    assert {row["node_type"] for row in report["types"]} >= {"demo_note", "demo_core", "demo_unowned"}


def test_census_reads_a_database_without_a_registry(tmp_path, capsys):
    """A database from before #4153 is reported, and is not migrated by the census."""
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            "CREATE TABLE graph_nodes (node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, "
            "label TEXT NOT NULL, properties TEXT);"
            "CREATE TABLE graph_node_owners (node_id TEXT NOT NULL, agent_id TEXT NOT NULL, "
            "PRIMARY KEY (node_id, agent_id));"
            "INSERT INTO graph_nodes VALUES ('h1', 'health_check', 'h', '{}');"
            "INSERT INTO graph_node_owners VALUES ('h1', 'did:example:alice');"
        )
        connection.commit()
    finally:
        connection.close()
    before = _state(db_path)

    assert census_module.main(["--sqlite", str(db_path)]) == 0

    assert _state(db_path) == before
    output = capsys.readouterr().out
    assert "no node-type registry table" in output
    assert "health_check: 1 row(s)" in output
    assert "owner: UNOWNED" in output


def test_census_refuses_a_missing_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        census_module.main(["--sqlite", str(tmp_path / "absent.db")])
    assert not (tmp_path / "absent.db").exists()


@pytest.fixture(scope="module")
def real_tree_sites():
    """The shipped tree's write sites, scanned once for the tests that read them."""
    return scan_write_sites()


def test_real_tree_write_sites(real_tree_sites):
    """Spot checks against the shipped source, including what #4135-5 must settle."""
    sites = real_tree_sites
    by_type: dict = {}
    for site in sites:
        by_type.setdefault(site.node_type, []).append(site)

    assert [(s.path, s.function) for s in by_type["todo_item"]] == [
        ("kestrel_sovereign/features/todo/feature.py", "TodoFeature._persist")
    ]
    # rv-4244-1 §5: one owner each, and the census finds one module for
    # ``decision``, which its seed names as its writer.
    assert {s.owning_feature for s in by_type["decision"]} == {CORE_OWNER}
    assert {s.module for s in by_type["decision"]} == {"kestrel_sovereign.storage.schema_router"}
    assert {s.owning_feature for s in by_type["strategy_decision"]} == {"StrategicMemoryFeature"}
    assert {s.module for s in by_type["strategy_decision"]} == {
        "kestrel_sovereign.features.strategic_memory.decision_index",
    }
    # The v2 seed migration retypes rows of both: a core site of no one type.
    assert (
        "kestrel_sovereign.storage.node_type_registry", "apply_seed_v2",
    ) in {(s.module, s.function) for s in by_type[None]}
    assert {s.module for s in by_type["feature_config"]} == {
        "kestrel_sovereign.features.base", "kestrel_sovereign.features.isolated_runtime",
    }
    # The generic store and the census itself are not writers.
    assert not {s.module for s in sites} & {
        census_module.GENERIC_GRAPH_STORE, census_module.CENSUS_MODULE,
    }
    # A GraphNode built only to be inspected is not a write.
    assert not any(s.module == "kestrel_sovereign.doctor" for s in sites)
    assert any(s.expression == "section.node_type" for s in by_type[None])

    # Writers of a node they read, or were handed, build no GraphNode: each is
    # a dynamic site. These rewrite the agent root (#4153 review).
    dynamic = {(s.module, s.function, s.expression) for s in by_type[None]}
    assert {
        ("kestrel_sovereign.agent.constitution",
         "ConstitutionMixin.ensure_doctrine_bundle_anchored", "agent_node"),
        ("kestrel_sovereign.agent.constitution",
         "ConstitutionMixin._read_governing_constitution", "agent_node"),
        ("kestrel_sovereign.agent.constitution",
         "ConstitutionMixin._persist_governance_receipt_node_owned.write", "node"),
        ("kestrel_sovereign.agent.doctrine_bundle", "anchor_doctrine_bundle", "agent_node"),
        ("kestrel_sovereign.agent.doctrine_bundle", "reanchor_doctrine_bundle", "agent_node"),
        ("kestrel_sovereign.identity.birth_record", "replicate_birth_record", "node"),
        ("kestrel_sovereign.identity.birth_record", "replicate_birth_record", "agent_node"),
    } <= dynamic
    # The storage facades are the write path: their callers are listed instead.
    assert not {(s.module, s.function) for s in sites} & census_module.GRAPH_WRITE_FORWARDERS


def _ruled_multi_owner_violations(sites, ruled, listed) -> dict:
    """The ruled types the census gives more than one owner, unless listed."""
    owners: dict = {}
    for site in sites:
        if site.node_type in ruled:
            owners.setdefault(site.node_type, set()).add(site.owning_feature)
    return {
        node_type: found
        for node_type, found in owners.items()
        if len(found) > 1 and node_type not in listed
    }


def test_a_ruled_type_has_one_owner_unless_listed(real_tree_sites):
    """rv-4244-1 §5: a ruled type with two owners fails unless an issue owns it.

    A ruled type's seed follows its ruling, not the census, so nothing else
    notices when a second owner writes it. ``decision`` had two that way.
    """
    assert _ruled_multi_owner_violations(real_tree_sites, RULED, RULED_MULTI_OWNER) == {}
    owners: dict = {}
    for site in real_tree_sites:
        owners.setdefault(site.node_type, set()).add(site.owning_feature)
    for node_type, issue in RULED_MULTI_OWNER.items():
        assert node_type in RULED, node_type
        assert isinstance(issue, int) and issue > 0, (node_type, issue)
        # A listed type that has one owner again comes off the list.
        assert len(owners.get(node_type, ())) > 1, node_type


def test_multi_owner_guard_fails_a_second_owner():
    """The guard fires on the shape it exists for, and a listing silences it."""

    def site(node_type, owner):
        return WriteSite("m", "m.py", 1, "f", SITE_GRAPH_NODE, node_type, "'t'", owner)

    sites = [
        site("ruled_type", "OneFeature"),
        site("ruled_type", CORE_OWNER),
        site("single_owner", CORE_OWNER),
        site("unruled", "OneFeature"),
        site("unruled", CORE_OWNER),
    ]
    ruled = {"ruled_type", "single_owner"}

    assert _ruled_multi_owner_violations(sites, ruled, {}) == {
        "ruled_type": {"OneFeature", CORE_OWNER},
    }
    assert _ruled_multi_owner_violations(sites, ruled, {"ruled_type": 4244}) == {}


def test_seeds_agree_with_the_census(real_tree_sites):
    """Every type the census attributes to one in-tree owner is seeded with that owner.

    A graph type's ``writer`` names every module with a site of that type, and
    any other module it names has a site the census could not type, under the
    same owner: a writer of a node it read or was handed.
    """
    by_type: dict = {}
    dynamic_by_owner: dict = {}
    for site in real_tree_sites:
        if site.node_type is None:
            dynamic_by_owner.setdefault(site.owning_feature, set()).add(site.module)
        else:
            by_type.setdefault(site.node_type, []).append(site)
    seeds = {entry.node_type: entry for entry in registry.SEED_ENTRIES}

    for node_type, type_sites in by_type.items():
        if node_type in RULED:
            continue
        owners = {site.owning_feature for site in type_sites}
        assert len(owners) == 1, (node_type, owners)
        entry = seeds[node_type]
        assert entry.owning_feature == owners.pop(), node_type
        if entry.canonical_home is registry.CanonicalHome.GRAPH:
            typed = {site.module for site in type_sites}
            assert typed <= set(entry.writer), node_type
            untyped = set(entry.writer) - typed
            assert untyped <= dynamic_by_owner.get(entry.owning_feature, set()), (node_type, untyped)
    assert set(seeds) - RULED == set(by_type) - RULED

    # The agent root's writers are mostly rewriters, so they appear only above.
    assert {
        "kestrel_sovereign.agent.constitution",
        "kestrel_sovereign.agent.doctrine_bundle",
        "kestrel_sovereign.identity.birth_record",
    } <= set(seeds["agent"].writer)


def test_census_cli_under_python_m_leaves_itself_out(tmp_path):
    """The documented ``python -m`` form excludes the census by its import name, not ``__main__``."""
    import os
    import subprocess
    import sys

    import kestrel_sovereign

    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            "CREATE TABLE graph_nodes (node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, "
            "label TEXT NOT NULL, properties TEXT);"
            "CREATE TABLE graph_node_owners (node_id TEXT NOT NULL, agent_id TEXT NOT NULL, "
            "PRIMARY KEY (node_id, agent_id));"
        )
        connection.commit()
    finally:
        connection.close()
    neutral = tmp_path / "cwd"
    neutral.mkdir()
    package_parent = Path(kestrel_sovereign.__file__).resolve().parent.parent

    completed = subprocess.run(
        [sys.executable, "-m", census_module.CENSUS_MODULE, "--sqlite", str(db_path), "--json"],
        cwd=neutral,
        env={**os.environ, "PYTHONPATH": str(package_parent)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert census_module.CENSUS_MODULE in report["unscanned_modules"]
    assert "__main__" not in report["unscanned_modules"]
    modules = {site["module"] for site in report["dynamic_write_sites"]} | {
        site["module"] for row in report["types"] for site in row["write_sites"]
    }
    assert census_module.CENSUS_MODULE not in modules
    assert census_module.CENSUS_MODULE == census_module.__name__


def test_static_feature_owners_match_feature_discovery():
    """The census names the same in-tree features discovery loads."""
    import importlib

    from kestrel_sovereign.features import discover_feature_modules, find_feature_class

    root = census_module.default_package_root()
    modules = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = census_module._index_module(".".join(parts), path)
    static = set(census_module._feature_owners(root, modules).values())
    discovered = {
        find_feature_class(importlib.import_module(module)).__name__
        for module in discover_feature_modules()
    }
    assert static == discovered
