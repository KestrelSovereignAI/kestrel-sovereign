"""Existing SQLite doctrine receipts survive the public bookmark replacement."""

from collections import OrderedDict
from copy import deepcopy
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.agent.doctrine_bundle import (
    DoctrineBundleDriftError,
    DoctrineBundleError,
    anchor_doctrine_bundle,
    compute_doctrine_bundle_hash,
    resolve_anchored_paths,
    verify_doctrine_bundle,
)
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode


ROOT = Path(__file__).resolve().parents[2]
OLD = "docs/principles/KESTREL_CONSTITUTION.md"
ARCHIVE = "doctrine/constitutions/v1/KESTREL_CONSTITUTION.md"
DID = "did:test:doctrine-reader-receipt"


def _write(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _legacy_checkout(root, selection):
    _write(root, OLD, (ROOT / ARCHIVE).read_bytes())
    for path in ("AGENTS.md", "docs/TORTOISE_DOCTRINE.md"):
        _write(root, path, (ROOT / path).read_bytes())
    if selection == "relative":
        return [OLD]
    if selection == "absolute":
        return [str(root / OLD)]
    parent_spelling = "docs/principles/../principles/KESTREL_CONSTITUTION.md"
    if selection == "relative-parent":
        return [parent_spelling]
    if selection == "absolute-parent":
        return [str(root / parent_spelling)]
    if selection == "system-alias":
        if sys.platform != "darwin" or not root.is_relative_to("/private/var"):
            pytest.skip("requires the macOS /var alias for /private/var")
        alias = Path("/var") / root.relative_to("/private/var")
        assert (alias / OLD).samefile(root / OLD)
        return [str(alias / OLD)]
    alias = root.with_name("checkout-alias")
    alias.symlink_to(root, target_is_directory=True)
    assert (alias / OLD).samefile(root / OLD)
    if selection == "directory-alias":
        return [str(alias / OLD)]
    _write(root, "between.md", b"Between the repeated selections.\n")
    return [OLD, "between.md", str(root / parent_spelling), str(alias / OLD), OLD]


def _agent(storage, root):
    return SimpleNamespace(
        agent_id=DID, storage=storage,
        _resolve_project_root_for_doctrine=AsyncMock(return_value=root),
        context_builder=SimpleNamespace(_bootstrap_files=OrderedDict()),
    )


async def _read(agent, root, entry):
    if entry == "live":
        return await ConstitutionMixin.compute_live_doctrine_bundle_hash(agent)
    operation = anchor_doctrine_bundle if entry == "anchor" else verify_doctrine_bundle
    return (await operation(agent, project_root=root, bootstrap_files=OrderedDict())).hash


async def _persist_legacy(db, root, extras):
    async with AsyncStorage(str(db), agent_id=DID) as storage:
        await storage.add_node(GraphNode(
            node_id=DID, node_type="agent", label="Existing doctrine receipt",
            properties={"doctrine_anchored_paths": extras, "operator_note": "unchanged"},
        ))
        before = await anchor_doctrine_bundle(
            _agent(storage, root), project_root=root, bootstrap_files=OrderedDict(),
        )
        assert await verify_doctrine_bundle(
            _agent(storage, root), project_root=root, bootstrap_files=OrderedDict(),
        ) == before
        return before, deepcopy((await storage.get_node(DID)).properties)


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", [False, True], ids=["stock", "custom-archive"])
@pytest.mark.parametrize("selection", [
    "relative", "absolute", "relative-parent", "absolute-parent",
    "directory-alias", "system-alias", "repeated",
])
@pytest.mark.parametrize("entry", ["verify", "anchor", "live"])
async def test_existing_receipt_verifies_wrapper_alias_without_rewriting(tmp_path, caplog, selection, entry, custom):
    root = tmp_path / "checkout"
    extras = _legacy_checkout(root, selection)
    if custom:
        _write(root, OLD, (root / OLD).read_bytes() + b"\nOriginal operator doctrine.\r\n")
    db = tmp_path / "agent.db"
    before, properties = await _persist_legacy(db, root, extras)
    if selection != "repeated" and not custom:
        assert before.hash == "ccb76492c7c1e083aae5c7498998cb430f0a1ef20b24458ed9a0e79f55991118"
    _write(root, ARCHIVE, (root / OLD).read_bytes())
    _write(root, OLD, (ROOT / OLD).read_bytes())
    literal = compute_doctrine_bundle_hash(
        anchored_files=resolve_anchored_paths(project_root=root, extra_paths=extras),
        bootstrap_files=OrderedDict(),
    )
    assert literal.hash != before.hash
    async with AsyncStorage(str(db), agent_id=DID) as storage:
        caplog.set_level(logging.WARNING)
        assert await _read(_agent(storage, root), root, entry) == before.hash
        assert "historical reader mapping v1" in caplog.text
        assert str(root / ARCHIVE) in caplog.text
        for path in before.files[3:]:
            if Path(path).resolve() == (root / OLD).resolve():
                assert path in caplog.text, "report the operator's original spelling"
        assert (await storage.get_node(DID)).properties == properties
        verified = await verify_doctrine_bundle(
            _agent(storage, root), project_root=root, bootstrap_files=OrderedDict(),
        )
        expected_files = [
            str(root / ARCHIVE) if Path(path).resolve() == (root / OLD).resolve() else path
            for path in before.files
        ]
        assert verified.files == expected_files, "preserve every selection in its ordered position"
        assert verified.total_bytes == before.total_bytes

    # New selections have no historical receipt: the wrapper itself is selected.
    async with AsyncStorage(str(tmp_path / "new-agent.db"), agent_id=DID) as storage:
        await storage.add_node(GraphNode(
            node_id=DID, node_type="agent", label="New explicit selection",
            properties={"doctrine_anchored_paths": extras},
        ))
        caplog.clear()
        assert await _read(_agent(storage, root), root, "anchor") == literal.hash
        assert await _read(_agent(storage, root), root, entry) == literal.hash
        assert "historical reader mapping v1" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["verify", "anchor", "live"])
@pytest.mark.parametrize("selection", [
    "relative", "relative-parent", "absolute-parent", "directory-alias", "system-alias",
])
@pytest.mark.parametrize("tamper", [OLD, ARCHIVE, "AGENTS.md", "bootstrap", "missing-archive", "external-lookalike"])
async def test_existing_receipt_alias_still_detects_each_changed_input(tmp_path, entry, tamper, selection):
    root = tmp_path / "checkout"
    extras = _legacy_checkout(root, selection)
    db = tmp_path / "agent.db"
    before, properties = await _persist_legacy(db, root, extras)
    _write(root, ARCHIVE, (root / OLD).read_bytes())
    _write(root, OLD, (ROOT / OLD).read_bytes())
    async with AsyncStorage(str(db), agent_id=DID) as storage:
        agent = _agent(storage, root)
        assert await _read(agent, root, entry) == before.hash
        if tamper == "missing-archive":
            (root / ARCHIVE).unlink()
        elif tamper == "bootstrap":
            agent.context_builder._bootstrap_files["SOUL.md"] = "Changed bootstrap"
        elif tamper == "external-lookalike":
            outside = tmp_path / "elsewhere" / "KESTREL_CONSTITUTION.md"
            _write(outside.parent, outside.name, (ROOT / OLD).read_bytes())
            properties["doctrine_anchored_paths"] = [str(outside)]
            node = await storage.get_node(DID)
            node.properties = deepcopy(properties)
            await storage.add_node(node)
        else:
            _write(root, tamper, (root / tamper).read_bytes() + b"\nChanged input.\n")
        if entry == "live":
            assert await _read(agent, root, entry) != before.hash
        else:
            operation = anchor_doctrine_bundle if entry == "anchor" else verify_doctrine_bundle
            error = DoctrineBundleError if entry == "anchor" else DoctrineBundleDriftError
            with pytest.raises(error, match="drift"):
                await operation(agent, project_root=root, bootstrap_files=agent.context_builder._bootstrap_files)
        assert (await storage.get_node(DID)).properties == properties


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["verify", "anchor", "live"])
async def test_custom_literal_receipt_survives_stock_archive_then_detects_change(tmp_path, caplog, entry):
    root = tmp_path / "checkout"
    extras = _legacy_checkout(root, "repeated")
    custom = (root / OLD).read_bytes() + b"\nOperator's original additional doctrine.\r\n"
    _write(root, OLD, custom)
    db = tmp_path / "agent.db"
    before, properties = await _persist_legacy(db, root, extras)
    _write(root, ARCHIVE, (ROOT / ARCHIVE).read_bytes())
    async with AsyncStorage(str(db), agent_id=DID) as storage:
        agent = _agent(storage, root)
        caplog.set_level(logging.WARNING)
        assert await _read(agent, root, entry) == before.hash
        assert (root / OLD).read_bytes() == custom
        assert (await storage.get_node(DID)).properties == properties
        verified = await verify_doctrine_bundle(agent, project_root=root, bootstrap_files=OrderedDict())
        assert verified.files == before.files, "custom operator selections stay literal"
        _write(root, OLD, custom + b"Changed custom input.\n")
        if entry == "live":
            assert await _read(agent, root, entry) != before.hash
        else:
            with pytest.raises(DoctrineBundleError, match="drift"):
                await _read(agent, root, entry)
        assert (await storage.get_node(DID)).properties == properties
