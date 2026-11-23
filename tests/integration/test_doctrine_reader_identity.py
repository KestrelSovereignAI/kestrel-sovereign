"""Descriptor identity and whole-receipt acceptance for #4220 rv-4220-2."""

import ast
from collections import OrderedDict
import hashlib
import inspect
import os
from pathlib import Path, PureWindowsPath
import subprocess
from types import SimpleNamespace

import pytest

from kestrel_sovereign.agent import doctrine_bundle as doctrine
from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.agent.system_prompt_assembler import assemble_system_prompt
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sdk.signals import Status
from tests.integration.test_doctrine_reader_receipts import (
    ARCHIVE, DID, OLD, ROOT, _agent, _legacy_checkout, _persist_legacy, _read, _write,
)
from tests.unit.test_signals_dispatcher_constitution_injection import (
    _AuditingAgent, _cognition_reg, _drain, _make_dispatcher, _signal,
)


def _literal(root, extras):
    return doctrine.compute_doctrine_bundle_hash(
        anchored_files=doctrine.resolve_anchored_paths(project_root=root, extra_paths=extras),
        bootstrap_files=OrderedDict(),
    )


def _upgrade(root):
    _write(root, ARCHIVE, (root / OLD).read_bytes())
    _write(root, OLD, (ROOT / OLD).read_bytes())


async def _assert_dispatcher_refuses(root, storage, expected, literal):
    agent = _agent(storage, root)
    live = await ConstitutionMixin.compute_live_doctrine_bundle_hash(agent)
    assert live is not None and live == literal.hash and live != expected
    observer = _AuditingAgent(anchored_bundle_hash=expected)
    async def native_live():
        return await ConstitutionMixin.compute_live_doctrine_bundle_hash(agent)
    observer.compute_live_doctrine_bundle_hash = native_live
    env = await _make_dispatcher(root, observer)
    template = root / "signal-template.md"
    template.write_text("{payload}", encoding="utf-8")
    env.registry.register(_cognition_reg(template, constitution_injection="full"))
    try:
        result = await env.dispatcher.dispatch_signal(_signal("cog_src"))
        assert result.status == Status.DROPPED_VALIDATION
        assert "doctrine_bundle_drift" in (result.error or "")
        assert observer.process_input_calls == []
    finally:
        await _drain(env)
        await env.dispatcher.shutdown_durable_delivery()
        await env.backend.close()


@pytest.mark.asyncio
async def test_full_prompt_reads_historical_bytes_under_configured_name(tmp_path):
    root = tmp_path / "checkout"
    _legacy_checkout(root, "relative")
    alias = root / "my-constitution.md"
    alias.symlink_to(OLD)
    original = (root / OLD).read_bytes() + b"\r\nLocal doctrine.\r\n"
    _write(root, OLD, original)
    before_text = alias.read_text(encoding="utf-8")
    bootstrap = OrderedDict([("SOUL.md", "Agent bootstrap")])
    db = tmp_path / "agent.db"
    before, properties = await _persist_legacy(db, root, [alias.name], bootstrap_files=bootstrap)
    _upgrade(root)

    async with AsyncStorage(str(db), agent_id=DID) as storage:
        agent = _agent(storage, root)
        agent.context_builder._bootstrap_files = bootstrap
        assert await ConstitutionMixin.compute_live_doctrine_bundle_hash(agent) == before.hash
        prompt_files = await ConstitutionMixin.get_anchored_doctrine_files(agent)
        assembled = assemble_system_prompt(
            constitution="Agent constitution", bootstrap_files=bootstrap, anchored_doctrine=prompt_files,
        )
        assert before_text.strip() in dict(assembled.subsections)[alias.name]
        assert "Local doctrine." in assembled.prompt
        assert prompt_files[alias.name] == before_text
        assert "KESTREL_CONSTITUTION.md" not in prompt_files
        assert (await storage.get_node(DID)).properties == properties


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["verify", "anchor", "live"])
@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
@pytest.mark.parametrize("layout", ["minimal", "stock"])
async def test_a1_n2_alias_keeps_its_own_logical_name(tmp_path, caplog, entry, alias, layout):
    root = tmp_path / "checkout"
    _legacy_checkout(root, "relative")
    if layout == "minimal":
        (root / "AGENTS.md").unlink()
        (root / "docs/TORTOISE_DOCTRINE.md").unlink()
    extra = root / "my-constitution.md"
    if alias == "symlink":
        extra.symlink_to(OLD)
    else:
        extra.hardlink_to(root / OLD)
    before, properties = await _persist_legacy(tmp_path / "agent.db", root, [extra.name])
    expected = ("1f27de2964ad174c052372b89ff328897ad4cd6aa661669479044ff14abc50e4"
                if layout == "minimal" else
                "316b6de5260803d69cdebb88cfad21bddfecdd9c72a2c1bad572286a81f23cba")
    assert before.hash == expected
    _upgrade(root)
    async with AsyncStorage(str(tmp_path / "agent.db"), agent_id=DID) as storage:
        assert await _read(_agent(storage, root), root, entry) == before.hash
        assert (await storage.get_node(DID)).properties == properties
        verified = await doctrine.verify_doctrine_bundle(
            _agent(storage, root), project_root=root, bootstrap_files=OrderedDict(),
        )
        assert verified.files == _literal(root, [extra.name]).files
        assert verified.files[-1] == str(extra)
        assert verified.total_bytes == before.total_bytes
        mappings = next(
            record.args[1] for record in reversed(caplog.records)
            if "historical reader mapping v1 verified" in record.msg
        )
        assert mappings[-1] == (extra.name, str(extra), str(root / ARCHIVE))


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [OLD, ARCHIVE])
async def test_alias_tamper_returns_literal_and_dispatcher_refuses(tmp_path, changed):
    root = tmp_path / "checkout"
    _legacy_checkout(root, "relative")
    (root / "my-constitution.md").symlink_to(OLD)
    extras = ["my-constitution.md"]
    before, _ = await _persist_legacy(tmp_path / "agent.db", root, extras)
    _upgrade(root)
    _write(root, changed, (root / changed).read_bytes() + b"!")
    literal = _literal(root, extras)
    async with AsyncStorage(str(tmp_path / "agent.db"), agent_id=DID) as storage:
        agent = _agent(storage, root)
        with pytest.raises(doctrine.DoctrineBundleDriftError) as refused:
            await doctrine.verify_doctrine_bundle(agent, project_root=root, bootstrap_files=OrderedDict())
        assert refused.value.live_hash == literal.hash
        await _assert_dispatcher_refuses(root, storage, before.hash, literal)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", [b"\n", b"\r\n"], ids=["LF", "CRLF"])
@pytest.mark.parametrize("name", ["my-constitution.md", "KESTREL_CONSTITUTION.md"])
async def test_n1_a2_copied_wrapper_is_not_an_alias_through_dispatcher(tmp_path, ending, name):
    root = tmp_path / "checkout"
    _legacy_checkout(root, "relative")
    _write(root, name, (root / OLD).read_bytes())
    before, _ = await _persist_legacy(tmp_path / "agent.db", root, [name])
    _upgrade(root)
    _write(root, name, (ROOT / OLD).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending))
    literal = _literal(root, [name])
    async with AsyncStorage(str(tmp_path / "agent.db"), agent_id=DID) as storage:
        await _assert_dispatcher_refuses(root, storage, before.hash, literal)


@pytest.mark.asyncio
async def test_c1_self_referencing_bookmark_keeps_live_hash_and_refuses(tmp_path):
    root = tmp_path / "checkout"
    extras = _legacy_checkout(root, "relative")
    before, _ = await _persist_legacy(tmp_path / "agent.db", root, extras)
    _upgrade(root)
    (root / OLD).unlink()
    (root / OLD).symlink_to(Path(OLD).name)
    literal = _literal(root, extras)
    async with AsyncStorage(str(tmp_path / "agent.db"), agent_id=DID) as storage:
        await _assert_dispatcher_refuses(root, storage, before.hash, literal)


@pytest.mark.asyncio
async def test_selected_open_error_retains_literal_even_if_candidate_would_match(tmp_path):
    root = tmp_path / "checkout"
    extras = _legacy_checkout(root, "relative") + ["unreadable.md"]
    (root / "unreadable.md").mkdir()
    before, _ = await _persist_legacy(tmp_path / "agent.db", root, extras)
    _upgrade(root)
    async with AsyncStorage(str(tmp_path / "agent.db"), agent_id=DID) as storage:
        await _assert_dispatcher_refuses(root, storage, before.hash, _literal(root, extras))


@pytest.mark.asyncio
async def test_bookmark_replaced_between_descriptor_reads_returns_literal(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    _legacy_checkout(root, "relative")
    (root / "my-constitution.md").symlink_to(OLD)
    extras = ["my-constitution.md"]
    before, _ = await _persist_legacy(tmp_path / "agent.db", root, extras)
    _upgrade(root)
    literal = _literal(root, extras)
    bookmark = root / OLD
    original_identity = bookmark.stat()
    real_fstat = os.fstat
    replaced = False

    def replace_after_read(fd):
        nonlocal replaced
        info = real_fstat(fd)
        if not replaced and (info.st_dev, info.st_ino) == (
            original_identity.st_dev, original_identity.st_ino,
        ):
            replacement = root / "replacement.md"
            replacement.write_bytes(bookmark.read_bytes())
            replacement.replace(bookmark)
            replaced = True
        return info

    monkeypatch.setattr(doctrine, "os", SimpleNamespace(fstat=replace_after_read))
    agent_db = tmp_path / "agent.db"
    async with AsyncStorage(str(agent_db), agent_id=DID) as storage:
        agent = _agent(storage, root)
        assert await ConstitutionMixin.compute_live_doctrine_bundle_hash(agent) == literal.hash
        assert replaced


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["zero", "missing", "error"])
async def test_a3_unavailable_descriptor_identity_retains_literal(tmp_path, monkeypatch, failure):
    root = tmp_path / "checkout"
    extras = _legacy_checkout(root, "relative")
    before, _ = await _persist_legacy(tmp_path / "agent.db", root, extras)
    _upgrade(root)
    literal = _literal(root, extras)
    real_fstat = os.fstat
    def unavailable(fd):
        if failure == "error":
            raise OSError("identity unavailable")
        info = real_fstat(fd)
        return SimpleNamespace(st_dev=info.st_dev, st_ino=0 if failure == "zero" else None)
    # Only the compatibility module's os binding changes; SQLite retains real OS I/O.
    monkeypatch.setattr(doctrine, "os", SimpleNamespace(fstat=unavailable))
    async with AsyncStorage(str(tmp_path / "agent.db"), agent_id=DID) as storage:
        await _assert_dispatcher_refuses(root, storage, before.hash, literal)


@pytest.mark.asyncio
async def test_p1_moved_custom_archive_is_receipt_proven_then_tamper_refuses(tmp_path):
    root = tmp_path / "checkout"
    extras = _legacy_checkout(root, "relative")
    _write(root, OLD, (root / OLD).read_bytes() + b"\nOperator custom doctrine.\n")
    before, properties = await _persist_legacy(tmp_path / "agent.db", root, extras)
    _upgrade(root)
    async with AsyncStorage(str(tmp_path / "agent.db"), agent_id=DID) as storage:
        for entry in ("verify", "anchor", "live"):
            assert await _read(_agent(storage, root), root, entry) == before.hash
        assert (await storage.get_node(DID)).properties == properties
        _write(root, ARCHIVE, (root / ARCHIVE).read_bytes() + b"!")
        await _assert_dispatcher_refuses(root, storage, before.hash, _literal(root, extras))


@pytest.mark.asyncio
async def test_order_duplicates_and_missing_selection_are_retained(tmp_path):
    root = tmp_path / "checkout"
    _legacy_checkout(root, "relative")
    extras = [OLD, "AGENTS.md", "missing.md", OLD]
    before, properties = await _persist_legacy(tmp_path / "agent.db", root, extras)
    _upgrade(root)
    async with AsyncStorage(str(tmp_path / "agent.db"), agent_id=DID) as storage:
        agent = _agent(storage, root)
        for entry in ("verify", "anchor", "live"):
            assert await _read(agent, root, entry) == before.hash
        literal = _literal(root, extras)
        result = doctrine.compute_doctrine_bundle_for_receipt(
            project_root=root, extra_paths=extras, bootstrap_files=OrderedDict(), anchored_hash=before.hash,
        )
        assert result.files == literal.files and len(result.files) == 6
        node = await storage.get_node(DID)
        node.properties["doctrine_anchored_paths"] = ["AGENTS.md", OLD, "missing.md", OLD]
        await storage.add_node(node)
        with pytest.raises(doctrine.DoctrineBundleDriftError):
            await doctrine.verify_doctrine_bundle(agent, project_root=root, bootstrap_files=OrderedDict())
        assert properties["doctrine_anchored_paths"] == extras


def test_unreadable_archive_metadata_retains_the_archive_selection(tmp_path, monkeypatch):
    real_lstat = Path.lstat
    archive = tmp_path / ARCHIVE

    def denied_for_archive(path):
        if path == archive:
            raise PermissionError("archive metadata unavailable")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", denied_for_archive)
    paths = doctrine.resolve_anchored_paths(project_root=tmp_path)
    assert paths[0] == archive
    snapshot = doctrine.compute_doctrine_bundle_for_receipt(
        project_root=tmp_path, extra_paths=[], anchored_hash="previous-receipt",
        bootstrap_files=OrderedDict([("SOUL.md", "bootstrap")]),
    )
    assert snapshot.hash and snapshot.files == ["SOUL.md"]


def test_10_no_path_comparisons_or_resolution_in_compatibility():
    source = inspect.getsource(doctrine)
    tree = ast.parse(source)
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if isinstance(call.func, ast.Attribute):
            assert call.func.attr not in {"resolve", "realpath", "samefile", "stat"}
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "read_doctrine_bundle_for_receipt")
    for comparison in (node for node in ast.walk(function) if isinstance(node, ast.Compare)):
        compared_names = {node.id for node in ast.walk(comparison) if isinstance(node, ast.Name)}
        assert not compared_names.intersection({"path", "paths", "project_root", "bookmark", "archive", "source"})
    helper = inspect.getsource(doctrine._read_doctrine_identity)
    assert helper.count('.open(') == 1
    assert 'handle.read()' in helper and 'os.fstat(handle.fileno())' in helper


def test_11_windows_logical_names_match_receipt_era_derivation():
    path = PureWindowsPath(r"C:\\checkout\\my-constitution.md")
    data = b"operator bytes\r\n"
    snapshot = doctrine._hash_doctrine_contents([(path, data)], OrderedDict())
    section = doctrine._file_section(path, path.name, data)
    assert snapshot.hash == hashlib.sha256(section).hexdigest()
    assert path.name == "my-constitution.md" and snapshot.files == [str(path)]


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True], ids=["stock", "explicit"])
async def test_4_real_git_autocrlf_checkout_preserves_old_receipt(tmp_path, explicit):
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
    git = ["git", "-C", str(root), "-c", "core.autocrlf=true"]

    # Rebuild the receipt-era index from the archived blob in this checkout.
    # This also runs in depth-one CI checkouts and source distributions.
    paths = [OLD, "AGENTS.md", "docs/TORTOISE_DOCTRINE.md"]
    for logical in paths:
        source = ARCHIVE if logical == OLD else logical
        _write(root, logical, (ROOT / source).read_bytes())
    subprocess.run([*git, "add", "--", *paths], check=True, capture_output=True)
    for logical in paths:
        (root / logical).unlink()
    subprocess.run([*git, "checkout-index", "--all"], check=True, capture_output=True)
    assert b"\r\n" in (root / OLD).read_bytes()
    extras = [OLD] if explicit else []
    before, _ = await _persist_legacy(tmp_path / "agent.db", root, extras)
    _write(root, ARCHIVE, (root / OLD).read_bytes())
    # Materialize the new wrapper with Git's actual checkout conversion as well.
    _write(root, OLD, (ROOT / OLD).read_bytes())
    subprocess.run([*git, "add", "--", OLD], check=True, capture_output=True)
    (root / OLD).unlink()
    subprocess.run([*git, "checkout-index", "--", OLD], check=True, capture_output=True)
    assert b"\r\n" in (root / OLD).read_bytes()
    async with AsyncStorage(str(tmp_path / "agent.db"), agent_id=DID) as storage:
        for entry in ("verify", "anchor", "live"):
            assert await _read(_agent(storage, root), root, entry) == before.hash


@pytest.mark.asyncio
async def test_5_git_symlinks_false_materialized_link_stays_literal(tmp_path, caplog):
    root = tmp_path / "checkout"
    _legacy_checkout(root, "relative")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
    git = ["git", "-C", str(root), "-c", "core.symlinks=false"]
    blob = subprocess.check_output([*git, "hash-object", "-w", "--stdin"], input=OLD.encode()).decode().strip()
    subprocess.run([*git, "update-index", "--add", "--cacheinfo", "120000", blob, "my-constitution.md"], check=True, capture_output=True)
    subprocess.run([*git, "checkout-index", "--force", "--all"], check=True, capture_output=True)
    assert not (root / "my-constitution.md").is_symlink()
    assert (root / "my-constitution.md").read_text() == OLD
    before, properties = await _persist_legacy(tmp_path / "agent.db", root, ["my-constitution.md"])
    _upgrade(root)
    async with AsyncStorage(str(tmp_path / "agent.db"), agent_id=DID) as storage:
        caplog.clear()
        for entry in ("verify", "anchor", "live"):
            assert await _read(_agent(storage, root), root, entry) == before.hash
        assert (await storage.get_node(DID)).properties == properties
        assert "historical reader mapping" not in caplog.text
