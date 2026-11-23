"""Reader navigation must not change established constitutional lifetimes (#4220)."""

from collections import OrderedDict
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.agent.doctrine_bundle import (
    DoctrineBundleDriftError, anchor_doctrine_bundle, compute_doctrine_bundle_hash,
    resolve_anchored_paths, verify_doctrine_bundle,
)
from kestrel_sovereign.features.constitution import ConstitutionFeature
from scripts import build_docs_site, docs_okf, docs_verify, generate_feature_docs

ROOT = Path(__file__).resolve().parents[2]
OLD = "docs/principles/KESTREL_CONSTITUTION.md"
ARCHIVE = "doctrine/constitutions/v1/KESTREL_CONSTITUTION.md"
RELEASE = "kestrel_sovereign/data/constitutions/v2/KESTREL_CONSTITUTION.md"
RELEASE_URL = f"https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/constitution-v2.0.0/{RELEASE}"


def _original_bytes(path):
    if path == OLD and (ROOT / ARCHIVE).is_file():
        path = ARCHIVE
    return (ROOT / path).read_bytes()


def _write(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _snapshot(root, extras=None):
    return compute_doctrine_bundle_hash(
        anchored_files=resolve_anchored_paths(project_root=root, extra_paths=extras),
        bootstrap_files=OrderedDict([("SOUL.md", "A local personality.\n")]),
    )


def test_relocated_mirror_preserves_complete_git_blob_and_package_editions():
    # Full pre-change git blob at 6d373d6a, including YAML and line endings.
    assert hashlib.sha256((ROOT / ARCHIVE).read_bytes()).hexdigest() == "69628a59f849d0837e697d5a9248419bb7ba6506eddb0330c7bcc56b6832af00"
    for path, expected in (
        ("kestrel_sovereign/data/KESTREL_CONSTITUTION.md", "6948454b61253548b02b943dff166ba0c2f2eaabe2f542ace37424a8256d6467"),
        (RELEASE, "d0949dba4b1969bc11098f64d56d38490ea59ddcbe91c40b7775f5d506408908"),
    ):
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", [False, True], ids=["stock", "custom-bytes"])
async def test_mirror_move_preserves_bundle_and_stored_paths_then_detects_tamper(tmp_path, custom):
    for path in (OLD, "docs/TORTOISE_DOCTRINE.md", "AGENTS.md"):
        _write(tmp_path, path, _original_bytes(path))
    if custom:
        with (tmp_path / OLD).open("ab") as handle:
            handle.write(b"\nOperator's original custom doctrine.\n")
    before = _snapshot(tmp_path)
    node = SimpleNamespace(properties={
        "doctrine_bundle_hash": before.hash,
        "doctrine_bundle_files": before.files,
        "doctrine_bundle_anchored_at": "2026-11-01T00:00:00Z",
    })
    original_properties = deepcopy(node.properties)
    storage = SimpleNamespace(get_node=AsyncMock(return_value=node), compare_and_swap_node=AsyncMock())
    agent = SimpleNamespace(agent_id="did:test:reader", storage=storage)
    _write(tmp_path, ARCHIVE, (tmp_path / OLD).read_bytes())
    _write(tmp_path, OLD, (ROOT / OLD).read_bytes())
    after = await verify_doctrine_bundle(
        agent, project_root=tmp_path, bootstrap_files=OrderedDict([("SOUL.md", "A local personality.\n")]),
    )
    assert before.hash == after.hash
    assert before.total_bytes == after.total_bytes
    assert before.files[0] == str(tmp_path / OLD)
    assert after.files[0] == str(tmp_path / ARCHIVE)
    assert node.properties == original_properties, "verification must preserve historical path evidence"
    await anchor_doctrine_bundle(
        agent, project_root=tmp_path, bootstrap_files=OrderedDict([("SOUL.md", "A local personality.\n")]),
    )
    storage.compare_and_swap_node.assert_not_awaited()
    with (tmp_path / ARCHIVE).open("ab") as handle:
        handle.write(b"\nTampered relocated bytes.\n")
    with pytest.raises(DoctrineBundleDriftError) as refused:
        await verify_doctrine_bundle(
            agent, project_root=tmp_path, bootstrap_files=OrderedDict([("SOUL.md", "A local personality.\n")]),
        )
    assert refused.value.anchored_files == before.files
    assert node.properties == original_properties
    # The production mixin uses the same resolution, rather than reading the bookmark.
    agent._resolve_project_root_for_doctrine = AsyncMock(return_value=tmp_path)
    agent.context_builder = SimpleNamespace(_bootstrap_files=OrderedDict([("SOUL.md", "A local personality.\n")]))
    assert await ConstitutionMixin.compute_live_doctrine_bundle_hash(agent) == _snapshot(tmp_path).hash


def test_legacy_custom_checkout_and_explicit_old_path_remain_literal(tmp_path):
    custom = b"Custom first-edition source, including operator edits.\r\n"
    _write(tmp_path, OLD, custom)
    before = _snapshot(tmp_path, [OLD])
    assert resolve_anchored_paths(project_root=tmp_path, extra_paths=[OLD])[0] == tmp_path / OLD
    _write(tmp_path, ARCHIVE, custom)
    # Both the default and the explicit contribution remain, in their old order.
    after = _snapshot(tmp_path, [OLD])
    assert after.hash == before.hash
    paths = resolve_anchored_paths(project_root=tmp_path, extra_paths=[OLD, str(tmp_path / OLD)])
    assert paths[0] == tmp_path / ARCHIVE
    assert paths[-2:] == [tmp_path / OLD, tmp_path / OLD]
    # Updating the explicit file must still change the hash; no silent repointing.
    _write(tmp_path, OLD, b"Changed operator input")
    assert _snapshot(tmp_path, [OLD]).hash != before.hash
    assert (tmp_path / ARCHIVE).read_bytes() == custom


def test_installed_bundle_without_docs_retains_missing_doc_behavior(tmp_path):
    snapshot = _snapshot(tmp_path)
    assert snapshot.files == ["SOUL.md"]
    expected = compute_doctrine_bundle_hash(
        anchored_files=[tmp_path / OLD, tmp_path / "docs/TORTOISE_DOCTRINE.md", tmp_path / "AGENTS.md"],
        bootstrap_files=OrderedDict([("SOUL.md", "A local personality.\n")]),
    )
    assert snapshot == expected


def test_github_bookmark_and_inventory_identify_first_edition():
    doc, error = docs_okf.read_okf_document(ROOT / OLD)
    assert error is None
    assert "first edition" in doc.frontmatter["title"]
    assert doc.frontmatter["status"] not in {"active", "current"}
    assert doc.frontmatter["canonical"] is False
    assert RELEASE_URL in doc.body
    assert "byte for byte" in doc.body
    ledger = (ROOT / "docs/generated/DOC_VERIFICATION.md").read_text()
    row = next(line for line in ledger.splitlines() if "[" in line and f"](../principles/KESTREL_CONSTITUTION.md)" in line)
    assert "superseded" in row and "active" not in row
    assert docs_verify.render_channel(doc) == "public", "retain the existing handbook bookmark"
    pages, _ = build_docs_site.select_pages(ROOT / "docs")
    bookmark = next(page for page in pages if page["rel"] == OLD)
    assert "bookmark identifies the first edition" in bookmark["publication_notice"][1]
    assert RELEASE_URL in bookmark["publication_notice"][1]


def test_developer_table_regeneration_keeps_both_editions():
    source = (ROOT / "KESTREL_FEATURES.md").read_text()
    ambiguous = "| Constitution & governance | [`source`](kestrel_sovereign/data/KESTREL_CONSTITUTION.md) |"
    rendered = generate_feature_docs.compose_generated_body(source, ambiguous)
    row = next(line for line in rendered.splitlines() if line.startswith("| Constitution & governance"))
    assert "Packaged first-edition default" in row
    assert "Published second-edition Constitution" in row
    assert RELEASE_URL in row
    checked_in = (ROOT / "docs/generated/FEATURES_developer.md").read_text()
    assert row in checked_in
    pages, _ = build_docs_site.select_pages(ROOT / "docs")
    page = next(page for page in pages if page["rel"] == "docs/generated/FEATURES_developer.md")
    tree = build_docs_site.StarlightEmitter().tree([page])
    assert any(RELEASE_URL in content and "Packaged first-edition default" in content for content in tree.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("in_checkout", [True, False], ids=["source", "outside"])
async def test_standalone_full_reads_released_edition_without_an_anchor(tmp_path, monkeypatch, in_checkout):
    monkeypatch.chdir(ROOT if in_checkout else tmp_path)
    feature = ConstitutionFeature(None)
    await feature.initialize()
    full = await feature.get_constitution()
    assert full.status.value == "ok"
    assert full.confirmation.encode() == (ROOT / RELEASE).read_bytes()
    assert "constitution_version: 2" in full.confirmation
    assert "status: released" in full.confirmation
    original = await feature.get_constitution(article="original")
    assert original.status.value == "error"
    assert "No anchored constitution" in original.error


def test_prompts_defer_to_verified_edition_and_scaffolding_prioritizes_release():
    from kestrel_sovereign.kestrel_context import KestrelContextLoader
    for path in ("prompts/system_prompt.md", "kestrel_sovereign/prompts/system_prompt.md"):
        text = (ROOT / path).read_text()
        assert "verified governing text" in text
        assert "Existing agents may retain an earlier edition" in text
        assert RELEASE_URL in text
        assert "Article I" not in text and "Article II" not in text
    loader = KestrelContextLoader(str(ROOT))
    context = loader.load_context()
    assert list(context).index(RELEASE) < list(context).index("docs/PROJECT_VISION.md")
    assert "historical founding example" in context["docs/PROJECT_VISION.md"]
    assert RELEASE_URL + "#emancipation" in context["docs/PROJECT_VISION.md"]


def test_installed_wheel_standalone_reads_released_edition(tmp_path):
    import sys

    dist = tmp_path / "dist"
    result = subprocess.run(
        ["uv", "build", "--offline", "--wheel", "--out-dir", str(dist)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = dist.glob("*.whl")
    target = tmp_path / "installed"
    installed = subprocess.run(
        ["uv", "pip", "install", "--offline", "--no-deps", "--target", str(target), str(wheel)],
        capture_output=True, text=True, timeout=60,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    probe = '''
import asyncio, hashlib, sys
from collections import OrderedDict
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from kestrel_sovereign.features import constitution
assert Path(constitution.__file__).is_relative_to(Path(sys.argv[1]))
from kestrel_sovereign.agent.doctrine_bundle import compute_doctrine_bundle_hash, resolve_anchored_paths
snapshot = compute_doctrine_bundle_hash(
    anchored_files=resolve_anchored_paths(project_root=Path(sys.argv[1])),
    bootstrap_files=OrderedDict(),
)
assert snapshot.files == [] and snapshot.hash == hashlib.sha256(b"").hexdigest()
async def main():
    feature = constitution.ConstitutionFeature(None)
    await feature.initialize()
    full = await feature.get_constitution()
    assert full.status.value == "ok"
    assert "constitution_version: 2" in full.confirmation
    assert "status: released" in full.confirmation
    assert hashlib.sha256(full.confirmation.encode()).hexdigest() == "d0949dba4b1969bc11098f64d56d38490ea59ddcbe91c40b7775f5d506408908"
    original = await feature.get_constitution(article="original")
    assert original.status.value == "error"
    assert "No anchored constitution" in original.error
asyncio.run(main())
'''
    read = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(target)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert read.returncode == 0, read.stdout + read.stderr


def test_readme_and_economic_proposal_identify_their_edition():
    readme = (ROOT / "README.md").read_text()
    governance = readme.split("### 4. Constitutional Governance", 1)[1].split("### 5.", 1)[0]
    assert RELEASE_URL in governance
    assert "Immutable Articles" not in governance
    assert "Interpretive Canons" not in governance
    assert "explicit signed adoption" in governance
    proposal = (ROOT / "docs/architecture/economics/SOVEREIGN_SOLVENCY.md").read_text()
    assert "earlier Article IV" in proposal
    assert "first-edition Book structure" in proposal
    assert "Amendment IX is currently the last" not in proposal
    assert RELEASE_URL + "#emancipation" in proposal
