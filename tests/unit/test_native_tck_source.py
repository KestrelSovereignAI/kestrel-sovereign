"""TCK preparation uses real pinned Git objects and checks its complete tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tests.utils.native_tck_source import prepared_tck_source


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


@pytest.fixture
def pinned_source(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "--quiet")
    (upstream / "LICENSE").write_bytes(b"Test-only source license\n")
    (upstream / "specification").mkdir()
    version = {"branch": "v1.0.1", "commitHash": "1" * 40}
    (upstream / "specification/version.json").write_text(json.dumps(version))
    (upstream / "runner.py").write_text("VALUE = 'base'\n")
    git(upstream, "add", ".")
    git(upstream, "-c", "user.name=TCK Test", "-c", "user.email=tck-test@example.invalid",
        "commit", "--quiet", "-m", "Pinned source")
    commit = git(upstream, "rev-parse", "HEAD")
    (upstream / "runner.py").write_text("VALUE = 'reviewed'\n")
    patch = subprocess.check_output(["git", "diff", "--binary"], cwd=upstream)
    git(upstream, "add", "runner.py")
    tree = git(upstream, "write-tree")
    # The caller's dirty working copy must not become the materialized source.
    (upstream / "runner.py").write_text("VALUE = 'unreviewed dirt'\n")
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "UPSTREAM.patch").write_bytes(patch)
    license_bytes = (upstream / "LICENSE").read_bytes()
    (provenance / "LICENSE").write_bytes(license_bytes)
    manifest = {
        "schema_version": 1, "upstream_repository": str(upstream),
        "upstream_commit": commit, "patched_tree": tree,
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "license_sha256": hashlib.sha256(license_bytes).hexdigest(),
        "protocol_version": "1.0.1", "spec_commit": version["commitHash"],
    }
    return provenance, manifest


def write_manifest(provenance, manifest):
    (provenance / "SOURCE.toml").write_text("\n".join(
        f"{key} = {json.dumps(value)}" for key, value in manifest.items()
    ) + "\n")


def test_tck_checkout_applies_only_the_reviewed_patch_and_owns_cleanup(tmp_path, pinned_source):
    provenance, manifest = pinned_source
    write_manifest(provenance, manifest)
    with prepared_tck_source(tmp_path, provenance_dir=provenance) as root:
        assert (root / "runner.py").read_text() == "VALUE = 'reviewed'\n"
        assert git(root, "rev-parse", "HEAD") == manifest["upstream_commit"]
        assert git(root, "write-tree") == manifest["patched_tree"]
        assert not (root / ".venv").exists()
        materialized = root
    assert not materialized.exists()
    assert (Path(manifest["upstream_repository"]) / "runner.py").read_text() == "VALUE = 'unreviewed dirt'\n"


@pytest.mark.parametrize("filename", ["UPSTREAM.patch", "LICENSE"])
def test_changed_tck_inputs_fail_before_creating_a_checkout(tmp_path, pinned_source, filename):
    provenance, manifest = pinned_source
    write_manifest(provenance, manifest)
    (provenance / filename).write_bytes(b"Changed after review")
    with pytest.raises(ValueError, match="custody digest mismatch"):
        with prepared_tck_source(tmp_path, provenance_dir=provenance):
            pytest.fail("Unreviewed source was admitted")
    assert not list(tmp_path.glob("native-a2a-tck-*"))


@pytest.mark.parametrize(("field", "value", "message"), [
    ("schema_version", 2, "Unsupported"),
    ("upstream_commit", "main", "immutable upstream_commit"),
    ("patched_tree", "2" * 40, "patched tree"),
    ("spec_commit", "3" * 40, "specification differs"),
    ("protocol_version", "0.3.0", "specification differs"),
    ("upstream_repository", "missing-repository", "preparation failed"),
])
def test_invalid_tck_provenance_fails_and_removes_owned_source(
    tmp_path, pinned_source, field, value, message,
):
    provenance, manifest = pinned_source
    manifest[field] = value
    write_manifest(provenance, manifest)
    with pytest.raises((ValueError, RuntimeError), match=message):
        with prepared_tck_source(tmp_path, provenance_dir=provenance):
            pytest.fail("Unreviewed source was admitted")
    assert not list(tmp_path.glob("native-a2a-tck-*"))


def test_test_failure_still_removes_the_owned_tck_source(tmp_path, pinned_source):
    provenance, manifest = pinned_source
    write_manifest(provenance, manifest)
    with pytest.raises(RuntimeError, match="original test error"):
        with prepared_tck_source(tmp_path, provenance_dir=provenance) as root:
            materialized = root
            raise RuntimeError("original test error")
    assert not materialized.exists()


def test_tck_checkout_bytes_do_not_inherit_operator_crlf_configuration(
    tmp_path, pinned_source, monkeypatch,
):
    provenance, manifest = pinned_source
    write_manifest(provenance, manifest)
    configuration = tmp_path / "operator-gitconfig"
    configuration.write_text("[core]\n\tautocrlf = true\n\teol = crlf\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(configuration))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    with prepared_tck_source(tmp_path, provenance_dir=provenance) as root:
        assert (root / "LICENSE").read_bytes() == (provenance / "LICENSE").read_bytes()
        assert (root / "runner.py").read_bytes() == b"VALUE = 'reviewed'\n"
