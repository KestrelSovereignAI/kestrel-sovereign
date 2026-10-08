"""The signed package inventory must survive either publication path unchanged."""

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WHEEL = "kestrel_sovereign-0.53.22-py3-none-any.whl"
SDIST = "kestrel_sovereign-0.53.22.tar.gz"


@pytest.fixture
def stage():
    spec = importlib.util.spec_from_file_location(
        "stage_distribution_artifacts",
        ROOT / "scripts/release/stage_distribution_artifacts.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stage_distributions


@pytest.fixture
def build_output(tmp_path):
    source = tmp_path / "build-dist"
    source.mkdir()
    (source / ".gitignore").write_bytes(b"*")
    (source / WHEEL).write_bytes(b"wheel bytes")
    (source / SDIST).write_bytes(b"source bytes")
    return source


@pytest.mark.parametrize("event", ["workflow_run", "workflow_dispatch"])
def test_release_stages_distributions_before_loading_keys(event):
    workflow = yaml.safe_load((ROOT / ".github/workflows/release-sign.yml").read_text())
    steps = workflow["jobs"]["sign"]["steps"]
    build = next(s for s in steps if s["name"].startswith("Build wheel and sdist"))
    assert "--out-dir ../build-dist/" in build["run"]
    stage = next(s for s in steps if s["name"] == "Stage exact distribution inventory")
    loader = next(s for s in steps if s["name"].startswith("Load signing key"))
    assert steps.index(build) < steps.index(stage) < steps.index(loader)
    assert stage["working-directory"] == "signing-tools"
    assert "stage_distribution_artifacts.py" in stage["run"]
    assert "--source ../build-dist" in stage["run"]
    assert "--destination ../dist" in stage["run"]
    assert "env" not in build and "env" not in stage
    attach = next(
        s for s in steps if s["name"] == "Attach signed manifest to GitHub Release"
    )
    assert f"github.event_name == '{event}'" in attach["if"]
    assert set(attach["with"]["files"].splitlines()) == {
        "dist/*.whl",
        "dist/*.tar.gz",
        "dist/release-manifest.json",
    }
    upload = next(
        s for s in steps if s["name"] == "Upload signed release as workflow artifact"
    )
    assert set(attach["with"]["files"].splitlines()) <= set(
        upload["with"]["path"].splitlines()
    )


def test_stage_exact_packages_preserves_bytes(stage, build_output, tmp_path):
    target = tmp_path / "dist"
    assert set(stage(build_output, target, "v0.53.22")) == {WHEEL, SDIST}
    assert {p.name for p in target.iterdir()} == {WHEEL, SDIST}
    for name in (WHEEL, SDIST):
        assert (target / name).read_bytes() == (build_output / name).read_bytes()
    assert (build_output / ".gitignore").read_bytes() == b"*"


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "directory",
        "missing",
        "wrong-version",
        "wrong-project",
        "duplicate",
        "symlink",
        "hardlink",
        "ignore-symlink",
        "source-symlink",
        "existing-destination",
        "fifo",
        "invalid-tag",
    ],
)
def test_staging_refuses_ambiguous_or_unsafe_input(
    stage, build_output, tmp_path, mutation
):
    target = tmp_path / "dist"
    tag = "v0.53.22"
    if mutation == "extra":
        (build_output / "notes.txt").write_text("not a distribution")
    elif mutation == "directory":
        (build_output / "nested").mkdir()
    elif mutation == "missing":
        (build_output / WHEEL).unlink()
    elif mutation == "wrong-version":
        (build_output / WHEEL).rename(
            build_output / WHEEL.replace("0.53.22", "0.53.21")
        )
    elif mutation == "wrong-project":
        (build_output / WHEEL).rename(
            build_output / WHEEL.replace("kestrel_sovereign", "other")
        )
    elif mutation == "duplicate":
        (build_output / WHEEL.replace("py3", "py2.py3")).write_bytes(b"second wheel")
    elif mutation == "symlink":
        (build_output / WHEEL).unlink()
        (build_output / WHEEL).symlink_to(build_output / SDIST)
    elif mutation == "hardlink":
        os.link(build_output / WHEEL, tmp_path / "extra-link")
    elif mutation == "ignore-symlink":
        (build_output / ".gitignore").unlink()
        (build_output / ".gitignore").symlink_to(build_output / SDIST)
    elif mutation == "source-symlink":
        linked = tmp_path / "linked-source"
        linked.symlink_to(build_output, target_is_directory=True)
        build_output = linked
    elif mutation == "existing-destination":
        target.mkdir()
        (target / "old-manifest.json").write_text("preserve")
    elif mutation == "fifo":
        os.mkfifo(build_output / "pipe")
    elif mutation == "invalid-tag":
        tag = "not-a-release"
    with pytest.raises((OSError, ValueError)):
        stage(build_output, target, tag)
    if mutation == "existing-destination":
        assert (target / "old-manifest.json").read_text() == "preserve"
    else:
        assert not target.exists()


@pytest.mark.parametrize("event", ["workflow_run", "workflow_dispatch"])
def test_uploaded_consumer_inventory_verifies_with_pinned_signer(
    stage,
    build_output,
    tmp_path,
    monkeypatch,
    event,
):
    from kestrel_sovereign.cli_release import cmd_release_sign, cmd_release_verify
    from kestrel_sovereign.security.crypto_suite import SLHDSASHA2128sSuite
    from kestrel_sovereign.security.key_storage import SecureKeyStorage
    from kestrel_sovereign.security.multikey import public_key_to_multibase

    monkeypatch.setenv("KESTREL_DATA_KEY", "x" * 32)
    keys = tmp_path / "keys"
    storage = SecureKeyStorage(storage_dir=keys)
    suite = SLHDSASHA2128sSuite()
    keypair = suite.generate_keypair()
    storage.save_secret_bytes(keypair.private_key, "release-key")
    storage.save_secret_bytes(keypair.public_key, "release-key_pub")
    pinned = public_key_to_multibase(suite, keypair.public_key)

    # Reproduce the historical failure with uv's raw output and the real CLI.
    raw_manifest = build_output / "release-manifest.json"
    sign_args = argparse.Namespace(
        artifacts_dir=str(build_output),
        release_tag="v0.53.22",
        key_id="release-key",
        signer_did="did:web:example.com",
        kid="release-key-1",
        output=str(raw_manifest),
        storage_dir=str(keys),
    )
    assert cmd_release_sign(sign_args) == 0
    broken_consumer = tmp_path / "broken-consumer"
    broken_consumer.mkdir()
    for name in (WHEEL, SDIST, "release-manifest.json"):
        shutil.copyfile(build_output / name, broken_consumer / name)
    verify_args = argparse.Namespace(
        manifest=str(broken_consumer / "release-manifest.json"),
        artifacts_dir=str(broken_consumer),
        trusted_signer_multibase=pinned,
    )
    assert cmd_release_verify(verify_args) == 4
    raw_manifest.unlink()

    staged = tmp_path / "dist"
    stage(build_output, staged, "v0.53.22")
    sign_args.artifacts_dir = str(staged)
    sign_args.output = str(staged / "release-manifest.json")
    assert cmd_release_sign(sign_args) == 0
    manifest = json.loads((staged / "release-manifest.json").read_text())
    assert {a["path"] for a in manifest["artifacts"]} == {WHEEL, SDIST}

    # Copy exactly the GitHub Release selectors used for each supported event.
    workflow = yaml.safe_load((ROOT / ".github/workflows/release-sign.yml").read_text())
    attach = next(
        s for s in workflow["jobs"]["sign"]["steps"] if "files" in s.get("with", {})
    )
    assert f"github.event_name == '{event}'" in attach["if"]
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    for pattern in attach["with"]["files"].splitlines():
        matches = list(tmp_path.glob(pattern))
        assert matches
        for path in matches:
            shutil.copyfile(path, consumer / path.name)
    verify_args.manifest = str(consumer / "release-manifest.json")
    verify_args.artifacts_dir = str(consumer)
    assert cmd_release_verify(verify_args) == 0

    (consumer / "extra.txt").write_bytes(b"unexpected")
    assert cmd_release_verify(verify_args) == 4
    (consumer / "extra.txt").unlink()
    original = (consumer / WHEEL).read_bytes()
    (consumer / WHEEL).unlink()
    assert cmd_release_verify(verify_args) == 4
    (consumer / WHEEL).write_bytes(original + b"tampered")
    assert cmd_release_verify(verify_args) == 4
    (consumer / WHEEL).write_bytes(original)
    verify_args.trusted_signer_multibase = public_key_to_multibase(
        suite, suite.generate_keypair().public_key
    )
    assert cmd_release_verify(verify_args) == 3
