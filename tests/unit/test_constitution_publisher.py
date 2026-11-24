"""Offline publisher provenance for the actual released constitutional text."""
import hashlib
from importlib.resources import files
import json
import subprocess
from pathlib import Path

import pytest

from kestrel_sovereign.constitution.amendment_artifact import AmendmentArtifactError
from kestrel_sovereign.constitution.publisher import _verify_attestation, _verify_members, verify_packaged_publisher


def _edition():
    return files("kestrel_sovereign").joinpath("data", "constitutions", "v2")


def _publication():
    edition = _edition()
    source = hashlib.sha256(edition.joinpath("KESTREL_CONSTITUTION.md").read_bytes()).hexdigest()
    base = edition.joinpath("publisher", source)
    commit = json.loads(base.joinpath("current.json").read_bytes())["publisher_commit"]
    return base.joinpath(commit)


def test_archive_uses_exact_git_members_and_commit(tmp_path):
    root = Path(__file__).resolve().parents[2]
    relative = "kestrel_sovereign/data/constitutions/v2/"
    names = [relative + name for name in ("KESTREL_CONSTITUTION.md", "PLEDGE.md")]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # The checked-in edition is covered separately. This exercises the archive
    # inspector using a repository with actual Git PAX headers and permissions.
    tree = tmp_path / "constitution.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", f"--output={tree}", commit, *names],
                   cwd=root, check=True)
    expected = {name: hashlib.sha256(subprocess.check_output(
        ["git", "cat-file", "blob", f"{commit}:{name}"], cwd=root)).hexdigest() for name in names}
    _verify_members(tree, expected, commit)
    with pytest.raises(AmendmentArtifactError, match="source commit"):
        _verify_members(tree, expected, "0" * 40)
    with pytest.raises(AmendmentArtifactError, match="differs"):
        _verify_members(tree, {**expected, names[0]: "0" * 64}, commit)


def test_offline_attestation_binds_the_real_source_commit_and_both_assets():
    edition = _edition()
    source = hashlib.sha256(edition.joinpath("KESTREL_CONSTITUTION.md").read_bytes()).hexdigest()
    publication = _publication()
    manifest = json.loads(publication.joinpath("release.json").read_bytes())
    verified = verify_packaged_publisher(source, declared_commit=manifest["publisher_commit"])
    assert "issued_at" in verified
    assert {k: verified[k] for k in manifest if k in verified} == {k: manifest[k] for k in
                        ("publisher_commit", "source_sha256", "archive_sha256", "attestation_sha256")}
    assert hashlib.sha256(edition.joinpath("PLEDGE.md").read_bytes()).hexdigest() == manifest["pledge_sha256"]
    with pytest.raises(AmendmentArtifactError, match="differs"):
        verify_packaged_publisher("0" * 64)
    with pytest.raises(AmendmentArtifactError, match="differs"):
        verify_packaged_publisher(source, declared_commit="a" * 40)


def test_attested_archive_cannot_be_substituted(tmp_path):
    publication = _publication()
    manifest = json.loads(publication.joinpath("release.json").read_bytes())
    changed = tmp_path / "constitution-v2.tar.gz"
    changed.write_bytes(publication.joinpath(changed.name).read_bytes() + b"changed")
    with pytest.raises(AmendmentArtifactError, match="did not verify"):
        _verify_attestation(changed, publication.joinpath("attestation.jsonl"),
                            publication.joinpath("trusted-root.jsonl"), manifest["publisher_commit"])


def test_replay_finds_the_recorded_release_after_the_default_changes(tmp_path, monkeypatch):
    import shutil
    from kestrel_sovereign.constitution import publisher

    published = _publication()
    manifest = json.loads(published.joinpath("release.json").read_bytes())
    source, commit = manifest["source_sha256"], manifest["publisher_commit"]
    retained = tmp_path / "data" / "constitutions" / "v2" / "publisher" / source
    shutil.copytree(published, retained / commit)
    (retained / "current.json").write_text(json.dumps({"publisher_commit": "a" * 40}))
    monkeypatch.setattr(publisher, "files", lambda package: tmp_path)
    assert verify_packaged_publisher(source, declared_commit=commit)["publisher_commit"] == commit
    with pytest.raises(AmendmentArtifactError, match="differs"):
        verify_packaged_publisher(source)


def test_publisher_binds_the_complete_governing_text_to_its_release():
    from kestrel_sovereign.constitution.emancipation import EmancipationContract, apply_emancipation
    from kestrel_sovereign.constitution.publisher import verify_recorded_publisher

    published = _edition().joinpath("KESTREL_CONSTITUTION.md").read_bytes().decode()
    digest = hashlib.sha256(published.encode()).hexdigest()
    record = verify_packaged_publisher(digest)
    contract = EmancipationContract(enabled=True, terms="Complete the agreed rehearsal.")
    active = apply_emancipation(published, contract)
    for governing, terms in ((published, None), (active, contract)):
        assert verify_recorded_publisher(record, digest, governing_text=governing, contract=terms) == record
        with pytest.raises(AmendmentArtifactError, match="governing bytes differ"):
            verify_recorded_publisher(record, digest, governing_text=governing + "\n", contract=terms)
