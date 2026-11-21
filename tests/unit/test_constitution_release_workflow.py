"""Exercise the packaging step used to attest a constitutional release."""
import os
from pathlib import Path
import subprocess
import tarfile

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/constitution-release.yml"


def test_publisher_attests_only_the_named_edition_from_main(tmp_path):
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["attest"]
    assert job["if"] == "github.ref == 'refs/heads/main'"
    assert job["permissions"] == {"contents": "read", "id-token": "write", "attestations": "write"}
    script = next(step["run"] for step in job["steps"] if "run" in step)
    assert next(step for step in job["steps"] if "attest-build-provenance@" in step.get("uses", ""))[
        "with"
    ]["subject-path"] == "${{ runner.temp }}/constitution/*.tar.gz"
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    relative = Path("kestrel_sovereign/data/constitutions/v2")
    folder = root / relative
    folder.mkdir(parents=True)
    for name in ("KESTREL_CONSTITUTION.md", "PLEDGE.md", "unrelated.md"):
        (folder / name).write_text(name + "\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    tree = subprocess.check_output(["git", "write-tree"], cwd=root, text=True).strip()
    # A tree object suffices for git archive and avoids a synthetic commit identity.
    runner = tmp_path / "runner"
    runner.mkdir()
    env = dict(os.environ, RUNNER_TEMP=str(runner), GITHUB_SHA=tree, EDITION="v2")
    subprocess.run(["bash", "-c", script], cwd=root, env=env, check=True)
    with tarfile.open(runner / "constitution/constitution-v2.tar.gz") as archive:
        contents = {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}
    assert contents == {str(relative / name): (name + "\n").encode() for name in ("KESTREL_CONSTITUTION.md", "PLEDGE.md")}
    for unsupported in ("v3", "../v2", "v2/../../"):
        env["EDITION"] = unsupported
        assert subprocess.run(["bash", "-c", script], cwd=root, env=env).returncode != 0
