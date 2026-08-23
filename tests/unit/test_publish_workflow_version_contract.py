"""Release publishing must bind the package version to an explicit semver tag."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tomllib

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"


def _build_job() -> dict[str, object]:
    workflow = yaml.safe_load(PUBLISH_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["build"]


def _version_check_step() -> dict[str, object]:
    steps = _build_job()["steps"]
    return next(
        step
        for step in steps
        if step.get("name") == "Verify the version metadata matches the tag"
    )


def _version_check_script() -> str:
    return _version_check_step()["run"]


def _run_version_check(release_ref: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["RELEASE_REF"] = release_ref
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _version_check_script()],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_publish_version_check_accepts_matching_tag_for_push_and_dispatch() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]

    for release_ref in (f"v{version}", f"refs/tags/v{version}"):
        result = _run_version_check(release_ref)
        assert result.returncode == 0, result.stderr
        assert f"tag v{version} matches" in result.stdout


@pytest.mark.parametrize(
    "release_ref",
    ("main", "631bcf32", "v0.53", "v0.53.4-rc1", "refs/heads/v0.53.4"),
)
def test_publish_version_check_rejects_non_release_refs(release_ref: str) -> None:
    result = _run_version_check(release_ref)

    assert result.returncode != 0
    assert "release ref must be a vX.Y.Z tag" in result.stdout


def test_publish_version_check_rejects_mismatched_tag() -> None:
    result = _run_version_check("v999.999.999")

    assert result.returncode != 0
    assert "does not match pyproject.toml version" in result.stdout


def test_manual_publish_input_is_described_as_a_release_tag() -> None:
    workflow_text = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "Release tag to publish" in workflow_text
    assert "Tag or commit to publish" not in workflow_text


def test_version_check_runs_for_tag_pushes_and_manual_dispatches() -> None:
    step = _version_check_step()

    assert "if" not in step
    assert step["env"]["RELEASE_REF"] == "${{ inputs.ref || github.ref_name }}"


def test_build_checks_out_the_same_release_ref_that_is_verified() -> None:
    checkout = next(
        step
        for step in _build_job()["steps"]
        if step.get("name") == "Checkout"
    )

    assert checkout["with"]["ref"] == "${{ inputs.ref || github.ref }}"
