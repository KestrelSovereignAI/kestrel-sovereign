"""``publish.yml`` must grant at least what ``ci.yml`` asks for.

``publish.yml`` calls ``ci.yml`` as a reusable workflow for its release gate.
A called workflow's jobs cannot hold permissions the calling job lacks, and
GitHub validates that **at run-creation time, before any ``if:`` is
evaluated** — so a job that could never run still counts.

When the ceiling is exceeded the entire publish run dies with
``startup_failure`` and zero jobs: no CI gate, no clean-install gate, no
build, no PyPI upload, and no logs. Reproduced directly — a nested job
requesting ``actions: write`` behind ``if: false``, under a caller capped at
``contents: read``, produced ``conclusion=startup_failure, jobs=0`` in four
seconds.

The failure is silent until someone pushes a release tag, which is the worst
possible time to discover it, so it gets a unit test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
PUBLISH = WORKFLOWS / "publish.yml"

# GitHub's permission lattice for the purposes of "is X at least Y".
_RANK = {"none": 0, "read": 1, "write": 2}

# Enough of GitHub's scope list to make `write-all` meaningfully exceed a
# `contents: read` ceiling.
_ALL_SCOPES = (
    "actions", "attestations", "checks", "contents", "deployments", "discussions",
    "id-token", "issues", "packages", "pages", "pull-requests",
    "repository-projects", "security-events", "statuses",
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _nested_calls() -> dict[str, tuple[dict, Path]]:
    """Every job in publish.yml that calls a local reusable workflow.

    Not just ci.yml: `clean-install.yml` is called the same way and carries the
    same trap. It declares no `permissions:` today, so it is safe — but one
    added line would brick the next release identically, and a test that only
    looked at ci.yml would not notice.
    """
    found = {}
    for name, job in _load(PUBLISH)["jobs"].items():
        uses = str(job.get("uses", ""))
        if uses.startswith("./.github/workflows/"):
            found[name] = (job, WORKFLOWS / Path(uses).name)
    assert found, "publish.yml no longer calls any local reusable workflow"
    return found


def _calling_job() -> dict:
    """The job in publish.yml that invokes ci.yml."""
    calls = [job for job, path in _nested_calls().values() if path.name == "ci.yml"]
    assert len(calls) == 1, f"expected exactly one job calling ci.yml, found {len(calls)}"
    return calls[0]


def _effective_ceiling() -> dict[str, str]:
    """What the calling job actually grants the called workflow.

    A job-level ``permissions:`` replaces the workflow-level default outright;
    it does not merge with it.
    """
    job = _calling_job()
    if "permissions" in job:
        return job["permissions"]
    return _load(PUBLISH).get("permissions", {}) or {}


def _requested(path: Path = CI) -> dict[str, dict[str, str]]:
    """Every ``permissions:`` block in a called workflow, job-level and above.

    Workflow-level blocks count: they apply to every job and are validated
    against the same ceiling. Omitting them left a hole where adding
    ``permissions: {id-token: write}`` at the top of ci.yml re-armed the
    startup failure with every test green.

    The shorthand string forms (``write-all`` / ``read-all`` / ``{}``) are
    normalised rather than skipped, since ``permissions: write-all`` on a job
    is the most direct way to exceed a ceiling.
    """
    document = _load(path)
    blocks: dict[str, dict[str, str]] = {}

    def record(label: str, value) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if value == "write-all":
                blocks[label] = {scope: "write" for scope in _ALL_SCOPES}
            elif value == "read-all":
                blocks[label] = {scope: "read" for scope in _ALL_SCOPES}
            return
        if isinstance(value, dict):
            blocks[label] = value

    record(f"{path.name} (workflow-level)", document.get("permissions"))
    for name, job in document["jobs"].items():
        record(name, job.get("permissions"))
    return blocks


def test_ci_is_actually_called_by_publish():
    """If this stops being true the rest of the file is vacuous."""
    assert _calling_job()["uses"].endswith("ci.yml")


def test_every_nested_workflow_is_within_its_callers_ceiling():
    """Covers every reusable workflow publish.yml calls, not only ci.yml."""
    violations = []
    for caller, (job, called) in _nested_calls().items():
        ceiling = job.get("permissions") or _load(PUBLISH).get("permissions") or {}
        for label, perms in _requested(called).items():
            for scope, level in perms.items():
                granted = ceiling.get(scope, "none")
                if _RANK[str(level)] > _RANK[str(granted)]:
                    violations.append(
                        f"  {called.name} {label!r} requests {scope}: {level}, but "
                        f"publish.yml job {caller!r} grants {scope}: {granted}"
                    )
    assert not violations, (
        "publish.yml would fail at startup with zero jobs on the next release "
        "tag.\n" + "\n".join(violations)
    )


def test_every_requested_scope_is_within_the_ceiling():
    ceiling = _effective_ceiling()
    violations = []
    for job_name, perms in _requested().items():
        for scope, level in perms.items():
            granted = ceiling.get(scope, "none")
            if _RANK[str(level)] > _RANK[str(granted)]:
                violations.append(
                    f"  ci.yml job {job_name!r} requests {scope}: {level}, "
                    f"but publish.yml's CI gate grants {scope}: {granted}"
                )
    assert not violations, (
        "publish.yml would fail at startup with zero jobs on the next release "
        "tag.\n" + "\n".join(violations) + "\n\nWiden the `permissions:` block on "
        "publish.yml's `ci:` job, or drop the request from ci.yml."
    )


@pytest.mark.parametrize(
    "job_name,scope,level",
    [
        ("lint-and-imports", "pull-requests", "read"),
        ("cancel-expensive-tiers-when-unit-fails", "actions", "write"),
    ],
)
def test_known_escalations_are_still_declared(job_name, scope, level):
    """Pins the two jobs that drove the ceiling change.

    Not redundant with the subset check: if someone deletes a job's
    ``permissions:`` block, the subset test still passes while the job silently
    loses the scope it needs — ``gh pr list`` or the cancel call would 403 into
    a no-op rather than fail.
    """
    requested = _requested()
    assert job_name in requested, f"{job_name} no longer declares permissions"
    assert requested[job_name].get(scope) == level


def test_ceiling_is_declared_on_the_calling_job_not_inherited():
    """The workflow-level default in publish.yml is `contents: read`, which is
    narrower than ci.yml needs. The grant has to be on the `ci:` job itself."""
    assert "permissions" in _calling_job(), (
        "publish.yml's CI gate relies on the workflow-level `contents: read` "
        "default, which is too narrow for ci.yml's jobs"
    )
