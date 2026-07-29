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


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _calling_job() -> dict:
    """The job in publish.yml that invokes ci.yml."""
    jobs = _load(PUBLISH)["jobs"]
    calls = [
        (name, job)
        for name, job in jobs.items()
        if str(job.get("uses", "")).endswith("ci.yml")
    ]
    assert len(calls) == 1, f"expected exactly one job calling ci.yml, found {[c[0] for c in calls]}"
    return calls[0][1]


def _effective_ceiling() -> dict[str, str]:
    """What the calling job actually grants the called workflow.

    A job-level ``permissions:`` replaces the workflow-level default outright;
    it does not merge with it.
    """
    job = _calling_job()
    if "permissions" in job:
        return job["permissions"]
    return _load(PUBLISH).get("permissions", {}) or {}


def _requested() -> dict[str, dict[str, str]]:
    """Every ``permissions:`` block declared by a job in ci.yml."""
    return {
        name: job["permissions"]
        for name, job in _load(CI)["jobs"].items()
        if isinstance(job.get("permissions"), dict)
    }


def test_ci_is_actually_called_by_publish():
    """If this stops being true the rest of the file is vacuous."""
    assert _calling_job()["uses"].endswith("ci.yml")


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
