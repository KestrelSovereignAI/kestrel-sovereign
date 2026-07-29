"""Decision table for the ``duplicate-run-gate`` job in ``.github/workflows/ci.yml``.

That job decides whether the expensive CI tiers (integration-tests, and
llm-tests via its ``needs:``) should run at all. It exists because a
``kestrel-talon`` branch matches BOTH the ``issue-*`` push trigger and the
``pull_request`` trigger, so once its PR is open every commit fires two
complete CI runs on the same SHA. See #2800.

The gate is shell embedded in YAML, which nothing else type-checks or
exercises. Getting it wrong in the "skip" direction silently stops testing
real changes, so these tests run the ACTUAL script extracted from the
workflow — not a copy — with ``gh`` stubbed and the ``${{ }}`` expressions
substituted the way Actions would.

The invariant: the gate skips in exactly one situation — a push to a branch
that already has an open PR onto a base this workflow actually runs for
(``main`` or ``epic/**``). Everything else must run, including a PR onto an
uncovered base such as a stacked PR, and every error path.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

CI_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

# Stubs for the one external command the gate calls. It asks for
# ``--json baseRefName --jq '.[].baseRefName'``, so the stub emits one base
# branch per line — the shape that matters, because an open PR only makes a
# push redundant when that PR's base is one this workflow runs for.
GH_STUBS = {
    "no-pr": "gh() { :; }",
    "pr-onto-main": "gh() { echo main; }",
    "pr-onto-epic": "gh() { echo 'epic/semantic-kb'; }",
    "two-prs-onto-main": "gh() { printf 'main\\nmain\\n'; }",
    # Stacked PR: base is another issue-* branch, which the `pull_request`
    # trigger filter (main, epic/**) ignores — so NO pull_request run exists
    # and this push run is the only coverage.
    "pr-onto-issue-branch": "gh() { echo 'issue-2793-ci-unchain-jobs'; }",
    "pr-onto-release-branch": "gh() { echo 'release/0.50'; }",
    "mixed-stacked-and-main": "gh() { printf 'issue-2793-x\\nmain\\n'; }",
    "api-failure": "gh() { return 1; }",
}


def _gate_script() -> str:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    steps = workflow["jobs"]["duplicate-run-gate"]["steps"]
    return steps[0]["run"]


def _render(script: str, event: str, ref: str) -> str:
    """Substitute the Actions expressions, as the runner would."""
    ref_name = re.sub(r"^refs/(heads|tags)/", "", ref)
    rendered = (
        script.replace("${{ github.event_name }}", event)
        .replace("${{ github.ref }}", ref)
        .replace("${{ github.ref_name }}", ref_name)
        .replace("${{ github.repository }}", "KestrelSovereignAI/kestrel-sovereign")
    )
    assert "${{" not in rendered, f"unsubstituted expression:\n{rendered}"
    return rendered


def _decide(event: str, ref: str, gh_mode: str) -> str:
    """Run the real gate and return its ``run_expensive`` output."""
    handle = tempfile.NamedTemporaryFile("w", delete=False, suffix=".env")
    handle.close()
    try:
        body = GH_STUBS[gh_mode] + "\n" + _render(_gate_script(), event, ref)
        subprocess.run(
            ["bash", "-c", body],
            env={**os.environ, "GITHUB_OUTPUT": handle.name},
            capture_output=True,
            text=True,
            timeout=30,
        )
        written = Path(handle.name).read_text()
    finally:
        os.unlink(handle.name)

    matches = re.findall(r"^run_expensive=(\w+)$", written, re.MULTILINE)
    assert matches, f"gate wrote no run_expensive decision; GITHUB_OUTPUT was:\n{written!r}"
    # The script exits at its first decision, so exactly one line is correct.
    assert len(matches) == 1, f"gate wrote {len(matches)} decisions: {matches}"
    return matches[0]


class TestSkipsOnlyTheDuplicate:
    """The single case the gate exists to suppress."""

    @pytest.mark.parametrize(
        "gh_mode", ["pr-onto-main", "pr-onto-epic", "two-prs-onto-main", "mixed-stacked-and-main"]
    )
    def test_branch_push_with_covered_pr_skips(self, gh_mode):
        assert _decide("push", "refs/heads/issue-2800-dedupe", gh_mode) == "false"

    def test_branch_push_before_pr_exists_runs(self):
        """The pre-PR window is real: talon opens its PR 35 min to 2.7 h after
        the first push, so this run is the only signal until then."""
        assert _decide("push", "refs/heads/issue-2800-dedupe", "no-pr") == "true"


class TestPrMustActuallyTriggerThisWorkflow:
    """An open PR only makes a push redundant if that PR runs this workflow.

    The ``pull_request`` trigger is filtered to base branches ``main`` and
    ``epic/**``. A PR onto anything else produces no ``pull_request`` run, so
    suppressing the push run would leave the commit with no integration
    coverage at all. Stacked PRs onto another ``issue-*`` branch are the
    realistic case.
    """

    @pytest.mark.parametrize("gh_mode", ["pr-onto-issue-branch", "pr-onto-release-branch"])
    def test_pr_onto_uncovered_base_still_runs(self, gh_mode):
        assert _decide("push", "refs/heads/issue-2800-dedupe", gh_mode) == "true"


class TestEveryOtherTriggerRuns:
    """No trigger other than a duplicate branch push may be suppressed."""

    @pytest.mark.parametrize(
        "event,ref,gh_mode",
        [
            ("pull_request", "refs/pull/2801/merge", "pr-onto-main"),
            ("push", "refs/heads/main", "pr-onto-main"),
            ("workflow_dispatch", "refs/heads/some-branch", "no-pr"),
            # publish.yml calls this workflow as its release gate. On a tag
            # push the caller's ref is the tag; a workflow_dispatch release
            # from main arrives as event=workflow_dispatch.
            ("push", "refs/tags/v0.49.5", "no-pr"),
            ("push", "refs/tags/v0.49.5", "pr-onto-main"),
        ],
    )
    def test_runs(self, event, ref, gh_mode):
        assert _decide(event, ref, gh_mode) == "true"


class TestFailsOpen:
    """Any uncertainty must run the tests, never skip them."""

    @pytest.mark.parametrize("gh_mode", ["api-failure"])
    def test_unusable_pr_query_still_runs(self, gh_mode):
        assert _decide("push", "refs/heads/issue-2800-dedupe", gh_mode) == "true"


class TestWiring:
    """The gate is only worth testing if the expensive tiers consult it."""

    def test_integration_tests_consults_the_gate(self):
        jobs = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]
        integration = jobs["integration-tests"]
        assert "duplicate-run-gate" in integration["needs"]
        assert "needs.duplicate-run-gate.outputs.run_expensive == 'true'" in integration["if"]

    def test_llm_tests_inherits_the_skip(self):
        """llm-tests is not gated directly; it depends on integration-tests, and
        a job whose dependency is skipped is itself skipped."""
        jobs = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]
        assert jobs["llm-tests"]["needs"] == "integration-tests"

    def test_required_checks_are_never_gated(self):
        """lint-and-imports and unit-tests are required status checks. A skipped
        job still posts a check run, so gating them would put a `skipped`
        conclusion on the merge gate. Deliberately out of scope."""
        jobs = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]
        for name in ("lint-and-imports", "unit-tests"):
            assert "duplicate-run-gate" not in str(jobs[name].get("needs", ""))
            assert "duplicate-run-gate" not in str(jobs[name].get("if", ""))
