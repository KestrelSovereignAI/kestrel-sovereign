"""Decision table for the duplicate-run gate in ``.github/workflows/ci.yml``.

The gate is a step of the ``lint-and-imports`` job — the root of the graph, so
every tier can read its output regardless of how the tiers are wired to each
other. It decides whether the expensive tiers (``integration-tests`` and
``llm-tests``) run at all. It exists because a ``kestrel-talon`` branch matches
BOTH the ``issue-*`` push trigger and the ``pull_request`` trigger, so once its
PR is open every commit fires two complete CI runs on the same SHA. See #2800.

It is shell embedded in YAML, which nothing else type-checks or exercises, and
an error in the *skip* direction silently stops testing real changes. Worst
case, it skips the integration tier inside publish.yml's release gate, where
skipped jobs still let the ``ci`` job report success and publishing proceeds.
So these tests run the ACTUAL script extracted from the workflow — not a copy
— with ``gh`` stubbed and the real ``env:`` block supplied.

The invariant: the gate skips in exactly one situation — a push to a branch
that already has an open, non-conflicting, same-repository PR onto a base this
workflow actually runs for (``main`` or ``epic/**``). Everything else runs,
including stacked PRs onto an uncovered base, fork PRs from a branch with a
colliding name, and every error path.

Three layers are covered, because a bug in any one of them disarms the gate:

1. the script's decisions (``TestSkips*`` / ``TestEveryOtherTriggerRuns`` / ``TestFailsOpen``)
2. the ``--jq`` filter, which runs inside ``gh`` and so is invisible to a
   stubbed ``gh`` (``TestPrQueryFilter``)
3. the wiring between producer and consumer — step id, ``outputs:`` key, and
   the consuming ``if:`` (``TestWiring``). A one-character typo here skips the
   expensive tier on every trigger while every decision test stays green.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

CI_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

GATE_STEP_ID = "dedupe-gate"
GATE_HOST_JOB = "lint-and-imports"
GATE_OUTPUT = "run_expensive"

# ``gh`` is stubbed, so these emit what the *filtered* query would return: one
# base branch per line. The filter itself is covered by TestPrQueryFilter.
GH_STUBS = {
    "no-pr": "gh() { :; }",
    "pr-onto-main": "gh() { echo main; }",
    "pr-onto-epic": "gh() { echo 'epic/semantic-kb'; }",
    "two-prs-onto-main": "gh() { printf 'main\\nmain\\n'; }",
    # Stacked PR: base is another issue-* branch, which the pull_request
    # trigger filter (main, epic/**) ignores — so no pull_request run exists
    # and this push run is the only coverage.
    "pr-onto-issue-branch": "gh() { echo 'issue-2793-ci-unchain-jobs'; }",
    "pr-onto-release-branch": "gh() { echo 'release/0.50'; }",
    "mixed-stacked-and-main": "gh() { printf 'issue-2793-x\\nmain\\n'; }",
    "api-failure": "gh() { return 1; }",
}


def _jobs() -> dict:
    return yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]


def _gate_step() -> dict:
    steps = _jobs()[GATE_HOST_JOB]["steps"]
    matching = [s for s in steps if s.get("id") == GATE_STEP_ID]
    assert len(matching) == 1, (
        f"expected exactly one step with id={GATE_STEP_ID!r} in {GATE_HOST_JOB}, "
        f"found {len(matching)}. The job output and the test harness both locate "
        f"the gate by that id."
    )
    return matching[0]


def _env_for(event: str, ref: str) -> dict:
    """The ``env:`` block the runner would hand the step.

    The script reads only these — nothing is interpolated into its body, which
    is the point: a hostile branch name cannot inject shell.
    """
    return {
        "EVENT_NAME": event,
        "REF": ref,
        "REF_NAME": re.sub(r"^refs/(heads|tags)/", "", ref),
        "REPO": "KestrelSovereignAI/kestrel-sovereign",
        "REPO_OWNER": "KestrelSovereignAI",
        "GH_TOKEN": "stub",
    }


def _decide(event: str, ref: str, gh_mode: str) -> str:
    """Run the real gate script and return its decision."""
    script = _gate_step()["run"]
    assert "${{" not in script, (
        "the gate interpolates an Actions expression into its script body; route "
        f"it through env: instead, or a branch name can inject shell:\n{script}"
    )

    handle = tempfile.NamedTemporaryFile("w", delete=False, suffix=".env")
    handle.close()
    try:
        proc = subprocess.run(
            # -e matches the runner's default shell (`bash -e {0}`).
            ["bash", "-e", "-c", GH_STUBS[gh_mode] + "\n" + script],
            env={**os.environ, "GITHUB_OUTPUT": handle.name, **_env_for(event, ref)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        written = Path(handle.name).read_text()
    finally:
        os.unlink(handle.name)

    assert proc.returncode == 0, (
        f"gate exited {proc.returncode}. It runs inside {GATE_HOST_JOB}, a REQUIRED "
        f"status check, so a non-zero exit blocks merges.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    matches = re.findall(rf"^{GATE_OUTPUT}=(\w+)$", written, re.MULTILINE)
    assert matches, f"gate wrote no decision; GITHUB_OUTPUT was:\n{written!r}"
    assert len(matches) == 1, f"gate wrote {len(matches)} decisions: {matches}"
    return matches[0]


class TestSkipsOnlyTheDuplicate:
    """The single case the gate exists to suppress."""

    @pytest.mark.parametrize(
        "gh_mode",
        ["pr-onto-main", "pr-onto-epic", "two-prs-onto-main", "mixed-stacked-and-main"],
    )
    def test_branch_push_with_covered_pr_skips(self, gh_mode):
        assert _decide("push", "refs/heads/issue-2800-dedupe", gh_mode) == "false"

    def test_branch_push_before_pr_exists_runs(self):
        """The pre-PR window is real: talon opens its PR 35 min to 2.7 h after the
        first push, so this run is the only signal until then."""
        assert _decide("push", "refs/heads/issue-2800-dedupe", "no-pr") == "true"


class TestPrMustActuallyTriggerThisWorkflow:
    """An open PR only makes a push redundant if that PR runs this workflow.

    ``pull_request`` is filtered to base branches ``main`` and ``epic/**``. A PR
    onto anything else produces no run, so suppressing the push would leave the
    commit with no integration coverage. Stacked PRs are the realistic case.
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
            # publish.yml calls this workflow as its release gate. A tag push
            # arrives with the caller's tag ref; a workflow_dispatch release
            # from main arrives as event=workflow_dispatch.
            ("push", "refs/tags/v0.49.5", "no-pr"),
            ("push", "refs/tags/v0.49.5", "pr-onto-main"),
        ],
    )
    def test_runs(self, event, ref, gh_mode):
        assert _decide(event, ref, gh_mode) == "true"


class TestFailsOpen:
    """Any uncertainty must run the tests, never skip them."""

    def test_unusable_pr_query_still_runs(self):
        assert _decide("push", "refs/heads/issue-2800-dedupe", "api-failure") == "true"


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
class TestPrQueryFilter:
    """The ``--jq`` filter runs inside ``gh``, so a stubbed ``gh`` never exercises
    it. Run it against representative ``gh pr list`` JSON instead.

    It has to reject two classes of PR that ``--head`` alone cannot, because
    ``--head`` matches on branch NAME only and takes no owner qualifier.

    One caveat: ``gh --jq`` evaluates with gojq (embedded in gh), while this
    runs the ``jq`` binary. ``env.*`` and ``select`` behave identically in
    both, so the filter is faithful today — but a filter using a
    gojq/jq-divergent construct would pass here and misbehave in CI.
    """

    @staticmethod
    def _filter(prs: list[dict], owner: str = "KestrelSovereignAI") -> list[str]:
        script = _gate_step()["run"]
        match = re.search(r"pr_filter='(.*?)'", script, re.DOTALL)
        assert match, f"could not locate pr_filter='…' in the gate script:\n{script}"
        proc = subprocess.run(
            ["jq", "-r", match.group(1)],
            input=json.dumps(prs),
            env={**os.environ, "REPO_OWNER": owner},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"jq failed: {proc.stderr}"
        return proc.stdout.split()

    @staticmethod
    def _pr(base="main", owner="KestrelSovereignAI", mergeable="MERGEABLE"):
        return {
            "baseRefName": base,
            "headRepositoryOwner": {"login": owner},
            "mergeable": mergeable,
        }

    def test_ordinary_pr_is_kept(self):
        assert self._filter([self._pr()]) == ["main"]

    def test_fork_pr_with_colliding_branch_name_is_rejected(self):
        """kestrel-sovereign is public. Anyone can fork it, push a branch named
        exactly like ours, and open a PR onto main. ``gh pr list --head`` would
        return it, and without this filter their PR would suppress our run —
        denial-of-testing with no write access required."""
        assert self._filter([self._pr(owner="some-fork-account")]) == []

    def test_conflicting_pr_is_rejected(self):
        """A CONFLICTING PR has no computable merge ref, so no pull_request run
        covers the SHA and the push run is the only remaining signal."""
        assert self._filter([self._pr(mergeable="CONFLICTING")]) == []

    def test_unknown_mergeability_is_kept(self):
        """GitHub reports UNKNOWN while it lazily computes the merge ref, which
        is the normal answer right after the push that triggered this gate;
        treating it as conflicting would run the duplicate nearly every time.

        The cost is bounded but real: if the PR then turns out to conflict, no
        pull_request run is created and that commit gets no integration or llm
        coverage. Neither is a required check and the conflict-resolving push
        re-runs everything, so the exposure is one commit wide."""
        assert self._filter([self._pr(mergeable="UNKNOWN")]) == ["main"]

    def test_keeps_only_the_qualifying_pr_among_several(self):
        prs = [
            self._pr(base="issue-2793-x"),
            self._pr(owner="fork-account"),
            self._pr(mergeable="CONFLICTING"),
            self._pr(base="main"),
        ]
        assert self._filter(prs) == ["issue-2793-x", "main"]


class TestWiring:
    """The producer/consumer link. Every assertion here corresponds to a mutation
    that leaves all the decision tests above green while disarming the gate."""

    def test_output_is_wired_to_the_gate_step(self):
        """A typo here (``steps.dcide``) yields an empty output on every run."""
        job = _jobs()[GATE_HOST_JOB]
        assert job["outputs"][GATE_OUTPUT] == (
            "${{ steps." + GATE_STEP_ID + ".outputs." + GATE_OUTPUT + " }}"
        )

    def test_gate_step_exists_exactly_once(self):
        assert _gate_step()["run"].strip()

    def test_gate_host_job_runs_on_every_trigger(self):
        """An ``if:`` on the host job would skip the gate — and its dependents."""
        assert "if" not in _jobs()[GATE_HOST_JOB]

    def test_integration_tests_consults_the_gate(self):
        integration = _jobs()["integration-tests"]
        assert GATE_HOST_JOB in str(integration["needs"])
        assert f"needs.{GATE_HOST_JOB}.outputs.{GATE_OUTPUT}" in integration["if"]

    def test_consumer_fails_open(self):
        """``!= 'false'`` not ``== 'true'``: if the output is ever empty, the
        tier must still run. ``== 'true'`` would skip it everywhere, including
        publish.yml's release gate, where skipped jobs still report success."""
        integration = _jobs()["integration-tests"]
        assert f"needs.{GATE_HOST_JOB}.outputs.{GATE_OUTPUT} != 'false'" in integration["if"]
        assert f"{GATE_OUTPUT} == 'true'" not in integration["if"]

    def test_llm_tests_states_its_conditions_outright(self):
        """llm-tests runs in parallel now, so it no longer inherits anything from
        integration-tests. Both the main-push skip and the dedupe clause have to
        be stated on it directly, byte-identical to integration-tests, or it
        starts running on main pushes and ignoring the gate."""
        jobs = _jobs()
        llm, integration = jobs["llm-tests"], jobs["integration-tests"]
        assert GATE_HOST_JOB in str(llm["needs"])
        assert " ".join(llm["if"].split()) == " ".join(integration["if"].split())

    def test_required_checks_are_not_themselves_gated(self):
        """lint-and-imports and unit-tests are required status checks. A skipped
        job still posts a check run, so neither may be conditioned on the gate."""
        jobs = _jobs()
        for name in ("lint-and-imports", "unit-tests"):
            assert GATE_OUTPUT not in str(jobs[name].get("if", ""))


class TestTierConditions:
    """The `if:` on both expensive tiers, clause by clause.

    Three separate comment paragraphs in ci.yml call this condition
    load-bearing, and until now nothing tested it. Each assertion below
    corresponds to a mutation that left every other test green.
    """

    TIERS = ("integration-tests", "llm-tests")

    @pytest.mark.parametrize("tier", TIERS)
    def test_skips_main_branch_pushes(self, tier):
        """Dropping this clause reruns the heaviest tiers on every squash-merge
        to main — the regression #682 removed."""
        condition = " ".join(_jobs()[tier]["if"].split())
        assert "github.ref != 'refs/heads/main'" in condition

    @pytest.mark.parametrize("tier", TIERS)
    def test_keeps_the_event_name_clause(self, tier):
        """Reducing the condition to the ref check alone would skip both tiers
        inside a workflow_dispatch release from main — where github.ref IS
        refs/heads/main — letting the `ci` gate report success on a partial run
        and publishing to PyPI."""
        condition = " ".join(_jobs()[tier]["if"].split())
        assert "github.event_name != 'push'" in condition
        assert "(github.event_name != 'push' || github.ref != 'refs/heads/main')" in condition

    @pytest.mark.parametrize("tier", TIERS)
    def test_tiers_are_not_chained_to_each_other(self, tier):
        """Re-adding unit-tests (or integration-tests) to a tier's needs: would
        silently restore the serial chain and give back the ~7 min saving."""
        needs = str(_jobs()[tier].get("needs"))
        assert needs == GATE_HOST_JOB, (
            f"{tier} must depend on {GATE_HOST_JOB} alone to stay parallel; got {needs}"
        )


class TestGateQueryInvocation:
    """The `gh pr list` invocation itself. Invisible to the decision tests,
    because those stub `gh` out entirely."""

    @staticmethod
    def _script() -> str:
        return _gate_step()["run"]

    def test_queries_only_open_prs(self):
        """`--state all` would mean that once any PR from a branch name closes
        or merges, every future push to that name skips the expensive tiers
        forever."""
        assert "--state open" in self._script()

    def test_queries_this_branch(self):
        assert '--head "$REF_NAME"' in self._script()

    def test_requests_the_fields_the_filter_needs(self):
        script = self._script()
        for field in ("baseRefName", "headRepositoryOwner", "mergeable"):
            assert field in script, f"gh pr list must request {field}"


class TestFailFast:
    """The tiers run in parallel, so the old `needs:` chain no longer stops the
    12.8-minute integration tier when the unit tier goes red. A dedicated job
    restores that, and its shape is load-bearing."""

    JOB = "cancel-expensive-tiers-when-unit-fails"

    def test_exists_and_triggers_on_unit_failure(self):
        job = _jobs()[self.JOB]
        assert job["needs"] == "unit-tests"
        condition = " ".join(job["if"].split())
        # An `if:` with no status function is implicitly ANDed with success(),
        # so the bare comparison would never fire.
        assert "always()" in condition
        assert "needs.unit-tests.result == 'failure'" in condition

    def test_does_not_fire_on_a_lint_failure(self):
        """`failure()` walks transitive ancestors, so a lint-and-imports failure
        would trigger it — cancelling the run for no benefit, since every tier
        is already skipped by needs:, and converting a clean `failure`
        conclusion into `cancelled`."""
        condition = " ".join(_jobs()[self.JOB]["if"].split())
        assert condition != "failure()"
        assert "needs.unit-tests.result" in condition

    def test_is_disabled_in_the_release_gate(self):
        """Jobs of a called workflow belong to the CALLER's run, so github.run_id
        here is publish.yml's. Cancelling would kill the in-flight clean-install
        matrix — the only place the billed macOS/Windows axis runs per release."""
        condition = " ".join(_jobs()[self.JOB]["if"].split())
        assert "inputs.ref == ''" in condition

    def test_actually_calls_the_cancel_endpoint(self):
        """Replacing the POST with an echo leaves every other assertion green
        and the fail-fast silently dead."""
        step = _jobs()[self.JOB]["steps"][0]
        assert "gh api --method POST" in step["run"]
        assert "actions/runs/$RUN_ID/cancel" in step["run"]

    def test_cancels_by_run_id_not_run_number(self):
        """github.run_number is a per-workflow counter, not an API id; the cancel
        would 404."""
        assert _jobs()[self.JOB]["steps"][0]["env"]["RUN_ID"] == "${{ github.run_id }}"

    def test_is_one_directional(self):
        """It must depend on unit-tests ONLY. Adding integration-tests would make
        an integration failure cancel the still-running unit tier — throwing away
        the result the author needs next, and 8 of 11 recent failures were
        integration-tests itself."""
        assert _jobs()[self.JOB]["needs"] == "unit-tests"

    def test_is_a_separate_job_not_a_step_in_unit_tests(self):
        """Cancelling from inside a still-running job marks that job `cancelled`,
        hiding which tier broke. As its own job, unit-tests has already concluded
        `failure` before the cancel lands."""
        assert "cancel" not in str(_jobs()["unit-tests"]["steps"]).lower()

    def test_has_the_permission_it_needs(self):
        assert _jobs()[self.JOB]["permissions"]["actions"] == "write"

    def test_no_test_tier_can_outlive_the_cancel(self):
        """Every expensive tier must start no earlier than the gate host, so the
        cancel can reach it. If a tier ever depended on nothing, it could finish
        before unit-tests failed."""
        jobs = _jobs()
        for tier in ("integration-tests", "llm-tests"):
            assert jobs[tier]["needs"], f"{tier} must declare a dependency"
