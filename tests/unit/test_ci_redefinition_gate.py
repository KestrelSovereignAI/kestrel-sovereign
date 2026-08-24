"""The F811 gate in ``.github/workflows/ci.yml``.

``lint-and-imports`` was named for a linter it did not run. Its checks were
``compileall`` and an import validation, and a module with two top-level
definitions of the same name passes both: duplicate ``def`` is valid Python and
the module imports fine. The second definition wins and the first is dead code.

That is exactly what a three-way merge produces when the same function was
added on both sides — a branch's unsquashed commit against main's squashed
copy — and it reached main once (#3065), caught only because a behavioural test
happened to exercise the shadowed path. See #3067.

The gate is shell embedded in YAML, which nothing else type-checks, and its
failure mode is silent: every path that cannot resolve a base commit passes.
Get the base resolution wrong and the gate is *inert* rather than absent, which
is worse — a green check reporting on an empty set. So these tests run the
ACTUAL script extracted from the workflow, against REAL git repositories with
real merge-bases, stubbing only ``uvx`` so the argv handed to ruff is
observable. Nothing about the gate is restated here.

Four layers are covered, because a bug in any one disarms it:

1. base resolution per trigger (``TestChangedSet`` / ``TestUnresolvableBase``)
2. the changed set itself — deletions, non-Python files, merge-base semantics
3. that a finding actually fails the step (``TestFindingFailsTheStep``)
4. wiring — step id, host job, ``fetch-depth``, rule selection (``TestWiring``)

Layer 4 matters most. ``fetch-depth: 0`` is what makes the diff possible at
all; with the default depth-1 checkout every run resolves to "nothing changed"
and every test above still passes, because they supply their own repository.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

CI_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

GATE_STEP_ID = "f811-gate"
GATE_HOST_JOB = "lint-and-imports"

ZERO_SHA = "0" * 40
# A well-formed SHA that no repository built here will contain.
ABSENT_SHA = "dead" * 10


def _jobs() -> dict:
    return yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]


def _gate_step() -> dict:
    steps = _jobs()[GATE_HOST_JOB]["steps"]
    matching = [s for s in steps if s.get("id") == GATE_STEP_ID]
    assert len(matching) == 1, (
        f"expected exactly one step with id={GATE_STEP_ID!r} in {GATE_HOST_JOB}, "
        f"found {len(matching)}. This harness locates the gate by that id."
    )
    return matching[0]


def _ruff_version() -> str:
    return str(_gate_step()["env"]["RUFF_VERSION"])


# ---------------------------------------------------------------------------
# Resolving the step's own env: block against a modelled Actions context
# ---------------------------------------------------------------------------

_EXPRESSION = re.compile(r"^\$\{\{\s*([a-z_]+(?:\.[a-zA-Z_]+)+)\s*\}\}$")


def _actions_context(
    event: str,
    ref: str,
    *,
    before: str = "",
    pr_base_sha: str = "",
    called_ref: str = "",
) -> dict:
    """A faithful Actions context for one trigger.

    The absent halves are modelled as absent, not as empty strings, so the
    resolver below has to reproduce GitHub's own "missing path renders empty"
    behaviour rather than being handed it: ``github.event.before`` does not
    exist on a ``pull_request`` event, and ``github.event.pull_request`` does
    not exist on a ``push``.
    """
    event_payload: dict = {}
    if before:
        event_payload["before"] = before
    if pr_base_sha:
        event_payload["pull_request"] = {"base": {"sha": pr_base_sha}}
    return {
        "github": {
            "event_name": event,
            "ref": ref,
            "ref_name": re.sub(r"^refs/(heads|tags)/", "", ref),
            "repository": "KestrelSovereignAI/kestrel-sovereign",
            "event": event_payload,
        },
        "inputs": {"ref": called_ref},
    }


def _lookup(context: dict, dotted: str) -> str:
    """GitHub renders a missing context path as the empty string."""
    node = context
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return ""
        node = node[part]
    return str(node)


def _env_for(context: dict) -> dict:
    """Resolve the gate's OWN ``env:`` block — never a hand-written stand-in.

    The env block is the entire interface between the workflow and the script.
    Fabricating it would leave every expression in it unguarded: swapping
    ``github.event.before`` for ``github.event.after`` would diff a commit
    against itself and pass on everything, while every decision test below
    stayed green.
    """
    resolved = {}
    for name, expression in _gate_step()["env"].items():
        text = str(expression).strip()
        match = _EXPRESSION.match(text)
        if not match:
            assert "${{" not in text, (
                f"gate env {name}={expression!r} embeds an expression this "
                f"harness cannot resolve faithfully"
            )
            resolved[name] = text  # a literal, e.g. the pinned RUFF_VERSION
            continue
        resolved[name] = _lookup(context, match.group(1))
    return resolved


# ---------------------------------------------------------------------------
# Real git repositories
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(repo),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        },
    )
    return proc.stdout.strip()


def _commit(repo: Path, message: str, files: dict[str, str | None]) -> str:
    for name, body in files.items():
        target = repo / name
        if body is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    return path


# ---------------------------------------------------------------------------
# Running the real gate script
# ---------------------------------------------------------------------------


class GateRun:
    def __init__(self, proc: subprocess.CompletedProcess, argv_file: Path):
        self.proc = proc
        self._argv_file = argv_file

    @property
    def exit_code(self) -> int:
        return self.proc.returncode

    @property
    def output(self) -> str:
        return self.proc.stdout + self.proc.stderr

    @property
    def ran_ruff(self) -> bool:
        return self._argv_file.exists()

    @property
    def argv(self) -> list[str]:
        assert self.ran_ruff, f"the gate never invoked ruff:\n{self.output}"
        raw = self._argv_file.read_bytes().decode()
        return [part for part in raw.split("\0") if part]

    @property
    def linted(self) -> list[str]:
        """The paths handed to ruff — everything after the ``--`` separator."""
        argv = self.argv
        return sorted(argv[argv.index("--") + 1 :])


def _run_gate(
    repo: Path,
    tmp_path: Path,
    context: dict,
    *,
    ruff_exit: int = 0,
    stub_ruff: bool = True,
) -> GateRun:
    script = _gate_step()["run"]
    assert "${{" not in script, (
        "the gate interpolates an Actions expression into its script body; route "
        f"it through env: instead, or a branch name can inject shell:\n{script}"
    )

    argv_file = tmp_path / "ruff-argv"
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir(exist_ok=True)

    path = "/usr/bin:/bin:/usr/local/bin"
    if stub_ruff:
        # Only the external tool is stubbed. A shell function shadows the PATH
        # lookup, so the gate's own quoting and argument order are what gets
        # recorded here.
        prelude = f'uvx() {{ printf "%s\\0" "$@" > "{argv_file}"; return {ruff_exit}; }}\n'
    else:
        prelude = ""
        uvx = shutil.which("uvx")
        assert uvx, "caller must skip when uvx is unavailable"
        path = f"{Path(uvx).parent}:{path}"

    proc = subprocess.run(
        # -e matches the runner's default shell (`bash -e {0}`).
        ["bash", "-e", "-c", prelude + script],
        cwd=repo,
        env={
            "PATH": path,
            "HOME": str(repo),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "RUNNER_TEMP": str(runner_temp),
            **_env_for(context),
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    return GateRun(proc, argv_file)


BASE_TREE = {
    "pkg/__init__.py": "",
    "pkg/mod.py": "def a():\n    return 1\n",
    "pkg/doomed.py": "def gone():\n    return 1\n",
    "README.md": "# readme\n",
}


class TestChangedSet:
    """What the gate hands to ruff, per trigger."""

    def test_branch_push_lints_only_what_the_push_changed(self, repo, tmp_path):
        before = _commit(repo, "base", BASE_TREE)
        _commit(
            repo,
            "change",
            {"pkg/mod.py": "def a():\n    return 2\n", "pkg/added.py": "x = 1\n"},
        )

        run = _run_gate(
            repo,
            tmp_path,
            _actions_context("push", "refs/heads/issue-3067-gate", before=before),
        )

        assert run.exit_code == 0, run.output
        assert run.linted == ["pkg/added.py", "pkg/mod.py"]

    def test_pull_request_excludes_commits_that_landed_on_the_base(
        self, repo, tmp_path
    ):
        """The base moving after the branch point must not widen the changed set.

        Without merge-base semantics a long-lived PR lints the whole base
        branch's churn, and the 22 pre-existing F811 sites start failing PRs
        that never touched them — which is precisely the outcome the
        changed-files scope exists to avoid.
        """
        _commit(repo, "base", BASE_TREE)
        _git(repo, "checkout", "-q", "-b", "feature")
        head = _commit(repo, "pr work", {"pkg/mod.py": "def a():\n    return 2\n"})

        _git(repo, "checkout", "-q", "main")
        base_sha = _commit(repo, "moved on", {"pkg/unrelated.py": "y = 1\n"})

        # actions/checkout gives a pull_request the MERGE commit, not the head.
        _git(repo, "checkout", "-q", "-b", "pr-merge")
        _git(repo, "merge", "-q", "--no-ff", "-m", "merge", head)

        run = _run_gate(
            repo,
            tmp_path,
            _actions_context(
                "pull_request", "refs/pull/1/merge", pr_base_sha=base_sha
            ),
        )

        assert run.exit_code == 0, run.output
        assert run.linted == ["pkg/mod.py"]

    def test_non_python_changes_alone_invoke_nothing(self, repo, tmp_path):
        before = _commit(repo, "base", BASE_TREE)
        _commit(repo, "docs", {"README.md": "# rewritten\n"})

        run = _run_gate(
            repo, tmp_path, _actions_context("push", "refs/heads/docs", before=before)
        )

        assert run.exit_code == 0, run.output
        assert not run.ran_ruff

    def test_deleted_python_files_are_not_linted(self, repo, tmp_path):
        before = _commit(repo, "base", BASE_TREE)
        _commit(
            repo,
            "delete one, touch another",
            {"pkg/doomed.py": None, "pkg/mod.py": "def a():\n    return 3\n"},
        )

        run = _run_gate(
            repo, tmp_path, _actions_context("push", "refs/heads/cleanup", before=before)
        )

        assert run.exit_code == 0, run.output
        assert run.linted == ["pkg/mod.py"]

    def test_deleting_the_only_python_change_invokes_nothing(self, repo, tmp_path):
        """A deletion-only push has no file left to lint, and ruff errors on a
        path that does not exist — so it must not be invoked at all."""
        before = _commit(repo, "base", BASE_TREE)
        _commit(repo, "delete", {"pkg/doomed.py": None})

        run = _run_gate(
            repo, tmp_path, _actions_context("push", "refs/heads/cleanup", before=before)
        )

        assert run.exit_code == 0, run.output
        assert not run.ran_ruff


class TestUnresolvableBase:
    """Every state that cannot name a base passes, and passes deliberately.

    Erroring instead would fail a REQUIRED check on every new branch. The next
    ordinary push re-covers the same files against a base that does resolve.
    """

    @pytest.fixture
    def two_commits(self, repo):
        _commit(repo, "base", BASE_TREE)
        _commit(repo, "change", {"pkg/mod.py": "def a():\n    return 2\n"})
        return repo

    def test_first_push_of_a_branch(self, two_commits, tmp_path):
        run = _run_gate(
            two_commits,
            tmp_path,
            _actions_context("push", "refs/heads/brand-new", before=ZERO_SHA),
        )
        assert run.exit_code == 0, run.output
        assert not run.ran_ruff

    def test_force_push_leaves_a_base_this_clone_does_not_have(
        self, two_commits, tmp_path
    ):
        run = _run_gate(
            two_commits,
            tmp_path,
            _actions_context("push", "refs/heads/rebased", before=ABSENT_SHA),
        )
        assert run.exit_code == 0, run.output
        assert not run.ran_ruff

    def test_push_event_with_no_before_sha_at_all(self, two_commits, tmp_path):
        run = _run_gate(
            two_commits, tmp_path, _actions_context("push", "refs/heads/odd")
        )
        assert run.exit_code == 0, run.output
        assert not run.ran_ruff

    def test_tag_push_is_publish_yml_release_gate_not_a_change(
        self, two_commits, tmp_path
    ):
        """`gh`-style trap: a tag push carries a `before`, but it names whatever
        the tag ref pointed at — not this tree's parent."""
        before = _git(two_commits, "rev-parse", "HEAD~1")
        run = _run_gate(
            two_commits,
            tmp_path,
            _actions_context("push", "refs/tags/v0.52.1", before=before),
        )
        assert run.exit_code == 0, run.output
        assert not run.ran_ruff

    def test_workflow_call_pinned_to_a_release_sha(self, two_commits, tmp_path):
        before = _git(two_commits, "rev-parse", "HEAD~1")
        run = _run_gate(
            two_commits,
            tmp_path,
            _actions_context(
                "push", "refs/heads/main", before=before, called_ref="abc123"
            ),
        )
        assert run.exit_code == 0, run.output
        assert not run.ran_ruff

    def test_workflow_dispatch(self, two_commits, tmp_path):
        run = _run_gate(
            two_commits,
            tmp_path,
            _actions_context("workflow_dispatch", "refs/heads/main"),
        )
        assert run.exit_code == 0, run.output
        assert not run.ran_ruff


class TestFindingFailsTheStep:
    def test_nonzero_ruff_exit_fails_the_gate(self, repo, tmp_path):
        """A gate that reports a finding and still exits 0 is the whole bug."""
        before = _commit(repo, "base", BASE_TREE)
        _commit(repo, "change", {"pkg/mod.py": "def a():\n    return 2\n"})

        run = _run_gate(
            repo,
            tmp_path,
            _actions_context("push", "refs/heads/issue-3067", before=before),
            ruff_exit=1,
        )

        assert run.exit_code != 0, (
            "ruff reported a finding and the gate still exited 0 — the step "
            f"would go green on a duplicated definition:\n{run.output}"
        )

    def test_ruff_is_invoked_with_the_pinned_version_and_only_f811(
        self, repo, tmp_path
    ):
        before = _commit(repo, "base", BASE_TREE)
        _commit(repo, "change", {"pkg/mod.py": "def a():\n    return 2\n"})

        argv = _run_gate(
            repo,
            tmp_path,
            _actions_context("push", "refs/heads/issue-3067", before=before),
        ).argv

        assert argv[0] == f"ruff@{_ruff_version()}", (
            f"ruff must be version-pinned so a new release cannot change what "
            f"this required check means; got {argv[0]!r}"
        )
        assert "check" in argv
        assert argv[argv.index("--select") + 1] == "F811", (
            "F401 (~51 pre-existing hits) and F821 (~28) are untriaged against "
            "this tree; widening the selection here turns the gate red on work "
            "that did not cause it. See #3067."
        )


# The #3065 shape: git's three-way merge saw the branch's definition and
# main's as two independent additions and kept BOTH. The second one wins, so
# the guard in the first is present, unreachable, and shipped as dead code.
MERGED_DUPLICATE = (
    "def parse_issue_ref(ref):\n"
    "    if '/' not in ref:\n"
    "        return None, int(ref.lstrip('#'))\n"
    "    return ref, 0\n"
    "\n\n"
    "def parse_issue_ref(ref):\n"
    "    return ref, 0\n"
)

SINGLE_DEFINITION = "def parse_issue_ref(ref):\n    return ref, 0\n"


@pytest.fixture
def pinned_ruff():
    """The exact ruff the workflow pins, or a skip if it cannot be resolved."""
    if shutil.which("uvx") is None:
        pytest.skip("uvx unavailable; cannot resolve the pinned ruff")
    pinned = f"ruff@{_ruff_version()}"
    try:
        probe = subprocess.run(
            ["uvx", pinned, "--version"], capture_output=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # offline runner
        pytest.skip(f"could not resolve {pinned}: {exc}")
    if probe.returncode != 0:
        pytest.skip(f"could not resolve {pinned}: {probe.stderr.decode()}")
    return pinned


class TestRuffActuallyCatchesTheDefect:
    """The rule selection is only worth anything if F811 reports this shape."""

    @pytest.fixture
    def check(self, pinned_ruff, tmp_path):
        def run(source: str) -> subprocess.CompletedProcess:
            target = tmp_path / "sample.py"
            target.write_text(source)
            return subprocess.run(
                [
                    "uvx",
                    pinned_ruff,
                    "check",
                    "--no-cache",
                    "--select",
                    "F811",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

        return run

    def test_a_merge_keeping_both_copies_of_a_function_is_reported(self, check):
        result = check(MERGED_DUPLICATE)
        assert result.returncode != 0, (
            "ruff F811 did not report two top-level definitions of one name — "
            "the gate is selecting a rule that does not cover the defect it "
            f"was added for:\n{result.stdout}"
        )
        assert "F811" in result.stdout

    def test_a_single_definition_is_clean(self, check):
        result = check(SINGLE_DEFINITION)
        assert result.returncode == 0, result.stdout


class TestEndToEnd:
    """The whole chain, unstubbed: real git history, real ruff, real script.

    Every other test here proves one link. These prove they are connected —
    that a push carrying the #3065 defect turns this step red, and that the
    two things which must NOT turn it red do not.
    """

    def test_a_push_introducing_a_duplicate_definition_fails(
        self, pinned_ruff, repo, tmp_path
    ):
        before = _commit(repo, "base", {"pkg/issue_selection.py": SINGLE_DEFINITION})
        _commit(repo, "bad merge", {"pkg/issue_selection.py": MERGED_DUPLICATE})

        run = _run_gate(
            repo,
            tmp_path,
            _actions_context("push", "refs/heads/issue-3065", before=before),
            stub_ruff=False,
        )

        assert run.exit_code != 0, (
            "a push that duplicated a top-level definition passed the gate — "
            f"the exact defect #3067 was filed for:\n{run.output}"
        )
        assert "F811" in run.output

    def test_a_clean_push_passes(self, pinned_ruff, repo, tmp_path):
        before = _commit(repo, "base", {"pkg/issue_selection.py": SINGLE_DEFINITION})
        _commit(
            repo,
            "ordinary change",
            {"pkg/issue_selection.py": SINGLE_DEFINITION.replace("ref, 0", "ref, 1")},
        )

        run = _run_gate(
            repo,
            tmp_path,
            _actions_context("push", "refs/heads/ordinary", before=before),
            stub_ruff=False,
        )

        assert run.exit_code == 0, run.output

    def test_a_pre_existing_duplicate_in_an_untouched_file_does_not_fail_the_push(
        self, pinned_ruff, repo, tmp_path
    ):
        """The property the changed-files scope exists for.

        F811 reports 22 sites on the current tree (#3094). Whatever this gate
        does, it must not charge a 22-site cleanup to the next unrelated PR —
        that is why there is no full-tree scan and no `per-file-ignores`
        baseline.
        """
        before = _commit(
            repo,
            "base",
            {
                "pkg/legacy.py": MERGED_DUPLICATE,  # stands in for the 22
                "pkg/mod.py": "def a():\n    return 1\n",
            },
        )
        _commit(repo, "touch something else", {"pkg/mod.py": "def a():\n    return 2\n"})

        run = _run_gate(
            repo,
            tmp_path,
            _actions_context("push", "refs/heads/unrelated", before=before),
            stub_ruff=False,
        )

        assert run.exit_code == 0, (
            "a pre-existing F811 in a file this push never touched failed the "
            f"gate; the changed-file scope is not holding:\n{run.output}"
        )


class TestWiring:
    """A gate wired to nothing is a green check reporting on an empty set."""

    def test_gate_lives_in_the_job_every_tier_depends_on(self):
        _gate_step()  # asserts it is in GATE_HOST_JOB, exactly once

        dependents = {
            name
            for name, job in _jobs().items()
            if GATE_HOST_JOB
            in (
                [job["needs"]] if isinstance(job.get("needs"), str) else job.get("needs", [])
            )
        }
        assert dependents, (
            f"no job declares `needs: {GATE_HOST_JOB}`, so the gate no longer "
            f"blocks anything downstream"
        )

    def test_checkout_fetches_enough_history_to_diff(self):
        """Without this the gate is inert, not absent: every diff resolves to
        an empty changed set and the step passes on everything."""
        checkout = next(
            s
            for s in _jobs()[GATE_HOST_JOB]["steps"]
            if str(s.get("uses", "")).startswith("actions/checkout@")
        )
        assert checkout.get("with", {}).get("fetch-depth") == 0, (
            "the F811 gate diffs against a base commit; a depth-1 checkout has "
            "no ancestors and silently lints nothing"
        )

    def test_gate_cannot_be_waved_through(self):
        step = _gate_step()
        assert not step.get("continue-on-error"), (
            "continue-on-error turns the gate into a warning"
        )
        assert "if" not in step, (
            "the gate must run on every trigger; the unresolvable-base cases "
            "are handled inside the script, where they are commented"
        )

    def test_pinned_ruff_version_is_exact(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", _ruff_version()), (
            f"RUFF_VERSION={_ruff_version()!r} is not an exact pin; a range "
            f"lets a ruff release change what this required check means"
        )
