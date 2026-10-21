"""``kestrel update`` continues in a fresh interpreter after it installs (#3702).

On 2026-10-19 a host's update ran ``uv sync`` and then ``feature sync`` in the
same process. The sync had removed ``importlib_metadata``, which that process
had already imported and whose finder stdlib ``importlib.metadata`` still
dispatched to, so the sync died on its first version lookup with every feature
package pruned. Pinned here: nothing runs after the install in the process
that imported the replaced code; the continuation is the same update minus the
steps already done, and its exit status is the update's.

The pull does NOT hand off. Until the install, the pulled code runs against the
packages the old code was installed with, and it need not import against them:
an end-to-end run across the same 2026-10-19 revisions died importing
``fastapi.routing.iter_route_contexts`` in exactly that window. So the
interpreter that imported the old code drives the install, and the manifest
preflight runs before the pull, against the lock the pull will land.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

import pytest

from kestrel_sovereign import cli, cli_lifecycle

INCOMING = "c0ffee0000000000000000000000000000000000"


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: tmp_path)
    monkeypatch.setattr(cli, "_project_dir_is_git", lambda _: True)
    monkeypatch.setattr(cli, "_git_working_tree_dirty", lambda _: (False, ""))
    return tmp_path


@pytest.fixture
def manifest(checkout):
    path = checkout / ".kestrel-host-features.toml"
    path.write_text('[[feature]]\nname = "arize-phoenix"\npypi = ""\n')
    return path


@pytest.fixture
def steps(checkout, manifest, monkeypatch):
    """Record every step the PARENT runs, and the continuation it hands off to."""
    calls = []
    monkeypatch.setattr(cli, "_host_manifest_path", lambda args: manifest)
    monkeypatch.setattr(
        cli, "_run_git_fetch", lambda _: calls.append("fetch") or (0, "")
    )
    monkeypatch.setattr(cli, "_pull_target_revision", lambda _: INCOMING)
    monkeypatch.setattr(
        cli, "_run_sync_preflight",
        lambda path, lock_revision=None, **kw: calls.append(("preflight", lock_revision)) or 0,
    )
    monkeypatch.setattr(
        cli, "_run_git_pull", lambda _: calls.append("pull") or (0, "")
    )
    monkeypatch.setattr(
        cli, "_run_uv_sync", lambda _: calls.append("uv sync") or (0, "")
    )
    monkeypatch.setattr(
        cli, "cmd_feature_sync", lambda args: calls.append("feature sync") or 0
    )
    monkeypatch.setattr(
        cli_lifecycle, "_run_feature_reconcile",
        lambda *a, **kw: calls.append("reconcile") or 0,
    )
    monkeypatch.setattr(cli_lifecycle, "_report_core_lock", lambda: None)
    monkeypatch.setattr(
        cli, "cmd_restart", lambda args: calls.append("restart") or 0
    )
    return calls


def _record_continuation(monkeypatch, calls, rc=0):
    seen = []

    def continuation(argv):
        calls.append("continue")
        seen.append(argv)
        return rc

    monkeypatch.setattr(cli, "_run_update_continuation", continuation)
    return seen


def _parse(argv):
    return cli.build_parser().parse_args(argv)


def _update(*argv):
    return cli.cmd_update(_parse(["update", *argv]))


def test_after_the_install_nothing_more_runs_in_this_process(steps, monkeypatch):
    seen = _record_continuation(monkeypatch, steps, rc=0)

    assert _update() == 0

    assert steps == [
        "fetch", ("preflight", INCOMING), "pull", "uv sync", "continue",
    ]
    (argv,) = seen
    assert argv[-3:] == ["--no-pull", "--no-install", "--after-install"]
    continued = _parse(argv)
    assert (continued.pull, continued.install) == (False, False)
    assert continued.features and continued.restart and continued.after_install


def test_the_continuation_exit_status_is_the_update_exit_status(steps, monkeypatch):
    _record_continuation(monkeypatch, steps, rc=7)

    assert _update() == 7


def test_without_an_install_nothing_hands_off(steps, monkeypatch):
    """The pull alone does not: the code it lands may not import yet."""
    _record_continuation(monkeypatch, steps)

    assert _update("--no-install") == 0

    assert steps == ["pull", "feature sync", "reconcile", "restart"]


def test_a_dry_run_never_hands_off_and_still_checks_the_incoming_lock(
    steps, monkeypatch,
):
    _record_continuation(monkeypatch, steps)

    assert _update("--dry-run") == 0

    # Read-only: the remote-tracking refs as last fetched, no fetch; the
    # reconcile step previews its plan.
    assert steps == [("preflight", INCOMING), "reconcile"]


def test_a_refused_preflight_stops_the_update_before_the_pull(steps, monkeypatch):
    _record_continuation(monkeypatch, steps)
    monkeypatch.setattr(
        cli, "_run_sync_preflight",
        lambda path, lock_revision=None, **kw: steps.append(("preflight", lock_revision)) or 1,
    )

    assert _update() == 1

    assert steps == ["fetch", ("preflight", INCOMING)]


def test_without_a_pull_the_preflight_reads_the_lock_on_disk(steps, monkeypatch):
    _record_continuation(monkeypatch, steps)

    assert _update("--no-pull") == 0

    assert steps[:2] == [("preflight", None), "uv sync"]


def test_an_empty_manifest_has_nothing_to_preflight(steps, manifest, monkeypatch):
    manifest.write_text("")
    _record_continuation(monkeypatch, steps)

    assert _update() == 0

    assert steps == ["pull", "uv sync", "continue"]


def test_the_phases_run_every_step_once_in_order(steps):
    """End to end through the real parser, with the continuation re-entered here."""
    assert _update("--allow-dirty", "--startup-timeout", "75.5") == 0

    assert steps == [
        "fetch", ("preflight", INCOMING), "pull", "uv sync",
        "feature sync", "reconcile", "restart",
    ]


def test_a_continuation_still_judges_every_agent(steps, monkeypatch):
    """With ``--no-features`` the continuation changes nothing itself, but the
    install before it did: the restart must still check every agent."""
    restarts = []
    monkeypatch.setattr(cli, "cmd_restart", lambda args: restarts.append(args) or 0)

    assert _update("Kite", "--no-features") == 0

    (restart,) = restarts
    assert restart.name == "Kite"
    assert restart.constitution_check_all_agents is True


# --- the argv the continuation is handed -----------------------------------


def _update_parser():
    subparsers = argparse.ArgumentParser().add_subparsers()
    cli_lifecycle.add_lifecycle_subparsers(subparsers)
    return subparsers.choices["update"]


def _non_default_argvs():
    """One argv per way each update action can differ from its default.

    Generated from the parser, so a flag added later is covered without
    editing this list, and :func:`cli_lifecycle._update_argv` must render it.
    """
    for action in _update_parser()._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if not action.option_strings:
            yield ["Kite"]
        elif isinstance(action, argparse.BooleanOptionalAction):
            yield [action.option_strings[0]]
            yield [action.option_strings[1]]
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            yield [action.option_strings[0]]
        else:
            yield [action.option_strings[0], "12.5"]


@pytest.mark.parametrize("flags", list(_non_default_argvs()), ids=shlex.join)
def test_every_update_flag_reaches_the_continuation(flags):
    parsed = _parse(["update", *flags])

    assert _parse(cli_lifecycle._update_argv(parsed)) == parsed


def test_flags_combine_in_the_continuation():
    parsed = _parse([
        "update", "Kite", "--no-pull", "--prefer-pypi", "--no-uv-sync",
        "--continue-on-error", "--manifest", "/m m.toml", "--force",
        "--allow-constitution-safe-mode", "--startup-timeout", "9",
    ])

    assert _parse(cli_lifecycle._update_argv(parsed)) == parsed


def test_an_action_it_cannot_render_is_refused_not_dropped(monkeypatch):
    real = cli_lifecycle.add_lifecycle_subparsers

    def with_an_append_flag(subparsers):
        real(subparsers)
        subparsers.choices["update"].add_argument("--tag", action="append")

    monkeypatch.setattr(cli_lifecycle, "add_lifecycle_subparsers", with_an_append_flag)

    with pytest.raises(TypeError, match="--tag"):
        cli_lifecycle._update_argv(argparse.Namespace(tag=["x"]))


# --- the spawn itself --------------------------------------------------------


def test_the_continuation_is_this_venvs_kestrel_with_inherited_stdio(monkeypatch):
    """The real spawn, minus ``update``: ``--version`` exits before any step."""
    monkeypatch.setattr(
        cli, "_run_update_continuation", cli_lifecycle._run_update_continuation,
    )
    runs = []
    real_run = subprocess.run

    def recording_run(command, **kwargs):
        runs.append((command, kwargs))
        return real_run(command, **kwargs)

    monkeypatch.setattr(cli_lifecycle.subprocess, "run", recording_run)

    assert cli._run_update_continuation(["--version"]) == 0

    ((command, kwargs),) = runs
    assert command == [
        sys.executable, "-P", "-m", "kestrel_sovereign.cli", "--version",
    ]
    # No capture, env, or cwd override: the child writes to the operator's
    # terminal and resolves the project exactly as this process did.
    assert set(kwargs) == {"check"}


def test_a_child_killed_by_a_signal_exits_as_a_shell_would(monkeypatch):
    monkeypatch.setattr(
        cli_lifecycle.subprocess, "run",
        lambda command, **kw: subprocess.CompletedProcess(command, -15),
    )

    assert cli_lifecycle._run_update_continuation(["update"]) == 143


def test_an_interpreter_that_will_not_start_says_how_to_finish(monkeypatch, capsys):
    def refuse(command, **kw):
        raise OSError("no such file")

    monkeypatch.setattr(cli_lifecycle.subprocess, "run", refuse)

    assert cli_lifecycle._run_update_continuation(["update", "--no-pull"]) == 1
    assert "kestrel update --no-pull" in capsys.readouterr().err


def test_the_continuation_ignores_a_checkout_in_the_working_directory(
    monkeypatch, tmp_path,
):
    """Run from inside another checkout, ``python -m`` would import ITS
    ``kestrel_sovereign`` ahead of the venv's; the console script never did."""
    impostor = tmp_path / "kestrel_sovereign"
    impostor.mkdir()
    (impostor / "__init__.py").write_text("")
    (impostor / "cli.py").write_text("import sys\nsys.exit(42)\n")
    monkeypatch.chdir(tmp_path)

    assert cli_lifecycle._run_update_continuation(["--version"]) == 0


@pytest.mark.parametrize(
    ("flags", "fetch", "allow_dirty"),
    [((), True, False), (("--dry-run",), False, False), (("--allow-dirty",), True, True)],
)
def test_the_preflight_pulls_as_the_update_would(steps, monkeypatch, flags, fetch, allow_dirty):
    """A dry run fetches nothing; ``--allow-dirty`` reaches the pull the sync
    makes of a declared core, as it reaches the sync itself."""
    _record_continuation(monkeypatch, steps)
    seen = []
    monkeypatch.setattr(
        cli, "_run_sync_preflight", lambda path, **kw: seen.append(kw) or 0,
    )

    assert _update(*flags) == 0

    ((kw),) = seen
    assert (kw["fetch"], kw["allow_dirty"]) == (fetch, allow_dirty)


def test_the_continuation_does_not_judge_the_revision_again(steps, monkeypatch):
    """The adoption gate ran before the install. Run again in the continuation,
    a refusal could only come after `uv sync` pruned the features it restores;
    the restart judges the installed code, after the restore."""
    gates = []
    monkeypatch.setattr(
        cli_lifecycle, "_constitution_adoption_gate",
        lambda *a, **kw: gates.append(kw["code_label"]) or 0,
    )

    assert _update() == 0

    assert len(gates) == 1
    assert "feature sync" in steps and "restart" in steps


def test_the_fetch_leaves_fetch_head_alone(tmp_path):
    """The coordinator checks out FETCH_HEAD after its own fetch; a preflight
    fetch in between must not retarget it."""
    def git(cwd, *args):
        return subprocess.run(
            ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True,
        ).stdout.strip()

    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "a")
    git(origin, "tag", "release")
    git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "b")
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(origin), str(clone))
    git(clone, "fetch", "-q", "origin", "release")
    fetched = git(clone, "rev-parse", "FETCH_HEAD")

    rc, _ = cli_lifecycle._run_git_fetch(clone)

    assert rc == 0
    assert git(clone, "rev-parse", "FETCH_HEAD") == fetched
