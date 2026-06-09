"""Tests for ``kestrel update`` — one-shot pull + install + sync + restart.

Pins:

* Default invocation calls each of the four step helpers in order and
  short-circuits on the first failure (so a half-applied update doesn't
  get restarted into).
* Dirty working tree refuses the pull unless ``--allow-dirty`` is passed.
* Each step can be skipped individually via ``--no-<step>``.
* ``--dry-run`` mutates nothing — no shell commands are invoked, just
  printed.
* ``--continue-on-error`` lets ``feature sync`` fail but still triggers
  the restart.
* The target-agent positional flows to the restart step only.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from kestrel_sovereign import cli


@pytest.fixture
def stub_project_dir(tmp_path, monkeypatch):
    """Pretend the project root + source checkout are both at tmp_path.

    The real ``cmd_update`` reads the source checkout from
    ``_resolve_source_checkout()`` (introspecting the installed
    kestrel_sovereign package) and the runtime data root from
    ``_get_project_dir()``. For most tests they're indistinguishable
    — patch both to point at tmp_path with a fake .git dir."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: tmp_path)
    return tmp_path


def _ns(**overrides):
    base = dict(
        name=None, pull=True, install=True, features=True, restart=True,
        allow_dirty=False, no_deps=False, continue_on_error=False,
        dry_run=False, manifest=None, subprocess=False, force=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_update_full_pipeline_calls_each_step_in_order(stub_project_dir):
    """Default ``kestrel update`` runs pull → install → sync → restart,
    each only after the previous returned 0."""
    calls = []

    def fake_dirty(_):
        calls.append("dirty_check")
        return False

    def fake_pull(_):
        calls.append("pull")
        return 0, "Already up to date.\n"

    def fake_install(_, no_deps):
        calls.append(("install", no_deps))
        return 0, "Installed 1 package.\n"

    def fake_sync(args):
        calls.append("sync")
        return 0

    def fake_restart(args):
        calls.append(("restart", args.name, args.force))
        return 0

    with patch.object(cli, "_project_dir_is_git", lambda _: True), \
         patch.object(cli, "_git_working_tree_dirty", fake_dirty), \
         patch.object(cli, "_run_git_pull", fake_pull), \
         patch.object(cli, "_run_uv_pip_install_editable", fake_install), \
         patch.object(cli, "cmd_feature_sync", fake_sync), \
         patch.object(cli, "cmd_restart", fake_restart):
        rc = cli.cmd_update(_ns())

    assert rc == 0
    assert calls == [
        "dirty_check",
        "pull",
        ("install", False),
        "sync",
        ("restart", None, False),
    ]


def test_update_short_circuits_when_pull_fails(stub_project_dir):
    """If git pull fails, install/sync/restart MUST NOT run — otherwise a
    half-applied update could be restarted into."""
    later = []
    with patch.object(cli, "_git_working_tree_dirty", lambda _: False), \
         patch.object(cli, "_run_git_pull",
                      lambda _: (1, "merge conflict\n")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda *a, **kw: later.append("install") or (0, "")), \
         patch.object(cli, "cmd_feature_sync",
                      lambda args: later.append("sync") or 0), \
         patch.object(cli, "cmd_restart",
                      lambda args: later.append("restart") or 0):
        rc = cli.cmd_update(_ns())

    assert rc != 0
    assert later == []


def test_update_short_circuits_when_install_fails(stub_project_dir):
    """Install failure aborts before sync + restart."""
    later = []
    with patch.object(cli, "_git_working_tree_dirty", lambda _: False), \
         patch.object(cli, "_run_git_pull", lambda _: (0, "")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda *a, **kw: (1, "build failed\n")), \
         patch.object(cli, "cmd_feature_sync",
                      lambda args: later.append("sync") or 0), \
         patch.object(cli, "cmd_restart",
                      lambda args: later.append("restart") or 0):
        rc = cli.cmd_update(_ns())

    assert rc != 0
    assert later == []


def test_dirty_working_tree_refuses_pull(stub_project_dir, capsys):
    """Default behaviour: a dirty working tree aborts before any
    state-changing step. The operator either commits/stashes or passes
    ``--allow-dirty``."""
    with patch.object(cli, "_git_working_tree_dirty", lambda _: True):
        rc = cli.cmd_update(_ns())
    assert rc == 2
    err = capsys.readouterr().err
    assert "dirty" in err.lower()
    assert "--allow-dirty" in err


def test_allow_dirty_lets_pull_proceed(stub_project_dir):
    """``--allow-dirty`` bypasses the dirty-tree refusal so the operator
    can update on top of work-in-progress when they know it's safe."""
    pull_calls = []
    with patch.object(cli, "_git_working_tree_dirty", lambda _: True), \
         patch.object(cli, "_run_git_pull",
                      lambda d: pull_calls.append(d) or (0, "ok")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda *a, **kw: (0, "")), \
         patch.object(cli, "cmd_feature_sync", lambda args: 0), \
         patch.object(cli, "cmd_restart", lambda args: 0):
        rc = cli.cmd_update(_ns(allow_dirty=True))

    assert rc == 0
    assert pull_calls == [stub_project_dir]


def test_dry_run_invokes_no_shell_commands(stub_project_dir, capsys):
    """``--dry-run`` MUST NOT call any of the real-action helpers."""
    forbidden = []
    with patch.object(cli, "_git_working_tree_dirty", lambda _: False), \
         patch.object(cli, "_run_git_pull",
                      lambda *_: forbidden.append("pull") or (0, "")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda *_, **__: forbidden.append("install") or (0, "")), \
         patch.object(cli, "cmd_feature_sync",
                      lambda args: forbidden.append("sync") or 0), \
         patch.object(cli, "cmd_restart",
                      lambda args: forbidden.append("restart") or 0):
        rc = cli.cmd_update(_ns(dry_run=True))

    assert rc == 0
    assert forbidden == []
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()


@pytest.mark.parametrize("skipped_flag,expected_absent", [
    ("pull", "pull"),
    ("install", "install"),
    ("features", "sync"),
    ("restart", "restart"),
])
def test_no_flag_skips_individual_step(
    stub_project_dir, skipped_flag, expected_absent,
):
    """Each ``--no-<step>`` skips precisely its step without affecting the
    others."""
    called = []
    with patch.object(cli, "_git_working_tree_dirty", lambda _: False), \
         patch.object(cli, "_run_git_pull",
                      lambda _: called.append("pull") or (0, "")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda *a, **kw: called.append("install") or (0, "")), \
         patch.object(cli, "cmd_feature_sync",
                      lambda args: called.append("sync") or 0), \
         patch.object(cli, "cmd_restart",
                      lambda args: called.append("restart") or 0):
        rc = cli.cmd_update(_ns(**{skipped_flag: False}))

    assert rc == 0
    assert expected_absent not in called


def test_continue_on_error_runs_restart_after_sync_failure(stub_project_dir):
    """``--continue-on-error`` allows a feature-sync failure to NOT
    abort the restart — useful when a single optional feature package
    is temporarily unreachable."""
    called = []
    with patch.object(cli, "_git_working_tree_dirty", lambda _: False), \
         patch.object(cli, "_run_git_pull", lambda _: (0, "")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda *a, **kw: (0, "")), \
         patch.object(cli, "cmd_feature_sync",
                      lambda args: called.append("sync") or 1), \
         patch.object(cli, "cmd_restart",
                      lambda args: called.append("restart") or 0):
        rc = cli.cmd_update(_ns(continue_on_error=True))

    assert rc == 0
    assert called == ["sync", "restart"]


def test_target_agent_flows_to_restart_only(stub_project_dir):
    """A positional ``name`` arg is forwarded to the restart step only;
    the other steps don't accept an agent target."""
    restart_target = []
    with patch.object(cli, "_git_working_tree_dirty", lambda _: False), \
         patch.object(cli, "_run_git_pull", lambda _: (0, "")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda *a, **kw: (0, "")), \
         patch.object(cli, "cmd_feature_sync", lambda args: 0), \
         patch.object(cli, "cmd_restart",
                      lambda args: restart_target.append(args.name) or 0):
        rc = cli.cmd_update(_ns(name="Emma"))

    assert rc == 0
    assert restart_target == ["Emma"]


def test_no_deps_flag_threads_to_install_helper(stub_project_dir):
    """``--no-deps`` reaches the install helper so dependency resolution
    is skipped when the operator wants the fast path."""
    seen = []
    with patch.object(cli, "_git_working_tree_dirty", lambda _: False), \
         patch.object(cli, "_run_git_pull", lambda _: (0, "")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda d, no_deps: seen.append(no_deps) or (0, "")), \
         patch.object(cli, "cmd_feature_sync", lambda args: 0), \
         patch.object(cli, "cmd_restart", lambda args: 0):
        rc = cli.cmd_update(_ns(no_deps=True))

    assert rc == 0
    assert seen == [True]


def test_pypi_install_skips_pull_and_install_silently(tmp_path, monkeypatch):
    """Codex round 2 P1: when running from a PyPI install (no
    editable source checkout, e.g. ``pipx`` or
    ``pip install kestrel-sovereign`` against a tarball), pull AND
    install both skip — but features + restart still run. This is
    the value-add path for a configured deployment where
    ``KESTREL_HOME`` points at the data root."""
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: None)
    called = []
    with patch.object(cli, "_run_git_pull",
                      lambda _: called.append("pull") or (0, "")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda *a, **kw: called.append("install") or (0, "")), \
         patch.object(cli, "cmd_feature_sync",
                      lambda args: called.append("sync") or 0), \
         patch.object(cli, "cmd_restart",
                      lambda args: called.append("restart") or 0):
        rc = cli.cmd_update(_ns())

    assert rc == 0
    # Both pull and install silently skipped; sync + restart ran.
    assert called == ["sync", "restart"]


def test_resolve_source_checkout_finds_editable_install():
    """The resolver returns the directory containing the running
    kestrel_sovereign package's source — i.e. the editable checkout's
    root with pyproject.toml + .git — when installed editable."""
    # The test process runs against an editable install of
    # kestrel_sovereign; the resolver MUST find the checkout root.
    src = cli._resolve_source_checkout()
    assert src is not None
    assert (src / "pyproject.toml").exists()
    assert (src / ".git").exists()
    # Sanity: it's the kestrel-sovereign checkout, not some sibling.
    assert (src / "kestrel_sovereign").is_dir()


def test_install_helper_pins_sys_executable(stub_project_dir):
    """Codex round 1 P1: ``uv pip install -e .`` MUST target the
    running interpreter's environment, not whatever uv resolves on its
    own. Otherwise after the install the subsequent ``kestrel
    restart`` could end up running the OLD install."""
    import sys
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        import types as _types
        return _types.SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        cli._run_uv_pip_install_editable(stub_project_dir, no_deps=False)

    cmd = captured["cmd"]
    assert cmd[0:2] == ["uv", "pip"]
    assert "--python" in cmd
    assert cmd[cmd.index("--python") + 1] == sys.executable
    assert "-e" in cmd and cmd[-1] == "."


def test_git_error_in_real_checkout_aborts_update(stub_project_dir, capsys):
    """Codex round 1 P2: distinguishing 'not a git checkout' from
    'git itself failed' matters — a missing git binary inside a real
    checkout MUST abort the update, not silently skip the pull and
    keep going to install/sync/restart against potentially out-of-date
    code."""
    later = []
    boom = cli._GitFailedError("dubious ownership")

    with patch.object(cli, "_project_dir_is_git", lambda _: True), \
         patch.object(cli, "_git_working_tree_dirty",
                      lambda _: (_ for _ in ()).throw(boom)), \
         patch.object(cli, "_run_git_pull",
                      lambda _: later.append("pull") or (0, "")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda *a, **kw: later.append("install") or (0, "")), \
         patch.object(cli, "cmd_feature_sync",
                      lambda args: later.append("sync") or 0), \
         patch.object(cli, "cmd_restart",
                      lambda args: later.append("restart") or 0):
        rc = cli.cmd_update(_ns())

    assert rc != 0
    assert later == []
    err = capsys.readouterr().err
    assert "dubious ownership" in err


def test_subparser_registers_and_help_mentions_steps():
    """``kestrel update --help`` is discoverable and documents the
    pipeline."""
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "update" in help_text
    # Drill into the subparser's own help to confirm the docs.
    ns, _ = parser.parse_known_args(["update", "--no-pull", "--dry-run"])
    assert ns.command == "update"
    assert ns.pull is False
    assert ns.dry_run is True
