"""F234: update_then_restart must restore out-of-tree feature packages that a
bare `uv sync` prunes — mirroring what `kestrel update` does — instead of
restarting into a host with its isolated/entry-point features missing."""

import pytest

from kestrel_sovereign import paths
from kestrel_sovereign.features.restart_coordinator.update_profiles import (
    UPDATE_PROFILES,
)

PROFILE = UPDATE_PROFILES["sovereign_local_uv_sync"]


@pytest.fixture(autouse=True)
def _project_home(monkeypatch, tmp_path):
    """``KESTREL_HOME`` is this test's own directory, and the cwd a sibling.

    The profile reads the project directory's manifest, the one the host reads
    its feature enablement from, and never the host process cwd (#3502).
    """
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))
    launch = tmp_path.parent / f"{tmp_path.name}-launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    return launch


def _step_names(steps):
    return [s.name for s in steps]


def test_feature_sync_step_added_when_manifest_present(tmp_path):
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')

    steps = PROFILE.build_steps(
        repo_path="/repo", target_ref="main", allow_migrations=False
    )
    names = _step_names(steps)
    assert "feature_sync" in names
    # Ordering: restore runs AFTER install, and resolve_ref stays last.
    assert names.index("install") < names.index("feature_sync")
    assert names[-1] == "resolve_ref"

    fs = next(s for s in steps if s.name == "feature_sync")
    # Invoked via the running interpreter, not a bare `kestrel` PATH lookup.
    import sys
    assert fs.argv[:5] == [
        sys.executable, "-m", "kestrel_sovereign.cli", "feature", "sync",
    ]
    # Absolute manifest passed so discovery doesn't depend on the step cwd.
    assert "--manifest" in fs.argv
    assert str(manifest.resolve()) in fs.argv


def test_a_linked_manifest_is_passed_as_the_project_directorys_path(tmp_path):
    """Not the link's target: a relative ``editable`` is relative to the
    manifest's directory, so the step must name the manifest where every other
    reader finds it, in the project directory."""
    target = tmp_path / "elsewhere" / "features.toml"
    target.parent.mkdir()
    target.write_text('[[feature]]\nname = "voice"\neditable = "../voice"\n')
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.symlink_to(target)

    steps = PROFILE.build_steps(
        repo_path="/repo", target_ref="main", allow_migrations=False
    )

    fs = next(s for s in steps if s.name == "feature_sync")
    assert fs.argv[fs.argv.index("--manifest") + 1] == str(manifest)


def test_no_feature_sync_step_when_manifest_absent(_project_home):
    # No manifest in the project directory, and one in the cwd that is not it.
    (_project_home / ".kestrel-host-features.toml").write_text(
        '[[feature]]\nname = "voice"\n'
    )
    steps = PROFILE.build_steps(
        repo_path="/repo", target_ref="main", allow_migrations=False
    )
    names = _step_names(steps)
    assert "feature_sync" not in names
    # Unchanged shape for a host with no out-of-tree features.
    assert names == [
        "fetch", "checkout", "reattach_branch", "install", "resolve_ref",
    ]


def test_reattach_branch_step_shape():
    """The reattach step lands the local branch on the fetched commit.

    It is a coordinator-native routine (a single argv command cannot
    express the tag/branch-collision guard) and allow_failure, so tag/sha
    targets stay detached without aborting the update.
    """
    steps = PROFILE.build_steps(
        repo_path="/repo", target_ref="main", allow_migrations=False
    )
    names = _step_names(steps)
    # Reattach runs after the detach checkout, before install.
    assert names.index("checkout") < names.index("reattach_branch")
    assert names.index("reattach_branch") < names.index("install")

    reattach = next(s for s in steps if s.name == "reattach_branch")
    # argv documents the mutating command the native routine may run.
    assert reattach.argv == [
        "git", "-C", "/repo", "checkout", "-B", "main", "FETCH_HEAD",
    ]
    assert reattach.native == "reattach_branch"
    assert reattach.native_args == ("/repo", "main")
    assert reattach.allow_failure is True
    assert reattach.read_only is False
    # The mutating steps stay fatal-on-failure and non-native.
    for name in ("fetch", "checkout", "install"):
        step = next(s for s in steps if s.name == name)
        assert step.allow_failure is False
        assert step.native is None
