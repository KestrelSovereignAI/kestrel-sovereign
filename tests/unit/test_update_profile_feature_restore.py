"""F234: update_then_restart must restore out-of-tree feature packages that a
bare `uv sync` prunes — mirroring what `kestrel update` does — instead of
restarting into a host with its isolated/entry-point features missing."""

from kestrel_sovereign.features.restart_coordinator.update_profiles import (
    UPDATE_PROFILES,
)

PROFILE = UPDATE_PROFILES["sovereign_local_uv_sync"]


def _step_names(steps):
    return [s.name for s in steps]


def test_feature_sync_step_added_when_manifest_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
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


def test_no_feature_sync_step_when_manifest_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .kestrel-host-features.toml here
    steps = PROFILE.build_steps(
        repo_path="/repo", target_ref="main", allow_migrations=False
    )
    names = _step_names(steps)
    assert "feature_sync" not in names
    # Unchanged shape for a host with no out-of-tree features.
    assert names == [
        "fetch", "checkout", "reattach_branch", "install", "resolve_ref",
    ]


def test_reattach_branch_step_shape(tmp_path, monkeypatch):
    """The reattach step lands the local branch on the fetched commit.

    It is a coordinator-native routine (a single argv command cannot
    express the tag/branch-collision guard) and allow_failure, so tag/sha
    targets stay detached without aborting the update.
    """
    monkeypatch.chdir(tmp_path)  # no manifest
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
