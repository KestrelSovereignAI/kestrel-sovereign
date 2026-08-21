"""Tests for the ``kestrel update`` reconcile step (#1788).

Covers the execution + wiring half (the planning half is in
``test_feature_reconcile.py``): a missing allowlisted feature gets installed, an
unresolvable class aborts with a clear error, ``--dry-run`` mutates nothing, and
editable vs PyPI update modes dispatch to ``git pull`` vs ``pip --upgrade``.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kestrel_sovereign import cli, cli_lifecycle
from kestrel_sovereign import feature_reconcile as fr
from kestrel_sovereign.cli_features import CORE_UNSAFE
from kestrel_sovereign.feature_registry import FeaturePackageInfo
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from tests.utils.fake_uv import CORE, FakeUv, use_fake_uv


def _registry():
    return {
        "voice": FeaturePackageInfo(
            name="voice", package="kestrel-feature-voice",
            git="https://example/voice.git",
            features=["VoiceFeature"], description="", core=False,
        ),
    }


def _ok(stdout="", stderr="", rc=0):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def _unguarded():
    """A core guard with nothing to protect.

    ``_execute_reconcile_action`` requires a guard — a feature install without
    one is the #2949 defect. These tests exercise the dispatch (git pull vs pip
    --upgrade vs git fallback), not the guard, so they pass the explicit
    no-policy guard rather than inheriting a silent default.
    """
    from kestrel_sovereign.cli_features import CoreInstallGuard

    return CoreInstallGuard.unguarded()


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Patch the shared seams so reconcile runs against a fake host."""
    ma = MultiAgentConfig(agents={
        "emma": LocalAgentConfig(
            data_dir="agent_data/emma", port=8801, features=["VoiceFeature"],
        ),
    })
    monkeypatch.setattr(cli.MultiAgentConfig, "load", classmethod(lambda c, p, **k: ma))
    monkeypatch.setattr(
        "kestrel_sovereign.feature_registry.load_registry", lambda *a, **k: _registry(),
    )
    # No source map on disk by default.
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: tmp_path / "missing.toml")
    return tmp_path


# --------------------------------------------------------------------------
# _run_feature_reconcile
# --------------------------------------------------------------------------

def test_reconcile_installs_missing_allowlisted_feature(patched, capsys):
    """An allowlisted class missing from the venv is installed from its
    registry PyPI source."""
    install_calls = []

    def fake_install(pip_args, *, constraints=None, reinstall=None, timeout=None):
        install_calls.append(pip_args)
        return _ok(stdout="Successfully installed kestrel-feature-voice-0.3.0")

    with patch("importlib.metadata.version", side_effect=__import__("importlib.metadata", fromlist=["PackageNotFoundError"]).PackageNotFoundError), \
         patch.object(cli, "_editable_install_path", lambda p: None), \
         patch.object(cli, "_extension_install_run", fake_install):
        rc = cli_lifecycle._run_feature_reconcile(
            patched, manifest_override=None, dry_run=False,
            allow_dirty=False, continue_on_error=False, prefer=None,
        )

    assert rc == 0
    assert install_calls == [["kestrel-feature-voice"]]
    out = capsys.readouterr().out
    assert "kestrel-feature-voice" in out


def test_reconcile_errors_on_unresolvable_class(monkeypatch, tmp_path, capsys):
    """A class no registry package provides is a hard error (no blind
    fallback) — reconcile returns non-zero and names the class."""
    ma = MultiAgentConfig(agents={
        "x": LocalAgentConfig(data_dir="d/x", port=8801, features=["GhostFeature"]),
    })
    monkeypatch.setattr(cli.MultiAgentConfig, "load", classmethod(lambda c, p, **k: ma))
    monkeypatch.setattr(
        "kestrel_sovereign.feature_registry.load_registry", lambda *a, **k: _registry(),
    )
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: tmp_path / "missing.toml")

    rc = cli_lifecycle._run_feature_reconcile(
        tmp_path, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=None,
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "GhostFeature" in err


def test_reconcile_dry_run_mutates_nothing(patched):
    """--dry-run previews the install without running pip/git."""
    pnf = __import__("importlib.metadata", fromlist=["PackageNotFoundError"]).PackageNotFoundError
    with patch("importlib.metadata.version", side_effect=pnf), \
         patch.object(cli, "_editable_install_path", lambda p: None), \
         patch.object(cli, "_extension_install_run") as install, \
         patch.object(cli_lifecycle, "_editable_git_pull") as pull:
        rc = cli_lifecycle._run_feature_reconcile(
            patched, manifest_override=None, dry_run=True,
            allow_dirty=False, continue_on_error=False, prefer=None,
        )
    assert rc == 0
    install.assert_not_called()
    pull.assert_not_called()


# --------------------------------------------------------------------------
# _execute_reconcile_action
# --------------------------------------------------------------------------

def test_execute_editable_update_runs_git_pull_only():
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="update", mode="editable",
        source="/co/voice", required_by=["VoiceFeature"],
    )
    with patch.object(cli_lifecycle, "_editable_git_pull", return_value=(0, "Updated.")) as pull, \
         patch.object(cli, "_extension_install_run") as install:
        ok, detail = cli_lifecycle._execute_reconcile_action(action, {}, allow_dirty=False, guard=_unguarded())
    assert ok is True
    pull.assert_called_once()
    install.assert_not_called()  # present editable: pull is enough, no reinstall


def test_execute_editable_install_pulls_then_links():
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="install", mode="editable",
        source="/co/voice", relink=True,
    )
    with patch.object(cli_lifecycle, "_editable_git_pull", return_value=(0, "")) as pull, \
         patch.object(cli, "_extension_install_run", return_value=_ok()) as install:
        ok, _ = cli_lifecycle._execute_reconcile_action(action, {}, allow_dirty=False, guard=_unguarded())
    assert ok is True
    pull.assert_called_once()
    install.assert_called_once()
    assert install.call_args[0][0][0] == "-e"


def test_execute_editable_relink_switches_from_pypi_install():
    """A package installed from PyPI but now declared editable must be
    re-linked via pip install -e, not just git-pulled (codex round 3 P2)."""
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="update", mode="editable",
        source="/co/voice", current_version="0.3.0", relink=True,
    )
    with patch.object(cli_lifecycle, "_editable_git_pull", return_value=(0, "")) as pull, \
         patch.object(cli, "_extension_install_run", return_value=_ok()) as install:
        ok, _ = cli_lifecycle._execute_reconcile_action(action, {}, allow_dirty=False, guard=_unguarded())
    assert ok is True
    pull.assert_called_once()
    install.assert_called_once()  # relink despite op == "update"


def test_execute_editable_pull_failure_reports_and_skips_link():
    """A non-fast-forward / dirty collision from the pull is reported and the
    pip link is NOT attempted."""
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="install", mode="editable",
        source="/co/voice", relink=True,
    )
    with patch.object(cli_lifecycle, "_editable_git_pull", return_value=(2, "REFUSED — dirty")) as pull, \
         patch.object(cli, "_extension_install_run") as install:
        ok, detail = cli_lifecycle._execute_reconcile_action(action, {}, allow_dirty=False, guard=_unguarded())
    assert ok is False
    assert "REFUSED" in detail
    install.assert_not_called()


def test_execute_pypi_update_runs_upgrade():
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="update", mode="pypi",
        source="kestrel-feature-voice>=0.3,<0.4",
    )
    with patch.object(cli, "_extension_install_run", return_value=_ok()) as install:
        ok, _ = cli_lifecycle._execute_reconcile_action(action, {}, allow_dirty=False, guard=_unguarded())
    assert ok is True
    assert install.call_args[0][0] == ["--upgrade", "kestrel-feature-voice>=0.3,<0.4"]


def test_execute_pinned_pypi_does_not_fall_back_to_unpinned_git():
    """A pinned entry that fails pip must NOT install the unpinned git HEAD —
    that would violate the operator's declared pin (codex round 7 P2)."""
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="update", mode="pypi",
        source="kestrel-feature-voice>=0.3,<0.4", source_declared=True,
    )
    with patch.object(cli, "_extension_install_run", return_value=_ok(rc=1, stderr="no match")) as install:
        ok, _ = cli_lifecycle._execute_reconcile_action(
            action, {"kestrel-feature-voice": "https://example/voice.git"},
            allow_dirty=False, guard=_unguarded(),
        )
    assert ok is False
    assert install.call_count == 1  # NO git fallback for a pinned entry


def test_plan_marks_an_unpinned_pypi_declaration_as_a_declared_source():
    """``pypi = ""`` pins no version but still NAMES the index as the source.

    The plan has to carry that, because the executor's only other option is the
    registry's git URL — repo HEAD, a different source entirely. Truthiness on
    the spec cannot tell ``""`` (declared, any version) from absent (legacy, no
    declaration), and reads both as "substitution allowed".
    """
    info = _registry()["voice"]
    actions, _ = fr.plan_reconcile(
        {"kestrel-feature-voice": info},
        {"kestrel-feature-voice": fr.SourceEntry(
            package="kestrel-feature-voice", pypi="",
        )},
        {"kestrel-feature-voice": None},
        {},
        {"VoiceFeature": "kestrel-feature-voice"},
    )
    assert actions[0].source == "kestrel-feature-voice"  # no version pin
    assert actions[0].source_declared is True

    # The legacy entry — no source at all — is the one that may fall back.
    legacy, _ = fr.plan_reconcile(
        {"kestrel-feature-voice": info},
        {"kestrel-feature-voice": fr.SourceEntry(package="kestrel-feature-voice")},
        {"kestrel-feature-voice": None},
        {},
        {"VoiceFeature": "kestrel-feature-voice"},
    )
    assert legacy[0].source_declared is False


def test_execute_unpinned_pypi_declaration_does_not_fall_back_to_git():
    """The executor half: a failed index install for a ``pypi = ""`` entry is a
    failure, not a licence to install repo HEAD from somewhere else."""
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="install", mode="pypi",
        source="kestrel-feature-voice", source_declared=True,
    )
    with patch.object(cli, "_extension_install_run", return_value=_ok(rc=1, stderr="no match")) as install:
        ok, _ = cli_lifecycle._execute_reconcile_action(
            action, {"kestrel-feature-voice": "https://example/voice.git"},
            allow_dirty=False, guard=_unguarded(),
        )
    assert ok is False
    assert install.call_count == 1


def test_execute_pypi_force_reinstall_switches_off_editable():
    """A force-reinstall pypi action reinstalls so the wheel replaces an
    editable link (codex round 9 P2) — scoped to the package being switched.

    A bare ``--force-reinstall`` in the argv would apply to the whole resolve,
    and core is a resolved dependency of every feature package (#2949).
    """
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="update", mode="pypi",
        source="kestrel-feature-voice>=0.3,<0.4", force_reinstall=True,
    )
    with patch.object(cli, "_extension_install_run", return_value=_ok()) as install:
        ok, _ = cli_lifecycle._execute_reconcile_action(action, {}, allow_dirty=False, guard=_unguarded())
    assert ok is True
    args = install.call_args[0][0]
    assert "--upgrade" in args
    assert "--force-reinstall" not in args
    assert install.call_args[1]["reinstall"] == "kestrel-feature-voice"


def test_execute_pypi_falls_back_to_git_url():
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="install", mode="pypi",
        source="kestrel-feature-voice",
    )
    results = [_ok(rc=1, stderr="not found"), _ok(rc=0)]
    with patch.object(cli, "_extension_install_run", side_effect=results) as install:
        ok, _ = cli_lifecycle._execute_reconcile_action(
            action, {"kestrel-feature-voice": "https://example/voice.git"},
            allow_dirty=False, guard=_unguarded(),
        )
    assert ok is True
    assert install.call_count == 2
    assert install.call_args[0][0] == ["git+https://example/voice.git"]


def _reconcile(patched, **kw):
    """Run the reconcile step against the fake host."""
    return cli_lifecycle._run_feature_reconcile(
        patched, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=None, **kw
    )


# --------------------------------------------------------------------------
# distribution identity — live metadata, registry, and source map name the
# same package three ways (#2949)
# --------------------------------------------------------------------------

def test_reconcile_honours_the_source_map_when_live_metadata_uses_underscores(
    monkeypatch, patched, tmp_path,
):
    """``ep.dist.name`` is whatever the installed package's METADATA spells.

    A project built as ``kestrel_feature_voice`` reports the underscore form,
    while the registry and the operator's source map spell it with hyphens —
    PEP 503 says those are one distribution. Compared raw, the plan's package
    misses its own source entry, the declared pin evaporates, and the package is
    reported ``present (no managed source)``: reconcile silently stops managing
    a package the operator explicitly declared.
    """
    manifest = tmp_path / "hosts.toml"
    manifest.write_text(
        '[[feature]]\nname = "kestrel-feature-voice"\npypi = ">=0.3,<0.4"\n'
    )
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: manifest)
    monkeypatch.setattr(
        "kestrel_sovereign.features.discover_entrypoint_feature_dists",
        lambda: {"VoiceFeature": "kestrel_feature_voice"},
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.discover_local_feature_class_names", lambda: set(),
    )

    with patch("importlib.metadata.version", lambda pkg: "0.3.0"), \
         patch.object(cli, "_editable_install_path", lambda p: None), \
         patch.object(cli, "_extension_install_run", return_value=_ok()) as install:
        rc = _reconcile(patched)

    assert rc == 0
    # The declared pin was applied — not dropped as "no managed source".
    assert install.called, "the source map's entry was not matched to the package"
    assert install.call_args[0][0] == ["--upgrade", "kestrel-feature-voice>=0.3,<0.4"]


def test_reconcile_git_fallback_finds_a_catalog_row_spelled_differently(
    monkeypatch, patched,
):
    """The executor's registry lookup is keyed like the plan's package identity.

    An action carries the canonical distribution name, so a catalog row spelling
    its own ``package`` another way must still resolve to a git URL — otherwise
    a package that HAS a remote source is reported as unrecoverable.
    """
    monkeypatch.setattr(
        "kestrel_sovereign.feature_registry.load_registry",
        lambda *a, **k: {"voice": FeaturePackageInfo(
            name="voice", package="Kestrel_Feature_Voice",
            git="https://example/voice.git",
            features=["VoiceFeature"], description="", core=False,
        )},
    )
    pnf = __import__("importlib.metadata", fromlist=["PackageNotFoundError"]).PackageNotFoundError
    results = [_ok(rc=1, stderr="not found"), _ok(rc=0)]
    with patch("importlib.metadata.version", side_effect=pnf), \
         patch.object(cli, "_editable_install_path", lambda p: None), \
         patch.object(cli, "_extension_install_run", side_effect=results) as install:
        rc = _reconcile(patched)

    assert rc == 0
    assert install.call_count == 2
    assert install.call_args[0][0] == ["git+https://example/voice.git"]


def test_resolve_packages_canonicalizes_live_distribution_names():
    """The planning half: whichever spelling live metadata reports, the package
    identity handed to the plan is the one the source index is keyed on."""
    pkg_infos, class_to_pkg, unresolved = fr.resolve_packages(
        {"VoiceFeature"}, _registry(),
        entrypoint_dists={"VoiceFeature": "Kestrel_Feature_Voice"},
    )
    assert unresolved == []
    assert class_to_pkg == {"VoiceFeature": "kestrel-feature-voice"}
    # Canonical enough to find the catalogued row (git URL, extras) too, rather
    # than synthesizing a sourceless info from the live spelling.
    assert list(pkg_infos) == ["kestrel-feature-voice"]
    assert pkg_infos["kestrel-feature-voice"].git == "https://example/voice.git"


# --------------------------------------------------------------------------
# core install guard (#2949) — reconcile holds core to the SAME source-map
# policy the feature commands do.
# --------------------------------------------------------------------------

def test_reconcile_pins_core_to_the_editable_checkout(monkeypatch, patched, capsys):
    """The regression, through `kestrel update`: editable core at X plus a
    feature requiring core > X must fail the resolve, not swap the link.

    The registry's git-URL fallback is pinned too — an unguarded fallback would
    be a hole the size of the original bug.
    """
    venv = FakeUv(core_checkout="/src/core")
    use_fake_uv(monkeypatch, venv)

    rc = _reconcile(patched)

    assert rc == 1  # loud, not silent
    assert venv.editable[CORE] == "/src/core"  # link intact
    assert venv.installed[CORE] == "0.52.0"
    assert "kestrel-feature-voice" not in venv.installed
    assert venv.pins == ["==0.52.0", "==0.52.0"]  # pip attempt + git fallback
    assert "No solution found" in capsys.readouterr().out


def test_reconcile_pin_does_not_block_a_compatible_feature(monkeypatch, patched):
    """The pin must not manufacture failures: a feature the checkout satisfies
    installs normally and core is untouched."""
    venv = FakeUv(core_checkout="/src/core", feature_requires=">=0.52")
    use_fake_uv(monkeypatch, venv)

    rc = _reconcile(patched)

    assert rc == 0
    assert venv.installed["kestrel-feature-voice"] == "0.4.0"
    assert venv.editable[CORE] == "/src/core"
    assert venv.pins == ["==0.52.0"]


def test_reconcile_pins_core_to_the_manifests_declared_pypi_window(
    monkeypatch, patched, tmp_path, capsys,
):
    """A `pypi` core entry is a declaration, not a waiver: reconcile holds the
    batch to the declared window even though core is not editable."""
    manifest = tmp_path / "hosts.toml"
    manifest.write_text(
        '[[feature]]\nname = "kestrel-sovereign"\npypi = ">=0.52,<0.53"\n'
        '[[feature]]\nname = "voice"\npypi = ">=0.3"\n'
    )
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: manifest)
    venv = FakeUv(core_checkout=None)  # a declared wheel, not a checkout
    use_fake_uv(monkeypatch, venv)

    rc = _reconcile(patched)

    # The feature needs core >=0.53; the manifest forbids it. The window holds.
    assert rc == 1
    assert venv.pins == [">=0.52,<0.53"]  # pinned entry: no git fallback
    assert venv.installed[CORE] == "0.52.0"
    assert "kestrel-feature-voice" not in venv.installed
    assert "No solution found" in capsys.readouterr().out


def test_reconcile_fails_when_an_install_replaced_the_editable_core(
    monkeypatch, patched, capsys,
):
    """Detection half: an install that bypassed the pin leaves reconcile
    non-zero, names the change, and re-links the checkout."""
    venv = FakeUv(core_checkout="/src/core", honours_constraints=False)
    use_fake_uv(monkeypatch, venv)

    rc = _reconcile(patched)

    assert rc == 1  # the feature installed, but reconcile does NOT report success
    assert venv.installed["kestrel-feature-voice"] == "0.4.0"
    # The swap happened, was named, and the checkout was re-linked.
    assert venv.editable[CORE] == "/src/core"
    assert venv.installed[CORE] == "0.52.0"
    err = capsys.readouterr().err
    assert "was replaced during the install batch" in err
    assert (
        f"restored: uv pip install --python {shlex.quote(sys.executable)} "
        "-e /src/core"
    ) in err


def test_reconcile_reports_a_core_that_could_not_be_restored(
    monkeypatch, patched, capsys,
):
    """A failed re-link is the worst case: core is still a wheel copy nobody
    declared. Reconcile must hand the operator the exact command, not a
    reassuring 'restored'."""
    venv = FakeUv(
        core_checkout="/src/core", honours_constraints=False, repair_fails=True,
    )
    use_fake_uv(monkeypatch, venv)

    rc = _reconcile(patched)

    # CORE_UNSAFE, not the generic 1: an unrepaired core is a safety failure
    # that no caller may continue past, and `1` is the code --continue-on-error
    # is entitled to ignore.
    assert rc == CORE_UNSAFE
    assert venv.editable.get(CORE) is None  # still swapped
    err = capsys.readouterr().err
    assert (
        "RESTORE FAILED — run `uv pip install --python "
        f"{shlex.quote(sys.executable)} -e /src/core` by hand."
    ) in err


def test_reconcile_repaired_core_is_an_error_but_not_unsafe(monkeypatch, patched):
    """A drift that WAS repaired still fails the command — nothing reports
    success over a replaced core — but the venv is correct again, so it must not
    claim the un-continuable status reserved for a core still left wrong."""
    venv = FakeUv(core_checkout="/src/core", honours_constraints=False)
    use_fake_uv(monkeypatch, venv)

    rc = _reconcile(patched)

    assert rc == 1 and rc != CORE_UNSAFE
    assert venv.editable.get(CORE) == "/src/core"  # restored


def test_reconcile_source_switch_leaves_the_editable_core_linked(
    monkeypatch, patched, tmp_path,
):
    """The same regression through `kestrel update`, which is where a source
    switch is most likely to run: an editable core at X survives a feature
    moving off its checkout while the index carries that same X.

    The version pin cannot see this one — a same-version wheel satisfies
    `==0.52.0` — so what holds core is the reinstall being scoped to the
    feature. `feature_requires=">=0.52"` removes the version-skew route
    deliberately: anything that happens to core here is the cascade.
    """
    manifest = tmp_path / "hosts.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ">=0.3,<0.5"\n')
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: manifest)
    venv = FakeUv(core_checkout="/src/core", feature_requires=">=0.52")
    venv.installed["kestrel-feature-voice"] = "0.3.1"
    venv.editable["kestrel-feature-voice"] = "/src/voice"
    use_fake_uv(monkeypatch, venv)

    rc = _reconcile(patched)

    assert rc == 0
    assert venv.editable[CORE] == "/src/core"  # link intact
    assert venv.installed[CORE] == "0.52.0"
    # The switch still took effect — voice came off its checkout.
    assert venv.installed["kestrel-feature-voice"] == "0.4.0"
    assert "kestrel-feature-voice" not in venv.editable
    assert not any("--force-reinstall" in c for c in venv.commands)
    cmd = venv.commands[0]
    assert cmd[cmd.index("--reinstall-package") + 1] == "kestrel-feature-voice"
