"""Tests for `kestrel feature upgrade`.

The command discovers which packages back the loaded features from their live
entry-points (no curated requirements file), skips editable/dev installs (they
track a local checkout), and pip-upgrades the rest. These tests pin that
behaviour without touching the real network or pip.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from kestrel_sovereign import cli
from kestrel_sovereign import cli_features


def _dist(name, version, editable_path=None, entries=None):
    return {
        "dist": name,
        "version": version,
        "editable_path": editable_path,
        "entries": entries or [f"features:{name}"],
    }


@pytest.fixture
def fake_registry(monkeypatch):
    """Stub load_registry so name resolution + git fallback have data."""
    registry = {
        "github": types.SimpleNamespace(
            package="kestrel-feature-github",
            git="https://github.com/KestrelSovereignAI/kestrel-feature-github.git",
            features=["GitHubFeature"],
        ),
        "voice": types.SimpleNamespace(
            package="kestrel-feature-voice",
            git="https://github.com/KestrelSovereignAI/kestrel-feature-voice.git",
            features=["VoiceFeature"],
        ),
    }
    import kestrel_sovereign.feature_registry as fr

    monkeypatch.setattr(fr, "load_registry", lambda: registry)
    return registry


class _PipSpy:
    """Records install invocations and returns a canned success result."""

    def __init__(self, stdout="", stderr=""):
        self.calls = []
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=self.stdout, stderr=self.stderr)


def _installed_versions(monkeypatch, versions):
    """Point installed-metadata reads at *versions* — the venv AFTER an install.

    What an upgrade did is read back from the venv (`_installed_version`), not
    parsed out of the installer's stdout, so this is the double that has to
    move. Keyed canonically because that is how a distribution is identified
    (PEP 503), whichever of its spellings the caller happens to hold.
    """
    import importlib.metadata as md

    from kestrel_sovereign.feature_reconcile import canonical_package

    canonical = {canonical_package(k): v for k, v in versions.items()}

    def version(name):
        try:
            return canonical[canonical_package(name)]
        except KeyError:
            raise md.PackageNotFoundError(name) from None

    monkeypatch.setattr(md, "version", version)


def test_editable_install_path_reads_direct_url(monkeypatch):
    import kestrel_sovereign.cli as climod

    class _Dist:
        def __init__(self, payload):
            self._payload = payload

        def read_text(self, name):
            return self._payload

    import importlib.metadata as md

    editable = '{"url": "file:///Volumes/x/pkg", "dir_info": {"editable": true}}'
    monkeypatch.setattr(md, "distribution", lambda n: _Dist(editable))
    assert climod._editable_install_path("pkg") == "/Volumes/x/pkg"

    regular = '{"url": "https://pypi/...", "dir_info": {}}'
    monkeypatch.setattr(md, "distribution", lambda n: _Dist(regular))
    assert climod._editable_install_path("pkg") is None


def test_installed_distribution_discovery_includes_all_xai_voice_roles(monkeypatch):
    """Canonical speech and conversation groups collapse to one distribution."""
    import importlib.metadata as md
    import kestrel_sovereign.feature_registry as feature_registry

    dist = types.SimpleNamespace(name="kestrel-voice-xai")
    entries = [
        (
            "kestrel_feature_voice_providers",
            types.SimpleNamespace(name="XAITTSProvider", dist=dist),
        ),
        (
            "kestrel_sovereign.conversation_providers",
            types.SimpleNamespace(name="XAIRealtime", dist=dist),
        ),
    ]
    monkeypatch.setattr(
        feature_registry,
        "iter_extension_entry_points",
        lambda: iter(entries),
    )
    monkeypatch.setattr(md, "version", lambda _name: "0.1.1")
    monkeypatch.setattr(cli, "_editable_install_path", lambda _name: None)

    assert cli._installed_extension_distributions() == [
        {
            "dist": "kestrel-voice-xai",
            "version": "0.1.1",
            "editable_path": None,
            "entries": [
                "kestrel_feature_voice_providers:XAITTSProvider",
                "conversation_providers:XAIRealtime",
            ],
        },
    ]


def test_upgrade_dry_run_changes_nothing(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [
            _dist("kestrel-feature-github", "0.1.0"),
            _dist("kestrel-feature-voice", "0.1.0", editable_path="/src/voice"),
        ],
    )
    spy = _PipSpy()
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=True))

    assert rc == 0
    assert spy.calls == []  # dry-run never shells out
    out = capsys.readouterr().out
    assert "would upgrade" in out
    assert "skip (editable -> /src/voice)" in out


def test_upgrade_skips_editable_and_pip_upgrades_others(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [
            _dist("kestrel-feature-github", "0.1.0"),
            _dist("kestrel-feature-voice", "0.1.0", editable_path="/src/voice"),
        ],
    )
    _installed_versions(monkeypatch, {"kestrel-feature-github": "0.2.0"})
    spy = _PipSpy()
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 0
    # exactly one install call — the non-editable package. The command is
    # routed through the uv-aware helper, so the trailing args are the pip
    # args regardless of whether it ran via `uv pip install` or `python -m pip`.
    assert len(spy.calls) == 1
    assert spy.calls[0][-2:] == ["--upgrade", "kestrel-feature-github"]
    out = capsys.readouterr().out
    assert "upgraded -> 0.2.0" in out
    assert "1 package(s) upgraded" in out
    assert "kestrel-feature-voice" in out and "skip (editable" in out


@pytest.mark.parametrize("core_rc", sorted(cli_features.NON_CONTINUABLE_CORE))
def test_upgrade_returns_every_core_state_verbatim(
    monkeypatch, fake_registry, capsys, core_rc
):
    """A gate that hand-compares one code is a hole the next code falls through.

    Driven with EVERY non-continuable code rather than the one this gate was
    written for, so adding a code and adding a gate are caught by the same
    check. `feature sync`'s gate has the same test in test_cli_feature_sync.py.
    """
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("kestrel-feature-github", "0.1.0")],
    )
    _installed_versions(monkeypatch, {"kestrel-feature-github": "0.2.0"})
    monkeypatch.setattr(cli.subprocess, "run", _PipSpy())
    monkeypatch.setattr(
        cli_features.CoreInstallGuard, "verify", lambda self: core_rc,
    )

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == core_rc


# --- what an upgrade did is a fact about the venv (#2949) -------------------


def test_upgrade_reports_the_version_uv_actually_installed(
    monkeypatch, fake_registry, capsys
):
    """uv is the DEFAULT backend, and it does not write pip's prose.

    `uv pip install` reports an install as `+ pkg==ver` on stderr; pip writes
    `Successfully installed pkg-ver` on stdout. Reading the new version out of
    pip's line meant every upgrade on the backend this host actually uses
    reported "up to date" — and the restart notice, which only prints when
    something moved, went with it.
    """
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("kestrel-feature-github", "0.1.0")],
    )
    _installed_versions(monkeypatch, {"kestrel-feature-github": "0.2.0"})
    spy = _PipSpy(
        stderr=(
            "Resolved 14 packages in 412ms\n"
            "Installed 1 package in 31ms\n"
            " + kestrel-feature-github==0.2.0\n"
        ),
    )
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 0
    out = capsys.readouterr().out
    assert "upgraded -> 0.2.0" in out
    assert "1 package(s) upgraded" in out
    assert "Restart the host/agents" in out


def test_upgrade_reads_back_a_dotted_distribution_name(
    monkeypatch, fake_registry, capsys
):
    """`Kestrel.Feature.Github` names the same distribution as the registry's.

    PEP 503 folds runs of `.`, `-` and `_` into one separator, so the read-back
    canonicalizes. Swapping underscores for dashes — the old normalization —
    leaves the dotted spelling unmatched and calls a real upgrade "up to date".
    """
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("Kestrel.Feature.Github", "0.1.0")],
    )
    _installed_versions(monkeypatch, {"kestrel-feature-github": "0.2.0"})
    # pip-shaped output on purpose: the ONLY thing that can go wrong here is
    # the spelling the version is looked up under.
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        _PipSpy(stdout="Successfully installed kestrel-feature-github-0.2.0"),
    )

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 0
    assert "upgraded -> 0.2.0" in capsys.readouterr().out


def test_upgrade_that_moved_nothing_is_not_reported_as_an_upgrade(
    monkeypatch, fake_registry, capsys
):
    """The other direction: an install that resolved to what was already there.

    Reading the venv must not turn every successful exit into an "upgraded"
    line — the restart notice would then print on runs that changed nothing.
    """
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("kestrel-feature-github", "0.1.0")],
    )
    _installed_versions(monkeypatch, {"kestrel-feature-github": "0.1.0"})
    monkeypatch.setattr(cli.subprocess, "run", _PipSpy())

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 0
    out = capsys.readouterr().out
    assert "up to date" in out
    assert "0 package(s) upgraded" in out
    assert "Restart the host/agents" not in out


def test_upgrade_subset_by_name_only_targets_match(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [
            _dist("kestrel-feature-github", "0.1.0"),
            _dist("kestrel-feature-voice", "0.1.0"),
        ],
    )
    _installed_versions(monkeypatch, {"kestrel-feature-github": "0.2.0"})
    spy = _PipSpy()
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(
        types.SimpleNamespace(names=["github"], dry_run=False)
    )

    assert rc == 0
    assert len(spy.calls) == 1
    assert spy.calls[0][-1] == "kestrel-feature-github"


def test_upgrade_matches_a_distribution_spelled_differently(
    monkeypatch, fake_registry, capsys
):
    """One distribution, three spellings — the operator's, the registry's, and
    the one the package's own METADATA wrote.

    `Kestrel_Feature_Voice` is a legal `Name:` for the project the registry
    calls `kestrel-feature-voice`, and that is what `ep.dist.name` reports.
    Comparing raw strings declared the installed package missing, refused to
    upgrade it, and lost its git fallback with it (issue #2949) — PEP 503 says
    those are the same distribution.
    """
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("Kestrel_Feature_Voice", "0.1.0")],
    )

    def flaky_run(cmd, capture_output=True, text=True, timeout=None):
        calls.append(cmd)
        if "git+" in cmd[-1]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="Successfully installed kestrel-feature-voice-0.2.0",
                stderr="",
            )
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="No matching distribution"
        )

    calls = []
    monkeypatch.setattr(cli.subprocess, "run", flaky_run)

    rc = cli.cmd_feature_upgrade(
        types.SimpleNamespace(names=["voice"], dry_run=False)
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "not an installed extension package" not in out
    # ...and the registry's git URL is still reachable for the fallback.
    assert len(calls) == 2
    assert calls[1][-1] == (
        "git+https://github.com/KestrelSovereignAI/kestrel-feature-voice.git"
    )


def test_upgrade_unmatched_name_is_reported(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("kestrel-feature-github", "0.1.0")],
    )
    spy = _PipSpy()
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(
        types.SimpleNamespace(names=["does-not-exist"], dry_run=True)
    )

    assert rc == 1
    assert spy.calls == []
    assert "not an installed extension package" in capsys.readouterr().out


def test_upgrade_falls_back_to_git_on_pip_failure(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("kestrel-feature-github", "0.1.0")],
    )

    calls = []

    def flaky_run(cmd, capture_output=True, text=True, timeout=None):
        calls.append(cmd)
        # First (PyPI) attempt fails; git fallback succeeds.
        if "git+" in cmd[-1]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="Successfully installed kestrel-feature-github-0.2.0", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No matching distribution")

    monkeypatch.setattr(cli.subprocess, "run", flaky_run)

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 0
    assert len(calls) == 2
    assert calls[1][-1].startswith("git+https://github.com/")


def test_upgrade_routes_through_uv_aware_helper(monkeypatch, fake_registry, capsys):
    """upgrade goes through _extension_install_run (uv-aware), not bare python -m pip."""
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("kestrel-feature-github", "0.1.0")],
    )

    calls = []

    def fake_install(pip_args, *, constraints=None, constraint_path=None, reinstall=None, timeout=None):
        calls.append(pip_args)
        return subprocess.CompletedProcess(
            pip_args, 0, stdout="Successfully installed kestrel-feature-github-0.2.0", stderr=""
        )

    # Patch the seam the command actually reaches — `cli._extension_install_run`,
    # via CoreInstallGuard. Patching the cli_features global instead leaves the
    # command shelling out to a real `uv pip install`.
    monkeypatch.setattr(cli, "_extension_install_run", fake_install)

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 0
    assert calls == [["--upgrade", "kestrel-feature-github"]]


def test_upgrade_no_installed_extensions(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(cli, "_installed_extension_distributions", lambda: [])
    spy = _PipSpy()
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 0
    assert spy.calls == []
    assert "No installed feature/provider packages" in capsys.readouterr().out


def test_host_feature_entry_point_group_is_captured(monkeypatch):
    """Host features (e.g. fleet observability, #2444) must be enumerated so
    `feature sync --capture` records them and `feature sync` restores them after
    `uv sync` prunes the out-of-tree package."""
    from kestrel_sovereign import cli_features
    from kestrel_sovereign import feature_registry
    from kestrel_sovereign.host_features.discovery import (
        HOST_FEATURE_ENTRY_POINT_GROUP,
    )

    assert (
        HOST_FEATURE_ENTRY_POINT_GROUP
        in feature_registry.EXTENSION_ENTRY_POINT_GROUPS
    )

    fake_dist = types.SimpleNamespace(name="kestrel-feature-observability-fleet")
    fake_ep = types.SimpleNamespace(name="fleet", dist=fake_dist)
    monkeypatch.setattr(
        feature_registry,
        "iter_extension_entry_points",
        lambda: iter([(HOST_FEATURE_ENTRY_POINT_GROUP, fake_ep)]),
    )
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda name: "0.4.0")
    monkeypatch.setattr(cli, "_editable_install_path", lambda name: None)

    dists = cli_features._installed_extension_distributions()

    names = {d["dist"] for d in dists}
    assert "kestrel-feature-observability-fleet" in names
    fleet = next(d for d in dists if d["dist"] == "kestrel-feature-observability-fleet")
    assert "host_features:fleet" in fleet["entries"]


def test_upgrade_reports_core_drift_when_the_install_is_interrupted(
    monkeypatch, fake_registry, capsys,
):
    """`feature upgrade` is the third install path (#2962)."""
    from tests.utils.fake_uv import CORE, FakeUv, use_fake_uv

    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("kestrel-feature-voice", "0.1.0")],
    )
    venv = FakeUv(
        core_checkout="/src/core",
        honours_constraints=False,
        feature_install_interrupted=True,
    )
    use_fake_uv(monkeypatch, venv)

    with pytest.raises(KeyboardInterrupt):
        cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    err = capsys.readouterr().err
    assert "INTERRUPTED" in err
    assert venv.editable.get(CORE) != "/src/core"
