"""Tests for `kestrel feature sync` — the restore counterpart to `upgrade`.

`upgrade` discovers packages from live entry-points (useless once a package is
pruned and its entry-point is gone). `sync` reads a host-local manifest and
reinstalls whatever is missing. These tests pin that behaviour without touching
the network, pip, or uv.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from kestrel_sovereign import cli


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


class _InstallSpy:
    """Records _extension_install_run invocations; returns canned result."""

    def __init__(self, returncode=0, stderr=""):
        self.calls = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, pip_args):
        self.calls.append(list(pip_args))
        return subprocess.CompletedProcess(pip_args, self.returncode, stdout="", stderr=self.stderr)


def _versions(monkeypatch, mapping):
    """Make importlib.metadata.version answer from *mapping* (else NotFound)."""
    import importlib.metadata as md

    def fake_version(name):
        if name in mapping:
            return mapping[name]
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", fake_version)


def _args(manifest, capture=False, dry_run=False):
    return types.SimpleNamespace(manifest=str(manifest), capture=capture, dry_run=dry_run)


# --- pure helpers ----------------------------------------------------------


def test_pip_spec_renders_extras():
    assert cli._pip_spec("pkg", []) == "pkg"
    assert cli._pip_spec("pkg", ["local"]) == "pkg[local]"
    assert cli._pip_spec("pkg", ["a", "b"]) == "pkg[a,b]"


def test_extension_install_run_prefers_uv(monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output=True, text=True):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")

    cli._extension_install_run(["kestrel-feature-voice"])
    assert seen["cmd"][:3] == ["uv", "pip", "install"]
    assert "--python" in seen["cmd"]  # pinned to this interpreter
    assert seen["cmd"][-1] == "kestrel-feature-voice"


def test_extension_install_run_falls_back_to_pip(monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output=True, text=True):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: None)  # no uv

    cli._extension_install_run(["kestrel-feature-voice"])
    assert seen["cmd"][1:3] == ["-m", "pip"]
    assert seen["cmd"][0] == cli.sys.executable


# --- capture ---------------------------------------------------------------


def test_capture_writes_manifest_from_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [
            {"dist": "kestrel-feature-github", "editable_path": "/src/github"},
            {"dist": "kestrel-feature-voice", "editable_path": None},
        ],
    )
    manifest = tmp_path / ".kestrel-host-features.toml"
    rc = cli.cmd_feature_sync(_args(manifest, capture=True))

    assert rc == 0
    entries = cli._load_host_manifest(manifest)
    by_name = {e["name"]: e for e in entries}
    assert by_name["kestrel-feature-github"]["editable"] == "/src/github"
    assert by_name["kestrel-feature-voice"]["editable"] is None


# --- sync ------------------------------------------------------------------


def test_sync_missing_manifest_returns_guidance(tmp_path, capsys):
    rc = cli.cmd_feature_sync(_args(tmp_path / "nope.toml"))
    assert rc == 1
    out = capsys.readouterr().out
    assert "--capture" in out


def test_sync_installs_missing_package(monkeypatch, fake_registry, tmp_path, capsys):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')
    _versions(monkeypatch, {})  # nothing installed
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == [["kestrel-feature-voice"]]
    assert "installed" in capsys.readouterr().out


def test_sync_skips_present_package(monkeypatch, fake_registry, tmp_path, capsys):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.2.1"})
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == []  # already present -> no install
    assert "present" in capsys.readouterr().out


def test_sync_editable_extras_install(monkeypatch, fake_registry, tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[feature]]\nname = "voice"\neditable = "/src/voice"\nextras = ["local"]\n'
    )
    _versions(monkeypatch, {})  # missing
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == [["-e", "/src/voice[local]"]]


def test_sync_reinstalls_on_editable_path_mismatch(monkeypatch, fake_registry, tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\neditable = "/src/voice"\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.2.1"})  # installed...
    # ...but editable from a DIFFERENT path than the manifest wants
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: "/old/voice")
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == [["-e", "/src/voice"]]


def test_sync_editable_present_when_path_matches(monkeypatch, fake_registry, tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\neditable = "/src/voice"\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.2.1"})
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: "/src/voice")
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == []  # editable already points at the wanted checkout


def test_sync_ensures_extras_even_when_editable_path_matches(monkeypatch, fake_registry, tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\neditable = "/src/voice"\nextras = ["local"]\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.2.1"})
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: "/src/voice")  # path matches
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    # right checkout, but extras can't be probed -> idempotent re-install
    assert spy.calls == [["-e", "/src/voice[local]"]]


def test_sync_dry_run_changes_nothing(monkeypatch, fake_registry, tmp_path, capsys):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')
    _versions(monkeypatch, {})
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest, dry_run=True))

    assert rc == 0
    assert spy.calls == []
    assert "would install" in capsys.readouterr().out


# --- pypi source form (#1788) ----------------------------------------------


def test_manifest_parses_pypi_source_form(tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ">=0.3,<0.4"\n')
    (entry,) = cli._load_host_manifest(manifest)
    assert entry["pypi"] == ">=0.3,<0.4"
    assert entry["editable"] is None


def test_manifest_rejects_editable_and_pypi_together(tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[feature]]\nname = "voice"\neditable = "/src/voice"\npypi = ">=0.3"\n'
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli._load_host_manifest(manifest)


def test_manifest_rejects_non_string_pypi(tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = 3\n')
    with pytest.raises(ValueError, match="version spec"):
        cli._load_host_manifest(manifest)


def test_sync_installs_pinned_pypi_spec(monkeypatch, fake_registry, tmp_path):
    """A pypi version spec pins the installed package (``pkg>=0.3,<0.4``)."""
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ">=0.3,<0.4"\n')
    _versions(monkeypatch, {})  # missing
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == [["kestrel-feature-voice>=0.3,<0.4"]]


def test_sync_git_fallback_when_pip_fails(monkeypatch, fake_registry, tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')
    _versions(monkeypatch, {})

    class _FailThenGit:
        def __init__(self):
            self.calls = []

        def __call__(self, pip_args):
            self.calls.append(list(pip_args))
            rc = 1 if not str(pip_args[0]).startswith("git+") else 0
            return subprocess.CompletedProcess(pip_args, rc, stdout="", stderr="boom")

    spy = _FailThenGit()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls[0] == ["kestrel-feature-voice"]
    assert spy.calls[1][0].startswith("git+")


def test_sync_unknown_name_treated_as_raw_package(monkeypatch, fake_registry, tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "some-random-pkg"\n')
    _versions(monkeypatch, {})
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == [["some-random-pkg"]]


def test_load_host_manifest_rejects_entry_without_name(tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\neditable = "/src/x"\n')
    with pytest.raises(ValueError):
        cli._load_host_manifest(manifest)


def test_load_host_manifest_rejects_string_extras(tmp_path):
    # A bare string must not silently iterate into characters.
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\nextras = "local"\n')
    with pytest.raises(ValueError):
        cli._load_host_manifest(manifest)


def test_sync_ensures_extras_when_base_already_installed(monkeypatch, fake_registry, tmp_path, capsys):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\nextras = ["local"]\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.2.1"})  # base present, extras unknown
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    # extras can't be probed -> re-run the (idempotent) install to guarantee them
    assert spy.calls == [["kestrel-feature-voice[local]"]]
    assert "ensured" in capsys.readouterr().out


def test_sync_git_fallback_carries_extras(monkeypatch, fake_registry, tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\nextras = ["local"]\n')
    _versions(monkeypatch, {})  # missing

    class _FailThenGit:
        def __init__(self):
            self.calls = []

        def __call__(self, pip_args):
            self.calls.append(list(pip_args))
            rc = 0 if "git+" in str(pip_args[0]) else 1
            return subprocess.CompletedProcess(pip_args, rc, stdout="", stderr="boom")

    spy = _FailThenGit()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls[0] == ["kestrel-feature-voice[local]"]
    # PEP 508 form keeps extras across the git fallback
    assert spy.calls[1] == ["kestrel-feature-voice[local] @ git+"
                            "https://github.com/KestrelSovereignAI/kestrel-feature-voice.git"]


def test_toml_basic_string_escapes_backslashes_and_quotes():
    assert cli._toml_basic_string(r"C:\src\voice") == r'"C:\\src\\voice"'
    assert cli._toml_basic_string('a"b') == '"a\\"b"'


def test_capture_roundtrips_windows_style_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [{"dist": "kestrel-feature-voice", "editable_path": r"C:\src\voice"}],
    )
    manifest = tmp_path / "m.toml"
    cli.cmd_feature_sync(_args(manifest, capture=True))
    # The captured file must parse back to the exact path (no TOML escape break)
    entries = cli._load_host_manifest(manifest)
    assert entries[0]["editable"] == r"C:\src\voice"
