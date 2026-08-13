"""Tests for `kestrel feature sync` — the restore counterpart to `upgrade`.

`upgrade` discovers packages from live entry-points (useless once a package is
pruned and its entry-point is gone). `sync` reads a host-local manifest and
reinstalls whatever is missing. These tests pin that behaviour without touching
the network, pip, or uv.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import types

import pytest

from kestrel_sovereign import cli
from kestrel_sovereign import cli_features
from tests.utils.fake_uv import CHECKOUT, CORE, FakeUv, use_fake_uv


@pytest.fixture
def fake_registry(monkeypatch):
    """Stub load_registry so name resolution + git fallback have data.

    Carries a bundled row whose package IS core, like the real registry: dozens
    of bundled features share `kestrel-sovereign`, so that is how a manifest's
    core entry resolves to a package at all.
    """
    registry = {
        "identity": types.SimpleNamespace(
            package=CORE,
            git="https://github.com/KestrelSovereignAI/kestrel-sovereign.git",
            features=["IdentityFeature"],
            core=True,
        ),
        "github": types.SimpleNamespace(
            package="kestrel-feature-github",
            git="https://github.com/KestrelSovereignAI/kestrel-feature-github.git",
            features=["GitHubFeature"],
            core=False,
        ),
        "voice": types.SimpleNamespace(
            package="kestrel-feature-voice",
            git="https://github.com/KestrelSovereignAI/kestrel-feature-voice.git",
            features=["VoiceFeature"],
            core=False,
        ),
    }
    import kestrel_sovereign.feature_registry as fr

    monkeypatch.setattr(fr, "load_registry", lambda: registry)
    return registry


class _InstallSpy:
    """Records _extension_install_run invocations; returns canned result."""

    def __init__(self, returncode=0, stderr=""):
        self.calls = []
        self.constraints = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, pip_args, *, constraints=None, timeout=None):
        self.calls.append(list(pip_args))
        self.constraints.append(list(constraints or []))
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

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")

    cli._extension_install_run(["kestrel-feature-voice"], constraints=None)
    assert seen["cmd"][:3] == ["uv", "pip", "install"]
    assert "--python" in seen["cmd"]  # pinned to this interpreter
    assert seen["cmd"][-1] == "kestrel-feature-voice"


def test_extension_install_run_falls_back_to_pip(monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: None)  # no uv

    cli._extension_install_run(["kestrel-feature-voice"], constraints=None)
    assert seen["cmd"][1:3] == ["-m", "pip"]
    assert seen["cmd"][0] == cli.sys.executable


def _direct_url_record(url):
    """A `direct_url.json` payload for an editable install of *url*."""

    class _Dist:
        def read_text(self, name):
            return json.dumps({"url": url, "dir_info": {"editable": True}})

    return _Dist()


@pytest.mark.parametrize(
    "url, path",
    [
        # The plain case, unchanged.
        ("file:///src/kestrel-sovereign", "/src/kestrel-sovereign"),
        # A space in the checkout path is recorded percent-encoded.
        ("file:///src/My%20Kestrel/core", "/src/My Kestrel/core"),
        # Windows: urlsplit leaves the URL's own leading slash in front of the
        # drive letter, so the naive de-scheme yields the unusable `/C:/...`.
        ("file:///C:/src/kestrel-sovereign", r"C:\src\kestrel-sovereign"),
        ("file:///C:/src/My%20Kestrel", "C:\\src\\My Kestrel"),
        ("file:///c|/src/kestrel", r"c:\src\kestrel"),
        # RFC 8089: an empty authority and `localhost` mean the same thing.
        ("file://localhost/src/kestrel-sovereign", "/src/kestrel-sovereign"),
        # A real authority is a UNC share, not part of the path.
        ("file://build-01/share/kestrel", r"\\build-01\share\kestrel"),
    ],
)
def test_editable_path_decodes_the_url_direct_url_json_actually_stores(
    monkeypatch, url, path
):
    """`direct_url.json` records a URL, not a path — so decode it as one.

    PEP 610 stores the checkout as a `file:` URL: spaces arrive as `%20`, and a
    Windows checkout arrives behind a drive letter or a UNC authority. Chopping
    `file://` off textually leaves `/src/My%20Kestrel` or `/C:/src/...`, which
    name nothing. That is not cosmetic here: `CoreInstallGuard` captures this
    value as the checkout core must be restored FROM (issue #2949), so a mangled
    path turns a recoverable swap into an unrecoverable one.
    """
    import importlib.metadata as md

    monkeypatch.setattr(md, "distribution", lambda n: _direct_url_record(url))
    assert cli._editable_install_path("kestrel-sovereign") == path


def test_render_command_quotes_for_the_shell_that_will_run_it(monkeypatch):
    """`shlex.join` speaks POSIX sh, and Windows needs its own quoting.

    A Windows interpreter path routinely contains a space (`C:\\Program
    Files\\...`), and POSIX single quotes are not quoting to any Windows shell —
    they become part of the program name.
    """
    argv = [r"C:\Program Files\venv\python.exe", "-m", "pip", "install", "-e", CHECKOUT]

    monkeypatch.setattr(cli_features, "_is_windows", lambda: True)
    assert cli._render_command(argv) == (
        r"& 'C:\Program Files\venv\python.exe' -m pip install -e " + CHECKOUT
    )
    assert cli._render_shell() == "PowerShell"

    monkeypatch.setattr(cli_features, "_is_windows", lambda: False)
    assert cli._render_command(["uv", "pip", "install", "kestrel-sovereign>=0.52,<0.53"]) == (
        "uv pip install 'kestrel-sovereign>=0.52,<0.53'"
    )
    assert cli._render_shell() is None


def test_render_command_does_not_leave_a_pypi_range_as_redirection(monkeypatch):
    """A version window is `>=0.52,<0.53` — three shell metacharacters.

    `subprocess.list2cmdline` quotes for the MS C runtime the CHILD parses, not
    for the shell that reads the line first, so it hands the spec back bare. A
    Windows shell then reads `>` as redirection: the pasted "recovery" writes a
    file called `=0.52,` and installs an unpinned core off the index — the very
    swap this guard exists to prevent (#2949), re-entered through its own fix.
    """
    monkeypatch.setattr(cli_features, "_is_windows", lambda: True)

    rendered = cli._render_command(
        ["uv", "pip", "install", "--force-reinstall", "kestrel-sovereign>=0.52,<0.53"]
    )

    assert rendered == (
        "& uv pip install --force-reinstall 'kestrel-sovereign>=0.52,<0.53'"
    )
    # The spec is one argument, not a redirection: every metacharacter is inside
    # the quotes that PowerShell strips before the program ever sees it.
    assert ">" not in rendered.split("'")[0]
    assert "<" not in rendered.split("'")[0]


def test_render_command_escapes_a_literal_quote(monkeypatch):
    """A checkout under `C:\\src\\jason's kestrel` must not end the quoting.

    PowerShell's only escape inside a literal string is a doubled quote; getting
    this wrong splits one path into two arguments and installs from neither.
    """
    monkeypatch.setattr(cli_features, "_is_windows", lambda: True)

    assert cli._render_command(["uv", "pip", "install", "-e", r"C:\src\jason's kestrel"]) == (
        r"& uv pip install -e 'C:\src\jason''s kestrel'"
    )


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
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: None)
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == [["kestrel-feature-voice>=0.3,<0.4"]]


def test_sync_pypi_spec_with_extras_orders_extras_first(monkeypatch, fake_registry, tmp_path):
    """pip requires ``pkg[extra]>=x``; never ``pkg>=x[extra]``."""
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[feature]]\nname = "voice"\npypi = ">=0.3,<0.4"\nextras = ["local"]\n'
    )
    _versions(monkeypatch, {})  # missing
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: None)
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == [["kestrel-feature-voice[local]>=0.3,<0.4"]]


def test_sync_repins_installed_version_violating_pypi_spec(monkeypatch, fake_registry, tmp_path, capsys):
    """An installed-but-out-of-range package must be re-pinned, not reported
    `present` (codex round 2 P2)."""
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ">=0.3,<0.4"\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.2.1"})  # below the pin
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: None)  # non-editable
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == [["kestrel-feature-voice>=0.3,<0.4"]]  # plain reinstall, no force
    assert "reinstalled" in capsys.readouterr().out


def test_sync_present_when_installed_satisfies_pypi_spec(monkeypatch, fake_registry, tmp_path, capsys):
    """An installed version already within the pin needs no action."""
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ">=0.3,<0.4"\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.3.2"})  # in range
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: None)  # non-editable
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == []
    assert "present" in capsys.readouterr().out


def test_sync_switches_editable_install_to_pypi(monkeypatch, fake_registry, tmp_path, capsys):
    """Manifest changed editable->pypi: an installed editable package must be
    reinstalled from PyPI, not reported `present` (codex round 8 P2)."""
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ">=0.3,<0.4"\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.3.1"})  # in-range version...
    # ...but currently installed EDITABLE from a checkout
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: "/src/voice")
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    # force-reinstall so the wheel replaces the editable link (round 10 P2)
    assert spy.calls == [["--force-reinstall", "kestrel-feature-voice>=0.3,<0.4"]]
    assert "reinstalled" in capsys.readouterr().out


def test_sync_pinned_pypi_no_git_fallback(monkeypatch, fake_registry, tmp_path, capsys):
    """A pinned pypi entry that fails pip must not silently install unpinned
    git HEAD (codex round 7 P2)."""
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ">=0.3,<0.4"\n')
    _versions(monkeypatch, {})  # missing
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: None)
    spy = _InstallSpy(returncode=1, stderr="no matching distribution")
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 1
    # one constrained attempt, NO git fallback
    assert spy.calls == [["kestrel-feature-voice>=0.3,<0.4"]]
    assert "FAILED" in capsys.readouterr().out


def test_sync_unpinned_pypi_declaration_no_git_fallback(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """``pypi = ""`` is a SOURCE declaration that happens to pin no version.

    The manifest documents it as "from PyPI, any version". The git URL is a
    different source — repo HEAD, unpinned — so substituting it when the index
    install fails installs something the operator never declared and then
    reports success. Only the legacy entry (no ``pypi`` key at all) may fall
    back, and truthiness cannot tell the two apart.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ""\n')
    _versions(monkeypatch, {})  # missing
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: None)
    spy = _InstallSpy(returncode=1, stderr="no matching distribution")
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 1  # the failure is reported, not papered over
    assert spy.calls == [["kestrel-feature-voice"]]  # index only, no git+
    assert "FAILED" in capsys.readouterr().out


def test_sync_unpinned_pypi_core_declaration_never_falls_back_to_git(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """The same hole aimed at core, which is what makes it #2949 and not a nit.

    The registry carries a git URL for ``kestrel-sovereign`` itself, so a failed
    index install of a ``pypi = ""`` core entry would have installed core from
    repo HEAD — the operator's core replaced by an unpinned build from a source
    they did not declare, reported as a successful sync.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text(f'[[feature]]\nname = "{CORE}"\npypi = ""\n')
    _versions(monkeypatch, {})  # core absent → the entry is actually installed
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: None)
    spy = _InstallSpy(returncode=1, stderr="no matching distribution")
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 1
    assert not any("git+" in str(arg) for call in spy.calls for arg in call)
    assert "FAILED" in capsys.readouterr().out


def test_sync_git_fallback_when_pip_fails(monkeypatch, fake_registry, tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')
    _versions(monkeypatch, {})

    class _FailThenGit:
        def __init__(self):
            self.calls = []

        def __call__(self, pip_args, *, constraints=None, timeout=None):
            self.calls.append(list(pip_args))
            rc = 1 if not str(pip_args[0]).startswith("git+") else 0
            return subprocess.CompletedProcess(pip_args, rc, stdout="", stderr="boom")

    spy = _FailThenGit()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls[0] == ["kestrel-feature-voice"]
    assert spy.calls[1][0].startswith("git+")


def test_sync_git_fallback_finds_a_catalog_row_spelled_differently(
    monkeypatch, fake_registry, tmp_path,
):
    """The fallback's registry lookup is keyed like every other package identity.

    The entry resolves to the canonical distribution name, so a catalog row that
    spells its own ``package`` another way (PEP 503 says they are the same
    distribution) must still be found — otherwise a legacy entry that HAS a
    remote source is reported as an unrecoverable failure.
    """
    fake_registry["voice"] = types.SimpleNamespace(
        package="Kestrel_Feature_Voice",
        git="https://github.com/KestrelSovereignAI/kestrel-feature-voice.git",
        features=["VoiceFeature"],
        core=False,
    )
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')
    _versions(monkeypatch, {})

    class _FailThenGit:
        def __init__(self):
            self.calls = []

        def __call__(self, pip_args, *, constraints=None, timeout=None):
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

        def __call__(self, pip_args, *, constraints=None, timeout=None):
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
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: None)  # core: wheel
    manifest = tmp_path / "m.toml"
    cli.cmd_feature_sync(_args(manifest, capture=True))
    # The captured file must parse back to the exact path (no TOML escape break)
    entries = cli._load_host_manifest(manifest)
    assert entries[0]["editable"] == r"C:\src\voice"


# --- core install guard (#2949) --------------------------------------------
#
# `uv pip` is project-blind: it never learns the lock declares the root as
# `source = { editable = "." }`, so `kestrel-sovereign` is an ordinary
# dependency. A feature whose core requirement the installed core fails used to
# make uv resolve core from the index, replacing the operator's core with a
# wheel copy — invisibly, because cwd=checkout keeps shadowing site-packages.
#
# These tests substitute at the `subprocess.run` seam rather than at
# `_extension_install_run`, so the constraint-file plumbing itself is under
# test: delete the `-c` handling and they fail.

# The venv+resolver double lives in `tests/utils/fake_uv.py`: `feature sync`,
# `feature install`, `feature upgrade`, `kestrel update`'s reconcile and the
# HTTP install endpoint all claim to guard core identically, so they are tested
# against ONE model of what an install does to a venv.

def _voice_manifest(tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ">=0.4"\n')
    return manifest


def test_unconstrained_feature_install_swaps_editable_core(monkeypatch):
    """Fidelity check on the resolver double: WITHOUT a constraint the swap
    really happens. Without this the "core survived" assertions below could
    pass vacuously."""
    venv = FakeUv()  # editable core 0.52.0; voice needs core >=0.53
    use_fake_uv(monkeypatch, venv)

    result = cli._extension_install_run(["kestrel-feature-voice>=0.4"], constraints=None)

    assert result.returncode == 0  # resolve "succeeded"...
    assert venv.editable.get(CORE) is None  # ...by dropping the editable link
    assert venv.installed[CORE] == "0.53.0"  # for an index wheel


def test_sync_constraint_stops_a_feature_replacing_editable_core(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """The regression: editable core at X + a feature requiring core > X must
    fail the resolve, not swap the install underneath the operator."""
    venv = FakeUv()
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(_voice_manifest(tmp_path)))

    assert rc == 1  # loud, not silent
    assert venv.editable[CORE] == CHECKOUT  # link intact
    assert venv.installed[CORE] == "0.52.0"  # still the checkout's version
    assert "kestrel-feature-voice" not in venv.installed  # feature not installed
    # The pin was actually handed to the resolver, and the real skew surfaced.
    assert venv.pins == ["==0.52.0"]
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "No solution found" in out


def test_sync_constraint_does_not_block_a_compatible_feature(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """The pin must not manufacture failures: a feature the checkout satisfies
    installs normally and core is untouched."""
    venv = FakeUv(feature_requires=">=0.52")
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(_voice_manifest(tmp_path)))

    assert rc == 0
    assert venv.installed["kestrel-feature-voice"] == "0.4.0"
    assert venv.editable[CORE] == CHECKOUT
    assert venv.installed[CORE] == "0.52.0"
    assert "installed" in capsys.readouterr().out


def test_sync_relinks_core_when_an_install_bypasses_the_constraint(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """Detection half: an install that ignores the pin still can't leave sync
    reporting success over a replaced core."""
    venv = FakeUv(honours_constraints=False)
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(_voice_manifest(tmp_path)))

    assert rc == 1
    # The swap happened, was named, and the checkout was re-linked.
    assert venv.editable[CORE] == CHECKOUT
    assert venv.installed[CORE] == "0.52.0"  # the checkout's build, restored
    err = capsys.readouterr().err
    assert "was replaced during the install batch" in err
    assert (
        f"restored: uv pip install --python {shlex.quote(sys.executable)} "
        f"-e {CHECKOUT}"
    ) in err


# --- the manifest's core entry governs the whole batch ----------------------


def test_sync_applies_core_entry_before_features_whatever_the_manifest_order(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """A manifest is a declaration, not a program.

    Core moved to a new checkout (0.53.0) but listed LAST, with a feature that
    needs core >=0.53. The batch must apply the core entry first and pin the
    version core BECAME — pinning the pre-switch 0.52.0 would fail a feature the
    operator's own manifest makes installable.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[feature]]\nname = "voice"\npypi = ">=0.4"\n'
        f'[[feature]]\nname = "kestrel-sovereign"\neditable = "{CHECKOUT}-next"\n'
    )
    venv = FakeUv(checkouts={f"{CHECKOUT}-next": "0.53.0"})
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    # Core first (unpinned — never constrained against itself), then voice
    # pinned to what core became.
    assert venv.pins == [None, "==0.53.0"]
    assert venv.editable[CORE] == f"{CHECKOUT}-next"
    assert venv.installed[CORE] == "0.53.0"
    assert venv.installed["kestrel-feature-voice"] == "0.4.0"


def test_sync_guards_core_declared_under_its_underscore_spelling(
    monkeypatch, fake_registry, tmp_path
):
    """`kestrel_sovereign` and `kestrel-sovereign` are ONE distribution.

    Same manifest as the test above, written with the underscore spelling pip
    accepts. Resolving the install target under normalized-name rules while
    keying the source index on the raw string made the batch execute the entry
    as core but read core's policy from the *live* link — so the guard pinned
    the pre-switch version, failed the feature, then "restored" core to the
    checkout the manifest had just moved it off.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[feature]]\nname = "voice"\npypi = ">=0.4"\n'
        f'[[feature]]\nname = "kestrel_sovereign"\neditable = "{CHECKOUT}-next"\n'
    )
    venv = FakeUv(checkouts={f"{CHECKOUT}-next": "0.53.0"})
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert venv.pins == [None, "==0.53.0"]  # core first, then what core became
    assert venv.editable[CORE] == f"{CHECKOUT}-next"  # the DECLARED checkout
    assert venv.installed[CORE] == "0.53.0"
    assert venv.installed["kestrel-feature-voice"] == "0.4.0"


def test_sync_holds_core_inside_the_manifest_declared_pypi_window(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """A `pypi` core entry is a declaration, not a waiver.

    The operator pinned core to >=0.52,<0.53. A feature requiring >=0.53 must
    fail loudly — not quietly drag core to 0.53.0 and leave the venv violating
    the manifest it just "synced".
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[feature]]\nname = "kestrel-sovereign"\npypi = ">=0.52,<0.53"\n'
        '[[feature]]\nname = "voice"\npypi = ">=0.4"\n'
    )
    venv = FakeUv()
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 1
    assert venv.pins == [None, ">=0.52,<0.53"]  # core itself, then the window
    assert venv.installed[CORE] == "0.52.0"  # inside the declared window
    assert venv.editable.get(CORE) is None  # the declared wheel took effect
    assert "kestrel-feature-voice" not in venv.installed
    assert "No solution found" in capsys.readouterr().out


def test_sync_pins_core_to_the_declared_pypi_window_without_blocking_it(
    monkeypatch, fake_registry, tmp_path
):
    """The window is a window: core may move inside it, and a feature that needs
    the newer core installs cleanly."""
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[feature]]\nname = "kestrel-sovereign"\npypi = ">=0.53"\n'
        '[[feature]]\nname = "voice"\npypi = ">=0.4"\n'
    )
    venv = FakeUv()
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert venv.pins == [None, ">=0.53"]
    assert venv.editable.get(CORE) is None  # the declared wheel took effect
    assert venv.installed[CORE] == "0.53.0"
    assert venv.installed["kestrel-feature-voice"] == "0.4.0"


def test_sync_restores_core_pushed_outside_the_declared_pypi_window(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """Detection half for a PyPI-declared core: an install that bypassed the
    window is named and rolled back inside it."""
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[feature]]\nname = "kestrel-sovereign"\npypi = ">=0.52,<0.53"\n'
        '[[feature]]\nname = "voice"\npypi = ">=0.4"\n'
    )
    venv = FakeUv(honours_constraints=False)
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 1
    assert venv.installed[CORE] == "0.52.0"  # pulled back into the window
    err = capsys.readouterr().err
    assert "was replaced during the install batch" in err
    # The spec is shell-quoted: `>=`/`<` unquoted would make the advertised
    # command a shell redirection rather than an install.
    assert (
        f"restored: uv pip install --python {shlex.quote(sys.executable)} "
        f"'{CORE}>=0.52,<0.53'"
    ) in err


def test_sync_does_not_call_a_no_op_reinstall_a_restore(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """A repair is judged by re-reading the venv, not by an exit code.

    An installer can exit 0 and leave core exactly where it was (a resolve that
    decided the wheel already "satisfies" the request). Reporting `restored:`
    off that exit code would hand the operator a receipt for something that
    never happened.
    """
    venv = FakeUv(honours_constraints=False, repair_noops=True)
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(_voice_manifest(tmp_path)))

    assert rc == 1
    assert venv.editable.get(CORE) is None  # still swapped — the repair did nothing
    err = capsys.readouterr().err
    assert "restored:" not in err
    assert (
        "RESTORE FAILED — run `uv pip install --python "
        f"{shlex.quote(sys.executable)} -e {CHECKOUT}` by hand."
    ) in err


def test_sync_recovery_command_names_pip_on_a_host_without_uv(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """The hand-run command must be the one that actually ran.

    On a host with no uv the installer falls back to this interpreter's pip.
    Advertising `uv pip install ...` there hands the operator a command their
    host cannot run — and, without `--python`, one that would install core into
    whatever environment happens to be active. Re-targeting core is the failure
    this guard exists to prevent, so the recovery command may not commit it.
    """
    venv = FakeUv(honours_constraints=False, repair_fails=True)
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr("shutil.which", lambda name: None)  # no uv on PATH

    rc = cli.cmd_feature_sync(_args(_voice_manifest(tmp_path)))

    assert rc == 1
    assert venv.editable.get(CORE) is None  # swapped, and the re-link failed
    err = capsys.readouterr().err
    assert (
        f"RESTORE FAILED — run `{shlex.quote(sys.executable)} -m pip install "
        f"-e {CHECKOUT}` by hand."
    ) in err
    assert "uv pip install" not in err


def test_sync_recovery_command_is_runnable_on_windows(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """...and quoted for, and labelled with, the shell that host actually has.

    Windows is a supported platform, and its interpreter path routinely contains
    a space (`C:\\Program Files\\...`). `shlex.join` quotes that POSIX-style, and
    no Windows shell will start `'C:\\Program Files\\venv\\python.exe'` — the
    quotes become part of the program name. PowerShell needs the `&` call
    operator to run a quoted path at all. A recovery command the operator's
    shell rejects is worth no more than printing none, which on a failed restore
    leaves core swapped with no way back.
    """
    venv = FakeUv(honours_constraints=False, repair_fails=True)
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr("shutil.which", lambda name: None)  # no uv on PATH
    monkeypatch.setattr(cli_features, "_is_windows", lambda: True)
    monkeypatch.setattr(
        cli_features.sys, "executable", r"C:\Program Files\kestrel\venv\python.exe"
    )

    rc = cli.cmd_feature_sync(_args(_voice_manifest(tmp_path)))

    assert rc == 1
    assert venv.editable.get(CORE) is None  # swapped, and the re-link failed
    err = capsys.readouterr().err
    assert (
        "RESTORE FAILED — run `& 'C:\\Program Files\\kestrel\\venv\\python.exe' "
        f"-m pip install -e {CHECKOUT}` in PowerShell by hand."
    ) in err
    # The interpreter is invoked, not echoed: a bare quoted path is a string
    # literal to PowerShell, so dropping `&` prints the path and repairs nothing.
    assert "run `'C:\\Program Files" not in err


# --- the single-package commands are guarded too ----------------------------


def test_feature_install_pins_core_to_the_editable_checkout(
    monkeypatch, fake_registry, capsys
):
    """`kestrel feature install` is the shortest path to the #2949 swap: one
    command, one feature, core replaced. It carries the same guard as sync."""
    venv = FakeUv()
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_install(types.SimpleNamespace(name="voice"))

    assert rc == 1
    assert venv.editable[CORE] == CHECKOUT  # link intact
    assert venv.installed[CORE] == "0.52.0"
    assert "kestrel-feature-voice" not in venv.installed
    # PyPI attempt AND the git fallback are both pinned — the fallback is not
    # a hole in the guard.
    assert venv.pins == ["==0.52.0", "==0.52.0"]
    assert "No solution found" in capsys.readouterr().out


def test_feature_upgrade_pins_core_to_the_editable_checkout(
    monkeypatch, fake_registry, capsys
):
    """`--upgrade` is the likeliest command to drag core forward, so it is
    guarded exactly like sync."""
    venv = FakeUv()
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [{
            "dist": "kestrel-feature-voice", "version": "0.3.0",
            "editable_path": None, "entries": [],
        }],
    )

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 1
    assert venv.editable[CORE] == CHECKOUT
    assert venv.installed[CORE] == "0.52.0"
    assert venv.pins == ["==0.52.0", "==0.52.0"]  # pip attempt + git fallback
    assert "FAILED" in capsys.readouterr().out
