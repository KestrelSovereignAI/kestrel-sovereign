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
from kestrel_sovereign import feature_reconcile as fr
from kestrel_sovereign.cli_features import CORE_UNSAFE
from tests.utils.fake_uv import (
    CHECKOUT,
    CORE,
    SDK,
    SDK_CHECKOUT,
    FakeUv,
    use_fake_uv,
)


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
        self.reinstalls = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, pip_args, *, constraints=None, reinstall=None, timeout=None):
        self.calls.append(list(pip_args))
        self.constraints.append(list(constraints or []))
        self.reinstalls.append(reinstall)
        return subprocess.CompletedProcess(pip_args, self.returncode, stdout="", stderr=self.stderr)


def _versions(monkeypatch, mapping):
    """Make importlib.metadata.version answer from *mapping* (else NotFound)."""
    import importlib.metadata as md

    def fake_version(name):
        if name in mapping:
            return mapping[name]
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", fake_version)


def _args(manifest, capture=False, dry_run=False, allow_dirty=False):
    return types.SimpleNamespace(
        manifest=str(manifest), capture=capture, dry_run=dry_run,
        allow_dirty=allow_dirty,
    )


# --- pure helpers ----------------------------------------------------------


def test_pip_spec_renders_extras():
    assert cli._pip_spec("pkg", []) == "pkg"
    assert cli._pip_spec("pkg", ["local"]) == "pkg[local]"
    assert cli._pip_spec("pkg", ["a", "b"]) == "pkg[a,b]"


def test_worst_rc_ranks_by_severity_not_by_value():
    """These exit codes are NOT ordered by their numbers.

    CORE_STALE is 3 and CORE_UNSAFE is 2, so `max()` on the raw ints ranks a
    stale core above an *undeclared* one and returns the weaker code for the
    more dangerous state. The ranking is explicit for exactly that reason.
    """
    from kestrel_sovereign.cli_features import CORE_STALE, _worst_rc

    assert _worst_rc(0, 0) == 0
    assert _worst_rc(0, 1) == 1
    # A core state always outranks an ordinary package failure...
    assert _worst_rc(1, CORE_STALE) == CORE_STALE
    assert _worst_rc(1, CORE_UNSAFE) == CORE_UNSAFE
    # ...and an undeclared core outranks a merely stale one.
    assert _worst_rc(CORE_STALE, CORE_UNSAFE) == CORE_UNSAFE
    # The trap this exists to avoid, stated as the fact it is:
    assert max(CORE_STALE, CORE_UNSAFE) == CORE_STALE


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
    # ...but currently installed EDITABLE from a checkout. Stub provenance, the
    # single seam both `_editable_install_path` and `_from_index` read, so the
    # modelled package cannot be editable to one helper and index-resolved to
    # the other.
    monkeypatch.setattr(
        cli, "_direct_url_provenance", lambda dist: fr.Provenance.direct("/src/voice", editable=True),
    )
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: "/src/voice")
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    # The wheel must replace the editable link (round 10 P2) — via a reinstall
    # scoped to this package, never a blanket --force-reinstall that would
    # cascade into core (#2949).
    assert spy.calls == [["kestrel-feature-voice>=0.3,<0.4"]]
    assert spy.reinstalls == ["kestrel-feature-voice"]
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
    # Stub provenance alongside it, or the shape read falls through to this
    # venv's real metadata and the assertions describe the developer's machine
    # instead of the modelled host.
    monkeypatch.setattr(cli, "_direct_url_provenance", lambda dist: fr.Provenance.from_index_install())
    spy = _InstallSpy(returncode=1, stderr="no matching distribution")
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    # CORE_UNSAFE, not 1: the declared core never installed, so the venv is left
    # without the core the manifest names — not merely an optional package that
    # failed, and not something --continue-on-error may carry past.
    assert rc == CORE_UNSAFE
    assert not any("git+" in str(arg) for call in spy.calls for arg in call)
    assert "FAILED" in capsys.readouterr().out


def test_sync_git_fallback_when_pip_fails(monkeypatch, fake_registry, tmp_path):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')
    _versions(monkeypatch, {})

    class _FailThenGit:
        def __init__(self):
            self.calls = []

        def __call__(self, pip_args, *, constraints=None, reinstall=None, timeout=None):
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

        def __call__(self, pip_args, *, constraints=None, reinstall=None, timeout=None):
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

        def __call__(self, pip_args, *, constraints=None, reinstall=None, timeout=None):
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
    # ...and its provenance, or capture reads THIS venv's editable core and
    # writes a real local path ahead of the entry under test.
    monkeypatch.setattr(cli, "_direct_url_provenance", lambda dist: fr.Provenance.from_index_install())
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

    assert rc == CORE_UNSAFE
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

    assert rc == CORE_UNSAFE
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

    assert rc == CORE_UNSAFE
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


# --- a feature's source switch must not reinstall core (#2949) --------------
#
# The version pin above bounds core's VERSION. It says nothing about core's
# SOURCE, and cannot: the index publishes the same version the checkout builds,
# so a same-version wheel satisfies `==0.52.0` exactly. The one command that
# reaches for that wheel anyway is a source switch — `--force-reinstall`
# applies to the whole resolve, and core is a resolved dependency of every
# feature package. Scoping the reinstall to the package being switched is what
# holds the source; these tests pin that, on both installer backends.

def _switch_voice_to_pypi(tmp_path):
    """Manifest that moves an editable `voice` install onto the index."""
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ">=0.3,<0.5"\n')
    return manifest


def _editable_voice(**kw):
    """A venv where BOTH core and the feature are editable checkouts.

    `feature_requires=">=0.52"` removes the version-skew route deliberately:
    the installed core satisfies the feature, so anything that happens to core
    here is the reinstall cascade and nothing else. The index carries 0.52.0 —
    the very version the checkout builds — which is what makes the pin blind
    to the swap.
    """
    venv = FakeUv(feature_requires=">=0.52", **kw)
    venv.installed["kestrel-feature-voice"] = "0.3.1"
    venv.editable["kestrel-feature-voice"] = "/src/voice"
    return venv


def test_blanket_force_reinstall_swaps_editable_core_through_the_pin(monkeypatch):
    """Fidelity check on the double: the pin does NOT stop a blanket reinstall.

    Without this the "core survived" assertions below could pass vacuously —
    and this is the exact claim the fix rests on, so it is asserted rather
    than assumed.
    """
    venv = _editable_voice()
    use_fake_uv(monkeypatch, venv)

    result = cli._extension_install_run(
        ["--force-reinstall", "kestrel-feature-voice>=0.3,<0.5"],
        constraints=[f"{CORE}==0.52.0"],
    )

    assert result.returncode == 0  # the install "succeeded"...
    assert venv.pins == ["==0.52.0"]  # ...with the pin applied...
    assert venv.installed[CORE] == "0.52.0"  # ...at the pinned version...
    assert venv.editable.get(CORE) is None  # ...and the link gone anyway.


def _pypi_core_guard(monkeypatch, **kw):
    """A guard whose manifest says core comes from the index, over an editable
    core — so :meth:`resolve` has a real repair to perform."""
    from kestrel_sovereign.cli_features import CoreInstallGuard
    from kestrel_sovereign.feature_reconcile import SourceEntry

    venv = FakeUv(**kw)
    use_fake_uv(monkeypatch, venv)
    index = {CORE: SourceEntry(package=CORE, pypi=">=0.52,<0.53")}
    return venv, CoreInstallGuard.snapshot(index)


def test_core_repair_does_not_drag_editable_dependencies_off_their_checkouts(
    monkeypatch,
):
    """The repair is scoped too — it is an install like any other.

    Putting core back from the index means displacing an editable link, and the
    blanket flag that does it reinstalls everything the resolve touches. Core's
    own dependencies are in that set: an editable SDK checkout comes back as an
    index wheel. That is issue #2949 committed by the code written to undo it,
    so the repair names the one package it is for.
    """
    venv, guard = _pypi_core_guard(monkeypatch)

    outcome = guard.resolve()

    assert outcome.repaired
    assert venv.editable.get(CORE) is None  # core is the declared wheel now...
    assert venv.editable[SDK] == SDK_CHECKOUT  # ...and its SDK is still linked
    assert not any("--force-reinstall" in c for c in venv.commands)
    cmd = venv.commands[-1]
    assert cmd[cmd.index("--reinstall-package") + 1] == CORE
    # What the operator is told to run is what ran — scoping included.
    assert outcome.command == (
        f"uv pip install --python {shlex.quote(sys.executable)} "
        f"--reinstall-package {CORE} '{CORE}>=0.52,<0.53'"
    )


def test_core_repair_recovery_command_carries_both_pip_passes(monkeypatch, capsys):
    """A pip host's repair is two commands, and the operator gets both.

    pip cannot scope a reinstall, so the repair splits (see
    `_install_commands`). Printing only the first pass advertises a restore
    that resolves dependencies and never replaces the link — half a repair,
    handed over as the whole one.
    """
    venv, guard = _pypi_core_guard(monkeypatch, repair_fails=True)
    monkeypatch.setattr("shutil.which", lambda name: None)  # no uv on PATH

    assert guard.verify() == CORE_UNSAFE  # repair failed: core is still wrong
    assert venv.editable[CORE] == CHECKOUT  # the repair failed; nothing moved
    py = shlex.quote(sys.executable)
    assert (
        f"RESTORE FAILED — run `{py} -m pip install --upgrade "
        f"'{CORE}>=0.52,<0.53' && {py} -m pip install --force-reinstall "
        f"--no-deps '{CORE}>=0.52,<0.53'` by hand."
    ) in capsys.readouterr().err


def test_windows_recovery_command_keeps_the_destructive_pass_conditional(
    monkeypatch, capsys
):
    """The printed sequence has to keep the ordering that makes it safe.

    Pass 2 replaces core with `--no-deps` and is only correct once pass 1's
    resolve has succeeded (`_install_commands`). PowerShell 5.1 has no `&&`,
    but joining with a bare `;` runs pass 2 regardless — so the command handed
    to a Windows operator would install a core whose dependencies the pass
    before it just refused to resolve. That is the hazard the two-pass split
    exists to prevent, re-entered through the printed fix.
    """
    venv, guard = _pypi_core_guard(monkeypatch, repair_fails=True)
    monkeypatch.setattr("shutil.which", lambda name: None)  # no uv on PATH
    monkeypatch.setattr(cli_features, "_is_windows", lambda: True)
    monkeypatch.setattr(
        cli_features.sys, "executable", r"C:\Program Files\kestrel\venv\python.exe"
    )

    assert guard.verify() == CORE_UNSAFE  # repair failed: core is still wrong
    assert venv.editable[CORE] == CHECKOUT  # the repair failed; nothing moved
    py = r"& 'C:\Program Files\kestrel\venv\python.exe'"
    assert (
        f"RESTORE FAILED — run `{py} -m pip install --upgrade "
        f"'{CORE}>=0.52,<0.53'; if ($?) {{ {py} -m pip install "
        f"--force-reinstall --no-deps '{CORE}>=0.52,<0.53' }}` in PowerShell "
        "by hand."
    ) in capsys.readouterr().err


# --- a repair is judged by re-reading core, not by an exit code (#2949) -----
#
# The no-op repair below already pins one direction: exit 0 over an unchanged
# venv is not a restore. These pin the mirror image, which is where the same
# mistake costs an operator a working host: an installer that ended badly
# AFTER core was already home.


def test_a_repair_whose_last_pass_failed_is_still_judged_by_where_core_is(
    monkeypatch, capsys
):
    """pip's repair is two passes, and the first one can restore core.

    `--upgrade` installs the declared wheel over the editable link; the
    `--no-deps` pass that follows only has to displace it. When THAT pass
    fails — a dropped connection, a build error — the installer exits nonzero
    over a core that already conforms. Reading the exit code sends the
    operator to run a restore that has happened, and fails the HTTP install of
    a host that is fine.

    Core starts at 0.51.0, OUTSIDE the declared `>=0.52,<0.53`, because pass 1
    is deliberately non-forcing (see `_install_commands`): it only writes when
    the installed version does not already satisfy the spec. A same-version
    source switch no-ops in pass 1 by design — that is the case pass 2 exists
    for, and modelling it as a write would assert the opposite of what the
    production code documents.
    """
    venv, guard = _pypi_core_guard(
        monkeypatch, repair_last_pass_fails=True, core_version="0.51.0",
    )
    monkeypatch.setattr("shutil.which", lambda name: None)  # no uv on PATH

    assert guard.verify() == 1  # the swap happened, so it is still reported...

    err = capsys.readouterr().err
    assert "restored:" in err  # ...but as one that was put back
    assert "RESTORE FAILED" not in err
    assert len(venv.commands) == 2  # both passes ran...
    assert venv.editable.get(CORE) is None  # ...and core is the declared wheel
    assert venv.installed[CORE] == "0.52.0"
    assert guard.verify() == 0  # nothing left to report on a second look


def test_repair_displaces_a_non_editable_direct_url_core_at_a_satisfying_version(
    monkeypatch, capsys,
):
    """Detection and repair must ask the same question about core's source.

    A core installed from a VCS ref (or local path, or archive) at a version
    the declared window accepts is a source violation the guard now names. But
    pip and uv judge "already satisfied" by VERSION, so re-resolving the spec
    writes nothing — and a repair that only scopes its reinstall for *editable*
    cores leaves this one exactly where it was. The drift would then be
    permanently unfixable: every run reports CORE_UNSAFE and prints a manual
    command that no-ops for the operator too.
    """
    venv, guard = _pypi_core_guard(
        monkeypatch,
        core_checkout=None,  # NOT editable — this is the case is_editable misses
        direct_urls={CORE: "git+https://example.invalid/core@abc"},
    )
    # Preconditions: the version is fine and the install is not editable, so
    # the ONLY thing wrong is where core came from.
    assert venv.installed[CORE] == "0.52.0"
    assert venv.editable.get(CORE) is None

    rc = guard.verify()

    # 1, not CORE_UNSAFE: the drift is still reported (nothing claims success
    # over a core that was replaced) but it was actually put back.
    assert rc == 1
    assert venv.direct_urls.get(CORE) is None  # the git copy is gone
    err = capsys.readouterr().err
    assert "RESTORE FAILED" not in err
    assert "--reinstall-package" in err  # the reinstall was scoped, not blanket
    assert guard.verify() == 0  # and it stays fixed


def test_dry_run_previews_the_source_only_core_drift_that_sync_repairs(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """A preview that says `present` for something sync reinstalls is a lie.

    Planning judged a `pypi` core by "is it editable?" while execution judges it
    by "did it come from an index?". A non-editable direct-URL core inside the
    declared window therefore previewed as `present` in `sync --dry-run` and
    `feature status`, while a real sync reinstalled it. Both now ask the guard's
    predicate, so the preview and the run cannot disagree.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text(f'[[feature]]\nname = "{CORE}"\npypi = ">=0.52,<0.53"\n')
    venv = FakeUv(
        core_checkout=None,
        direct_urls={CORE: "git+https://example.invalid/core@abc"},
    )
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(manifest, dry_run=True))

    assert rc == 0
    assert venv.commands == []  # a preview mutates nothing
    assert "would reinstall" in capsys.readouterr().out


def test_sync_succeeds_switching_a_direct_url_core_to_the_declared_index(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """Sync's own install must make the switch, not leave it to the guard.

    Scoping the reinstall only for EDITABLE cores made sync's install a
    same-version no-op against a direct-URL core. The final guard then did the
    real repair and reported drift, so `feature sync` — and a plain `kestrel
    update` — exited 1 over a run that had reached exactly the declared state.
    A command that did the right thing must not report failure for it.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text(f'[[feature]]\nname = "{CORE}"\npypi = ">=0.52,<0.53"\n')
    venv = FakeUv(
        core_checkout=None,
        direct_urls={CORE: "git+https://example.invalid/core@abc"},
    )
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0  # the switch succeeded, so nothing reports failure
    assert venv.direct_urls.get(CORE) is None  # and it actually happened
    err = capsys.readouterr().err
    assert "core: ERROR" not in err  # the guard had nothing left to repair


def test_unreadable_core_provenance_is_repaired_not_waved_through(
    monkeypatch, fake_registry, tmp_path,
):
    """End to end: damaged provenance must not pass as an index install.

    The version sits inside the declared window, so the only thing between this
    core and a clean `present` is whether "unknown" reads as "from the index".
    It does not — sync reinstalls from the declared source rather than trusting
    metadata it could not read.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text(f'[[feature]]\nname = "{CORE}"\npypi = ">=0.52,<0.53"\n')
    venv = FakeUv(core_checkout=None, unreadable_provenance={CORE})
    use_fake_uv(monkeypatch, venv)
    assert venv.installed[CORE] == "0.52.0"  # version is fine; source is unknown

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    # A scoped reinstall ran, rather than a same-version no-op that would have
    # left the unverifiable install in place.
    assert any("--reinstall-package" in cmd for cmd in venv.commands), venv.commands


def test_a_pypi_declared_feature_installed_from_git_is_drift_not_present(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """`pypi` names a SOURCE for features too, not only for core.

    A feature that reached the venv from a git URL sits at a satisfying version
    and is not editable, so a check built on editability called it `present`
    and sync never moved it back to the declared index source. There is no
    reinstall loop to fear here: sync's git fallback is gated on `pypi_want is
    None`, so a declared entry never falls back.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = ">=0.3,<0.4"\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.3.1"})  # in range...
    # ...but installed from a git ref, not the index.
    monkeypatch.setattr(
        cli,
        "_direct_url_provenance",
        lambda dist: fr.Provenance.direct("git+https://example.invalid/voice@abc"),
    )
    monkeypatch.setattr(cli, "_editable_install_path", lambda dist: None)
    spy = _InstallSpy()
    monkeypatch.setattr(cli, "_extension_install_run", spy)

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert spy.calls == [["kestrel-feature-voice>=0.3,<0.4"]]
    # Scoped, or the same-version index wheel is a no-op and nothing moves.
    assert spy.reinstalls == ["kestrel-feature-voice"]
    assert "reinstalled" in capsys.readouterr().out


def test_a_repair_killed_after_the_write_is_not_reported_as_a_failed_restore(
    monkeypatch,
):
    """A timeout ends a process; it does not undo what the process did.

    The HTTP surface bounds its repair, so the kill can land anywhere —
    including after the write that put core back. Treating "we stopped
    waiting" as "the restore failed" turns a host that conforms into a 500
    naming a command the operator does not need to run.
    """
    venv, guard = _pypi_core_guard(monkeypatch, repair_hangs_after_restore=True)

    outcome = guard.resolve(timeout=300)

    assert venv.editable.get(CORE) is None  # the write landed before the kill
    assert venv.installed[CORE] == "0.52.0"
    assert outcome.repaired
    assert outcome.conforming
    assert outcome.drift is not None  # the swap is still reported...
    assert outcome.output == ""  # ...but not as an unrepaired one


def test_sync_source_switch_leaves_the_editable_core_linked(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """The regression: an editable core at X survives a feature source switch
    while the index carries that same X."""
    venv = _editable_voice()
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(_switch_voice_to_pypi(tmp_path)))

    assert rc == 0
    assert venv.editable[CORE] == CHECKOUT  # the link the operator declared
    assert venv.installed[CORE] == "0.52.0"
    # The switch still took effect: voice IS the reinstall, off its checkout.
    assert venv.installed["kestrel-feature-voice"] == "0.4.0"
    assert "kestrel-feature-voice" not in venv.editable
    # No command may carry the blanket flag; the reinstall names one package.
    assert not any("--force-reinstall" in c for c in venv.commands)
    cmd = venv.commands[0]
    assert cmd[cmd.index("--reinstall-package") + 1] == "kestrel-feature-voice"
    captured = capsys.readouterr()
    assert "reinstalled" in captured.out
    # Prevention, not a rescue: the guard had nothing to detect or repair.
    assert "was replaced" not in captured.err


def test_sync_source_switch_keeps_core_on_a_host_without_uv(
    monkeypatch, fake_registry, tmp_path
):
    """pip has no `--reinstall-package`, and the protection is not dropped there.

    The switch becomes two passes, in this order: a non-forcing `--upgrade`
    that resolves the requested version's dependencies (so an editable core it
    still satisfies keeps its link), then a `--no-deps` force that replaces the
    package itself and can reach no dependency at all.
    """
    venv = _editable_voice()
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr("shutil.which", lambda name: None)  # no uv on PATH

    rc = cli.cmd_feature_sync(_args(_switch_voice_to_pypi(tmp_path)))

    assert rc == 0
    assert venv.editable[CORE] == CHECKOUT
    assert venv.installed[CORE] == "0.52.0"
    assert venv.installed["kestrel-feature-voice"] == "0.4.0"
    assert "kestrel-feature-voice" not in venv.editable  # switch took effect
    assert [c[:4] for c in venv.commands] == [
        [sys.executable, "-m", "pip", "install"],
        [sys.executable, "-m", "pip", "install"],
    ]
    # Resolve first, and it forces nothing...
    assert "--upgrade" in venv.commands[0]
    assert "--force-reinstall" not in venv.commands[0]
    # ...then replace, reaching nothing but the package itself.
    assert "--force-reinstall" in venv.commands[1] and "--no-deps" in venv.commands[1]
    # The pin still travels on both passes — bounding the version is still the
    # other half of the guard.
    assert venv.pins == ["==0.52.0", "==0.52.0"]


def test_sync_pip_source_switch_fails_before_replacing_the_feature(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """A switch whose dependencies cannot be satisfied changes nothing.

    The index build of `voice` wants a core the manifest's own pin forbids.
    That has no solution, and the pass that finds it out is the one that runs
    FIRST — so the sync fails with the working editable install still in place,
    rather than reporting a failure over a venv it already replaced.

    (uv resolves the whole switch before writing anything, so this is pip's
    ordering being asserted, not a behaviour uv shares by accident.)
    """
    venv = _editable_voice()
    venv.feature_requires = ">=0.53"  # the index build outgrew the pinned core
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr("shutil.which", lambda name: None)

    rc = cli.cmd_feature_sync(_args(_switch_voice_to_pypi(tmp_path)))

    assert rc == 1
    # The feature is untouched: same version, still linked to its checkout.
    assert venv.installed["kestrel-feature-voice"] == "0.3.1"
    assert venv.editable["kestrel-feature-voice"] == "/src/voice"
    assert venv.editable[CORE] == CHECKOUT  # and so is core
    assert len(venv.commands) == 1  # the destructive pass never ran
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "No solution found" in out


def test_sync_pip_source_switch_does_not_run_the_second_pass_after_a_failure(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """A failed first pass is reported, not papered over by a follow-up that
    would replace the package the failed resolve was about."""
    venv = _editable_voice(feature_install_fails=True)
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr("shutil.which", lambda name: None)

    rc = cli.cmd_feature_sync(_args(_switch_voice_to_pypi(tmp_path)))

    assert rc == 1
    assert len(venv.commands) == 1  # no second pass
    assert venv.editable[CORE] == CHECKOUT  # and core is still where it was
    assert "FAILED" in capsys.readouterr().out


# --- provenance reading, at the metadata boundary (#2949) -------------------


@pytest.mark.parametrize("body", [
    '{}',                                  # parses, but no url
    '{"url": ""}',                         # present and empty
    '{"url": "   "}',                      # whitespace only
    '{"url": 42}',                         # wrong type
    '{"dir_info": {"editable": true}}',    # editable flag, no url to link to
    '[]',                                  # valid JSON, wrong shape
    'not json at all',
    '',                                    # the file exists and says nothing
])
def test_a_damaged_direct_url_file_reads_as_unknown_not_as_an_index_install(
    monkeypatch, body,
):
    """PEP 610 requires ``url``. Anything that parses without a usable one is
    damage, and damage is UNKNOWN — every one of these shapes previously
    reported "no direct URL", which is the positive evidence of an index
    install, so a declared `pypi` source was satisfied by a core nobody could
    verify.
    """
    import importlib.metadata as md

    class _Dist:
        def read_text(self, name):
            return body

    monkeypatch.setattr(md, "distribution", lambda name: _Dist())

    prov = cli_features._direct_url_provenance("kestrel-sovereign")

    assert not prov.is_from_index
    assert not prov.known
    assert prov.editable_path is None


def test_a_real_direct_url_file_still_reads_as_that_source(monkeypatch):
    """The counterpart: a well-formed file is read, not rejected — otherwise
    'fail closed' would just mean 'always fail'."""
    import importlib.metadata as md

    class _Dist:
        def read_text(self, name):
            return '{"url": "git+https://example.invalid/c@abc"}'

    monkeypatch.setattr(md, "distribution", lambda name: _Dist())

    prov = cli_features._direct_url_provenance("kestrel-sovereign")

    assert prov.known and not prov.is_from_index
    assert prov.url == "git+https://example.invalid/c@abc"


def test_no_direct_url_file_is_a_known_index_install(monkeypatch):
    """And the absent file remains the one thing that DOES mean 'from an
    index' — the distinction the whole type exists to preserve."""
    import importlib.metadata as md

    class _Dist:
        def read_text(self, name):
            return None

    monkeypatch.setattr(md, "distribution", lambda name: _Dist())

    prov = cli_features._direct_url_provenance("kestrel-sovereign")

    assert prov.known and prov.is_from_index


def test_a_declared_editable_core_is_pulled_before_it_is_linked(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """Linking a declared editable core without pulling it restarts on stale code.

    Nothing else covers this checkout: `kestrel update` step 1 pulls the one the
    *currently installed* core lives in — None when core is a wheel, the wrong
    one when the manifest declares a different checkout — and reconcile, which
    does pull every editable entry it installs, excludes core because it is
    bundled. So the fleet could come back up on whatever happened to be on disk.
    """
    checkout = tmp_path / "core-checkout"
    checkout.mkdir()
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[[feature]]\nname = "{CORE}"\neditable = "{checkout}"\n'
    )
    # core is a wheel; the manifest says otherwise. `checkouts` models what
    # that checkout builds, the way a real `-e` install takes its version.
    venv = FakeUv(core_checkout=None, checkouts={str(checkout): "0.52.0"})
    use_fake_uv(monkeypatch, venv)

    pulled = []
    monkeypatch.setattr(
        "kestrel_sovereign.cli_lifecycle._editable_git_pull",
        lambda path, allow_dirty: pulled.append((str(path), allow_dirty)) or (0, "Already up to date."),
    )

    rc = cli.cmd_feature_sync(_args(manifest))

    assert rc == 0
    assert pulled == [(str(checkout), False)]  # pulled, and never over a dirty tree
    assert venv.editable.get(CORE) == str(checkout)  # ...and then linked


def test_a_dirty_core_checkout_is_reported_not_silently_linked_stale(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """A refused pull must not fail the sync — but must not be silent either.

    A dirty core checkout is an ordinary dev state, so refusing to link over it
    would be hostile. Saying nothing would put the operator back where they
    started: restarted onto code that did not move, with no way to know.
    """
    checkout = tmp_path / "core-checkout"
    checkout.mkdir()
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[[feature]]\nname = "{CORE}"\neditable = "{checkout}"\n'
    )
    venv = FakeUv(core_checkout=None, checkouts={str(checkout): "0.52.0"})
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr(
        "kestrel_sovereign.cli_lifecycle._editable_git_pull",
        lambda path, allow_dirty: (2, "REFUSED — checkout has modified tracked files"),
    )

    rc = cli.cmd_feature_sync(_args(manifest))

    out = capsys.readouterr().out
    # CORE_STALE, not the generic 1: the link is still the right end state, but
    # `kestrel update` must not restart and report SUCCESS over code that never
    # moved — and `--continue-on-error` ignores 1, so reporting non-zero was not
    # enough on its own.
    assert rc == cli_features.CORE_STALE
    assert "REFUSED" in out
    assert "NOT updated" in out  # the staleness is named
    assert venv.editable.get(CORE) == str(checkout)


def test_a_later_feature_failure_cannot_downgrade_a_stale_core(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """CORE_STALE must survive an ordinary package failure later in the batch.

    Core is synced FIRST, so every optional feature is installed after it. `rc`
    carried both facts in one int, so a failed optional feature overwrote
    CORE_STALE with 1 — and `--continue-on-error` ignores 1, restarting the
    fleet onto the stale checkout and exiting 0. That is precisely the outcome
    CORE_STALE was added to prevent, reintroduced one step later.
    """
    checkout = tmp_path / "core-checkout"
    checkout.mkdir()
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[[feature]]\nname = "{CORE}"\neditable = "{checkout}"\n'
        '\n[[feature]]\nname = "voice"\n'
    )
    # `feature_requires` that the INSTALLED core already satisfies: the feature
    # fails on its own build, so nothing here moves core. Any CORE_STALE can
    # only have come from the refused pull, not from drift the feature caused.
    venv = FakeUv(
        core_checkout=None,
        checkouts={str(checkout): "0.52.0"},
        feature_requires=">=0.5",
        feature_install_fails=True,
    )
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr(
        "kestrel_sovereign.cli_lifecycle._editable_git_pull",
        lambda path, allow_dirty: (2, "REFUSED — checkout has modified tracked files"),
    )

    rc = cli.cmd_feature_sync(_args(manifest))

    out = capsys.readouterr().out
    # Both facts are real in this run — the premise of the test, not its claim.
    assert "NOT updated" in out       # core is stale
    assert "Failed to build" in out   # and an optional feature genuinely failed
    # The claim: the stronger code survives the weaker one.
    assert rc == cli_features.CORE_STALE


def test_an_unreadable_direct_url_file_is_unknown_not_an_index_install(
    monkeypatch, tmp_path,
):
    """`read_text` returning None does NOT mean the file is absent.

    `importlib.metadata`'s PathDistribution suppresses FileNotFoundError and
    PermissionError together and returns None for both, so a direct_url.json
    that exists but cannot be read arrives identical to one never written — and
    "never written" is the positive evidence of an index install. A
    permission-damaged editable core would therefore have read as from-index and
    left the guard unconstrained, which is the fail-open this reader exists to
    close.

    Uses a real unreadable file rather than a stub: the bug lives in what the
    stdlib does with the exception, so a double that raises would test the
    double.
    """
    import importlib.metadata as md

    dist_info = tmp_path / "kestrel_sovereign-0.53.0.dist-info"
    dist_info.mkdir()
    target = dist_info / "direct_url.json"
    target.write_text('{"url": "/src/core", "dir_info": {"editable": true}}')
    target.chmod(0o000)
    try:
        readable = False
        try:
            target.read_text()
            readable = True   # running as root, or a filesystem without perms
        except PermissionError:
            pass
        if readable:
            pytest.skip("filesystem does not enforce file permissions here")

        monkeypatch.setattr(
            md, "distribution", lambda name: md.PathDistribution(dist_info),
        )
        prov = cli_features._direct_url_provenance("kestrel-sovereign")
    finally:
        target.chmod(0o600)

    assert not prov.known          # present but unreadable
    assert not prov.is_from_index  # and so NOT positive evidence of an index


def test_a_genuinely_absent_direct_url_file_is_still_an_index_install(
    monkeypatch, tmp_path,
):
    """The counterpart, so 'fail closed' does not degrade into 'always hold':
    an install with no direct_url.json is exactly what an index resolution
    looks like, and must keep reading as one."""
    import importlib.metadata as md

    dist_info = tmp_path / "kestrel_sovereign-0.53.0.dist-info"
    dist_info.mkdir()
    monkeypatch.setattr(
        md, "distribution", lambda name: md.PathDistribution(dist_info),
    )

    prov = cli_features._direct_url_provenance("kestrel-sovereign")

    assert prov.known and prov.is_from_index


def test_provenance_carries_the_vcs_revision_not_just_the_repo_url(monkeypatch):
    """PEP 610 stores the revision in `vcs_info`, not in `url`.

    Two commits of one repository share a url, so a Provenance built from the
    url alone reports them identical — and a same-version replacement between
    them reads as no change. That is the version-pin mistake this change exists
    to correct, moved one field inward.
    """
    import importlib.metadata as md

    def _dist(commit):
        class _D:
            def read_text(self, name):
                return json.dumps({
                    "url": "https://github.com/example/core",
                    "vcs_info": {"vcs": "git", "commit_id": commit,
                                 "requested_revision": "main"},
                })
        return _D()

    monkeypatch.setattr(md, "distribution", lambda name: _dist("aaa111"))
    a = cli_features._direct_url_provenance("kestrel-sovereign")
    monkeypatch.setattr(md, "distribution", lambda name: _dist("bbb222"))
    b = cli_features._direct_url_provenance("kestrel-sovereign")

    assert a.url == b.url                # same repository...
    assert a.revision != b.revision      # ...different commit
    assert a.source_id != b.source_id    # ...so a different SOURCE
    assert "aaa111" in a.describe()      # and the operator is told which


def test_provenance_carries_archive_hash_and_subdirectory(monkeypatch):
    """The same for the other identity fields PEP 610 keeps outside `url`."""
    import importlib.metadata as md

    def _dist(payload):
        class _D:
            def read_text(self, name):
                return json.dumps(payload)
        return _D()

    base = {"url": "https://example.invalid/core.tar.gz"}
    monkeypatch.setattr(md, "distribution", lambda name: _dist(
        {**base, "archive_info": {"hashes": {"sha256": "aaa"}}}))
    a = cli_features._direct_url_provenance("kestrel-sovereign")
    monkeypatch.setattr(md, "distribution", lambda name: _dist(
        {**base, "archive_info": {"hashes": {"sha256": "bbb"}}}))
    b = cli_features._direct_url_provenance("kestrel-sovereign")
    assert a.url == b.url and a.source_id != b.source_id

    monkeypatch.setattr(md, "distribution", lambda name: _dist(
        {**base, "subdirectory": "pkg_a"}))
    c = cli_features._direct_url_provenance("kestrel-sovereign")
    monkeypatch.setattr(md, "distribution", lambda name: _dist(
        {**base, "subdirectory": "pkg_b"}))
    d = cli_features._direct_url_provenance("kestrel-sovereign")
    assert c.url == d.url and c.source_id != d.source_id


def test_provenance_keeps_the_vcs_kind(monkeypatch):
    """PEP 610 requires `vcs_info.vcs` and keeps it OUTSIDE `url`.

    `url` is the bare transport address, so the same address served by two
    different VCS at one revision string produced identical identities. The
    derived source_id picks this up automatically once the field exists — which
    is the point of deriving it.
    """
    import importlib.metadata as md

    def _dist(vcs):
        class _D:
            def read_text(self, name):
                return json.dumps({
                    "url": "https://example.invalid/core",
                    "vcs_info": {"vcs": vcs, "commit_id": "abc123"},
                })
        return _D()

    monkeypatch.setattr(md, "distribution", lambda name: _dist("git"))
    a = cli_features._direct_url_provenance("kestrel-sovereign")
    monkeypatch.setattr(md, "distribution", lambda name: _dist("hg"))
    b = cli_features._direct_url_provenance("kestrel-sovereign")

    assert a.url == b.url and a.revision == b.revision  # indistinguishable before
    assert a.vcs == "git" and b.vcs == "hg"
    assert a.source_id != b.source_id
    assert "git" in a.describe()


def test_a_failed_core_action_stops_the_batch(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """Features must not install against a core that failed to reach its source.

    `install_core` refreshes the guard from whatever core is NOW — the old
    version, or a partial write — so every remaining entry would resolve and pin
    against a core the manifest says is wrong. A later successful repair can
    then move core to the declared version those features were never resolved
    against, and `--continue-on-error` would restart the fleet on exactly that
    combination: a venv no single step reports as broken.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[[feature]]\nname = "voice"\npypi = ">=0.3,<0.5"\n'
        f'[[feature]]\nname = "{CORE}"\npypi = ">=0.52,<0.53"\n'
    )
    # Core is editable, the manifest says index — a real transition — and the
    # core install fails.
    venv = FakeUv(repair_fails=True)
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_args(manifest))

    out = capsys.readouterr().out
    assert rc != 0
    assert "rest of this batch is skipped" in out
    # The feature was never installed against the wrong core. Core is ordered
    # first (_core_entry_first), so nothing before it ran either.
    assert not any(
        "kestrel-feature-voice" in " ".join(cmd) for cmd in venv.commands
    ), venv.commands


def test_the_manifest_rejects_a_malformed_pypi_spec(tmp_path):
    """Caught where the operator can see which line is wrong.

    Downstream it fails silently: the spec renders into a constraint naming a
    different package, and every version satisfies the window. Failing at load
    keeps a typo from becoming a guard that reports success over an unprotected
    core.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname = "voice"\npypi = "banana"\n')

    with pytest.raises(ValueError) as exc:
        cli_features._load_host_manifest(manifest)

    msg = str(exc.value)
    assert "not a usable PEP 440" in msg and "banana" in msg

    # `===` parses as a SpecifierSet but carries no version operand, so it
    # matches nothing. The loader uses the GUARD's validator precisely so the
    # two cannot disagree about what is usable.
    manifest.write_text('[[feature]]\nname = "voice"\npypi = "==="\n')
    with pytest.raises(ValueError, match="not a usable PEP 440"):
        cli_features._load_host_manifest(manifest)

    # The valid forms still load, including "" (any version from the index).
    for spec in (">=0.3,<0.4", "", "==1.2.3"):
        manifest.write_text(f'[[feature]]\nname = "voice"\npypi = "{spec}"\n')
        assert cli_features._load_host_manifest(manifest)[0]["pypi"] == spec


def test_a_failed_core_extras_ensure_does_not_skip_the_rest_of_the_batch(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """"The core action failed" and "core is off its source" are not the same.

    A core entry that already conforms but declares extras yields an `ensure`
    action. A failed optional extra says nothing about where core came from —
    but the batch-stop treated ANY core failure as a failed source transition
    and skipped every remaining manifest entry, so `--continue-on-error`
    restarted with packages still pruned by the preceding `uv sync`.

    The core install is failed at the guard seam rather than through a FakeUv
    knob: the knobs that fail an install also govern the guard's own repair, so
    they would move the very core state this asserts is untouched.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[[feature]]\nname = "{CORE}"\neditable = "{CHECKOUT}"\nextras = ["local"]\n'
        '[[feature]]\nname = "voice"\npypi = ">=0.3,<0.5"\n'
    )
    # Core already editable at CHECKOUT, so it conforms. `feature_requires` is
    # relaxed so the following entry is genuinely installable against this core
    # — otherwise the pin blocks it for a legitimate version reason and the test
    # would pass without proving the batch continued.
    venv = FakeUv(feature_requires=">=0.52")
    use_fake_uv(monkeypatch, venv)

    real_install_core = cli_features.CoreInstallGuard.install_core

    def _failing_install_core(self, pip_args, **kw):
        # Nonzero, and the venv is left exactly as it was: the extra did not
        # install, core did not move.
        return subprocess.CompletedProcess(
            ["uv", "pip", "install", *pip_args], 1, stdout="", stderr="no such extra",
        )

    monkeypatch.setattr(
        cli_features.CoreInstallGuard, "install_core", _failing_install_core,
    )

    rc = cli.cmd_feature_sync(_args(manifest))

    out = capsys.readouterr().out
    assert rc != 0                                   # the extra really did fail
    assert "rest of this batch is skipped" not in out
    assert venv.editable[CORE] == CHECKOUT           # core never moved
    # ...so the entry after it was still restored.
    assert venv.installed.get("kestrel-feature-voice") == "0.4.0", venv.commands


def test_allow_dirty_reaches_the_core_pull(monkeypatch, fake_registry, tmp_path):
    """The flag has to arrive at the step that honours it.

    Both public routes dropped it: `feature sync` never registered the option,
    and `cmd_update` omitted it when building sync_args — so the pull was always
    called with allow_dirty=False while the failure message told the operator to
    "pass --allow-dirty", a flag one parser rejected and the other ignored.
    """
    manifest = tmp_path / "m.toml"
    manifest.write_text(f'[[feature]]\nname = "{CORE}"\neditable = "{CHECKOUT}"\n')

    def _run(allow_dirty):
        # Fresh state each time: the first sync links core, so a second run has
        # nothing to install and would never reach the pull at all.
        venv = FakeUv(core_checkout=None, checkouts={CHECKOUT: "0.52.0"})
        use_fake_uv(monkeypatch, venv)
        seen: list = []
        monkeypatch.setattr(
            "kestrel_sovereign.cli_lifecycle._editable_git_pull",
            lambda checkout, allow_dirty: seen.append(allow_dirty) or (0, "ok"),
        )
        cli.cmd_feature_sync(_args(manifest, allow_dirty=allow_dirty))
        return seen

    assert _run(True) == [True], "sync dropped --allow-dirty before the pull"
    assert _run(False) == [False], "sync invented an allow_dirty nobody asked for"


def test_update_forwards_allow_dirty_to_feature_sync(tmp_path):
    """`kestrel update --allow-dirty` must not lose the flag at the sync step."""
    import argparse
    from unittest.mock import patch

    from kestrel_sovereign import cli_lifecycle

    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").touch()
    # The sync step is gated on the manifest EXISTING — without one, update
    # skips the restore entirely and the test would pass by never running it.
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')
    captured: list = []

    with patch.object(cli, "_git_working_tree_dirty", lambda _: (False, "")), \
         patch.object(cli, "_run_git_pull", lambda _: (0, "")), \
         patch.object(cli, "_run_uv_pip_install_editable", lambda *a, **kw: (0, "")), \
         patch.object(cli, "_host_manifest_path", lambda a: manifest), \
         patch.object(cli, "cmd_feature_sync",
                      lambda a: captured.append(getattr(a, "allow_dirty", "MISSING")) or 0), \
         patch.object(cli_lifecycle, "_run_feature_reconcile", lambda *a, **kw: 0), \
         patch.object(cli, "cmd_restart", lambda a: 0), \
         patch.object(cli, "_get_project_dir", lambda: tmp_path), \
         patch.object(cli, "_resolve_source_checkout", lambda: tmp_path):
        ns = argparse.Namespace(
            name=None, pull=False, install=False, features=True, restart=False,
            allow_dirty=True, no_deps=False, continue_on_error=False,
            dry_run=False, manifest=None, subprocess=False, force=False,
            uv_sync=None,
        )
        cli.cmd_update(ns)

    assert captured == [True], f"update dropped --allow-dirty: {captured}"
