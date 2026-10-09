"""Feature installs are held to core's ``uv.lock`` (#3502).

A host ran ``anthropic`` 1.11.0 against a lock pinning 0.117.0: ``uv sync`` put
the locked version in, and the update's feature installs moved it straight
back out, because ``uv pip`` never reads the lock and reconcile's ``--upgrade``
is eager. Every feature-install surface goes through one guard, so these drive
each of them against the shared fake venv (``tests/utils/fake_uv.py``) and pin
that the lock's versions reach the resolver and hold. The drift check that
reports a venv off the lock is driven through ``kestrel doctor`` and
``kestrel update`` here; its comparison is tested in ``test_core_lock.py``.
"""

from __future__ import annotations

import argparse
import shlex
import tempfile
import types
from pathlib import Path

import pytest

from kestrel_sovereign import cli, cli_lifecycle, paths
from kestrel_sovereign import core_lock as cl
from kestrel_sovereign import feature_reconcile as fr
from kestrel_sovereign.cli_features import (
    CORE_UNSAFE, DEFAULT_HOST_MANIFEST, CoreInstallGuard,
)
from kestrel_sovereign.doctor import DoctorReport, _check_core_lock, diagnose
from kestrel_sovereign.feature_registry import FeaturePackageInfo
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from tests.utils.fake_uv import CORE, SDK, SDK_CHECKOUT, FakeUv, use_fake_uv

VOICE = "kestrel-feature-voice"


def _lock_text(anthropic="0.117.0", voice=None) -> str:
    """A lock that pins anthropic, core, and core's SDK, and *voice* if given.

    The SDK is locked AND (in the fake venv) linked from a checkout at the
    locked version. Unless a manifest declares that checkout, its pin is the
    lock's like any other, and at that version it keeps the link.
    """
    text = (
        "version = 1\nrevision = 3\n"
        '\n[[package]]\nname = "anthropic"\n'
        f'version = "{anthropic}"\nsource = {{ registry = "https://pypi.org/simple" }}\n'
        f'\n[[package]]\nname = "{CORE}"\nversion = "0.52.0"\nsource = {{ editable = "." }}\n'
        f'\n[[package]]\nname = "{SDK}"\nversion = "0.36.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    if voice is not None:
        text += (
            f'\n[[package]]\nname = "{VOICE}"\nversion = "{voice}"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
        )
    return text


def _undetermined_lock_text(package="anthropic") -> str:
    """``_lock_text()`` plus a fork of *package* whose markers will not evaluate.

    The lock reads and pins everything else, but which version of *package*
    this environment installs cannot be told, so the reader names it
    ``undetermined`` rather than guess. It has no version to pin.
    """
    return _lock_text() + (
        f'\n[[package]]\nname = "{package}"\nversion = "1.11.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        'resolution-markers = ["this is not a marker"]\n'
    )


def _checkout(tmp_path, name="kestrel-sovereign", lock: str | None = None) -> str:
    """A core checkout on disk, with *lock* as its uv.lock (None: no lock)."""
    checkout = tmp_path / name
    checkout.mkdir()
    if lock is not None:
        (checkout / "uv.lock").write_text(lock, encoding="utf-8")
    return str(checkout)


def _venv(core_checkout, *, voice_requires=("anthropic>=0.77",), **kw) -> FakeUv:
    """Editable core at *core_checkout*, anthropic installed at the locked 0.117.0.

    The index publishes 1.11.0 too, so an install free to move anthropic does.
    """
    venv = FakeUv(core_checkout=core_checkout, feature_requires=">=0.52", **kw)
    venv.installed["anthropic"] = "0.117.0"
    venv.package_index["anthropic"] = ["0.117.0", "1.11.0"]
    venv.installed_requires[VOICE] = list(voice_requires)
    return venv


def _lines(constraint_file: str) -> list:
    return [line for line in constraint_file.splitlines() if line.strip()]


@pytest.fixture(autouse=True)
def _own_project_home(monkeypatch, tmp_path):
    """A guard given no manifest reads the project directory's host manifest.

    Every test here gets its own project directory (``KESTREL_HOME``), so the
    manifest it reads is the one the test writes there, or none. The cwd is a
    sibling, never the project directory: the guard must not read it.
    """
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))
    launch = tmp_path.parent / f"{tmp_path.name}-launch"
    launch.mkdir()
    monkeypatch.chdir(launch)


@pytest.fixture
def fake_registry(monkeypatch):
    registry = {
        "voice": types.SimpleNamespace(
            package=VOICE,
            git="https://github.com/KestrelSovereignAI/kestrel-feature-voice.git",
            features=["VoiceFeature"],
            core=False,
        ),
    }
    monkeypatch.setattr(
        "kestrel_sovereign.feature_registry.load_registry", lambda: registry,
    )
    return registry


def _sync_args(manifest):
    return types.SimpleNamespace(
        manifest=str(manifest), capture=False, dry_run=False, allow_dirty=False,
    )


def _voice_manifest(tmp_path, extra=""):
    manifest = tmp_path / "m.toml"
    manifest.write_text(extra + '[[feature]]\nname = "voice"\npypi = ">=0.4"\n')
    return manifest


# --- feature sync (the path the live host's update reported) -----------------


def test_feature_sync_passes_the_lock_pins_to_the_resolver(
    monkeypatch, fake_registry, tmp_path,
):
    venv = _venv(_checkout(tmp_path, lock=_lock_text()))
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    assert rc == 0
    assert venv.installed[VOICE] == "0.4.0"
    (constraints,) = venv.constraint_files
    lines = _lines(constraints)
    assert "anthropic===0.117.0" in lines
    # Core keeps its own pin, never the lock's line for it.
    assert f"{CORE}==0.52.0" in lines
    assert lines.count(f"{CORE}==0.52.0") == 1
    # The SDK is linked from a checkout no manifest declares, so the lock's
    # version holds it like any other. At that version the pin keeps the link.
    assert f"{SDK}===0.36.0" in lines
    assert venv.editable[SDK]


def test_a_feature_needing_a_version_the_lock_does_not_pin_fails_without_moving_it(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    venv = _venv(
        _checkout(tmp_path, lock=_lock_text()), voice_requires=("anthropic>=1",),
    )
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    out = capsys.readouterr().out
    assert rc == 1
    assert "No solution found" in out
    assert venv.installed["anthropic"] == "0.117.0"
    assert VOICE not in venv.installed
    # The operator is told the LOCK refused it, and what moving it takes.
    assert "core's uv.lock" in out
    assert "uv lock --upgrade-package" in out


def test_without_a_lock_the_same_install_moves_the_package(
    monkeypatch, fake_registry, tmp_path,
):
    """Control for the test above: the lock is what refused, not the double."""
    venv = _venv(_checkout(tmp_path, lock=None), voice_requires=("anthropic>=1",))
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    assert rc == 0
    assert venv.installed["anthropic"] == "1.11.0"


def _unusable_lock(checkout: str, kind: str) -> None:
    """Make *checkout*'s uv.lock exist and unusable, as *kind* says.

    ``undetermined`` reads, but names no single version of anthropic here, so
    an install cannot be held at a locked version of it.
    """
    path = Path(checkout) / "uv.lock"
    if kind == "unparseable":
        path.write_text("this is = not [a lock", encoding="utf-8")
    elif kind == "directory":
        path.mkdir()
    elif kind == "undetermined":
        path.write_text(_undetermined_lock_text(), encoding="utf-8")


@pytest.mark.parametrize("kind", ["unparseable", "directory", "undetermined"])
def test_an_unreadable_lock_refuses_feature_installs_without_running_one(
    monkeypatch, fake_registry, tmp_path, capsys, kind,
):
    """A ``uv.lock`` that exists and will not read is not "no lock".

    The directory case is the one ``is_file()`` read as absent, so the install
    ran with no lock pins at all (#3502). The undetermined case read, and ran
    held to every locked package but the one it named no version of.
    """
    checkout = _checkout(tmp_path)
    _unusable_lock(checkout, kind)
    venv = _venv(checkout)
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    out = capsys.readouterr().out
    assert rc == 1
    assert venv.commands == []  # nothing ran, git fallback included
    assert "could not hold this install" in out
    if kind == "undetermined":
        assert "no single version for this environment of: anthropic" in out


def test_a_lock_naming_no_single_version_refuses_feature_install(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """The reported repro: anthropic's lock markers will not evaluate here.

    Before the fix the guard read the lock, found no error, and ran the
    install held to core and the SDK only, so a feature needing anthropic 1.x
    moved it. The drift check named the package, but only after the install.
    """
    venv = _venv(
        _checkout(tmp_path, lock=_undetermined_lock_text()),
        voice_requires=("anthropic>=1",),
    )
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_install(argparse.Namespace(name="voice"))

    out = capsys.readouterr().out
    assert rc != 0
    assert venv.commands == []
    assert venv.installed["anthropic"] == "0.117.0"
    assert "no single version for this environment of: anthropic" in out


def test_moving_core_holds_the_rest_of_the_batch_to_the_new_checkouts_lock(
    monkeypatch, fake_registry, tmp_path,
):
    old = _checkout(tmp_path, "old", lock=_lock_text("0.117.0"))
    new = _checkout(tmp_path, "new", lock=_lock_text("0.118.0"))
    venv = _venv(old, checkouts={new: "0.53.0"})
    venv.package_index["anthropic"].append("0.118.0")
    manifest = _voice_manifest(
        tmp_path, f'[[feature]]\nname = "{CORE}"\neditable = "{new}"\n',
    )
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(manifest))

    assert rc == 0
    assert len(venv.commands) == 2  # core, then voice
    # Core itself is the operator moving core: never pinned against itself,
    # bound by the manifest's windows (#3106) and by the lock beside the
    # checkout it is moving TO, which is what CI tested that core against.
    core_lines = _lines(venv.constraint_files[0])
    assert core_lines[0] == f"{VOICE}>=0.4"
    assert "anthropic===0.118.0" in core_lines
    assert "anthropic===0.117.0" not in core_lines
    assert venv.pins[0] is None
    # The feature after it is held to the lock beside the checkout core became.
    lines = _lines(venv.constraint_files[-1])
    assert "anthropic===0.118.0" in lines
    assert "anthropic===0.117.0" not in lines
    assert venv.installed["anthropic"] == "0.118.0"


def test_a_core_install_refused_by_its_target_lock_names_that_lock(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """The note names the lock that bound the CORE install: the target's."""
    old = _checkout(tmp_path, "old", lock=_lock_text("0.117.0"))
    new = _checkout(tmp_path, "new", lock=_lock_text("0.200.0"))  # never published
    venv = _venv(old, checkouts={new: "0.53.0"})
    venv.installed_requires[CORE] = ["anthropic>=0.77"]
    manifest = tmp_path / "m.toml"
    manifest.write_text(f'[[feature]]\nname = "{CORE}"\neditable = "{new}"\n')
    use_fake_uv(monkeypatch, venv)

    cli.cmd_feature_sync(_sync_args(manifest))

    out = capsys.readouterr().out
    assert "No solution found" in out
    assert f"core's uv.lock ({Path(new) / 'uv.lock'})" in out
    assert venv.installed["anthropic"] == "0.117.0"


@pytest.mark.parametrize("kind", ["unparseable", "undetermined"])
def test_an_unreadable_target_lock_refuses_the_core_install(
    monkeypatch, fake_registry, tmp_path, capsys, kind,
):
    """A core entry whose checkout's lock will not read installs nothing.

    Core already conforms (the entry only asks for extras), so nothing follows
    the refusal: no install ran at all, and the operator is told why. A lock
    naming no single version of a package refuses the same way: the extras'
    resolution would be free to move it.
    """
    checkout = _checkout(tmp_path)
    _unusable_lock(checkout, kind)
    venv = _venv(checkout)
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[[feature]]\nname = "{CORE}"\neditable = "{checkout}"\n'
        'extras = ["observability"]\n'
    )
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(manifest))

    out = capsys.readouterr().out
    assert rc != 0
    assert venv.commands == []  # no installer ran
    assert venv.editable[CORE] == checkout
    assert "could not hold this install" in out


def _pinned(lines, package) -> bool:
    return any(line.startswith(f"{package}===") for line in lines)


def _held(lines, package, checkout) -> bool:
    return cl.checkout_hold_line(package, checkout) in lines


# --- the rule: lock + declared checkouts, nothing else ----------------------


@pytest.mark.parametrize(
    "installed_as", ["index", "editable", "direct_url", "unknown_provenance"],
)
def test_a_locked_package_is_pinned_however_it_is_installed(
    monkeypatch, fake_registry, tmp_path, capsys, installed_as,
):
    """The ruling's repro: a feature needs ``anthropic>=1`` in a plain install.

    Anthropic is locked at 0.117.0 and installed at it from the index, linked
    from a checkout, installed from a URL, or with a ``direct_url.json`` that
    will not read. However it got there, the install carries
    ``anthropic===0.117.0``, so it is refused and anthropic stays as it was.
    Each rule that read the venv to leave a linked package free failed some
    state of it (#3502).
    """
    venv = _venv(
        _checkout(tmp_path, lock=_lock_text()), voice_requires=("anthropic>=1",),
    )
    if installed_as == "editable":
        venv.editable["anthropic"] = "/src/anthropic"
    elif installed_as == "direct_url":
        venv.direct_urls["anthropic"] = "git+https://example.invalid/anthropic@abc"
    elif installed_as == "unknown_provenance":
        venv.unreadable_provenance.add("anthropic")
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    assert rc == 1
    assert "anthropic===0.117.0" in _lines(venv.constraint_files[0])
    assert venv.installed["anthropic"] == "0.117.0"
    assert VOICE not in venv.installed
    if installed_as == "editable":
        assert venv.editable["anthropic"] == "/src/anthropic"
    assert "core's uv.lock" in capsys.readouterr().out


def test_without_a_lock_a_linked_package_is_replaced_from_the_index(
    monkeypatch, fake_registry, tmp_path,
):
    """Control for the test above: the fake does replace a free link."""
    venv = _venv(_checkout(tmp_path, lock=None), voice_requires=("anthropic>=1",))
    venv.editable["anthropic"] = "/src/anthropic"
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    assert rc == 0
    assert venv.installed["anthropic"] == "1.11.0"
    assert "anthropic" not in venv.editable


def test_the_lock_lines_read_no_package_provenance(monkeypatch, tmp_path):
    """Wiring: nothing about how a locked package is installed reaches the rule.

    Every provenance read but core's own (which locates the lock) raises, and
    the guard still produces the same lines.
    """
    venv = _venv(_checkout(tmp_path, lock=_lock_text()))
    use_fake_uv(monkeypatch, venv)
    expected = CoreInstallGuard.snapshot().install_constraints
    real = cli._direct_url_provenance

    def core_only(dist):
        if fr.canonical_package(dist) != CORE:
            raise AssertionError(f"provenance of {dist} was read")
        return real(dist)

    monkeypatch.setattr(cli, "_direct_url_provenance", core_only)

    assert CoreInstallGuard.snapshot().install_constraints == expected
    assert "anthropic===0.117.0" in expected
    assert f"{SDK}===0.36.0" in expected


def test_the_lines_do_not_depend_on_the_install_that_carries_them(
    monkeypatch, fake_registry, tmp_path,
):
    """Not from the install's arguments: one set of lines for every install.

    A plain request, an upgrade, a link, a named and a bare git URL all carry
    the same lock lines. The previous rule parsed these arguments and each
    spelling it missed left a locked package free (#3502).
    """
    venv = _venv(_checkout(tmp_path, lock=_lock_text(voice="0.4.0")))
    use_fake_uv(monkeypatch, venv)
    guard = CoreInstallGuard.snapshot()
    git_url = f"git+{fake_registry['voice'].git}"

    for pip_args in (
        [VOICE],
        ["--upgrade", VOICE],
        ["-e", "/src/kestrel-feature-voice"],
        [f"{VOICE} @ {git_url}"],
        ["--upgrade", git_url],
    ):
        guard.run(pip_args)

    lock_lines = [line for line in guard.install_constraints if "===" in line]
    assert f"{VOICE}===0.4.0" in lock_lines
    for constraints in venv.constraint_files:
        assert _lines(constraints) == guard.install_constraints


@pytest.mark.parametrize("upgrade", [False, True], ids=["install", "upgrade"])
def test_an_undeclared_linked_package_off_the_lock_is_put_back_to_it(
    monkeypatch, reconcile_host, upgrade,
):
    """The accepted cost: a hand-linked SDK at another version is pinned back.

    No manifest declares the SDK's checkout, so the lock's version holds it,
    and the install that resolves it replaces the link with that version
    rather than leave it off the lock. Declaring the checkout is how an
    operator keeps it (see the declared-checkout tests).
    """
    venv = _venv(_checkout(reconcile_host, lock=_lock_text()))
    venv.installed[SDK] = "0.36.5"
    venv.installed_requires[VOICE].append(f"{SDK}>=0.36")
    venv.package_index[SDK] = ["0.36.0", "0.37.0"]
    use_fake_uv(monkeypatch, venv)
    guard = CoreInstallGuard.snapshot()

    result = guard.run(["--upgrade", VOICE] if upgrade else [VOICE])

    assert result.returncode == 0
    assert f"{SDK}===0.36.0" in _lines(venv.constraint_files[0])
    assert venv.installed[SDK] == "0.36.0"
    assert SDK not in venv.editable


# --- a checkout the manifest declares --------------------------------------


def _sdk_declared(tmp_path, *, sdk_lock="0.35.0", voice_requires_sdk=">=0.36"):
    """Voice depends on the SDK; the manifest declares the SDK's checkout.

    The lock pins the SDK at *sdk_lock*, the checkout builds 0.36.0, and the
    index publishes 0.35.0, 0.37.0 and 0.40.0, so an install that resolves
    the SDK from the index has somewhere to go.
    """
    lock = _lock_text().replace(
        f'name = "{SDK}"\nversion = "0.36.0"', f'name = "{SDK}"\nversion = "{sdk_lock}"',
    )
    venv = _venv(_checkout(tmp_path, lock=lock))
    venv.installed_requires[VOICE].append(f"{SDK}{voice_requires_sdk}")
    venv.package_index[SDK] = ["0.35.0", "0.37.0", "0.40.0"]
    manifest = _voice_manifest(
        tmp_path, f'[[feature]]\nname = "{SDK}"\neditable = "{SDK_CHECKOUT}"\n',
    )
    return venv, manifest


def test_a_declared_checkout_is_held_on_it_and_not_pinned(
    monkeypatch, fake_registry, tmp_path,
):
    """The manifest's editable entry is the SDK's declared source.

    Its line holds it on that checkout. The locked version is not pinned: the
    operator chose to run what the checkout builds (0.36.0, against a lock of
    0.35.0), and a pin would replace the link. Everything else stays pinned.
    """
    venv, manifest = _sdk_declared(tmp_path)
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(manifest))

    assert rc == 0
    (constraints,) = venv.constraint_files  # the SDK entry was already present
    lines = _lines(constraints)
    assert _held(lines, SDK, SDK_CHECKOUT)
    assert not _pinned(lines, SDK)
    assert "anthropic===0.117.0" in lines
    assert venv.editable[SDK] == SDK_CHECKOUT
    assert venv.installed[SDK] == "0.36.0"


def test_a_declared_checkout_is_never_resolved_from_the_index(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """Pressure on the held checkout: a feature needs an SDK it does not build.

    The install is refused, and the SDK stays linked at what its checkout
    builds. Leaving it unconstrained let the resolver replace the link with an
    index wheel (the control below).
    """
    venv, manifest = _sdk_declared(tmp_path, voice_requires_sdk=">=0.40")
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(manifest))

    assert rc == 1
    assert _held(_lines(venv.constraint_files[0]), SDK, SDK_CHECKOUT)
    assert venv.editable[SDK] == SDK_CHECKOUT
    assert venv.installed[SDK] == "0.36.0"
    assert VOICE not in venv.installed
    assert "core's uv.lock" in capsys.readouterr().out


def test_without_the_hold_pressure_replaces_the_declared_checkout(
    monkeypatch, fake_registry, tmp_path,
):
    """Control: with no lock, nothing holds the SDK and the index wins."""
    venv, manifest = _sdk_declared(tmp_path, voice_requires_sdk=">=0.40")
    (Path(tmp_path) / "kestrel-sovereign" / "uv.lock").unlink()
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(manifest))

    assert rc == 0
    assert venv.installed[SDK] == "0.40.0"
    assert SDK not in venv.editable


def test_reconcile_upgrade_keeps_a_declared_checkout_linked(
    monkeypatch, reconcile_host,
):
    """uv's eager ``--upgrade`` would take the newest index SDK; the hold keeps the link."""
    venv, manifest = _sdk_declared(reconcile_host)
    venv.installed[VOICE] = "0.3.1"
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: manifest)
    use_fake_uv(monkeypatch, venv)

    rc = cli_lifecycle._run_feature_reconcile(
        reconcile_host, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=None,
    )

    assert rc == 0
    (command,) = venv.commands
    assert "--upgrade" in command
    assert _held(_lines(venv.constraint_files[0]), SDK, SDK_CHECKOUT)
    assert venv.editable[SDK] == SDK_CHECKOUT
    assert venv.installed[SDK] == "0.36.0"


def test_the_install_that_links_a_declared_checkout_is_held_to_it_alone(
    monkeypatch, fake_registry, tmp_path,
):
    """The explicit direct-source install: ``-e <declared checkout>``.

    Voice is locked at 0.4.0 and installed from the index; the manifest now
    declares its checkout, which builds 0.5.0. The link carries a hold on that
    checkout rather than the locked version, so it lands, and every other
    locked package is still pinned.
    """
    checkout = "/src/voice-new"
    venv = _venv(
        _checkout(tmp_path, lock=_lock_text(voice="0.4.0")),
        dependency_checkouts={checkout: (VOICE, "0.5.0")},
    )
    venv.installed[VOICE] = "0.4.0"
    manifest = tmp_path / "m.toml"
    manifest.write_text(f'[[feature]]\nname = "voice"\neditable = "{checkout}"\n')
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(manifest))

    assert rc == 0
    (command,) = venv.commands
    assert command[-2:] == ["-e", checkout]
    lines = _lines(venv.constraint_files[0])
    assert _held(lines, VOICE, checkout)
    assert not _pinned(lines, VOICE)
    assert "anthropic===0.117.0" in lines
    assert f"{SDK}===0.36.0" in lines
    assert venv.editable[VOICE] == checkout
    assert venv.installed[VOICE] == "0.5.0"


def test_an_undetermined_package_with_a_declared_checkout_is_no_reason_to_refuse(
    monkeypatch, fake_registry, tmp_path,
):
    """The hold on a checkout needs no version, so the lock naming none is no gap.

    Without the declaration the same lock refuses every install (the
    undetermined tests above): the SDK would have no line at all.
    """
    venv = _venv(_checkout(tmp_path, lock=_undetermined_lock_text(SDK)))
    manifest = _voice_manifest(
        tmp_path, f'[[feature]]\nname = "{SDK}"\neditable = "{SDK_CHECKOUT}"\n',
    )
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(manifest))

    assert rc == 0
    lines = _lines(venv.constraint_files[0])
    assert "anthropic===0.117.0" in lines
    assert _held(lines, SDK, SDK_CHECKOUT)


def test_an_undetermined_linked_package_without_a_declaration_refuses(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """Linked is not declared: the SDK has no line it could be held by."""
    venv = _venv(_checkout(tmp_path, lock=_undetermined_lock_text(SDK)))
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    assert rc == 1
    assert venv.commands == []
    assert f"no single version for this environment of: {SDK}" in capsys.readouterr().out


def test_pip_refuses_an_install_whose_lock_covers_a_declared_checkout(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """pip refuses an editable constraint, so it cannot carry the hold.

    The install is refused rather than run with the SDK free to leave its
    checkout, and the operator is told why and what to do.
    """
    venv, manifest = _sdk_declared(tmp_path)
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr("shutil.which", lambda name: None)

    rc = cli.cmd_feature_sync(_sync_args(manifest))

    out = capsys.readouterr().out
    assert rc == 1
    assert venv.commands == []
    assert f"declares {SDK} editable" in out
    assert "Install uv" in out


def test_pip_carries_the_lock_pins_when_no_checkout_is_declared(
    monkeypatch, fake_registry, tmp_path,
):
    """Only a hold on a checkout needs uv: version pins ride on pip too."""
    venv = _venv(_checkout(tmp_path, lock=_lock_text()))
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr("shutil.which", lambda name: None)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    assert rc == 0
    (command,) = venv.commands
    assert command[1:4] == ["-m", "pip", "install"]
    assert "anthropic===0.117.0" in _lines(venv.constraint_files[0])


# --- a package the manifest declares from the index --------------------------


def _linked_voice(tmp_path, linked: str) -> FakeUv:
    """Voice linked at 0.5.0, *linked* as ``editable`` or a ``direct_url``.

    The lock pins voice at 0.4.0 and the index publishes both, so an install
    free to move voice lands 0.5.0, and one held to the lock lands 0.4.0.
    """
    venv = _venv(_checkout(tmp_path, lock=_lock_text(voice="0.4.0")))
    venv.installed[VOICE] = "0.5.0"
    venv.package_index[VOICE] = ["0.4.0", "0.5.0"]
    if linked == "editable":
        venv.editable[VOICE] = "/src/kestrel-feature-voice"
    else:
        venv.direct_urls[VOICE] = "git+https://example.invalid/voice.git@abc"
    return venv


@pytest.mark.parametrize("linked", ["editable", "direct_url"])
def test_a_linked_package_the_manifest_moves_to_pypi_lands_the_locked_version(
    monkeypatch, fake_registry, tmp_path, linked,
):
    """A switch to the index is held to the lock, linked or not.

    A rule that left voice free because it was linked when the guard looked
    let the install that REPLACED the link take any version in the manifest's
    window (#3502).
    """
    venv = _linked_voice(tmp_path, linked)
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    assert rc == 0
    (constraints,) = venv.constraint_files
    assert f"{VOICE}===0.4.0" in _lines(constraints)
    assert venv.installed[VOICE] == "0.4.0"
    assert VOICE not in venv.editable
    assert VOICE not in venv.direct_urls


@pytest.mark.parametrize("linked", ["editable", "direct_url"])
def test_a_package_the_manifest_declares_from_pypi_is_pinned_while_linked(
    monkeypatch, tmp_path, linked,
):
    """``pypi`` is a declared index source: the window and the lock both bind."""
    venv = _linked_voice(tmp_path, linked)
    use_fake_uv(monkeypatch, venv)

    guard = CoreInstallGuard.snapshot(
        {VOICE: fr.SourceEntry(package=VOICE, pypi=">=0.4")}
    )

    lines = guard.install_constraints
    assert f"{VOICE}>=0.4" in lines
    assert f"{VOICE}===0.4.0" in lines


def test_reconcile_moving_a_linked_package_to_pypi_lands_the_locked_version(
    monkeypatch, reconcile_host,
):
    """``kestrel update``'s reconcile performs the same switch, with ``--upgrade``."""
    venv = _linked_voice(reconcile_host, "editable")
    manifest = _voice_manifest(reconcile_host)
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: manifest)
    use_fake_uv(monkeypatch, venv)

    rc = cli_lifecycle._run_feature_reconcile(
        reconcile_host, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=None,
    )

    assert rc == 0
    (command,) = venv.commands
    assert "--upgrade" in command
    assert f"{VOICE}===0.4.0" in _lines(venv.constraint_files[0])
    assert venv.installed[VOICE] == "0.4.0"
    assert VOICE not in venv.editable


# --- `kestrel update --prefer-*`: the plan and the lock read one declaration --


VOICE_CHECKOUT = "/src/kestrel-feature-voice"


def _reconcile(host, prefer) -> int:
    return cli_lifecycle._run_feature_reconcile(
        host, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=prefer,
    )


def _record_guards(monkeypatch) -> list:
    """Every guard ``CoreInstallGuard.snapshot`` builds, as built."""
    guards = []
    snapshot = CoreInstallGuard.snapshot.__func__

    def recording(cls, source_index=None):
        guards.append(snapshot(cls, source_index))
        return guards[-1]

    monkeypatch.setattr(CoreInstallGuard, "snapshot", classmethod(recording))
    return guards


def test_reconcile_prefer_pypi_holds_an_editable_entry_to_the_lock_not_its_checkout(
    monkeypatch, reconcile_host,
):
    """``--prefer-pypi`` moves voice off the checkout the manifest declares.

    The plan reinstalls voice from the index, so the lock pins it at 0.4.0 and
    nothing holds it on the checkout the preference replaced. With the guard
    reading the manifest's own declaration, the plan asked for the index while
    the lock line held voice on its checkout, and nothing pinned the version
    the switch landed (#3502).
    """
    venv = _linked_voice(reconcile_host, "editable")
    manifest = reconcile_host / "m.toml"
    manifest.write_text(f'[[feature]]\nname = "voice"\neditable = "{VOICE_CHECKOUT}"\n')
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: manifest)
    use_fake_uv(monkeypatch, venv)

    rc = _reconcile(reconcile_host, "pypi")

    assert rc == 0
    (command,) = venv.commands
    assert "--upgrade" in command and command[-1] == VOICE
    lines = _lines(venv.constraint_files[0])
    assert f"{VOICE}===0.4.0" in lines
    assert not any(line.startswith(f"-e {VOICE} @ ") for line in lines)
    assert venv.installed[VOICE] == "0.4.0"
    assert VOICE not in venv.editable


def test_reconcile_without_a_preference_keeps_the_declared_checkout_held(
    monkeypatch, reconcile_host,
):
    """Control for the test above: the same manifest, no preference.

    The plan pulls the checkout and installs nothing, and the guard holds voice
    on it. So the hold the test above asserts is absent is really there to be
    dropped.
    """
    venv = _linked_voice(reconcile_host, "editable")
    manifest = reconcile_host / "m.toml"
    manifest.write_text(f'[[feature]]\nname = "voice"\neditable = "{VOICE_CHECKOUT}"\n')
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: manifest)
    guards = _record_guards(monkeypatch)
    use_fake_uv(monkeypatch, venv)  # stubs the checkout's git pull too

    rc = _reconcile(reconcile_host, None)

    assert rc == 0
    assert venv.commands == []
    (guard,) = guards
    lines = guard.install_constraints
    assert _held(lines, VOICE, VOICE_CHECKOUT)
    assert not _pinned(lines, VOICE)
    assert venv.editable[VOICE] == VOICE_CHECKOUT


def test_reconcile_prefer_source_holds_the_checkout_it_keeps(
    monkeypatch, reconcile_host,
):
    """``--prefer-source`` keeps voice on its linked checkout over a ``pypi`` entry.

    The plan only pulls that checkout, and every install the batch runs is
    held to the same choice: voice stays on the checkout, with neither its
    locked version nor the index window the preference replaced. Either line
    would let another install in the batch replace the checkout the plan kept.
    """
    venv = _linked_voice(reconcile_host, "editable")
    manifest = _voice_manifest(reconcile_host)
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: manifest)
    guards = _record_guards(monkeypatch)
    use_fake_uv(monkeypatch, venv)
    pulled = []
    monkeypatch.setattr(
        cli_lifecycle, "_editable_git_pull",
        lambda checkout, dirty: pulled.append(str(checkout)) or (0, ""),
    )

    rc = _reconcile(reconcile_host, "source")

    assert rc == 0
    assert pulled == [VOICE_CHECKOUT]
    assert venv.commands == []
    (guard,) = guards
    lines = guard.install_constraints
    assert _held(lines, VOICE, VOICE_CHECKOUT)
    assert not _pinned(lines, VOICE)
    assert f"{VOICE}>=0.4" not in lines
    assert "anthropic===0.117.0" in lines


@pytest.mark.parametrize(
    ("linked_at", "after", "still_linked"),
    [("0.36.0", "0.36.0", True), ("0.36.5", "0.36.0", False)],
    ids=["linked-at-the-lock", "linked-off-the-lock"],
)
def test_an_upgrade_cannot_move_a_linked_package_past_the_lock(
    monkeypatch, reconcile_host, linked_at, after, still_linked,
):
    """uv's ``--upgrade`` replaces a linked dependency with the newest index wheel.

    Measured on uv 0.9.22. Held, a link at the locked version stays, and one
    off it can move only to the lock.
    """
    venv = _venv(_checkout(reconcile_host, lock=_lock_text()))
    venv.installed[VOICE] = "0.3.1"
    venv.installed[SDK] = linked_at
    venv.installed_requires[VOICE].append(f"{SDK}>=0.36")
    venv.package_index[SDK] = ["0.36.0", "0.37.0"]
    use_fake_uv(monkeypatch, venv)

    rc = cli_lifecycle._run_feature_reconcile(
        reconcile_host, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=None,
    )

    assert rc == 0
    (command,) = venv.commands
    assert "--upgrade" in command
    assert f"{SDK}===0.36.0" in _lines(venv.constraint_files[0])
    assert venv.installed[SDK] == after
    assert (SDK in venv.editable) is still_linked


def test_without_the_pin_an_upgrade_replaces_the_link_with_the_newest(
    monkeypatch, reconcile_host,
):
    """Control for the test above: the fake moves a free link the way uv does."""
    venv = _venv(_checkout(reconcile_host, lock=None))
    venv.installed[VOICE] = "0.3.1"
    venv.installed_requires[VOICE].append(f"{SDK}>=0.36")
    venv.package_index[SDK] = ["0.36.0", "0.37.0"]
    use_fake_uv(monkeypatch, venv)

    rc = cli_lifecycle._run_feature_reconcile(
        reconcile_host, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=None,
    )

    assert rc == 0
    assert venv.installed[SDK] == "0.37.0"
    assert SDK not in venv.editable


# --- git fallbacks: what the production callers actually pass -------------


def _private_voice(tmp_path, *, git_builds, linked: bool) -> FakeUv:
    """Voice locked at 0.4.0, published by no index, its git URL building *git_builds*.

    A private feature package is why a git fallback runs at all: the index
    request fails, and the fallback installs from the registry's git URL.
    *linked*: voice is on disk from that git URL; otherwise from an index.
    No manifest declares voice's source, so the lock's version holds it on
    the fallback as on every other install.
    """
    venv = _venv(
        _checkout(tmp_path, lock=_lock_text(voice="0.4.0")),
        feature_on_index=False, feature_git_version=git_builds,
    )
    venv.installed[VOICE] = "0.4.0"
    if linked:
        venv.direct_urls[VOICE] = "git+https://example.invalid/voice.git@abc"
    return venv


def _upgrade_voice(monkeypatch, venv):
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr(
        cli, "_installed_extension_distributions",
        lambda: [{"dist": VOICE, "version": "0.4.0", "editable_path": None}],
    )
    return cli.cmd_feature_upgrade(argparse.Namespace(names=[], dry_run=False))


@pytest.mark.parametrize("linked", [True, False], ids=["from-git", "from-index"])
def test_feature_upgrade_bare_git_fallback_cannot_leave_the_lock(
    monkeypatch, fake_registry, tmp_path, linked,
):
    """``feature upgrade``'s own fallback, a bare ``--upgrade git+<url>``.

    Its checkout builds 0.5.0 against a lock of 0.4.0, so the fallback carries
    ``kestrel-feature-voice===0.4.0`` and is refused, however voice is on disk:
    a git HEAD the lock never named is not what CI tested.
    """
    venv = _private_voice(tmp_path, git_builds="0.5.0", linked=linked)

    rc = _upgrade_voice(monkeypatch, venv)

    assert rc == 1
    index_request, fallback = venv.commands
    assert index_request[-2:] == ["--upgrade", VOICE]
    assert fallback[-2:] == ["--upgrade", f"git+{fake_registry['voice'].git}"]
    assert f"{VOICE}===0.4.0" in _lines(venv.constraint_files[1])
    assert venv.installed[VOICE] == "0.4.0"


def test_feature_upgrade_git_fallback_lands_the_locked_version(
    monkeypatch, fake_registry, tmp_path,
):
    """The same fallback succeeds when git builds the version the lock pins."""
    venv = _private_voice(tmp_path, git_builds="0.4.0", linked=False)

    rc = _upgrade_voice(monkeypatch, venv)

    assert rc == 0
    assert f"{VOICE}===0.4.0" in _lines(venv.constraint_files[1])
    assert venv.direct_urls[VOICE] == f"git+{fake_registry['voice'].git}"


@pytest.mark.parametrize(
    ("git_builds", "rc_expected"), [("0.5.0", 1), ("0.4.0", 0)],
    ids=["off-the-lock", "at-the-lock"],
)
def test_reconcile_git_fallback_without_extras_is_held_to_the_lock(
    monkeypatch, reconcile_host, git_builds, rc_expected,
):
    """``kestrel update``'s reconcile fallback: no extras, so a bare git URL.

    Voice is allowlisted, named by no manifest, and catalogued with a git URL.
    Reconcile upgrades it, the index has no copy, and the fallback carries the
    lock's pin like the request before it.
    """
    venv = _private_voice(reconcile_host, git_builds=git_builds, linked=True)
    use_fake_uv(monkeypatch, venv)

    rc = cli_lifecycle._run_feature_reconcile(
        reconcile_host, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=None,
    )

    assert rc == rc_expected
    index_request, fallback = venv.commands
    assert index_request[-2:] == ["--upgrade", VOICE]
    assert fallback[-2:] == ["--upgrade", "git+https://example/voice.git"]
    assert f"{VOICE}===0.4.0" in _lines(venv.constraint_files[1])
    assert venv.installed[VOICE] == "0.4.0"


@pytest.fixture
def core_provenance_unknown(monkeypatch, tmp_path):
    """Core's own direct_url.json will not read; its checkout HAS a lock."""
    checkout = _checkout(tmp_path, lock=_lock_text())
    venv = _venv(checkout, unreadable_provenance={CORE})
    use_fake_uv(monkeypatch, venv)
    return venv


def test_unknown_core_provenance_refuses_feature_installs(
    core_provenance_unknown, fake_registry, capsys,
):
    """Which lock holds this core cannot be told, so nothing can be held to it.

    Before #3502's fix this read as "no lock", and installs ran unheld on
    exactly the host whose install was already damaged.
    """
    rc = cli.cmd_feature_install(argparse.Namespace(name="voice"))

    out = capsys.readouterr().out
    assert rc != 0
    assert core_provenance_unknown.commands == []
    assert "--reinstall-package kestrel-sovereign" in out


def test_doctor_warns_when_core_provenance_hides_the_lock(core_provenance_unknown):
    report = DoctorReport()

    _check_core_lock(report)

    (warning,) = report.warn
    assert "venv was not compared against core's uv.lock" in warning
    assert "nothing to compare" not in warning


# --- the automatic core restore after a feature install ---------------------


@pytest.fixture
def restore_dir(tmp_path, monkeypatch):
    """Where a restore's constraints file lands, so a test can find it."""
    directory = tmp_path / "constraints"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


def _restore_files(directory: Path) -> list:
    return list(directory.glob("kestrel-restore-constraints-*.txt"))


def _swapping_venv(core_checkout, **kw) -> FakeUv:
    """A feature install that drops editable core for an index wheel.

    ``honours_constraints=False`` is a path that bypasses core's pin (a
    feature's own build step), so the guard finds core off its checkout
    afterwards and restores it. The checkout requires a newer anthropic than
    the venv holds (it was pulled ahead of the venv), so the restore has to
    move anthropic, and the index publishes 1.11.0 beside the 0.118.0 the
    checkout's lock pins.
    """
    venv = FakeUv(core_checkout=core_checkout, honours_constraints=False, **kw)
    venv.installed["anthropic"] = "0.117.0"
    venv.package_index["anthropic"] = ["0.117.0", "0.118.0", "1.11.0"]
    venv.installed_requires[CORE] = ["anthropic>=0.118"]
    return venv


def _core_restores(venv: FakeUv, checkout: str) -> list:
    """The index of every install that targeted core's checkout."""
    return [
        index for index, command in enumerate(venv.commands)
        if "-e" in command and command[-1] == checkout
    ]


@pytest.mark.parametrize(
    ("lock", "anthropic_after"),
    [(_lock_text("0.118.0"), "0.118.0"), (None, "1.11.0")],
    ids=["locked", "no-lock-control"],
)
def test_a_failed_feature_install_restores_core_held_to_the_checkouts_lock(
    monkeypatch, fake_registry, tmp_path, lock, anthropic_after,
):
    """The restore resolves core's dependencies, so the lock has to hold it too.

    The feature install was held to the lock, but core moved anyway and the
    install failed. The repair puts core back from its checkout. Without the
    lock's pins it resolves anthropic to whatever the index has, which is the
    drift the feature install was just prevented from causing. The control
    shows the fake really moves anthropic when nothing holds it.
    """
    checkout = _checkout(tmp_path, lock=lock)
    venv = _swapping_venv(checkout, feature_install_fails=True)
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_install(argparse.Namespace(name="voice"))

    assert rc != 0
    assert venv.editable[CORE] == checkout  # core was put back
    (restore,) = _core_restores(venv, checkout)
    assert venv.installed["anthropic"] == anthropic_after
    if lock is None:
        assert "-c" not in venv.commands[restore]
        return
    # The restore read the lock beside the checkout it restored, through the
    # same constraints-file plumbing every install uses.
    assert "-c" in venv.commands[restore]
    restore_lines = _lines(venv.constraint_files[-1])
    assert "anthropic===0.118.0" in restore_lines
    # Core's own pin is absent: this install exists to put core back. The SDK
    # is pinned like any locked package no manifest declares a checkout for.
    assert not any(line.startswith(f"{CORE}==") for line in restore_lines)
    assert f"{SDK}===0.36.0" in restore_lines


def test_a_restore_that_fails_prints_a_command_held_to_the_lock(
    monkeypatch, fake_registry, tmp_path, restore_dir, capsys,
):
    """The printed recovery command is the bounded install, not an unbounded one.

    ``-c`` has no inline form, so a command naming no file is an install free
    to move what the lock pins the moment an operator pastes it.
    """
    checkout = _checkout(tmp_path, lock=_lock_text("0.118.0"))
    venv = _swapping_venv(checkout, feature_install_fails=True, repair_fails=True)
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_install(argparse.Namespace(name="voice"))

    err = capsys.readouterr().err
    assert rc == CORE_UNSAFE
    assert "RESTORE FAILED" in err
    (published,) = _restore_files(restore_dir)
    assert f"-c {shlex.quote(str(published))}" in err
    assert "anthropic===0.118.0" in _lines(published.read_text(encoding="utf-8"))
    # The file the operator is pointed at is the one the attempt used.
    (restore,) = _core_restores(venv, checkout)
    command = venv.commands[restore]
    assert Path(command[command.index("-c") + 1]) == published
    # ...and the operator is told that it is the lock that bounds it.
    assert f"core's uv.lock ({Path(checkout) / 'uv.lock'})" in err


def test_an_interrupted_feature_install_prints_a_restore_held_to_the_lock(
    monkeypatch, fake_registry, tmp_path, restore_dir, capsys,
):
    """Ctrl-C never repairs, so the printed command is the only restore that runs."""
    checkout = _checkout(tmp_path, lock=_lock_text("0.118.0"))
    venv = _swapping_venv(checkout, feature_install_interrupted=True)
    use_fake_uv(monkeypatch, venv)

    with pytest.raises(KeyboardInterrupt):
        cli.cmd_feature_install(argparse.Namespace(name="voice"))

    err = capsys.readouterr().err
    assert "NOT RESTORED" in err
    (published,) = _restore_files(restore_dir)
    assert f"-c {shlex.quote(str(published))}" in err
    assert "anthropic===0.118.0" in _lines(published.read_text(encoding="utf-8"))
    assert _core_restores(venv, checkout) == []  # reported, never repaired


def _unreadable_restore_guard(tmp_path, kind="unparseable"):
    """Core live at a checkout with a good lock; declared at one whose lock is broken."""
    live = _checkout(tmp_path, "live", lock=_lock_text())
    declared = _checkout(tmp_path, "declared")
    _unusable_lock(declared, kind)
    venv = FakeUv(core_checkout=live, checkouts={declared: "0.53.0"})
    return venv, declared


@pytest.mark.parametrize("kind", ["unparseable", "directory", "undetermined"])
def test_a_restore_whose_lock_will_not_read_is_refused_not_attempted(
    monkeypatch, tmp_path, restore_dir, kind,
):
    """The same rule as a feature install: no lock to hold it, no install."""
    venv, declared = _unreadable_restore_guard(tmp_path, kind)
    use_fake_uv(monkeypatch, venv)
    guard = CoreInstallGuard.snapshot({
        CORE: fr.SourceEntry(package=CORE, editable=declared),
    })

    outcome = guard.resolve()

    assert outcome.drift is not None  # the drift is still named
    assert outcome.repaired is False
    assert outcome.attempted is False
    assert venv.commands == []  # asserted on the venv, not a flag
    assert _restore_files(restore_dir) == []  # nothing for a command that does not exist
    assert outcome.command == ""
    instruction = outcome.restore_instruction
    assert instruction.startswith("NOT RESTORED")
    assert "could not hold this install" in instruction
    assert "pip install" not in instruction
    # Not the "nothing declares where core belongs" sentence: something does.
    assert "no declared source" not in instruction
    if kind == "undetermined":
        assert "no single version for this environment of: anthropic" in instruction


def test_an_interrupt_names_a_restore_lock_that_will_not_read(
    monkeypatch, tmp_path, restore_dir, capsys,
):
    """The interrupt path fails closed on the same lock, rather than print an unheld command."""
    venv, declared = _unreadable_restore_guard(tmp_path)
    venv.feature_install_interrupted = True
    use_fake_uv(monkeypatch, venv)
    guard = CoreInstallGuard.snapshot({
        CORE: fr.SourceEntry(package=CORE, editable=declared),
    })

    with pytest.raises(KeyboardInterrupt):
        guard.run([f"{VOICE}>=0.4"])

    err = capsys.readouterr().err
    assert "INTERRUPTED" in err
    assert "NOT RESTORED" in err
    assert "could not hold this install" in err
    assert "pip install" not in err
    assert _restore_files(restore_dir) == []


# --- kestrel update's reconcile: the eager --upgrade --------------------------


@pytest.fixture
def reconcile_host(monkeypatch, tmp_path):
    """One agent allowlisting VoiceFeature, voice installed at 0.3.1 from the index."""
    ma = MultiAgentConfig(agents={
        "emma": LocalAgentConfig(
            data_dir="agent_data/emma", port=8801, features=["VoiceFeature"],
        ),
    })
    monkeypatch.setattr(cli.MultiAgentConfig, "load", classmethod(lambda c, p, **k: ma))
    monkeypatch.setattr(
        "kestrel_sovereign.feature_registry.load_registry",
        lambda *a, **k: {"voice": FeaturePackageInfo(
            name="voice", package=VOICE, git="https://example/voice.git",
            features=["VoiceFeature"], description="", core=False,
        )},
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.discover_entrypoint_feature_dists",
        lambda: {"VoiceFeature": VOICE},
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.discover_local_feature_class_names", lambda: set(),
    )
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: tmp_path / "missing.toml")
    return tmp_path


@pytest.mark.parametrize(
    ("lock", "anthropic_after"),
    [(_lock_text(), "0.117.0"), (None, "1.11.0")],
    ids=["locked", "no-lock-control"],
)
def test_reconcile_upgrade_cannot_drag_a_locked_package_off_the_lock(
    monkeypatch, reconcile_host, lock, anthropic_after,
):
    """The reported failure: `--upgrade` moved anthropic past the lock.

    The control shows the fake resolver really does move it when nothing holds
    it, so the locked case is not passing vacuously.
    """
    venv = _venv(_checkout(reconcile_host, lock=lock))
    venv.installed[VOICE] = "0.3.1"
    use_fake_uv(monkeypatch, venv)

    rc = cli_lifecycle._run_feature_reconcile(
        reconcile_host, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=None,
    )

    assert rc == 0
    assert "--upgrade" in venv.commands[0]
    assert venv.installed[VOICE] == "0.4.0"
    assert venv.installed["anthropic"] == anthropic_after


def test_reconcile_names_the_lock_when_it_refuses_an_install(
    monkeypatch, reconcile_host, capsys,
):
    venv = _venv(
        _checkout(reconcile_host, lock=_lock_text()), voice_requires=("anthropic>=1",),
    )
    venv.installed[VOICE] = "0.3.1"
    use_fake_uv(monkeypatch, venv)

    rc = cli_lifecycle._run_feature_reconcile(
        reconcile_host, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=None,
    )

    assert rc == 1
    assert venv.installed["anthropic"] == "0.117.0"
    assert "core's uv.lock" in capsys.readouterr().out


# --- feature install / upgrade ------------------------------------------------


def test_feature_install_is_held_to_the_lock(monkeypatch, fake_registry, tmp_path):
    venv = _venv(_checkout(tmp_path, lock=_lock_text()))
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_install(argparse.Namespace(name="voice"))

    assert rc == 0
    assert "anthropic===0.117.0" in _lines(venv.constraint_files[0])


@pytest.mark.parametrize("surface", ["install", "upgrade"])
def test_feature_install_and_upgrade_name_the_lock_when_it_refuses(
    monkeypatch, fake_registry, tmp_path, capsys, surface,
):
    venv = _venv(
        _checkout(tmp_path, lock=_lock_text()), voice_requires=("anthropic>=1",),
    )
    use_fake_uv(monkeypatch, venv)
    if surface == "install":
        rc = cli.cmd_feature_install(argparse.Namespace(name="voice"))
    else:
        venv.installed[VOICE] = "0.3.1"
        monkeypatch.setattr(
            cli, "_installed_extension_distributions",
            lambda: [{"dist": VOICE, "version": "0.3.1", "editable_path": None}],
        )
        rc = cli.cmd_feature_upgrade(argparse.Namespace(names=[], dry_run=False))

    assert rc == 1
    assert venv.installed["anthropic"] == "0.117.0"
    assert "core's uv.lock" in capsys.readouterr().out


def test_feature_upgrade_cannot_move_a_locked_package(
    monkeypatch, fake_registry, tmp_path,
):
    venv = _venv(_checkout(tmp_path, lock=_lock_text()))
    venv.installed[VOICE] = "0.3.1"
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr(
        cli, "_installed_extension_distributions",
        lambda: [{"dist": VOICE, "version": "0.3.1", "editable_path": None}],
    )

    rc = cli.cmd_feature_upgrade(argparse.Namespace(names=[], dry_run=False))

    assert rc == 0
    assert venv.installed[VOICE] == "0.4.0"
    assert venv.installed["anthropic"] == "0.117.0"


def _run_surface(monkeypatch, venv, surface) -> int:
    """``feature install voice`` or ``feature upgrade`` (voice at 0.3.1)."""
    if surface == "install":
        return cli.cmd_feature_install(argparse.Namespace(name="voice"))
    venv.installed[VOICE] = "0.3.1"
    monkeypatch.setattr(
        cli, "_installed_extension_distributions",
        lambda: [{"dist": VOICE, "version": "0.3.1", "editable_path": None}],
    )
    return cli.cmd_feature_upgrade(argparse.Namespace(names=[], dry_run=False))


def _host_manifest(tmp_path, text: str) -> None:
    (tmp_path / DEFAULT_HOST_MANIFEST).write_text(text, encoding="utf-8")


@pytest.mark.parametrize("surface", ["install", "upgrade"])
@pytest.mark.parametrize("declared", [True, False], ids=["declared", "undeclared"])
def test_install_and_upgrade_hold_a_checkout_the_host_manifest_declares(
    monkeypatch, fake_registry, tmp_path, surface, declared,
):
    """The commands that take no manifest still honour its declared checkouts.

    The host manifest declares the SDK's checkout, which builds 0.36.0 against
    a lock of 0.35.0. ``feature install`` and ``feature upgrade`` hold the SDK
    on it, as ``feature sync`` would. Without the declaration the lock's
    version pins it, and the link is put back to 0.35.0.
    """
    venv, _ = _sdk_declared(tmp_path, voice_requires_sdk=">=0.35")
    if declared:
        _host_manifest(
            tmp_path, f'[[feature]]\nname = "{SDK}"\neditable = "{SDK_CHECKOUT}"\n',
        )
    use_fake_uv(monkeypatch, venv)

    rc = _run_surface(monkeypatch, venv, surface)

    assert rc == 0
    lines = _lines(venv.constraint_files[0])
    assert "anthropic===0.117.0" in lines
    if declared:
        assert _held(lines, SDK, SDK_CHECKOUT)
        assert not _pinned(lines, SDK)
        assert venv.editable[SDK] == SDK_CHECKOUT
        assert venv.installed[SDK] == "0.36.0"
    else:
        assert f"{SDK}===0.35.0" in lines
        assert venv.installed[SDK] == "0.35.0"
        assert SDK not in venv.editable


@pytest.mark.parametrize("surface", ["install", "upgrade"])
def test_an_unreadable_host_manifest_refuses_an_install_held_to_a_lock(
    monkeypatch, fake_registry, tmp_path, capsys, surface,
):
    """Which checkouts it declares is unknown, and "none" could replace one."""
    venv = _venv(_checkout(tmp_path, lock=_lock_text()))
    _host_manifest(tmp_path, "this is = not [toml")
    use_fake_uv(monkeypatch, venv)

    rc = _run_surface(monkeypatch, venv, surface)

    out = capsys.readouterr().out
    assert rc == 1
    assert venv.commands == []
    assert f"host manifest {tmp_path / DEFAULT_HOST_MANIFEST} would not read" in out


def test_without_a_lock_the_host_manifest_is_never_read(
    monkeypatch, fake_registry, tmp_path,
):
    """No lock, nothing for the declarations to decide: installs run as before."""
    venv = _venv(_checkout(tmp_path, lock=None))
    _host_manifest(tmp_path, "this is = not [toml")
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_install(argparse.Namespace(name="voice"))

    assert rc == 0
    assert venv.installed[VOICE] == "0.4.0"


# --- where the host manifest is: the project directory, never the cwd -------


def test_the_default_host_manifest_is_the_project_directorys():
    """``KESTREL_HOME`` names the project directory; the cwd is elsewhere.

    The manifest a command reads by default is the project directory's, the
    one the host reads its feature enablement from. An explicit ``--manifest``
    stays exactly what the operator named.
    """
    home = Path(paths.project_dir())
    assert Path.cwd() != home

    assert cli._host_manifest_path(types.SimpleNamespace(manifest=None)) == (
        home / DEFAULT_HOST_MANIFEST
    )
    assert cli._host_manifest_path(types.SimpleNamespace(manifest="x/m.toml")) == (
        Path("x/m.toml")
    )


@pytest.mark.parametrize("surface", ["install", "upgrade"])
def test_the_host_manifest_is_the_project_directorys_not_the_cwds(
    monkeypatch, fake_registry, tmp_path, surface,
):
    """``KESTREL_HOME`` and the launch directory differ, as on a service host.

    The project directory's manifest declares the SDK's checkout; the cwd's
    declares another one. ``feature install`` / ``upgrade`` hold the SDK on
    the project directory's checkout, the one the host itself reads. Read from
    the cwd, the guard held the SDK on a checkout the host never declared, or,
    with no manifest there, pinned it and put the link back to the locked
    0.35.0 (#3502).
    """
    venv, _ = _sdk_declared(tmp_path, voice_requires_sdk=">=0.35")
    _host_manifest(
        tmp_path, f'[[feature]]\nname = "{SDK}"\neditable = "{SDK_CHECKOUT}"\n',
    )
    launch = Path.cwd()
    assert launch != tmp_path
    (launch / DEFAULT_HOST_MANIFEST).write_text(
        f'[[feature]]\nname = "{SDK}"\neditable = "/src/elsewhere"\n',
        encoding="utf-8",
    )
    use_fake_uv(monkeypatch, venv)

    rc = _run_surface(monkeypatch, venv, surface)

    assert rc == 0
    lines = _lines(venv.constraint_files[0])
    assert _held(lines, SDK, SDK_CHECKOUT)
    assert not _held(lines, SDK, "/src/elsewhere")
    assert not _pinned(lines, SDK)
    assert venv.editable[SDK] == SDK_CHECKOUT
    assert venv.installed[SDK] == "0.36.0"


def test_a_relative_checkout_in_the_host_manifest_is_resolved_beside_it(
    monkeypatch, fake_registry, tmp_path,
):
    """``editable = "sdk"`` names the checkout next to the manifest.

    The guard holds the SDK on the project directory's ``sdk``, not on an
    ``sdk`` beside whatever directory the command was launched from.
    """
    checkout = tmp_path / "sdk"
    venv = _venv(_checkout(tmp_path, lock=_lock_text()), sdk_checkout=str(checkout))
    _host_manifest(tmp_path, f'[[feature]]\nname = "{SDK}"\neditable = "sdk"\n')
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_install(argparse.Namespace(name="voice"))

    assert rc == 0
    lines = _lines(venv.constraint_files[0])
    assert _held(lines, SDK, str(checkout))
    assert not _held(lines, SDK, str(Path.cwd() / "sdk"))
    assert venv.editable[SDK] == str(checkout)


def test_the_manifest_loader_resolves_relative_checkouts_against_its_directory(
    monkeypatch, tmp_path,
):
    """Every reader gets the same absolute checkout, whatever its cwd."""
    home = tmp_path / "home"
    home.mkdir()
    (home / DEFAULT_HOST_MANIFEST).write_text(
        '[[feature]]\nname = "a"\neditable = "sdk"\n'
        '[[feature]]\nname = "b"\neditable = "../b"\n'
        '[[feature]]\nname = "c"\neditable = "/abs/c"\n'
        '[[feature]]\nname = "d"\neditable = "~/d"\n'
        '[[feature]]\nname = "e"\neditable = ""\n'
        '[[feature]]\nname = "f"\npypi = ">=1"\n'
        "[[feature]]\nname = \"g\"\neditable = 'C:\\src\\g'\n",
        encoding="utf-8",
    )
    expected = {
        "a": str(home / "sdk"),
        "b": str(tmp_path / "b"),
        "c": "/abs/c",
        "d": "~/d",
        "e": "",
        "f": None,
        # Drive-anchored: absolute on Windows, and relative to nothing here.
        "g": r"C:\src\g",
    }

    def editables(manifest):
        return {e["name"]: e["editable"] for e in cli._load_host_manifest(manifest)}

    assert editables(home / DEFAULT_HOST_MANIFEST) == expected
    # A relative --manifest is relative to the cwd, and its checkouts to it.
    monkeypatch.chdir(tmp_path)
    assert editables(Path("home") / DEFAULT_HOST_MANIFEST) == expected


@pytest.mark.parametrize("present", [False, True], ids=["absent", "present"])
def test_without_a_host_manifest_every_locked_package_is_pinned_and_the_note_says_so(
    monkeypatch, fake_registry, tmp_path, capsys, present,
):
    """No manifest declares no checkout: the lock pins everything it covers.

    That is the lock's own hold, and the refusal says which manifest the guard
    looked for, so the pins on a host that keeps linked checkouts are never
    unexplained.
    """
    venv = _venv(_checkout(tmp_path, lock=_lock_text()), voice_requires=("anthropic>=1",))
    if present:
        _host_manifest(tmp_path, '[[feature]]\nname = "voice"\npypi = ">=0.4"\n')
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_install(argparse.Namespace(name="voice"))

    out = capsys.readouterr().out
    assert rc == 1
    assert venv.installed["anthropic"] == "0.117.0"
    lines = _lines(venv.constraint_files[0])
    assert "anthropic===0.117.0" in lines
    assert f"{SDK}===0.36.0" in lines
    absent_note = f"No host manifest exists at {tmp_path / DEFAULT_HOST_MANIFEST}"
    assert (absent_note in out) is not present


@pytest.mark.parametrize(
    "unreachable",
    ["directory", "dangling_link", "unknown_home_user", "project_directory", "no_home"],
)
def test_a_host_manifest_that_cannot_be_reached_refuses_an_install_held_to_a_lock(
    monkeypatch, fake_registry, tmp_path, capsys, unreachable,
):
    """Absent is the one state that declares nothing; every other one refuses.

    A directory or a link to nowhere at the manifest's path, a checkout under
    a home directory this host does not have, or a project directory that
    cannot be resolved (unreadable, or no home for its ``~/.kestrel``
    fallback), leaves which packages are declared editable unknown. Reading
    that as "none" could replace a declared checkout from the index, so no
    install runs, and none of them is a traceback.
    """
    venv = _venv(_checkout(tmp_path, lock=_lock_text()))
    manifest = tmp_path / DEFAULT_HOST_MANIFEST
    if unreachable == "directory":
        manifest.mkdir()
    elif unreachable == "dangling_link":
        manifest.symlink_to(tmp_path / "nowhere.toml")
    elif unreachable == "unknown_home_user":
        _host_manifest(
            tmp_path,
            f'[[feature]]\nname = "{SDK}"\neditable = "~no-such-user-3502/sdk"\n',
        )
    else:
        failure = (
            PermissionError(13, "Permission denied", str(tmp_path))
            if unreachable == "project_directory"
            else RuntimeError("Could not determine home directory.")
        )

        def unresolvable():
            raise failure

        monkeypatch.setattr(cli, "_get_project_dir", unresolvable)
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_install(argparse.Namespace(name="voice"))

    out = capsys.readouterr().out
    assert rc == 1
    assert venv.commands == []
    assert "declares editable is unknown" in out


def test_a_guard_built_without_a_manifest_reads_the_host_manifests_declarations(
    monkeypatch, tmp_path,
):
    """Only ``snapshot`` was given a manifest; every other guard reads the host's.

    A guard whose declarations were never supplied must not treat "none" as
    fact when a lock binds it: that pins a declared checkout and puts the link
    back from the index. ``unguarded()`` binds a lock for a core install or
    restore, so it reads the project directory's manifest as ``snapshot()``
    does.
    """
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    _host_manifest(
        tmp_path, f'[[feature]]\nname = "{SDK}"\neditable = "{SDK_CHECKOUT}"\n',
    )
    checkout = Path(_checkout(tmp_path, lock=_lock_text()))

    lock, error, lines = CoreInstallGuard.unguarded()._bind_lock(
        lambda: cl.load_core_lock(checkout),
    )

    assert error is None
    assert _held(lines, SDK, SDK_CHECKOUT)
    assert not _pinned(lines, SDK)
    assert "anthropic===0.117.0" in lines


# --- the drift check: doctor and update report it, neither repairs it -------


@pytest.fixture
def drifted_host(monkeypatch, tmp_path):
    """Core editable at a checkout locking anthropic 0.117.0; 1.11.0 installed."""
    checkout = _checkout(tmp_path, lock=_lock_text())
    monkeypatch.setattr(
        cli, "_core_install_shape",
        lambda: fr.CoreInstallShape(
            version="0.52.0",
            provenance=fr.Provenance.direct(checkout, editable=True),
        ),
    )
    installed = {"anthropic": "1.11.0", SDK: "0.36.0"}
    monkeypatch.setattr(
        cl, "installed_versions",
        lambda names: {name: installed[name] for name in names if name in installed},
    )
    return installed


def test_doctor_warns_on_every_package_off_the_lock(drifted_host):
    report = DoctorReport()

    _check_core_lock(report)

    assert report.ready  # a warning: the venv still boots
    (warning,) = report.warn
    assert "venv differs from core's uv.lock" in warning
    assert "anthropic 1.11.0 installed, uv.lock pins 0.117.0" in warning


def test_kestrel_doctor_runs_the_lock_check(drifted_host, tmp_path):
    report = diagnose(tmp_path)

    assert any(
        "anthropic 1.11.0 installed, uv.lock pins 0.117.0" in line
        for line in report.warn
    ), report.warn


def test_doctor_reports_a_venv_on_the_lock(drifted_host):
    drifted_host["anthropic"] = "0.117.0"
    report = DoctorReport()

    _check_core_lock(report)

    assert report.warn == []
    assert any("venv matches core's uv.lock" in line for line in report.ok)


def test_update_reports_drift_before_the_restart_and_does_not_fail_on_it(
    drifted_host, monkeypatch, tmp_path, capsys,
):
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: None)
    seen_before_restart = []

    def restart(args):
        seen_before_restart.append(capsys.readouterr())
        return 0

    monkeypatch.setattr(cli, "cmd_restart", restart)

    rc = cli.cmd_update(argparse.Namespace(
        name=None, pull=False, install=False, features=False, restart=True,
        allow_dirty=False, no_deps=False, continue_on_error=False,
        dry_run=False, manifest=None, force=False, uv_sync=None,
    ))

    assert rc == 0  # reported, not enforced
    (before,) = seen_before_restart
    assert "lock: WARNING" in before.err
    assert "anthropic 1.11.0 installed, uv.lock pins 0.117.0" in before.err
    assert drifted_host["anthropic"] == "1.11.0"  # reported, never repaired


def test_update_leaves_a_locked_package_at_the_locked_version(
    monkeypatch, fake_registry, reconcile_host, capsys,
):
    """The issue's gate, on the fake venv: after `kestrel update` installs
    features, anthropic is the locked version and the lock check says so."""
    venv = _venv(_checkout(reconcile_host, lock=_lock_text()))
    venv.installed[VOICE] = "0.3.1"
    use_fake_uv(monkeypatch, venv)
    monkeypatch.setattr(
        cl, "installed_versions",
        lambda names: {
            name: venv.installed[name] for name in names if name in venv.installed
        },
    )
    monkeypatch.setattr(cli, "_get_project_dir", lambda: reconcile_host)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: None)
    manifest = _voice_manifest(reconcile_host)

    rc = cli.cmd_update(argparse.Namespace(
        name=None, pull=False, install=False, features=True, restart=False,
        allow_dirty=False, no_deps=False, continue_on_error=False,
        dry_run=False, manifest=str(manifest), force=False, uv_sync=None,
    ))

    assert rc == 0
    assert venv.installed["anthropic"] == "0.117.0"
    assert any("--upgrade" in command for command in venv.commands)
    assert "lock: venv matches core's uv.lock" in capsys.readouterr().out
