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

from kestrel_sovereign import cli, cli_lifecycle
from kestrel_sovereign import core_lock as cl
from kestrel_sovereign import feature_reconcile as fr
from kestrel_sovereign.cli_features import CORE_UNSAFE, CoreInstallGuard
from kestrel_sovereign.doctor import DoctorReport, _check_core_lock, diagnose
from kestrel_sovereign.feature_registry import FeaturePackageInfo
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from tests.utils.fake_uv import CORE, SDK, FakeUv, use_fake_uv

VOICE = "kestrel-feature-voice"


def _lock_text(anthropic="0.117.0") -> str:
    """A lock that pins anthropic, core, and core's SDK.

    The SDK is locked AND (in the fake venv) linked from a checkout, so it is
    the case a pin must leave alone.
    """
    return (
        "version = 1\nrevision = 3\n"
        '\n[[package]]\nname = "anthropic"\n'
        f'version = "{anthropic}"\nsource = {{ registry = "https://pypi.org/simple" }}\n'
        f'\n[[package]]\nname = "{CORE}"\nversion = "0.52.0"\nsource = {{ editable = "." }}\n'
        f'\n[[package]]\nname = "{SDK}"\nversion = "0.36.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )


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
    assert "anthropic==0.117.0" in lines
    # Core keeps its own pin, never the lock's line for it.
    assert f"{CORE}==0.52.0" in lines
    assert lines.count(f"{CORE}==0.52.0") == 1
    # The SDK is locked but linked from a checkout: a pin would replace it.
    assert not any(line.startswith(f"{SDK}==") for line in lines)


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
    assert "could not be used" in out
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


def test_an_undetermined_package_linked_from_a_checkout_is_no_reason_to_refuse(
    monkeypatch, fake_registry, tmp_path,
):
    """The SDK is linked, so it is left free whatever the lock says of it.

    A pin would replace that link, so it never carried one; the lock naming no
    version of it takes nothing out of the hold, and the rest stays pinned.
    """
    venv = _venv(_checkout(tmp_path, lock=_undetermined_lock_text(SDK)))
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    assert rc == 0
    lines = _lines(venv.constraint_files[0])
    assert "anthropic==0.117.0" in lines
    assert not any(line.startswith(f"{SDK}==") for line in lines)


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
    assert "anthropic==0.118.0" in core_lines
    assert "anthropic==0.117.0" not in core_lines
    assert venv.pins[0] is None
    # The feature after it is held to the lock beside the checkout core became.
    lines = _lines(venv.constraint_files[-1])
    assert "anthropic==0.118.0" in lines
    assert "anthropic==0.117.0" not in lines
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
    assert "could not be used" in out


def test_a_manifest_editable_entry_of_a_locked_package_is_left_unpinned(
    monkeypatch, tmp_path,
):
    venv = _venv(_checkout(tmp_path, lock=_lock_text()))
    use_fake_uv(monkeypatch, venv)

    guard = CoreInstallGuard.snapshot({
        "anthropic": fr.SourceEntry(package="anthropic", editable="/src/anthropic"),
    })

    assert not any(line.startswith("anthropic==") for line in guard.install_constraints)


def test_a_locked_package_whose_provenance_will_not_read_keeps_its_pin(
    monkeypatch, fake_registry, tmp_path, capsys,
):
    """Unknown provenance is not a deliberate link (#3502).

    Reading "the metadata would not read" as "linked from a checkout" took a
    damaged package out from under the lock, so a feature needing another
    version moved it. Unknown keeps the pin, and the install is refused.
    """
    venv = _venv(
        _checkout(tmp_path, lock=_lock_text()),
        voice_requires=("anthropic>=1",),
        unreadable_provenance={"anthropic"},
    )
    use_fake_uv(monkeypatch, venv)

    rc = cli.cmd_feature_sync(_sync_args(_voice_manifest(tmp_path)))

    assert rc == 1
    assert "anthropic==0.117.0" in _lines(venv.constraint_files[0])
    assert venv.installed["anthropic"] == "0.117.0"
    assert "core's uv.lock" in capsys.readouterr().out


def _dist_info(site: Path, name: str, direct_url=None) -> None:
    """``<name>-1.0.dist-info`` on *site*, with *direct_url* as its direct_url.json.

    *direct_url* is the file's text, ``b"..."`` for raw bytes, a ``Path`` to
    make it a directory (unreadable), or None for no file (an index install).
    """
    dist_info = site / f"{name.replace('-', '_')}-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8",
    )
    target = dist_info / "direct_url.json"
    if isinstance(direct_url, Path):
        target.mkdir()
    elif isinstance(direct_url, bytes):
        target.write_bytes(direct_url)
    elif direct_url is not None:
        target.write_text(direct_url, encoding="utf-8")


def test_real_provenance_reads_unknown_for_damaged_metadata_and_keeps_the_pin(
    monkeypatch, tmp_path,
):
    """The same rule through the real PEP 610 reader, on fixture metadata.

    Malformed, non-UTF-8 and unreadable ``direct_url.json`` are all unknown and
    keep their pins; a well-formed editable link is the one left out, and an
    index install (no file) is pinned as before.
    """
    site = tmp_path / "site"
    linked_checkout = tmp_path / "linked-checkout"
    linked_checkout.mkdir()
    damaged = {
        "lockfix-malformed": "{not json",
        "lockfix-not-utf8": b"\xff\xfe{}",
        "lockfix-unreadable": tmp_path / "is-a-directory",
    }
    for name, direct_url in damaged.items():
        _dist_info(site, name, direct_url)
    _dist_info(site, "lockfix-index")
    _dist_info(
        site, "lockfix-linked",
        f'{{"url": "{linked_checkout.as_uri()}", "dir_info": {{"editable": true}}}}',
    )
    monkeypatch.syspath_prepend(str(site))
    names = [*damaged, "lockfix-index", "lockfix-linked"]
    lock = "version = 1\nrevision = 3\n" + "".join(
        f'\n[[package]]\nname = "{name}"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        for name in names
    )
    checkout = _checkout(tmp_path, lock=lock)
    monkeypatch.setattr(
        cli, "_core_install_shape",
        lambda: fr.CoreInstallShape(
            version="0.52.0",
            provenance=fr.Provenance.direct(checkout, editable=True),
        ),
    )
    for name in damaged:
        assert not cli._direct_url_provenance(name).known, name

    lines = CoreInstallGuard.snapshot().install_constraints

    for name in [*damaged, "lockfix-index"]:
        assert f"{name}==1.0" in lines, name
    assert not any(line.startswith("lockfix-linked==") for line in lines)


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
    assert "anthropic==0.118.0" in restore_lines
    # Core's own pin is absent: this install exists to put core back. The SDK
    # is locked but linked from a checkout, so a pin would replace it.
    assert not any(line.startswith(f"{CORE}==") for line in restore_lines)
    assert not any(line.startswith(f"{SDK}==") for line in restore_lines)


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
    assert "anthropic==0.118.0" in _lines(published.read_text(encoding="utf-8"))
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
    assert "anthropic==0.118.0" in _lines(published.read_text(encoding="utf-8"))
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
    assert "could not be used" in instruction
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
    assert "could not be used" in err
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
    assert "anthropic==0.117.0" in _lines(venv.constraint_files[0])


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
