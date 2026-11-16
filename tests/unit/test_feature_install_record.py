"""``feature install`` records what it installs, so ``kestrel update`` keeps it (#4127).

Measured on a fresh Windows install: ``kestrel feature install
channel_telegram``, then ``kestrel update``. The update's ``uv sync`` removed
the package (core's lock does not name it), and ``feature sync`` restored
nothing, because the host manifest it restores from had never heard of it. The
agent came back without its channel, and nothing said so.

Two halves, pinned here against the shared fake venv:

* both install commands append the package to the host manifest, without
  rewriting what an operator wrote there;
* ``kestrel update`` refuses, before it changes anything, to uninstall an
  installed extension that nothing records (installed before the fix, or by
  hand), and prints the entry that records it.
"""

from __future__ import annotations

import argparse
import re
import tomllib
import types
from pathlib import Path

import pytest

from kestrel_sovereign import cli, cli_lifecycle, paths
from kestrel_sovereign.cli_features import (
    DEFAULT_HOST_MANIFEST,
    FeatureRecordError,
    UnrecordedExtension,
    record_feature_install,
    unrecorded_extensions,
)
from tests.utils.fake_uv import CORE, SDK, FakeUv, use_fake_uv

VOICE = "kestrel-feature-voice"
TELEGRAM = "kestrel-channel-telegram"


def _lock_text(*extra) -> str:
    """A lock pinning core and its SDK, and each ``(name, version)`` in *extra*."""
    text = (
        "version = 1\nrevision = 3\n"
        f'\n[[package]]\nname = "{CORE}"\nversion = "0.52.0"\nsource = {{ editable = "." }}\n'
        f'\n[[package]]\nname = "{SDK}"\nversion = "0.36.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    for name, version in extra:
        text += (
            f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
        )
    return text


@pytest.fixture(autouse=True)
def project(monkeypatch, tmp_path):
    """This test's own project directory, and a registry that knows voice.

    The cwd is a sibling: the manifest is the project directory's, never the
    cwd's (#3502).
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv(paths.HOME_ENV, str(home))
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    registry = {
        "voice": types.SimpleNamespace(
            package=VOICE,
            git="https://github.com/KestrelSovereignAI/kestrel-feature-voice.git",
            features=["VoiceFeature"],
            core=False,
        ),
    }
    monkeypatch.setattr("kestrel_sovereign.feature_registry.load_registry", lambda: registry)
    return home


@pytest.fixture
def venv(monkeypatch, tmp_path):
    """Editable core on a checkout whose lock names neither voice nor telegram.

    Its extensions are what the fake venv has installed of those two, as live
    entry-point discovery would report them.
    """
    checkout = tmp_path / "kestrel-sovereign"
    checkout.mkdir()
    (checkout / "uv.lock").write_text(_lock_text(), encoding="utf-8")
    venv = FakeUv(core_checkout=str(checkout), feature_requires=">=0.52")
    use_fake_uv(monkeypatch, venv)

    def extensions():
        return [
            {
                "dist": name, "version": venv.installed[name],
                "editable_path": venv.editable.get(name), "entries": [],
            }
            for name in (TELEGRAM, VOICE) if name in venv.installed
        ]

    monkeypatch.setattr(cli, "_installed_extension_distributions", extensions)
    return venv


def _manifest(project) -> Path:
    return project / DEFAULT_HOST_MANIFEST


def _declared(project) -> dict:
    return {
        entry["name"]: entry for entry in cli._load_host_manifest(_manifest(project))
    }


# --- recording an install -----------------------------------------------------


def test_an_install_on_a_host_with_no_manifest_creates_one(venv, project, capsys):
    assert cli.cmd_feature_install(argparse.Namespace(name="voice")) == 0

    assert venv.installed[VOICE] == "0.4.0"
    assert _declared(project) == {
        VOICE: {"name": VOICE, "editable": None, "pypi": "", "extras": []},
    }
    assert f"Recorded {VOICE} in {_manifest(project)}" in capsys.readouterr().out


OPERATOR_MANIFEST = (
    "# Pinned by hand: keep this comment.\n"
    "[host_features]\n"
    "default_enabled = false\n"
    "\n"
    "[[feature]]\n"
    'name = "arize-phoenix"\n'
    'pypi = ">=20,<21"  # trailing comment\n'
)


@pytest.mark.parametrize(
    "existing",
    [OPERATOR_MANIFEST, OPERATOR_MANIFEST.rstrip("\n"), OPERATOR_MANIFEST.replace("\n", "\r\n"), ""],
    ids=["trailing-newline", "no-trailing-newline", "crlf", "empty"],
)
def test_an_install_appends_and_keeps_every_byte_already_there(project, existing):
    manifest = _manifest(project)
    manifest.write_bytes(existing.encode("utf-8"))

    assert record_feature_install(VOICE) == manifest

    written = manifest.read_bytes().decode("utf-8")
    assert written.startswith(existing)
    entries = _declared(project)
    assert entries[VOICE]["pypi"] == ""
    if existing:
        assert entries["arize-phoenix"]["pypi"] == ">=20,<21"
        assert tomllib.loads(written)["host_features"] == {"default_enabled": False}


@pytest.mark.parametrize(
    "spelling",
    [VOICE, "voice", "Kestrel_Feature_Voice", f"{VOICE}[local]"],
    ids=["dist", "registry-name", "unnormalized", "inline-extras"],
)
def test_a_package_the_manifest_already_declares_is_left_alone(project, spelling):
    """Whatever source it names: the operator's declaration is theirs."""
    manifest = _manifest(project)
    text = f'[[feature]]\nname = "{spelling}"\neditable = "/src/voice"\n'
    manifest.write_text(text, encoding="utf-8")

    assert record_feature_install(VOICE) is None

    assert manifest.read_text(encoding="utf-8") == text


def test_a_git_fallback_install_is_recorded_without_a_source(venv, project, capsys):
    """The legacy entry is the one ``feature sync`` falls back to git for."""
    venv.feature_on_index = False

    assert cli.cmd_feature_install(argparse.Namespace(name="voice")) == 0

    assert "trying git" in capsys.readouterr().out
    assert _declared(project)[VOICE]["pypi"] is None


@pytest.mark.parametrize(
    "existing",
    ["this is = not [toml", 'feature = []\n', '[[feature]]\npypi = ""\n'],
    ids=["not-toml", "feature-is-not-an-array-of-tables", "entry-without-a-name"],
)
def test_a_manifest_that_cannot_take_the_entry_is_refused_unchanged(project, existing):
    manifest = _manifest(project)
    manifest.write_text(existing, encoding="utf-8")

    with pytest.raises(FeatureRecordError) as raised:
        record_feature_install(VOICE)

    assert manifest.read_text(encoding="utf-8") == existing
    described = raised.value.describe()
    assert str(manifest) in described
    assert f'[[feature]]\nname = "{VOICE}"\npypi = ""\n' in described


def test_an_install_that_cannot_be_recorded_fails_after_installing(
    venv, project, capsys,
):
    """A manifest every reader loads (it declares nothing) that no entry can join."""
    _manifest(project).write_text("feature = []\n", encoding="utf-8")

    assert cli.cmd_feature_install(argparse.Namespace(name="voice")) == 1

    assert venv.installed[VOICE] == "0.4.0"
    err = capsys.readouterr().err
    assert f"{VOICE} is installed, but recording it" in err
    assert "The next `kestrel update` will uninstall it" in err


def test_a_failed_install_records_nothing(venv, project):
    """Voice needs a core newer than the pinned checkout, from the index and git alike."""
    venv.feature_requires = ">=0.53"

    assert cli.cmd_feature_install(argparse.Namespace(name="voice")) == 1

    assert VOICE not in venv.installed
    assert not _manifest(project).exists()


# --- the journey: install, then update -----------------------------------------


@pytest.fixture
def update(venv, tmp_path, monkeypatch):
    """``kestrel update --no-pull`` on the fake venv, ``uv sync`` pruning as uv does.

    The sync removes every installed package the lock does not name. The
    feature sync is the real one, restoring from the host manifest through the
    fake installer; reconcile and the restart are recorded, not run.
    """
    checkout = tmp_path / "kestrel-sovereign"
    (checkout / ".git").mkdir()
    steps = []
    locked = {CORE, SDK}

    def uv_sync(_checkout):
        steps.append("uv sync")
        for name in [name for name in venv.installed if name not in locked]:
            del venv.installed[name]
            venv.editable.pop(name, None)
        return 0, ""

    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: checkout)
    monkeypatch.setattr(cli, "_git_head", lambda _: None)
    monkeypatch.setattr(cli, "_run_uv_sync", uv_sync)
    monkeypatch.setattr(cli_lifecycle, "_run_feature_reconcile", lambda *a, **kw: 0)
    monkeypatch.setattr(cli_lifecycle, "_report_core_lock", lambda: None)
    monkeypatch.setattr(cli, "cmd_restart", lambda args: steps.append("restart") or 0)
    return steps


def _update(*flags):
    return cli.cmd_update(cli.build_parser().parse_args(["update", "--no-pull", *flags]))


def test_a_feature_installed_with_feature_install_survives_an_update(
    venv, project, update,
):
    """The issue, end to end: the second command no longer undoes the first."""
    assert cli.cmd_feature_install(argparse.Namespace(name="voice")) == 0

    assert _update() == 0

    assert update == ["uv sync", "restart"]
    assert venv.installed[VOICE] == "0.4.0"


def test_an_update_refuses_to_uninstall_an_extension_nothing_records(
    venv, project, update, capsys,
):
    """Installed before ``feature install`` recorded installs: refused, untouched."""
    venv.installed[VOICE] = "0.4.0"

    assert _update() == 1

    assert update == []
    assert venv.installed[VOICE] == "0.4.0"
    err = capsys.readouterr().err
    assert "• extensions: REFUSED" in err
    assert f"    {VOICE} 0.4.0" in err
    assert "Nothing was pulled, installed or restarted" in err
    # No manifest yet: the one command that writes it is named.
    assert "kestrel feature sync --capture" in err


def test_the_entry_the_refusal_prints_is_the_fix(venv, project, update, capsys):
    """Append exactly what it printed, and the same update keeps the package."""
    venv.installed[VOICE] = "0.4.0"
    _manifest(project).write_text(OPERATOR_MANIFEST, encoding="utf-8")
    assert _update() == 1
    err = capsys.readouterr().err
    assert f"append these entries to {_manifest(project)}" in err
    printed = re.search(r"^(    \[\[feature\]\]\n(?:    \w+ = .*\n)+)", err, re.M)
    entry = "\n" + "\n".join(line[4:] for line in printed.group(1).splitlines()) + "\n"
    with open(_manifest(project), "a", encoding="utf-8") as handle:
        handle.write(entry)

    assert _update() == 0

    assert update == ["uv sync", "restart"]
    assert venv.installed[VOICE] == "0.4.0"


def test_continue_on_error_updates_anyway_and_says_what_it_removes(
    venv, project, update, capsys,
):
    venv.installed[VOICE] = "0.4.0"

    assert _update("--continue-on-error") == 0

    assert update == ["uv sync", "restart"]
    assert VOICE not in venv.installed
    err = capsys.readouterr().err
    assert "• extensions: WARNING" in err
    assert "the update will uninstall them" in err


def test_a_dry_run_says_the_update_would_refuse(venv, project, update, capsys):
    venv.installed[VOICE] = "0.4.0"

    assert _update("--dry-run") == 1

    assert update == []
    assert "• extensions: REFUSED" in capsys.readouterr().err


def test_an_install_that_does_not_prune_is_not_checked(venv, project, update, monkeypatch):
    """``uv pip install -e .`` removes nothing, so there is nothing to refuse."""
    venv.installed[VOICE] = "0.4.0"
    monkeypatch.setattr(
        cli, "_run_uv_pip_install_editable",
        lambda *a, **kw: update.append("pip install -e") or (0, ""),
    )

    assert _update("--no-uv-sync") == 0

    assert update == ["pip install -e", "restart"]


def test_an_unreadable_lock_is_not_read_as_an_empty_one(venv, project, update, tmp_path, capsys):
    """Every installed extension would look unrecorded; say the lock is the problem."""
    venv.installed[VOICE] = "0.4.0"
    (tmp_path / "kestrel-sovereign" / "uv.lock").write_text("not = [a lock", encoding="utf-8")

    assert _update() == 1

    assert update == []
    err = capsys.readouterr().err
    assert "core's uv.lock would not read" in err
    assert "REFUSED" not in err


# --- what survives an update ---------------------------------------------------


def _unrecorded(project, *, locked=(), provisioned=()):
    return [
        extension.package
        for extension in unrecorded_extensions(
            _manifest(project), locked=locked, provisioned=provisioned,
        )
    ]


def test_each_way_a_package_survives(venv, project):
    venv.installed[VOICE] = "0.4.0"
    venv.installed[TELEGRAM] = "0.1.10"

    assert _unrecorded(project) == [TELEGRAM, VOICE]
    assert _unrecorded(project, locked=[VOICE]) == [TELEGRAM]
    assert _unrecorded(project, provisioned=[TELEGRAM]) == [VOICE]
    _manifest(project).write_text(
        f'[[feature]]\nname = "{TELEGRAM}[bot]"\npypi = ">=0.1"\n', encoding="utf-8",
    )
    assert _unrecorded(project) == [VOICE]


def test_a_package_a_survivor_requires_comes_back_with_it(venv, project):
    """Under the extras the manifest asks the survivor for, and only those."""
    venv.installed[VOICE] = "0.4.0"
    venv.installed[TELEGRAM] = "0.1.10"
    venv.installed_requires[TELEGRAM] = [f'{VOICE}>=0.4; extra == "speech"']
    manifest = _manifest(project)

    manifest.write_text(f'[[feature]]\nname = "{TELEGRAM}"\npypi = ""\n', encoding="utf-8")
    assert _unrecorded(project) == [VOICE]

    manifest.write_text(
        f'[[feature]]\nname = "{TELEGRAM}"\npypi = ""\nextras = ["speech"]\n',
        encoding="utf-8",
    )
    assert _unrecorded(project) == []


def test_an_editable_extension_is_recorded_on_its_checkout(venv, project):
    venv.installed[VOICE] = "0.4.0"
    venv.editable[VOICE] = "C:\\src\\kestrel-feature-voice"

    (extension,) = unrecorded_extensions(_manifest(project), locked=(), provisioned=())

    assert extension == UnrecordedExtension(VOICE, "0.4.0", "C:\\src\\kestrel-feature-voice")
    parsed = tomllib.loads(extension.entry())["feature"]
    assert parsed == [{"name": VOICE, "editable": "C:\\src\\kestrel-feature-voice"}]


def test_an_extension_an_agent_allowlists_is_kept_by_reconcile(
    venv, project, update, monkeypatch, capsys,
):
    """Reconcile reinstalls what an allowlist names, so the update goes ahead.

    The allowlist is set where reconcile derives it, so no other gate meets
    an agent this test does not otherwise model.
    """
    monkeypatch.setattr(
        "kestrel_sovereign.feature_reconcile.compute_required_classes",
        lambda multi_agent: ({"VoiceFeature"}, []),
    )
    venv.installed[VOICE] = "0.4.0"

    assert _update() == 0

    assert update == ["uv sync", "restart"]
    assert "each installed extension is in uv.lock" in capsys.readouterr().out
