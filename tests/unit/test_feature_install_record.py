"""``feature install`` records what it installs, so ``kestrel update`` keeps it (#4127).

Measured on a fresh Windows install: ``kestrel feature install
channel_telegram``, then ``kestrel update``. The update's ``uv sync`` removed
the package (core's lock does not name it), and ``feature sync`` restored
nothing, because the host manifest it restores from had never heard of it. The
agent came back without its channel, and nothing said so.

Two halves, pinned here against the shared fake venv:

* both install commands append the package to the host manifest, without
  rewriting what an operator wrote there;
* ``kestrel update`` refuses, before its ``uv sync`` changes anything, to
  uninstall an installed extension that nothing records (installed before the
  fix, or by hand), and prints the entry that records it. What the sync
  removes is asked of uv itself.
"""

from __future__ import annotations

import argparse
import re
import subprocess
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
def test_an_install_appends_and_keeps_every_byte_already_there(venv, project, existing):
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
def test_a_package_the_manifest_already_declares_is_left_alone(venv, project, spelling):
    """Whatever source it names: the operator's declaration is theirs."""
    manifest = _manifest(project)
    text = f'[[feature]]\nname = "{spelling}"\neditable = "/src/voice"\n'
    manifest.write_text(text, encoding="utf-8")

    assert record_feature_install(VOICE) is None

    assert manifest.read_text(encoding="utf-8") == text


def test_the_install_is_recorded_in_the_manifest_its_guard_read(
    venv, project, tmp_path, monkeypatch,
):
    """One file: the manifest whose checkouts held the install records it."""
    elsewhere = tmp_path / "elsewhere" / DEFAULT_HOST_MANIFEST
    elsewhere.parent.mkdir()
    read = []

    def manifest_path(args):
        read.append(args)
        return elsewhere

    monkeypatch.setattr(cli, "_host_manifest_path", manifest_path)

    assert cli.cmd_feature_install(argparse.Namespace(name="voice")) == 0

    assert read, "the guard and the record name the manifest through one seam"
    assert cli._load_host_manifest(elsewhere)[0]["name"] == VOICE
    assert not _manifest(project).exists()


def test_an_install_records_the_copy_installed_not_the_one_asked_for(venv, project):
    """Already linked from a checkout, voice is "already satisfied": the install
    leaves the link, so the entry names the checkout. ``pypi = ""`` would have
    the next sync replace it from the index."""
    venv.installed[VOICE] = "0.4.0"
    venv.editable[VOICE] = "/src/kestrel-feature-voice"

    assert cli.cmd_feature_install(argparse.Namespace(name="voice")) == 0

    assert venv.editable[VOICE] == "/src/kestrel-feature-voice"
    assert _declared(project)[VOICE] == {
        "name": VOICE, "editable": "/src/kestrel-feature-voice", "pypi": None, "extras": [],
    }


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
def test_a_manifest_that_cannot_take_the_entry_is_refused_unchanged(venv, project, existing):
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

    def removals(_checkout, revision=None):
        return tuple(sorted(name for name in venv.installed if name not in locked))

    def uv_sync(_checkout):
        steps.append("uv sync")
        for name in removals(_checkout):
            del venv.installed[name]
            venv.editable.pop(name, None)
        return 0, ""

    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: checkout)
    monkeypatch.setattr(cli, "_git_head", lambda _: None)
    monkeypatch.setattr(cli, "_uv_sync_removals", removals)
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


def test_no_answer_from_uv_is_not_an_empty_one(venv, project, update, monkeypatch, capsys):
    """Every installed extension would look kept; say uv is the problem."""
    venv.installed[VOICE] = "0.4.0"

    def refuse(_checkout):
        raise cli_lifecycle.UvDryRunError("`uv sync --dry-run` exited 2: unexpected argument")

    monkeypatch.setattr(cli, "_uv_sync_removals", lambda checkout, revision=None: refuse(checkout))

    assert _update() == 1

    assert update == []
    err = capsys.readouterr().err
    assert "• extensions: FAILED — `uv sync --dry-run` exited 2" in err
    assert "REFUSED" not in err


# --- what survives an update ---------------------------------------------------


def _unrecorded(project, *, removals, provisioned=()):
    asked = []

    def provisioned_without(gone):
        asked.append(gone)
        return provisioned

    found = [
        extension.package
        for extension in unrecorded_extensions(
            _manifest(project), removals=removals, provisioned_without=provisioned_without,
        )
    ]
    return found, asked


def test_each_way_a_package_survives(venv, project):
    venv.installed[VOICE] = "0.4.0"
    venv.installed[TELEGRAM] = "0.1.10"
    both = (TELEGRAM, VOICE)

    assert _unrecorded(project, removals=both)[0] == [TELEGRAM, VOICE]
    # uv keeps it (its lock installs it).
    assert _unrecorded(project, removals=(TELEGRAM,))[0] == [TELEGRAM]
    # Reconcile provisions it for an allowlist.
    assert _unrecorded(project, removals=both, provisioned=[TELEGRAM])[0] == [VOICE]
    # The manifest declares it, spelled with inline extras.
    _manifest(project).write_text(
        f'[[feature]]\nname = "{TELEGRAM}[bot]"\npypi = ">=0.1"\n', encoding="utf-8",
    )
    assert _unrecorded(project, removals=both)[0] == [VOICE]


def test_reconcile_is_asked_without_what_will_be_gone(venv, project):
    """Its classes are resolved as reconcile will see the venv: without the
    packages uv removes and the manifest does not restore."""
    venv.installed[VOICE] = "0.4.0"
    venv.installed[TELEGRAM] = "0.1.10"
    _manifest(project).write_text(
        f'[[feature]]\nname = "{TELEGRAM}"\npypi = ""\n', encoding="utf-8",
    )

    _, asked = _unrecorded(project, removals=(TELEGRAM, VOICE))

    assert asked == [frozenset({VOICE})]


def test_a_package_a_survivor_requires_comes_back_with_it(venv, project):
    """Under the extras the manifest asks the survivor for, and only those."""
    venv.installed[VOICE] = "0.4.0"
    venv.installed[TELEGRAM] = "0.1.10"
    venv.installed_requires[TELEGRAM] = [f'{VOICE}>=0.4; extra == "speech"']
    manifest = _manifest(project)
    both = (TELEGRAM, VOICE)

    manifest.write_text(f'[[feature]]\nname = "{TELEGRAM}"\npypi = ""\n', encoding="utf-8")
    assert _unrecorded(project, removals=both)[0] == [VOICE]

    manifest.write_text(
        f'[[feature]]\nname = "{TELEGRAM}"\npypi = ""\nextras = ["speech"]\n',
        encoding="utf-8",
    )
    assert _unrecorded(project, removals=both)[0] == []


def test_a_checkout_comes_back_only_from_the_manifest(venv, project):
    """Reconcile, or a survivor's requirement, would bring it back from the index."""
    venv.installed[VOICE] = "0.4.0"
    venv.editable[VOICE] = "/src/kestrel-feature-voice"

    assert _unrecorded(project, removals=(VOICE,), provisioned=[VOICE])[0] == [VOICE]

    _manifest(project).write_text(
        f'[[feature]]\nname = "{VOICE}"\neditable = "/src/kestrel-feature-voice"\n',
        encoding="utf-8",
    )
    assert _unrecorded(project, removals=(VOICE,))[0] == []


def test_an_editable_extension_is_recorded_on_its_checkout(venv, project):
    venv.installed[VOICE] = "0.4.0"
    venv.editable[VOICE] = "C:\\src\\kestrel-feature-voice"

    (extension,) = unrecorded_extensions(
        _manifest(project), removals=(VOICE,), provisioned_without=lambda gone: (),
    )

    assert extension.editable_path == "C:\\src\\kestrel-feature-voice"
    parsed = tomllib.loads(extension.entry())["feature"]
    assert parsed == [{"name": VOICE, "editable": "C:\\src\\kestrel-feature-voice"}]


@pytest.mark.parametrize(
    "from_index, expected",
    [(True, {"name": VOICE, "pypi": ""}), (False, {"name": VOICE})],
    ids=["index", "direct-url"],
)
def test_the_printed_entry_names_the_source_it_was_installed_from(from_index, expected):
    """``pypi = ""`` would stop ``feature sync`` falling back to git for a
    direct-URL install, so that one gets the entry that names no source."""
    extension = UnrecordedExtension(VOICE, "0.4.0", None, from_index=from_index)

    assert tomllib.loads(extension.entry())["feature"] == [expected]


def test_a_git_installed_extension_is_printed_without_a_source(venv, project, monkeypatch):
    """Provenance is read, not assumed: a direct URL is not an index install."""
    venv.installed[VOICE] = "0.4.0"
    monkeypatch.setattr(cli, "_from_index", lambda dist: dist != VOICE)

    (extension,) = unrecorded_extensions(
        _manifest(project), removals=(VOICE,), provisioned_without=lambda gone: (),
    )

    assert extension.from_index is False


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
    assert "every extension `uv sync` removes is reinstalled" in capsys.readouterr().out


def test_a_class_only_the_removed_package_declares_is_not_provisioned(
    venv, project, update, monkeypatch, capsys,
):
    """Not catalogued and found only through its own entry point: once uv removes
    it, reconcile cannot resolve the class, so it is not reinstalled."""
    monkeypatch.setattr(
        "kestrel_sovereign.feature_reconcile.compute_required_classes",
        lambda multi_agent: ({"TelegramChannel"}, []),
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.discover_entrypoint_feature_dists",
        lambda: {"TelegramChannel": TELEGRAM},
    )
    venv.installed[TELEGRAM] = "0.1.10"

    assert _update() == 1

    assert update == []
    assert f"    {TELEGRAM} 0.1.10" in capsys.readouterr().err


# --- asking uv what the sync removes --------------------------------------------


def _uv_answers(monkeypatch, *, stdout="", returncode=0, stderr=""):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(cli_lifecycle.subprocess, "run", run)
    return calls


def _changes(*changes):
    import json

    return json.dumps({
        "schema": {"version": "preview"},
        "sync": {"changes": [
            {"name": name, "version": version, "action": action}
            for name, version, action in changes
        ]},
    })


def test_uv_is_asked_exactly_as_the_sync_runs(monkeypatch, tmp_path):
    calls = _uv_answers(monkeypatch, stdout=_changes())

    assert cli_lifecycle._uv_sync_removals(tmp_path) == ()

    ((argv, kwargs),) = calls
    sync, env = cli_lifecycle._uv_sync_invocation()
    assert argv == [*sync, "--dry-run", "--output-format", "json", "--project", str(tmp_path)]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["env"].get("VIRTUAL_ENV") == env.get("VIRTUAL_ENV")


def test_a_version_change_is_not_a_removal(monkeypatch, tmp_path):
    """Measured with uv 0.9.22: an upgrade is an uninstall and an install."""
    _uv_answers(monkeypatch, stdout=_changes(
        ("idna", "3.20", "uninstalled"),
        ("Six", "1.16.0", "uninstalled"),
        ("six", "1.17.0", "installed"),
    ))

    assert cli_lifecycle._uv_sync_removals(tmp_path) == ("idna",)


@pytest.mark.parametrize(
    "answer",
    [
        {"returncode": 2, "stderr": "error: unexpected argument '--output-format'"},
        # An exit status is part of the answer, whatever stdout says.
        {"returncode": 1, "stdout": _changes(), "stderr": "error: lock is out of date"},
        {"stdout": "Would uninstall 1 package\n - six==1.17.0\n"},
        {"stdout": '{"sync": {}}'},
        {"stdout": '{"sync": {"changes": [{"name": "six"}]}}'},
        {"stdout": '{"sync": {"changes": [{"name": "six", "action": null}]}}'},
    ],
    ids=["uv-fails", "fails-with-an-answer", "not-json", "no-changes", "change-without-action", "null-action"],
)
def test_an_answer_that_is_not_one_raises(monkeypatch, tmp_path, answer):
    _uv_answers(monkeypatch, **answer)

    with pytest.raises(cli_lifecycle.UvDryRunError):
        cli_lifecycle._uv_sync_removals(tmp_path)


# --- uv is asked about the revision the pull lands, before it lands ------------

REAL_RUN = subprocess.run


def _git(cwd, *args):
    return REAL_RUN(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    """A checkout whose next commit changes pyproject.toml and uv.lock."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "pyproject.toml").write_text("# old pyproject\n", encoding="utf-8")
    (repo / "uv.lock").write_text("# old lock\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "old")
    old = _git(repo, "rev-parse", "HEAD")
    (repo / "pyproject.toml").write_text("# new pyproject\n", encoding="utf-8")
    (repo / "uv.lock").write_text("# new lock\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "new")
    new = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", old)
    return repo, old, new


def _uv_sees(monkeypatch):
    """Real git; uv answers nothing, and records the project files it was shown."""
    seen = []

    def run(argv, **kwargs):
        if argv[0] == "git":
            return REAL_RUN(argv, **kwargs)
        project = Path(argv[argv.index("--project") + 1])
        seen.append({path.name: path.read_text() for path in project.iterdir()})
        return subprocess.CompletedProcess(argv, 0, stdout=_changes(), stderr="")

    monkeypatch.setattr(cli_lifecycle.subprocess, "run", run)
    return seen


def test_uv_reads_the_project_as_the_revision_records_it(checkout, monkeypatch):
    repo, _old, new = checkout
    seen = _uv_sees(monkeypatch)

    assert cli_lifecycle._uv_sync_removals(repo, new) == ()

    assert seen == [{"pyproject.toml": "# new pyproject\n", "uv.lock": "# new lock\n"}]
    # Read from git: the checkout itself has not moved.
    assert (repo / "uv.lock").read_text() == "# old lock\n"


def test_local_edits_to_the_project_files_are_what_lands(checkout, monkeypatch):
    """As the lock preflight reads it: git carries the edits across the pull."""
    repo, _old, new = checkout
    (repo / "uv.lock").write_text("# edited lock\n", encoding="utf-8")
    seen = _uv_sees(monkeypatch)

    cli_lifecycle._uv_sync_removals(repo, new)

    assert seen == [{"pyproject.toml": "# old pyproject\n", "uv.lock": "# edited lock\n"}]


def test_a_revision_without_a_lock_is_no_answer(checkout, monkeypatch):
    repo, old, _new = checkout
    _git(repo, "rm", "-q", "uv.lock")
    _git(repo, "commit", "-qm", "no lock")
    _uv_sees(monkeypatch)

    with pytest.raises(cli_lifecycle.UvDryRunError, match="no uv.lock"):
        cli_lifecycle._uv_sync_removals(repo, _git(repo, "rev-parse", "HEAD"))


# --- the console's remove forgets what its install recorded ---------------------

from kestrel_sovereign.cli_features import FeatureForgetError, forget_feature_install

AROUND = (
    "# Header comment.\n"
    "[host_features]\n"
    "default_enabled = false\n"
    "\n"
    "[[feature]]\n"
    'name = "arize-phoenix"\n'
    'pypi = ">=20,<21"\n'
    "\n"
)
AFTER = (
    "# The comment above the next entry is the next entry's.\n"
    "[[feature]]\n"
    'name = "kestrel-feature-github"\n'
    'pypi = ""\n'
)


@pytest.mark.parametrize(
    "spelling", [VOICE, "voice", f"{VOICE}[local]"], ids=["dist", "registry", "extras"],
)
def test_forgetting_removes_only_that_entry(venv, project, spelling):
    entry = f'[[feature]]\nname = "{spelling}"\npypi = ""\n# a note on voice\n\n'
    _manifest(project).write_text(AROUND + entry + AFTER, encoding="utf-8")

    assert forget_feature_install(VOICE) == _manifest(project)

    assert _manifest(project).read_text(encoding="utf-8") == AROUND + AFTER


def test_install_then_forget_leaves_the_operators_text(venv, project):
    _manifest(project).write_text(AROUND + AFTER, encoding="utf-8")
    assert record_feature_install(VOICE) is not None

    forget_feature_install(VOICE)

    assert _manifest(project).read_text(encoding="utf-8").rstrip("\n") == (AROUND + AFTER).rstrip("\n")
    assert VOICE not in _declared(project)


@pytest.mark.parametrize("existing", [None, AROUND + AFTER], ids=["no-manifest", "not-declared"])
def test_nothing_to_forget_changes_nothing(project, existing):
    if existing is not None:
        _manifest(project).write_text(existing, encoding="utf-8")

    assert forget_feature_install(VOICE) is None

    if existing is None:
        assert not _manifest(project).exists()
    else:
        assert _manifest(project).read_text(encoding="utf-8") == existing


def test_a_header_in_another_value_does_not_confuse_it(venv, project):
    """Spans that do not parse alone are skipped; the result is still checked."""
    github = (
        '[[feature]]\nname = "kestrel-feature-github"\npypi = ""\n'
        'note = """\n[[feature]]\nname = "x"\n"""\n'
    )
    _manifest(project).write_text(
        github + f'[[feature]]\nname = "{VOICE}"\npypi = ""\n', encoding="utf-8",
    )

    assert forget_feature_install(VOICE) == _manifest(project)

    assert _manifest(project).read_text(encoding="utf-8") == github


def test_a_manifest_that_does_not_split_by_line_is_refused_unchanged(venv, project):
    """A header-looking line inside the entry's own value cuts it short."""
    text = (
        '[[feature]]\nname = "kestrel-feature-github"\npypi = ""\n'
        f'[[feature]]\nname = "{VOICE}"\npypi = ""\n'
        'note = """\n[[feature]]\nname = "x"\n"""\n'
    )
    _manifest(project).write_text(text, encoding="utf-8")

    with pytest.raises(FeatureForgetError) as raised:
        forget_feature_install(VOICE)

    assert _manifest(project).read_text(encoding="utf-8") == text
    assert "reinstalls it unless" in raised.value.describe()


def test_the_gate_asks_uv_about_the_revision_the_pull_lands(venv, project, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(
        cli, "_uv_sync_removals",
        lambda checkout, revision=None: asked.append((checkout, revision)) or (),
    )

    rc = cli_lifecycle._unrecorded_extension_gate(
        project, tmp_path, manifest=None, landing=lambda: "c0ffee", continue_on_error=False,
    )

    assert rc == 0
    assert asked == [(tmp_path, "c0ffee")]
