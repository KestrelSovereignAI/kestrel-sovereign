"""``feature sync --preflight``: what a sync would certainly refuse, before it runs (#3702).

``kestrel update`` ran ``uv sync`` (which prunes every package its lock does
not name) and then learned from ``feature sync`` that the host manifest pinned
``arize-phoenix==17.7.0`` and ``pydantic-ai-slim==1.107.1`` while the incoming
lock pinned 20.23.0 and 2.57.0. Every restore is held to the lock's line AND
the manifest's window, so none could succeed, and the host was left without
its features. The contradiction is in the two declarations, so it is decided
here from them alone, before anything is installed, and the update refuses on
it. Driven against the shared fake venv so the preflight's verdict can be
checked against what the guarded sync actually does.
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import pytest

from kestrel_sovereign import cli, cli_lifecycle, paths
from kestrel_sovereign import feature_reconcile as fr
from kestrel_sovereign.cli_features import run_sync_preflight, sync_preflight
from tests.utils.fake_uv import CORE, SDK, FakeUv, use_fake_uv


def _lock_text(**pins) -> str:
    """A lock that pins core, its SDK, and each ``name=version`` in *pins*."""
    text = (
        "version = 1\nrevision = 3\n"
        f'\n[[package]]\nname = "{CORE}"\nversion = "0.52.0"\nsource = {{ editable = "." }}\n'
        f'\n[[package]]\nname = "{SDK}"\nversion = "0.36.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    for name, version in pins.items():
        text += (
            f'\n[[package]]\nname = "{name.replace("_", "-")}"\nversion = "{version}"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
        )
    return text


def _checkout(tmp_path, lock, name="kestrel-sovereign") -> str:
    checkout = tmp_path / name
    checkout.mkdir()
    if lock is not None:
        (checkout / "uv.lock").write_text(lock, encoding="utf-8")
    return str(checkout)


def _manifest(tmp_path, body: str):
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text(body, encoding="utf-8")
    return manifest


def _index(manifest):
    return fr.build_source_index(cli._load_host_manifest(manifest), {})


@pytest.fixture(autouse=True)
def _own_project_home(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))
    launch = tmp_path.parent / f"{tmp_path.name}-launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    monkeypatch.setattr("kestrel_sovereign.feature_registry.load_registry", lambda: {})


@pytest.fixture
def venv(monkeypatch, tmp_path):
    """Editable core on a checkout whose lock pins phoenix 20.23.0 and slim 2.57.0.

    Phoenix is the one package the double installs, published at both versions
    and accepting this core, so the only thing that can fail its install is the
    contradiction under test.
    """
    venv = FakeUv(
        core_checkout=_checkout(
            tmp_path,
            _lock_text(**{"arize-phoenix": "20.23.0", "pydantic-ai-slim": "2.57.0"}),
        ),
        feature="arize-phoenix",
        feature_version="20.23.0",
        feature_requires=">=0.52",
    )
    venv.package_index["arize-phoenix"] = ["17.7.0", "20.23.0"]
    use_fake_uv(monkeypatch, venv)
    return venv


PHOENIX_STALE = '[[feature]]\nname = "arize-phoenix"\npypi = "==17.7.0"\n'


# The host manifest as it stood on 2026-10-19.
STALE = (
    '[[feature]]\nname = "arize-phoenix"\npypi = "==17.7.0"\n'
    '[[feature]]\nname = "pydantic-ai-slim[anthropic,mcp]"\npypi = "==1.107.1"\n'
)


def test_a_window_that_excludes_the_locked_version_is_a_conflict(venv, tmp_path):
    result = sync_preflight(_index(_manifest(tmp_path, STALE)))

    assert result.refuses
    assert [(c.package, c.window, c.locked) for c in result.conflicts] == [
        ("arize-phoenix", "==17.7.0", "20.23.0"),
        # Named without its inline extras, as the lock names it.
        ("pydantic-ai-slim", "==1.107.1", "2.57.0"),
    ]


def test_the_guarded_sync_refuses_exactly_what_the_preflight_names(
    venv, tmp_path, capsys,
):
    """Not a second opinion: the sync it predicts fails on the same entry, and
    for the reason it names (the window and the lock's line together)."""
    manifest = _manifest(tmp_path, PHOENIX_STALE)
    sync = types.SimpleNamespace(
        manifest=str(manifest), capture=False, dry_run=False, allow_dirty=False,
    )

    assert run_sync_preflight(manifest) == 1
    capsys.readouterr()
    assert cli.cmd_feature_sync(sync) != 0

    out = capsys.readouterr().out
    assert "arize-phoenix==17.7.0 and arize-phoenix===20.23.0" in out
    assert "arize-phoenix" not in venv.installed


@pytest.mark.parametrize(
    "body",
    [
        '[[feature]]\nname = "arize-phoenix"\npypi = ""\n',
        '[[feature]]\nname = "arize-phoenix"\npypi = ">=20.23,<21"\n',
        '[[feature]]\nname = "arize-phoenix"\n',
    ],
    ids=["any-version", "window-contains-it", "legacy-no-source"],
)
def test_a_window_that_contains_the_locked_version_passes(venv, tmp_path, body):
    result = sync_preflight(_index(_manifest(tmp_path, body)))

    assert not result.refuses
    assert result.lock_path is not None


def test_the_sync_installs_what_the_preflight_passes(venv, tmp_path):
    manifest = _manifest(tmp_path, '[[feature]]\nname = "arize-phoenix"\npypi = ""\n')
    sync = types.SimpleNamespace(
        manifest=str(manifest), capture=False, dry_run=False, allow_dirty=False,
    )

    assert run_sync_preflight(manifest) == 0
    assert cli.cmd_feature_sync(sync) == 0
    assert venv.installed["arize-phoenix"] == "20.23.0"


def test_a_package_the_lock_does_not_pin_has_nothing_to_contradict(venv, tmp_path):
    result = sync_preflight(_index(_manifest(
        tmp_path, '[[feature]]\nname = "kestrel-feature-voice"\npypi = "==0.1.0"\n',
    )))

    assert not result.refuses
    assert result.compared == 0


def test_a_locked_pre_release_is_inside_a_window_that_admits_it(monkeypatch, tmp_path):
    use_fake_uv(monkeypatch, FakeUv(
        core_checkout=_checkout(tmp_path, _lock_text(**{"arize-phoenix": "21.0.0rc1"})),
    ))

    result = sync_preflight(_index(_manifest(
        tmp_path, '[[feature]]\nname = "arize-phoenix"\npypi = ">=20.23"\n',
    )))

    assert not result.refuses
    assert result.compared == 1


def test_an_entry_held_on_its_checkout_is_not_a_window(venv, tmp_path):
    phoenix = tmp_path / "phoenix-checkout"
    phoenix.mkdir()
    result = sync_preflight(_index(_manifest(
        tmp_path, f'[[feature]]\nname = "arize-phoenix"\neditable = "{phoenix}"\n',
    )))

    assert not result.refuses


def test_a_lock_that_cannot_hold_the_sync_refuses(monkeypatch, tmp_path):
    use_fake_uv(monkeypatch, FakeUv(
        core_checkout=_checkout(tmp_path, "this is not a lock [[["),
    ))

    result = sync_preflight(_index(_manifest(
        tmp_path, '[[feature]]\nname = "arize-phoenix"\npypi = ""\n',
    )))

    assert result.refuses
    assert result.error


def test_core_declared_from_the_index_brings_no_lock(venv, tmp_path):
    result = sync_preflight(_index(_manifest(
        tmp_path, f'[[feature]]\nname = "{CORE}"\npypi = ">=0.52"\n' + STALE,
    )))

    assert not result.refuses


def test_core_declared_on_another_checkout_is_held_to_that_checkouts_lock(
    venv, tmp_path,
):
    other = _checkout(
        tmp_path, _lock_text(**{"arize-phoenix": "17.7.0"}), name="other-core",
    )
    result = sync_preflight(_index(_manifest(
        tmp_path,
        f'[[feature]]\nname = "{CORE}"\neditable = "{other}"\n'
        '[[feature]]\nname = "arize-phoenix"\npypi = "==17.7.0"\n',
    )))

    assert not result.refuses
    assert str(result.lock_path).startswith(other)


def test_the_cli_names_each_conflict_and_the_fix(venv, tmp_path, capsys):
    manifest = _manifest(tmp_path, STALE)

    rc = cli.cmd_feature_sync(argparse.Namespace(
        manifest=str(manifest), capture=False, dry_run=False, allow_dirty=False,
        preflight=True,
    ))

    assert rc == 1
    err = capsys.readouterr().err
    assert "arize-phoenix: the manifest declares arize-phoenix==17.7.0, the lock pins 20.23.0" in err
    assert 'pypi = ""' in err
    assert "arize-phoenix" not in venv.installed


def test_a_manifest_that_will_not_read_refuses(tmp_path, capsys):
    assert run_sync_preflight(_manifest(tmp_path, "[[feature]]\npypi = 3\n")) == 1
    assert "would not read" in capsys.readouterr().err


# --- kestrel update refuses before the install prunes ------------------------


@pytest.fixture
def update(venv, tmp_path, monkeypatch):
    """``kestrel update`` on the fake venv, recording the steps that change it."""
    checkout = tmp_path / "kestrel-sovereign"
    (checkout / ".git").mkdir()
    steps = []
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: checkout)
    monkeypatch.setattr(cli, "_run_uv_sync", lambda _: steps.append("uv sync") or (0, ""))
    monkeypatch.setattr(
        cli, "cmd_feature_sync", lambda args: steps.append("feature sync") or 0,
    )
    monkeypatch.setattr(
        cli_lifecycle, "_run_feature_reconcile", lambda *a, **kw: 0,
    )
    monkeypatch.setattr(cli_lifecycle, "_report_core_lock", lambda: None)
    monkeypatch.setattr(cli, "cmd_restart", lambda args: steps.append("restart") or 0)
    return steps


def _update(*flags):
    return cli.cmd_update(cli.build_parser().parse_args(["update", "--no-pull", *flags]))


def test_update_refuses_a_contradicted_manifest_before_installing(
    update, tmp_path, capsys,
):
    _manifest(tmp_path, STALE)

    assert _update() == 1

    assert update == []
    err = capsys.readouterr().err
    assert "preflight: REFUSED" in err
    assert "arize-phoenix==17.7.0" in err


def test_update_with_continue_on_error_installs_anyway(update, tmp_path, capsys):
    _manifest(tmp_path, STALE)

    assert _update("--continue-on-error") == 0

    assert update == ["uv sync", "feature sync", "restart"]
    assert "continuing (--continue-on-error)" in capsys.readouterr().err


def test_update_installs_when_the_manifest_agrees(update, tmp_path):
    _manifest(tmp_path, '[[feature]]\nname = "arize-phoenix"\npypi = ""\n')

    assert _update() == 0

    assert update == ["uv sync", "feature sync", "restart"]


def test_update_without_features_has_no_sync_to_predict(update, tmp_path):
    _manifest(tmp_path, STALE)

    assert _update("--no-features") == 0

    assert update == ["uv sync", "restart"]


# --- the lock the pull will land, read before it lands -----------------------

import subprocess as _subprocess

from kestrel_sovereign.core_lock import CoreLockError, read_core_lock_at

REAL_RUN = _subprocess.run


def _git(cwd, *args):
    return REAL_RUN(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit_lock(repo, lock, message):
    (repo / "uv.lock").write_text(lock, encoding="utf-8")
    _git(repo, "add", "uv.lock")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def incoming(tmp_path, monkeypatch):
    """Core's checkout on a lock pinning phoenix 17.7.0; its upstream now pins 20.23.0.

    The venv's metadata is the fake's; git is real, because the revision the
    pull lands on is what this reads.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _commit_lock(origin, _lock_text(**{"arize-phoenix": "17.7.0"}), "old lock")
    clone = tmp_path / "kestrel-sovereign"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    new = _commit_lock(origin, _lock_text(**{"arize-phoenix": "20.23.0"}), "new lock")
    _git(clone, "fetch", "-q")
    use_fake_uv(monkeypatch, FakeUv(
        core_checkout=str(clone), feature="arize-phoenix",
        feature_version="17.7.0", feature_requires=">=0.52",
    ))
    monkeypatch.setattr(_subprocess, "run", REAL_RUN)
    return clone, new


def test_the_preflight_reads_the_lock_the_revision_records(incoming, tmp_path):
    clone, new = incoming
    index = _index(_manifest(tmp_path, PHOENIX_STALE))

    # On disk, the lock still agrees with the manifest...
    assert not sync_preflight(index).refuses
    # ...but the revision the pull lands on does not.
    result = sync_preflight(index, lock_revision=new)

    assert [(c.package, c.locked) for c in result.conflicts] == [
        ("arize-phoenix", "20.23.0"),
    ]
    assert str(result.lock_path).endswith(f"uv.lock@{new}")


def test_a_revision_without_a_lock_has_none(incoming):
    clone, _ = incoming
    _git(clone, "rm", "-q", "uv.lock")
    _git(clone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "no lock")

    assert read_core_lock_at(clone, "HEAD") is None


def test_a_revision_the_checkout_does_not_have_is_an_error(incoming):
    clone, _ = incoming

    with pytest.raises(CoreLockError, match="no revision"):
        read_core_lock_at(clone, "f" * 40)


@pytest.fixture
def real_update(incoming, tmp_path, monkeypatch):
    """``kestrel update`` on *incoming*'s clone: real git, recorded install steps."""
    clone, new = incoming
    project = tmp_path / "project"
    project.mkdir()
    steps = []
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: clone)
    monkeypatch.setattr(cli, "_run_uv_sync", lambda _: steps.append("uv sync") or (0, ""))
    monkeypatch.setattr(
        cli, "cmd_feature_sync", lambda args: steps.append("feature sync") or 0,
    )
    monkeypatch.setattr(cli_lifecycle, "_run_feature_reconcile", lambda *a, **kw: 0)
    monkeypatch.setattr(cli_lifecycle, "_report_core_lock", lambda: None)
    monkeypatch.setattr(cli, "cmd_restart", lambda args: steps.append("restart") or 0)
    return clone, new, project, steps


def test_update_refuses_the_incoming_lock_before_pulling_anything(
    real_update, capsys,
):
    clone, new, project, steps = real_update
    _manifest(project, PHOENIX_STALE)
    head = _git(clone, "rev-parse", "HEAD")

    rc = cli.cmd_update(cli.build_parser().parse_args(["update"]))

    assert rc == 1
    assert steps == []
    assert _git(clone, "rev-parse", "HEAD") == head
    err = capsys.readouterr().err
    assert "arize-phoenix==17.7.0, the lock pins 20.23.0" in err
    assert "nothing was pulled, installed or restarted" in err


def test_update_pulls_and_installs_when_the_incoming_lock_agrees(real_update):
    clone, new, project, steps = real_update
    _manifest(project, '[[feature]]\nname = "arize-phoenix"\npypi = ""\n')

    rc = cli.cmd_update(cli.build_parser().parse_args(["update"]))

    assert rc == 0
    assert _git(clone, "rev-parse", "HEAD") == new
    assert steps == ["uv sync", "feature sync", "restart"]


def test_core_declared_on_the_checkout_being_pulled_is_read_at_the_revision(
    incoming, tmp_path,
):
    """``feature sync --capture`` declares an editable core on its own checkout:
    that is the checkout the caller's pull or checkout moves, so the revision
    the caller names is the one, not the one a pull of its upstream would land
    (the coordinator checks out ``FETCH_HEAD``, which may be a tag)."""
    clone, new = incoming
    _git(clone, "checkout", "-q", "-b", "release")
    tag = _commit_lock(clone, _lock_text(**{"arize-phoenix": "21.0.0"}), "release lock")
    _git(clone, "checkout", "-q", "main")
    index = _index(_manifest(
        tmp_path,
        f'[[feature]]\nname = "{CORE}"\neditable = "{clone}"\n' + PHOENIX_STALE,
    ))

    assert [c.locked for c in sync_preflight(index, lock_revision=new).conflicts] == [
        "20.23.0",
    ]
    assert [c.locked for c in sync_preflight(index, lock_revision=tag).conflicts] == [
        "21.0.0",
    ]
    # Nothing moves it: the lock on disk.
    assert not sync_preflight(index).refuses


def test_core_declared_on_another_checkout_keeps_that_checkouts_lock(
    incoming, tmp_path,
):
    """The pull moves the installed core's checkout; a declared core elsewhere
    is not that checkout, so the pulled revision says nothing about its lock."""
    clone, new = incoming
    other = _checkout(
        tmp_path, _lock_text(**{"arize-phoenix": "17.7.0"}), name="other-core",
    )
    index = _index(_manifest(
        tmp_path,
        f'[[feature]]\nname = "{CORE}"\neditable = "{other}"\n' + PHOENIX_STALE,
    ))

    result = sync_preflight(index, lock_revision=new)

    assert not result.refuses
    assert str(result.lock_path).startswith(other)


def test_a_core_the_sync_will_pull_is_read_at_that_pulls_revision(
    incoming, tmp_path, capsys,
):
    """A manifest moving core to another checkout: ``feature sync`` pulls that
    checkout before linking it, so ITS pending pull decides the lock."""
    clone, new = incoming
    origin = tmp_path / "other-origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _commit_lock(origin, _lock_text(**{"arize-phoenix": "17.7.0"}), "old lock")
    other = tmp_path / "other-core"
    _git(tmp_path, "clone", "-q", str(origin), str(other))
    _commit_lock(origin, _lock_text(**{"arize-phoenix": "20.23.0"}), "new lock")
    manifest = _manifest(
        tmp_path,
        f'[[feature]]\nname = "{CORE}"\neditable = "{other}"\n'
        '[[feature]]\nname = "arize-phoenix"\npypi = ">=20.23"\n',
    )

    # On disk the other checkout pins 17.7.0, which the window excludes; the
    # pull the sync will make lands 20.23.0, which it contains.
    assert run_sync_preflight(manifest) == 0, capsys.readouterr().err


@pytest.mark.parametrize(
    ("extras", "refuses"), [('extras = ["observability"]\n', False), ("", True)],
    ids=["with-extras-the-sync-pulls-it", "linked-as-declared-nothing-pulls-it"],
)
def test_core_on_the_installed_checkout_is_read_where_the_sync_leaves_it(
    incoming, tmp_path, extras, refuses,
):
    """With no pull of its own (``--no-pull``), the update leaves the checkout
    to the sync, which pulls a declared core it has to act on: one whose extras
    it ensures. One linked exactly as declared it leaves alone."""
    clone, new = incoming
    index = _index(_manifest(
        tmp_path,
        f'[[feature]]\nname = "{CORE}"\neditable = "{clone}"\n{extras}'
        '[[feature]]\nname = "arize-phoenix"\npypi = ">=20.23"\n',
    ))

    assert sync_preflight(index).refuses is refuses


def test_a_dry_run_preflight_fetches_nothing(incoming, tmp_path, monkeypatch):
    clone, new = incoming
    other = tmp_path / "other-core"
    _git(tmp_path, "clone", "-q", str(clone), str(other))
    manifest = _manifest(
        tmp_path,
        f'[[feature]]\nname = "{CORE}"\neditable = "{other}"\n'
        '[[feature]]\nname = "arize-phoenix"\npypi = ""\n',
    )
    fetches = []
    monkeypatch.setattr(cli, "_run_git_fetch", lambda c: fetches.append(c) or (0, ""))

    rc = cli.cmd_feature_sync(argparse.Namespace(
        manifest=str(manifest), capture=False, dry_run=True, allow_dirty=False,
        preflight=True,
    ))

    assert rc == 0
    assert fetches == []


def test_the_revision_belongs_to_the_checkout_the_caller_names(incoming, tmp_path):
    """The coordinator moves its repo, which need not be the checkout core is
    linked from now: a manifest declaring that repo is read at the revision
    the coordinator lands it on, and nothing is fetched there."""
    clone, new = incoming
    other = tmp_path / "other-core"
    _git(tmp_path, "clone", "-q", str(clone), str(other))
    _git(other, "checkout", "-q", "-b", "release")
    tag = _commit_lock(other, _lock_text(**{"arize-phoenix": "21.0.0"}), "release lock")
    _git(other, "checkout", "-q", "main")
    index = _index(_manifest(
        tmp_path,
        f'[[feature]]\nname = "{CORE}"\neditable = "{other}"\n' + PHOENIX_STALE,
    ))

    result = sync_preflight(index, lock_revision=tag, lock_checkout=other)

    assert [c.locked for c in result.conflicts] == ["21.0.0"]


def test_local_lock_edits_are_the_lock_that_lands(incoming, tmp_path):
    """``--allow-dirty``: git carries a local ``uv.lock`` edit across a pull
    that leaves the file alone, so that edit is what the sync is held to."""
    clone, _ = incoming
    origin = Path(_git(clone, "remote", "get-url", "origin"))
    (origin / "README").write_text("unrelated\n")
    _git(origin, "add", "README")
    _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "readme")
    _git(clone, "fetch", "-q")
    _git(clone, "merge", "-q", "--ff-only", "origin/main~1")  # HEAD at the 20.23.0 lock
    incoming_rev = _git(clone, "rev-parse", "origin/main")
    (clone / "uv.lock").write_text(_lock_text(**{"arize-phoenix": "17.7.0"}))
    index = _index(_manifest(
        tmp_path, '[[feature]]\nname = "arize-phoenix"\npypi = "==20.23.0"\n',
    ))

    result = sync_preflight(index, lock_revision=incoming_rev)

    assert [c.locked for c in result.conflicts] == ["17.7.0"]
