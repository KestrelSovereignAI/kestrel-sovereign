"""``kestrel feature sync`` names every locked package it leaves below uv.lock (#3324).

On the prod host an ``update_then_restart`` moved ``kestrel-sovereign-sdk``
from the locked 0.38.1 to 0.37.1, because an editable ``kestrel-claws``
checkout capped it ``<0.38``, and every step of the update reported rc=0. The
install hold from #3502 refuses that install today, but a checkout the
manifest declares, or an install that bypasses the constraints file, can
still leave a locked package below the lock.

These pin that the sync names each such package with both versions and what
forced it, even when its manifest declares nothing to install; that this is a
warning which never changes the exit status; that a sync with nothing below
the lock prints exactly what it printed before; and that every finding reaches
the ``update_log`` of ``update_then_restart``, whole, through the report the
sync writes for it, however many there are. The report
and the comparison itself (``core_lock.lock_downgrades``) are pinned at the
bottom.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import types
from pathlib import Path

import pytest

from kestrel_sovereign import cli, cli_features, paths
from kestrel_sovereign import core_lock as cl
from kestrel_sovereign import feature_reconcile as fr
from tests.utils.fake_uv import CORE, SDK, FakeUv, use_fake_uv

REPO_ROOT = Path(__file__).resolve().parents[2]

CLAWS = "kestrel-claws"
SDK_LOCKED = "0.38.1"
#: The line the prod host's update would have printed.
CLAWS_FINDING = (
    f"{SDK} 0.37.1 installed, below the {SDK_LOCKED} uv.lock pins; forced by "
    f"{CLAWS} (requires {SDK}<0.38)"
)


def _lock_text(also_at_2=()) -> str:
    """A lock pinning core, its SDK at 0.38.1, and anthropic (not installed),
    and each of *also_at_2* at 2.0.0."""
    return (
        "version = 1\nrevision = 3\n"
        f'\n[[package]]\nname = "{CORE}"\nversion = "0.52.0"\n'
        'source = { editable = "." }\n'
        f'\n[[package]]\nname = "{SDK}"\nversion = "{SDK_LOCKED}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        '\n[[package]]\nname = "anthropic"\nversion = "0.117.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        + "".join(
            f'\n[[package]]\nname = "{name}"\nversion = "2.0.0"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
            for name in also_at_2
        )
    )


@pytest.fixture(autouse=True)
def _own_project_home(monkeypatch, tmp_path):
    """The project directory is this test's own, and the cwd a sibling of it."""
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))
    launch = tmp_path.parent / f"{tmp_path.name}-launch"
    launch.mkdir()
    monkeypatch.chdir(launch)


@pytest.fixture(autouse=True)
def _no_ambient_lock_report(monkeypatch):
    """No sync here writes a lock report unless its test asks for one."""
    monkeypatch.delenv(cl.SYNC_LOCK_REPORT_ENV, raising=False)


@pytest.fixture(autouse=True)
def _no_registry(monkeypatch):
    """Every manifest label here names its distribution directly."""
    monkeypatch.setattr(
        "kestrel_sovereign.feature_registry.load_registry", lambda: {},
    )


def _core_checkout(tmp_path, lock: str | None = _lock_text()) -> str:
    """Core's checkout on disk, with *lock* as its uv.lock (None: no lock)."""
    checkout = tmp_path / "kestrel-sovereign"
    checkout.mkdir()
    if lock is not None:
        (checkout / "uv.lock").write_text(lock, encoding="utf-8")
    return str(checkout)


def _host(
    tmp_path,
    *,
    sdk: str,
    claws_requires=(f"{SDK}<0.38",),
    claws_installed=True,
    lock: str | None = _lock_text(),
    **kw,
) -> FakeUv:
    """Editable core whose checkout holds *lock*, the SDK at *sdk*, and claws.

    Claws is the feature from the issue: an editable checkout whose
    requirement on the SDK is *claws_requires*. ``claws_installed=False`` is
    the venv ``uv sync`` leaves behind: the feature pruned, so the sync
    installs it again. The index publishes the SDK below, at, and above the
    lock, so an install free to move it can go either way.
    """
    venv = FakeUv(
        core_checkout=_core_checkout(tmp_path, lock),
        sdk_checkout=None,
        feature=CLAWS,
        feature_version="0.4.2",
        feature_requires=">=0.52",
        **kw,
    )
    venv.installed[SDK] = sdk
    venv.package_index[SDK] = ["0.37.1", SDK_LOCKED, "0.39.0"]
    venv.installed_requires[CLAWS] = list(claws_requires)
    if claws_installed:
        venv.installed[CLAWS] = "0.4.2"
        venv.editable[CLAWS] = _claws_checkout(tmp_path)
    return venv


def _claws_checkout(tmp_path) -> str:
    return str(tmp_path / CLAWS)


def _manifest(tmp_path, text: str | None = None) -> Path:
    manifest = tmp_path / "features.toml"
    manifest.write_text(
        text if text is not None
        else f'[[feature]]\nname = "{CLAWS}"\neditable = "{_claws_checkout(tmp_path)}"\n'
    )
    return manifest


def _sync(manifest, *, dry_run=False) -> int:
    return cli.cmd_feature_sync(types.SimpleNamespace(
        manifest=str(manifest), capture=False, dry_run=dry_run, allow_dirty=False,
    ))


def _source_index(manifest):
    return fr.build_source_index(cli._load_host_manifest(Path(manifest)), {})


def _warnings(caplog) -> list:
    return [
        record.getMessage() for record in caplog.records
        if record.name == cli_features.__name__ and record.levelno == logging.WARNING
    ]


# --- 1. a feature that caps a locked package below the lock -----------------


def test_a_feature_capping_a_locked_package_below_the_lock_is_named(
    monkeypatch, tmp_path, capsys, caplog,
):
    """The prod host's venv after the update: claws present, its cap met by
    an SDK one minor version under the lock. The sync installs nothing, so
    before #3324 it said nothing at all."""
    venv = _host(tmp_path, sdk="0.37.1")
    use_fake_uv(monkeypatch, venv)
    manifest = _manifest(tmp_path)

    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc = _sync(manifest)

    out = capsys.readouterr().out
    assert rc == 0  # a warning, never a failure
    assert not venv.commands  # nothing was installed: the finding is the venv's
    (finding,) = cli.sync_lock_findings(_source_index(manifest))
    assert finding == cl.LockDowngrade(
        name=SDK,
        locked=SDK_LOCKED,
        installed="0.37.1",
        forced_by=(cl.LockForcer(CLAWS, f"requires {SDK}<0.38"),),
    )
    assert f"WARNING: {CLAWS_FINDING}" in out
    assert "not running what CI tested" in out
    assert _warnings(caplog) == [f"feature sync: {CLAWS_FINDING}"]


def test_a_sync_that_moves_a_locked_package_below_the_lock_names_it(
    monkeypatch, tmp_path, capsys, caplog,
):
    """The update itself: ``uv sync`` left the SDK at the lock and pruned
    claws, and the sync's install of claws resolved its dependencies without
    the constraints file (a build step of its own), moving the SDK down."""
    venv = _host(
        tmp_path, sdk=SDK_LOCKED, claws_installed=False,
        dependencies_bypass_constraints=True,
    )
    use_fake_uv(monkeypatch, venv)

    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc = _sync(_manifest(tmp_path))

    out = capsys.readouterr().out
    assert rc == 0
    assert venv.installed[CLAWS] == "0.4.2"
    assert venv.installed[SDK] == "0.37.1"  # the sync is what moved it
    assert CLAWS_FINDING in out
    assert _warnings(caplog) == [f"feature sync: {CLAWS_FINDING}"]


def test_a_declared_checkout_below_the_lock_names_the_manifest(
    monkeypatch, tmp_path, capsys,
):
    """The hold keeps a declared checkout at whatever it builds, so a stale
    SDK checkout is the one downgrade #3502 leaves in place on purpose."""
    sdk_checkout = str(tmp_path / "kestrel-sovereign-sdk")
    venv = _host(
        tmp_path, sdk="0.37.1", claws_installed=False,
        dependency_checkouts={sdk_checkout: (SDK, "0.37.1")},
    )
    venv.editable[SDK] = sdk_checkout
    use_fake_uv(monkeypatch, venv)
    manifest = _manifest(
        tmp_path, f'[[feature]]\nname = "{SDK}"\neditable = "{sdk_checkout}"\n',
    )

    rc = _sync(manifest)

    out = capsys.readouterr().out
    assert rc == 0
    (finding,) = cli.sync_lock_findings(_source_index(manifest))
    assert finding.forced_by == (
        cl.LockForcer("host manifest", f"holds it on checkout {sdk_checkout}"),
    )
    assert (
        f"{SDK} 0.37.1 installed, below the {SDK_LOCKED} uv.lock pins; forced "
        f"by host manifest (holds it on checkout {sdk_checkout})"
    ) in out


def test_a_downgrade_nothing_installed_explains_is_named_with_an_unknown_cause(
    monkeypatch, tmp_path, capsys,
):
    venv = _host(tmp_path, sdk="0.37.1", claws_requires=())
    use_fake_uv(monkeypatch, venv)

    rc = _sync(_manifest(tmp_path))

    out = capsys.readouterr().out
    assert rc == 0
    assert (
        f"{SDK} 0.37.1 installed, below the {SDK_LOCKED} uv.lock pins; forced "
        "by unknown"
    ) in out


# --- 2. nothing below the lock: no finding, output unchanged -----------------


def _sync_output(monkeypatch, tmp_path, name, *, check: bool) -> tuple:
    """``(rc, home, venv)`` of the claws sync on a fresh host at *tmp_path*/*name*.

    Claws accepts the locked SDK, so nothing ends below the lock.

    ``check=False`` runs it with the lock comparison taken out, which is the
    sync as it was before #3324.
    """
    home = tmp_path / name
    home.mkdir()
    with monkeypatch.context() as m:
        m.setenv(paths.HOME_ENV, str(home))
        venv = _host(
            home, sdk=SDK_LOCKED, claws_installed=False,
            claws_requires=(f"{SDK}>=0.37.1,<0.39",),
        )
        use_fake_uv(m, venv)
        if not check:
            m.setattr(cli_features, "sync_lock_findings", lambda *_: ())
        rc = _sync(_manifest(home))
        assert venv.installed[CLAWS] == "0.4.2"
        return rc, str(home), venv


def test_a_sync_with_nothing_below_the_lock_prints_what_it_printed_before(
    monkeypatch, tmp_path, capsys, caplog,
):
    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc, home, venv = _sync_output(monkeypatch, tmp_path, "checked", check=True)
    checked = capsys.readouterr().out.replace(home, "<home>")
    rc_before, home_before, _ = _sync_output(
        monkeypatch, tmp_path, "unchecked", check=False,
    )
    before = capsys.readouterr().out.replace(home_before, "<home>")

    assert rc == rc_before == 0
    assert checked == before
    assert _warnings(caplog) == []
    # Claws is installed and unlocked; the SDK is at the lock.
    assert venv.installed[SDK] == SDK_LOCKED
    with monkeypatch.context() as m:
        m.setenv(paths.HOME_ENV, home)
        use_fake_uv(m, venv)
        assert cli.sync_lock_findings(_source_index(Path(home) / "features.toml")) == ()


# --- 3. moved above the lock: no finding --------------------------------------


def test_a_sync_that_moves_a_locked_package_above_the_lock_names_nothing(
    monkeypatch, tmp_path, capsys, caplog,
):
    venv = _host(
        tmp_path, sdk=SDK_LOCKED, claws_installed=False,
        claws_requires=(f"{SDK}>=0.39",), dependencies_bypass_constraints=True,
    )
    use_fake_uv(monkeypatch, venv)
    manifest = _manifest(tmp_path)

    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc = _sync(manifest)

    out = capsys.readouterr().out
    assert rc == 0
    assert venv.installed[SDK] == "0.39.0"  # it did move, upward
    assert cli.sync_lock_findings(_source_index(manifest)) == ()
    assert "uv.lock" not in out
    assert "WARNING" not in out
    assert _warnings(caplog) == []


# --- 4. no usable lock: one finding, and the sync still succeeds -------------


def test_a_core_checkout_without_a_uv_lock_is_one_finding(
    monkeypatch, tmp_path, capsys, caplog,
):
    venv = _host(tmp_path, sdk="0.37.1", claws_installed=False, lock=None)
    use_fake_uv(monkeypatch, venv)

    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc = _sync(_manifest(tmp_path))

    out = capsys.readouterr().out
    assert rc == 0
    assert venv.installed[CLAWS] == "0.4.2"
    expected = (
        "installed versions were not compared against core's uv.lock: there "
        f"is no uv.lock at {Path(venv.editable[CORE]) / 'uv.lock'}"
    )
    assert f"WARNING: {expected}" in out
    assert _warnings(caplog) == [f"feature sync: {expected}"]


@pytest.mark.parametrize("text, says", [
    ("this is [[ not toml", "Expected '='"),
    ("version = 1\n", "no [[package]] entries"),
])
def test_a_uv_lock_that_will_not_read_is_one_finding(
    monkeypatch, tmp_path, capsys, caplog, text, says,
):
    """Claws is present, so the sync installs nothing: the lock that will not
    read refuses installs (#3502), and that refusal is not this finding."""
    venv = _host(tmp_path, sdk="0.37.1", lock=text)
    use_fake_uv(monkeypatch, venv)

    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc = _sync(_manifest(tmp_path))

    out = capsys.readouterr().out
    assert rc == 0
    (warning,) = _warnings(caplog)
    assert warning.startswith(
        "feature sync: installed versions were not compared against core's "
        f"uv.lock: {Path(venv.editable[CORE]) / 'uv.lock'}"
    )
    assert says in warning
    assert warning.removeprefix("feature sync: ") in out
    assert "below the" not in out


def test_core_provenance_that_will_not_read_is_one_finding(monkeypatch, tmp_path):
    """Damaged install metadata hides which checkout, and so which lock,
    describes core: not "no lock", which would say nothing."""
    venv = _host(tmp_path, sdk="0.37.1")
    venv.unreadable_provenance.add(CORE)
    use_fake_uv(monkeypatch, venv)

    (finding,) = cli.sync_lock_findings(_source_index(_manifest(tmp_path)))

    assert isinstance(finding, cl.LockNotCompared)
    assert "direct_url.json" in finding.reason


def test_a_core_with_no_checkout_has_no_lock_to_miss(monkeypatch, tmp_path, capsys):
    """An index wheel ships no uv.lock, so there is nothing missing to report:
    a pip-installed host would otherwise warn on every sync."""
    venv = _host(tmp_path, sdk="0.37.1", lock=None)
    venv.editable.pop(CORE)
    use_fake_uv(monkeypatch, venv)
    manifest = _manifest(tmp_path)

    assert cli.sync_lock_findings(_source_index(manifest)) == ()


# --- an empty manifest installs nothing and still compares -------------------
#
# ``update_then_restart`` runs the sync whenever the manifest exists, empty or
# not, so the venv ``uv sync`` left behind is compared even when there is
# nothing to restore.

def _nothing_to_sync(manifest) -> str:
    """What the sync prints for a manifest with no entries, before #3324 too."""
    return f"Manifest {manifest} declares no features. Nothing to sync.\n"


def test_an_empty_manifest_still_names_a_package_below_the_lock(
    monkeypatch, tmp_path, capsys, caplog,
):
    """Claws is installed but no longer declared: the sync has nothing to do,
    and the SDK claws holds below the lock is still named."""
    venv = _host(tmp_path, sdk="0.37.1")
    use_fake_uv(monkeypatch, venv)
    manifest = _manifest(tmp_path, "")

    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc = _sync(manifest)

    out = capsys.readouterr().out
    assert rc == 0
    assert not venv.commands
    assert out.startswith(_nothing_to_sync(manifest))
    assert f"WARNING: {CLAWS_FINDING}" in out
    assert "not running what CI tested" in out
    assert _warnings(caplog) == [f"feature sync: {CLAWS_FINDING}"]


@pytest.mark.parametrize("lock, says", [
    (None, "there is no uv.lock at"),
    ("version = 1\n", "no [[package]] entries"),
])
def test_an_empty_manifest_with_no_usable_lock_is_one_finding(
    monkeypatch, tmp_path, capsys, caplog, lock, says,
):
    venv = _host(tmp_path, sdk="0.37.1", lock=lock)
    use_fake_uv(monkeypatch, venv)
    report = tmp_path / "lock-findings.json"
    monkeypatch.setenv(cl.SYNC_LOCK_REPORT_ENV, str(report))

    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc = _sync(_manifest(tmp_path, ""))

    out = capsys.readouterr().out
    assert rc == 0
    (warning,) = _warnings(caplog)
    assert warning.startswith(
        "feature sync: installed versions were not compared against core's "
        "uv.lock: "
    )
    assert says in warning
    assert f"WARNING: {warning.removeprefix('feature sync: ')}" in out
    (finding,) = cl.read_lock_findings_report(report)
    assert finding["kind"] == "lock_not_compared"
    assert finding["message"] == warning.removeprefix("feature sync: ")


def test_an_empty_manifest_with_nothing_below_the_lock_prints_what_it_did_before(
    monkeypatch, tmp_path, capsys, caplog,
):
    """And reports an empty list: compared and clean."""
    venv = _host(tmp_path, sdk=SDK_LOCKED)
    use_fake_uv(monkeypatch, venv)
    manifest = _manifest(tmp_path, "")
    report = tmp_path / "lock-findings.json"
    monkeypatch.setenv(cl.SYNC_LOCK_REPORT_ENV, str(report))

    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc = _sync(manifest)

    assert rc == 0
    assert capsys.readouterr().out == _nothing_to_sync(manifest)
    assert _warnings(caplog) == []
    assert cl.read_lock_findings_report(report) == []


def test_an_empty_manifest_dry_run_compares_nothing(
    monkeypatch, tmp_path, capsys, caplog,
):
    """A dry run on an empty manifest is unchanged: it neither compares nor
    writes a report, whatever the venv holds."""
    venv = _host(tmp_path, sdk="0.37.1")
    use_fake_uv(monkeypatch, venv)
    manifest = _manifest(tmp_path, "")
    report = tmp_path / "lock-findings.json"
    monkeypatch.setenv(cl.SYNC_LOCK_REPORT_ENV, str(report))

    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc = _sync(manifest, dry_run=True)

    assert rc == 0
    assert capsys.readouterr().out == _nothing_to_sync(manifest)
    assert _warnings(caplog) == []
    assert not report.exists()


# --- 5. the finding reaches what update_then_restart keeps -------------------


_BOOTSTRAP = """\
import runpy
import sys

sys.path.insert(0, {repo!r})

import pytest

from tests.utils.fake_uv import SDK, FakeUv, use_fake_uv

venv = FakeUv(
    core_checkout={core!r}, sdk_checkout=None, feature={claws!r},
    feature_version="0.4.2", feature_requires=">=0.52",
)
venv.installed[SDK] = "0.37.1"
venv.installed[{claws!r}] = "0.4.2"
venv.editable[{claws!r}] = {claws_checkout!r}
venv.installed_requires[{claws!r}] = [SDK + "<0.38"]
for name in {fillers!r}:
    venv.installed[name] = "1.0.0"
for name in {downgraded!r}:
    venv.installed[name] = "1.0.0"
    venv.installed_requires[{claws!r}].append(name + "<2")
use_fake_uv(pytest.MonkeyPatch(), venv)
runpy.run_module("kestrel_sovereign.cli", run_name="__main__", alter_sys=True)
"""


_LOCK_FROM_DOWNGRADED = object()


def _fake_host_sync_step(
    tmp_path, *, fillers=(), downgraded=(), declares_claws=True,
    lock=_LOCK_FROM_DOWNGRADED,
):
    """The profile's own ``feature_sync`` step, run against a fake host.

    The step is the profile's argv with one change: a bootstrap that points
    the child's venv reads at the fake host runs ``kestrel_sovereign.cli`` as
    ``__main__``. Claws caps the SDK below the lock, and each of *downgraded*
    (locked at 2.0.0, installed at 1.0.0) below 2. *fillers* are manifest
    entries already installed, which the sync lists and leaves alone.
    ``declares_claws=False`` leaves claws installed but out of the manifest.
    *lock* is core's uv.lock (None: there is none); by default it pins core,
    the SDK, and every one of *downgraded*.
    """
    from kestrel_sovereign.features.restart_coordinator.update_profiles import (
        UPDATE_PROFILES,
    )

    (tmp_path / ".kestrel-host-features.toml").write_text(
        (
            f'[[feature]]\nname = "{CLAWS}"\neditable = "{_claws_checkout(tmp_path)}"\n'
            if declares_claws else ""
        )
        + "".join(f'[[feature]]\nname = "{name}"\n' for name in fillers)
    )
    if lock is _LOCK_FROM_DOWNGRADED:
        lock = _lock_text(downgraded)
    bootstrap = tmp_path / "fake_host_cli.py"
    bootstrap.write_text(_BOOTSTRAP.format(
        repo=str(REPO_ROOT),
        core=_core_checkout(tmp_path, lock),
        claws=CLAWS,
        claws_checkout=_claws_checkout(tmp_path),
        fillers=list(fillers),
        downgraded=list(downgraded),
    ))
    repo = tmp_path / "repo"
    repo.mkdir()
    steps = UPDATE_PROFILES["sovereign_local_uv_sync"].build_steps(
        repo_path=str(repo), target_ref="main", allow_migrations=False,
    )
    profile_step = next(step for step in steps if step.name == "feature_sync")
    assert profile_step.argv[:5] == [
        sys.executable, "-m", "kestrel_sovereign.cli", "feature", "sync",
    ]
    # The profile, not this test, asks the coordinator to keep the findings.
    assert profile_step.reports_lock_findings is True
    return dataclasses.replace(
        profile_step,
        argv=[sys.executable, str(bootstrap), *profile_step.argv[3:]],
    )


async def test_the_finding_reaches_the_update_log_of_update_then_restart(tmp_path):
    """Runs the profile's own ``feature_sync`` step through the coordinator's
    step runner, which is what ``update_log`` records.

    Forty present entries push the sync's output past the tail the log keeps,
    so this also pins that the finding is printed where the tail keeps it.
    """
    from kestrel_sovereign.features.restart_coordinator.feature import (
        _OUTPUT_TAIL_CHARS,
        RestartCoordinatorFeature,
    )

    step = _fake_host_sync_step(
        tmp_path, fillers=[f"kestrel-filler-{i:02d}" for i in range(40)],
    )

    feature = RestartCoordinatorFeature(types.SimpleNamespace(did="did:test:host"))
    outcome = await feature._run_update_step(step)

    assert outcome["ok"], outcome
    assert len(outcome["stdout_tail"]) == _OUTPUT_TAIL_CHARS  # it was cut
    assert "kestrel-filler-00" not in outcome["stdout_tail"]
    # Labelled on its own line, so a tail that cuts the note still says why.
    assert f"WARNING: {CLAWS_FINDING}" in outcome["stdout_tail"]
    assert CLAWS_FINDING in outcome["stderr_tail"]  # the WARNING log record
    # And kept whole, beside the output.
    assert outcome["lock_findings"] == [{
        "kind": "lock_downgrade",
        "package": SDK,
        "locked": SDK_LOCKED,
        "installed": "0.37.1",
        "forced_by": [{"source": CLAWS, "reason": f"requires {SDK}<0.38"}],
        "cause": f"{CLAWS} (requires {SDK}<0.38)",
        "message": CLAWS_FINDING,
    }]
    assert "lock_findings_unavailable" not in outcome


async def _logged_sync_step(tmp_path, monkeypatch, sync_step) -> dict:
    """*sync_step* run by the coordinator's update, as its request's
    ``update_log`` row records it."""
    from kestrel_sovereign.auth import CallerContext, caller_context_scope
    from kestrel_sovereign.features.restart_coordinator.feature import (
        RestartCoordinatorFeature,
    )
    from kestrel_sovereign.features.restart_coordinator.store import (
        ensure_restart_requests_table,
        get_request,
        insert_request,
        record_update_log,
    )
    from kestrel_sovereign.features.restart_coordinator.update_profiles import (
        UpdateProfile,
        UpdateStep,
    )
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db import SQLiteBackend

    profile = UpdateProfile(
        name="probe", description="", supports_migrations=False,
        _build=lambda *_: [
            sync_step,
            UpdateStep(
                "resolve_ref", [sys.executable, "-c", "print('deadbeef')"],
                read_only=True,
            ),
        ],
    )

    raw = SQLiteBackend(str(tmp_path / "restart.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    try:
        await ensure_restart_requests_table(db)
        feature = RestartCoordinatorFeature(types.SimpleNamespace(
            did="did:test:host", agent_id="did:test:host",
            _raw_storage=types.SimpleNamespace(db=db), storage=None,
        ))
        feature._db = db
        # A whole-host request is the sovereign's to make.
        monkeypatch.setenv("KESTREL_API_KEY", "lock-report-test-key")
        with caller_context_scope(CallerContext.sovereign(
            identity="test-sovereign", credential="lock-report-test-key",
        )):
            row = await insert_request(
                db, requested_by_agent="did:test:host", reason="update",
            )
        req = types.SimpleNamespace(
            id=row.id, update_repo_path=str(tmp_path / "repo"),
            update_target_ref="main", update_allow_migrations=False,
        )

        update = await feature._run_update(req, profile)
        await record_update_log(db, req.id, json.dumps(update))
        logged = (await get_request(db, req.id)).update_log_dict()
    finally:
        await db.close()

    (sync,) = [s for s in logged["steps"] if s["step"] == "feature_sync"]
    return sync


async def test_every_finding_reaches_the_update_log_when_they_outrun_the_tail(
    tmp_path, monkeypatch,
):
    """Thirty-one findings, which alone are longer than the tail the log keeps
    of each stream: the first ones are cut from both tails, and every one is
    still in the request's ``update_log``, read back from the row."""
    from kestrel_sovereign.features.restart_coordinator.feature import (
        _OUTPUT_TAIL_CHARS,
    )

    downgraded = [f"kestrel-locked-{i:02d}" for i in range(30)]
    expected = [CLAWS_FINDING] + [
        f"{name} 1.0.0 installed, below the 2.0.0 uv.lock pins; forced by "
        f"{CLAWS} (requires {name}<2)"
        for name in downgraded
    ]
    assert sum(len(message) for message in expected) > _OUTPUT_TAIL_CHARS

    sync = await _logged_sync_step(
        tmp_path, monkeypatch, _fake_host_sync_step(tmp_path, downgraded=downgraded),
    )

    assert sync["ok"], sync
    # The tails lost the first findings...
    first = next(m for m in expected if m.startswith("kestrel-locked-00 "))
    assert first not in sync["stdout_tail"]
    assert first not in sync["stderr_tail"]
    # ...and the update log has every one of them, whole.
    assert sorted(f["message"] for f in sync["lock_findings"]) == sorted(expected)
    assert all(f["kind"] == "lock_downgrade" for f in sync["lock_findings"])


@pytest.mark.parametrize("lock, expected", [
    pytest.param(
        _LOCK_FROM_DOWNGRADED,
        {"kind": "lock_downgrade", "message": CLAWS_FINDING},
        id="below-the-lock",
    ),
    pytest.param(
        None,
        {"kind": "lock_not_compared", "message": "there is no uv.lock at"},
        id="no-lock",
    ),
])
async def test_an_empty_manifest_reaches_the_update_log(
    tmp_path, monkeypatch, lock, expected,
):
    """The coordinator keeps an empty manifest's findings too: a list, not
    the ``lock_findings: None`` of a sync that never compared."""
    sync = await _logged_sync_step(
        tmp_path, monkeypatch,
        _fake_host_sync_step(tmp_path, declares_claws=False, lock=lock),
    )

    assert sync["ok"], sync
    assert "declares no features" in sync["stdout_tail"]
    assert "lock_findings_unavailable" not in sync
    (finding,) = sync["lock_findings"]
    assert finding["kind"] == expected["kind"]
    assert expected["message"] in finding["message"]
    assert f"WARNING: {finding['message']}" in sync["stdout_tail"]


def test_a_damaged_dist_info_does_not_crash_the_scan_for_a_cause(
    monkeypatch, tmp_path,
):
    """Read from real metadata: METADATA that is not UTF-8 raises from
    ``importlib.metadata``, and the scan only runs on a host something already
    moved below the lock. One damaged package must not crash the sync."""
    site = tmp_path / "site"
    for dist, metadata in (
        ("good_pkg-1.0.dist-info", b"Metadata-Version: 2.1\nName: Good_Pkg\nVersion: 1.0\n"),
        ("badutf-1.0.dist-info", b"\xff\xfe\xfa not utf-8"),
        ("no_metadata-1.0.dist-info", None),
    ):
        (site / dist).mkdir(parents=True)
        if metadata is not None:
            (site / dist / "METADATA").write_bytes(metadata)
    monkeypatch.syspath_prepend(str(site))

    names = cli._installed_distribution_names()

    assert "good-pkg" in names
    assert len(names) == len(set(names))


# --- the comparison: core_lock.lock_downgrades --------------------------------


def _drift(installed, locked=SDK_LOCKED, name=SDK) -> cl.LockDrift:
    return cl.LockDrift(name=name, locked=locked, installed=installed)


@pytest.mark.parametrize("installed, below", [
    ("0.37.1", True),
    ("0.38.0rc1", True),
    ("0.39.0", False),
    ("0.38.1+local", False),
    ("not-a-version", False),
])
def test_below_is_a_pep440_comparison(installed, below):
    assert _drift(installed).below is below


def _downgrades(drift, requirements, **kw):
    return cl.lock_downgrades(
        drift,
        distributions=lambda: list(requirements),
        requires=lambda dist: requirements[dist],
        **kw,
    )


def test_every_requirement_excluding_the_locked_version_is_named():
    (finding,) = _downgrades([_drift("0.37.1")], {
        "kestrel-claws": [f"{SDK}<0.38"],
        "Kestrel_Feature_Voice": [f"{SDK}>=0.36,!=0.38.1"],
        "kestrel-feature-github": [f"{SDK}>=0.37.1,<0.39"],  # accepts the lock
        "kestrel-sovereign": [f"{SDK}[tracing]>=0.37.1,<0.39"],
        SDK: [f"{SDK}<0.38"],  # its own metadata is not a cause
    })

    assert finding.forced_by == (
        cl.LockForcer("kestrel-claws", f"requires {SDK}<0.38"),
        cl.LockForcer("kestrel-feature-voice", f"requires {SDK}!=0.38.1,>=0.36"),
    )


def test_a_requirement_whose_markers_do_not_hold_is_not_a_cause():
    (finding,) = _downgrades([_drift("0.37.1")], {
        "kestrel-claws": [
            f'{SDK}<0.38; sys_platform == "no-such-platform"',
            f'{SDK}<0.38; extra == "dev"',
            "this is not a requirement",
        ],
    })

    assert finding.forced_by == ()
    assert finding.cause == cl.UNKNOWN_CAUSE


def test_an_extra_the_host_asked_for_makes_its_requirement_a_cause():
    (finding,) = _downgrades(
        [_drift("0.37.1")],
        {"kestrel-claws": [f'{SDK}<0.38; extra == "local"']},
        extras={"Kestrel_Claws": ["local"]},
    )

    assert finding.forced_by == (
        cl.LockForcer("kestrel-claws", f"requires {SDK}<0.38"),
    )


def test_drift_above_or_beside_the_lock_is_not_a_downgrade_and_reads_nothing():
    def unread():
        raise AssertionError("nothing is below the lock: the venv is not read")

    assert cl.lock_downgrades(
        [_drift("0.39.0"), _drift("not-a-version")],
        distributions=unread,
        requires=unread,
    ) == ()


# --- the report update_then_restart reads the findings from -------------------


def test_the_sync_writes_every_finding_to_the_report_it_is_asked_for(
    monkeypatch, tmp_path, capsys,
):
    venv = _host(tmp_path, sdk="0.37.1", lock=_lock_text(["kestrel-locked-00"]))
    venv.installed["kestrel-locked-00"] = "1.0.0"
    use_fake_uv(monkeypatch, venv)
    report = tmp_path / "report" / "lock-findings.json"
    report.parent.mkdir()
    monkeypatch.setenv(cl.SYNC_LOCK_REPORT_ENV, str(report))

    rc = _sync(_manifest(tmp_path))

    out = capsys.readouterr().out
    assert rc == 0
    findings = cl.read_lock_findings_report(report)
    assert [f["message"] for f in findings] == [
        line.removeprefix("  WARNING: ")
        for line in out.splitlines() if line.startswith("  WARNING: ")
    ]
    assert {f["package"] for f in findings} == {SDK, "kestrel-locked-00"}
    assert not list(report.parent.glob("*.partial"))


def test_a_sync_with_nothing_below_the_lock_reports_an_empty_list(
    monkeypatch, tmp_path, capsys,
):
    """An empty report, not none: compared and clean, not never compared."""
    venv = _host(tmp_path, sdk=SDK_LOCKED, claws_requires=(f"{SDK}<0.39",))
    use_fake_uv(monkeypatch, venv)
    report = tmp_path / "lock-findings.json"
    monkeypatch.setenv(cl.SYNC_LOCK_REPORT_ENV, str(report))

    assert _sync(_manifest(tmp_path)) == 0
    assert "WARNING" not in capsys.readouterr().out
    assert cl.read_lock_findings_report(report) == []


def test_a_lock_that_will_not_read_is_reported_as_one_finding(monkeypatch, tmp_path):
    venv = _host(tmp_path, sdk="0.37.1", lock="version = 1\n")
    use_fake_uv(monkeypatch, venv)
    report = tmp_path / "lock-findings.json"
    monkeypatch.setenv(cl.SYNC_LOCK_REPORT_ENV, str(report))

    assert _sync(_manifest(tmp_path)) == 0
    (finding,) = cl.read_lock_findings_report(report)
    assert finding["kind"] == "lock_not_compared"
    assert "no [[package]] entries" in finding["reason"]
    assert finding["message"].startswith(
        "installed versions were not compared against core's uv.lock: "
    )


def test_a_report_that_cannot_be_written_is_said_and_fails_nothing(
    monkeypatch, tmp_path, capsys, caplog,
):
    venv = _host(tmp_path, sdk="0.37.1")
    use_fake_uv(monkeypatch, venv)
    unwritable = tmp_path / "no-such-dir" / "lock-findings.json"
    monkeypatch.setenv(cl.SYNC_LOCK_REPORT_ENV, str(unwritable))

    with caplog.at_level(logging.WARNING, logger=cli_features.__name__):
        rc = _sync(_manifest(tmp_path))

    out = capsys.readouterr().out
    assert rc == 0
    assert f"WARNING: {CLAWS_FINDING}" in out  # still printed
    assert f"could not write the lock report to {unwritable}" in out
    assert any(
        "could not write the lock report" in message for message in _warnings(caplog)
    )


def test_no_report_is_none_and_the_coordinator_says_why(tmp_path):
    from kestrel_sovereign.features.restart_coordinator.feature import (
        _lock_findings_fields,
    )

    report = tmp_path / "lock-findings.json"
    assert cl.read_lock_findings_report(report) is None
    fields = _lock_findings_fields(report)
    assert fields["lock_findings"] is None
    assert "wrote no lock report" in fields["lock_findings_unavailable"]


@pytest.mark.parametrize("raw, says", [
    pytest.param(b"not json", "is not JSON", id="not-json"),
    pytest.param(b"\xff\xfe", "is not JSON", id="not-utf8"),
    # Explicit ids: as their own ids, these two payloads printed 100 KB and
    # 1 MB lines that `gh run view --log` stops reading at (#3669).
    pytest.param(b"[" * 100_000, "is not JSON", id="nested-past-recursion-limit"),
    pytest.param(b'{"findings": {}}', "is not a lock report", id="findings-not-a-list"),
    pytest.param(
        b'{"findings": [{"kind": "lock_downgrade"}]}', "is not a lock report",
        id="finding-missing-fields",
    ),
    pytest.param(b'["findings"]', "is not a lock report", id="not-an-object"),
    pytest.param(
        b" " * (cl.LOCK_REPORT_MAX_BYTES + 1), "is larger than", id="over-size-cap",
    ),
])
def test_a_report_that_will_not_read_is_unavailable_never_empty(tmp_path, raw, says):
    from kestrel_sovereign.features.restart_coordinator.feature import (
        _lock_findings_fields,
    )

    report = tmp_path / "lock-findings.json"
    report.write_bytes(raw)

    with pytest.raises(cl.LockReportError, match=says):
        cl.read_lock_findings_report(report)
    fields = _lock_findings_fields(report)
    assert fields["lock_findings"] is None
    assert says in fields["lock_findings_unavailable"]


def test_a_report_round_trips_every_field(tmp_path):
    findings = (
        cl.LockDowngrade(
            name=SDK, locked=SDK_LOCKED, installed="0.37.1",
            forced_by=(
                cl.LockForcer(CLAWS, f"requires {SDK}<0.38"),
                cl.LockForcer("host manifest", "holds it on checkout /x"),
            ),
        ),
        cl.LockDowngrade(name="anthropic", locked="0.117.0", installed="0.100.0"),
        cl.LockNotCompared("there is no uv.lock at /y/uv.lock"),
    )
    report = tmp_path / "lock-findings.json"

    cl.write_lock_findings_report(report, findings)

    read = cl.read_lock_findings_report(report)
    assert read == [f.as_report() for f in findings]
    assert [f["message"] for f in read] == [f.describe() for f in findings]
    assert read[1]["cause"] == cl.UNKNOWN_CAUSE


async def test_only_a_reporting_step_is_given_a_report_file(tmp_path):
    """Every other step runs in this process's own environment, untouched."""
    from kestrel_sovereign.features.restart_coordinator.feature import (
        RestartCoordinatorFeature,
    )
    from kestrel_sovereign.features.restart_coordinator.update_profiles import (
        UpdateStep,
    )

    show = [
        sys.executable, "-c",
        f"import os; print(os.environ.get({cl.SYNC_LOCK_REPORT_ENV!r}, 'unset'))",
    ]
    feature = RestartCoordinatorFeature(types.SimpleNamespace(did="did:test:host"))

    plain = await feature._run_update_step(UpdateStep("install", show))
    reporting = await feature._run_update_step(
        UpdateStep("feature_sync", show, reports_lock_findings=True),
    )

    assert plain["stdout_tail"] == "unset"
    assert "lock_findings" not in plain
    assert reporting["stdout_tail"].endswith("lock-findings.json")
    assert not Path(reporting["stdout_tail"]).parent.exists()  # cleaned up
    assert reporting["lock_findings"] is None  # it wrote none
