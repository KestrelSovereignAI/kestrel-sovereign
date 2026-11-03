"""The offline migration gate on deploys (#3832).

On 2026-10-24 an agent's ``update_then_restart`` installed Core 0.54 (native
A2A) and restarted. Startup refused every agent whose store still held draft
A2A data, and the fleet was down for 31 hours. These tests pin the gate every
deploy path now asks first:

* a release declares the offline migrations it requires in
  ``kestrel_sovereign/data/offline_migrations.toml``; the declaration is read
  from the incoming revision with git, before anything is installed;
* each agent database is asked, read-only and in a fresh interpreter, whether
  each declared migration is done. The native A2A probe is the startup
  preflight itself. A declaration this code has no probe for, or a database
  that cannot be read, blocks: nothing was learned;
* ``kestrel update`` refuses before it pulls, installs or restarts, naming the
  agents, the databases and the runbook, unless
  ``--allow-pending-offline-migration``; with ``--no-restart`` only while an
  affected agent runs. ``kestrel restart`` refuses before terminating anything.

The restart coordinator's use of the same gate is pinned in
``test_restart_coordinator.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign import cli, cli_lifecycle, offline_migrations
from kestrel_sovereign.constitution_adoption import ConstitutionAdoptionCheck
from kestrel_sovereign.offline_migrations import (
    COMPLETE,
    MANIFEST_RELPATH,
    OFFLINE_MIGRATION_REQUIRED,
    OVERRIDE_FLAG,
    PENDING,
    POSTGRES_DATABASE,
    PROBES,
    UNKNOWN,
    UNVERIFIED,
    OfflineMigrationError,
    check_offline_migrations,
    check_record,
    check_revision_offline_migrations,
    manifest_at,
    packaged_manifest,
    parse_manifest,
    refusal_reason,
)
from tests.utils.constitution_anchor import (
    commit_constitution,
    git,
    origin_and_clone,
    seed_anchored_agents,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = "docs/architecture/a2a/OFFLINE_CUTOVER.md"
NATIVE = "native-a2a-cutover"


def _manifest(*entries: dict) -> bytes:
    lines = ["version = 1", ""]
    for entry in entries:
        lines.append("[[migration]]")
        lines += [f"{key} = {json.dumps(value)}" for key, value in entry.items()]
        lines.append("")
    return "\n".join(lines).encode()


def _entry(mid: str = NATIVE, runbook: str = RUNBOOK) -> dict:
    return {"id": mid, "summary": f"{mid} summary", "runbook": runbook}


@pytest.fixture(autouse=True)
def _no_real_process_lifecycle(monkeypatch):
    """Nothing here may start, stop or signal a real process.

    A reproduction of this change once reached the real host lifecycle and
    started a second server on a live host. The process launcher and both
    lifecycle commands fail the test unless the test replaces them; ``kestrel
    update``'s continuation already runs in this process (``tests/conftest.py``).
    """

    def refuse(*_args, **_kwargs):
        pytest.fail("reached the real process lifecycle")

    class _Refusing(type):
        def __getattr__(cls, _name):
            refuse()

        def __call__(cls, *args, **kwargs):
            refuse()

    monkeypatch.setattr(
        cli_lifecycle, "ProcessManager", _Refusing("ProcessManager", (), {})
    )
    monkeypatch.setattr(cli, "cmd_terminate", refuse)
    monkeypatch.setattr(cli, "cmd_start", refuse)


@pytest.fixture(autouse=True)
def _hermetic_backend(monkeypatch):
    """The agents read their own SQLite files unless a test says otherwise."""
    for name in ("KESTREL_DB_BACKEND", "KESTREL_DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def project(tmp_path):
    """Emma and Kite, each with an agent database holding no A2A data."""
    path = tmp_path / "project"
    path.mkdir()
    seed_anchored_agents(path, {"Emma": "e" * 64, "Kite": "k" * 64})
    return path


def _db(project: Path, name: str) -> Path:
    return project / "agent_data" / name.lower() / "kestrel_prime.db"


def _leave_draft_a2a_data(db_path: Path) -> None:
    """The store a pre-native release left: an undrained draft outbound task."""
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("CREATE TABLE a2a_outbound_tasks (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO a2a_outbound_tasks VALUES ('draft-1')")
        conn.commit()


def _snapshot(directory: Path) -> dict:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
    }


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def test_the_packaged_manifest_declares_the_native_a2a_cutover():
    declared = parse_manifest(packaged_manifest(), source="the packaged manifest")

    assert [m.id for m in declared] == [NATIVE]
    assert declared[0].runbook == RUNBOOK


def test_every_declared_migration_has_a_probe_and_every_probe_is_declared():
    """A declaration with no probe blocks every host; a probe nobody declares is dead."""
    declared = parse_manifest(packaged_manifest(), source="the packaged manifest")

    assert {m.id for m in declared} == set(PROBES)


def test_every_declared_runbook_is_in_the_checkout():
    for migration in parse_manifest(packaged_manifest(), source="the packaged manifest"):
        assert (REPO_ROOT / migration.runbook).is_file(), migration.runbook


def test_the_packaged_manifest_is_where_a_revision_carries_it():
    assert (REPO_ROOT / MANIFEST_RELPATH).read_bytes() == packaged_manifest()


@pytest.mark.parametrize(
    ("content", "why"),
    [
        (b"version = 2\n", "format version 2"),
        (b"version = true\n", "format version True"),
        (b"[[migration]]\n", "format version None"),
        (b"version = 1\nrequires = []\n", "unknown fields ['requires']"),
        (b"version = 1\nmigration = 'x'\n", "not an array of tables"),
        (
            _manifest({**_entry(), "release": "0.54.0"}),
            "exactly the fields",
        ),
        (_manifest({"id": NATIVE, "summary": "s"}), "exactly the fields"),
        (_manifest({**_entry(), "summary": " "}), "non-empty string"),
        (_manifest(_entry("Native A2A")), "invalid migration id"),
        (_manifest(_entry(), _entry()), "declared twice"),
        (_manifest(_entry(runbook="/etc/passwd")), "not a path inside"),
        (_manifest(_entry(runbook="docs/../../x.md")), "not a path inside"),
        (b"version = \n", "cannot read"),
        (b"\xff\xfe", "cannot read"),
    ],
)
def test_a_manifest_this_code_cannot_read_whole_is_refused(content, why):
    with pytest.raises(OfflineMigrationError, match="cannot read the offline") as caught:
        parse_manifest(content, source="the incoming revision")
    assert why in str(caught.value)
    assert "the incoming revision" in str(caught.value)


def test_a_manifest_declaring_nothing_is_read_as_nothing():
    assert parse_manifest(b"version = 1\n", source="x") == ()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    (path / "README.md").write_text("before declarations\n")
    git(path, "add", "README.md")
    git(path, "commit", "-q", "-m", "a release that predates #3832")
    return path


def _commit_manifest(repo: Path, content: bytes) -> str:
    path = repo / MANIFEST_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    git(repo, "add", MANIFEST_RELPATH)
    git(repo, "commit", "-q", "-m", "declare an offline migration")
    return git(repo, "rev-parse", "HEAD").strip()


def test_manifest_at_reads_what_a_revision_declares(repo):
    before = git(repo, "rev-parse", "HEAD").strip()
    after = _commit_manifest(repo, _manifest(_entry()))
    # The working tree moves on; the revision is what is read.
    (repo / MANIFEST_RELPATH).write_bytes(b"not committed")

    assert manifest_at(repo, after) == _manifest(_entry())
    assert manifest_at(repo, "HEAD") == _manifest(_entry())
    assert manifest_at(repo, before) is None


def test_manifest_at_refuses_a_revision_it_cannot_resolve(repo):
    with pytest.raises(OfflineMigrationError, match="cannot resolve"):
        manifest_at(repo, "no-such-ref")


def test_manifest_at_refuses_a_manifest_that_is_not_a_file(repo):
    if sys.platform == "win32":
        pytest.skip("symbolic links")
    path = repo / MANIFEST_RELPATH
    path.parent.mkdir(parents=True)
    path.symlink_to("elsewhere.toml")
    git(repo, "add", MANIFEST_RELPATH)
    git(repo, "commit", "-q", "-m", "a link")

    with pytest.raises(OfflineMigrationError, match="other than a file"):
        manifest_at(repo, "HEAD")


# ---------------------------------------------------------------------------
# The check, answered by a fresh interpreter
# ---------------------------------------------------------------------------


def _check(project, *entries, **kwargs):
    content = _manifest(*(entries or (_entry(),)))
    return check_offline_migrations(project, manifest=lambda: content, **kwargs)


def test_draft_a2a_data_is_a_pending_migration_named_with_its_agent(project):
    _leave_draft_a2a_data(_db(project, "Emma"))

    check = _check(project, code_label="the incoming revision")

    assert not check.safe
    assert check.blocking_agents == ("Emma",)
    by_agent = {f.agents: f for f in check.findings}
    emma, kite = by_agent[("Emma",)], by_agent[("Kite",)]
    assert emma.status == PENDING
    assert emma.database == str(_db(project, "Emma").resolve())
    # The detail is the startup preflight's own refusal.
    assert "Draft A2A data remains in a2a_outbound_tasks" in emma.detail
    assert kite.status == COMPLETE
    reason = refusal_reason(check)
    for text in ("Emma", RUNBOOK, "the incoming revision", "an agent cannot"):
        assert text in reason


def test_unversioned_task_history_is_pending_too(project):
    with closing(sqlite3.connect(str(_db(project, "Kite")))) as conn:
        conn.execute("CREATE TABLE a2a_tasks (id TEXT)")
        conn.execute("INSERT INTO a2a_tasks VALUES ('t')")
        conn.commit()

    check = _check(project)

    assert check.blocking_agents == ("Kite",)
    assert "Unversioned A2A task data" in check.blocking[0].detail


def test_a_migrated_store_is_complete(project):
    """The protocol revision the cutover publishes last is what startup accepts."""
    for name in ("Emma", "Kite"):
        with closing(sqlite3.connect(str(_db(project, name)))) as conn:
            conn.execute("CREATE TABLE a2a_tasks (id TEXT)")
            conn.execute("INSERT INTO a2a_tasks VALUES ('t')")
            conn.execute(
                "CREATE TABLE a2a_protocol_schema "
                "(singleton INTEGER PRIMARY KEY, revision INTEGER)"
            )
            conn.execute("INSERT INTO a2a_protocol_schema VALUES (1, 1)")
            conn.commit()

    check = _check(project)

    assert check.safe
    assert {f.status for f in check.findings} == {COMPLETE}
    assert check_record(check)["result"] == "passed"


def test_the_check_reads_without_changing_anything(project):
    _leave_draft_a2a_data(_db(project, "Emma"))
    directory = _db(project, "Emma").parent
    before = _snapshot(directory)

    assert not _check(project).safe

    assert _snapshot(directory) == before


def test_an_agent_with_no_database_yet_has_nothing_to_migrate(project):
    _db(project, "Kite").unlink()

    check = _check(project, agent_names=["Kite"])

    assert check.safe
    assert check.findings[0].detail == "no database yet"
    assert not _db(project, "Kite").exists()


def test_a_database_that_cannot_be_read_blocks(project):
    _db(project, "Emma").write_bytes(b"this is not a sqlite database" * 100)

    check = _check(project)

    assert check.blocking_agents == ("Emma",)
    (finding,) = check.blocking
    assert finding.status == UNVERIFIED
    assert "could not be read" in refusal_reason(check)


def test_a_migration_this_code_cannot_check_blocks_every_agent(project):
    """A release newer than the code judging it: only an operator crosses it."""
    check = _check(project, _entry(), _entry("a-future-migration"))

    (unknown,) = check.blocking
    assert unknown.status == UNKNOWN
    assert unknown.migration == "a-future-migration"
    assert unknown.agents == ("Emma", "Kite")
    assert "no check for it" in refusal_reason(check)


def test_agent_names_limit_the_check(project):
    _leave_draft_a2a_data(_db(project, "Emma"))

    assert _check(project, agent_names=["Kite"]).safe
    assert not _check(project, agent_names=["Emma"]).safe


def test_no_selected_agent_reads_no_declarations_and_starts_nothing(
    tmp_path, monkeypatch
):
    def unreachable(*args, **kwargs):
        raise AssertionError("nothing to judge")

    monkeypatch.setattr(offline_migrations.subprocess, "run", unreachable)
    empty = tmp_path / "empty"
    empty.mkdir()

    check = check_offline_migrations(empty, manifest=unreachable)

    assert check.safe and check.findings == ()


def test_a_named_agent_the_registry_lacks_is_refused_not_passed(
    project, monkeypatch
):
    """An empty pass would let a launcher that resolves the name elsewhere
    restart a database no gate read (#3832)."""
    def unreachable(*args, **kwargs):
        raise AssertionError("nothing to judge")

    monkeypatch.setattr(offline_migrations.subprocess, "run", unreachable)

    with pytest.raises(OfflineMigrationError, match="no local agent named Nobody"):
        check_offline_migrations(
            project, agent_names=["Nobody"], manifest=unreachable
        )


def test_a_release_declaring_nothing_starts_nothing(project, monkeypatch):
    def unreachable(*args, **kwargs):
        raise AssertionError("nothing to judge")

    monkeypatch.setattr(offline_migrations.subprocess, "run", unreachable)

    assert check_offline_migrations(project, manifest=lambda: None).safe
    assert check_offline_migrations(project, manifest=lambda: b"version = 1\n").safe


def test_the_installed_code_judges_its_own_declarations(project):
    """No manifest given: the fresh interpreter reads its packaged one."""
    _leave_draft_a2a_data(_db(project, "Emma"))

    check = check_offline_migrations(project)

    assert {f.migration for f in check.findings} == {NATIVE}
    assert check.blocking_agents == ("Emma",)


def test_a_revision_is_judged_by_what_it_declares(project, repo):
    _leave_draft_a2a_data(_db(project, "Emma"))
    before = git(repo, "rev-parse", "HEAD").strip()

    assert check_revision_offline_migrations(
        project, repo, before, code_label="old"
    ).safe

    after = _commit_manifest(repo, _manifest(_entry()))
    check = check_revision_offline_migrations(
        project, repo, after, code_label="new"
    )
    assert check.blocking_agents == ("Emma",)
    assert check.code_label == "new"


def test_a_postgres_runtime_is_one_database_and_its_dsn_stays_out(
    project, monkeypatch
):
    """Every agent shares it; a failed connect names no credential."""
    secret = "s3cret-password-3832"
    dsn = f"postgresql://kestrel:{secret}@127.0.0.1:1/kestrel"
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", dsn)

    check = _check(project)

    (finding,) = check.findings
    assert finding.status == UNVERIFIED
    assert finding.database == POSTGRES_DATABASE
    assert finding.agents == ("Emma", "Kite")
    assert "kestrel doctor" in finding.detail
    rendered = json.dumps(check_record(check)) + refusal_reason(check)
    assert secret not in rendered and dsn not in rendered


# ---------------------------------------------------------------------------
# Reading the fresh interpreter's answer
# ---------------------------------------------------------------------------


def _answer(payload, *, returncode=0, prefix=True, stderr=b""):
    line = json.dumps(payload)
    if prefix:
        line = offline_migrations._FRESH_CHECK_RESULT_PREFIX + line
    return subprocess.CompletedProcess(
        args=[], returncode=returncode,
        stdout=b"import noise\n" + line.encode() + b"\n", stderr=stderr,
    )


def _finding(**overrides):
    base = {
        "migration": NATIVE, "status": COMPLETE, "summary": "s",
        "runbook": RUNBOOK, "database": "db", "agents": ["Emma"], "detail": "",
    }
    base.update(overrides)
    return base


def _read(result, selected=("Emma",), declared=None, env=None):
    return offline_migrations._answered_findings(
        result, list(selected), declared, env or {}
    )


def test_a_well_formed_answer_is_read():
    (finding,) = _read(_answer({"protocol": 1, "findings": [_finding()]}))

    assert finding.status == COMPLETE and finding.agents == ("Emma",)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (_answer({"protocol": 2, "findings": []}), "unknown protocol"),
        (_answer({"protocol": 1}), "cannot read"),
        (_answer({"protocol": 1, "findings": [_finding(status="fine")]}),
         "unknown finding status"),
        (_answer({"protocol": 1, "findings": [_finding(agents="Emma")]}),
         "not a list"),
        (_answer({"protocol": 1, "findings": [_finding(agents=["Mallory"])]}),
         "asked about"),
        (_answer({"protocol": 1, "findings": [_finding(detail=3)]}),
         "not a string"),
        (_answer({"protocol": 1, "findings": [_finding(extra=1)]}), "cannot read"),
        (_answer({"protocol": 1, "error": "registry is broken"}),
         "registry is broken"),
        (_answer({"protocol": 1, "findings": []}, prefix=False), "could not check"),
        (_answer({"protocol": 1, "findings": []}, returncode=1, stderr=b"boom"),
         "boom"),
    ],
)
def test_an_answer_this_process_cannot_read_is_refused(result, message):
    with pytest.raises(OfflineMigrationError, match=message):
        _read(result)


def test_an_answer_that_skips_an_agent_or_a_migration_is_refused():
    declared = parse_manifest(_manifest(_entry(), _entry("other")), source="x")

    with pytest.raises(OfflineMigrationError, match="every agent"):
        _read(
            _answer({"protocol": 1, "findings": [_finding()]}),
            selected=("Emma", "Kite"),
        )
    with pytest.raises(OfflineMigrationError, match="asked about"):
        _read(_answer({"protocol": 1, "findings": [_finding()]}), declared=declared)


def test_an_answer_never_carries_the_dsn():
    dsn = "postgresql://kestrel:pw3832@db/kestrel"
    env = {"KESTREL_DATABASE_URL": dsn}

    (finding,) = _read(
        _answer({"protocol": 1, "findings": [_finding(detail=f"at {dsn}")]}),
        env=env,
    )
    assert dsn not in finding.detail
    with pytest.raises(OfflineMigrationError) as caught:
        _read(_answer({"protocol": 1, "error": f"cannot reach {dsn}"}), env=env)
    assert dsn not in str(caught.value)
    with pytest.raises(OfflineMigrationError) as caught:
        _read(_answer({}, returncode=1, stderr=f"Traceback {dsn}".encode()), env=env)
    assert dsn not in str(caught.value)


# ---------------------------------------------------------------------------
# kestrel restart
# ---------------------------------------------------------------------------


def _restart_args(**overrides):
    base = {"name": None, "force": False, "startup_timeout": 30}
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def lifecycle_calls(project, monkeypatch):
    """Point the CLI at ``project``, record terminate/start, pass the constitution."""
    calls = []
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(
        cli, "cmd_terminate", lambda args: calls.append("terminate") or 0
    )
    monkeypatch.setattr(cli, "cmd_start", lambda args: calls.append("start") or 0)
    # The constitution gate has its own tests; here it always passes.
    monkeypatch.setattr(
        cli_lifecycle, "check_installed_constitution_adoption",
        lambda *a, **k: ConstitutionAdoptionCheck(),
    )
    return calls


def test_restart_refuses_a_pending_migration_before_terminating_anything(
    project, lifecycle_calls, capsys
):
    _leave_draft_a2a_data(_db(project, "Emma"))

    rc = cli.cmd_restart(_restart_args())

    assert rc == OFFLINE_MIGRATION_REQUIRED
    assert lifecycle_calls == []
    err = capsys.readouterr().err
    for text in ("Emma", NATIVE, RUNBOOK, "no agent was stopped", OVERRIDE_FLAG):
        assert text in err


def test_restart_proceeds_when_nothing_is_pending(project, lifecycle_calls, capsys):
    assert cli.cmd_restart(_restart_args()) == 0
    assert lifecycle_calls == ["terminate", "start"]
    assert f"none pending for what the installed code requires ({NATIVE})" in (
        capsys.readouterr().out
    )


def test_the_override_restarts_deliberately(project, lifecycle_calls, capsys):
    _leave_draft_a2a_data(_db(project, "Emma"))

    rc = cli.cmd_restart(_restart_args(allow_pending_offline_migration=True))

    assert rc == 0
    assert lifecycle_calls == ["terminate", "start"]
    assert "proceeding deliberately" in capsys.readouterr().err


def test_a_named_restart_judges_its_agent_unless_the_package_changed(
    project, lifecycle_calls
):
    _leave_draft_a2a_data(_db(project, "Emma"))

    assert cli.cmd_restart(_restart_args(name="Kite")) == 0
    lifecycle_calls.clear()
    rc = cli.cmd_restart(
        _restart_args(name="Kite", constitution_check_all_agents=True)
    )
    assert rc == OFFLINE_MIGRATION_REQUIRED
    assert lifecycle_calls == []


def test_the_parser_offers_the_override_on_restart_and_update():
    parser = cli.build_parser()
    for command in (["restart"], ["update"]):
        assert parser.parse_args(command).allow_pending_offline_migration is False
        assert parser.parse_args(
            command + [OVERRIDE_FLAG]
        ).allow_pending_offline_migration is True


def test_the_update_continuation_keeps_the_override():
    """A fresh interpreter that lost the flag would run a different update."""
    args = cli.build_parser().parse_args(["update", OVERRIDE_FLAG])

    assert OVERRIDE_FLAG in cli_lifecycle._update_argv(args)


# ---------------------------------------------------------------------------
# kestrel update
# ---------------------------------------------------------------------------


def _update_args(**overrides):
    base = {
        "name": None, "pull": True, "install": True, "features": False,
        "restart": True, "allow_dirty": False, "no_deps": False,
        "continue_on_error": False, "dry_run": False, "manifest": None,
        "force": False, "uv_sync": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def upstream(tmp_path, project, monkeypatch):
    """A checkout whose upstream now declares the native A2A cutover.

    The step helpers that would change the host are recorded instead of run;
    the gate's own git reads are real, and so is the pull, which moves only
    this test's clone. The constitution gate always passes.
    """
    origin, clone = origin_and_clone(tmp_path, b"# Kestrel Constitution\n")
    _commit_manifest(origin, packaged_manifest())
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: clone)
    monkeypatch.setattr(
        cli_lifecycle, "check_constitution_adoption",
        lambda *a, **k: ConstitutionAdoptionCheck(),
    )
    steps = []
    real_pull = cli._run_git_fast_forward
    monkeypatch.setattr(
        cli,
        "_run_git_fast_forward",
        lambda c, revision: steps.append("pull") or real_pull(c, revision),
    )
    monkeypatch.setattr(
        cli,
        "_run_uv_pip_install_editable",
        lambda *a, **kw: steps.append("install") or (0, ""),
    )
    monkeypatch.setattr(
        cli, "cmd_restart", lambda args: steps.append(("restart", args)) or 0
    )
    return origin, clone, steps


def _head(clone: Path) -> str:
    return git(clone, "rev-parse", "HEAD").strip()


def test_update_refuses_a_pending_migration_before_changing_anything(
    project, upstream, capsys
):
    """The incident: native A2A over undrained draft data."""
    _, clone, steps = upstream
    _leave_draft_a2a_data(_db(project, "Emma"))
    head = _head(clone)

    rc = cli.cmd_update(_update_args())

    assert rc == OFFLINE_MIGRATION_REQUIRED
    assert steps == []
    assert _head(clone) == head
    assert not (clone / MANIFEST_RELPATH).exists()
    captured = capsys.readouterr()
    assert "checking incoming revision" in captured.out
    for text in (
        "Emma", NATIVE, RUNBOOK, "Draft A2A data remains",
        "nothing was pulled, installed or restarted", OVERRIDE_FLAG,
        "`kestrel update --no-restart`",
    ):
        assert text in captured.err


def test_update_proceeds_onto_a_release_whose_migration_is_done(
    project, upstream, capsys
):
    _, clone, steps = upstream

    rc = cli.cmd_update(_update_args())

    assert rc == 0
    assert steps[:2] == ["pull", "install"]
    assert (clone / MANIFEST_RELPATH).exists()
    assert "none pending for what the revision" in capsys.readouterr().out


def test_update_override_proceeds_and_reaches_the_restart(project, upstream):
    _, _, steps = upstream
    _leave_draft_a2a_data(_db(project, "Emma"))

    rc = cli.cmd_update(_update_args(allow_pending_offline_migration=True))

    assert rc == 0
    assert steps[:2] == ["pull", "install"]
    (_, restart_args), = [s for s in steps if isinstance(s, tuple)]
    assert restart_args.allow_pending_offline_migration is True
    assert restart_args.constitution_check_all_agents is True


def test_update_reports_a_restart_the_installed_code_refuses(
    project, upstream, monkeypatch, capsys
):
    _, _, steps = upstream
    monkeypatch.setattr(
        cli, "cmd_restart",
        lambda args: steps.append("restart") or OFFLINE_MIGRATION_REQUIRED,
    )

    rc = cli.cmd_update(_update_args())

    assert rc == OFFLINE_MIGRATION_REQUIRED
    assert steps == ["pull", "install", "restart"]
    assert (
        "REFUSED by the offline migration check above; no agent was stopped, "
        "but the update itself is installed."
    ) in capsys.readouterr().err


def test_update_onto_a_release_that_declares_nothing_passes(project, upstream):
    origin, _, steps = upstream
    git(origin, "reset", "-q", "--hard", "HEAD~1")
    commit_constitution(origin, b"# amended\n", "a release with no declarations")
    _leave_draft_a2a_data(_db(project, "Emma"))

    assert cli.cmd_update(_update_args()) == 0
    assert steps[:2] == ["pull", "install"]


def test_update_without_a_restart_installs_while_nothing_runs(
    project, upstream, capsys
):
    """Terminate, install, migrate, start: the offline migration's own path."""
    _, _, steps = upstream
    _leave_draft_a2a_data(_db(project, "Emma"))

    rc = cli.cmd_update(_update_args(restart=False))

    assert rc == 0
    assert steps == ["pull", "install"]
    assert "none of these agents is running" in capsys.readouterr().err


def test_update_without_a_restart_refuses_while_an_affected_agent_runs(
    project, upstream, capsys
):
    from kestrel_sovereign.multi_agent.process_manager import ProcessManager

    _, _, steps = upstream
    _leave_draft_a2a_data(_db(project, "Emma"))
    host_pid = project / "logs" / ".host.pid"
    host_pid.parent.mkdir()
    # This test process stands in for the live host serving Emma.
    ProcessManager.write_pid(host_pid, os.getpid(), root=project)

    rc = cli.cmd_update(_update_args(restart=False))

    assert rc == OFFLINE_MIGRATION_REQUIRED
    assert steps == []
    err = capsys.readouterr().err
    assert "Emma is running (host PID" in err
    assert "the migration needs its database to itself" in err


def test_update_refuses_when_the_incoming_revision_cannot_be_read(
    project, upstream, monkeypatch, capsys
):
    _, _, steps = upstream
    monkeypatch.setattr(cli, "_run_git_fetch", lambda _: (1, "network is down"))

    assert cli.cmd_update(_update_args()) == OFFLINE_MIGRATION_REQUIRED
    assert steps == []
    assert "network is down" in capsys.readouterr().err


def test_a_host_with_no_agents_does_no_git_work_for_the_gate(
    tmp_path, monkeypatch
):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / ".git").mkdir()
    monkeypatch.setattr(cli, "_get_project_dir", lambda: empty)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: empty)
    git_work = []
    monkeypatch.setattr(
        cli_lifecycle, "manifest_at", lambda *a: pytest.fail("no agent to judge")
    )
    monkeypatch.setattr(
        cli, "_run_git_fetch", lambda _: git_work.append("fetch") or (0, "")
    )
    monkeypatch.setattr(
        cli,
        "_resolve_pull_target",
        lambda _: cli_lifecycle._PullTarget(
            upstream="a" * 40, revision=None, head="a" * 40
        ),
    )
    monkeypatch.setattr(cli, "_git_head", lambda _: "a" * 40)
    monkeypatch.setattr(
        cli, "_run_git_fast_forward", lambda *_: git_work.append("pull") or (0, "")
    )
    monkeypatch.setattr(cli, "_git_working_tree_dirty", lambda _: (False, ""))
    monkeypatch.setattr(cli, "_run_uv_pip_install_editable", lambda *a, **k: (0, ""))

    assert cli.cmd_update(_update_args(restart=False)) == 0
    # The pull's own fetch, once; nothing read a revision for the gate.
    assert git_work == ["fetch", "pull"]


def test_the_pull_bringing_nothing_judges_the_checkout_on_disk(
    project, upstream, capsys
):
    _, clone, steps = upstream
    git(clone, "pull", "-q", "--ff-only")
    _leave_draft_a2a_data(_db(project, "Emma"))

    rc = cli.cmd_update(_update_args())

    assert rc == OFFLINE_MIGRATION_REQUIRED
    assert steps == []
    assert "the pull brings no new revision" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# What the gate judges is what the restart boots
# ---------------------------------------------------------------------------


def _register_only(path: Path, name: str, port: int) -> None:
    """Write a registry at ``path`` naming one agent, its data relative to the host."""
    from kestrel_sovereign.multi_agent.config import (
        HostConfig,
        LocalAgentConfig,
        MultiAgentConfig,
    )

    MultiAgentConfig(
        host=HostConfig(),
        agents={
            name: LocalAgentConfig(data_dir=f"agent_data/{name.lower()}", port=port)
        },
    ).save(path)


@pytest.fixture
def external_registry(tmp_path, project, monkeypatch):
    """A host whose registry lives outside the project and names only Emma.

    The project's own ``multi_agent.toml`` names Kite alone, so a gate that
    read it instead of the host's selection would judge the wrong agent.
    """
    _register_only(project / "multi_agent.toml", "Kite", 8802)
    registry = tmp_path / "fleet" / "fleet.toml"
    registry.parent.mkdir()
    # Relative data_dir: the host resolves it against its directory, the project.
    _register_only(registry, "Emma", 8801)
    monkeypatch.setenv("KESTREL_MULTI_AGENT_CONFIG", str(registry))
    return registry


def test_the_gate_judges_the_registry_the_host_loads(project, external_registry):
    """The reviewer's reproduction: an external registry naming Emma."""
    from kestrel_sovereign.multi_agent.config import load_host_registry
    from kestrel_sovereign.paths import spawned_agent_env

    _leave_draft_a2a_data(_db(project, "Emma"))

    host = load_host_registry(spawned_agent_env(project), project)
    assert list(host.get_local_agents()) == ["Emma"]

    check = _check(project, _entry())

    assert not check.safe
    (finding,) = check.blocking
    assert finding.agents == ("Emma",)
    assert finding.database == str(_db(project, "Emma"))
    assert "Draft A2A data remains" in finding.detail


def test_liveness_reads_the_registry_the_host_loads(project, external_registry):
    """An --no-restart install must see the host serving the external agent."""
    from kestrel_sovereign.multi_agent.process_manager import ProcessManager

    host_pid = project / "logs" / ".host.pid"
    host_pid.parent.mkdir()
    ProcessManager.write_pid(host_pid, os.getpid(), root=project)

    assert list(cli_lifecycle._running_agents(project, ["Emma"])) == ["Emma"]
    assert list(cli_lifecycle._running_agents(project, None)) == ["Emma"]


def test_a_missing_external_registry_falls_back_as_the_host_does(
    tmp_path, project, monkeypatch
):
    """The server loads the project's file when the selected one is absent.

    Not the agents auto-discovery would find: Emma's data is still on disk.
    """
    _register_only(project / "multi_agent.toml", "Kite", 8802)
    monkeypatch.setenv("KESTREL_MULTI_AGENT_CONFIG", str(tmp_path / "absent.toml"))
    _leave_draft_a2a_data(_db(project, "Emma"))
    _leave_draft_a2a_data(_db(project, "Kite"))

    check = _check(project, _entry())

    assert check.blocking_agents == ("Kite",)


def _recording_launcher(monkeypatch, project: Path) -> list:
    """The real ``kestrel start``/``terminate`` over a launcher that runs nothing.

    Returns the record of each agent they start or terminate. The commands
    resolve their agents for real; only the process work is replaced. The
    launcher demands the registry the agent was found in, as the real one
    needs it: without it ``ProcessManager.start_agent`` reloads the project's
    default file (#3832).
    """
    calls: list = []

    class Launcher:
        def __init__(self, _project_dir):
            self._agents = {}

        def start_agent(self, name, cfg, _bind, *, roster, **_kwargs):
            assert roster.get_local_agents()[name] == cfg
            calls.append(("start", name, cfg.port))

        def wait_for_health(self, _port, timeout=None):
            return True

        def register_agent(self, name, _cfg):
            self._agents[name] = SimpleNamespace(pid=4242, stop_failure="")

        def terminate_agent(self, name):
            calls.append(("terminate", name))
            return True

        @staticmethod
        def is_port_in_use(_port, _bind):
            return False

    monkeypatch.setattr(cli_lifecycle, "ProcessManager", Launcher)
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_maybe_first_run_setup", lambda _project: None)
    monkeypatch.setattr(cli, "cmd_terminate", cli_lifecycle.cmd_terminate)
    monkeypatch.setattr(cli, "cmd_start", cli_lifecycle.cmd_start)
    return calls


def test_start_and_terminate_name_agents_from_the_registry_the_gates_judge(
    project, external_registry, monkeypatch
):
    """One registry for the gates and the launcher: the host's (#3832)."""
    calls = _recording_launcher(monkeypatch, project)

    assert cli.cmd_terminate(_restart_args(name="Emma")) == 0
    assert cli.cmd_start(_restart_args(name="Emma")) == 0
    assert calls == [("terminate", "Emma"), ("start", "Emma", 8801)]
    # Kite is only in the project's default file, which the host does not load.
    assert cli.cmd_terminate(_restart_args(name="Kite")) == 1
    assert cli.cmd_start(_restart_args(name="Kite")) == 1
    assert calls == [("terminate", "Emma"), ("start", "Emma", 8801)]


def test_a_named_restart_of_a_hosted_agent_judges_its_database(
    project, external_registry, monkeypatch
):
    calls = _recording_launcher(monkeypatch, project)
    monkeypatch.setattr(
        cli_lifecycle, "check_installed_constitution_adoption",
        lambda *a, **k: ConstitutionAdoptionCheck(),
    )
    _leave_draft_a2a_data(_db(project, "Emma"))

    assert cli.cmd_restart(_restart_args(name="Emma")) == OFFLINE_MIGRATION_REQUIRED
    assert calls == []


def test_a_named_restart_cannot_stop_and_start_an_agent_no_gate_read(
    tmp_path, project, monkeypatch, capsys
):
    """The reviewer's reproduction: Emma is named only by the project's
    default file, the host loads an external registry. The gates used to judge
    nothing for Emma while terminate and start found her in the other file."""
    from kestrel_sovereign.constitution_adoption import (
        CONSTITUTION_ADOPTION_REQUIRED,
    )

    _register_only(project / "multi_agent.toml", "Emma", 8801)
    registry = tmp_path / "fleet" / "fleet.toml"
    registry.parent.mkdir()
    _register_only(registry, "Kite", 8802)
    monkeypatch.setenv("KESTREL_MULTI_AGENT_CONFIG", str(registry))
    _leave_draft_a2a_data(_db(project, "Emma"))
    calls = _recording_launcher(monkeypatch, project)

    # Each gate refuses an agent it cannot find, rather than judging nothing.
    assert cli.cmd_restart(_restart_args(name="Emma")) == CONSTITUTION_ADOPTION_REQUIRED
    assert "no local agent named Emma" in capsys.readouterr().err
    monkeypatch.setattr(
        cli_lifecycle, "check_installed_constitution_adoption",
        lambda *a, **k: ConstitutionAdoptionCheck(),
    )
    assert cli.cmd_restart(_restart_args(name="Emma")) == OFFLINE_MIGRATION_REQUIRED
    assert "no local agent named Emma" in capsys.readouterr().err
    # And the launcher reads the registry they read.
    assert cli.cmd_terminate(_restart_args(name="Emma")) == 1
    assert cli.cmd_start(_restart_args(name="Emma")) == 1
    assert calls == []


def _upstream_moves_before_the_pull(origin, clone, monkeypatch):
    """Publish the cutover upstream once the checks have judged, before the pull.

    Hooked on the dirty-tree check, the pull step's first act. A fetch in the
    checkout moves its remote-tracking ref too, as a concurrent fetch (or
    ``git pull``'s own) would.
    """
    judged = []

    def dirty(checkout):
        judged.append(git(clone, "rev-parse", "refs/remotes/origin/main").strip())
        _commit_manifest(origin, packaged_manifest())
        git(clone, "fetch", "-q")
        return False, ""

    monkeypatch.setattr(cli, "_git_working_tree_dirty", dirty)
    return judged


def test_the_pull_lands_the_revision_the_checks_judged(
    tmp_path, project, monkeypatch, capsys
):
    """Upstream gains a migration requirement after the checks: not landed."""
    origin, clone = origin_and_clone(tmp_path, b"# Kestrel Constitution\n")
    (origin / "README.md").write_text("a release that declares nothing\n")
    git(origin, "add", "README.md")
    git(origin, "commit", "-q", "-m", "an ordinary release")
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: clone)
    monkeypatch.setattr(
        cli_lifecycle, "check_constitution_adoption",
        lambda *a, **k: ConstitutionAdoptionCheck(),
    )
    monkeypatch.setattr(
        cli, "_run_uv_pip_install_editable", lambda *a, **kw: (0, "")
    )
    monkeypatch.setattr(cli, "cmd_restart", lambda args: 0)
    _leave_draft_a2a_data(_db(project, "Emma"))
    judged = _upstream_moves_before_the_pull(origin, clone, monkeypatch)

    assert cli.cmd_update(_update_args(restart=False)) == 0

    assert f"checking incoming revision {judged[0][:12]}" in capsys.readouterr().out
    assert _head(clone) == judged[0]
    assert (clone / "README.md").exists()
    assert not (clone / MANIFEST_RELPATH).exists()


def test_a_pull_the_checks_found_empty_moves_nothing(
    tmp_path, project, monkeypatch
):
    """No incoming revision when checked; upstream moves; HEAD stays put."""
    origin, clone = origin_and_clone(tmp_path, b"# Kestrel Constitution\n")
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: clone)
    monkeypatch.setattr(
        cli_lifecycle, "check_constitution_adoption",
        lambda *a, **k: ConstitutionAdoptionCheck(),
    )
    installs = []
    monkeypatch.setattr(
        cli, "_run_uv_pip_install_editable",
        lambda *a, **kw: installs.append("install") or (0, ""),
    )
    _leave_draft_a2a_data(_db(project, "Emma"))
    head = _head(clone)
    judged = _upstream_moves_before_the_pull(origin, clone, monkeypatch)

    assert cli.cmd_update(_update_args(restart=False)) == 0

    assert judged == [head]
    assert _head(clone) == head
    assert not (clone / MANIFEST_RELPATH).exists()
    assert installs == ["install"]


def test_the_pull_refuses_a_checkout_with_no_upstream(
    tmp_path, project, monkeypatch, capsys
):
    """git pull's own refusal, kept: the update aborts before installing."""
    _, clone = origin_and_clone(tmp_path, b"# Kestrel Constitution\n")
    git(clone, "branch", "--unset-upstream")
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: clone)
    monkeypatch.setattr(
        cli, "_run_uv_pip_install_editable",
        lambda *a, **kw: pytest.fail("installed after a failed pull"),
    )

    assert cli.cmd_update(_update_args(restart=False)) == 1
    assert "no upstream branch to pull from" in capsys.readouterr().err


def _detached_behind_a_judged_upstream(tmp_path, project, monkeypatch):
    """A detached checkout one release behind its upstream, which the checks judge.

    Returns ``(origin, clone, detached, judged, installs)``: the upstream
    release declares nothing, so the checks pass it. The pull, which moves
    only this test's clone, is real; the install is recorded.
    """
    origin, clone = origin_and_clone(tmp_path, b"# Kestrel Constitution\n")
    detached = _head(clone)
    (origin / "README.md").write_text("a release that declares nothing\n")
    git(origin, "add", "README.md")
    git(origin, "commit", "-q", "-m", "an ordinary release")
    judged = git(origin, "rev-parse", "HEAD").strip()
    git(clone, "checkout", "-q", "--detach", detached)
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: clone)
    monkeypatch.setattr(
        cli_lifecycle, "check_constitution_adoption",
        lambda *a, **k: ConstitutionAdoptionCheck(),
    )
    installs = []
    monkeypatch.setattr(
        cli, "_run_uv_pip_install_editable",
        lambda *a, **kw: installs.append("install") or (0, ""),
    )
    _leave_draft_a2a_data(_db(project, "Emma"))
    return origin, clone, detached, judged, installs


def test_detached_recovery_lands_the_judged_commit(tmp_path, project, monkeypatch):
    _, clone, _, judged, installs = _detached_behind_a_judged_upstream(
        tmp_path, project, monkeypatch
    )

    assert cli.cmd_update(_update_args(restart=False)) == 0

    assert git(clone, "symbolic-ref", "--short", "HEAD").strip() == "main"
    assert _head(clone) == judged
    assert installs == ["install"]


def test_detached_recovery_refuses_a_local_branch_past_the_judged_commit(
    tmp_path, project, monkeypatch, capsys
):
    """The reviewer's reproduction: the checks judge the upstream commit, but
    the local branch recovery reattaches to is already past it, at a commit
    requiring the cutover Emma still needs. Checking it out installed that."""
    _, clone, detached, judged, installs = _detached_behind_a_judged_upstream(
        tmp_path, project, monkeypatch
    )
    git(clone, "fetch", "-q")
    git(clone, "branch", "-q", "-f", "main", judged)
    git(clone, "checkout", "-q", "main")
    ahead = _commit_manifest(clone, packaged_manifest())
    git(clone, "checkout", "-q", "--detach", detached)

    assert cli.cmd_update(_update_args(restart=False)) == 1

    assert installs == []
    # Nothing moved: not HEAD, not the branch, not the working tree.
    assert _head(clone) == detached
    assert git(clone, "rev-parse", "refs/heads/main").strip() == ahead
    assert not (clone / MANIFEST_RELPATH).exists()
    assert "pull: FAILED" in capsys.readouterr().err


def test_nothing_installs_unless_the_pull_left_head_on_the_judged_commit(
    tmp_path, project, monkeypatch, capsys
):
    """However the pull got there: a merge that lands anywhere else stops the
    update before the install."""
    origin, clone, _, judged, installs = _detached_behind_a_judged_upstream(
        tmp_path, project, monkeypatch
    )
    git(clone, "checkout", "-q", "main")
    beyond = _commit_manifest(origin, packaged_manifest())
    git(clone, "fetch", "-q")
    real_pull = cli._run_git_fast_forward
    monkeypatch.setattr(
        cli, "_run_git_fast_forward", lambda c, _revision: real_pull(c, beyond)
    )
    monkeypatch.setattr(cli, "_run_git_fetch", lambda _checkout: (0, ""))
    monkeypatch.setattr(
        cli,
        "_resolve_pull_target",
        lambda _checkout: cli_lifecycle._PullTarget(
            upstream=judged, revision=judged, head=_head(clone)
        ),
    )

    assert cli.cmd_update(_update_args(restart=False)) == 1

    assert installs == []
    assert f"not {judged[:12]}, the revision the checks judged" in (
        capsys.readouterr().err
    )


# ---------------------------------------------------------------------------
# The feature sync installs the core the checks judged, and no other
# ---------------------------------------------------------------------------


def _declare_core(project: Path, checkout: Path) -> Path:
    """A host manifest declaring core editable from ``checkout``, with an extra.

    The extra is what makes the sync act on a core already linked from the
    checkout (``ensure``), which is where it used to pull it again.
    """
    manifest = project / ".kestrel-host-features.toml"
    manifest.write_text(
        "[[feature]]\n"
        'name = "kestrel-sovereign"\n'
        f'editable = "{checkout}"\n'
        'extras = ["observability"]\n'
    )
    return manifest


def _fake_venv_with_real_git(monkeypatch, checkout: Path, *, linked: bool = True, **venv_options):
    """The sync's installs land in a modelled venv; git still runs for real.

    ``FakeUv`` replaces ``subprocess.run`` everywhere, git included. Git is
    routed back to the real one so the sync's pull, had it run, would move
    this test's clone, and so is this interpreter, which runs the offline
    migration probe. Core is modelled linked from ``checkout``, or with
    ``linked=False`` as an index wheel. Returns the double and the install
    argvs as the sync issued them: the double models no extras on a
    checkout, so it is shown the checkout alone.
    """
    from tests.utils.fake_uv import FakeUv, use_fake_uv

    real_run = subprocess.run
    venv = FakeUv(
        core_checkout=str(checkout) if linked else None, sdk_checkout=None,
        **venv_options,
    )
    use_fake_uv(monkeypatch, venv)
    # use_fake_uv stubs the pull. The real one is the point here.
    monkeypatch.setattr(
        cli_lifecycle, "_editable_git_pull", _REAL_EDITABLE_GIT_PULL
    )
    issued = []

    def run(cmd, *args, **kwargs):
        if cmd and (
            Path(str(cmd[0])).name == "git" or str(cmd[0]) == sys.executable
        ):
            return real_run(cmd, *args, **kwargs)
        issued.append(list(cmd))
        extra = "[observability]"
        if str(cmd[-1]).endswith(extra):
            cmd = [*cmd[:-1], str(cmd[-1])[: -len(extra)]]
        return venv.run(
            cmd,
            capture_output=kwargs.get("capture_output", True),
            text=kwargs.get("text", True),
            timeout=kwargs.get("timeout"),
        )

    monkeypatch.setattr(subprocess, "run", run)
    return venv, issued


_REAL_EDITABLE_GIT_PULL = cli_lifecycle._editable_git_pull


@pytest.fixture
def judged_core_update(tmp_path, project, monkeypatch):
    """``kestrel update`` over a checkout the host manifest declares as core.

    Upstream holds one ordinary release, which the checks judge. Once the pull
    has landed it, the install step publishes a release requiring the cutover
    Emma still needs, and fetches it into the clone, as a concurrent fetch (or
    the sync's own pull) would. Reconcile, the lock report and the restart are
    out of scope; the sync and git are real.
    """
    origin, clone = origin_and_clone(tmp_path, b"# Kestrel Constitution\n")
    (origin / "README.md").write_text("a release that declares nothing\n")
    git(origin, "add", "README.md")
    git(origin, "commit", "-q", "-m", "an ordinary release")
    judged = git(origin, "rev-parse", "HEAD").strip()
    manifest = _declare_core(project, clone)
    _leave_draft_a2a_data(_db(project, "Emma"))
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: clone)
    monkeypatch.setattr(
        cli_lifecycle, "check_constitution_adoption",
        lambda *a, **k: ConstitutionAdoptionCheck(),
    )
    # The lock preflight has its own tests; this clone carries no uv.lock.
    monkeypatch.setattr(cli, "_run_sync_preflight", lambda *a, **k: 0)
    monkeypatch.setattr(cli_lifecycle, "_run_feature_reconcile", lambda *a, **k: 0)
    monkeypatch.setattr(cli_lifecycle, "_report_core_lock", lambda: None)
    venv, issued = _fake_venv_with_real_git(monkeypatch, clone)
    beyond = []

    def install(*_args, **_kwargs):
        beyond.append(_commit_manifest(origin, packaged_manifest()))
        git(clone, "fetch", "-q")
        return 0, ""

    monkeypatch.setattr(cli, "_run_uv_pip_install_editable", install)
    return SimpleNamespace(
        clone=clone, manifest=manifest, judged=judged, beyond=beyond, venv=venv,
        issued=issued,
    )


def test_the_feature_sync_links_core_at_the_judged_commit_without_pulling_it(
    judged_core_update, capsys
):
    """The reviewer's reproduction: the checks judged the ordinary release,
    upstream then gained the cutover, and the sync's own ``git pull`` of the
    declared core landed it and reached the core installer."""
    run = judged_core_update

    rc = cli.cmd_update(
        _update_args(features=True, restart=False, manifest=str(run.manifest))
    )

    assert rc == 0, capsys.readouterr()
    assert run.beyond, "upstream never moved past the judged commit"
    assert _head(run.clone) == run.judged
    assert not (run.clone / MANIFEST_RELPATH).exists()
    # Core was still linked: the judged checkout, at the judged commit.
    assert [cmd[-1] for cmd in run.issued if "-e" in cmd] == [
        f"{run.clone}[observability]"
    ], run.issued
    out = capsys.readouterr().out
    assert f"--judged-core-revision {run.judged}" in out
    assert f"not pulled: {run.clone} is at {run.judged[:12]}" in out


def test_the_feature_sync_refuses_a_core_checkout_moved_off_the_judged_commit(
    judged_core_update, monkeypatch, capsys
):
    """Whatever moved it after the install: the sync links no core then."""
    from kestrel_sovereign.cli_features import CORE_UNJUDGED

    run = judged_core_update
    install = cli._run_uv_pip_install_editable

    def install_then_move(*args, **kwargs):
        result = install(*args, **kwargs)
        git(run.clone, "merge", "-q", "--ff-only", run.beyond[-1])
        return result

    monkeypatch.setattr(cli, "_run_uv_pip_install_editable", install_then_move)

    rc = cli.cmd_update(
        _update_args(features=True, restart=False, manifest=str(run.manifest))
    )

    assert rc == CORE_UNJUDGED
    assert not [cmd for cmd in run.issued if "-e" in cmd], run.issued
    captured = capsys.readouterr()
    assert f"not {run.judged[:12]}, the commit the checks judged" in captured.out
    assert "--continue-on-error does not cover this" in captured.err


def test_update_refuses_a_manifest_declaring_core_elsewhere_before_anything(
    tmp_path, project, upstream, monkeypatch, capsys
):
    """The sync would pull and link a core no check judged: refused before
    the fetch, the pull and the install."""
    from kestrel_sovereign.cli_features import CORE_UNJUDGED

    _, clone, steps = upstream
    elsewhere = tmp_path / "another-core"
    elsewhere.mkdir()
    manifest = _declare_core(project, elsewhere)
    fetches = []
    monkeypatch.setattr(
        cli, "_run_git_fetch", lambda checkout: fetches.append(checkout) or (0, "")
    )
    head = _head(clone)

    rc = cli.cmd_update(
        _update_args(features=True, restart=False, manifest=str(manifest))
    )

    assert rc == CORE_UNJUDGED
    assert steps == [] and fetches == []
    assert _head(clone) == head
    err = capsys.readouterr().err
    assert str(elsewhere) in err and "Nothing was pulled" in err


@pytest.fixture
def index_core_update(tmp_path, project, monkeypatch):
    """``kestrel update`` on a core installed from the index (#3832).

    There is no source checkout to pull or install, so the checks judge the
    release installed now. The host manifest declares core editable from a
    scratch clone whose upstream has since gained a release requiring a
    migration this code has no check for: the sync's own pull would land it.
    The gates and git are real; reconcile, the lock report and the restart
    are out of scope.
    """
    origin, clone = origin_and_clone(tmp_path, b"# Kestrel Constitution\n")
    beyond = _commit_manifest(origin, _manifest(_entry("a-future-cutover")))
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: None)
    monkeypatch.setattr(
        cli_lifecycle, "check_constitution_adoption",
        lambda *a, **k: ConstitutionAdoptionCheck(),
    )
    reconciled = []
    monkeypatch.setattr(
        cli_lifecycle, "_run_feature_reconcile",
        lambda *a, **k: reconciled.append(k) or 0,
    )
    monkeypatch.setattr(cli_lifecycle, "_report_core_lock", lambda: None)
    return SimpleNamespace(
        origin=origin, clone=clone, beyond=beyond, head=_head(clone),
        reconciled=reconciled,
    )


def test_an_index_core_update_refuses_a_manifest_declaring_a_core_checkout(
    index_core_update, project, monkeypatch, capsys
):
    """The reviewer's reproduction: the installed-code gate passed, then the
    feature sync pulled the declared core checkout, landing a release whose
    migration no check read, and reached its installer."""
    run = index_core_update
    manifest = _declare_core(project, run.clone)
    venv, issued = _fake_venv_with_real_git(monkeypatch, run.clone, linked=False)
    assert check_revision_offline_migrations(
        project, run.origin, run.beyond, code_label="the release beyond"
    ).safe is False, "the release beyond the judged one is not refused"

    rc = cli.cmd_update(
        _update_args(features=True, restart=False, manifest=str(manifest))
    )

    from kestrel_sovereign.cli_features import CORE_UNJUDGED

    assert rc == CORE_UNJUDGED
    assert _head(run.clone) == run.head
    assert not (run.clone / MANIFEST_RELPATH).exists()
    assert issued == [] and venv.installed["kestrel-sovereign"] == "0.52.0"
    assert run.reconciled == []
    err = capsys.readouterr().err
    assert str(run.clone) in err and "Nothing was pulled" in err
    assert "the release the checks judged" in err


def test_an_index_core_update_holds_core_at_the_judged_release(
    index_core_update, project, monkeypatch, capsys
):
    """A missing feature wants a newer core than the installed wheel. Without
    the hold, the sync resolves that newer release from the index: one no
    check read. The update holds every sync install to the installed release,
    and hands reconcile the same release to hold."""
    from kestrel_sovereign.cli_features import JudgedCore

    run = index_core_update
    manifest = project / ".kestrel-host-features.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')
    venv, _ = _fake_venv_with_real_git(
        monkeypatch, run.clone, linked=False, feature_requires=">=0.53",
    )

    rc = cli.cmd_update(
        _update_args(
            features=True, restart=False, manifest=str(manifest),
            continue_on_error=True,
        )
    )

    assert rc == 0, capsys.readouterr()
    assert venv.installed["kestrel-sovereign"] == "0.52.0"
    assert venv.pins and set(venv.pins) == {"==0.52.0"}
    (reconcile,) = run.reconciled
    assert reconcile["judged"] == JudgedCore(version="0.52.0")


def test_the_continuation_carries_the_judged_commit():
    """A fresh interpreter that lost it would pull the core checkout again."""
    args = cli.build_parser().parse_args(
        ["update", "--judged-core-revision", "a" * 40]
    )

    argv = cli_lifecycle._update_argv(args)

    assert argv[argv.index("--judged-core-revision") + 1] == "a" * 40


# ---------------------------------------------------------------------------
# The launcher starts the agent from the registry it was found in
# ---------------------------------------------------------------------------


def test_a_named_start_hands_the_launcher_the_registry_it_read(
    tmp_path, project, monkeypatch
):
    """The reviewer's reproduction, through the real ``ProcessManager``.

    Both registries name Emma, at different data directories. ``cmd_start``
    found her in the host's external registry, but ``start_agent`` reloaded
    the project's default file and refused her as not matching "the active
    launcher roster", after a named restart had already terminated her.
    Spawning is replaced; everything before it runs.
    """
    from kestrel_sovereign.multi_agent.config import (
        HostConfig,
        LocalAgentConfig,
        MultiAgentConfig,
    )
    from kestrel_sovereign.multi_agent.process_manager import ProcessManager

    registry = tmp_path / "fleet" / "fleet.toml"
    registry.parent.mkdir()
    MultiAgentConfig(
        host=HostConfig(),
        agents={"Emma": LocalAgentConfig(data_dir="agent_data/emma-fleet", port=18803)},
    ).save(registry)
    from tests.utils.constitution_anchor import write_agent_anchor

    write_agent_anchor(
        project / "agent_data" / "emma-fleet" / "kestrel_prime.db",
        name="Emma",
        constitution_hash="e" * 64,
    )
    monkeypatch.setenv("KESTREL_MULTI_AGENT_CONFIG", str(registry))
    monkeypatch.delenv("KESTREL_API_KEY", raising=False)
    spawned = []

    def spawn_detached(self, cmd, env, log_file, pid_file, *, port):
        spawned.append((port, str(pid_file)))
        return 4242

    def refuse(*_args, **_kwargs):
        pytest.fail("reached the real process lifecycle")

    monkeypatch.setattr(cli_lifecycle, "ProcessManager", ProcessManager)
    monkeypatch.setattr(ProcessManager, "_spawn_detached", spawn_detached)
    monkeypatch.setattr(ProcessManager, "_spawn", refuse)
    monkeypatch.setattr(
        ProcessManager, "is_port_in_use", staticmethod(lambda port, host="": False)
    )
    monkeypatch.setattr(
        ProcessManager, "wait_for_health", staticmethod(lambda *a, **k: True)
    )
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_maybe_first_run_setup", lambda _project: None)

    assert cli_lifecycle.cmd_start(_restart_args(name="Emma")) == 0
    ((port, pid_file),) = spawned
    assert port == 18803
    assert Path(pid_file).is_relative_to(project / "agent_data" / "emma-fleet")
