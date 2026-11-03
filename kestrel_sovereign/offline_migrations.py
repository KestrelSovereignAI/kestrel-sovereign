"""Refuse a deploy onto a release that needs an offline migration not yet done (#3832).

Some releases refuse to start on data an earlier release wrote until an
operator migrates it offline. On 2026-10-24 an agent's ``update_then_restart``
installed Core 0.54 (native A2A, #3803) and restarted. Native startup refused
every agent whose store still held draft A2A data, and the fleet was down for
31 hours. Nothing had asked, before the install, whether the incoming release
could start on these databases.

A release declares the offline migrations it requires in
``kestrel_sovereign/data/offline_migrations.toml``. Every deploy path asks this
module before it changes anything:

- ``kestrel update``, before the pull, about the revision the pull will land
  on, for every local agent;
- ``kestrel restart`` (including ``kestrel update``'s restart step), about the
  installed code;
- the restart coordinator, about the installed code before a plain restart,
  and for an ``update_then_restart`` about the fetched revision before checkout
  and the installed code before the restart.

The declarations are data, read from the incoming revision with git: the
incoming code cannot be imported before it is installed. Whether a declared
migration is still pending in a database is answered by a probe the judging
code knows by the migration's id, a read-only run of the check startup applies.
A declaration the judging code has no probe for cannot be judged, so it blocks:
the release is newer than the code judging it, and only an operator crosses it.
A database that cannot be read blocks too, since nothing was learned about it.

The probes run in a fresh interpreter launched the way the host is. A
PostgreSQL server that never answers is bounded by the process deadline, and
the installed-code check runs the code a restart would boot rather than the
code this process imported.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kestrel_sovereign.multi_agent.config import launched_host_registry
from kestrel_sovereign.paths import spawned_agent_env

#: Where a release declares its offline migrations, relative to the checkout
#: root (and to the directory holding the installed package).
MANIFEST_RELPATH = "kestrel_sovereign/data/offline_migrations.toml"

#: Exit status of a ``kestrel update`` / ``kestrel restart`` this gate refused.
#: Distinct from the core-state codes (2-4) and the constitution gate's 5.
OFFLINE_MIGRATION_REQUIRED = 6

#: The operator flag that proceeds anyway.
OVERRIDE_FLAG = "--allow-pending-offline-migration"

#: How the operator reads a check of the code installed on disk.
INSTALLED_CODE = "the installed code"

#: How a PostgreSQL database is named in a finding: never by its DSN.
POSTGRES_DATABASE = "the PostgreSQL database (KESTREL_DATABASE_URL)"

#: The database holds nothing the migration still has to convert.
COMPLETE = "complete"
#: The database holds data the release refuses to start on.
PENDING = "pending"
#: The database could not be read, so nothing was learned about it.
UNVERIFIED = "unverified"
#: The judging code has no probe for the declared migration.
UNKNOWN = "unknown"
_STATUSES = frozenset({COMPLETE, PENDING, UNVERIFIED, UNKNOWN})

#: The manifest format this code reads.
_MANIFEST_VERSION = 1
_MIGRATION_FIELDS = frozenset({"id", "summary", "runbook"})
_MIGRATION_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

_GIT_TIMEOUT_SECONDS = 60

#: Tree-entry modes ``git ls-tree`` reports for a regular file.
_GIT_FILE_MODES = frozenset({b"100644", b"100755"})

#: The module a fresh interpreter runs to answer the check.
_FRESH_CHECK_MODULE = "kestrel_sovereign.offline_migrations"

#: Version of the answer a fresh-interpreter check prints. The child may be
#: newer code than the process asking it.
_FRESH_CHECK_PROTOCOL = 1

#: Marks the answer line in the child's stdout, which imports may also write to.
_FRESH_CHECK_RESULT_PREFIX = "kestrel-offline-migrations: "

#: Bound on the fresh-interpreter check: imports plus opening every agent
#: database, including a PostgreSQL connect that never answers.
_FRESH_CHECK_TIMEOUT_SECONDS = 180


class OfflineMigrationError(RuntimeError):
    """The gate could not establish what the code about to run requires.

    Never a finding about one database: nothing was learned about any of them,
    so it refuses as a pending migration does.
    """


@dataclass(frozen=True)
class OfflineMigration:
    """One offline migration a release declares it requires."""

    id: str
    summary: str
    #: The operator runbook, relative to the checkout root.
    runbook: str


@dataclass(frozen=True)
class MigrationFinding:
    """What one declared migration's probe found in one database."""

    migration: str
    status: str
    summary: str
    runbook: str
    #: A SQLite path or :data:`POSTGRES_DATABASE`; empty for :data:`UNKNOWN`.
    database: str = ""
    #: The agents that read this database (every selected agent for UNKNOWN).
    agents: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class OfflineMigrationCheck:
    """One gate evaluation: a finding per declared migration and database."""

    findings: tuple[MigrationFinding, ...] = ()
    #: What the declarations were read from, as the operator reads it.
    code_label: str = INSTALLED_CODE

    @property
    def blocking(self) -> tuple[MigrationFinding, ...]:
        """Findings that leave an agent unable to start, or unknown to be able."""
        return tuple(f for f in self.findings if f.status != COMPLETE)

    @property
    def safe(self) -> bool:
        return not self.blocking

    @property
    def blocking_agents(self) -> tuple[str, ...]:
        return tuple(sorted({agent for f in self.blocking for agent in f.agents}))


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def parse_manifest(content: bytes, *, source: str) -> tuple[OfflineMigration, ...]:
    """The migrations a manifest declares.

    Strict: a field or format version this code does not know may carry a
    requirement it would otherwise ignore, so it refuses rather than judging
    part of the declaration. ``source`` names where the bytes came from.

    Raises:
        OfflineMigrationError: The manifest is not one this code can read whole.
    """

    def refuse(why: str) -> OfflineMigrationError:
        return OfflineMigrationError(
            f"cannot read the offline migrations {source} declares: {why}"
        )

    try:
        data = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise refuse(str(exc)) from exc
    version = data.get("version")
    if type(version) is not int or version != _MANIFEST_VERSION:
        raise refuse(
            f"format version {version!r}; this code reads version {_MANIFEST_VERSION}"
        )
    unknown = sorted(set(data) - {"version", "migration"})
    if unknown:
        raise refuse(f"unknown fields {unknown}")
    entries = data.get("migration", [])
    if not isinstance(entries, list):
        raise refuse("`migration` is not an array of tables")
    migrations: list[OfflineMigration] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _MIGRATION_FIELDS:
            raise refuse(
                f"each migration needs exactly the fields {sorted(_MIGRATION_FIELDS)}"
            )
        if any(not isinstance(value, str) or not value.strip() for value in entry.values()):
            raise refuse("every migration field must be a non-empty string")
        if not _MIGRATION_ID.match(entry["id"]):
            raise refuse(f"invalid migration id {entry['id']!r}")
        if any(m.id == entry["id"] for m in migrations):
            raise refuse(f"migration {entry['id']!r} is declared twice")
        runbook = PurePosixPath(entry["runbook"])
        if runbook.is_absolute() or ".." in runbook.parts:
            raise refuse(
                f"runbook {entry['runbook']!r} is not a path inside the checkout"
            )
        migrations.append(OfflineMigration(**entry))
    return tuple(migrations)


def packaged_manifest() -> bytes:
    """This installation's own manifest: the one the installed code declares.

    Raises:
        OfflineMigrationError: The installed package has no readable manifest.
    """
    path = Path(__file__).parent.parent / MANIFEST_RELPATH
    try:
        return path.read_bytes()
    except OSError as exc:
        raise OfflineMigrationError(
            f"cannot read the installed code's offline migrations at {path}: {exc}"
        ) from exc


def manifest_at(repo_path: str | Path, revision: str) -> bytes | None:
    """The manifest checking out ``revision`` of ``repo_path`` would install.

    None when that revision has none: it predates declarations (#3832), and
    declares no offline migration. ``revision`` is resolved to one commit first,
    so a ref that moves meanwhile (``FETCH_HEAD``) cannot mix two revisions.

    Raises:
        OfflineMigrationError: git could not resolve or read the revision, or
            the revision leaves something other than a file at the manifest.
    """
    repo = Path(repo_path)
    commit = _git(repo, "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}")
    if commit.returncode != 0:
        raise OfflineMigrationError(
            f"cannot resolve {revision} in {repo}: {_git_detail(commit)}"
        )
    oid = commit.stdout.decode("ascii", "replace").strip()
    listed = _git(repo, "ls-tree", "-z", oid, "--", MANIFEST_RELPATH)
    if listed.returncode != 0:
        raise OfflineMigrationError(
            f"cannot list {MANIFEST_RELPATH} at {revision} in {repo}: "
            f"{_git_detail(listed)}"
        )
    record = listed.stdout.split(b"\0", 1)[0]
    if not record:
        return None
    meta, _, _name = record.partition(b"\t")
    mode, kind, blob = meta.split(b" ")
    if kind != b"blob" or mode not in _GIT_FILE_MODES:
        raise OfflineMigrationError(
            f"{revision} leaves {MANIFEST_RELPATH} in {repo} as something other "
            "than a file, so what it requires cannot be read"
        )
    shown = _git(repo, "cat-file", "blob", blob.decode("ascii"))
    if shown.returncode != 0:
        raise OfflineMigrationError(
            f"cannot read {MANIFEST_RELPATH} at {revision} in {repo}: "
            f"{_git_detail(shown)}"
        )
    return shown.stdout


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OfflineMigrationError(f"git is unavailable: {exc}") from exc


def _git_detail(result: subprocess.CompletedProcess) -> str:
    raw = result.stderr or result.stdout or b""
    return raw.decode("utf-8", "replace").strip() or f"git exited {result.returncode}"


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


async def _native_a2a_cutover(backend) -> str | None:
    """Pending while native A2A startup would refuse this store (#3803).

    The startup preflight itself, which only reads. Judged by the code that
    asks, so a store a newer runtime has already upgraded past the revisions
    this code knows also reads as pending: the operator decides that crossing.
    """
    from kestrel_sovereign.a2a.native_schema import (
        NativeA2AMigrationRequired,
        require_native_a2a_schema,
    )

    try:
        await require_native_a2a_schema(backend)
    except NativeA2AMigrationRequired as exc:
        return str(exc)
    return None


#: Every offline migration this code can judge, by the id a manifest declares.
#: A probe reads one connected database and returns why the migration is still
#: pending there, or None when nothing is left to migrate.
PROBES: dict[str, Callable[[object], Awaitable[str | None]]] = {
    "native-a2a-cutover": _native_a2a_cutover,
}


@dataclass(frozen=True)
class _Database:
    """One database the selected agents read, as their runtime opens it."""

    label: str
    agents: tuple[str, ...]
    sqlite_path: Path | None = None
    postgres_dsn: str = ""


def _runtime_reads_postgres(env: dict) -> bool:
    """Whether the agents' primary store is the shared PostgreSQL database.

    The runtime's rule (``AgentManager._hosted_agent_runtime_factory_configured``
    and doctor's ``_anchor_is_the_runtime_database``): PostgreSQL and a DSN,
    exactly, or the agent's own SQLite file.
    """
    backend = env.get("KESTREL_DB_BACKEND", "sqlite").lower()
    return backend == "postgres" and bool(env.get("KESTREL_DATABASE_URL"))


def _databases(
    project_dir: Path, env: dict, local: dict, selected: list[str]
) -> list[_Database]:
    if not selected:
        return []
    if _runtime_reads_postgres(env):
        return [
            _Database(
                label=POSTGRES_DATABASE,
                agents=tuple(selected),
                postgres_dsn=env["KESTREL_DATABASE_URL"],
            )
        ]
    by_path: dict[Path, list[str]] = {}
    for name in selected:
        path = local[name].resolve_data_dir(project_dir) / "kestrel_prime.db"
        by_path.setdefault(path, []).append(name)
    return [
        _Database(label=str(path), agents=tuple(names), sqlite_path=path)
        for path, names in by_path.items()
    ]


async def _probe_database(
    database: _Database, probes: dict
) -> dict[str, tuple[str, str]]:
    """``{migration id: (status, detail)}`` for one database, read-only."""
    from kestrel_sovereign.storage.db.interface import DatabaseError
    from kestrel_sovereign.storage.db.sqlite import ColdReadUnavailable

    if database.sqlite_path is not None:
        if not database.sqlite_path.exists():
            # The runtime creates native stores fresh: nothing to migrate.
            return {mid: (COMPLETE, "no database yet") for mid in probes}
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

        backend = SQLiteBackend(str(database.sqlite_path), cold_read=True)
    else:
        from kestrel_sovereign.storage.db.postgres import PostgresBackend

        backend = PostgresBackend(
            database.postgres_dsn, min_pool_size=1, max_pool_size=1
        )
    try:
        try:
            await backend.connect()
            results = {}
            for mid, probe in probes.items():
                reason = await probe(backend)
                results[mid] = (PENDING, reason) if reason else (COMPLETE, "")
            if database.sqlite_path is not None:
                backend.assert_cold_read_still_valid()
            return results
        finally:
            await backend.close()
    except (DatabaseError, ColdReadUnavailable, OSError) as exc:
        if database.sqlite_path is None:
            # A driver message can carry connection identity; the agents'
            # record of this check must not. `kestrel doctor` diagnoses it.
            detail = (
                f"{type(exc).__name__} reading it; run `kestrel doctor` to "
                "diagnose the connection"
            )
        else:
            detail = f"{type(exc).__name__}: {exc}"
        return {mid: (UNVERIFIED, detail) for mid in probes}


def _local_agents(project_dir: Path, env: dict) -> dict:
    """The local agents a host started in the project loads.

    From the registry ``kestrel start`` and ``kestrel terminate`` read too
    (:func:`~kestrel_sovereign.multi_agent.config.launched_host_registry`,
    ``KESTREL_MULTI_AGENT_CONFIG`` included), so the gate judges the
    databases a restart would open rather than whatever another file names.
    """
    try:
        registry = launched_host_registry(project_dir, env=env)
    except (OSError, ValueError) as exc:
        raise OfflineMigrationError(
            f"multi-agent configuration is invalid: {exc}"
        ) from exc
    return registry.get_local_agents()


def _select(local: dict, agent_names: Iterable[str] | None) -> list[str]:
    """The agents to judge; a named one the registry lacks refuses.

    Judging nothing for it would pass a restart the launcher might resolve
    elsewhere.
    """
    if agent_names is None:
        return list(local)
    names = list(agent_names)
    missing = [name for name in names if name not in local]
    if missing:
        raise OfflineMigrationError(
            f"no local agent named {', '.join(missing)} in the registry the "
            "host loads, so its database cannot be judged"
        )
    return names


def judge(
    project_dir: Path,
    migrations: tuple[OfflineMigration, ...],
    *,
    agent_names: Iterable[str] | None = None,
) -> tuple[MigrationFinding, ...]:
    """Probe each selected agent's database for each declared migration, here.

    The fresh interpreter's half of :func:`check_offline_migrations`.
    ``agent_names`` None means every local agent.

    Raises:
        OfflineMigrationError: The registry cannot be loaded or lacks a named
            agent.
    """
    env = spawned_agent_env(project_dir)
    local = _local_agents(project_dir, env)
    selected = _select(local, agent_names)
    databases = _databases(project_dir, env, local, selected)
    probes = {m.id: PROBES[m.id] for m in migrations if m.id in PROBES}
    results: dict[str, dict[str, tuple[str, str]]] = {}
    if probes and databases:

        async def probe_all() -> None:
            for database in databases:
                results[database.label] = await _probe_database(database, probes)

        asyncio.run(probe_all())
    findings: list[MigrationFinding] = []
    for migration in migrations:
        common = {
            "migration": migration.id,
            "summary": migration.summary,
            "runbook": migration.runbook,
        }
        if migration.id not in PROBES:
            findings.append(
                MigrationFinding(
                    **common,
                    status=UNKNOWN,
                    agents=tuple(selected),
                    detail="this code has no check for it",
                )
            )
            continue
        for database in databases:
            status, detail = results[database.label][migration.id]
            findings.append(
                MigrationFinding(
                    **common,
                    status=status,
                    database=database.label,
                    agents=database.agents,
                    detail=detail,
                )
            )
    return tuple(findings)


# ---------------------------------------------------------------------------
# The check a deploy runs
# ---------------------------------------------------------------------------


def check_offline_migrations(
    project_dir: Path,
    *,
    agent_names: Iterable[str] | None = None,
    manifest: Callable[[], bytes | None] | None = None,
    code_label: str = INSTALLED_CODE,
) -> OfflineMigrationCheck:
    """Whether the selected agents' databases have every required offline migration.

    ``manifest`` None judges the installed code's own declarations. Otherwise
    it returns the declarations to judge, as a release's manifest file holds
    them (:func:`manifest_at` reads one from a revision), or None when that
    release declares none. It is asked only once an agent is selected, so a
    host with no agents does no git work. ``agent_names`` limits the agents
    judged; None means every local agent in the registry a host started in
    the project loads (``KESTREL_MULTI_AGENT_CONFIG`` included).
    ``code_label`` names what is judged, for the operator.

    Each probe runs in a fresh interpreter launched the way the host is: this
    process's interpreter, the project as its working directory and the
    environment the launcher hands a spawned host. Read-only.

    Raises:
        OfflineMigrationError: The manifest or registry cannot be read, the
            registry lacks a named agent, or the fresh interpreter could not
            answer: it failed to start or to
            check, timed out, or answered in a form this process cannot read.
    """
    # Absolute, because the child runs with the project as its working
    # directory and would resolve a relative path against it a second time.
    project_dir = Path(project_dir).absolute()
    env = spawned_agent_env(project_dir)
    selected = _select(_local_agents(project_dir, env), agent_names)
    if not selected:
        return OfflineMigrationCheck(code_label=code_label)
    declared = content = None
    if manifest is not None:
        content = manifest()
        declared = () if content is None else parse_manifest(content, source=code_label)
        if not declared:
            return OfflineMigrationCheck(code_label=code_label)
    command = [
        sys.executable, "-m", _FRESH_CHECK_MODULE,
        "--project-dir", str(project_dir),
        *(f"--agent={name}" for name in selected),
    ]
    if content is not None:
        command.append("--manifest-stdin")
    try:
        result = subprocess.run(
            command,
            cwd=str(project_dir),
            env=env,
            input=content if content is not None else b"",
            capture_output=True,
            check=False,
            timeout=_FRESH_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise OfflineMigrationError(
            f"checking offline migrations timed out after "
            f"{_FRESH_CHECK_TIMEOUT_SECONDS}s"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise OfflineMigrationError(
            f"cannot start {sys.executable} to check offline migrations: {exc}"
        ) from exc
    return OfflineMigrationCheck(
        findings=_answered_findings(result, selected, declared, env),
        code_label=code_label,
    )


def check_revision_offline_migrations(
    project_dir: Path,
    repo_path: str | Path,
    revision: str,
    *,
    agent_names: Iterable[str] | None = None,
    code_label: str,
) -> OfflineMigrationCheck:
    """:func:`check_offline_migrations` for what checking out ``revision`` installs.

    A revision without a manifest declares nothing and passes.

    Raises:
        OfflineMigrationError: As :func:`manifest_at` and
            :func:`check_offline_migrations`.
    """
    return check_offline_migrations(
        project_dir,
        agent_names=agent_names,
        manifest=lambda: manifest_at(repo_path, revision),
        code_label=code_label,
    )


def _redact(text: str, env: dict) -> str:
    dsn = env.get("KESTREL_DATABASE_URL")
    return text.replace(dsn, "<KESTREL_DATABASE_URL>") if dsn else text


def _answered_findings(
    result: subprocess.CompletedProcess,
    selected: list[str],
    declared: tuple[OfflineMigration, ...] | None,
    env: dict,
) -> tuple[MigrationFinding, ...]:
    """The findings a fresh-interpreter check answered with, validated.

    Anything short of a known finding for every selected agent under every
    migration is a failure to check, never a pass: the child may be newer code,
    and an answer this process misreads must not let a deploy through.
    """
    stdout = result.stdout.decode("utf-8", "replace")
    answer = next(
        (
            line[len(_FRESH_CHECK_RESULT_PREFIX):]
            for line in reversed(stdout.splitlines())
            if line.startswith(_FRESH_CHECK_RESULT_PREFIX)
        ),
        None,
    )
    if result.returncode != 0 or answer is None:
        output = result.stderr.decode("utf-8", "replace") or stdout
        detail = _redact(output.strip()[-2000:], env)
        raise OfflineMigrationError(
            f"{INSTALLED_CODE} could not check offline migrations "
            f"(exit {result.returncode})" + (f": {detail}" if detail else "")
        )
    try:
        payload = json.loads(answer)
        if payload.get("protocol") != _FRESH_CHECK_PROTOCOL:
            raise ValueError(f"unknown protocol {payload.get('protocol')!r}")
        if "error" in payload:
            raise OfflineMigrationError(_redact(str(payload["error"]), env))
        findings = []
        for fields in payload["findings"]:
            if not isinstance(fields["agents"], list):
                raise ValueError("a finding's agents are not a list")
            findings.append(
                MigrationFinding(**{**fields, "agents": tuple(fields["agents"])})
            )
        covered: dict[str, set[str]] = {}
        for finding in findings:
            text = (finding.migration, finding.status, finding.summary,
                    finding.runbook, finding.database, finding.detail,
                    *finding.agents)
            if not all(isinstance(value, str) for value in text):
                raise ValueError("a finding field is not a string")
            if finding.status not in _STATUSES:
                raise ValueError(f"unknown finding status {finding.status!r}")
            if not set(finding.agents) <= set(selected):
                raise ValueError(f"findings for {list(finding.agents)}, asked about {selected}")
            covered.setdefault(finding.migration, set()).update(finding.agents)
        if any(agents != set(selected) for agents in covered.values()):
            raise ValueError(f"findings do not cover every agent asked about ({selected})")
        if declared is not None and set(covered) != {m.id for m in declared}:
            raise ValueError(
                f"findings for {sorted(covered)}, asked about "
                f"{sorted(m.id for m in declared)}"
            )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OfflineMigrationError(
            f"{INSTALLED_CODE} answered the offline migration check in a form "
            f"this process cannot read: {exc}"
        ) from exc
    return tuple(
        dataclasses.replace(finding, detail=_redact(finding.detail, env))
        for finding in findings
    )


# ---------------------------------------------------------------------------
# What the operator and the restart request read
# ---------------------------------------------------------------------------


def _describe(finding: MigrationFinding) -> str:
    agents = ", ".join(finding.agents)
    if finding.status == PENDING:
        return (
            f"{finding.migration}: still pending in {finding.database} "
            f"({agents}): {finding.detail}"
        )
    if finding.status == UNVERIFIED:
        return (
            f"{finding.migration}: {finding.database} ({agents}) could not be "
            f"read, so whether it is done is unknown: {finding.detail}"
        )
    return (
        f"{finding.migration} ({finding.summary}): the code judging it has no "
        f"check for it, so whether {agents} still need it is unknown"
    )


def _runbooks(check: OfflineMigrationCheck) -> dict[str, str]:
    return {f.migration: f.runbook for f in check.blocking}


def refusal_lines(check: OfflineMigrationCheck) -> list[str]:
    """The operator-facing explanation of a blocking check, and how to migrate."""
    statuses = {f.status for f in check.blocking}
    lines = [
        (
            f"{check.code_label} requires offline migrations not known to be "
            f"done; restarting onto it would leave "
            f"{len(check.blocking_agents)} agent(s) unable to start:"
        ),
        *(f"  {_describe(finding)}" for finding in check.blocking),
        *(
            f"Runbook for {migration}: {runbook}"
            for migration, runbook in _runbooks(check).items()
        ),
        (
            "Migrate offline before restarting onto this code: stop every "
            "agent (`kestrel terminate`, and any server started without "
            "`kestrel start`), install without restarting (`kestrel update "
            "--no-restart`, which proceeds once none of them is running), "
            "follow the runbook, then `kestrel start`."
        ),
    ]
    if UNVERIFIED in statuses:
        lines.append(
            "  A database that could not be read: fix what kept it from being "
            "read and retry; `kestrel doctor` diagnoses it."
        )
    if UNKNOWN in statuses:
        lines.append(
            "  A migration this kestrel has no check for: once the release is "
            "installed, `kestrel restart` judges it with the installed code."
        )
    return lines


def refusal_reason(check: OfflineMigrationCheck) -> str:
    """One-line terminal reason for a refused restart request."""
    found = "; ".join(_describe(finding) for finding in check.blocking)
    runbooks = ", ".join(
        f"{migration}: {runbook}" for migration, runbook in _runbooks(check).items()
    )
    return (
        f"offline migration required: restarting onto {check.code_label} would "
        f"leave agents unable to start ({found}). An operator must run the "
        f"offline migration first; an agent cannot. Runbook: {runbooks}."
    )


def check_record(check: OfflineMigrationCheck | None, *, error: str = "") -> dict:
    """What one evaluation of the gate decided, as a restart request records it.

    Every finding, done or not, so a passed check is evidence of what it read.
    ``check`` None with ``error`` records a gate that learned nothing.
    """
    if check is None:
        return {"result": "unverifiable", "code": "", "error": error, "findings": []}
    return {
        "result": "passed" if check.safe else "refused",
        "code": check.code_label,
        "error": "",
        "findings": [dataclasses.asdict(f) for f in check.findings],
    }


def main(argv: list[str] | None = None) -> int:
    """Answer :func:`check_offline_migrations` from this interpreter.

    Prints one result line; the exit status is 0 whenever a result was
    printed, including a check that could not establish what is required.
    """
    parser = argparse.ArgumentParser(
        prog=f"python -m {_FRESH_CHECK_MODULE}",
        description=(
            "Ask each agent database whether the offline migrations a release "
            "requires are done (#3832)."
        ),
    )
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument(
        "--agent", action="append", default=[],
        help="An agent to check; repeat for more. Default: every local agent.",
    )
    parser.add_argument(
        "--manifest-stdin", action="store_true",
        help="Judge the manifest on stdin instead of this installation's own.",
    )
    args = parser.parse_args(argv)
    payload: dict = {"protocol": _FRESH_CHECK_PROTOCOL}
    try:
        if args.manifest_stdin:
            manifest, source = sys.stdin.buffer.read(), "the manifest judged"
        else:
            manifest, source = packaged_manifest(), INSTALLED_CODE
        findings = judge(
            args.project_dir,
            parse_manifest(manifest, source=source),
            agent_names=args.agent or None,
        )
    except OfflineMigrationError as exc:
        payload["error"] = str(exc)
    else:
        payload["findings"] = [dataclasses.asdict(f) for f in findings]
    print(_FRESH_CHECK_RESULT_PREFIX + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
