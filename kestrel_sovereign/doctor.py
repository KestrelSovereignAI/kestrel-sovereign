"""``kestrel doctor`` — diagnose readiness without making any changes.

The doctor produces a structured :class:`DoctorReport` that the CLI
formats. It is also reused as the ``verify`` step at the end of
``kestrel setup``.

Checks performed:

  - ``KESTREL_DATA_KEY`` set in ``.env``
  - ``[llm]`` section present with a non-empty ``route_priority``
  - For each cloud route in ``route_priority``, one of its accepted
    credential env vars is set in ``.env`` (e.g. OpenRouter accepts
    either ``OPENROUTER_API_KEY`` or ``OPENROUTER_MANAGEMENT_API_KEY``)
  - At least one agent registered in ``multi_agent.toml``
  - For each registered agent, ``kestrel_prime.db`` exists
  - For each registered agent, the anchored ``constitution_hash``
    matches the SHA256 of the canonical KESTREL_CONSTITUTION.md.
    Drift here means the agent is silently governing itself by an
    older constitution than what's on disk — see ``_check_constitution_drift``.
  - For each registered agent, the ``governed_by`` graph edge targets the
    anchored ``constitution_hash`` (integrity proof 2) and any per-agent
    ``CONSTITUTION.md`` overlay is anchored and unmodified (#1722). The
    fail-closed integrity audit safe-modes an agent on either at boot;
    doctor surfaces them pre-upgrade so operators can reanchor first
    (#2616) — see ``_check_anchor_consistency``.
  - PostgreSQL hosts provide a reachable
    ``KESTREL_HOLD_EVIDENCE_DATABASE_URL`` on an independent cluster, and
    both runtime roles can read ``pg_catalog.pg_control_system()`` so that
    independence is proved before setup reports ready.
  - Legacy local identity exports are inspected by metadata only. Unsafe
    directory/file modes, links, and non-regular entries are reported without
    opening or parsing package contents.
  - Every pinned semantic resource still matches its manifest digest. One
    mismatch refuses agent boot wholesale, and a CRLF-smudged checkout breaks
    all of them at once — see ``_check_semantic_registry`` (#2924).

This is deliberately minimal. We avoid reaching out to Ollama / OpenAI
— that's flaky in CI and out of scope for "is the config sane?"
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

from kestrel_sovereign._doctor_postgres_probe import (
    ERROR_KIND_CONNECTION,
    ERROR_KIND_DIAGNOSTIC,
    ERROR_KIND_QUERY,
)
from kestrel_sovereign.identity.protected_export import (
    audit_legacy_identity_exports,
    effective_identity_export_roots,
)
from kestrel_sovereign.llm.route_credentials import accepted_credential_envs
from kestrel_sovereign.multi_agent.config import (
    MULTI_AGENT_CONFIG_FILENAME,
    MultiAgentConfig,
)
from kestrel_sovereign.setup.env_file import read_env
from kestrel_sovereign.setup.toml_file import read_toml


@dataclass
class DoctorReport:
    """Outcome of a doctor run.

    Each entry in ``ok`` / ``warn`` / ``fail`` is a single line a user
    can act on. ``fail`` makes ``ready`` False; ``warn`` is informational.
    """

    ok: list[str] = field(default_factory=list)
    warn: list[str] = field(default_factory=list)
    fail: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.fail


def diagnose(project_dir: Path) -> DoctorReport:
    """Run all v1 readiness checks and return a structured report."""
    report = DoctorReport()
    env_path = project_dir / ".env"
    toml_path = project_dir / "kestrel.toml"
    multi_agent_path = project_dir / MULTI_AGENT_CONFIG_FILENAME

    env = read_env(env_path)
    config = read_toml(toml_path)

    # Two different questions, two different readings, deliberately.
    # ``env`` is the *file's* contents: "is KESTREL_DATA_KEY written down where
    # setup would find it" is answered by the file, not by this shell.
    # ``resolved`` is what the agents would actually boot with, and it is the
    # only thing that can say which database holds their governance.
    resolved = runtime_env(project_dir)

    _check_data_key(env, env_path, report)
    _check_llm(config, env, toml_path, report)
    _check_multi_agent(multi_agent_path, project_dir, report)

    # Read each agent's governance ONCE and give the same reading to both
    # checks. They used to resolve and read independently, which on an
    # unreachable PostgreSQL meant paying the connection timeout twice per
    # agent — a ten-agent fleet waiting 100s to be told the database is down,
    # from a tool whose bound is five seconds and whose whole purpose is to
    # answer quickly when the database is down.
    readings = _read_agent_governance(multi_agent_path, project_dir, resolved)

    _check_constitution_drift(readings, report)
    _check_anchor_consistency(readings, report)
    _check_postgres_hold_readiness(resolved, project_dir, readings, report)
    _check_legacy_identity_exports(project_dir, report)
    _check_semantic_registry(report)

    return report


def _check_semantic_registry(report: DoctorReport) -> None:
    """Report pinned semantic resources that would refuse agent boot.

    A single failing pin makes every agent fail to load with "semantic runtime
    capability is unavailable", so this is a readiness question, not a
    curiosity. It reuses the registry's own classifier rather than asking
    ``git check-attr``: bytes answer the question in a wheel install too, where
    there is no git to ask.
    """
    from kestrel_sovereign.knowledge.registry import (
        KnowledgeRegistryError,
        ResourceIntegrityIssue,
        audit_semantic_resources,
    )

    try:
        findings = audit_semantic_resources()
    except (KnowledgeRegistryError, OSError, ImportError) as exc:
        # A manifest this tool cannot parse is precisely what it exists to
        # report. ``KnowledgeRegistryError`` covers that (including malformed
        # TOML, which the registry converts rather than letting escape); the
        # rest cover a manifest that cannot be read at all.
        report.fail.append(f"semantic registry is unusable: {exc}")
        return

    if not findings:
        report.ok.append("semantic registry: all pinned resources verified")
        return

    for issue in ResourceIntegrityIssue:
        affected = [finding for finding in findings if finding.issue is issue]
        if not affected:
            continue
        # Every distinct path, not an example. A CRLF-smudged checkout breaks
        # all 29 pins at once, and naming one of them sends an operator to
        # repair a single file while the fleet stays unbootable.
        paths = sorted({finding.package_resource for finding in affected})
        detail = (
            f"{len(affected)} semantic resource(s) fail their pin "
            f"({issue.value}), which refuses agent boot: {', '.join(paths)}"
        )
        # The remedy is checkout-scoped, so one line repairs the whole group.
        remedy = affected[0].remedy
        report.fail.append(f"{detail} — {remedy}" if remedy else detail)


def _check_legacy_identity_exports(
    project_dir: Path,
    report: DoctorReport,
) -> None:
    """Report unsafe legacy exports without reading sensitive package bytes."""

    roots = effective_identity_export_roots(project_dir)
    findings = audit_legacy_identity_exports(roots)
    if not findings:
        return
    affected_roots = len({finding.root for finding in findings})
    report.warn.append(
        f"{len(findings)} unsafe legacy identity-export filesystem finding(s) "
        f"under {affected_roots} configured data root(s). This metadata-only "
        "check did not read package contents. Run `kestrel identity "
        "harden-exports` to restrict eligible operator-owned exports to 0600."
    )


def format_report(report: DoctorReport) -> str:
    """Render a report as human-readable text suitable for stdout."""
    lines: list[str] = []
    for msg in report.ok:
        lines.append(f"  ✅ {msg}")
    for msg in report.warn:
        lines.append(f"  ⚠️  {msg}")
    for msg in report.fail:
        lines.append(f"  ❌ {msg}")
    if report.ready:
        lines.append("")
        lines.append("Ready. Start with: kestrel start")
    else:
        lines.append("")
        lines.append("Not ready. Fix the items above, or run: kestrel setup")
    return "\n".join(lines)


def _check_data_key(env: dict, env_path: Path, report: DoctorReport) -> None:
    if env.get("KESTREL_DATA_KEY"):
        report.ok.append("KESTREL_DATA_KEY is set")
    else:
        report.fail.append(f"KESTREL_DATA_KEY missing in {env_path}")


def _check_llm(config: dict, env: dict, toml_path: Path, report: DoctorReport) -> None:
    llm = config.get("llm") or {}
    priority = llm.get("route_priority") or []
    if not priority:
        report.fail.append(
            f"[llm] route_priority is empty in {toml_path} — no provider configured"
        )
        return

    report.ok.append(f"[llm] route_priority: {', '.join(priority)}")
    vendors = llm.get("vendors") or {}
    for route_id in priority:
        vendor_key, _, route_key = route_id.partition(":")
        vendor = vendors.get(vendor_key) or {}
        route = (vendor.get("routes") or {}).get(route_key) or {}
        accepted = accepted_credential_envs(route_id, route)
        if not accepted:
            continue
        satisfied = next((name for name in accepted if env.get(name)), None)
        if satisfied:
            report.ok.append(f"{satisfied} set for {route_id}")
        else:
            report.fail.append(
                f"{' or '.join(accepted)} not set in .env (required for {route_id})"
            )


def _check_multi_agent(
    multi_agent_path: Path, project_dir: Path, report: DoctorReport
) -> None:
    multi_agent = MultiAgentConfig.load(multi_agent_path, auto_discover_fallback=False)
    agents = multi_agent.get_local_agents()
    if not agents:
        report.fail.append(
            f"No local agents in {multi_agent_path} — run `kestrel setup agent`"
        )
        return

    report.ok.append(f"{len(agents)} agent(s) registered: {', '.join(agents.keys())}")
    for name, cfg in agents.items():
        db_path = (project_dir / cfg.data_dir / "kestrel_prime.db").resolve()
        if db_path.exists():
            report.ok.append(f"{name}: kestrel_prime.db present")
        else:
            report.fail.append(
                f"{name}: kestrel_prime.db missing at {db_path} — re-run inception"
            )


def runtime_env(project_dir: Path) -> dict:
    """The environment the agents would boot with — without becoming it.

    Delegates to ``paths.spawned_agent_env``, the body of the launcher's own
    ``ProcessManager._load_env``. Doctor iterates the local agents in
    ``multi_agent.toml``, and those are precisely the processes that launcher
    spawns, so its answer is the authoritative one — including its precedence,
    where the project ``.env`` overwrites a conflicting export. Reimplementing
    that here (this function first used ``load_project_env``'s opposite
    ``setdefault`` order) means doctor inspects one database while the agents
    open another, which is this issue's own defect wearing a different hat.

    Reading ``os.environ`` alone was the bug before that. On a standard install
    the PostgreSQL settings live only in the project ``.env`` — that is what
    ``.env.example`` documents and what ``kestrel setup`` writes — and neither
    ``cmd_doctor`` nor ``setup --check`` loads it, so doctor saw an unset
    backend and reported the birth record as current governance on exactly the
    hosts #2892 is about.

    A diagnostic reports on a process; it does not become one. Nothing here is
    exported.
    """
    from kestrel_sovereign.paths import spawned_agent_env

    return spawned_agent_env(project_dir)


def _anchor_is_the_runtime_database(env: dict) -> bool:
    """Whether the local ``kestrel_prime.db`` is what the agents actually read.

    Same rule as ``agent_manager._initialize_agent`` and
    ``setup.constitution_reanchor.resolve_reanchor_target``: PostgreSQL **and**
    a DSN, or SQLite. Copied rather than imported so doctor keeps its light
    dependency profile — and copied *exactly*, with no ``.strip()`` and no
    refusal on an unknown backend, because being stricter than the runtime is
    the same bug with the databases exchanged.

    ``env`` is :func:`runtime_env`'s resolution, not ``os.environ``.
    """
    backend = env.get("KESTREL_DB_BACKEND", "sqlite").lower()
    return not (backend == "postgres" and env.get("KESTREL_DATABASE_URL"))


_POSTGRES_CLUSTER_ID_SQL = (
    "SELECT system_identifier::text FROM pg_catalog.pg_control_system()"
)


def _postgres_cluster_probe_source(
    dsn: str,
    env: dict[str, str],
    project_dir: Path,
) -> _GovernanceSource:
    """Build the existing redaction/launch context for a cluster probe."""

    return _GovernanceSource(
        anchor_path=Path("."),
        agent_did="",
        dsn=dsn,
        dsn_identity=_postgres_redaction_identity(dsn, env),
        postgres_home=env.get("HOME") or env.get("USERPROFILE"),
        postgres_env=dict(env),
        postgres_cwd=str(project_dir.resolve()),
    )


def _read_postgres_cluster_identity(
    dsn: str,
    *,
    label: str,
    env: dict[str, str],
    project_dir: Path,
    report: DoctorReport,
) -> str | None:
    """Read the same privileged cluster identity Hold requires at boot."""

    source = _postgres_cluster_probe_source(dsn, env, project_dir)
    try:
        rows = _fetch_postgres_rows_isolated(
            dsn,
            _POSTGRES_CLUSTER_ID_SQL,
            postgres_home=source.postgres_home,
            postgres_env=source.postgres_env,
            postgres_cwd=source.postgres_cwd,
            dsn_identity=source.dsn_identity,
        )
    except Exception as exc:  # noqa: BLE001 - asyncpg has its own exception tree
        failure = _postgres_probe_failure_kind(exc)
        if failure == "connection":
            report.fail.append(
                f"PostgreSQL Hold {label} database is unreachable: "
                f"{_safe(exc, source)}"
            )
        else:
            report.fail.append(
                f"PostgreSQL Hold {label} cluster identity NOT verified: "
                "the runtime role requires EXECUTE on "
                f"pg_catalog.pg_control_system() ({_safe(exc, source)})"
            )
        return None

    if (
        len(rows) != 1
        or len(rows[0]) != 1
        or not isinstance(rows[0][0], str)
        or not rows[0][0].strip()
    ):
        report.fail.append(
            f"PostgreSQL Hold {label} cluster identity NOT verified: "
            "pg_catalog.pg_control_system() returned invalid data"
        )
        return None
    return rows[0][0]


def _runtime_postgres_connection_failed(readings: list[_AgentGovernance]) -> bool:
    """Reuse an already-observed runtime outage instead of reconnecting."""

    return any(
        isinstance(reading.source, _UnreadableDB)
        and reading.source.postgres_failure
        in {"connection", "diagnostic_timeout", "diagnostic_tooling"}
        for reading in readings
    )


def _check_postgres_hold_readiness(
    env: dict[str, str],
    project_dir: Path,
    readings: list[_AgentGovernance],
    report: DoctorReport,
) -> None:
    """Verify the mandatory independent PostgreSQL Hold custody service."""

    backend = env.get("KESTREL_DB_BACKEND", "sqlite").lower()
    primary_dsn = env.get("KESTREL_DATABASE_URL")
    if backend != "postgres" or not primary_dsn:
        return

    evidence_dsn = env.get("KESTREL_HOLD_EVIDENCE_DATABASE_URL")
    if not evidence_dsn:
        report.fail.append(
            "KESTREL_HOLD_EVIDENCE_DATABASE_URL is required for PostgreSQL "
            "Hold rollback evidence"
        )
        return
    if evidence_dsn == primary_dsn:
        report.fail.append(
            "KESTREL_HOLD_EVIDENCE_DATABASE_URL must identify an independent "
            "PostgreSQL cluster"
        )
        return
    primary_identity = None
    if _runtime_postgres_connection_failed(readings):
        report.fail.append(
            "PostgreSQL Hold primary cluster identity NOT verified because "
            "runtime database reachability was not established"
        )
    else:
        primary_identity = _read_postgres_cluster_identity(
            primary_dsn,
            label="primary",
            env=env,
            project_dir=project_dir,
            report=report,
        )
    evidence_identity = _read_postgres_cluster_identity(
        evidence_dsn,
        label="evidence",
        env=env,
        project_dir=project_dir,
        report=report,
    )
    if primary_identity is None or evidence_identity is None:
        return
    if primary_identity == evidence_identity:
        report.fail.append(
            "PostgreSQL Hold evidence requires an independent PostgreSQL cluster"
        )
        return
    report.ok.append("PostgreSQL Hold evidence: independent clusters verified")


@dataclass(frozen=True)
class _GovernanceSource:
    """Where an agent's **current** governance lives, and how to read it.

    Doctor reads agent databases without the AsyncStorage stack — deliberately,
    so a diagnostic tool cannot fail because the thing it is diagnosing fails
    to import. That was right on SQLite, where ``kestrel_prime.db`` *is* the
    database the runtime reads, and wrong on a PostgreSQL host, where the
    anchor holds the birth record (#2871) and the live governance is elsewhere.
    Doctor then reported birth-time state as current — permanently flagging
    drift after any legitimate reanchor, and prescribing a repair that
    correctly answers "nothing to do" (#2892). Two governance tools
    contradicting each other is how operators learn to ignore the one that
    cries wolf.

    Staying out of the storage stack costs nothing here: every property doctor
    reads is plaintext JSON in ``graph_nodes.properties`` and every edge is a
    plain row, so no ``KESTREL_DATA_KEY`` is involved. PostgreSQL reads run in
    a bounded child using the same asyncpg runtime as the agent.

    ``agent_did`` always comes from the **anchor**, which is where identity is
    born on every backend (#2871, #2894), and it is required on both. On
    PostgreSQL one database holds every local agent, so an unscoped read would
    pick a tenant by incidental row order. On SQLite the file holds one agent,
    but the DID is still needed: the runtime's store is bound on that backend
    too, and its reads carry an ownership predicate keyed by DID.
    """

    anchor_path: Path
    agent_did: str
    dsn: str | None = None
    #: Explicit DSN and environment-derived identities to redact from worker
    #: failures.  Asyncpg receives the launcher environment unchanged, so the
    #: redaction boundary must cover values that never appear in ``dsn``.
    dsn_identity: tuple | None = None
    #: HOME seen by asyncpg in the spawned agent, retained only for redaction
    #: of default credential and TLS paths in driver errors.
    postgres_home: str | None = None
    #: Complete environment resolved by the launcher for the spawned agent.
    #: The asyncpg worker receives this copy unchanged, just like the agent.
    postgres_env: dict[str, str] | None = None
    #: Working directory ProcessManager gives the spawned agent. Relative GSS
    #: configuration/cache locations must resolve beneath the same directory.
    postgres_cwd: str | None = None
    #: Whether the #2649 ownership backfill has already been recorded as
    #: complete. Boot re-runs it exactly once; until it has, a row with no
    #: witness is repaired at the next start, so its absence predicts nothing.
    ownership_settled: bool = False
    #: Whether ``graph_node_owners`` / ``graph_edge_owners`` exist here.
    #: False on a database predating the ownership migration (#2649), where
    #: the scoped reads below cannot run at all — see ``_has_ownership_ledger``.
    ownership_ledger: bool = True

    @property
    def reads_the_anchor(self) -> bool:
        return self.dsn is None

    def describe(self) -> str:
        return str(self.anchor_path) if self.reads_the_anchor else "PostgreSQL"


def _discover_agent_did(anchor_path: Path):
    """The DID of the agent this anchor belongs to, or a sentinel.

    Deliberately the one unscoped read doctor performs, and deliberately
    against the local file on every backend. It asks *whose anchor is this*,
    and identity is born in ``kestrel_prime.db`` regardless of where governance
    later lives (#2871, #2894) — so scoping this read by the very DID it exists
    to discover would be circular.
    """
    try:
        with sqlite3.connect(str(anchor_path)) as conn:
            rows = conn.execute(_DISCOVER_AGENT_SQLITE).fetchall()
    except sqlite3.DatabaseError as exc:
        # 'file is not a database' (sqlcipher-encrypted) lands here.
        return _UnreadableDB(reason=f"DB unreadable ({exc})")
    except sqlite3.Error as exc:
        return _UnreadableDB(reason=f"sqlite error ({exc})")

    if len(rows) > 1:
        # The same refusal boot makes, for the same reason: a damaged or
        # half-imported anchor holding two agent roots has no single answer to
        # "whose governance is this", and choosing one would have doctor
        # certify a tenant it picked by row order.
        return _UnreadableDB(
            reason=(
                f"the local anchor {anchor_path} holds more than one agent "
                "root, so it cannot say whose governance to check — the "
                "runtime refuses to boot from it for the same reason"
            )
        )
    if not rows or not rows[0][0]:
        return _NoAgentNode()
    return rows[0][0]


def _has_ownership_ledger(source: _GovernanceSource) -> bool:
    """Whether this database has the ownership tables at all.

    A database written before the ownership migration (#2649) has neither
    ``graph_node_owners`` nor ``graph_edge_owners``; ``AsyncStorage`` creates
    and backfills them at boot. Doctor exists to be run *before* that boot
    (#2616), so it meets un-migrated databases routinely, and the tenant-scoped
    reads simply cannot execute against one.

    Absent is not the same as unwitnessed, and conflating them was a real
    regression: the scoped query raised ``no such table``, which became a
    warning, which skipped the hash and edge checks entirely — and warnings
    leave ``ready`` true, so doctor would certify governance it never examined.
    When the ledger is absent, the legacy unscoped reads are the faithful ones:
    every row in a per-agent file belongs to that agent, and the backfill is
    what will shortly say so.

    Returns ``_UnreadableDB`` rather than a guess when the probe itself cannot
    run. Guessing ``True`` there looked harmless and was not: on an unreachable
    PostgreSQL this waited out a whole connection timeout, discarded the
    failure, and the node read then opened the same DSN and waited again —
    reintroducing, one function earlier, the doubled timeout that sharing a
    per-agent reading had just removed.
    """
    if source.reads_the_anchor:
        try:
            with sqlite3.connect(str(source.anchor_path)) as conn:
                row = conn.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='graph_node_owners'"
                ).fetchone()
                settled = conn.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='schema_backfills'"
                ).fetchone()
                if settled is not None:
                    settled = conn.execute(
                        "SELECT 1 FROM schema_backfills WHERE name = ?",
                        (_OWNERSHIP_BACKFILL,),
                    ).fetchone()
        except sqlite3.DatabaseError as exc:
            return _UnreadableDB(reason=f"DB unreadable ({exc})")
        except sqlite3.Error as exc:
            return _UnreadableDB(reason=f"sqlite error ({exc})")
        return _LedgerState(present=row is not None, settled=settled is not None)

    try:
        rows = _fetch_rows(
            source,
            "SELECT 1",
            "SELECT to_regclass('graph_nodes') IS NOT NULL, "
            "to_regclass('graph_node_owners') IS NOT NULL, "
            "to_regclass('schema_backfills') IS NOT NULL",
        )
        if rows and not rows[0][0]:
            # No graph schema at all: a PostgreSQL database that has never been
            # booted against. ``AsyncDatabase.postgres()`` creates the schema
            # and replicates the anchor on first boot, so this is a valid
            # starting state, not a broken one — and the governance that will
            # be audited is the anchor's. Reporting it unreadable made doctor
            # answer "Not ready" to a correctly configured first boot.
            return _SchemaAbsent()
    except Exception as exc:  # noqa: BLE001 — asyncpg raises its own tree
        return _postgres_unreadable(
            exc,
            reason=f"cannot read {source.describe()} ({_safe(exc, source)})",
        )
    # The marker is read in a *second* statement, on purpose. PostgreSQL
    # resolves every relation in a statement before evaluating any of it, so a
    # ``SELECT ... FROM schema_backfills`` folded into the probe above raises
    # ``undefined_table`` on a database that does not have it yet — which is
    # precisely the never-booted database the ``to_regclass`` checks exist to
    # recognise. One statement made ``_SchemaAbsent`` unreachable and put a
    # valid first boot back to "unreadable", undoing the fix three commits
    # earlier. Costs nothing in practice: this whole probe is memoised per DSN.
    if not (rows and rows[0][2]):
        return _LedgerState(present=bool(rows and rows[0][1]), settled=False)

    try:
        marker = _fetch_rows(
            source,
            "SELECT 1",
            "SELECT 1 FROM schema_backfills WHERE name = $1",
            postgres_params=(_OWNERSHIP_BACKFILL,),
        )
    except Exception as exc:  # noqa: BLE001 — asyncpg raises its own tree
        return _postgres_unreadable(
            exc,
            reason=f"cannot read {source.describe()} ({_safe(exc, source)})",
        )
    return _LedgerState(present=bool(rows[0][1]), settled=bool(marker))


def _resolve_governance_source(
    anchor_path: Path,
    env: dict,
    project_dir: Path,
    ledger_by_dsn: dict | None = None,
):
    """Resolve where to read this agent's governance from, and as whom.

    Returns a :class:`_GovernanceSource`, or an ``_UnreadableDB`` sentinel when
    the anchor cannot name the tenant to read as.

    ``ledger_by_dsn`` memoises the ownership-ledger probe across agents that
    share a database — including a *failed* probe, since one unreachable
    endpoint is unreachable for the whole fleet and re-proving that per agent
    is the outage cost this bound exists to avoid.
    """
    agent_did = _discover_agent_did(anchor_path)
    if isinstance(agent_did, _UnreadableDB):
        return agent_did
    if isinstance(agent_did, _NoAgentNode):
        return _UnreadableDB(
            reason=(
                f"the local anchor {anchor_path} names no agent, so there is "
                "no tenant to read its governance as"
            )
        )

    if _anchor_is_the_runtime_database(env):
        source = _GovernanceSource(anchor_path=anchor_path, agent_did=agent_did)
    else:
        runtime_dsn = env["KESTREL_DATABASE_URL"]
        try:
            _doctor_postgres_timeout_seconds(env)
        except _DoctorPostgresConfigurationError as exc:
            unsafe_source = _GovernanceSource(
                anchor_path=anchor_path,
                agent_did=agent_did,
                dsn=runtime_dsn,
                postgres_env=dict(env),
            )
            return _UnreadableDB(
                reason=(
                    "PostgreSQL doctor configuration is invalid "
                    f"({_safe(exc, unsafe_source)})"
                ),
                postgres_failure="doctor_configuration",
            )
        source = _GovernanceSource(
            anchor_path=anchor_path,
            agent_did=agent_did,
            dsn=runtime_dsn,
            dsn_identity=_postgres_redaction_identity(runtime_dsn, env),
            postgres_home=env.get("HOME") or env.get("USERPROFILE"),
            postgres_env=dict(env),
            postgres_cwd=str(project_dir.resolve()),
        )
    # Keyed on the DSN, or on the anchor path for a SQLite host where each
    # agent genuinely has its own file and its own answer.
    cache_key = source.dsn if source.dsn else str(source.anchor_path)
    if ledger_by_dsn is not None and cache_key in ledger_by_dsn:
        ledger = ledger_by_dsn[cache_key]
    else:
        ledger = _has_ownership_ledger(source)
        if ledger_by_dsn is not None:
            ledger_by_dsn[cache_key] = ledger

    if isinstance(ledger, (_UnreadableDB, _SchemaAbsent)):
        return ledger
    return replace(
        source,
        ownership_ledger=ledger.present,
        ownership_settled=ledger.settled,
    )


#: Seconds Doctor lets the runtime-equivalent asyncpg probe run before it is
#: killed and reaped.  This is a parent-process diagnostic deadline, not a
#: connection argument, so it cannot override DSN, service-file, or PG*
#: settings the spawned agent would use.
_CONNECT_TIMEOUT_SECONDS = 5
_POSTGRES_PROBE_GRACE_SECONDS = 5
_POSTGRES_TIMEOUT_ENV = "KESTREL_DOCTOR_POSTGRES_TIMEOUT_SECONDS"
_MAX_POSTGRES_TIMEOUT_SECONDS = 3600


class _DoctorPostgresConfigurationError(ValueError):
    """A doctor-only setting is invalid; the runtime is unaffected."""


def _doctor_postgres_timeout_seconds(env: dict) -> int:
    """Return the bounded probe budget configured in the launcher environment."""
    raw = env.get(_POSTGRES_TIMEOUT_ENV)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = str(_CONNECT_TIMEOUT_SECONDS)
    try:
        timeout = int(raw)
    except (TypeError, ValueError) as exc:
        raise _DoctorPostgresConfigurationError(
            f"{_POSTGRES_TIMEOUT_ENV} must be a positive integer"
        ) from exc
    if not 1 <= timeout <= _MAX_POSTGRES_TIMEOUT_SECONDS:
        raise _DoctorPostgresConfigurationError(
            f"{_POSTGRES_TIMEOUT_ENV} must be between 1 and "
            f"{_MAX_POSTGRES_TIMEOUT_SECONDS} seconds"
        )
    return timeout


def _safe(exc: object, source: _GovernanceSource) -> str:
    """Return a driver error with connection identities and secrets removed."""
    text = str(exc)
    if not source.dsn:
        return text

    text = text.replace(source.dsn, "<dsn>")
    env = source.postgres_env or {}
    secrets = set(_dsn_secrets(source.dsn))
    secrets.update(
        value
        for name in _POSTGRES_SECRET_ENV_FIELDS
        if isinstance((value := env.get(name)), str) and value
    )
    for secret in sorted(secrets, key=len, reverse=True):
        if len(secret) > 2:
            text = text.replace(secret, "<redacted>")
        else:
            text = re.sub(
                rf"(?<![\w-]){re.escape(secret)}(?![\w-])",
                "<redacted>",
                text,
            )

    for field_name, value in _postgres_connection_files(source.dsn, env):
        text = text.replace(value, f"<{field_name}>")
    identities = (
        source.dsn_identity
        if source.dsn_identity is not None
        else _postgres_redaction_identity(source.dsn, env)
    )
    for field_name, value in identities:
        text = text.replace(value, f"<{field_name}>")
    if source.postgres_home and len(source.postgres_home) > 2:
        text = text.replace(source.postgres_home, "<home>")
    return text


_POSTGRES_SECRET_ENV_FIELDS = ("PGPASSWORD", "PGSSLPASSWORD")
_POSTGRES_CONNECTION_FILE_QUERY_FIELDS = frozenset(
    {
        "passfile",
        "servicefile",
        "sslcert",
        "sslcrl",
        "sslkey",
        "sslrootcert",
    }
)
_POSTGRES_CONNECTION_FILE_ENV_FIELDS = {
    "PGPASSFILE": "passfile",
    "PGSERVICEFILE": "servicefile",
    "PGSSLCERT": "sslcert",
    "PGSSLCRL": "sslcrl",
    "PGSSLKEY": "sslkey",
    "PGSSLROOTCERT": "sslrootcert",
    "KRB5CCNAME": "kerberos_cache",
    "KRB5_CONFIG": "kerberos_config",
}
_POSTGRES_IDENTITY_ENV_FIELDS = {
    "PGHOST": "host",
    "PGUSER": "user",
    "PGDATABASE": "dbname",
    "PGSERVICE": "service",
}
_DSN_IDENTITY_FIELDS = ("host", "user", "dbname", "service")


def _dsn_query(dsn: str) -> tuple[tuple[str, str], ...]:
    """Return URI query fields for redaction without validating the runtime DSN."""
    try:
        return tuple(parse_qsl(urlsplit(dsn).query, keep_blank_values=True))
    except (TypeError, ValueError):
        return ()


def _dsn_secrets(dsn: str) -> tuple:
    """Return every URI-embedded credential token, longest first."""
    secrets: set[str] = set()
    for query_field, value in _dsn_query(dsn):
        if query_field in {"password", "sslpassword"} and value:
            secrets.add(value)

    try:
        raw_query = urlsplit(dsn).query
    except (TypeError, ValueError):
        raw_query = ""
    for match in re.finditer(
        r"(?:^|&)(?:password|sslpassword)=([^&#]+)", raw_query, re.IGNORECASE
    ):
        secrets.add(match.group(1))

    match = re.search(r"://[^:/?#]*:([^@/?#]+)@", dsn)
    if match:
        encoded_password = match.group(1)
        secrets.add(encoded_password)
        secrets.add(unquote(encoded_password))
    return tuple(sorted(secrets, key=len, reverse=True))


def _dsn_connection_files(dsn: str) -> tuple:
    """Return connection-file paths explicitly present in the runtime URI."""
    found = [
        (field, value)
        for field, value in _dsn_query(dsn)
        if field in _POSTGRES_CONNECTION_FILE_QUERY_FIELDS
        and isinstance(value, str)
        and len(value) > 2
    ]
    return tuple(sorted(found, key=lambda pair: len(pair[1]), reverse=True))


def _postgres_connection_files(dsn: str, env: dict) -> tuple:
    found = set(_dsn_connection_files(dsn))
    found.update(
        (field_name, value)
        for env_name, field_name in _POSTGRES_CONNECTION_FILE_ENV_FIELDS.items()
        if isinstance((value := env.get(env_name)), str) and len(value) > 2
    )
    return tuple(sorted(found, key=lambda pair: len(pair[1]), reverse=True))


def _dsn_identity(dsn: str, *, include_host: bool = True) -> tuple:
    """Return identity-bearing URI fields without asking a different driver."""
    found: set[tuple[str, str]] = set()
    try:
        parts = urlsplit(dsn)
    except (TypeError, ValueError):
        parts = None

    if parts is not None:
        if parts.username:
            found.add(("user", unquote(parts.username)))
        if include_host and parts.hostname:
            found.add(("host", unquote(parts.hostname)))
        if parts.path and parts.path != "/":
            found.add(("dbname", unquote(parts.path.removeprefix("/"))))

    for field_name, value in _dsn_query(dsn):
        normalized = "dbname" if field_name == "database" else field_name
        if normalized not in _DSN_IDENTITY_FIELDS:
            continue
        if normalized == "host" and not include_host:
            continue
        for member in value.split(",") if normalized == "host" else (value,):
            found.add((normalized, unquote(member)))

    return tuple(
        sorted(
            (
                (field_name, value)
                for field_name, value in found
                if isinstance(value, str) and len(value) > 2
            ),
            key=lambda pair: len(pair[1]),
            reverse=True,
        )
    )


def _postgres_redaction_identity(dsn: str, env: dict) -> tuple:
    """Cover identities supplied only through the spawned-agent environment."""
    found = set(_dsn_identity(dsn))
    found.update(
        (field_name, value)
        for env_name, field_name in _POSTGRES_IDENTITY_ENV_FIELDS.items()
        if isinstance((value := env.get(env_name)), str) and len(value) > 2
    )
    return tuple(sorted(found, key=lambda pair: len(pair[1]), reverse=True))


def _fetch_rows(
    source: _GovernanceSource,
    sqlite_sql: str,
    postgres_sql: str,
    sqlite_params: tuple = (),
    postgres_params: tuple = (),
) -> list:
    """Run one read-only query through the same database driver as the runtime."""
    if source.reads_the_anchor:
        with sqlite3.connect(str(source.anchor_path)) as conn:
            return conn.execute(sqlite_sql, sqlite_params).fetchall()

    return _fetch_postgres_rows_isolated(
        source.dsn,
        postgres_sql,
        postgres_params,
        postgres_home=source.postgres_home,
        postgres_env=source.postgres_env,
        postgres_cwd=source.postgres_cwd,
        dsn_identity=source.dsn_identity,
    )


class _PostgresProbeError(RuntimeError):
    """A safely transported failure from the isolated asyncpg probe."""


class _PostgresProbeConnectionError(_PostgresProbeError):
    """The spawned runtime's asyncpg connection failed."""


class _PostgresProbeQueryError(_PostgresProbeError):
    """Asyncpg connected, but the read-only governance query failed."""


class _PostgresProbeTimeoutError(_PostgresProbeError):
    """The bounded worker was terminated with its result still unknown."""

    def __init__(self, message: str, *, partial_diagnostic: str = "") -> None:
        super().__init__(message)
        self.partial_diagnostic = partial_diagnostic


def _postgres_probe_failure_kind(exc: BaseException) -> str:
    """Classify runtime impact without inferring it from rendered text."""
    if isinstance(exc, _PostgresProbeConnectionError):
        return "connection"
    if isinstance(exc, _PostgresProbeQueryError):
        return "runtime_database"
    if isinstance(exc, _PostgresProbeTimeoutError):
        return "diagnostic_timeout"
    return "diagnostic_tooling"


def _postgres_unreadable(exc: BaseException, *, reason: str):
    """Build an unreadable sentinel with explicit probe provenance."""
    return _UnreadableDB(
        reason=reason,
        postgres_failure=_postgres_probe_failure_kind(exc),
        postgres_partial_diagnostic=bool(
            isinstance(exc, _PostgresProbeTimeoutError) and exc.partial_diagnostic
        ),
    )


def _source_unreadable(
    source: _GovernanceSource,
    exc: BaseException,
    *,
    reason: str,
):
    """Preserve PostgreSQL probe provenance only for PostgreSQL reads."""
    if source.reads_the_anchor:
        return _UnreadableDB(reason=reason)
    return _postgres_unreadable(exc, reason=reason)


def _postgres_probe_env(resolved_env: dict[str, str]) -> dict[str, str]:
    """Copy the environment ProcessManager gives the spawned agent."""
    child_env = dict(resolved_env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    return child_env


def _postgres_fetch_rows_in_process(
    dsn: str,
    sql: str,
    params: tuple | list,
    *,
    connect=None,
) -> list:
    """Exercise the asyncpg worker seam in-process for focused tests."""
    from kestrel_sovereign._doctor_postgres_probe import (
        ProbeConnectionError,
        ProbeError,
        ProbeQueryError,
        fetch_rows_in_process,
    )

    try:
        return fetch_rows_in_process(dsn, sql, params, connect=connect)
    except ProbeConnectionError as exc:
        raise _PostgresProbeConnectionError(str(exc)) from exc
    except ProbeQueryError as exc:
        raise _PostgresProbeQueryError(str(exc)) from exc
    except ProbeError as exc:
        raise _PostgresProbeError(str(exc)) from exc


def _postgres_probe_timeout_seconds(env: dict) -> int:
    """Bound worker import, connect, query, serialization, and shutdown."""
    return _doctor_postgres_timeout_seconds(env) + _POSTGRES_PROBE_GRACE_SECONDS


def _probe_output_text(output: object) -> str:
    """Decode worker output without changing redaction-sensitive bytes."""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if not isinstance(output, str):
        return ""
    return output


def _redact_probe_output(
    output: str,
    dsn: str,
    postgres_home: str | None,
    dsn_identity: tuple | None,
    postgres_env: dict[str, str] | None,
) -> str:
    """Redact a worker fragment before it enters an exception."""
    if not output:
        return ""
    source = _GovernanceSource(
        anchor_path=Path("."),
        agent_did="",
        dsn=dsn,
        dsn_identity=dsn_identity,
        postgres_home=postgres_home,
        postgres_env=postgres_env,
    )
    return _safe(output, source)


def _redacted_probe_output_tail(
    output: object,
    dsn: str,
    postgres_home: str | None,
    dsn_identity: tuple | None,
    postgres_env: dict[str, str] | None,
    *,
    limit: int = 1000,
) -> str:
    """Redact raw worker output before normalizing and bounding it."""
    decoded = _probe_output_text(output)
    redacted = _redact_probe_output(
        decoded,
        dsn,
        postgres_home,
        dsn_identity,
        postgres_env,
    )
    normalized = " ".join(redacted.strip().split())
    return normalized[-limit:]


def _partial_probe_diagnostic(
    exc: subprocess.TimeoutExpired,
    stderr: object,
    *,
    limit: int = 2000,
) -> str:
    """Keep privacy-safe phase breadcrumbs, never timeout stdout/JSON rows."""
    allowed = {
        "PostgreSQL diagnostic phase: connecting",
        "PostgreSQL diagnostic phase: connected; querying",
    }
    breadcrumbs: list[str] = []
    for value in (exc.stderr, stderr):
        for line in _probe_output_text(value).splitlines():
            phase = line.strip()
            if phase in allowed and phase not in breadcrumbs:
                breadcrumbs.append(phase)
    return "; ".join(f"stderr: {phase}" for phase in breadcrumbs)[:limit]


def _fetch_postgres_rows_isolated(
    dsn: str,
    sql: str,
    params: tuple = (),
    *,
    postgres_home: str | None = None,
    postgres_env: dict[str, str] | None = None,
    postgres_cwd: str | None = None,
    dsn_identity: tuple | None = None,
) -> list:
    """Run asyncpg under the spawned agent's executable, environment, and cwd."""
    try:
        payload = json.dumps({"dsn": dsn, "sql": sql, "params": params})
    except (TypeError, ValueError) as exc:
        raise _PostgresProbeError(
            "PostgreSQL diagnostic query parameters are not transportable"
        ) from exc

    child_env = _postgres_probe_env(
        postgres_env if postgres_env is not None else os.environ
    )
    command = [
        sys.executable,
        "-m",
        "kestrel_sovereign._doctor_postgres_probe",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            cwd=postgres_cwd,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise _PostgresProbeError(
            "isolated PostgreSQL diagnostic process could not start"
        ) from exc

    try:
        stdout, stderr = process.communicate(
            payload,
            timeout=_postgres_probe_timeout_seconds(child_env),
        )
    except subprocess.TimeoutExpired as exc:
        process.kill()
        _stdout, stderr = process.communicate()
        detail = _partial_probe_diagnostic(exc, stderr)
        message = (
            "isolated PostgreSQL diagnostic process exceeded its bounded "
            "timeout and was terminated"
        )
        raise _PostgresProbeTimeoutError(
            f"{message}; partial diagnostic: {detail}" if detail else message,
            partial_diagnostic=detail,
        ) from exc
    except OSError as exc:
        try:
            process.kill()
        except OSError:
            pass
        process.wait()
        raise _PostgresProbeError(
            "isolated PostgreSQL diagnostic process communication failed"
        ) from exc
    except BaseException:
        try:
            process.kill()
        except OSError:
            pass
        process.wait()
        raise

    if process.returncode:
        detail = _redacted_probe_output_tail(
            stderr,
            dsn,
            postgres_home,
            dsn_identity,
            child_env,
        )
        message = "isolated PostgreSQL diagnostic process exited unexpectedly"
        raise _PostgresProbeError(f"{message}: {detail}" if detail else message)
    try:
        response = json.loads(stdout)
    except (TypeError, ValueError) as exc:
        raise _PostgresProbeError(
            "isolated PostgreSQL diagnostic process returned invalid data"
        ) from exc
    if not isinstance(response, dict):
        raise _PostgresProbeError(
            "isolated PostgreSQL diagnostic process returned invalid data"
        )
    if response.get("ok") is not True:
        error = response.get("error")
        message = (
            error
            if isinstance(error, str) and error
            else "PostgreSQL diagnostic query failed"
        )
        message = _redacted_probe_output_tail(
            message,
            dsn,
            postgres_home,
            dsn_identity,
            child_env,
        )
        error_type = {
            ERROR_KIND_CONNECTION: _PostgresProbeConnectionError,
            ERROR_KIND_DIAGNOSTIC: _PostgresProbeError,
            ERROR_KIND_QUERY: _PostgresProbeQueryError,
        }.get(response.get("kind"), _PostgresProbeError)
        raise error_type(message)

    rows = response.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, list) for row in rows):
        raise _PostgresProbeError(
            "isolated PostgreSQL diagnostic process returned invalid rows"
        )
    return [tuple(row) for row in rows]


@dataclass(frozen=True)
class _AgentGovernance:
    """One agent's governance, read once and shared by every check.

    ``source`` is a :class:`_GovernanceSource` or an ``_UnreadableDB``;
    ``node`` is ``(node_id, label, properties)`` or a read sentinel. Both
    checks below interpret the same values rather than fetching their own, so
    an agent costs one resolve and one row read per ``diagnose`` — not one per
    check, which on an unreachable PostgreSQL doubled every connection timeout.
    """

    name: str
    agent_dir: Path
    source: object
    node: object
    #: True when the runtime database holds no row for this agent yet and the
    #: reading above therefore comes from the anchor — see
    #: ``_read_agent_governance``.
    pending_replication: bool = False


def _is_placeholder_node(node: object) -> bool:
    """Whether this read is a boot-fabricated stand-in rather than a record.

    Delegates to ``identity.birth_record.is_fabricated_placeholder``, the
    predicate boot itself uses, rather than restating the shape here — a copy
    would drift, and this is the same "don't re-describe a constant you can
    import" mistake that silently disabled a fix in the sibling issue.

    ``is_fabricated_placeholder`` takes a ``GraphNode``; doctor holds a
    ``(node_id, label, properties)`` tuple read with stock drivers, so the shape
    is adapted here rather than doctor being made to build graph objects. The
    label is read from the row, never synthesised: the predicate matches label
    *and* properties, and handing it a label built from the node id would make
    it always agree on that half — a check loosened to fit its own adapter.
    """
    if not isinstance(node, tuple) or len(node) != 3:
        return False
    node_id, label, properties = node
    if not isinstance(properties, dict):
        return False

    from kestrel_sovereign.identity.birth_record import (
        is_fabricated_placeholder,
    )
    from kestrel_sovereign.storage.async_graph_store import GraphNode

    return is_fabricated_placeholder(
        GraphNode(
            node_id=node_id,
            node_type="agent",
            label=label,
            properties=properties,
        ),
        node_id,
    )


def _row_physically_exists(source: _GovernanceSource) -> bool:
    """Whether the agent row is in the table at all, ownership aside.

    The scoped read cannot tell "no row" from "a row this agent does not own",
    and the two need opposite handling: first-boot replication repairs the
    first and *cannot* repair the second, because ``AsyncGraphStore.add_node``
    refuses to claim an existing unowned or foreign-owned row. Treating an
    unowned row as an empty runtime let doctor judge the anchor instead and
    report Ready for a host whose boot cannot get past its own agent node.

    By ``node_id`` alone, without the ``node_type = 'agent'`` filter: a row
    holding this DID under another type still collides on the primary key, and
    ``add_node`` refuses it just the same, so it is not absent in any sense
    that matters to the caller.

    Fails closed to ``True`` — "something is there, do not call this empty" —
    so a probe that cannot run never manufactures a pending-replication verdict.
    """
    try:
        rows = _fetch_rows(
            source,
            "SELECT 1 FROM graph_nodes WHERE node_id = ?",
            "SELECT 1 FROM graph_nodes WHERE node_id = $1",
            sqlite_params=(source.agent_did,),
            postgres_params=(source.agent_did,),
        )
    except Exception:  # noqa: BLE001 — the caller's own read reports it
        return True
    return bool(rows)


def _read_agent_governance(
    multi_agent_path: Path, project_dir: Path, env: dict
) -> list[_AgentGovernance]:
    """Resolve and read every registered local agent's governance, once each.

    Agents whose anchor file is missing are skipped silently: that is already
    a fail from ``_check_multi_agent``, and re-reporting it would say the same
    thing twice in one report.

    The ownership-ledger probe is memoised per DSN. Every local agent on a
    PostgreSQL host shares one database, so probing per agent meant a
    black-holed endpoint cost the connection timeout once per agent — a
    ten-agent fleet waiting fifty seconds under a five-second bound. The
    schema question is a property of the database, not of the tenant asking.
    """
    multi_agent = MultiAgentConfig.load(multi_agent_path, auto_discover_fallback=False)
    ledger_by_dsn: dict = {}
    readings: list[_AgentGovernance] = []
    for name, cfg in multi_agent.get_local_agents().items():
        agent_dir = (project_dir / cfg.data_dir).resolve()
        db_path = agent_dir / "kestrel_prime.db"
        if not db_path.exists():
            continue

        source = _resolve_governance_source(db_path, env, project_dir, ledger_by_dsn)
        node = (
            source
            if isinstance(source, (_UnreadableDB, _SchemaAbsent))
            else _read_agent_node(source)
        )

        # Two ways a PostgreSQL runtime can hold nothing for this agent, and
        # boot treats them alike: the schema does not exist yet (a database
        # never booted against — ``AsyncDatabase.postgres()`` creates it), or
        # it exists and this tenant is not in it. Either way boot copies the
        # birth record out of the anchor (#2871) and *then* runs the integrity
        # audit, so the governance about to be audited is the anchor's.
        # Reporting "nothing to check" would let a stale anchor pass review and
        # safe-mode the agent moments later. Check what is about to be copied.
        pending = False
        on_postgres = isinstance(source, _SchemaAbsent) or (
            isinstance(source, _GovernanceSource) and not source.reads_the_anchor
        )
        # ...but only when the runtime really is empty. A row present without
        # this agent's ownership witness reads back identically to a missing
        # one and is the opposite situation: replication cannot claim it, so
        # the agent cannot boot and the anchor is not what will be audited.
        unowned = (
            isinstance(node, _NoAgentNode)
            and isinstance(source, _GovernanceSource)
            and source.ownership_settled
            and _row_physically_exists(source)
        )
        runtime_is_empty = (
            on_postgres
            and not unowned
            and (
                isinstance(node, (_NoAgentNode, _SchemaAbsent))
                # A boot-fabricated placeholder is *present* but is not a birth
                # record, and boot does not keep it: ``birth_record`` counts it as
                # an identity shortfall and replaces it from the anchor before the
                # audit. Reading it as populated made doctor judge a row nobody
                # will be governed by, warn only that it had no hash, and exit
                # Ready while the stale anchor about to replace it safe-modes the
                # agent.
                or _is_placeholder_node(node)
            )
        )
        if runtime_is_empty:
            anchor_source = _resolve_governance_source(
                db_path, {}, project_dir, ledger_by_dsn
            )
            if isinstance(anchor_source, _GovernanceSource):
                source, node, pending = (
                    anchor_source,
                    _read_agent_node(anchor_source),
                    True,
                )

        readings.append(
            _AgentGovernance(
                name=name,
                agent_dir=agent_dir,
                source=source,
                node=node,
                pending_replication=pending,
            )
        )
    return readings


def _check_constitution_drift(
    readings: list[_AgentGovernance], report: DoctorReport
) -> None:
    """Compare each agent's anchored constitution_hash against the on-disk file.

    For each local multi_agent agent: open ``kestrel_prime.db`` with stock
    ``sqlite3``, read the agent node's ``constitution_hash`` property
    (plain JSON in ``graph_nodes.properties``), and compare to the SHA256
    of the canonical ``KESTREL_CONSTITUTION.md`` shipped in the package.

    Why we don't need ``KESTREL_DATA_KEY`` here: the file content is
    Fernet-encrypted at the blob level in the ``files`` table, but the
    agent's anchor is just the *hash*, stored as plaintext JSON in the
    graph node properties. We compare hashes, not file content.

    Why we might still fail to read the DB:

      - ``KESTREL_DB_KEY`` was set at inception (whole-DB sqlcipher
        encryption). Stock sqlite3 can't open it. Warn + skip.
      - File is corrupt or partially written. Warn + skip with the
        underlying error so the user can debug.

    Per-agent overlay (``<agent_dir>/CONSTITUTION.md``) and the
    ``governed_by`` governance edge are NOT compared here — overlays ARE
    anchored since #1722, and the edge is integrity proof 2. Both are
    checked by ``_check_anchor_consistency`` (#2616).
    """
    if not readings:
        return

    canonical = _canonical_constitution_path()
    # Up-front readability guard: if the canonical governing source itself
    # cannot be read, no agent can be drift-checked against it, and the failure
    # is independent of any per-agent DB state. Surface it once and stop rather
    # than letting an empty/unreadable per-agent DB short-circuit the loop
    # before the canonical read is ever reached (#2463). The per-agent resolve
    # below still renders each agent's Amendment VIII contract for the actual
    # hash comparison; this only pre-checks that the file exists and is readable.
    try:
        canonical.read_bytes()
    except OSError as exc:
        report.warn.append(
            f"Constitution drift check skipped — cannot read canonical "
            f"{canonical}: {exc}"
        )
        return

    for reading in readings:
        name = reading.name
        source = reading.source
        if isinstance(source, _UnreadableDB):
            _report_unexamined(name, source.reason, source, report)
            continue

        node = reading.node
        if isinstance(node, _UnreadableDB):
            _report_unexamined(name, node.reason, node, report)
            continue
        if isinstance(node, _NoAgentNode):
            # The same verdict on either backend: the row is there, #2649 is
            # settled so boot will not backfill a witness, and replication
            # cannot claim an unowned row. Only the sentence differs, because
            # only the database does.
            if source.ownership_settled and _row_physically_exists(source):
                # The anchor is the file DID discovery just read a DID *out
                # of*, so the row is certainly there — and #2649's backfill is
                # already recorded complete, so boot will not re-run it and
                # repair this. (Table existence was the wrong proxy for that:
                # the tables are created by schema init, which says nothing
                # about whether the one-time backfill has run.)
                # The witness is missing or belongs to someone else, which
                # means the agent's own bound store cannot see its agent node
                # either: it fails at startup, and `add_node` will not
                # overwrite a foreign-owned row to repair it. Warning here let
                # doctor exit Ready, having also skipped the edge and overlay
                # checks, for a host that cannot boot.
                report.fail.append(
                    f"{name}: the agent row in {source.describe()} is not "
                    f"owned by {source.agent_did} — the agent's own storage "
                    f"cannot see it, so startup fails, and replication cannot "
                    f"claim an existing unowned row. This is an ownership "
                    f"ledger problem, not constitution drift; reanchoring will "
                    f"not clear it."
                )
                continue
            report.warn.append(
                f"{name}: constitution drift check skipped — no agent node "
                f"owned by {source.agent_did} in {source.describe()}"
            )
            continue
        _, _, properties = node

        stored_hash = _anchored_constitution_hash(properties)
        if isinstance(stored_hash, _NoHashProperty):
            report.warn.append(
                f"{name}: constitution drift check skipped — agent node missing "
                f"constitution_hash property (older agent? re-incept to anchor)"
            )
            continue

        # Recompute the EXPECTED hash the way the periodic integrity audit does
        # (#2463): resolve the packaged governing bytes through the shared
        # resolver, rendering this agent's anchored Amendment VIII emancipation
        # contract if it has one. Hashing raw package bytes here would false-flag
        # every emancipated agent as "drifted" and could not diagnose an active/
        # custom agent consistently with the runtime verifier.
        contract_json = _anchored_emancipation_contract(properties)
        try:
            from kestrel_sovereign.constitution.emancipation import (
                EmancipationConfigError,
                contract_from_json,
            )
            from kestrel_sovereign.constitution.resolver import (
                resolve_governing_constitution_bytes,
            )

            contract = contract_from_json(contract_json)
            on_disk_hash = hashlib.sha256(
                resolve_governing_constitution_bytes(
                    contract, constitution_path=str(canonical)
                )
            ).hexdigest()
        except FileNotFoundError as exc:
            report.warn.append(
                f"{name}: Constitution drift check skipped — cannot read "
                f"canonical {canonical}: {exc}"
            )
            continue
        except EmancipationConfigError as exc:
            report.fail.append(
                f"{name}: anchored emancipation contract is corrupted ({exc}); "
                f"the agent will fail its integrity audit. Re-anchor it."
            )
            continue
        except (OSError, ValueError) as exc:
            report.warn.append(
                f"{name}: constitution drift check skipped — cannot resolve "
                f"governing constitution: {exc}"
            )
            continue

        if reading.pending_replication:
            report.warn.append(
                f"{name}: PostgreSQL holds no record for this agent yet — boot "
                f"will copy the birth record from {source.anchor_path} and "
                f"audit that. The verdict below is about those pending bytes."
            )

        if stored_hash == on_disk_hash:
            report.ok.append(
                f"{name}: constitution anchored to current file ({stored_hash[:12]}…)"
            )
        else:
            report.fail.append(
                f"{name}: constitution drift — stored {stored_hash[:12]}… "
                f"does not match {canonical} ({on_disk_hash[:12]}…). "
                f"Run `kestrel constitution reanchor --agent-name {name} --force` "
                f"to update ({_rollback_advice(source)})."
            )


# Mirror kestrel_sovereign.setup.overlay_anchor — importing that module here
# would drag in the full AsyncStorage stack; doctor deliberately reads agent
# DBs with stock sqlite3 only.
_OVERLAY_FILENAME = "CONSTITUTION.md"
_OVERLAY_HASH_PROPERTY = "constitution_overlay_hash"


def _check_anchor_consistency(
    readings: list[_AgentGovernance], report: DoctorReport
) -> None:
    """Pre-upgrade anchor-drift checks beyond base-hash drift (#2616).

    The fail-closed constitution integrity audit (#2463 proofs, #2595
    durable safe mode, #2600 genesis-audit lifecycle) safe-modes an agent
    at boot on anchor inconsistencies the hash comparison in
    ``_check_constitution_drift`` cannot see:

      - Proof 2: the ``agent --governed_by--> constitution`` graph edge
        must target the anchored ``constitution_hash``. A historical
        pre-atomic reanchor could update the property + blob but leave
        the edge pointing at an ancient anchor (the 2026-07-18 incident:
        three prod agents fail-closed at first boot after upgrade).
      - Overlay: a per-agent ``CONSTITUTION.md`` overlay must be anchored
        (``constitution_overlay_hash``) and match the file on disk; an
        anchored overlay must still exist (#1722).

    Each drift fail names the trusted-channel remediation command so
    operators can reanchor BEFORE upgrading into the fail-closed
    enforcement. Unreadable DBs, missing agent nodes, and a missing base
    anchor are already surfaced by ``_check_constitution_drift``; this
    check stays silent for those instead of duplicating the warning.
    """
    for reading in readings:
        source = reading.source
        if isinstance(source, _UnreadableDB):
            # Already warned by _check_constitution_drift.
            continue

        node = reading.node
        if isinstance(node, (_UnreadableDB, _NoAgentNode)):
            # Already warned by _check_constitution_drift.
            continue
        node_id, _, properties = node
        if properties is None:
            # Unparseable properties — _check_constitution_drift already
            # warns (missing constitution_hash); nothing verifiable here.
            continue

        _check_governance_edge(reading.name, source, node_id, properties, report)
        _check_overlay_anchor(reading.name, reading.agent_dir, properties, report)


def _check_governance_edge(
    name: str,
    source: _GovernanceSource,
    node_id: str,
    properties: dict,
    report: DoctorReport,
) -> None:
    """Verify the ``governed_by`` edge targets the anchored constitution_hash."""
    stored_hash = properties.get("constitution_hash")
    if not stored_hash or not isinstance(stored_hash, str):
        # No anchor to agree with; _check_constitution_drift already warns.
        return

    targets = _read_governed_by_targets(source, node_id)
    if isinstance(targets, _UnreadableDB):
        if _edge_probe_left_governance_unexamined(targets):
            _report_unexamined(
                name,
                targets.reason,
                targets,
                report,
                database_was_reached=True,
            )
            return
        # FAIL, not warn: we could read the agent node moments ago, so the
        # DB is not sqlcipher-encrypted — the edges specifically cannot be
        # read (missing graph_edges table / corruption). The runtime fails
        # closed either way: an edge-read error is an integrity failure,
        # and a missing table is auto-created empty at startup, leaving
        # proof 2 with no edge. Reporting "Ready" here would upgrade the
        # operator straight into safe mode.
        report.fail.append(
            f"{name}: cannot verify the governed_by governance edge — "
            f"{targets.reason}. The fail-closed integrity audit (proof 2) "
            f"requires an edge at the anchored constitution and will "
            f"safe-mode this agent at next boot. Run `kestrel constitution "
            f"reanchor --agent-name {name} --force` with a signed artifact "
            f"to repair ({_rollback_advice(source)})."
        )
        return

    # An edge the runtime cannot see is not the same finding as no edge. The
    # writer deletes an *ownerless* correct edge and recreates it, so that one
    # a forced reanchor really does repair; an edge witnessed by another tenant
    # is refused by ``add_edge`` outright ("Cannot overwrite a graph edge owned
    # by another agent"). Promising a repair for the second would send an
    # operator to a command that cannot clear the finding it was given for.
    if stored_hash not in targets:
        physical = _read_governed_by_targets(source, node_id, any_owner=True)
        if isinstance(physical, _UnreadableDB) and (
            _edge_probe_left_governance_unexamined(physical)
        ):
            _report_unexamined(
                name,
                physical.reason,
                physical,
                report,
                database_was_reached=True,
            )
            return
        if not isinstance(physical, _UnreadableDB) and stored_hash in physical:
            report.fail.append(
                f"{name}: the governed_by edge to {stored_hash[:12]}… exists "
                f"in {source.describe()} but this agent does not own it, so "
                f"integrity proof 2 cannot see it and the agent safe-modes at "
                f"boot. If nobody owns it, `kestrel constitution reanchor "
                f"--agent-name {name} --force` re-creates it with its ledger "
                f"row; if another agent owns it, that is ledger damage no "
                f"reanchor can clear."
            )
            return

    if stored_hash in targets:
        report.ok.append(
            f"{name}: governed_by edge targets the anchored constitution "
            f"({stored_hash[:12]}…)"
        )
        stale = sorted({t for t in targets if t != stored_hash})
        if stale:
            # Proof 2 tolerates extra edges (it only requires one at the
            # anchor), so this won't safe-mode — but the agent nominally
            # has two governing constitutions. Reanchor cleans these up.
            report.warn.append(
                f"{name}: stale extra governed_by edge(s) alongside the "
                f"anchored constitution "
                f"({', '.join(t[:12] + '…' for t in stale)}). Boot will "
                f"succeed, but `kestrel constitution reanchor --agent-name "
                f"{name} --force` will remove them."
            )
    elif targets:
        report.fail.append(
            f"{name}: anchor drift — governed_by edge targets "
            f"{targets[0][:12]}… but constitution_hash is {stored_hash[:12]}…. "
            f"The fail-closed integrity audit (proof 2) will safe-mode this "
            f"agent at next boot. Run `kestrel constitution reanchor "
            f"--agent-name {name} --force` with a signed artifact to repair "
            f"({_rollback_advice(source)})."
        )
    else:
        report.fail.append(
            f"{name}: anchor drift — no governed_by edge to the anchored "
            f"constitution ({stored_hash[:12]}…). The fail-closed integrity "
            f"audit (proof 2) will safe-mode this agent at next boot. Run "
            f"`kestrel constitution reanchor --agent-name {name} --force` "
            f"with a signed artifact to repair ({_rollback_advice(source)})."
        )


def _edge_probe_left_governance_unexamined(result: _UnreadableDB) -> bool:
    """Whether an edge read failed without establishing integrity damage.

    ``runtime_database`` means the equivalent connection opened and the edge
    query itself failed, which is evidence the runtime cannot perform proof 2.
    Every other classified PostgreSQL failure stopped before that conclusion:
    it may block readiness, but it cannot justify a force-reanchor repair.
    Unclassified SQLite failures retain the established integrity finding.
    """
    return (
        result.postgres_failure is not None
        and result.postgres_failure != "runtime_database"
    )


def _describe_overlay_anchor(anchor: object) -> str:
    """Format an overlay anchor for a report line without assuming it's a hash.

    The runtime treats ANY truthy ``constitution_overlay_hash`` value as an
    anchor (truthiness, not isinstance), so a malformed non-string value must
    be reported, not slice-crashed on.
    """
    if isinstance(anchor, str):
        return f"{anchor[:12]}…"
    return f"a malformed non-string value ({anchor!r:.60})"


def _check_overlay_anchor(
    name: str, agent_dir: Path, properties: dict, report: DoctorReport
) -> None:
    """Verify any per-agent CONSTITUTION.md overlay is anchored + unmodified.

    Mirrors the runtime ``verify_constitution_overlay`` decision matrix
    (#1722) EXACTLY, including its edge states:

      - The anchor is "present" by truthiness, not type: a malformed
        non-string ``constitution_overlay_hash`` counts as anchored at
        runtime and can never equal the overlay's sha — that's a FAIL
        (drift if the overlay exists, tampering if it doesn't), never
        silently "unanchored".
      - An overlay file that exists but cannot be read is treated by the
        runtime as ABSENT (its sha never computes). With an anchor that is
        the "anchored overlay missing" tampering failure → FAIL here too.
        Without an anchor the runtime passes ("no overlay"), so doctor
        only warns that it could not verify the file.

    Overlay absent + no anchor is the normal case and stays silent.
    """
    overlay_path = agent_dir / _OVERLAY_FILENAME
    # Runtime truthiness (#1722): any truthy value is an anchor.
    anchor = properties.get(_OVERLAY_HASH_PROPERTY) or None

    if overlay_path.exists():
        try:
            overlay_hash = hashlib.sha256(overlay_path.read_bytes()).hexdigest()
        except OSError as exc:
            if anchor is not None:
                report.fail.append(
                    f"{name}: constitution overlay {overlay_path} exists but "
                    f"cannot be read ({exc}). The fail-closed integrity audit "
                    f"treats an unreadable overlay as absent, and an absent "
                    f"overlay with an anchor ("
                    f"{_describe_overlay_anchor(anchor)}) as tampering — this "
                    f"agent will safe-mode at next boot. Make the overlay "
                    f"readable, or restore it and re-run `kestrel constitution "
                    f"anchor-overlay --agent-name {name}`."
                )
            else:
                report.warn.append(
                    f"{name}: overlay anchor check incomplete — cannot read "
                    f"{overlay_path}: {exc}. The runtime treats an unreadable "
                    f"overlay as absent (no anchor is set, so boot will not "
                    f"safe-mode), but doctor could not verify the file."
                )
            return
        if anchor is None:
            report.fail.append(
                f"{name}: constitution overlay {overlay_path} is present but "
                f"NOT anchored — the fail-closed integrity audit will "
                f"safe-mode this agent at next boot. If the overlay is "
                f"legitimate, stop the agent and run `kestrel constitution "
                f"anchor-overlay --agent-name {name}`."
            )
        elif anchor != overlay_hash:
            report.fail.append(
                f"{name}: constitution overlay drift — anchored "
                f"{_describe_overlay_anchor(anchor)} does not match "
                f"{overlay_path} ({overlay_hash[:12]}…). If the "
                f"change is legitimate, stop the agent and run `kestrel "
                f"constitution anchor-overlay --agent-name {name}`."
            )
        else:
            report.ok.append(
                f"{name}: constitution overlay anchored ({overlay_hash[:12]}…)"
            )
    elif anchor is not None:
        report.fail.append(
            f"{name}: an anchored constitution overlay is missing from disk "
            f"(anchored {_describe_overlay_anchor(anchor)}, expected at "
            f"{overlay_path}) — the "
            f"fail-closed integrity audit treats this as tampering and will "
            f"safe-mode this agent at next boot. Restore the overlay file, or "
            f"re-run `kestrel constitution anchor-overlay --agent-name {name}` "
            f"after restoring the intended content."
        )


def _report_unexamined(
    name: str,
    reason: str,
    source: object,
    report: DoctorReport,
    *,
    database_was_reached: bool = False,
) -> None:
    """Record that this agent's governance could not be read.

    PostgreSQL failures retain explicit provenance from the isolated asyncpg
    worker. All leave governance unverified and therefore fail readiness, but
    the report must not turn diagnostic blindness into integrity corruption by
    guessing from words in ``reason``.

    An edge probe runs only after the agent-node read established the
    runtime-equivalent connection. ``database_was_reached`` preserves that
    evidence while still failing readiness for the incomplete governance read.

    On the local anchor it usually can. ``KESTREL_DB_KEY`` at inception gives a
    whole-DB sqlcipher file that stock ``sqlite3`` cannot open and the agent
    reads perfectly well — a supported configuration, in which doctor alone is
    blind. Failing there would mark every sqlcipher host permanently not-ready
    for a problem that does not exist. Corruption presents identically to
    encryption through stock ``sqlite3``, so the two cannot be told apart here;
    the warning stays a warning and names what it could not read.

    Either way this is deliberately not phrased as *drift*: the remedy is to
    make the database readable, not to reanchor. Prescribing a reanchor on the
    strength of a read that never happened is how an operator rewrites
    governance to fix a network problem.
    """
    postgres_failure = (
        source.postgres_failure if isinstance(source, _UnreadableDB) else None
    )
    if (
        isinstance(source, _UnreadableDB)
        and postgres_failure is not None
        and database_was_reached
    ):
        partial_advice = (
            " Inspect the preserved partial diagnostic, then fix its cause"
            if source.postgres_partial_diagnostic
            else " Fix the reported failure"
        )
        report.fail.append(
            f"{name}: governance NOT verified — {reason}. A runtime-equivalent "
            f"database connection succeeded while reading the agent node, but "
            f"the governance edge read did not complete.{partial_advice} and "
            f"re-run before treating this host as ready."
        )
        return
    if postgres_failure == "connection":
        report.fail.append(
            f"{name}: governance NOT verified — {reason}. Doctor cannot say "
            f"whether this agent is correctly anchored. The spawned runtime's "
            f"own asyncpg connection failed with its effective settings, so "
            f"runtime database access with those settings will fail too; "
            f"fix the access problem and re-run before treating this host as "
            f"ready."
        )
    elif postgres_failure == "runtime_database":
        report.fail.append(
            f"{name}: governance NOT verified — {reason}. The equivalent "
            f"diagnostic connection opened but could not read governance, so "
            f"the runtime cannot use this database for governance either; fix "
            f"the database access problem and re-run doctor."
        )
    elif postgres_failure == "doctor_configuration":
        report.fail.append(
            f"{name}: governance NOT verified — {reason}. Runtime database "
            f"reachability was not established by this diagnostic; fix the "
            f"doctor-only setting and re-run before treating this host as "
            f"ready."
        )
    elif postgres_failure == "diagnostic_timeout":
        partial_advice = (
            "inspect the preserved partial diagnostic, then fix "
            if source.postgres_partial_diagnostic
            else "fix "
        )
        report.fail.append(
            f"{name}: governance NOT verified — {reason}. Runtime database "
            f"reachability was not established before the finite diagnostic "
            f"deadline; {partial_advice}connectivity or adjust the doctor "
            f"timeout and re-run before "
            f"treating this host as ready."
        )
    elif postgres_failure == "diagnostic_tooling":
        report.fail.append(
            f"{name}: governance NOT verified — {reason}. Runtime database "
            f"reachability was not established by this diagnostic; repair the "
            f"local diagnostic tooling and re-run before "
            f"treating this host as ready."
        )
    else:
        report.warn.append(f"{name}: constitution drift check skipped — {reason}")


def _rollback_advice(source: _GovernanceSource) -> str:
    """What an operator actually gets to undo the prescribed repair.

    "DB is backed up first" is true of a SQLite anchor, which the reanchor
    copies aside before it writes. It is false of PostgreSQL: that path
    deliberately takes no backup — there is no file to copy — and the CLI tells
    operators to snapshot the database themselves. Now that doctor prescribes
    repairs against PostgreSQL, repeating the SQLite promise would send someone
    to mutate live governance believing a rollback copy exists.
    """
    if source.reads_the_anchor:
        return "DB is backed up first"
    return (
        "governance for this agent lives in PostgreSQL — there is no local "
        "file to copy, so snapshot that database first if you want to be able "
        "to undo it"
    )


def _canonical_constitution_path() -> Path:
    """Return the package's canonical constitution path (config.CONSTITUTION_PATH).

    Imported lazily — pulling kestrel_sovereign.config at module top would
    drag in a chunk of the package and slow doctor's cold-start path.
    """
    from kestrel_sovereign.config import CONSTITUTION_PATH

    return Path(CONSTITUTION_PATH)


#: The queries doctor issues, per backend.
#:
#: **Discovery** asks the anchor a different question from the rest: "whose
#: anchor is this?" Identity is born in the local ``kestrel_prime.db`` on every
#: backend (#2871, #2894), so this one is deliberately unscoped and takes the
#: single agent row. Everything below then scopes by the DID it returns.
#:
#: ``LIMIT 2``, not ``LIMIT 1``: the canonical reader
#: (``identity.local_anchor.read_anchor_agent_did``) *refuses* an anchor
#: holding more than one agent root rather than picking one by incidental row
#: order, and boot goes through it. Taking the first row would let doctor scope
#: its checks to an arbitrary tenant, find that one healthy, and report Ready
#: for an agent the runtime will not start at all. Two rows is all it takes to
#: know there are too many.
_DISCOVER_AGENT_SQLITE = (
    "SELECT node_id FROM graph_nodes WHERE node_type='agent' LIMIT 2"
)

#: **Governance** reads must see exactly what the booting agent sees, and the
#: agent's ``AsyncStorage`` is bound to its DID on *both* backends
#: (``kestrel_agent`` passes ``agent_id=self.did`` for SQLite and PostgreSQL
#: alike). A bound ``AsyncGraphStore`` does not filter on ``node_id``: its
#: ``_node_scope`` / ``_edge_scope`` require a matching ownership witness in
#: ``graph_node_owners`` / ``graph_edge_owners``. So a raw row without its
#: witness is invisible to the runtime — the integrity audit reads
#: ``storage.get_node`` and ``storage.get_edges_from``, both scoped — and
#: doctor matching on ``node_id`` alone would pronounce an agent healthy that
#: safe-modes at its next boot. False reassurance from a governance tool is
#: worse than the false alarm this issue started from.
_AGENT_NODE_SQLITE = (
    "SELECT node_id, label, properties FROM graph_nodes "
    "WHERE node_id = ? AND node_type = 'agent' AND EXISTS ("
    "  SELECT 1 FROM graph_node_owners AS owner "
    "  WHERE owner.node_id = graph_nodes.node_id AND owner.agent_id = ?)"
)
_AGENT_NODE_PG = (
    "SELECT node_id, label, properties FROM graph_nodes "
    "WHERE node_id = $1 AND node_type = 'agent' AND EXISTS ("
    "  SELECT 1 FROM graph_node_owners AS owner "
    "  WHERE owner.node_id = graph_nodes.node_id AND owner.agent_id = $2)"
)
#: Physical ``governed_by`` targets regardless of who witnesses them. Compared
#: against the scoped read to tell "no such edge" from "an edge this agent
#: cannot use" — different findings, with different remedies.
_GOVERNED_BY_ANY_OWNER_SQLITE = (
    "SELECT target_id FROM graph_edges WHERE source_id = ? AND label = 'governed_by'"
)
_GOVERNED_BY_ANY_OWNER_PG = (
    "SELECT target_id FROM graph_edges WHERE source_id = $1 AND label = 'governed_by'"
)
_GOVERNED_BY_SQLITE = (
    "SELECT target_id FROM graph_edges "
    "WHERE source_id = ? AND label = 'governed_by' AND EXISTS ("
    "  SELECT 1 FROM graph_edge_owners AS owner "
    "  WHERE owner.source_id = graph_edges.source_id "
    "  AND owner.target_id = graph_edges.target_id "
    "  AND owner.label = graph_edges.label AND owner.agent_id = ?)"
)
_GOVERNED_BY_PG = (
    "SELECT target_id FROM graph_edges "
    "WHERE source_id = $1 AND label = 'governed_by' AND EXISTS ("
    "  SELECT 1 FROM graph_edge_owners AS owner "
    "  WHERE owner.source_id = graph_edges.source_id "
    "  AND owner.target_id = graph_edges.target_id "
    "  AND owner.label = graph_edges.label AND owner.agent_id = $2)"
)

#: The same reads against a database predating the ownership migration
#: (#2649), where those ledgers do not exist yet. The second placeholder is
#: still bound — and still the DID — so the two forms stay parameter-compatible
#: and a caller cannot pick the wrong argument list with the wrong query.
_AGENT_NODE_SQLITE_LEGACY = (
    "SELECT node_id, label, properties FROM graph_nodes "
    "WHERE node_id = ? AND node_type = 'agent' AND ? IS NOT NULL"
)
_AGENT_NODE_PG_LEGACY = (
    "SELECT node_id, label, properties FROM graph_nodes "
    "WHERE node_id = $1 AND node_type = 'agent' AND $2::text IS NOT NULL"
)
_GOVERNED_BY_SQLITE_LEGACY = (
    "SELECT target_id FROM graph_edges "
    "WHERE source_id = ? AND label = 'governed_by' AND ? IS NOT NULL"
)
_GOVERNED_BY_PG_LEGACY = (
    "SELECT target_id FROM graph_edges "
    "WHERE source_id = $1 AND label = 'governed_by' AND $2::text IS NOT NULL"
)


@dataclass(frozen=True)
class _UnreadableDB:
    """Sentinel: doctor could not read the agent's governance database.

    ``postgres_failure`` records what was actually established. It is absent
    for SQLite and must never be reconstructed from prose in ``reason``.
    """

    reason: str
    postgres_failure: str | None = None
    #: True only when a timed-out isolated worker yielded redacted output.
    postgres_partial_diagnostic: bool = False


@dataclass(frozen=True)
class _NoAgentNode:
    """Sentinel: DB opened but no node_type='agent' row."""


@dataclass(frozen=True)
class _NoHashProperty:
    """Sentinel: agent node found but properties has no constitution_hash."""


#: The one-time #2649 ownership backfill's marker, as ``AsyncDatabase`` records
#: it in ``schema_backfills``. Boot runs that backfill exactly once; before it
#: has, a row with no ownership witness is repaired at the next start.
_OWNERSHIP_BACKFILL = "ownership_2649"


@dataclass(frozen=True)
class _LedgerState:
    """What the ownership ledgers say about this database.

    ``present`` decides which SQL doctor can run at all; ``settled`` decides
    whether a *missing* witness is permanent or merely pending. They are not
    the same fact — schema init creates the tables, and the backfill that fills
    them is recorded separately — and using the first as a proxy for the second
    turned a database awaiting its migration into a readiness failure.
    """

    present: bool
    settled: bool


@dataclass(frozen=True)
class _SchemaAbsent:
    """Sentinel: a PostgreSQL database with no Kestrel graph schema yet.

    Reachable, empty, and about to be initialised — ``AsyncDatabase.postgres()``
    creates the tables and replicates the anchor on first boot. Distinct from
    ``_UnreadableDB``, which means doctor could not look, and from
    ``_NoAgentNode``, which means the schema exists and this tenant is not in
    it. All three end at the same place — read the anchor, because that is what
    boot will copy — but only this one is a *healthy* state.
    """


def _anchored_constitution_hash(properties: dict | None):
    """The agent node's ``constitution_hash``, or ``_NoHashProperty()``.

    A pure reading of properties already fetched. It used to open its own
    connection, as did the emancipation-contract reader beside it, so doctor
    asked the same database for the same row three times per agent. On
    PostgreSQL those were three round trips that could disagree with each
    other — see :func:`_anchored_emancipation_contract` for what a disagreement
    cost.
    """
    if not isinstance(properties, dict):
        return _NoHashProperty()

    stored_hash = properties.get("constitution_hash")
    if not stored_hash or not isinstance(stored_hash, str):
        return _NoHashProperty()

    return stored_hash


def _anchored_emancipation_contract(properties: dict | None):
    """The agent node's ``emancipation_contract`` property, or ``None``.

    Doctor needs this to render Amendment VIII the way the runtime integrity
    audit does before hashing the governing bytes (#2463). ``None`` is the
    right answer for an agent that has no contract: a dormant contract yields
    the canonical bytes.

    ``None`` is emphatically *not* the right answer for a read that failed,
    which is why this no longer performs one. As a second query over a second
    connection it mapped every exception to ``None``, so one transient
    PostgreSQL blip on an emancipated agent made doctor render the *dormant*
    constitution, compare against the wrong hash, and report drift with advice
    to reanchor — manufacturing a governance failure out of a network hiccup.
    Reading the property from the row already fetched removes the failure mode
    rather than handling it.
    """
    if not isinstance(properties, dict):
        return None

    return properties.get("emancipation_contract")


def _read_agent_node(source: _GovernanceSource):
    """Read the agent node's id + properties as the *runtime* sees it.

    Scoped to the bound tenant on both backends — see the query constants for
    why an ownership witness, not a ``node_id`` match, is the runtime's scope.

    Returns:
        - ``(node_id, label, properties_dict)`` on success.
        - ``(node_id, label, None)`` when the properties column is missing,
          unparseable, or not a JSON object.
        - ``_UnreadableDB(reason=...)`` / ``_NoAgentNode()`` sentinels.
    """
    try:
        rows = _fetch_rows(
            source,
            _AGENT_NODE_SQLITE
            if source.ownership_settled
            else _AGENT_NODE_SQLITE_LEGACY,
            _AGENT_NODE_PG if source.ownership_settled else _AGENT_NODE_PG_LEGACY,
            sqlite_params=(source.agent_did, source.agent_did),
            postgres_params=(source.agent_did, source.agent_did),
        )
    except sqlite3.DatabaseError as exc:
        return _UnreadableDB(reason=f"DB unreadable ({exc})")
    except sqlite3.Error as exc:
        return _UnreadableDB(reason=f"sqlite error ({exc})")
    except Exception as exc:  # noqa: BLE001 — asyncpg raises its own tree
        return _source_unreadable(
            source,
            exc,
            reason=f"cannot read {source.describe()} ({_safe(exc, source)})",
        )

    if not rows:
        return _NoAgentNode()

    node_id, label, properties_raw = rows[0]
    if properties_raw is None:
        return node_id, label, None

    try:
        properties = (
            json.loads(properties_raw)
            if isinstance(properties_raw, (str, bytes, bytearray))
            else properties_raw
        )
    except (TypeError, ValueError):
        return node_id, label, None

    if not isinstance(properties, dict):
        return node_id, label, None

    return node_id, label, properties


def _read_governed_by_targets(
    source: _GovernanceSource, source_id: str, *, any_owner: bool = False
):
    """Return the targets of the agent's ``governed_by`` edges.

    Returns a tuple of target hashes (possibly empty), or
    ``_UnreadableDB(reason=...)`` when the edges cannot be read — a
    sqlcipher DB, corruption, or a legacy/synthetic DB with no
    ``graph_edges`` table.
    """
    try:
        rows = _fetch_rows(
            source,
            _GOVERNED_BY_ANY_OWNER_SQLITE
            if any_owner
            else _GOVERNED_BY_SQLITE
            if source.ownership_settled
            else _GOVERNED_BY_SQLITE_LEGACY,
            _GOVERNED_BY_ANY_OWNER_PG
            if any_owner
            else _GOVERNED_BY_PG
            if source.ownership_settled
            else _GOVERNED_BY_PG_LEGACY,
            # The any-owner form binds only ``source_id``; the scoped and
            # legacy forms bind the DID a second time. Passing the scoped
            # argument list to a one-placeholder query raises, which this
            # reader turns into ``_UnreadableDB`` — and the caller treats that
            # as "cannot tell", silently skipping the very branch it was added
            # for. Match the arguments to the query that was chosen.
            sqlite_params=(
                (source_id,) if any_owner else (source_id, source.agent_did)
            ),
            postgres_params=(
                (source_id,) if any_owner else (source_id, source.agent_did)
            ),
        )
    except sqlite3.DatabaseError as exc:
        return _UnreadableDB(reason=f"cannot read graph_edges ({exc})")
    except sqlite3.Error as exc:
        return _UnreadableDB(reason=f"sqlite error reading graph_edges ({exc})")
    except Exception as exc:  # noqa: BLE001 — asyncpg raises its own tree
        return _source_unreadable(
            source,
            exc,
            reason=(
                f"cannot read graph_edges in {source.describe()} ({_safe(exc, source)})"
            ),
        )

    return tuple(row[0] for row in rows if row[0])
