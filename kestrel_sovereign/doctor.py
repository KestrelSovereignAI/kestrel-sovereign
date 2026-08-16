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

import getpass
import hashlib
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

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


def _check_llm(
    config: dict, env: dict, toml_path: Path, report: DoctorReport
) -> None:
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

    Staying out of the stack costs nothing here: every property doctor reads is
    plaintext JSON in ``graph_nodes.properties`` and every edge is a plain row,
    so no ``KESTREL_DATA_KEY`` is involved on either backend, and
    ``psycopg2`` — already a hard dependency — is a *synchronous* driver, so
    there is no event loop to own.

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


def _has_ownership_ledger(source: "_GovernanceSource") -> bool:
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
        return _LedgerState(
            present=row is not None, settled=settled is not None
        )

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
    except Exception as exc:  # noqa: BLE001 — psycopg2 raises its own tree
        return _UnreadableDB(
            reason=f"cannot read {source.describe()} ({_safe(exc, source)})"
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
            "SELECT 1 FROM schema_backfills WHERE name = %s",
            postgres_params=(_OWNERSHIP_BACKFILL,),
        )
    except Exception as exc:  # noqa: BLE001 — psycopg2 raises its own tree
        return _UnreadableDB(
            reason=f"cannot read {source.describe()} ({_safe(exc, source)})"
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
            dsn = _doctor_postgres_dsn(runtime_dsn, env, project_dir)
        except ValueError as exc:
            # Bind the raw URI to the same redactor used for driver failures.
            # The translation error is intentionally generic, but this also
            # protects future parser messages from echoing URI credentials.
            unsafe_source = _GovernanceSource(
                anchor_path=anchor_path,
                agent_did=agent_did,
                dsn=runtime_dsn,
            )
            return _UnreadableDB(
                reason=f"cannot read PostgreSQL ({_safe(exc, unsafe_source)})"
            )
        source = _GovernanceSource(
            anchor_path=anchor_path, agent_did=agent_did, dsn=dsn
        )
    # Keyed on the DSN, or on the anchor path for a SQLite host where each
    # agent genuinely has its own file and its own answer.
    cache_key = source.dsn or str(source.anchor_path)
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


#: Seconds doctor will wait for a PostgreSQL connection before calling the
#: database unreadable. Doctor is the tool an operator reaches for *when the
#: database is unavailable*, and a black-holed or firewalled endpoint does not
#: refuse the connection — it drops the packets, and libpq's default is to wait
#: out the OS TCP timeout, minutes of an apparently hung diagnostic. Failing
#: fast turns that into the ``_UnreadableDB`` finding callers already report.
_CONNECT_TIMEOUT_SECONDS = 5


# Query parameters consumed by asyncpg 0.30's connection parser. Every other
# parameter is put in ``server_settings`` and sent in PostgreSQL's startup
# packet. This is intentionally asyncpg's vocabulary, not libpq's: libpq knows
# names such as ``connect_timeout`` and ``keepalives`` that the runtime sends
# to PostgreSQL as settings (where PostgreSQL rejects them). Letting libpq
# consume those names would make doctor report Ready for an agent that cannot
# connect.
_ASYNCPG_CONNECTION_QUERY_OPTIONS = frozenset(
    {
        "port",
        "host",
        "dbname",
        "database",
        "user",
        "password",
        "passfile",
        "sslmode",
        "sslcert",
        "sslkey",
        "sslrootcert",
        "sslnegotiation",
        "sslcrl",
        "sslpassword",
        "ssl_min_protocol_version",
        "ssl_max_protocol_version",
        "target_session_attrs",
        "krbsrvname",
        "gsslib",
    }
)


# Scalar environment variables asyncpg 0.30 reads, and the equivalent DSN
# parameter libpq accepts. PGHOST and PGPORT are resolved together below because
# asyncpg's host-list grammar assigns a possibly different port to every host.
# Deliberately absent are libpq-only variables such as PGCONNECT_TIMEOUT:
# inheriting one would give doctor connection semantics the spawned asyncpg
# process does not have. The diagnostic timeout below also states
# connect_timeout explicitly, preventing libpq from consulting that
# process-global default. Libpq-only variables are neutralised separately below
# so they cannot leak back in from doctor's own process environment.
_ASYNCPG_ENV_DSN_OPTIONS = (
    ("PGUSER", "user"),
    ("PGPASSWORD", "password"),
    ("PGDATABASE", "dbname"),
    ("PGPASSFILE", "passfile"),
    ("PGSSLMODE", "sslmode"),
    ("PGSSLNEGOTIATION", "sslnegotiation"),
    ("PGSSLROOTCERT", "sslrootcert"),
    ("PGSSLCRL", "sslcrl"),
    ("PGSSLKEY", "sslkey"),
    ("PGSSLCERT", "sslcert"),
    ("PGSSLMINPROTOCOLVERSION", "ssl_min_protocol_version"),
    ("PGSSLMAXPROTOCOLVERSION", "ssl_max_protocol_version"),
    ("PGTARGETSESSIONATTRS", "target_session_attrs"),
    ("PGKRBSRVNAME", "krbsrvname"),
    ("PGGSSLIB", "gsslib"),
)


# Libpq reads these variables but asyncpg does not. When one is present in the
# effective spawned-agent environment, explicitly state asyncpg's corresponding
# behaviour in the translated DSN so libpq cannot inherit a different value
# from doctor's process environment. Each row is capability-probed against the
# psycopg2-linked libpq before use: a newer process environment can contain
# variables that an older linked libpq neither reads nor accepts as parameters.
_LIBPQ_ONLY_ENV_DSN_DEFAULTS = (
    ("PGHOSTADDR", "hostaddr", ""),
    ("PGGSSENCMODE", "gssencmode", "disable"),
    ("PGCHANNELBINDING", "channel_binding", "disable"),
    ("PGCLIENTENCODING", "client_encoding", ""),
    ("PGAPPNAME", "application_name", ""),
    ("PGSSLCOMPRESSION", "sslcompression", "0"),
    ("PGSSLCERTMODE", "sslcertmode", "allow"),
    ("PGSSLCRLDIR", "sslcrldir", ""),
    ("PGSSLSNI", "sslsni", "1"),
    ("PGREQUIREPEER", "requirepeer", ""),
    (
        "PGREQUIREAUTH",
        "require_auth",
        "none,password,md5,scram-sha-256,gss,sspi",
    ),
    ("PGGSSDELEGATION", "gssdelegation", "0"),
    ("PGLOADBALANCEHOSTS", "load_balance_hosts", "disable"),
)


def _libpq_accepts_dsn_option(name: str, value: str) -> bool:
    """Whether the psycopg2-linked libpq can express one DSN option.

    The project can run against a system libpq older than the environment
    variable or runtime option being translated. Callers distinguish a safe
    omission (a libpq-only variable that this older libpq also cannot read)
    from an asyncpg runtime constraint, which must fail closed. An unavailable
    or incomplete psycopg2 is likewise a negative capability result.
    """
    try:
        from psycopg2 import extensions as _pg_ext

        _pg_ext.make_dsn(**{name: value})
    except Exception:  # noqa: BLE001 — an unusable libpq cannot express it
        return False
    return True


def _require_libpq_dsn_option(name: str, value: str) -> None:
    """Fail when doctor cannot preserve one effective runtime constraint."""
    if not _libpq_accepts_dsn_option(name, value):
        # Values can be credentials (``password`` / ``sslpassword``), so the
        # diagnostic deliberately identifies only the unsupported field.
        raise ValueError(
            "installed PostgreSQL diagnostic driver cannot represent "
            f"runtime connection option {name!r}"
        )


def _authority_fields(netloc: str) -> tuple[str, str, str]:
    """Return asyncpg's raw ``(user, password, hostspec)`` URI fields."""
    if "@" in netloc:
        auth, _, hostspec = netloc.partition("@")
        user, _, password = auth.partition(":")
        return user, password, hostspec
    return "", "", netloc


def _validate_asyncpg_ports(
    hosts: list[str], ports: int | list[int]
) -> list[int]:
    """Apply asyncpg 0.30's one-port-per-host validation."""
    if isinstance(ports, list):
        if len(ports) != len(hosts):
            raise ValueError(
                f"could not match {len(ports)} port numbers to "
                f"{len(hosts)} hosts"
            )
        return ports
    return [ports for _ in hosts]


def _asyncpg_default_ports(hostspecs: list[str], env: dict) -> list[int]:
    """Return the PGPORT/default ports asyncpg applies to a host list."""
    portspec = env.get("PGPORT")
    if portspec:
        ports = (
            [int(port) for port in portspec.split(",")]
            if "," in portspec
            else int(portspec)
        )
    else:
        ports = 5432
    return _validate_asyncpg_ports(hostspecs, ports)


def _parse_asyncpg_hostlist(
    hostlist: str,
    ports: list[int] | None,
    env: dict,
    *,
    unquote_hosts: bool = False,
) -> tuple[list[str], list[int]]:
    """Parse one host list with asyncpg 0.30's host/port rules."""
    hostspecs = hostlist.split(",")
    defaults = (
        _asyncpg_default_ports(hostspecs, env)
        if not ports
        else _validate_asyncpg_ports(hostspecs, ports)
    )
    hosts: list[str] = []
    resolved_ports: list[int] = []

    for index, hostspec in enumerate(hostspecs):
        if not hostspec:
            # asyncpg indexes the first character while parsing and rejects
            # empty members rather than treating them as libpq socket defaults.
            raise ValueError("empty host in PostgreSQL host list")
        if hostspec[0] == "/":
            host = hostspec
            host_port = ""
        elif hostspec[0] == "[":
            match = re.match(r"(?:\[([^\]]+)\])(?::([0-9]+))?", hostspec)
            if not match:
                raise ValueError("invalid IPv6 address in PostgreSQL host list")
            host = match.group(1)
            host_port = match.group(2) or ""
        else:
            host, _, host_port = hostspec.partition(":")

        if unquote_hosts:
            host = unquote(host)
        hosts.append(host)
        if not ports:
            if host_port and unquote_hosts:
                host_port = unquote(host_port)
            resolved_ports.append(
                int(host_port) if host_port else defaults[index]
            )

    return hosts, ports or resolved_ports


def _resolve_asyncpg_hosts(
    authority_hostspec: str, query: dict[str, str], env: dict
) -> tuple[list[str], list[int], list[str]]:
    """Resolve asyncpg's effective ordered hosts and per-host ports."""
    hosts: list[str] | None = None
    ports: list[int] | None = None
    auth_hosts: list[str] | None = None

    if authority_hostspec:
        hosts, ports = _parse_asyncpg_hostlist(
            authority_hostspec, None, env, unquote_hosts=True
        )
    else:
        query_port = query.get("port")
        if query_port:
            ports = [int(port) for port in query_port.split(",")]

        query_host = query.get("host")
        if query_host:
            hosts, ports = _parse_asyncpg_hostlist(query_host, ports, env)

    if not hosts:
        environment_host = env.get("PGHOST")
        # asyncpg deliberately ignores an empty PGHOST. Libpq does not: it
        # interprets it as its compiled socket directory, so it must never be
        # copied verbatim into the translated DSN.
        if environment_host:
            hosts, ports = _parse_asyncpg_hostlist(
                environment_host, ports, env
            )

    if not hosts:
        # asyncpg uses only ``localhost`` to select a default passfile entry,
        # even though it subsequently attempts every platform socket before
        # localhost. Keep that distinct authentication host list intact.
        auth_hosts = ["localhost"]
        hosts = (
            ["localhost"]
            if sys.platform == "win32"
            else [
                "/run/postgresql",
                "/var/run/postgresql",
                "/tmp",
                "/private/tmp",
                "localhost",
            ]
        )

    if not ports:
        ports = _asyncpg_default_ports(hosts, env)
    else:
        ports = _validate_asyncpg_ports(hosts, [int(port) for port in ports])
    return hosts, ports, auth_hosts or hosts


def _dsn_stated_options(parts, query: dict[str, str]) -> set[str]:
    """Connection options the runtime URI states before env resolution."""
    user, password, hostspec = _authority_fields(parts.netloc)
    stated = set(query).intersection(_ASYNCPG_CONNECTION_QUERY_OPTIONS)
    if hostspec:
        stated.add("host")
    if user:
        stated.add("user")
    if password:
        stated.add("password")
    if parts.path:
        stated.add("dbname")
    if "database" in stated:
        stated.add("dbname")
    return stated


def _libpq_option_fragment(name: str, value: str) -> str:
    """Encode one startup setting for libpq's command-line-style options."""
    # ``-c name=value`` has no escaping for its first equals delimiter. A NUL
    # cannot cross libpq's C-string boundary either. Reject these shapes
    # instead of probing a different setting from the one asyncpg will send in
    # its startup packet.
    if not name or "=" in name or "\x00" in name:
        raise ValueError(
            "runtime PostgreSQL startup setting name cannot be represented "
            "by the diagnostic driver"
        )
    if "\x00" in value:
        raise ValueError(
            "runtime PostgreSQL startup setting value cannot be represented "
            "by the diagnostic driver"
        )
    # libpq passes ``options`` through PostgreSQL's command-line splitter.
    # Backslash is its escape character. PostgreSQL splits on every whitespace
    # character, so escape tabs and newlines as well as ordinary spaces.
    setting = f"{name}={value}".replace("\\", "\\\\")
    setting = "".join(f"\\{char}" if char.isspace() else char for char in setting)
    return f"-c {setting}"


def _state_asyncpg_connection_defaults(
    parts,
    query: dict[str, str],
    hosts: list[str],
    user: str,
    env: dict,
) -> None:
    """State asyncpg's basic defaults for libpq's connection.

    The drivers have different implicit host resolution: asyncpg tries its
    platform socket list and then localhost, while libpq uses one compiled-in
    socket directory. Making every database-selecting default explicit keeps
    doctor on asyncpg's endpoint and also prevents an ambient ``PGSERVICE``
    recipe from filling a value that asyncpg never reads.
    """
    effective_user = unquote(user) if user else query.get("user")
    if not effective_user:
        effective_user = next(
            (
                env[name]
                for name in ("LOGNAME", "USER", "LNAME", "USERNAME")
                if env.get(name)
            ),
            None,
        )
        if not effective_user:
            try:
                effective_user = getpass.getuser()
            except (OSError, KeyError, ImportError) as exc:
                raise ValueError(
                    "runtime PostgreSQL user could not be determined"
                ) from exc
        query["user"] = effective_user

    if not parts.path and "dbname" not in query:
        query["dbname"] = effective_user

    if "sslmode" not in query:
        # State asyncpg's host-derived default unconditionally. Besides making
        # libpq match asyncpg, this prevents libpq-only PGREQUIRESSL from
        # changing TCP ``prefer`` or Unix-socket ``disable`` semantics.
        have_tcp_host = any(not host.startswith("/") for host in hosts)
        query["sslmode"] = "prefer" if have_tcp_host else "disable"

    if _libpq_accepts_dsn_option("target_session_attrs", "any"):
        query.setdefault("target_session_attrs", "any")


def _normalize_asyncpg_direct_tls(query: dict[str, str]) -> None:
    """Keep asyncpg's direct-TLS intent valid and fail-closed in libpq."""
    if query.get("sslnegotiation") != "direct":
        return

    sslmode = query.get("sslmode")
    if sslmode == "disable":
        # asyncpg never consults the negotiation mode when SSL is disabled.
        # State libpq's ordinary negotiation explicitly so an ambient
        # PGSSLNEGOTIATION cannot recreate its invalid direct+disable pair.
        query["sslnegotiation"] = "postgres"
    elif sslmode == "allow":
        # asyncpg tries plaintext first for ``allow`` and only retries with
        # TLS after a connection failure. libpq's ordinary negotiation has
        # the same ordering and, unlike direct negotiation, is valid with a
        # weak SSL mode.
        query["sslnegotiation"] = "postgres"
    elif sslmode == "prefer":
        # libpq only permits direct negotiation with require or a verifying
        # mode. Keep the direct-TLS path asyncpg attempts, while conservatively
        # declining asyncpg's weaker plaintext fallback in the diagnostic.
        query["sslmode"] = "require"


def _validate_asyncpg_connection_options(query: dict[str, str]) -> None:
    """Reject enum spellings asyncpg refuses before opening a socket."""
    allowed_values = {
        "sslmode": {
            "disable",
            "allow",
            "prefer",
            "require",
            "verify-ca",
            "verify-full",
        },
        "sslnegotiation": {"postgres", "direct"},
        "target_session_attrs": {
            "any",
            "primary",
            "standby",
            "prefer-standby",
            "read-write",
            "read-only",
        },
        "gsslib": {"gssapi", "sspi"},
    }
    for name, allowed in allowed_values.items():
        if name in query and query[name] not in allowed:
            raise ValueError(
                f"runtime PostgreSQL {name} connection option is not valid "
                "for asyncpg"
            )


_ASYNCPG_CONNECTION_FILE_OPTIONS = frozenset(
    {"passfile", "sslrootcert", "sslcrl", "sslkey", "sslcert"}
)


def _resolve_connection_file_paths(query: dict[str, str], project_dir: Path) -> None:
    """Resolve files from the spawned agent's working directory."""
    working_dir = project_dir.resolve()
    for name in _ASYNCPG_CONNECTION_FILE_OPTIONS:
        value = query.get(name)
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = (working_dir / path).resolve()
            query[name] = str(path)


def _read_asyncpg_passfile_password(
    passfile: Path,
    hosts: list[str],
    ports: list[int],
    database: str,
    user: str,
) -> str | None:
    """Select a passfile password with asyncpg 0.30's ordered-host rules."""
    try:
        if not passfile.is_file():
            return None
        if os.name != "nt" and passfile.stat().st_mode & 0o077:
            return None
        # Text-mode universal-newline handling matches asyncpg's file iterator;
        # splitting only on the translated newline avoids treating vertical
        # tabs and other Unicode line separators as record boundaries.
        lines = passfile.read_text().split("\n")
    except OSError:
        return None

    entries: list[tuple[str, ...]] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Match asyncpg's parser exactly: protect escaped backslashes while
        # splitting on unescaped colons, then restore them afterward.
        protected = line.replace(r"\\", "\n")
        entries.append(
            tuple(
                field.replace("\n", r"\\")
                for field in re.split(r"(?<!\\):", protected, maxsplit=4)
            )
        )

    for host, port in zip(hosts, ports):
        auth_host = "localhost" if host.startswith("/") else host
        for entry in entries:
            if len(entry) != 5:
                raise ValueError("runtime PostgreSQL passfile is malformed")
            phost, pport, pdatabase, puser, password = entry
            if phost not in {"*", auth_host}:
                continue
            if pport not in {"*", str(port)}:
                continue
            if pdatabase not in {"*", database}:
                continue
            if puser not in {"*", user}:
                continue
            return password
    return None


def _fold_asyncpg_passfile(
    parts,
    query: dict[str, str],
    hosts: list[str],
    ports: list[int],
    auth_hosts: list[str],
    env: dict,
    project_dir: Path,
) -> None:
    """Freeze asyncpg's passfile behavior into explicit libpq parameters.

    Asyncpg scans the ordered hosts once, selects the first matching passfile
    password, and reuses it for every connection attempt. Libpq consults the
    passfile again for each attempted host. Folding asyncpg's selection into
    an explicit password prevents doctor from authenticating as a different
    effective client after the first host fails.
    """
    authority_user, authority_password, _ = _authority_fields(parts.netloc)
    if authority_password:
        return
    if "password" in query:
        if query["password"] == "":
            query["passfile"] = _absent_passfile_path(project_dir)
        return
    if query.get("passfile") == "":
        query["passfile"] = _absent_passfile_path(project_dir)
        return

    user = unquote(authority_user) if authority_user else query["user"]
    if parts.path:
        database_path = (
            parts.path[1:] if parts.path.startswith("/") else parts.path
        )
        database = unquote(database_path)
    else:
        database = query["dbname"]

    if "passfile" in query:
        # An explicitly empty PGPASSFILE becomes ``Path('')`` in asyncpg and
        # therefore does not fall through to the default file.
        passfile = Path(query["passfile"]) if query["passfile"] else None
    elif sys.platform == "win32":
        # Asyncpg uses Windows' roaming AppData known folder rather than HOME.
        from asyncpg import compat as _asyncpg_compat

        home = _asyncpg_compat.get_pg_home_directory()
        passfile = home / "pgpass.conf" if home is not None else None
    else:
        try:
            # The agent process receives ``env`` wholesale. Path.home() reads
            # its HOME, including a project-.env override, before consulting
            # the OS account database.
            home = Path(env["HOME"] or "/") if "HOME" in env else Path.home()
        except (RuntimeError, KeyError):
            home = None
        if home is not None and not home.is_absolute():
            home = project_dir.resolve() / home
        passfile = home / ".pgpass" if home is not None else None

    password = (
        _read_asyncpg_passfile_password(
            passfile, auth_hosts, ports, database, user
        )
        if passfile is not None
        else None
    )
    if password is not None:
        query["password"] = password

    # With no match asyncpg attempts passwordless authentication; it does not
    # fall through to another passfile. A path beneath a directory verified
    # absent suppresses libpq's lookup silently. ``os.devnull`` is not a
    # regular file, so libpq warns once per attempted host when used as a
    # passfile sentinel.
    query["passfile"] = _absent_passfile_path(project_dir)


def _absent_passfile_path(project_dir: Path) -> str:
    """Return a stable path libpq can stat as absent without warning."""
    root = project_dir.resolve()
    suffix = 0
    while True:
        name = ".kestrel-doctor-no-passfile"
        if suffix:
            name = f"{name}-{suffix}"
        absent_directory = root / name
        if not absent_directory.exists():
            return str(absent_directory / "pgpass")
        suffix += 1


def _doctor_postgres_dsn(
    runtime_dsn: str, env: dict, project_dir: Path
) -> str:
    """Translate asyncpg's effective connection data into a libpq URI.

    ``runtime_dsn`` remains authoritative. Only fields it does not state are
    filled from the launcher environment, and query options asyncpg treats as
    server settings are carried through libpq's ``options`` parameter rather
    than stripped or accidentally reclassified as libpq connection options.

    The return value is the string stored on :class:`_GovernanceSource`, so a
    password folded in from ``env`` remains inside the redaction boundary used
    by every driver-error path.
    """
    try:
        parts = urlsplit(runtime_dsn)
        if parts.scheme not in {"postgres", "postgresql"}:
            raise ValueError("not an asyncpg PostgreSQL URI")
        raw_query = dict(parse_qsl(parts.query, strict_parsing=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime PostgreSQL DSN is not valid for asyncpg"
        ) from exc

    stated = _dsn_stated_options(parts, raw_query)
    # Keep asyncpg's classification intact through precedence resolution and
    # cross-driver normalization. Capability validation happens only after the
    # final effective option set exists; an explicit constraint must never be
    # silently deleted merely because the linked libpq predates it.
    query = {
        name: value
        for name, value in raw_query.items()
        if name in _ASYNCPG_CONNECTION_QUERY_OPTIONS
    }

    # asyncpg accepts ``database`` as an alias; libpq accepts ``dbname``.
    # A URI path wins over either query spelling in asyncpg, so ignored query
    # copies are removed instead of being allowed to override the path here.
    if parts.path:
        query.pop("dbname", None)
        query.pop("database", None)
    elif "dbname" in query:
        query.pop("database", None)
    elif "database" in query:
        query["dbname"] = query.pop("database")

    user, password, hostspec = _authority_fields(parts.netloc)
    hosts, ports, auth_hosts = _resolve_asyncpg_hosts(
        hostspec, raw_query, env
    )
    query["host"] = ",".join(hosts)
    query["port"] = ",".join(str(port) for port in ports)
    if user:
        query.pop("user", None)
    if password:
        query.pop("password", None)

    for env_name, dsn_name in _ASYNCPG_ENV_DSN_OPTIONS:
        value = env.get(env_name)
        if (
            value is not None
            and dsn_name not in stated
        ):
            query[dsn_name] = value

    # State the defaults even without PGSERVICE: asyncpg's host fallback list
    # differs from libpq's compiled-in socket default. A service-file path
    # alone remains inert, while an actual service recipe has no unstated
    # database-selecting value left to fill. This never mutates os.environ.
    has_libpq_service = "PGSERVICE" in env
    _state_asyncpg_connection_defaults(parts, query, hosts, user, env)
    _normalize_asyncpg_direct_tls(query)
    _validate_asyncpg_connection_options(query)

    for env_name, dsn_name, asyncpg_default in _LIBPQ_ONLY_ENV_DSN_DEFAULTS:
        should_neutralize = env_name in env or has_libpq_service
        if should_neutralize and _libpq_accepts_dsn_option(
            dsn_name, asyncpg_default
        ):
            query[dsn_name] = asyncpg_default

    _resolve_connection_file_paths(query, project_dir)
    _fold_asyncpg_passfile(
        parts, query, hosts, ports, auth_hosts, env, project_dir
    )

    for name, value in query.items():
        _require_libpq_dsn_option(name, value)

    # These are server settings under asyncpg even when libpq happens to have
    # a connection parameter with the same name. PostgreSQL processes the raw
    # ``options`` field before direct startup settings regardless of URI order,
    # so put it first and append every other setting through ``-c name=value``.
    option_fragments = [raw_query["options"]] if "options" in raw_query else []
    direct_settings = [
        (name, value)
        for name, value in raw_query.items()
        if name not in _ASYNCPG_CONNECTION_QUERY_OPTIONS and name != "options"
    ]
    if option_fragments and direct_settings:
        trailing_backslashes = len(option_fragments[0]) - len(
            option_fragments[0].rstrip("\\")
        )
        if trailing_backslashes % 2:
            raise ValueError(
                "runtime PostgreSQL options value cannot be combined "
                "losslessly with direct startup settings"
            )
    for name, value in direct_settings:
        option_fragments.append(_libpq_option_fragment(name, value))
    # State this unconditionally. ``options=`` is a valid explicit empty value
    # and prevents ambient PGOPTIONS from changing doctor's schema or startup
    # settings when the spawned asyncpg process would ignore it.
    query["options"] = " ".join(option_fragments)

    # ``urlencode`` defaults to quote_plus. libpq percent-decodes according to
    # RFC 3986 and keeps '+' literal, which would turn ``-c search_path=...``
    # into ``-c+search_path=...``. Use percent-encoded spaces explicitly.
    encoded_query = urlencode(query, quote_via=quote)
    # Building this directly preserves the URI's ``//`` marker when it has no
    # authority (``postgresql:///db``). ``urlunsplit`` normalizes that valid
    # asyncpg/libpq form to the invalid ``postgresql:/db``.
    # The normalized host list lives in query parameters so every host can
    # carry the exact per-host port asyncpg resolved. Retain only URI userinfo;
    # leaving the original authority hosts in place would let libpq reinterpret
    # an embedded port or an IPv6/socket spelling a second time.
    auth_netloc = (
        f"{parts.netloc.partition('@')[0]}@" if "@" in parts.netloc else ""
    )
    path = (
        parts.path
        if not parts.path or parts.path.startswith("/")
        else f"/{parts.path}"
    )
    effective = f"{parts.scheme}://{auth_netloc}{path}"
    return f"{effective}?{encoded_query}" if encoded_query else effective


def _safe(exc: object, source: "_GovernanceSource") -> str:
    """A driver error with the connection string taken out of it.

    libpq echoes the DSN it could not parse. An unmatched ``[`` in a URI is
    enough:

        invalid dsn: end of string reached when looking for matching "]"
        in IPv6 host address in URI: "postgresql://user:hunter2@[bad/kestrel"

    Doctor puts these straight into ``report.warn``, which an operator reads on
    a terminal and CI archives, so the unredacted form prints the database
    password. The whole DSN goes, not just the password: it also names a host
    and database an operator may not want in a build log, and a message that
    keeps everything except the secret is one parser bug away from keeping the
    secret too.
    """
    text = str(exc)
    if not source.dsn:
        return text

    text = text.replace(source.dsn, "<dsn>")

    # Then the individual fields, because the whole-string replace above only
    # fires on a byte-identical echo — and that is the *rare* case. A malformed
    # URI gets quoted back verbatim, but an ordinary DNS, connection, or
    # authentication failure names the parts instead:
    #
    #     could not translate host name "db.internal" to address
    #     FATAL:  password authentication failed for user "kestrel"
    #
    # so a redaction that only handled the verbatim echo left the database
    # topology and account names in every routine outage message. Password
    # first: it is the one field whose disclosure is unrecoverable, and
    # redacting longest-first stops a shorter token reappearing inside a
    # replacement.
    for secret in _dsn_secrets(source.dsn):
        text = text.replace(secret, "<redacted>")
    for field_name, value in _dsn_connection_files(source.dsn):
        text = text.replace(value, f"<{field_name}>")
    for field_name, value in _dsn_identity(source.dsn):
        text = text.replace(value, f"<{field_name}>")
    return text


def _dsn_secrets(dsn: str) -> tuple:
    """Every secret-bearing token in ``dsn``, longest first.

    Longest first so that redacting one token cannot leave a shorter one
    embedded in the replacement text.
    """
    secrets = set()
    try:
        from psycopg2.extensions import parse_dsn

        parsed = parse_dsn(dsn)
        for field in ("password", "sslpassword"):
            value = parsed.get(field)
            if value:
                secrets.add(value)
    except Exception:  # noqa: BLE001 — an unparseable DSN is why we are here
        pass

    # A bounded DSN is reserialized by libpq and is no longer byte-identical to
    # ``source.dsn``. Collect query credentials independently so their decoded
    # forms remain redacted even when libpq echoes only ``field=value``.
    try:
        for field, value in parse_qsl(
            urlsplit(dsn).query, strict_parsing=True
        ):
            if field in {"password", "sslpassword"} and value:
                secrets.add(value)
    except (TypeError, ValueError):
        pass

    # ``parse_dsn`` raises on exactly the malformed URIs that leak, so fall
    # back to the URI's own shape: everything between "://user:" and the "@".
    match = re.search(r"://[^:/?#]*:([^@/?#]+)@", dsn)
    if match:
        encoded_password = match.group(1)
        secrets.add(encoded_password)
        secrets.add(unquote(encoded_password))

    # ``_bounded_dsn`` reserializes a URI as libpq conninfo. In that spelling
    # apostrophes and backslashes gain a backslash, and whitespace-bearing
    # values are quoted. Redact both the logical value and the exact escaped
    # token a bounded-driver diagnostic can echo.
    for secret in tuple(secrets):
        conninfo_secret = re.sub(r"([\\'])", r"\\\1", secret)
        if any(character.isspace() for character in conninfo_secret):
            conninfo_secret = f"'{conninfo_secret}'"
        secrets.add(conninfo_secret)

    return tuple(sorted(secrets, key=len, reverse=True))


def _dsn_connection_files(dsn: str) -> tuple:
    """Resolved connection-file paths that driver errors must not disclose."""
    try:
        from psycopg2.extensions import parse_dsn

        parsed = parse_dsn(dsn)
    except Exception:  # noqa: BLE001 — unparseable; whole-DSN redaction remains
        return ()

    found = [
        (field, parsed[field])
        for field in _ASYNCPG_CONNECTION_FILE_OPTIONS
        if isinstance(parsed.get(field), str) and parsed[field]
    ]
    return tuple(sorted(found, key=lambda pair: len(pair[1]), reverse=True))


#: Identity-bearing DSN fields, and the placeholder each becomes. Not secrets,
#: but not things to scatter through CI logs either: together they are the map
#: of an operator's database estate.
_DSN_IDENTITY_FIELDS = ("host", "user", "dbname")


def _dsn_identity(dsn: str) -> tuple:
    """``(field, value)`` for each identity field in ``dsn``, longest first.

    Length-sorted for the same reason as the secrets, and short values are
    dropped: a one- or two-character host or user is common enough as an
    ordinary English fragment that replacing it would corrupt the message it
    is meant to keep readable.
    """
    try:
        from psycopg2.extensions import parse_dsn

        parsed = parse_dsn(dsn)
    except Exception:  # noqa: BLE001 — unparseable; the whole-DSN replace stands
        return ()

    found = []
    for field in _DSN_IDENTITY_FIELDS:
        value = parsed.get(field)
        if not isinstance(value, str):
            continue
        values = value.split(",") if field == "host" else [value]
        found.extend(
            (field, member) for member in values if len(member) > 2
        )
    return tuple(sorted(found, key=lambda pair: len(pair[1]), reverse=True))


def _bounded_dsn(dsn: str) -> str:
    """Apply doctor's unconditional outage bound to a translated DSN.

    PostgreSQL sources reach this function only after asyncpg-style query
    parameters have been translated. A runtime ``connect_timeout`` query is
    therefore a server setting in ``options``, never libpq's connection
    timeout. Stating five seconds here also prevents process-global
    ``PGCONNECT_TIMEOUT`` from changing the diagnostic independently of the
    spawned runtime.
    """
    try:
        from psycopg2.extensions import make_dsn

        return make_dsn(dsn, connect_timeout=_CONNECT_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001 — psycopg2.ProgrammingError on a bad DSN
        # The import is inside the ``try`` on purpose. An unparseable DSN is
        # the connection's own problem to report, with its own message, and a
        # psycopg2 that cannot offer these helpers is not a reason for doctor
        # to stop reading — either way the unbounded DSN is exactly the
        # behaviour that preceded this bound, and ``connect`` still speaks.
        return dsn


def _fetch_rows(
    source: "_GovernanceSource",
    sqlite_sql: str,
    postgres_sql: str,
    sqlite_params: tuple = (),
    postgres_params: tuple = (),
) -> list:
    """Run the backend's form of one read-only query and return its rows.

    Raises the driver's own error type; every caller maps that to
    ``_UnreadableDB``, the same shape it always did.

    Query *and* parameters are given per backend rather than templated. The
    placeholder style differs (``?`` vs ``%s``), and so does the scoping:
    SQLite reads the one agent in the anchor, PostgreSQL has to name its
    tenant. Spelling both out keeps that difference where a reader can see it.
    """
    if source.reads_the_anchor:
        # ``isolation_level=None`` and ``uri=False`` are the defaults; we rely
        # on them to avoid creating a transaction we won't commit.
        with sqlite3.connect(str(source.anchor_path)) as conn:
            return conn.execute(sqlite_sql, sqlite_params).fetchall()

    import psycopg2

    # A fresh connection per query: doctor makes a handful of reads per agent,
    # is not a hot path, and a connection that closes with its query cannot
    # outlive a diagnostic that fails halfway.
    connection = psycopg2.connect(_bounded_dsn(source.dsn))
    try:
        with connection.cursor() as cursor:
            cursor.execute(postgres_sql, postgres_params)
            return cursor.fetchall()
    finally:
        connection.close()


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


def _row_physically_exists(source: "_GovernanceSource") -> bool:
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
            "SELECT 1 FROM graph_nodes WHERE node_id = %s",
            sqlite_params=(source.agent_did,),
            postgres_params=(source.agent_did,),
        )
    except Exception:  # noqa: BLE001 — the caller's own read reports it
        return True
    return bool(rows)


def _read_agent_governance(
    multi_agent_path: Path, project_dir: Path, env: dict
) -> "list[_AgentGovernance]":
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

        source = _resolve_governance_source(
            db_path, env, project_dir, ledger_by_dsn
        )
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
        runtime_is_empty = on_postgres and not unowned and (
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
    readings: "list[_AgentGovernance]", report: DoctorReport
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
            _report_unexamined(name, node.reason, source, report)
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
    readings: "list[_AgentGovernance]", report: DoctorReport
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
    source: "_GovernanceSource",
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
        physical = _read_governed_by_targets(
            source, node_id, any_owner=True
        )
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
    name: str, reason: str, source: object, report: DoctorReport
) -> None:
    """Record that this agent's governance could not be read.

    Whether that is a *fail* turns on one question: can the **agent** open this
    database either?

    On PostgreSQL it cannot. A database that is down, misconfigured, or
    refusing this account refuses the runtime too, so readiness is false and
    must say so — ``ready`` is ``not report.fail``, and a warning here made
    ``kestrel doctor`` print "Ready" and exit 0 having inspected no governance
    at all, in exactly the state an operator most needs told about. "I did not
    check" and "I checked and it is fine" are different answers.

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
    if isinstance(source, _GovernanceSource) and not source.reads_the_anchor:
        agent_blind = True
    elif isinstance(source, _UnreadableDB):
        # Resolution failed before a source existed. Only PostgreSQL hosts
        # reach that through a network read; name it from the reason.
        agent_blind = "PostgreSQL" in reason
    else:
        agent_blind = False

    if agent_blind:
        report.fail.append(
            f"{name}: governance NOT verified — {reason}. Doctor cannot say "
            f"whether this agent is correctly anchored, and the runtime cannot "
            f"reach that database either; fix the access problem and re-run "
            f"before treating this host as ready."
        )
    else:
        report.warn.append(
            f"{name}: constitution drift check skipped — {reason}"
        )


def _rollback_advice(source: "_GovernanceSource") -> str:
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
    "WHERE node_id = %s AND node_type = 'agent' AND EXISTS ("
    "  SELECT 1 FROM graph_node_owners AS owner "
    "  WHERE owner.node_id = graph_nodes.node_id AND owner.agent_id = %s)"
)
#: Physical ``governed_by`` targets regardless of who witnesses them. Compared
#: against the scoped read to tell "no such edge" from "an edge this agent
#: cannot use" — different findings, with different remedies.
_GOVERNED_BY_ANY_OWNER_SQLITE = (
    "SELECT target_id FROM graph_edges "
    "WHERE source_id = ? AND label = 'governed_by'"
)
_GOVERNED_BY_ANY_OWNER_PG = (
    "SELECT target_id FROM graph_edges "
    "WHERE source_id = %s AND label = 'governed_by'"
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
    "WHERE source_id = %s AND label = 'governed_by' AND EXISTS ("
    "  SELECT 1 FROM graph_edge_owners AS owner "
    "  WHERE owner.source_id = graph_edges.source_id "
    "  AND owner.target_id = graph_edges.target_id "
    "  AND owner.label = graph_edges.label AND owner.agent_id = %s)"
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
    "WHERE node_id = %s AND node_type = 'agent' AND %s IS NOT NULL"
)
_GOVERNED_BY_SQLITE_LEGACY = (
    "SELECT target_id FROM graph_edges "
    "WHERE source_id = ? AND label = 'governed_by' AND ? IS NOT NULL"
)
_GOVERNED_BY_PG_LEGACY = (
    "SELECT target_id FROM graph_edges "
    "WHERE source_id = %s AND label = 'governed_by' AND %s IS NOT NULL"
)


@dataclass(frozen=True)
class _UnreadableDB:
    """Sentinel: the agent DB exists but stock sqlite3 can't open it."""

    reason: str


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


def _read_agent_node(source: "_GovernanceSource"):
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
            _AGENT_NODE_SQLITE if source.ownership_settled
            else _AGENT_NODE_SQLITE_LEGACY,
            _AGENT_NODE_PG if source.ownership_settled
            else _AGENT_NODE_PG_LEGACY,
            sqlite_params=(source.agent_did, source.agent_did),
            postgres_params=(source.agent_did, source.agent_did),
        )
    except sqlite3.DatabaseError as exc:
        return _UnreadableDB(reason=f"DB unreadable ({exc})")
    except sqlite3.Error as exc:
        return _UnreadableDB(reason=f"sqlite error ({exc})")
    except Exception as exc:  # noqa: BLE001 — psycopg2 raises its own tree
        return _UnreadableDB(
            reason=f"cannot read {source.describe()} ({_safe(exc, source)})"
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
    source: "_GovernanceSource", source_id: str, *, any_owner: bool = False
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
            _GOVERNED_BY_ANY_OWNER_SQLITE if any_owner
            else _GOVERNED_BY_SQLITE if source.ownership_settled
            else _GOVERNED_BY_SQLITE_LEGACY,
            _GOVERNED_BY_ANY_OWNER_PG if any_owner
            else _GOVERNED_BY_PG if source.ownership_settled
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
    except Exception as exc:  # noqa: BLE001 — psycopg2 raises its own tree
        return _UnreadableDB(
            reason=(
                f"cannot read graph_edges in {source.describe()} "
                f"({_safe(exc, source)})"
            )
        )

    return tuple(row[0] for row in rows if row[0])
