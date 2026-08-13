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

This is deliberately minimal. We avoid reaching out to Ollama / OpenAI
— that's flaky in CI and out of scope for "is the config sane?"
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path

from kestrel_sovereign.llm.route_credentials import accepted_credential_envs
from kestrel_sovereign.identity.protected_export import (
    audit_legacy_identity_exports,
    effective_identity_export_roots,
)
from kestrel_sovereign.multi_agent.config import MULTI_AGENT_CONFIG_FILENAME, MultiAgentConfig
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

    return report


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
    anchor_path: Path, env: dict, ledger_by_dsn: dict | None = None
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
        source = _GovernanceSource(
            anchor_path=anchor_path,
            agent_did=agent_did,
            dsn=env["KESTREL_DATABASE_URL"],
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

        password = parse_dsn(dsn).get("password")
        if password:
            secrets.add(password)
    except Exception:  # noqa: BLE001 — an unparseable DSN is why we are here
        pass

    # ``parse_dsn`` raises on exactly the malformed URIs that leak, so fall
    # back to the URI's own shape: everything between "://user:" and the "@".
    match = re.search(r"://[^:/?#]*:([^@/?#]+)@", dsn)
    if match:
        secrets.add(match.group(1))

    return tuple(sorted(secrets, key=len, reverse=True))


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

    found = [
        (field, parsed[field])
        for field in _DSN_IDENTITY_FIELDS
        if isinstance(parsed.get(field), str) and len(parsed[field]) > 2
    ]
    return tuple(sorted(found, key=lambda pair: len(pair[1]), reverse=True))


def _bounded_dsn(dsn: str) -> str:
    """``dsn`` with a connection timeout, unless it already states one.

    ``psycopg2.connect(dsn, connect_timeout=...)`` merges through
    ``make_dsn``, where the **keyword wins** over the DSN. Passing the default
    that way would silently overrule an operator who tuned a slow or distant
    link, and doctor would then report a reachable database as unreadable —
    the cry-wolf failure this whole change is about. So the default is applied
    only where the DSN is silent.
    """
    try:
        from psycopg2.extensions import make_dsn, parse_dsn

        if "connect_timeout" in parse_dsn(dsn):
            return dsn
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

    Fails closed to ``True`` — "something is there, do not call this empty" —
    so a probe that cannot run never manufactures a pending-replication verdict.
    """
    try:
        rows = _fetch_rows(
            source,
            "SELECT 1 FROM graph_nodes WHERE node_id = ? AND node_type = 'agent'",
            "SELECT 1 FROM graph_nodes WHERE node_id = %s AND node_type = 'agent'",
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

        source = _resolve_governance_source(db_path, env, ledger_by_dsn)
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
            anchor_source = _resolve_governance_source(db_path, {}, ledger_by_dsn)
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


def _read_governed_by_targets(source: "_GovernanceSource", source_id: str):
    """Return the targets of the agent's ``governed_by`` edges.

    Returns a tuple of target hashes (possibly empty), or
    ``_UnreadableDB(reason=...)`` when the edges cannot be read — a
    sqlcipher DB, corruption, or a legacy/synthetic DB with no
    ``graph_edges`` table.
    """
    try:
        rows = _fetch_rows(
            source,
            _GOVERNED_BY_SQLITE if source.ownership_settled
            else _GOVERNED_BY_SQLITE_LEGACY,
            _GOVERNED_BY_PG if source.ownership_settled
            else _GOVERNED_BY_PG_LEGACY,
            sqlite_params=(source_id, source.agent_did),
            postgres_params=(source_id, source.agent_did),
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
