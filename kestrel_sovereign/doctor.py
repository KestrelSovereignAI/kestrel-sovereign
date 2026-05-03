"""``kestrel doctor`` — diagnose readiness without making any changes.

The doctor produces a structured :class:`DoctorReport` that the CLI
formats. It is also reused as the ``verify`` step at the end of
``kestrel setup``.

Checks performed:

  - ``KESTREL_DATA_KEY`` set in ``.env``
  - ``[llm]`` section present with a non-empty ``route_priority``
  - For each cloud route in ``route_priority``, the matching
    ``api_key_env`` is set in ``.env``
  - At least one agent registered in ``rookery.toml``
  - For each registered agent, ``kestrel_prime.db`` exists
  - For each registered agent, the anchored ``constitution_hash``
    matches the SHA256 of the canonical KESTREL_CONSTITUTION.md.
    Drift here means the agent is silently governing itself by an
    older constitution than what's on disk — see ``_check_constitution_drift``.

This is deliberately minimal. We avoid reaching out to Ollama / OpenAI
— that's flaky in CI and out of scope for "is the config sane?"
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from kestrel_sovereign.rookery.config import ROOKERY_CONFIG_FILENAME, RookeryConfig
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
    rookery_path = project_dir / ROOKERY_CONFIG_FILENAME

    env = read_env(env_path)
    config = read_toml(toml_path)

    _check_data_key(env, env_path, report)
    _check_llm(config, env, toml_path, report)
    _check_rookery(rookery_path, project_dir, report)
    _check_constitution_drift(rookery_path, project_dir, report)

    return report


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
        api_key_env = route.get("api_key_env")
        if api_key_env and not env.get(api_key_env):
            report.fail.append(
                f"{api_key_env} not set in .env (required for {route_id})"
            )
        elif api_key_env:
            report.ok.append(f"{api_key_env} set for {route_id}")


def _check_rookery(
    rookery_path: Path, project_dir: Path, report: DoctorReport
) -> None:
    rookery = RookeryConfig.load(rookery_path, auto_discover_fallback=False)
    agents = rookery.get_local_agents()
    if not agents:
        report.fail.append(
            f"No local agents in {rookery_path} — run `kestrel setup agent`"
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


def _check_constitution_drift(
    rookery_path: Path, project_dir: Path, report: DoctorReport
) -> None:
    """Compare each agent's anchored constitution_hash against the on-disk file.

    For each local rookery agent: open ``kestrel_prime.db`` with stock
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

    Per-agent overlay (``<agent_dir>/CONSTITUTION.md``, #898) is
    deliberately NOT compared here. Overlays affect runtime grant
    lookups but are not anchored at inception, so there is no anchor
    for them to drift against. (If we ever start anchoring overlay
    text, this check needs an extension.)
    """
    rookery = RookeryConfig.load(rookery_path, auto_discover_fallback=False)
    agents = rookery.get_local_agents()
    if not agents:
        return

    canonical = _canonical_constitution_path()
    try:
        on_disk_hash = hashlib.sha256(canonical.read_bytes()).hexdigest()
    except OSError as exc:
        report.warn.append(
            f"Constitution drift check skipped — cannot read canonical "
            f"{canonical}: {exc}"
        )
        return

    for name, cfg in agents.items():
        db_path = (project_dir / cfg.data_dir / "kestrel_prime.db").resolve()
        if not db_path.exists():
            # Already reported by _check_rookery; don't duplicate.
            continue

        stored_hash = _read_anchored_constitution_hash(db_path)
        if isinstance(stored_hash, _UnreadableDB):
            report.warn.append(
                f"{name}: constitution drift check skipped — {stored_hash.reason}"
            )
            continue
        if isinstance(stored_hash, _NoAgentNode):
            report.warn.append(
                f"{name}: constitution drift check skipped — no agent node in DB"
            )
            continue
        if isinstance(stored_hash, _NoHashProperty):
            report.warn.append(
                f"{name}: constitution drift check skipped — agent node missing "
                f"constitution_hash property (older agent? re-incept to anchor)"
            )
            continue

        if stored_hash == on_disk_hash:
            report.ok.append(
                f"{name}: constitution anchored to current file ({stored_hash[:12]}…)"
            )
        else:
            # Don't name a remediation command that doesn't exist yet.
            # Drift detection ships before reanchor (the writer touches
            # five DB locations + RAG index — its own PR with proper
            # integration tests). The message tells the user exactly
            # what diverged so they can act; once the reanchor CLI
            # lands, this hint should be replaced with the command.
            report.fail.append(
                f"{name}: constitution drift — stored {stored_hash[:12]}… "
                f"does not match {canonical} ({on_disk_hash[:12]}…). "
                f"Agent is anchored to an older copy; reanchor support "
                f"is planned but not yet shipped."
            )


def _canonical_constitution_path() -> Path:
    """Return the package's canonical constitution path (config.CONSTITUTION_PATH).

    Imported lazily — pulling kestrel_sovereign.config at module top would
    drag in a chunk of the package and slow doctor's cold-start path.
    """
    from kestrel_sovereign.config import CONSTITUTION_PATH

    return Path(CONSTITUTION_PATH)


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


def _read_anchored_constitution_hash(db_path: Path):
    """Read the agent node's constitution_hash from a Kestrel DB.

    Returns:
        - ``str`` of the hash on success.
        - ``_UnreadableDB(reason=...)`` if the DB is sqlcipher-encrypted
          or otherwise unreadable.
        - ``_NoAgentNode()`` if there's no agent row.
        - ``_NoHashProperty()`` if there's an agent row but no
          ``constitution_hash`` property.
    """
    try:
        # ``isolation_level=None`` and ``uri=False`` are the default; we
        # rely on the default to avoid creating a transaction we won't commit.
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT properties FROM graph_nodes "
                "WHERE node_type='agent' LIMIT 1"
            ).fetchone()
    except sqlite3.DatabaseError as exc:
        # 'file is not a database' (sqlcipher-encrypted) lands here.
        return _UnreadableDB(reason=f"DB unreadable ({exc})")
    except sqlite3.Error as exc:
        return _UnreadableDB(reason=f"sqlite error ({exc})")

    if row is None:
        return _NoAgentNode()

    properties_raw = row[0]
    if properties_raw is None:
        return _NoHashProperty()

    try:
        properties = (
            json.loads(properties_raw)
            if isinstance(properties_raw, (str, bytes, bytearray))
            else properties_raw
        )
    except (TypeError, ValueError):
        return _NoHashProperty()

    if not isinstance(properties, dict):
        return _NoHashProperty()

    stored_hash = properties.get("constitution_hash")
    if not stored_hash or not isinstance(stored_hash, str):
        return _NoHashProperty()

    return stored_hash
