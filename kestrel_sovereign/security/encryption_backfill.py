"""One-shot backfill: encrypt pre-migration plaintext rows at rest (#1401).

The runtime write path (``AsyncConversationStore.add_message``,
``AsyncFileStore.store_file``) has set ``meta["enc"] = True`` for all
new rows since the encryption migration shipped, but pre-existing rows
were not backfilled. The opportunistic read-side
``_migrate_message`` runs only when an old row was encrypted under the
GLOBAL key and the agent key is now available — it doesn't fire for
truly plaintext rows. This module is the missing one-shot.

Operator workflow:

    # 1) Audit (read-only): see how many rows would be touched.
    kestrel migrate-encryption --data-dir agent_data/meridian --dry-run

    # 2) Stop the agent host (or shut down only the affected agent).
    kestrel stop

    # 3) Backfill.
    kestrel migrate-encryption --data-dir agent_data/meridian

    # 4) Verify: a second --dry-run should report 0 plaintext rows.
    kestrel migrate-encryption --data-dir agent_data/meridian --dry-run

    # 5) Restart.
    kestrel start

Invariants
----------

* **Idempotent.** Rows whose metadata already carries ``enc: true`` are
  excluded from the scan, so a second run is a no-op.

* **Metadata-preserving.** Existing metadata fields (``session_id``,
  ``sent_form``, ``privacy_mode``, …) survive the backfill. Only
  ``enc`` and ``key_version`` are added/updated.

* **Per-row atomic.** Each UPDATE is its own commit. A crash mid-run
  leaves the DB in a state that's safe to resume from (the unprocessed
  rows are simply still plaintext; re-running picks up where we left
  off because the WHERE clause is metadata-keyed).

* **Encrypt-only.** We never decrypt existing rows. Rows already
  encrypted under any key are untouched.

* **Soft-deleted rows included.** Trash rows can be restored by the
  user via the Trash UI (#763) and must be encrypted-at-rest too,
  otherwise a restore would silently make a previously-protected row
  visible-in-plain on disk.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from kestrel_sovereign.security.encryption import (
    get_agent_fernet,
    get_fernet,
)
from kestrel_sdk.security.encryption import (
    encrypt_bytes,
    encrypt_string_fernet,
)

logger = logging.getLogger(__name__)


# Mirror ``async_conversation_store.CURRENT_KEY_VERSION``. Duplicated
# (rather than imported) so the migration module doesn't pull in the
# async storage stack. Bump in lockstep if that constant ever changes.
CURRENT_KEY_VERSION = 1


def _agent_pid_file(data_dir: Path) -> Path:
    """Path to the agent's pid file under ``data_dir``.

    Inlined here (rather than imported from
    ``kestrel_sovereign.multi_agent.process_manager``) to keep this
    module's import graph small. Going through ``multi_agent``
    pulls in ``KestrelAgent`` + the full LLM stack, which makes the
    migration command share a fate with any in-flight breakage in
    those modules — exactly what stops an operator from running the
    migration as a recovery step.

    Must stay in lockstep with
    ``ProcessManager.agent_pid_file``; that's the canonical
    location and any change there should mirror here.
    """
    return data_dir / "agent.pid"


def _read_pid(pid_file: Path) -> Optional[int]:
    """Read a pid integer from ``pid_file`` or return None."""
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


def _is_process_running(pid: int) -> bool:
    """Cross-platform liveness check for ``pid``.

    Inlined for the same reason as ``_agent_pid_file`` — keep the
    import graph small. Logic mirrors
    ``ProcessManager.is_process_running``.
    """
    import sys as _sys
    if _sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:  # noqa: BLE001
            return False
    import os as _os
    try:
        _os.kill(pid, 0)
        return True
    except OSError:
        return False


@dataclass
class TableReport:
    """Per-table counts surfaced to the operator."""
    table: str
    scanned: int = 0
    plaintext: int = 0
    encrypted_now: int = 0
    skipped_no_fernet: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "scanned": self.scanned,
            "plaintext": self.plaintext,
            "encrypted_now": self.encrypted_now,
            "skipped_no_fernet": self.skipped_no_fernet,
            "errors": list(self.errors),
        }


@dataclass
class BackfillReport:
    """Combined report for ``migrate-encryption``."""
    db_path: str
    agent_id: Optional[str]
    dry_run: bool
    conversation: TableReport
    files: TableReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "agent_id": self.agent_id,
            "dry_run": self.dry_run,
            "conversation": self.conversation.to_dict(),
            "files": self.files.to_dict(),
        }


def discover_agent_id(db_path: str | Path) -> Optional[str]:
    """Read the agent's DID from ``graph_nodes`` (node_type='agent').

    Agent DBs created by the inception flow always have exactly one
    node_type='agent' row holding the agent's DID. Multi-agent hosts
    keep each agent's DID in that agent's own DB. Returns None when
    the table is absent (legacy schema) or no agent node exists —
    the caller decides whether to require ``--agent-id``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='graph_nodes'"
        )
        if cur.fetchone() is None:
            return None
        cur.execute(
            "SELECT node_id FROM graph_nodes "
            "WHERE node_type='agent' LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _is_plaintext(metadata_json: Optional[str]) -> bool:
    """A row is plaintext when its metadata lacks ``enc: true``.

    Rows with NULL metadata, malformed JSON, or ``enc: false`` are
    treated as plaintext. Rows with ``enc: true`` are encrypted
    regardless of ``key_version``.
    """
    if not metadata_json:
        return True
    try:
        meta = json.loads(metadata_json)
    except (ValueError, TypeError):
        return True
    if not isinstance(meta, dict):
        return True
    return not meta.get("enc")


def _parse_metadata(metadata_json: Optional[str]) -> Dict[str, Any]:
    """Best-effort metadata parse. Returns {} on NULL/malformed."""
    if not metadata_json:
        return {}
    try:
        meta = json.loads(metadata_json)
    except (ValueError, TypeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def backfill_conversation_history(
    db_path: str | Path,
    *,
    agent_id: str,
    dry_run: bool = False,
) -> TableReport:
    """Re-encrypt every plaintext ``conversation_history`` row.

    Per-agent encryption (``key_version=1``) when ``KESTREL_DATA_KEY``
    is configured and the agent_id resolves a fernet; otherwise the
    method is a no-op and reports ``skipped_no_fernet`` for the
    counted rows.
    """
    report = TableReport(table="conversation_history")
    agent_fernet = get_agent_fernet(agent_id) if agent_id else None
    global_fernet = get_fernet()
    fernet = agent_fernet or global_fernet

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        # Scope to this agent's rows. Multi-agent hosts share the DB
        # in some deployments; never touch a sibling agent's bytes.
        cur.execute(
            "SELECT id, content, metadata FROM conversation_history "
            "WHERE agent_id = ?",
            (agent_id,),
        )
        rows = cur.fetchall()

        for row_id, content, metadata_json in rows:
            report.scanned += 1
            if not _is_plaintext(metadata_json):
                continue
            report.plaintext += 1
            if fernet is None:
                report.skipped_no_fernet += 1
                continue
            if dry_run:
                continue
            meta = _parse_metadata(metadata_json)
            try:
                ciphertext, was_encrypted = encrypt_string_fernet(
                    content, fernet,
                )
            except Exception as exc:  # noqa: BLE001
                report.errors.append(
                    f"row id={row_id}: encrypt failed: {exc}"
                )
                continue
            if not was_encrypted:
                # ``encrypt_string_fernet`` returns ``was_encrypted=False``
                # when no fernet is configured — we already short-circuited
                # that above. Belt-and-suspenders: count it as skipped
                # rather than silently writing the cleartext back.
                report.skipped_no_fernet += 1
                continue
            meta["enc"] = True
            meta["key_version"] = (
                CURRENT_KEY_VERSION if agent_fernet is not None else 0
            )
            cur.execute(
                "UPDATE conversation_history SET content = ?, metadata = ? "
                "WHERE id = ? AND agent_id = ?",
                (ciphertext, json.dumps(meta), row_id, agent_id),
            )
            conn.commit()
            report.encrypted_now += 1
    finally:
        conn.close()
    return report


def backfill_files(
    db_path: str | Path,
    *,
    dry_run: bool = False,
) -> TableReport:
    """Re-encrypt every plaintext ``files`` row with the global fernet.

    The file store doesn't carry a per-agent key (no ``key_version``
    field in the ``files`` schema) — files are addressed by content
    hash, which collides across agents that store the same bytes.
    Use the global fernet only.
    """
    report = TableReport(table="files")
    fernet = get_fernet()

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='files'"
        )
        if cur.fetchone() is None:
            return report
        cur.execute("SELECT content_hash, content, metadata FROM files")
        rows = cur.fetchall()
        for content_hash, content_blob, metadata_json in rows:
            report.scanned += 1
            if not _is_plaintext(metadata_json):
                continue
            report.plaintext += 1
            if fernet is None:
                report.skipped_no_fernet += 1
                continue
            if dry_run:
                continue
            meta = _parse_metadata(metadata_json)
            try:
                ciphertext, was_encrypted = encrypt_bytes(content_blob, fernet)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(
                    f"hash={content_hash}: encrypt failed: {exc}"
                )
                continue
            if not was_encrypted:
                report.skipped_no_fernet += 1
                continue
            meta["enc"] = True
            cur.execute(
                "UPDATE files SET content = ?, metadata = ? "
                "WHERE content_hash = ?",
                (ciphertext, json.dumps(meta), content_hash),
            )
            conn.commit()
            report.encrypted_now += 1
    finally:
        conn.close()
    return report


def cli_run(args, *, stdout=None, stderr=None) -> int:
    """Entry point for ``kestrel migrate-encryption``.

    Implemented here rather than in ``cli.py`` so ``cli.py``'s
    top-level imports (which transitively pull the full
    KestrelAgent / LLM stack) don't sit between operators and a
    working migration. The CLI's ``cmd_migrate_encryption`` is a
    thin wrapper that delegates here.

    ``args`` must expose ``data_dir`` (str), ``agent_id``
    (Optional[str]), and ``dry_run`` (bool). ``stdout`` / ``stderr``
    default to ``sys.stdout`` / ``sys.stderr`` and are injectable so
    tests don't have to capture the global streams.

    Exit codes:
      * ``0`` — clean (no plaintext rows OR all backfilled)
      * ``1`` — bad arguments / missing DB
      * ``2`` — live agent process is holding the DB; refuse to mutate
      * ``3`` — migration completed but at least one row error
      * ``4`` — migration completed but rows were skipped because no
                encryption key was configured
    """
    import os as _os
    import sys as _sys

    out = stdout or _sys.stdout
    err = stderr or _sys.stderr

    data_dir = Path(args.data_dir).resolve()
    db_path = data_dir / "kestrel_prime.db"
    if not db_path.exists():
        print(
            f"No kestrel_prime.db at {db_path}. Pass --data-dir pointing "
            f"at an agent's data directory (e.g. agent_data/meridian).",
            file=err,
        )
        return 1

    # Refuse to write while the agent is actually running. SQLite
    # would honor file locks, but the kestrel daemon caches metadata
    # in-process and we'd see writes the daemon doesn't; safer to
    # require a stop. ``--dry-run`` is read-only so it's allowed.
    # Only block on a LIVE pid — a stale pid file from a crashed
    # daemon should not trap the operator (codex round-1 P2 on PR
    # #1405). ``ProcessManager.is_process_running`` does the kill(0)
    # check that confirms the pid actually exists.
    if not args.dry_run:
        pid_file = _agent_pid_file(data_dir)
        if pid_file.exists():
            pid = _read_pid(pid_file)
            if pid is not None and _is_process_running(pid):
                print(
                    f"Refusing to mutate: agent process is running "
                    f"(pid {pid}, pid file {pid_file}). Stop the host "
                    f"first (kestrel stop), then re-run. Use "
                    f"--dry-run to audit without writing.",
                    file=err,
                )
                return 2
            if pid is not None:
                print(
                    f"Note: ignoring stale pid file {pid_file} (pid "
                    f"{pid} is not running).",
                    file=err,
                )

    report = backfill_all(
        db_path,
        agent_id=getattr(args, "agent_id", None),
        dry_run=bool(args.dry_run),
    )

    mode = "DRY RUN — no changes written" if report.dry_run else "WRITE MODE"
    print(f"=== migrate-encryption ({mode}) ===", file=out)
    print(f"  db:       {report.db_path}", file=out)
    print(f"  agent_id: {report.agent_id or '(unresolved)'}", file=out)
    for tr in (report.conversation, report.files):
        print(f"  [{tr.table}]", file=out)
        print(f"    scanned:           {tr.scanned}", file=out)
        print(f"    plaintext rows:    {tr.plaintext}", file=out)
        print(f"    re-encrypted now:  {tr.encrypted_now}", file=out)
        if tr.skipped_no_fernet:
            print(
                f"    skipped (no key):  {tr.skipped_no_fernet} — "
                "KESTREL_DATA_KEY not set?",
                file=out,
            )
        if tr.errors:
            print(f"    errors: {len(tr.errors)}", file=out)
            for line in tr.errors[:5]:
                print(f"      - {line}", file=out)
            if len(tr.errors) > 5:
                print(f"      … {len(tr.errors) - 5} more", file=out)

    error_count = (
        len(report.conversation.errors) + len(report.files.errors)
    )
    skipped_no_fernet = (
        report.conversation.skipped_no_fernet
        + report.files.skipped_no_fernet
    )

    if report.dry_run:
        print(
            "\nRe-run without --dry-run to apply. The mutating run requires "
            "the agent to be stopped.",
            file=out,
        )
    else:
        total = (
            report.conversation.encrypted_now
            + report.files.encrypted_now
        )
        if total:
            print(
                f"\nDone — encrypted {total} row(s) at rest. Re-run with "
                f"--dry-run to verify zero plaintext rows remain.",
                file=out,
            )
        elif error_count == 0 and skipped_no_fernet == 0:
            print(
                "\nNo plaintext rows found. DB is already fully encrypted.",
                file=out,
            )

    if error_count:
        print(
            f"\nFailed: {error_count} error(s) reported across tables — "
            "see above. Exit code 3.",
            file=err,
        )
        return 3
    if skipped_no_fernet:
        print(
            f"\nFailed: {skipped_no_fernet} row(s) skipped because no "
            "encryption key was configured (KESTREL_DATA_KEY unset?). "
            "Exit code 4.",
            file=err,
        )
        return 4
    return 0


def backfill_all(
    db_path: str | Path,
    *,
    agent_id: Optional[str] = None,
    dry_run: bool = False,
) -> BackfillReport:
    """Run both backfills against a single agent's kestrel_prime.db.

    When ``agent_id`` is None, resolves from the DB's
    ``graph_nodes`` table. The migration is a no-op for
    ``conversation_history`` if the agent_id can't be resolved (the
    runtime write path is scoped by agent_id; backfilling without
    that scope would corrupt cross-agent isolation on multi-agent
    DBs). Files always backfill globally — they have no per-agent
    scope to violate.
    """
    resolved_id = agent_id or discover_agent_id(db_path)
    if resolved_id is None:
        conversation_report = TableReport(
            table="conversation_history",
            errors=[
                "agent_id could not be resolved from graph_nodes; pass "
                "--agent-id explicitly or run inception first"
            ],
        )
    else:
        conversation_report = backfill_conversation_history(
            db_path, agent_id=resolved_id, dry_run=dry_run,
        )

    files_report = backfill_files(db_path, dry_run=dry_run)

    return BackfillReport(
        db_path=str(db_path),
        agent_id=resolved_id,
        dry_run=dry_run,
        conversation=conversation_report,
        files=files_report,
    )
