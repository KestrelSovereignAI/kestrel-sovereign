"""Canonical audit timestamp handling.

Audit entries are aggregated across sources — ``security_audit_log`` (written
via SQLite ``CURRENT_TIMESTAMP`` → ``YYYY-MM-DD HH:MM:SS``, UTC, no offset) and
``destructive_audit_log`` (written via ``datetime.now(timezone.utc).isoformat()``
→ ISO-8601 with ``T`` and offset). Lexically comparing the two formats sorts
incorrectly (space ``0x20`` < ``T`` ``0x54``), which corrupts anchor boundary
computation and verify ranges (F092).

Everything that writes or compares an audit timestamp routes through here so
there is a single canonical form: UTC ISO-8601.
"""

from datetime import datetime, timezone

# Formats emitted by SQLite CURRENT_TIMESTAMP (no fractional seconds by default,
# but tolerate a fractional variant defensively).
_SQLITE_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")

# SQL fragment reused by the one-time legacy-row migration: rewrite a bare
# ``YYYY-MM-DD HH:MM:SS`` (exactly 19 chars, space separator, no offset) into
# canonical ``YYYY-MM-DDTHH:MM:SS.000000+00:00`` so it matches isoformat() output
# and sorts correctly against ISO rows. Idempotent — only touches legacy rows.
LEGACY_TS_MIGRATION_SUFFIX = "'.000000+00:00'"


def utc_now_iso() -> str:
    """The canonical 'now' stamp for a freshly written audit row."""
    return datetime.now(timezone.utc).isoformat()


def normalize_audit_timestamp(value) -> str:
    """Coerce any audit timestamp to canonical UTC ISO-8601.

    Accepts SQLite ``CURRENT_TIMESTAMP`` output, ISO-8601 (with or without a
    ``Z``/offset), and datetime objects. Returns ``""`` for empty input and the
    original string unchanged if it can't be parsed (never silently drops data).
    """
    if value in (None, ""):
        return ""
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return ""
        dt = None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            for fmt in _SQLITE_FORMATS:
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            return s
    if dt.tzinfo is None:
        # CURRENT_TIMESTAMP is UTC; a naive ISO stamp is treated as UTC too.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()
