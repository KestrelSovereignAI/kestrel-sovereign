"""What may be stored in ``conversation_history.created_at`` (#3009).

SQLite has no datetime type. ``TIMESTAMP`` gets NUMERIC affinity, an ISO string
cannot be losslessly converted to a number, so it is stored as **TEXT** —
whatever text the writer supplied. A date in SQLite is a convention, and until
this module nothing enforced the convention: the column was nullable, its
``DEFAULT CURRENT_TIMESTAMP`` is stripped on the way to SQLite (see
``normalize_schema``), and one writer copied foreign text in verbatim.

The readers paid for that. Each carried a fallback for a value the column let
through — a regex gating the parser to what ``julianday`` can order, a
projection fold that escalates an undatable row to a full transcript pass, two
separate undatable-row substitutions that had to be reconciled with each other.
None of those describe anything this system writes. They describe what the
column failed to refuse.

The rule, authored once
=======================

    ``created_at`` is never NULL, and on SQLite it is spelled
    ``YYYY-MM-DD HH:MM:SS`` in UTC — the spelling ``CURRENT_TIMESTAMP`` and
    ``datetime('now')`` already produce.

Three renderings have to agree, the same shape
:mod:`~kestrel_sovereign.storage.session_id_column` uses for session identity:
Python (the writers), SQLite (the CHECK, and the migration's bulk pass) and
PostgreSQL (where the column is a real ``timestamp`` and the only thing left to
forbid is NULL). Every spelling below is computed from
:data:`CANONICAL_FORMAT`; there is nowhere else to change it.

Why the second is the last field
--------------------------------

Because it is what every existing writer already produces, and because a
fixed-width value makes **lexicographic order the chronological order** — which
is what lets ``julianday`` come out of the ordering, the filtering and the index
once nothing non-canonical remains. Fractional seconds are truncated by the
migration. No writer in this codebase has ever produced them; ties inside a
second are broken by ``id``, which the read paths already do because
``CURRENT_TIMESTAMP`` is second-granularity too.

Two parsers, deliberately
-------------------------

:func:`parse_stored_timestamp` accepts more than
:func:`~kestrel_sovereign.storage.session_grouping.coerce_session_timestamp`
does, and that is not the drift this ticket exists to end. They answer
different questions:

* the reader asks *can this value be ordered*, and is narrowed to exactly what
  ``julianday`` can express, because a value it dated but SQL could not would
  sort at the far end of the canonical order and fall out of a ``LIMIT``;
* this module asks *can this value be repaired*, and a value Python can date
  unambiguously should become canonical rather than be thrown to
  :func:`derived_stamp`, even when SQLite cannot read it (``2026-01-02Z``, a
  lowercase ``t``, the basic form ``20260102T030405``).

After the migration the distinction stops mattering — everything stored is
inside both domains — and the reader's gate is deleted, leaving one parser.

Where the two disagree, order decides
-------------------------------------

SQLite's ``strftime`` silently *normalizes* an impossible day-of-month
(``2023-04-31`` becomes May 1) where Python refuses it outright. The migration
runs its SQL pass first, so such a value is normalized by the engine that can
read it and never reaches the Python pass. Stated here because it is the one
place the two renderings produce different answers for the same input, and the
resolution is an ordering rather than an agreement.

A row that cannot be repaired
-----------------------------

Some value may be readable by neither. It cannot keep the column's promise, and
NOT NULL cannot hold "unknown", so the fact has to live somewhere that can:
:data:`UNDATED_TABLE` records the original text beside the stamp that replaced
it. That stamp is not invented — :func:`derived_stamp` takes the row's nearest
readable neighbour, which is what every reader already computed on the fly for
such a row. The migration makes that derivation durable and says so, instead of
recomputing it invisibly on every read forever.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Sequence, Tuple

# ── The rule, authored once ──────────────────────────────────────────────
#
# Every spelling in this module is COMPUTED from this line.
CANONICAL_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Name of the CHECK that carries the rule. Also the migration's marker: the
#: constraint's presence is what says the repair has run, so there is no second
#: ledger that can disagree with the schema (``migrate_columns_once``'s rule).
CONSTRAINT_NAME = "conversation_history_created_at_canonical"

#: Where an unrepairable row's original text is kept.
UNDATED_TABLE = "conversation_history_undated"

#: The stamp for a row with no readable neighbour anywhere in its history.
#: Any instant would do; what matters is that it is the SAME instant every
#: time, so two derivations of one transcript agree. Shared with
#: ``session_grouping.UNDATABLE_ROW_FALLBACK``, which is the reader's name for
#: the identical decision.
UNDATABLE_EPOCH = datetime(1970, 1, 1)

#: Timestamp formats ``fromisoformat`` does not take. SQLite's
#: ``CURRENT_TIMESTAMP`` space form is one of them on older Pythons.
_LEGACY_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


def parse_stored_timestamp(value: Any) -> Optional[datetime]:
    """Parse a stored timestamp to aware UTC, or ``None`` if it has no date.

    The widest unambiguous reading of a stored value — ISO-8601 in any spelling
    Python accepts, plus the SQLite space form. ``None`` means the value cannot
    be dated at all, so a caller makes an explicit decision rather than silently
    mis-sorting a raw string.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        parsed = None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in _LEGACY_FORMATS:
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def canonical_created_at(value: Any) -> Optional[str]:
    """The canonical spelling of ``value``, or ``None`` if it has none.

    Always the string form, on both backends: it is what SQLite stores, and
    PostgreSQL casts it to ``timestamp`` on the way in. Callers binding to a
    PostgreSQL parameter want :func:`created_at_bind` instead.
    """
    parsed = parse_stored_timestamp(value)
    if parsed is None:
        return None
    return parsed.replace(tzinfo=None).strftime(CANONICAL_FORMAT)


def created_at_bind(backend_type: str, value: Any) -> Any:
    """The value to bind for ``created_at``, or ``None`` if it has no date.

    PostgreSQL's ``timestamp`` column takes a naive-UTC ``datetime`` through
    asyncpg; SQLite takes the canonical text. Both are the same instant in the
    same spelling — this only chooses which type carries it.
    """
    parsed = parse_stored_timestamp(value)
    if parsed is None:
        return None
    naive_utc = parsed.replace(tzinfo=None)
    if backend_type == "postgres":
        return naive_utc
    return naive_utc.strftime(CANONICAL_FORMAT)


def derived_stamp(
    predecessor: Optional[str], successor: Optional[str]
) -> Tuple[str, str]:
    """The stamp an unrepairable row takes, and the name of where it came from.

    Its nearest readable predecessor, else its nearest readable successor, else
    the epoch. The predecessor comes first because that is what the readers
    already do — an undatable row joins the session of the row before it — so
    materializing this changes no grouping that was not already being computed.
    The successor is second because it is still evidence from the transcript
    (a message cannot have been written after the one that follows it), and it
    is a far tighter bound than 1970 for a run of undatable rows at the very
    start of a history.

    Callers find the neighbours however their access path allows — a list scan
    during a restore, an indexed lookup during the migration — and the rule for
    choosing between them lives only here.
    """
    if predecessor is not None:
        return predecessor, "predecessor"
    if successor is not None:
        return successor, "successor"
    return UNDATABLE_EPOCH.strftime(CANONICAL_FORMAT), "epoch"


def fill_undatable(values: Sequence[Any]) -> list:
    """Canonicalize an ordered run of stored values, deriving what it must.

    Returns one ``(stamp, origin)`` per input, where ``origin`` is ``"stored"``
    for a value that carried its own readable date and otherwise names the
    neighbour :func:`derived_stamp` fell back to. For callers holding the whole
    ordered list — a restore reading a backup — where the neighbours are found
    by looking along it.
    """
    stamps = [canonical_created_at(value) for value in values]
    filled = []
    for index, stamp in enumerate(stamps):
        if stamp is not None:
            filled.append((stamp, "stored"))
            continue
        predecessor = next(
            (s for s in reversed(stamps[:index]) if s is not None), None
        )
        successor = next((s for s in stamps[index + 1:] if s is not None), None)
        filled.append(derived_stamp(predecessor, successor))
    return filled


# ── The same rule, rendered in SQL ───────────────────────────────────────

def canonical_sql(backend_type: str, expression: str) -> str:
    """Render ``expression`` in the canonical spelling, in SQL.

    NULL where the engine cannot read the value, on both backends: SQLite's
    ``strftime`` returns NULL for text it cannot parse, and on PostgreSQL the
    column is already a ``timestamp``, so the only unreadable value is NULL
    itself and the expression is the column.
    """
    if backend_type == "postgres":
        return expression
    return f"strftime('{CANONICAL_FORMAT}', {expression})"


def created_at_check(backend_type: str) -> str:
    """The CHECK expression carrying the whole rule for this backend.

    One door, not two. A plain ``NOT NULL`` cannot express the spelling half of
    the rule on SQLite, so it would only ever be a partial second enforcement
    point for the same invariant — and two doors on one rule is how the
    a2a-route ``CHECK`` and its column definition came apart in #2804. The
    NULL-ness is therefore stated inside the CHECK on both backends.

    ``IS`` rather than ``=`` because SQLite's ``=`` yields NULL against an
    unreadable value's NULL rendering, and a CHECK whose expression is NULL
    **passes**: written with ``=`` this constraint accepts ``'not a date'``.
    Measured on sqlite 3.50.4, and the reason the first version of it was
    decorative.
    """
    if backend_type == "postgres":
        return "created_at IS NOT NULL"
    return (
        "created_at IS NOT NULL AND created_at IS "
        + canonical_sql("sqlite", "created_at")
    )


def noncanonical_predicate(backend_type: str) -> str:
    """Rows whose ``created_at`` breaks the rule — the CHECK, negated.

    Derived from :func:`created_at_check` rather than spelled again, so the
    migration can never look for a different set of rows than the constraint
    will refuse.
    """
    return f"NOT ({created_at_check(backend_type)})"


def repairable_bulk_update(backend_type: str) -> str:
    """Re-spell every row the ENGINE can read, in one statement.

    The cheap pass, and on SQLite the only one that normally does anything: a
    ``T`` separator, a trailing ``Z``, an offset (converted to UTC), a bare
    date (midnight), fractional seconds (truncated). Rows the engine cannot
    read are left for the caller's Python pass — ``strftime`` returns NULL for
    them and this deliberately does not write that NULL back.

    Empty on PostgreSQL, where a ``timestamp`` column has no spelling to fix.
    """
    if backend_type == "postgres":
        return ""
    canonical = canonical_sql(backend_type, "created_at")
    return (
        "UPDATE conversation_history SET created_at = "
        + canonical
        + " WHERE "
        + canonical
        + " IS NOT NULL AND created_at IS NOT "
        + canonical
    )


def conversation_history_ddl(backend_type: str) -> str:
    """The canonical ``CREATE TABLE`` for ``conversation_history``.

    Written with the table name as ``{table}`` because
    ``AsyncDatabase.ensure_check_constraint`` uses the same template twice: to
    create the table fresh and to rebuild it when a legacy table has to gain
    the constraint. A second, hand-copied spelling for the rebuild is exactly
    the drift #2804 was.

    Declared here rather than in ``CORE_SCHEMA`` for two reasons. The CHECK is
    genuinely per-backend — on PostgreSQL the type does the spelling half of
    the work — and ``normalize_schema`` strips ``DEFAULT CURRENT_TIMESTAMP`` on
    its way to SQLite, which is why the live column has no default at all
    (see #3048). Both are answered by rendering the statement for the backend
    instead of translating one text into it.

    ``embedding_vec`` and ``embedding_profile_id`` are deliberately absent:
    they are added by later, backend-shaped migrations (a pgvector column on
    PostgreSQL, a BLOB on SQLite) and are not this template's to declare. The
    rebuild carries columns it does not know about across rather than dropping
    them.
    """
    id_column = (
        "id SERIAL PRIMARY KEY" if backend_type == "postgres"
        else "id INTEGER PRIMARY KEY AUTOINCREMENT"
    )
    return f"""CREATE TABLE {{table}} (
    {id_column},
    agent_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    rendered_content TEXT DEFAULT NULL,
    model TEXT DEFAULT NULL,
    provider TEXT DEFAULT NULL,
    metadata TEXT,
    session_id TEXT DEFAULT NULL,
    lexical_index_id TEXT DEFAULT NULL,
    lexical_index_version TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP DEFAULT NULL,
    archived_at TIMESTAMP DEFAULT NULL,
    CONSTRAINT {CONSTRAINT_NAME} CHECK ({created_at_check(backend_type)})
)"""


UNDATED_DDL = f"""CREATE TABLE IF NOT EXISTS {UNDATED_TABLE} (
    message_id INTEGER PRIMARY KEY,
    agent_id TEXT NOT NULL,
    original_created_at TEXT,
    derived_created_at TIMESTAMP NOT NULL,
    derived_from TEXT NOT NULL,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)"""
