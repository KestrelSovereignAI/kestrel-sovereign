"""#3009: ``conversation_history.created_at`` stops promising nothing.

The column was nullable and, on SQLite, held whatever text a writer supplied —
``TIMESTAMP`` there is NUMERIC affinity, so an ISO string is stored as TEXT.
Every reader carried a fallback for a value only that absence let through. This
file is about closing it, and it has three subjects.

**The rule has to mean the same thing in three languages.** Python spells it,
SQLite enforces it in a CHECK, PostgreSQL enforces the half of it its type does
not already cover. A corpus of spellings is run through each rendering and they
are required to agree, rather than trusted to.

**A database that already holds a violation has to be repairable.** The
constraint cannot be added while a row breaks it, so the migration re-spells
what it can — by the engine first, then by a parser that reads more — and dates
what it cannot from the row's own neighbours, recording the original text so a
repair is never a silent overwrite.

**The retrofit rebuilds the table, and a rebuild is where data goes missing.**
On SQLite there is no ``ADD CONSTRAINT``: the table is copied, dropped and
renamed. ``DROP TABLE`` takes the indexes and the #2959 change triggers with
it, and a copy that knows only the canonical template would leave behind every
column a later ``ALTER`` added — ``embedding_vec`` among them. Each of those is
pinned below, because none of them fails loudly.
"""

from __future__ import annotations

import sqlite3

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.session_grouping import (
    coerce_session_timestamp,
)
from kestrel_sovereign.storage.conversation_created_at import (
    CANONICAL_FORMAT,
    CONSTRAINT_NAME,
    UNDATED_TABLE,
    canonical_created_at,
    created_at_check,
    derived_stamp,
    fill_undatable,
)

AGENT = "did:test:created-at"
OTHER = "did:test:created-at-other"

#: The pre-#3009 table: nullable, unconstrained, and missing every column later
#: migrations add. What an upgrading host actually has on disk — taken from a
#: live agent database rather than imagined.
LEGACY_DDL = """CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP,
    deleted_at TIMESTAMP DEFAULT NULL
)"""

#: Spellings and what each must become. The offsets are the interesting rows:
#: an offset must be APPLIED, not truncated, or the row is dated an hour wrong
#: and still passes a "looks canonical" check.
REPAIRABLE = [
    ("2026-01-02 03:04:05", "2026-01-02 03:04:05"),   # already canonical
    ("2026-01-02T03:04:05", "2026-01-02 03:04:05"),   # T separator
    ("2026-01-02T03:04:05Z", "2026-01-02 03:04:05"),  # explicit UTC
    ("2026-01-02 04:04:05+01:00", "2026-01-02 03:04:05"),   # offset applied
    ("2026-01-02T02:04:05-01:00", "2026-01-02 03:04:05"),   # the other way
    ("2026-01-02", "2026-01-02 00:00:00"),            # bare date -> midnight
    ("2026-01-02 03:04:05.750", "2026-01-02 03:04:05"),     # fraction dropped
    ("2026-01-02Z", "2026-01-02 00:00:00"),           # julianday NULL; Python reads it
    ("20260102T030405", "2026-01-02 03:04:05"),       # basic form, Python only
]

#: Values no reading of any kind turns into an instant.
UNREPAIRABLE = ["not a date", "", "   ", "2026-13-45 99:99:99", None]


async def _legacy_database(tmp_path, name, rows, ddl=LEGACY_DDL):
    """A pre-#3009 SQLite file holding ``rows``, then opened and migrated.

    Seeded with the raw ``sqlite3`` driver against the legacy DDL, because the
    whole point is a value the current schema refuses: writing it through
    ``AsyncDatabase`` after ``_init_schema`` has run would be writing it through
    the constraint that exists to stop it. This is the state on disk of a host
    upgrading INTO this change, not a state manufactured to suit the test.
    """
    path = str(tmp_path / name)
    raw = sqlite3.connect(path)
    raw.execute(ddl)
    raw.executemany(
        "INSERT INTO conversation_history (agent_id, role, content, created_at) "
        "VALUES (?, ?, 'x', ?)",
        rows,
    )
    raw.commit()
    raw.close()
    return await AsyncDatabase.sqlite(path)


async def _stamps(db, agent_id=AGENT):
    return [
        row[0]
        for row in await db.fetchall(
            "SELECT created_at FROM conversation_history WHERE agent_id = ? "
            "ORDER BY id",
            (agent_id,),
        )
    ]


# ── The rule, in each of its renderings ──────────────────────────────────

def test_the_python_rendering_and_the_engines_agree_on_every_spelling():
    """The parser and the CHECK must draw the same line.

    A value Python canonicalizes into something the CHECK then refuses would
    make a writer's output unwritable; a value the CHECK accepts but Python
    would re-spell would leave two forms of the same instant in the column.
    Asked of the real engine rather than of a second copy of the rule.
    """
    engine = sqlite3.connect(":memory:")
    engine.execute(
        "CREATE TABLE t (created_at TIMESTAMP, "
        f"CHECK ({created_at_check('sqlite')}))"
    )
    for spelling, expected in REPAIRABLE:
        canonical = canonical_created_at(spelling)
        assert canonical == expected, (
            f"{spelling!r} was read as {canonical!r}, not {expected!r}"
        )
        engine.execute("INSERT INTO t (created_at) VALUES (?)", (canonical,))
    stored = [row[0] for row in engine.execute("SELECT created_at FROM t")]
    assert stored == [expected for _, expected in REPAIRABLE], (
        "the CHECK accepted every canonical value but stored something else"
    )


def test_the_check_refuses_a_value_that_looks_nothing_like_a_date():
    """``IS``, not ``=`` — the difference between a guard and a decoration.

    ``strftime`` returns NULL for text it cannot read, and a CHECK whose
    expression evaluates to NULL **passes** on SQLite. Written with ``=`` this
    constraint therefore accepts ``'not a date'`` while refusing
    ``'2026-01-02T03:04:05'``: it rejects the near-misses and waves through the
    garbage, which is the worst of both. Measured on sqlite 3.50.4.
    """
    engine = sqlite3.connect(":memory:")
    engine.execute(
        "CREATE TABLE t (created_at TIMESTAMP, "
        f"CHECK ({created_at_check('sqlite')}))"
    )
    for value in [spelling for spelling, _ in REPAIRABLE[1:]] + UNREPAIRABLE:
        with pytest.raises(sqlite3.IntegrityError):
            engine.execute("INSERT INTO t (created_at) VALUES (?)", (value,))


def test_the_check_admits_what_the_column_default_writes():
    """A constraint the table's own DEFAULT cannot satisfy is a broken table.

    Every insert in this codebase that omits ``created_at`` relies on it, and a
    CHECK that refused ``CURRENT_TIMESTAMP`` would turn those into hard errors
    at the moment the constraint landed.
    """
    engine = sqlite3.connect(":memory:")
    engine.execute(
        "CREATE TABLE t (created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        f"CHECK ({created_at_check('sqlite')}))"
    )
    engine.execute("INSERT INTO t DEFAULT VALUES")
    engine.execute("INSERT INTO t (created_at) VALUES (datetime('now'))")


def test_the_canonical_spelling_sorts_as_the_clock_does():
    """The reason the rule is a fixed-width spelling and not merely 'a date'.

    Lexicographic order over canonical values IS chronological order, which is
    what allows ``julianday`` out of the ordering, the filtering and the index.
    If this ever stops holding, that simplification stops being available.
    """
    engine = sqlite3.connect(":memory:")
    engine.execute("CREATE TABLE t (v TEXT)")
    values = [
        "2026-01-02 03:04:05", "1970-01-01 00:00:00", "2026-01-02 03:04:06",
        "2025-12-31 23:59:59", "0001-01-01 00:00:00", "9999-12-31 23:59:59",
    ]
    engine.executemany("INSERT INTO t VALUES (?)", [(v,) for v in values])
    text_order = [r[0] for r in engine.execute("SELECT v FROM t ORDER BY v")]
    clock_order = [
        r[0] for r in engine.execute("SELECT v FROM t ORDER BY julianday(v)")
    ]
    assert text_order == clock_order


def test_an_undatable_row_prefers_the_row_before_it():
    """The reader's rule, made durable rather than re-invented.

    Every reader dates such a row from its predecessor. The migration must make
    the same choice or it would move rows between sessions on the way past.
    """
    assert derived_stamp("2026-01-01 00:00:00", "2026-06-01 00:00:00") == (
        "2026-01-01 00:00:00", "predecessor",
    )
    assert derived_stamp(None, "2026-06-01 00:00:00") == (
        "2026-06-01 00:00:00", "successor",
    )
    assert derived_stamp(None, None) == ("1970-01-01 00:00:00", "epoch")


def test_a_leading_run_of_undatable_rows_borrows_forward():
    """A restore reads its rows as a list, so it can look both ways.

    Falling to 1970 for a history that simply opens with an unreadable row
    would put those messages 56 years before the rest of the transcript and
    manufacture a session gap that never happened.
    """
    filled = fill_undatable(
        ["junk", None, "2026-01-02T03:04:05", "junk", "2026-01-03 00:00:00"]
    )
    assert filled == [
        ("2026-01-02 03:04:05", "successor"),
        ("2026-01-02 03:04:05", "successor"),
        ("2026-01-02 03:04:05", "stored"),
        ("2026-01-02 03:04:05", "predecessor"),
        ("2026-01-03 00:00:00", "stored"),
    ]


# ── The migration, over a database that already holds violations ─────────

@pytest.mark.asyncio
async def test_every_spelling_a_legacy_database_holds_is_re_spelled(tmp_path):
    """The migration's whole job, asked one spelling at a time.

    Both passes are exercised here on purpose: SQLite reads most of these, and
    ``2026-01-02Z`` and the basic form are the two only Python reads. A version
    that ran just the SQL pass would leave those two undated and derive them
    from a neighbour — which still produces a plausible-looking timestamp, so
    the failure would be invisible without pinning the exact value.
    """
    db = await _legacy_database(
        tmp_path, "spellings.db",
        [(AGENT, "user", spelling) for spelling, _ in REPAIRABLE],
    )
    try:
        assert await _stamps(db) == [expected for _, expected in REPAIRABLE]
        assert not await db.fetchall(f"SELECT * FROM {UNDATED_TABLE}"), (
            "a row the migration could re-spell exactly was dated from a "
            "neighbour instead"
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_row_nothing_can_date_takes_its_neighbour_and_says_so(tmp_path):
    """The only lossy case, and the record that keeps it from being silent.

    The database is being rewritten in place, so the original text has nowhere
    else to survive. It goes into ``conversation_history_undated`` beside the
    stamp that replaced it and the name of where that stamp came from.
    """
    db = await _legacy_database(
        tmp_path, "undatable.db",
        [
            (AGENT, "user", "leading junk"),
            (AGENT, "user", "2026-01-02 03:04:05"),
            (AGENT, "assistant", "not a date"),
            (AGENT, "user", None),
        ],
    )
    try:
        assert await _stamps(db) == [
            "2026-01-02 03:04:05",   # borrowed forward from its successor
            "2026-01-02 03:04:05",
            "2026-01-02 03:04:05",   # inherited from its predecessor
            "2026-01-02 03:04:05",
        ]
        recorded = await db.fetchall(
            f"SELECT message_id, agent_id, original_created_at, "
            f"derived_created_at, derived_from FROM {UNDATED_TABLE} "
            "ORDER BY message_id"
        )
        assert recorded == [
            (1, AGENT, "leading junk", "2026-01-02 03:04:05", "successor"),
            (3, AGENT, "not a date", "2026-01-02 03:04:05", "predecessor"),
            (4, AGENT, None, "2026-01-02 03:04:05", "predecessor"),
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_stamp_is_never_borrowed_across_an_agent_boundary(tmp_path):
    """One agent's message may not be dated from another agent's history.

    Both agents' rows live in one table ordered by one id sequence, so the
    nearest row by id to an undatable row is frequently somebody else's. Taking
    it would date a message from a transcript it has nothing to do with.
    """
    db = await _legacy_database(
        tmp_path, "tenants.db",
        [
            (OTHER, "user", "2020-01-01 00:00:00"),
            (AGENT, "user", "junk"),
            (OTHER, "user", "2020-01-02 00:00:00"),
            (AGENT, "user", "2026-05-05 05:05:05"),
        ],
    )
    try:
        assert await _stamps(db, AGENT) == [
            "2026-05-05 05:05:05", "2026-05-05 05:05:05",
        ], "the undatable row took the other agent's stamp"
        assert await _stamps(db, OTHER) == [
            "2020-01-01 00:00:00", "2020-01-02 00:00:00",
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_the_engine_pass_never_writes_its_own_failure_back(tmp_path):
    """``strftime`` returns NULL for what it cannot read.

    The bulk UPDATE writes that expression into the column, so without a guard
    it would overwrite every value SQLite cannot read with NULL — turning a
    recoverable spelling into an unrecoverable absence, one statement before
    the pass that could have recovered it.
    """
    db = await _legacy_database(
        tmp_path, "nulls.db",
        [(AGENT, "user", "2026-01-02Z"), (AGENT, "user", "2026-01-03 00:00:00")],
    )
    try:
        assert await _stamps(db) == [
            "2026-01-02 00:00:00", "2026-01-03 00:00:00",
        ]
        assert not await db.fetchall(f"SELECT * FROM {UNDATED_TABLE}")
    finally:
        await db.close()


# ── The retrofit, and what a rebuild must not lose ───────────────────────

@pytest.mark.asyncio
async def test_the_rebuild_keeps_columns_the_template_never_heard_of(tmp_path):
    """``embedding_vec`` is added by an ALTER, not by the canonical DDL.

    A rebuild that copied only the columns both shapes share would drop it and
    every embedding in it — silently, because the column is re-added empty by
    the next boot's migration and nothing reports a row count that changed
    only in one field.
    """
    legacy = LEGACY_DDL
    db = None
    path = str(tmp_path / "carry.db")
    raw = sqlite3.connect(path)
    raw.execute(legacy)
    raw.execute("ALTER TABLE conversation_history ADD COLUMN embedding_vec BLOB")
    raw.execute(
        "ALTER TABLE conversation_history ADD COLUMN a_column_nobody_declares "
        "TEXT DEFAULT 'kept'"
    )
    raw.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, created_at, embedding_vec) "
        "VALUES (?, 'user', 'x', '2026-01-02T03:04:05', ?)",
        (AGENT, b"an embedding"),
    )
    raw.commit()
    raw.close()

    db = await AsyncDatabase.sqlite(path)
    try:
        row = await db.fetchone(
            "SELECT created_at, embedding_vec, a_column_nobody_declares "
            "FROM conversation_history WHERE agent_id = ?",
            (AGENT,),
        )
        assert row == ("2026-01-02 03:04:05", b"an embedding", "kept")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_the_rebuild_leaves_the_projection_unable_to_claim_currency(
    tmp_path,
):
    """``DROP TABLE`` takes the #2959 change triggers with it.

    A projection whose triggers vanished stops being told about writes, and a
    watermark that still says "current" would go on answering from rows the
    rebuild had just re-dated underneath it. The ordering in ``_init_schema``
    is what prevents that: the retrofit runs first, the projection's schema
    pass then finds its trigger family missing, reinstalls it, and rotates the
    generation — which is how a projection is told it can no longer be trusted.

    Reproduces a host that upgraded through #2959 and is now upgrading through
    #3009, which is every existing host, not an edge case.
    """
    path = str(tmp_path / "projection.db")
    db = await AsyncDatabase.sqlite(path)
    await db.execute(
        "INSERT INTO conversation_history (agent_id, role, content, created_at) "
        "VALUES (?, 'user', 'x', '2026-01-02 03:04:05')",
        (AGENT,),
    )
    before = await db.fetchone(
        "SELECT generation FROM conversation_history_changes "
        "WHERE agent_id = ? AND slot = 0",
        (AGENT,),
    )
    assert before and before[0], "the ledger never recorded a generation"
    await db.close()

    # Put the table back the way #2959 left it: same rows, no constraint.
    raw = sqlite3.connect(path)
    raw.execute("ALTER TABLE conversation_history RENAME TO ch_old")
    raw.execute(LEGACY_DDL)
    raw.execute(
        "INSERT INTO conversation_history "
        "(id, agent_id, role, content, metadata, created_at, deleted_at) "
        "SELECT id, agent_id, role, content, metadata, created_at, deleted_at "
        "FROM ch_old"
    )
    raw.execute("DROP TABLE ch_old")
    raw.commit()
    raw.close()

    db = await AsyncDatabase.sqlite(path)
    try:
        ddl = await db.fetchone(
            "SELECT sql FROM sqlite_master WHERE name = 'conversation_history'"
        )
        assert CONSTRAINT_NAME in ddl[0], "the retrofit did not run"
        triggers = await db.fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND tbl_name = 'conversation_history'"
        )
        assert len(triggers) == 3, (
            f"the rebuild left {len(triggers)} change trigger(s); a projection "
            "with no triggers is never told it went stale"
        )
        after = await db.fetchone(
            "SELECT generation FROM conversation_history_changes "
            "WHERE agent_id = ? AND slot = 0",
            (AGENT,),
        )
        assert after[0] != before[0], (
            "the generation survived a rebuild of the table it accounts for, "
            "so a projection built before it still reports itself current"
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_the_rebuild_puts_back_the_indexes_it_dropped(tmp_path):
    """A missing index degrades every conversation query without failing one.

    ``DROP TABLE`` removes them, and the ordering index in particular is what
    keeps the conversation list a bounded traversal rather than a scan of the
    agent's whole history.
    """
    db = await _legacy_database(
        tmp_path, "indexes.db", [(AGENT, "user", "2026-01-02T03:04:05")]
    )
    try:
        indexes = {
            row[0]
            for row in await db.fetchall(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'conversation_history'"
            )
        }
        for expected in (
            "idx_conversation_agent_canonical",
            "idx_conversation_agent_created_at",
            "idx_conversation_agent_session",
            "idx_conversation_agent_live_row_id",
        ):
            assert expected in indexes, f"the rebuild lost {expected}"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_the_constraint_is_the_marker_so_a_second_boot_does_nothing(
    tmp_path,
):
    """No separate ledger that could disagree with the schema.

    A migration whose "already done" lives anywhere but the schema can be told
    it has run against a database that does not have it. The constraint's own
    presence is the answer here, so this asks the cheap question: a second boot
    must leave the table, the triggers and the ledger exactly as it found them.
    """
    path = str(tmp_path / "twice.db")
    db = await _legacy_database(
        tmp_path, "twice.db", [(AGENT, "user", "2026-01-02T03:04:05")]
    )

    async def fingerprint(handle):
        objects = await handle.fetchall(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE tbl_name = 'conversation_history' ORDER BY type, name"
        )
        rows = await handle.fetchall(
            "SELECT id, created_at FROM conversation_history ORDER BY id"
        )
        return objects, rows

    first = await fingerprint(db)
    await db.close()

    db = await AsyncDatabase.sqlite(path)
    try:
        assert await fingerprint(db) == first, (
            "the second boot rebuilt a table that already carried the "
            "constraint, so every boot pays for a full copy of history"
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_fresh_database_is_born_with_the_constraint(tmp_path):
    """The template that creates the table is the one the rebuild uses.

    Two spellings of the same shape is how #2804 happened: a table created
    fresh carried its CHECK and one that gained the column by ALTER did not,
    permanently and undetectably.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "fresh.db"))
    try:
        ddl = await db.fetchone(
            "SELECT sql FROM sqlite_master WHERE name = 'conversation_history'"
        )
        assert CONSTRAINT_NAME in ddl[0]
        assert CANONICAL_FORMAT in ddl[0]
        assert "DEFAULT CURRENT_TIMESTAMP" in ddl[0], (
            "the column reached SQLite without its default, which is the hole "
            "#3048 describes and this DDL exists to route around"
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_the_day_the_two_readers_disagree_about_is_settled_by_the_engine(
    tmp_path,
):
    """``2023-02-29`` is read by SQLite and refused by Python.

    ``julianday`` and ``strftime`` NORMALIZE an impossible day-of-month rather
    than rejecting it — February 29th of a non-leap year becomes March 1st —
    while ``datetime.fromisoformat`` refuses it outright. That is the one input
    on which the two renderings of the rule genuinely disagree, and the
    migration resolves it by ORDER rather than by agreement: the engine's pass
    runs first, so the value is settled by the reading that can make sense of
    it and never reaches the parser that cannot.

    It matters which way round that falls. Derived from a neighbour, the row
    would take an unrelated session's stamp; normalized, it keeps the instant
    it was always closest to. This pins the outcome rather than the reasoning,
    because reversing the two passes produces a plausible timestamp either way.
    """
    engine = sqlite3.connect(":memory:")
    assert engine.execute(
        "SELECT julianday('2023-02-29T12:00:00')"
    ).fetchone()[0] is not None, "the premise died: SQLite now refuses this"
    assert canonical_created_at("2023-02-29T12:00:00") is None, (
        "the premise died: Python now reads this, so nothing diverges"
    )

    db = await _legacy_database(
        tmp_path, "impossible.db",
        [
            (AGENT, "user", "2023-02-28 09:00:00"),
            (AGENT, "user", "2023-02-29T12:00:00"),
        ],
    )
    try:
        assert await _stamps(db) == [
            "2023-02-28 09:00:00", "2023-03-01 12:00:00",
        ]
        assert not await db.fetchall(f"SELECT * FROM {UNDATED_TABLE}"), (
            "a value SQLite can read was treated as undatable and given its "
            "neighbour's stamp"
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_after_the_migration_both_readers_can_date_every_row(tmp_path):
    """The claim the reader-side fallbacks exist for, tested at the source.

    Ordering compares ``created_at`` through ``julianday``; the projection's
    fold parses the same column with ``coerce_session_timestamp``. They accept
    different sets of strings, in both directions, and every fallback in the
    read path is there to survive a row that falls in the gap. This asks the
    migrated database whether such a row can still exist.

    It cannot, and that is what makes the fallbacks removable rather than
    merely unused: no writer produces one, and the CHECK refuses one.
    """
    db = await _legacy_database(
        tmp_path, "domains.db",
        [
            (AGENT, "user", spelling)
            for spelling, _ in REPAIRABLE
        ] + [(AGENT, "user", value) for value in UNREPAIRABLE],
    )
    try:
        rows = await db.fetchall(
            "SELECT created_at, julianday(created_at) FROM conversation_history "
            "ORDER BY id"
        )
        assert rows, "nothing was seeded"
        for stored, ordering_key in rows:
            assert ordering_key is not None, (
                f"{stored!r} survived the migration unreadable to the ordering"
            )
            assert coerce_session_timestamp(stored) is not None, (
                f"{stored!r} survived the migration unreadable to the fold"
            )
            assert canonical_created_at(stored) == stored, (
                f"{stored!r} is stored in a spelling the writers do not produce"
            )
    finally:
        await db.close()
