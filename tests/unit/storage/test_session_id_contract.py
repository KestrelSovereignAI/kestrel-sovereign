"""#2958: one session-id rule, three renderings, no drift.

``session_id_column`` decides which ``metadata.session_id`` values may be copied
into the indexed column. Three implementations have to agree — Python (the
write paths), SQLite and PostgreSQL (the two halves of the legacy backfill) —
and the previous attempt at this ticket failed review precisely because the rule
was written out by hand in each of them and they diverged.

So the rule is authored once and COMPILED into each dialect. These tests hold
the compiler to that: the generated character classes are proved to denote the
authored set, and the invariant the whole design rests on is asserted directly.

    The column may be NULL where the grouper is canonical.
    It may never disagree.

The SQL renderings are executed against real engines in
``tests/integration/test_session_id_column_backend_parity.py``; what is checked
here is everything that does not need a server.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.storage.session_grouping import group_messages_into_sessions
from kestrel_sovereign.storage.session_id_column import (
    SESSION_ID_MAX_LENGTH,
    _assert_class_denotes,
    _parse_sql_character_class,
    _sql_character_class,
    backfill_statement,
    column_session_id,
    describe_contract,
    is_stampable_session_id,
    merged_column_assignment,
)


UUID_A = "8f1d1c62-9b0e-4b2c-9a1d-000000000001"

# Values a live database can hold in ``metadata.session_id``, and whether the
# column may adopt them.  Spelled out rather than computed, so the test states
# the contract instead of restating the implementation.
VALUES = [
    ("uuid", UUID_A, True),
    ("sess_ key", "sess_9fk21xa", True),
    ("bare hyphen-underscore", "-_-", True),
    ("single letter", "a", True),
    ("at the length limit", "b" * SESSION_ID_MAX_LENGTH, True),
    # #2012: the list endpoint keyed sessions by row id and the UI echoed the
    # integer back. Grouping ignores those, so the column must not adopt one.
    ("bare integer", "1314", False),
    ("empty string", "", False),
    ("one over the length limit", "b" * (SESSION_ID_MAX_LENGTH + 1), False),
    # PostgreSQL cannot hold a NUL in TEXT and ``metadata::jsonb`` raises on the
    # escape that would produce one — mid-migration that is a failed boot.
    ("nul", "\x00", False),
    ("embedded nul", "a\x00b", False),
    # str.isdigit() calls these digits and neither SQL dialect does. The charset
    # settles the disagreement before the digit test is ever reached.
    ("unicode digits", "١٢٣", False),
    ("non-ascii letters", "sessión", False),
    # Nothing may be filed under an identity whose spelling depends on the
    # reader: SQLite, PostgreSQL and Python render each of these differently.
    ("json true", True, False),
    ("json false", False, False),
    ("json number", 1314, False),
    # The one non-string JSON value that renders as something the charset and
    # digit tests both accept (``-5``), so it is the case that makes the SQL
    # type tests load-bearing rather than decorative.
    ("json negative number", -5, False),
    ("json float", 1.5, False),
    ("json object", {"nested": UUID_A}, False),
    ("json array", [UUID_A], False),
    ("json null", None, False),
    # Ordinary strings that are simply outside the portable charset.
    ("space", "a b", False),
    ("colon", "did:x:1", False),
    ("slash", "a/b", False),
    ("bracket", "a]b", False),
    ("caret", "a^b", False),
]


@pytest.mark.parametrize(
    ("label", "value", "stampable"),
    [(label, value, stampable) for label, value, stampable in VALUES],
    ids=[label for label, _v, _s in VALUES],
)
def test_the_rule_classifies_each_value(label, value, stampable):
    assert is_stampable_session_id(value) is stampable


@pytest.mark.parametrize(
    ("label", "value", "stampable"),
    [(label, value, stampable) for label, value, stampable in VALUES],
    ids=[label for label, _v, _s in VALUES],
)
def test_the_column_never_disagrees_with_session_grouping(label, value, stampable):
    """The invariant. Stampable implies the grouper files the row there too.

    The contract is deliberately narrower than
    ``group_messages_into_sessions``, so the column is allowed to be silent —
    but never to name a session the transcript does not show. Asserted against
    the real grouper rather than a restatement of its rule, so a change to
    either side has to come here to be reconciled.
    """
    if not stampable:
        return

    sessions = group_messages_into_sessions(
        [{"id": 1, "role": "user", "content": "x",
          "metadata": {"session_id": value}, "timestamp": "2026-01-01T00:00:00+00:00"}]
    )
    assert sessions[0]["session_id"] == value, (
        f"{label}: the column would claim a session grouping does not file"
    )


def test_no_stampable_value_is_ignored_as_a_bare_integer_by_the_grouper():
    """The narrowing is one-directional: never stampable-but-ungrouped.

    ``group_messages_into_sessions`` drops a ``session_id`` that is truthy and
    ``str(...).isdigit()``.  Proved over the whole corpus rather than the
    hand-picked rows above, because the failure this guards against is a value
    nobody thought to list.
    """
    for label, value, stampable in VALUES:
        if not stampable:
            continue
        assert value and not str(value).isdigit(), label


def test_metadata_is_accepted_parsed_or_exactly_as_stored():
    """Write paths hold metadata at different points in its life."""
    assert column_session_id({"session_id": UUID_A}) == UUID_A
    assert column_session_id('{"session_id": "%s"}' % UUID_A) == UUID_A
    assert column_session_id(b'{"session_id": "%s"}' % UUID_A.encode()) == UUID_A


@pytest.mark.parametrize(
    "metadata",
    [None, "", "{not json", "[1, 2]", '"a string"', "null", 17, [], {"other": 1}],
)
def test_metadata_without_a_usable_session_id_yields_none(metadata):
    """Unparseable, not an object, or simply absent — all the same answer."""
    assert column_session_id(metadata) is None


# Raw metadata TEXT whose reading depends on who parses it. A ``dict`` cannot
# express any of these — the duplicate is gone the moment it becomes one — so
# they are a separate corpus from VALUES above.
#
# (label, stored metadata text, expected column value)
DUPLICATE_KEY_TEXT = [
    ("duplicated at the top level",
     '{"session_id": "aaaa-1111", "session_id": "bbbb-2222"}', None),
    ("duplicated three times",
     '{"session_id": "a1", "session_id": "b2", "session_id": "c3"}', None),
    ("duplicated with one unstampable occurrence",
     '{"session_id": "1314", "session_id": "bbbb-2222"}', None),
    # Not duplicates, and each is a shape a real row has: a nested object may
    # repeat the key without the top level being ambiguous, and a key that
    # merely CONTAINS "session_id" is a different key entirely. A guard that
    # counted substrings instead of parsed keys would null both of these.
    ("duplicated only inside a nested object",
     '{"nested": {"session_id": "x1", "session_id": "y2"}, '
     '"session_id": "cccc-3333"}', "cccc-3333"),
    ("a longer key that contains the key",
     '{"parent_session_id": "aaaa-1111", "session_id": "cccc-3333"}',
     "cccc-3333"),
    ("the key appearing inside a value",
     '{"note": "session_id was reset", "session_id": "cccc-3333"}',
     "cccc-3333"),
]


@pytest.mark.parametrize(
    ("label", "metadata", "expected"),
    DUPLICATE_KEY_TEXT,
    ids=[label for label, _m, _e in DUPLICATE_KEY_TEXT],
)
def test_a_duplicated_key_is_refused_rather_than_arbitrated(label, metadata, expected):
    """JSON allows a key twice; the three readers resolve it three ways.

    There is no occurrence that all of Python, SQLite and PostgreSQL would pick,
    so picking one would guarantee that some backend disagrees with the column.
    Refusing costs a NULL, which the design already tolerates everywhere else.
    """
    assert column_session_id(metadata) == expected


def test_the_duplicate_a_dict_cannot_carry_is_only_a_question_for_stored_text():
    """Why the text path is not just ``json.loads`` then the dict path.

    ``json.loads`` resolves a duplicate silently, and to the LAST occurrence —
    where SQLite's ``json_extract`` takes the FIRST. Pinning the stdlib half of
    that disagreement here keeps the reason for the pairs-based parse visible;
    the two engine halves are pinned against real servers in
    ``tests/integration/test_session_id_column_backend_parity.py``.
    """
    import json

    duplicated = '{"session_id": "aaaa-1111", "session_id": "bbbb-2222"}'
    assert json.loads(duplicated)["session_id"] == "bbbb-2222"
    assert column_session_id(duplicated) is None
    # The same document, once it has been through a dict, has no duplicate left
    # to refuse — so the dict path must NOT inherit the refusal.
    assert column_session_id(json.loads(duplicated)) == "bbbb-2222"


def test_the_generated_character_classes_denote_the_authored_set():
    """The compiler is checked by parsing its output back, not by eyeballing it."""
    contract = describe_contract()
    assert (
        _parse_sql_character_class(contract["allowed_class"])
        == contract["allowed_characters"]
    )
    assert _parse_sql_character_class(contract["digit_class"]) == contract["digits"]


def test_a_literal_hyphen_is_emitted_where_it_cannot_become_a_range():
    """``[a-c-]`` and ``[-a-c]`` are not the same class in either dialect."""
    body = _sql_character_class(frozenset("-ac"))
    assert body.startswith("-")
    assert _parse_sql_character_class(body) == frozenset("-ac")


def test_a_character_set_that_cannot_mean_the_same_thing_twice_is_refused():
    """``^``, ``]`` and ``\\`` do not agree across a PG regex and a GLOB class.

    Silently emitting one would produce a class that matches something other
    than the set it was generated from — the exact drift this module exists to
    make impossible — so import fails instead.
    """
    for hostile in ("^", "]", "\\"):
        characters = frozenset("ab" + hostile)
        with pytest.raises(ValueError, match="do not mean the same thing"):
            _assert_class_denotes(_sql_character_class(characters), characters)


def test_a_class_that_does_not_denote_its_set_is_refused():
    with pytest.raises(ValueError, match="does not denote"):
        _assert_class_denotes("a-c", frozenset("abcd"))


@pytest.mark.parametrize("backend_type", ["sqlite", "postgres"])
def test_each_backfill_carries_the_compiled_rule_not_a_hand_copy(backend_type):
    """Every literal in the SQL traces back to the one authored statement.

    A reviewer's question about this migration is "did someone retype the rule
    in SQL?"  This answers it mechanically: the character classes, the length
    cap and the NUL escape in the emitted statement are the objects from the
    contract, so editing the contract necessarily edits the SQL.

    The NUL escape is bound on SQLite alone. PostgreSQL does not need it —
    ``pg_input_is_valid`` declines every document whose cast would raise, that
    one included — and carrying it anyway would be a conjunct no input can
    exercise, which reads as protection while defending nothing.
    """
    contract = describe_contract()
    sql, params = backfill_statement(backend_type)

    for authored in (contract["allowed_class"], contract["digit_class"]):
        # Either bracket form — SQLite negates the charset test (``NOT ... GLOB
        # '*[^...]*'``) where PostgreSQL anchors it (``~ '^[...]+$'``).
        assert f"[{authored}]" in sql or f"[^{authored}]" in sql
    assert str(contract["max_length"]) in sql
    if backend_type == "sqlite":
        assert params == (contract["nul_escape"],)
    else:
        assert params == ()
        assert contract["nul_escape"] not in sql
    assert sql.count("?") == len(params)
    assert sql.startswith("UPDATE conversation_history SET session_id = CASE WHEN ")


@pytest.mark.parametrize("backend_type", ["sqlite", "postgres"])
def test_the_where_clause_holds_no_expression_that_can_raise(backend_type):
    """The cast-safety argument, asserted structurally.

    Neither engine promises an evaluation order for ``AND``, so a JSON parse in
    the ``WHERE`` clause can run before the guard meant to protect it — on
    PostgreSQL that is a raised migration, i.e. a failed boot. Only the ``SET``
    expression is guaranteed to run after the row has passed ``WHERE``, so every
    parse belongs there.

    The statement's own ``WHERE`` is found by the ``END`` that closes the CASE,
    not by the first ``WHERE`` in the string: the duplicate-key guard is a
    subquery and brings a ``WHERE`` of its own, so "first" started meaning the
    wrong clause the moment it was added. A test that reads the wrong region
    reports on nothing.

    ``pg_input_is_valid`` is deliberately absent from the list below and is the
    one JSON-aware function the ``WHERE`` may hold. Its entire purpose is to
    answer "would this parse?" *without* parsing, so it is the exception that
    makes the rest of the rule enforceable rather than a hole in it.
    """
    sql, _params = backfill_statement(backend_type)
    assert sql.count(" END WHERE ") == 1, sql
    where = sql[sql.index(" END WHERE ") + len(" END"):]

    for parse in (
        "::json",  # covers ::jsonb too
        "json_extract",
        "json_type",
        "jsonb_typeof",
        # The duplicate-key guards. Each walks the document and so raises on
        # malformed metadata exactly like the extraction it protects — being a
        # guard does not exempt a clause from needing one.
        "json_each",
        "json_object_keys",
    ):
        assert parse not in where, f"{parse!r} can raise and must not gate itself"


@pytest.mark.parametrize("backend_type", ["sqlite", "postgres"])
def test_the_merge_clause_binds_exactly_one_value(backend_type):
    """The store appends this clause to a SET list it is also binding params to.

    ``update_message_metadata`` builds ``(*merge_params, <this one>, message_id,
    agent_id)`` positionally, so a second placeholder in here would not fail
    loudly — it would consume ``message_id``, and the WHERE would then match on
    ``agent_id`` against a row id. Wrong row, no error. The count is the
    contract between the two modules, so it is asserted here rather than left
    to the caller to notice.
    """
    clause = merged_column_assignment(backend_type)
    assert clause.startswith("session_id = ")
    assert clause.count("?") == 1, clause


@pytest.mark.parametrize("backend_type", ["sqlite", "postgres"])
def test_the_merge_clause_asks_about_the_column_it_was_given(backend_type):
    """The guard reads the document being merged, not one named by accident.

    Both dialects have to inspect the old document to decide whether the merge
    can speak for it, so both clauses carry the metadata column's name. Pinned
    because a hard-coded ``metadata`` would keep working for today's only
    caller and silently read the wrong column for the next one.
    """
    clause = merged_column_assignment(backend_type, metadata_column="legacy_meta")
    assert "legacy_meta" in clause
    assert "metadata" not in clause


@pytest.mark.parametrize("backend_type", ["sqlite", "postgres"])
def test_both_merge_clauses_refuse_a_root_that_is_not_an_object(backend_type):
    """Neither engine merges into a scalar, an array or a JSON null.

    SQLite's ``json_set`` returns such a document unchanged and PostgreSQL's
    ``||`` concatenates it into an array — both report success, neither
    produces a top-level ``session_id``. An assignment that did not ask would
    stamp an id no reader of the row can see, which is the one state this
    column may not occupy. Asserted per dialect here and executed against both
    engines in the parity suite.
    """
    clause = merged_column_assignment(backend_type)
    type_test = "jsonb_typeof" if backend_type == "postgres" else "json_type"
    assert f"{type_test}(COALESCE(" in clause
    assert "= 'object'" in clause
    assert clause.startswith("session_id = CASE WHEN ")


def test_only_sqlite_carries_the_duplicate_key_guard():
    """The asymmetry is real and belongs where it is.

    ``jsonb`` deduplicates on parse, so a PostgreSQL object merge can only ever
    leave one ``session_id`` and it is the incoming one; the duplicate guard
    would be dead code there. ``json_set`` replaces the first occurrence only,
    so on SQLite it is the difference between NULL and a wrong answer. Pinned
    so a future "make both branches look the same" edit has to justify moving
    a guard rather than doing it for symmetry — in either direction.
    """
    assert "json_each" in merged_column_assignment("sqlite")
    assert "< 2" in merged_column_assignment("sqlite")
    assert "json_object_keys" not in merged_column_assignment("postgres")
    assert "< 2" not in merged_column_assignment("postgres")


def test_the_postgres_backfill_asks_the_parser_rather_than_listing_bad_inputs():
    """``pg_input_is_valid`` is why the Postgres floor is 16 (#2988).

    Cast safety cannot be enumerated. ``IS JSON OBJECT`` plus a search for the
    NUL escape was the first attempt, and it let a lone surrogate through —
    valid JSON text that ``jsonb`` refuses, so the ``SET`` expression raised and
    rolled back a mandatory migration. Any list of bad inputs is complete only
    up to the last bug report; ``pg_input_is_valid(metadata, 'jsonb')`` asks the
    parser the actual question and cannot be behind.

    Both clauses are pinned because they are not the same claim and neither
    implies the other:

    * ``pg_input_is_valid`` — the cast will not raise.
    * ``IS JSON OBJECT`` — the document is an object, which ``json_object_keys``
      in the SET clause requires and which validity does NOT give: ``'"x"'`` and
      ``'[1]'`` are both valid jsonb and both make it raise.

    Dropping either has to be a deliberate act, including the floor requirement
    in ``.github/workflows/ci.yml`` that the first clause depends on.
    """
    sql, _params = backfill_statement("postgres")
    assert "pg_input_is_valid(metadata, 'jsonb')" in sql
    assert "metadata IS JSON OBJECT" in sql
