"""What may be stamped into ``conversation_history.session_id`` (#2958).

Session identity has only ever lived inside each row's ``metadata`` JSON. A
value inside a JSON blob cannot be indexed, so ``list_conversations`` could not
query by it and had to oversample history and hope the session it wanted fell
inside the window. Phase A of #2948 gives that value a column of its own.

The column is a **derived duplicate**, not a new source of truth: metadata stays
authoritative until the read path moves. This module owns the one question that
duplicate raises — *which* metadata values are eligible to be copied into it —
and answers it in three renderings that must agree: Python (the write paths that
stamp the column), SQLite, and PostgreSQL (the two halves of the legacy
backfill).

The invariant
=============

    The column may be NULL where the grouper is canonical.
    It may never disagree.

:func:`~kestrel_sovereign.storage.session_grouping.group_messages_into_sessions`
files a row under ``metadata.session_id`` when that value is truthy and not
``str(...).isdigit()``. This contract is deliberately **strictly narrower**: a
value it accepts is always one the grouper would also accept, so the column can
only ever be silent, never wrong. Everything a reader might have to reconcile is
therefore a NULL, and Phase C already has to tolerate NULL for rows that simply
never had a session id.

Narrower is not fussiness — it is what makes one rule expressible in three
languages. Two of the three cannot be asked Python's questions:

* ``str.isdigit()`` is true for non-ASCII digits (``"١٢٣"``), which neither SQL
  dialect's digit test matches. Restricting the charset to ASCII settles that
  disagreement before the digit test is reached, rather than trying to teach SQL
  Python's Unicode tables.
* A JSON value that is not a string renders differently in every reader:
  ``{"session_id": true}`` extracts as ``1`` in SQLite, ``'true'`` in Postgres
  and ``'True'`` under Python's ``str()``. Nothing may be filed under an
  identity whose spelling depends on who read it, so only JSON strings qualify.
* PostgreSQL cannot hold a NUL in ``TEXT`` at all, and its ``jsonb`` rejects a
  wider set of documents than JSON syntax does: the NUL escape, and any
  ill-formed UTF-16 escape such as a lone surrogate. ``metadata::jsonb`` raises
  on those rather than returning a bad value, and inside a mandatory migration
  a raise is a failed boot, not a skipped row. The backfill therefore asks
  ``pg_input_is_valid`` whether the cast will succeed instead of listing the
  documents on which it would not.
* An oversized id would be accepted by every writer and then fail when the
  composite ``(agent_id, session_id)`` B-tree entry exceeds PostgreSQL's
  ~2704-byte page limit — at index build time, which is again boot.
* JSON permits a key to appear twice, and the three readers resolve
  ``{"session_id": "a", "session_id": "b"}`` three different ways: SQLite's
  ``json_extract`` takes the FIRST, PostgreSQL's ``jsonb`` and Python's
  ``json.loads`` take the LAST. Measured, not assumed — on sqlite 3.50 and
  PostgreSQL 16.14. Whichever occurrence is "right", two of the readers
  disagree with it, so a duplicated key has no single answer and is refused
  outright. Only the TOP level is examined, because that is the only level any
  of the three consults for this key.

Real non-UUID ids exist and are legitimate (``sess_*`` keys are in live data),
so the charset admits them; the rule excludes bare integers, which are the
mis-filed legacy keys of #2012 that the grouper already ignores.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

# ── The rule, authored once ──────────────────────────────────────────────
#
# Every other spelling in this module is COMPUTED from these three lines. They
# are not documentation of a rule stated elsewhere; there is nowhere else.
_ALLOWED_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_-"
)
_DIGITS = frozenset("0123456789")

#: The metadata key the column duplicates. Named rather than spelled inline so
#: a caller that has to ask "is this write touching the indexed value?" — see
#: ``AsyncConversationStore.update_message_metadata`` — is provably asking
#: about the same key the backfill below reads.
SESSION_ID_KEY = "session_id"

#: Longest session id that may reach the column, in bytes. The charset above is
#: ASCII-only, so for any value that gets this far bytes and characters are the
#: same count — which is why each dialect may use its own cheapest length
#: function below without the two meaning different things.
SESSION_ID_MAX_LENGTH = 512

#: The JSON escape a NUL is stored as. **SQLite only.** PostgreSQL needs no
#: such term — ``pg_input_is_valid(metadata, 'jsonb')`` already declines every
#: document whose cast would raise, this one included — but SQLite parses a NUL
#: escape happily and then hands the result to ``length()`` and ``GLOB``, which
#: are NUL-terminated C string functions and cannot see past the first one.
#: ``"a\\u0000b"`` would measure 1 character and match the charset, so without a
#: raw-text search SQLite would stamp a value PostgreSQL cannot even store.
_NUL_ESCAPE = "\\u0000"


def _sql_character_class(characters: frozenset) -> str:
    """Render a character set as a bracket-expression body, ranges collapsed.

    One rendering serves both dialects because PostgreSQL's regex bracket
    expressions and SQLite's GLOB bracket expressions agree on everything used
    here: ``a-z`` is a range, and a ``-`` in leading position is a literal.

    A literal ``-`` is emitted first for that reason — anywhere else it would
    quietly become a range operator and widen the class. The characters that
    have no shared meaning across the two dialects (``^`` negates, ``]``
    closes, ``\\`` escapes in one and not the other) are refused outright by
    :func:`_assert_class_denotes`, which then proves the result by parsing it
    back.
    """
    literal_dash = "-" if "-" in characters else ""
    codepoints = sorted(ord(c) for c in characters if c != "-")
    runs: list = []
    for point in codepoints:
        if runs and point == runs[-1][1] + 1:
            runs[-1][1] = point
        else:
            runs.append([point, point])
    body = literal_dash
    for start, end in runs:
        if end - start >= 2:
            body += f"{chr(start)}-{chr(end)}"
        else:
            body += "".join(chr(c) for c in range(start, end + 1))
    return body


def _parse_sql_character_class(body: str) -> frozenset:
    """Read a bracket-expression body back as the set of characters it matches.

    The inverse of :func:`_sql_character_class`, used to PROVE the generated
    class denotes the authored set rather than to trust that it does. A rule
    stated once and compiled twice is only worth the compilation if the
    compilation is checked.
    """
    characters = set()
    index = 0
    while index < len(body):
        if index + 2 < len(body) and body[index + 1] == "-":
            characters.update(
                chr(c) for c in range(ord(body[index]), ord(body[index + 2]) + 1)
            )
            index += 3
        else:
            characters.add(body[index])
            index += 1
    return frozenset(characters)


def _assert_class_denotes(body: str, characters: frozenset) -> str:
    """Refuse to import a class that does not mean what it was generated from."""
    unportable = characters & frozenset("^]\\")
    if unportable:
        raise ValueError(
            f"session id character set contains {sorted(unportable)!r}, which "
            "do not mean the same thing in a PostgreSQL regex class and a "
            "SQLite GLOB class"
        )
    if _parse_sql_character_class(body) != characters:
        raise ValueError(
            f"character class {body!r} does not denote {sorted(characters)!r}"
        )
    return body


_ALLOWED_CLASS = _assert_class_denotes(
    _sql_character_class(_ALLOWED_CHARACTERS), _ALLOWED_CHARACTERS
)
_DIGIT_CLASS = _assert_class_denotes(
    _sql_character_class(_DIGITS), _DIGITS
)


def is_storable_session_id(value: Any) -> bool:
    """Whether ``conversation_sessions`` can hold this key on BOTH engines.

    A **weaker** question than :func:`is_stampable_session_id`, and a different
    one. That function asks what the indexed DUPLICATE may hold, and is narrow
    on purpose so one rule can be written in three languages; this asks only
    what a primary key can physically store, because #2960 keys the session
    projection on whatever the grouper produced — including the ``sms:{sender}``
    ids ``rasa_shim`` writes, which the column contract excludes and every
    reader opens without difficulty.

    Three limits, and each is PostgreSQL telling us so at insert time — which
    is inside the repair that runs on the first page of every conversation
    list, so a key past one of them does not lose a session, it fails the whole
    list:

    * **No NUL.** PostgreSQL cannot hold one in ``TEXT`` at all.
    * **Encodable.** A lone surrogate has no UTF-8 encoding, so the driver
      cannot send it.
    * **Bounded.** A composite B-tree entry over ~2704 bytes is refused.
      :data:`SESSION_ID_MAX_LENGTH` is the bound used rather than that number,
      because it is the length a session id already has everywhere else in this
      system and one number that two tables share cannot drift into two.

    Measured in BYTES, not characters: this admits non-ASCII, where the two
    differ by up to 4×, and it is bytes the index counts.
    """
    if not isinstance(value, str) or not value:
        return False
    if "\x00" in value:
        return False
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return len(encoded) <= SESSION_ID_MAX_LENGTH


def is_stampable_session_id(value: Any) -> bool:
    """Whether ``value`` may be copied into the indexed column.

    The Python rendering of the rule. See the module docstring for why each
    clause is narrower than session grouping's.
    """
    return (
        isinstance(value, str)
        and len(value) <= SESSION_ID_MAX_LENGTH
        and all(character in _ALLOWED_CHARACTERS for character in value)
        and any(character not in _DIGITS for character in value)
    )


def column_session_id(metadata: Any) -> Optional[str]:
    """The value ``conversation_history.session_id`` takes for this row.

    Write paths hand this the metadata they are about to persist, so the column
    is derived from the same bytes the row stores rather than from a variable
    that might have moved on. Callers hold that metadata at different points in
    its life, so it is accepted either parsed (a dict) or exactly as stored
    (JSON text); anything else — unparseable, not an object, absent — has no
    stampable id and leaves the column NULL.

    The two forms are not quite the same question, which is why the text form
    is not simply parsed into the dict form and handed on. A ``dict`` has
    already lost the one thing this has to see: JSON text may carry
    ``session_id`` twice, and ``json.loads`` silently keeps the last of them
    while SQLite would have read the first. Parsing to pairs preserves the
    duplicate so it can be refused instead of arbitrated (see the module
    docstring). A caller holding a ``dict`` cannot have a duplicate to refuse.
    """
    if isinstance(metadata, (str, bytes, bytearray)):
        try:
            # A JSON object parses to a TUPLE of pairs under this hook and an
            # array to a list, so — unlike ``object_pairs_hook=list`` — the two
            # cannot be confused, and ``[["session_id", "x"]]`` is not mistaken
            # for an object carrying that key.
            document = json.loads(metadata, object_pairs_hook=tuple)
        except (TypeError, ValueError):
            return None
        if not isinstance(document, tuple):
            return None
        occurrences = [
            item for key, item in document if key == SESSION_ID_KEY
        ]
        # 0 = absent, >1 = a duplicate no two readers resolve alike.
        if len(occurrences) != 1:
            return None
        value = occurrences[0]
    elif isinstance(metadata, dict):
        value = metadata.get(SESSION_ID_KEY)
    else:
        return None
    return value if is_stampable_session_id(value) else None


def merged_column_assignment(
    backend_type: str, metadata_column: str = "metadata"
) -> str:
    """The ``session_id = ...`` clause for an UPDATE that merges metadata.

    Returned with exactly one ``?`` placeholder, which the caller binds to
    :func:`column_session_id` of the *update* — the merge is last-writer-wins
    on this key, so on a document the merge can speak for, what
    ``metadata.session_id`` says afterwards is exactly what arrived. Deriving
    from the update rather than re-reading the row is what keeps this a single
    statement, and single is what keeps it safe from the read-modify-write race
    :meth:`AsyncConversationStore.update_message_metadata` exists to avoid.

    "A document the merge can speak for" is the part that is not free, and it
    is the whole reason this is a function rather than a constant string. The
    stamp is allowed only where the merged row is an OBJECT carrying the key
    exactly once; anywhere else the column takes NULL, which is the answer
    :func:`column_session_id` gives the same document and one Phase C already
    tolerates.

    Both dialects fail that in a shape that is easy to miss, because both keep
    reporting success while quietly declining to merge:

    * **A non-object root.** ``metadata`` is free text, and legacy rows hold
      ``42``, ``[]``, ``"text"`` and ``null`` as well as objects. Measured on
      sqlite 3.50.4, ``json_set('42', '$."session_id"', json('"x"'))`` returns
      ``42`` — the document's value unchanged (it is reserialized compactly,
      but nothing is added to it) and no error. Measured on PostgreSQL 16.14,
      ``'42'::jsonb || '{"session_id":"x"}'::jsonb`` returns
      ``[42, {"session_id": "x"}]`` — ``||`` promotes a non-object operand to a
      one-element array and CONCATENATES rather than merging. Neither result
      carries a top-level ``session_id``, so an unguarded assignment left the
      column naming a session no reader of that row can see. That is the exact
      state the column may never occupy, and it was reachable on both engines.
    * **A duplicated key, on SQLite only.** ``json_set`` replaces only the
      FIRST occurrence. Measured on 3.50.4: setting ``session_id`` on
      ``{"session_id":"a","session_id":"b"}`` yields
      ``{"session_id":"<new>","session_id":"b"}`` — ``json_extract`` then reads
      the new value while ``json.loads``, and so session grouping, still reads
      ``b``. PostgreSQL needs no counterpart: ``jsonb`` deduplicates on parse,
      so an object merge *repairs* a duplicated legacy document on its way
      past, leaving one key holding the incoming value.

    Both conditions are asked of the OLD document, because every ``SET``
    expression sees the pre-update row — and that is not an approximation of
    the question:

    * Objectness cannot be created or destroyed by this merge. The single
      placeholder is bound to ``column_session_id(update)``, which returns
      non-``NULL`` only for a mapping (or JSON object text) — so whenever there
      is anything to stamp, the patch is itself an object, and on both dialects
      merging an object patch leaves an object root an object and a non-object
      root a non-object. Old-is-object and merged-is-object are the same claim.
    * A document holding the key at most once can only hold it once after the
      merge; two or more and ``json_set`` cannot collapse them.

    ``COALESCE`` is inside both tests for the same reason it is inside the
    merge: a SQL ``NULL`` metadata column becomes ``{}``, a genuine object, and
    such a row must keep stamping normally. A JSON ``null`` is a different
    thing — it is a value, not an absence, ``COALESCE`` leaves it alone, and it
    is not an object.

    Collapsing the SQLite duplicate instead was the alternative, and it was
    declined: it would mean rewriting the whole document on every metadata
    update of any key, changing merge semantics well outside this column's
    remit for the sake of a legacy shape the reader has to tolerate anyway.

    ``json_type``/``jsonb_typeof`` and ``json_each`` all raise on malformed
    metadata — but so do the ``json_set`` and ``::jsonb`` they ride beside, so
    this adds no failure mode the statement did not already have.
    """
    if backend_type == "postgres":
        return (
            "session_id = CASE WHEN jsonb_typeof(COALESCE("
            f"{metadata_column}::jsonb, '{{}}'::jsonb)) = 'object' THEN ? END"
        )
    return (
        f"session_id = CASE WHEN json_type(COALESCE({metadata_column}, '{{}}')) "
        "= 'object' AND (SELECT count(*) FROM "
        f"json_each(COALESCE({metadata_column}, '{{}}')) "
        f"WHERE key = '{SESSION_ID_KEY}') < 2 THEN ? END"
    )


def backfill_statement(backend_type: str) -> Tuple[str, tuple]:
    """``(sql, params)`` lifting legacy ``metadata.session_id`` into the column.

    Declared as a ``backfills`` entry on the same ``migrate_columns_once`` call
    that adds the column, so the ALTER and this land in one transaction.

    **Where each half of the rule is spelled is load-bearing, not stylistic.**
    Neither dialect guarantees the evaluation order of ``AND`` in a ``WHERE``
    clause, and on PostgreSQL the JSON parse *raises* rather than returning
    false — so a guard that sits beside the parse in the same conjunction can be
    evaluated after it and never protect anything. (The ``CASE WHEN <guard>
    THEN <parse>`` shape has the same hole for a different reason: the guard is
    inside the row's projection, but so is the parse.) What IS guaranteed is
    that a row failing ``WHERE`` never has its ``SET`` expression evaluated at
    all. So the split is:

    * ``WHERE`` carries **only text-level predicates** — ones that cannot raise
      on any input, whatever order the planner picks. These establish that the
      JSON is parseable at all.
    * ``SET`` carries the value-level predicates, on rows already proven safe to
      parse.

    Rows whose metadata is absent, malformed, NUL-bearing, or simply has no
    session id keep the column's NULL default, which is what Phase C must
    tolerate anyway.

    The JSON-type test in each ``CASE`` is the clause that does the least
    obvious work, so it is the one most likely to be pruned by a later reader
    as redundant. It is not: a negative JSON number renders as ``-5``, which
    is inside the charset and is not all digits, so without the type test both
    dialects would stamp ``'-5'`` for a value Python declines outright. That
    is the cross-backend disagreement this module exists to prevent, and it is
    in both corpora as ``json negative number``.
    """
    if backend_type == "postgres":
        # ``pg_input_is_valid(metadata, 'jsonb')`` (PG16+) answers the only
        # question the ``SET`` clause below actually needs — *will this text
        # cast to jsonb?* — and answers it without performing the cast, so it
        # cannot raise on the input it exists to reject.
        #
        # It replaced an ``IS JSON OBJECT`` paired with a text search for
        # the NUL escape, and that is a change of KIND, not of degree: the
        # pair ENUMERATED the ways a document can be valid JSON that jsonb
        # cannot represent, and an enumeration is only ever as complete as
        # the last bug report. It knew the NUL escape and not the lone
        # surrogate — ``{"session_id": "\\ud800"}`` passes ``IS JSON OBJECT``
        # and carries no NUL, and the cast then fails with "Unicode low
        # surrogate must follow a high surrogate". Inside a mandatory
        # migration that is a rolled-back boot, forever, because the schema is
        # the marker and the next boot starts from the same row. Measured on
        # PostgreSQL 16.14: the old predicate raises on that document, this one
        # skips it. ``pg_input_is_valid`` covers both, and covers whatever the
        # next one turns out to be, because it asks the parser instead of
        # describing it.
        #
        # ``IS JSON OBJECT`` stays, and is NOT redundant beside it: validity is
        # not objectness. ``'"hello"'`` and ``'[1]'`` are both valid jsonb, and
        # ``json_object_keys`` in the SET clause raises on either. Verified —
        # ``pg_input_is_valid`` returns true for both.
        #
        # ``json_object_keys`` is asked of ``::json``, not ``::jsonb``: the
        # binary form deduplicates on the way in, so a duplicate key is already
        # gone by the time it could be counted. ``::json`` keeps the document
        # as written. Both casts are safe here — the row reached the SET clause
        # only by passing the text-level ``WHERE`` above, and jsonb validity
        # implies json validity (jsonb is the stricter of the two).
        return (
            "UPDATE conversation_history SET session_id = CASE WHEN "
            f"jsonb_typeof(metadata::jsonb -> '{SESSION_ID_KEY}') = 'string' "
            f"AND (SELECT count(*) FROM json_object_keys(metadata::json) key "
            f"WHERE key = '{SESSION_ID_KEY}') = 1 "
            f"AND length(metadata::jsonb ->> '{SESSION_ID_KEY}') "
            f"<= {SESSION_ID_MAX_LENGTH} "
            f"AND (metadata::jsonb ->> '{SESSION_ID_KEY}') "
            f"~ '^[{_ALLOWED_CLASS}]+$' "
            f"AND (metadata::jsonb ->> '{SESSION_ID_KEY}') "
            f"!~ '^[{_DIGIT_CLASS}]+$' "
            f"THEN metadata::jsonb ->> '{SESSION_ID_KEY}' END "
            "WHERE pg_input_is_valid(metadata, 'jsonb') "
            "AND metadata IS JSON OBJECT "
            f"AND position('{SESSION_ID_KEY}' in metadata) > 0",
            (),
        )
    # SQLite's json_extract RAISES on malformed metadata, which inside a
    # migration transaction is a failed boot rather than a skipped row, so
    # ``json_valid`` does here exactly what ``IS JSON OBJECT`` does above.
    #
    # There is deliberately no counterpart to the ``OBJECT`` half of that
    # test. ``json_type(metadata, '$.session_id')`` returns NULL — never a
    # type, never an error — for every document that is not an object with
    # that key, so requiring 'text' already says "an object, carrying a
    # session_id, holding a string". A separate ``json_type(metadata) =
    # 'object'`` clause cannot change the outcome of any input, which makes it
    # a guard no test can defend: it would read as protection while a mutation
    # removing it went unnoticed. The behaviour it looks like it provides is
    # asserted against both engines in the parity suite instead.
    #
    # ``json_each`` walks the document as written and so yields a duplicated
    # key twice, which is what makes the count a real test rather than a
    # tautology. It is also the clause that must NOT be moved into the WHERE:
    # it raises on malformed metadata, exactly like the extraction it guards.
    return (
        "UPDATE conversation_history SET session_id = CASE WHEN "
        f"json_type(metadata, '$.{SESSION_ID_KEY}') = 'text' "
        f"AND (SELECT count(*) FROM json_each(metadata) "
        f"WHERE key = '{SESSION_ID_KEY}') = 1 "
        f"AND length(json_extract(metadata, '$.{SESSION_ID_KEY}')) "
        f"<= {SESSION_ID_MAX_LENGTH} "
        f"AND NOT json_extract(metadata, '$.{SESSION_ID_KEY}') "
        f"GLOB '*[^{_ALLOWED_CLASS}]*' "
        f"AND json_extract(metadata, '$.{SESSION_ID_KEY}') "
        f"GLOB '*[^{_DIGIT_CLASS}]*' "
        f"THEN json_extract(metadata, '$.{SESSION_ID_KEY}') END "
        "WHERE json_valid(metadata) = 1 "
        "AND instr(metadata, ?) = 0 "
        f"AND instr(metadata, '{SESSION_ID_KEY}') > 0",
        (_NUL_ESCAPE,),
    )


def describe_contract() -> Dict[str, Any]:
    """The rule as data, for tests that assert the renderings agree."""
    return {
        "key": SESSION_ID_KEY,
        "allowed_characters": _ALLOWED_CHARACTERS,
        "digits": _DIGITS,
        "max_length": SESSION_ID_MAX_LENGTH,
        "allowed_class": _ALLOWED_CLASS,
        "digit_class": _DIGIT_CLASS,
        "nul_escape": _NUL_ESCAPE,
    }
