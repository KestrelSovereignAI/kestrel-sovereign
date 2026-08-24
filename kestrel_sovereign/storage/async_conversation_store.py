"""
Async Conversation Store for Kestrel Storage.

Provides async conversation history management with encryption at rest.
Encryption is enabled via KESTREL_DATA_KEY environment variable.

Key Versioning:
    - key_version: 0 (or missing) = global key (backward compat)
    - key_version: 1 = per-agent HKDF-derived key

All queries are scoped by agent_id for multi-tenant isolation.
"""
import json
import logging
import os
import re
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .async_database import AsyncDatabase
from .conversation_ids import coerce_persistent_message_id
from .conversation_created_at import UNDATED_TABLE
from .session_grouping import (
    UNDATABLE_ROW_FALLBACK,
    canonical_session_id,
    canonical_timestamp_sql,
    iso_session_timestamp,
    parse_message_metadata,
    coerce_session_timestamp,
    coalesce_sessions_by_session_id,
    group_messages_into_sessions,
    session_cursor_values,
    sort_sessions,
    summarize_sessions,
    timestamp_predicate,
    timestamp_query_param,
)
from .session_id_column import (
    SESSION_ID_KEY,
    column_session_id,
    merged_column_assignment,
)
from .destructive_audit import DestructiveAuditEvent, DestructiveAuditLog, hash_rows
from .sqla.embedding_profile import upsert_embedding_profile as _upsert_embedding_profile
from .lexical_memory_index import (
    ConversationLexicalIndex,
    LexicalIndexReplacement,
)
from .encryption import (
    get_fernet, get_agent_fernet, encrypt_string, decrypt_string, remove_enc_flag,
    DecryptionError
)
from kestrel_sovereign.security.input_guardrails import extract_raw_user_content

logger = logging.getLogger(__name__)


class ProjectionNotReady(RuntimeError):
    """The session index is not complete enough to answer the list (#2960).

    Its own type because the caller has to tell it from a storage failure: this
    one is temporary and self-correcting — the next request continues the walk
    — so it is a 503 rather than a 500.
    """


#: How many budgeted repairs one list request will run before refusing.
#:
#: Each is ``STEP_BUDGET * CHUNK_ROWS`` live rows, so this is tens of millions
#: of rows for one agent — far past anything measured, and past the point where
#: a synchronous rebuild belongs on a request at all. It is a backstop on the
#: refusal, not a work budget: the loop exits the moment the walk lands, which
#: for every agent this has been measured on is the first pass.
REPAIR_PASSES = 8

#: How many times a list request will re-read a page that was rebuilt under it.
#:
#: Each attempt is three primary-key reads and one bounded page query, so this
#: is cheap; it is a bound on a race rather than on work. Losing it three times
#: running means a projection being rebuilt continuously, which is a state to
#: report rather than to keep retrying inside a request.
_PAGE_ATTEMPTS = 3

# Current key version for new encryptions
CURRENT_KEY_VERSION = 1

# Keep exact-id purge statements below the oldest supported SQLite bind
# ceiling (999).  PostgreSQL permits much larger statements, but sharing one
# conservative batch size keeps the destructive path backend-neutral.
_EXACT_PURGE_BATCH_SIZE = 500
_PURGE_SELECT_PLACEHOLDER = "__KESTREL_PURGE_SELECT_COLUMNS__"
_PURGE_BASE_SELECT_COLUMNS = (
    "id, role, content, metadata, created_at, deleted_at"
)
_PURGE_LEXICAL_SELECT_COLUMNS = (
    f"{_PURGE_BASE_SELECT_COLUMNS}, lexical_index_id"
)
_PURGE_NO_LEXICAL_SELECT_COLUMNS = (
    f"{_PURGE_BASE_SELECT_COLUMNS}, "
    "CAST(NULL AS TEXT) AS lexical_index_id"
)


class ConversationLexicalSchemaError(RuntimeError):
    """The history/token-table halves of the lexical schema disagree."""


class ConversationSessionTimestampError(RuntimeError):
    """Exact session membership cannot be proven from stored chronology."""


@dataclass(slots=True)
class _PreparedConversationWrite:
    """Precomputed, not-yet-visible conversation row material.

    Semantic recall persistence must acquire the assertion lifecycle fence only
    for its final liveness check and INSERT.  Provider embedding and lexical
    token indexing are intentionally prepared before that narrow critical
    section; the lexical token handle remains available for cleanup if the
    final fenced write cannot be admitted.
    """

    role: str
    content: str
    rendered_content: Optional[str]
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
    embedding_profile_id: Optional[str]
    embedding_service: Optional[Any]
    model: Optional[str]
    provider: Optional[str]
    lexical_index_id: Optional[str]
    lexical_index_version: Optional[str]


def _escape_like_session_value(session_id: str) -> str:
    """JSON-escape then LIKE-escape a ``session_id`` for a ``metadata LIKE`` match.

    The stored metadata is JSON, so the value appears JSON-escaped; we match that
    form, then escape LIKE wildcards (``%``/``_``/``\\``) so a wildcard in the id
    can't broaden the match to every row (#1729). Pair with ``ESCAPE '\\'``. For
    an ordinary UUID session id every pass is a no-op.
    """
    json_frag = json.dumps(session_id)[1:-1]
    return (
        json_frag.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _metadata_like_forms(key: str, value_literal: str) -> tuple[str, str]:
    """Return the two ``metadata LIKE`` patterns for a ``"key": value`` fragment.

    Metadata JSON is written two ways with DIFFERENT spacing, so a single
    space-form ``LIKE`` silently misses half the rows:
      * Python ``json.dumps`` emits ``"key": value`` (space after the colon).
      * SQLite ``json_set()`` / ``json()`` (used by the atomic single-flag
        metadata updates, e.g. ``decay_protected`` #2158) re-emit the WHOLE
        object MINIFIED — ``"key":value`` (no space). Any row ever touched by a
        json_set update is therefore stored minified.
    Match both. ``value_literal`` is the already-JSON-rendered value, e.g.
    ``"true"`` for a bool flag or ``'"abc"'`` for a string. Callers OR the two:
    ``AND (metadata LIKE ? OR metadata LIKE ?)``.
    """
    return (f'%"{key}": {value_literal}%', f'%"{key}":{value_literal}%')


def _serialize_embedding(embedding: Sequence[float]) -> bytes:
    """Pack a Python embedding list to little-endian float32 bytes.

    Duplicated locally (mirrors ``async_rag_store._serialize_embedding`` and
    ``saved_items_store._serialize_embedding``) so the conversation_history
    write path doesn't need to import from a sibling store. Same on-disk
    shape SQLite's ``embedding_vec`` BLOB column expects and the
    PurePythonBackend's ``_unpack`` reads — explicit little-endian (``<``)
    so the bytes round-trip on big-endian hosts. The sibling helpers
    omit the prefix and rely on the host being little-endian; we fix
    that here for new writes. (Codex P3 on PR-B.)
    """
    return struct.pack(f'<{len(embedding)}f', *embedding)


def _format_pgvector_text(embedding: Sequence[float]) -> str:
    """Format an embedding as pgvector's bind-parameter text shape.

    ``[v1,v2,…]`` — used together with a ``::vector`` cast so asyncpg
    can hand a Python string to pgvector without a typed-binary
    adapter. Mirrors the formatting in
    ``SavedItemsStore._write_embedding_vec``.
    """
    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


def _rows_affected(result) -> int:
    """Normalise the return value of AsyncDatabase.execute_commit (#763).

    Different backends return slightly different shapes:
      * SQLite/Postgres backends return ``cursor.rowcount`` (int).
      * Some legacy paths returned a Result-like object with ``.rowcount``.

    Soft-delete needs an honest count so callers can distinguish "row
    was already in trash" from "row was just trashed." The pre-existing
    ``hasattr(result, 'rowcount') else True`` fallback lied (always
    True), which broke the no-op semantics. Use this helper instead.
    """
    if isinstance(result, int):
        return result
    if hasattr(result, "rowcount") and result.rowcount is not None:
        return result.rowcount
    return 0


def _is_missing_column_error(exc: Exception, *columns: str) -> bool:
    """Whether a backend error says one of ``columns`` is absent."""
    message = str(exc).lower()
    missing_column = any(
        marker in message
        for marker in ("no such column", "no column named", "does not exist")
    )
    return missing_column and any(column.lower() in message for column in columns)


def _is_missing_lexical_column_error(exc: Exception) -> bool:
    """Whether an INSERT failed only because the additive migration is absent."""
    return _is_missing_column_error(
        exc, "lexical_index_id", "lexical_index_version"
    )


# Stopwords excluded from tokenized fallback matching so that common
# filler words in natural-language queries don't inflate match scores.
#
# DELIBERATELY KEPT IN: negation tokens (``not``, ``no``, ``never``,
# ``neither``, ``nor``, ``without``). Stripping them would let a query
# like "do not use OpenAI" reduce to "use openai" and match memories
# with the OPPOSITE meaning. Negation is semantically load-bearing in
# recall queries; better to leave it in and let the threshold gate
# borderline matches.
_SEARCH_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "been", "do", "did", "does", "has", "have", "had",
    "this", "that", "these", "those", "my", "your", "our", "we", "i",
    "me", "you", "he", "she", "they", "them", "his", "her", "its",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "about", "some", "any", "all", "each", "every", "so", "if", "then",
})

# Minimum fraction of query terms that must appear in content for a
# tokenized fallback hit.  0.6 = at least 60 % of non-stopword query
# tokens must be present — conservative enough to avoid noise while
# still catching broad natural-language recall queries.
_TOKEN_MATCH_THRESHOLD = 0.6


def _tokenize_for_search(text: str) -> List[str]:
    """Split *text* into lowercase alphanumeric tokens, dropping stopwords."""
    return [
        tok for tok in re.findall(r"[a-z0-9]+", text.lower())
        if tok not in _SEARCH_STOPWORDS and len(tok) > 1
    ]


# Transport-wrapper markers that get baked into rendered user turns
# (see ``agent/context_builder.py`` and ``agent/memory_manager.py``).
# These are prompt-replay transport, NOT canonical conversation content,
# so they must never make an otherwise-unrelated row searchable (#1537).
#
# ``_resolve_canonical`` already strips wrappers for rows explicitly
# flagged ``sent_form``. This projection is defense-in-depth for the
# residual cases — rows where the flag is missing or the anchored
# ``extract_raw_user_content`` grammar didn't fully strip — so a query
# like ``"RELEVANT MEMORIES from past conversations"`` can't match a row
# whose actual user text is unrelated and whose returned content is later
# wrapper-stripped by ``MemoryFeature._strip_sent_form_for_recall``.
_SEARCH_WRAPPER_BLOCK_RES = (
    re.compile(r"<retrieved_context>.*?</retrieved_context>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<memories>.*?</memories>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<documents>.*?</documents>", re.DOTALL | re.IGNORECASE),
    # The plain-text memory block emitted by
    # ``MemoryManager.retrieve_memories`` / ``_build_memory_context``
    # (``--- RELEVANT MEMORIES (from past conversations) --- ... ---
    # END MEMORIES ---``). The XML ``<memories>`` envelope is only added
    # one layer up in ``ContextManager.build_context``; rows persisted
    # via an older rendering path (or where the envelope was stripped but
    # the inner block was not) carry the bare delimiters. The whole
    # region is retrieved-context transport — the recalled ``[Memory N]``
    # lines are copies of canonical rows that remain independently
    # searchable — so the entire span must be removed from the match
    # text, not just the heading line (#1549).
    re.compile(
        r"-*\s*RELEVANT MEMORIES.*?-+\s*END MEMORIES\s*-*",
        re.DOTALL | re.IGNORECASE,
    ),
)
# Display-only marker lines emitted when baking retrieved memories into
# the user template. These carry words ("relevant", "memories",
# "conversations", "retrieved", "earlier") that would otherwise satisfy
# the tokenized fallback on their own. Stripped line-by-line as a
# fallback for the residual case where the bare memory block above was
# truncated mid-render (token budget) and lost its ``END MEMORIES``
# footer, so the DOTALL span match can't anchor (#1549).
_SEARCH_WRAPPER_LINE_RES = (
    re.compile(r"-*\s*RELEVANT MEMORIES.*", re.IGNORECASE),
    re.compile(
        r"NOTE: These are retrieved from earlier conversations,"
        r" not the current session\.",
        re.IGNORECASE,
    ),
    re.compile(r"-+\s*END MEMORIES\s*-*", re.IGNORECASE),
)


def _strip_search_wrappers(text: str) -> str:
    """Project *text* to its wrapper-free form for SEARCH MATCHING only.

    Rendered user turns carry transport wrappers — ``<retrieved_context>``,
    ``<memories>``, ``<documents>`` blocks plus the plain-text
    ``--- RELEVANT MEMORIES ... --- END MEMORIES ---`` block (heading,
    ``NOTE:`` line, and recalled ``[Memory N]`` lines) — baked in for
    byte-stable prompt replay (#1402). Those wrappers are generated
    retrieval-context, not conversation content; matching against them
    reintroduces the #1500/#1537/#1549 false positive where a query of
    wrapper-only terms matches a row whose actual user text is unrelated.

    This is a matching-only projection: the row's canonical ``content``
    (already split out by ``_resolve_canonical``) is what gets returned —
    this stripped form only decides whether the row is a hit. Idempotent
    on canonical rows that carry no wrappers.
    """
    if not text:
        return text
    s = text
    for pattern in _SEARCH_WRAPPER_BLOCK_RES:
        s = pattern.sub(" ", s)
    for pattern in _SEARCH_WRAPPER_LINE_RES:
        s = pattern.sub(" ", s)
    return s.replace("<user_input>", " ").replace("</user_input>", " ")


# Negation tokens form a semantic-equivalence class for the fallback
# matcher. When the query contains ANY negator, the content must contain
# AT LEAST ONE negator from this class (word-boundary-matched, so "no"
# doesn't substring-hit inside "normally") or the row is rejected as
# opposite-meaning. This handles both the substring false-positive
# (codex r2 P2) and the cross-negator equivalence — "not" in the query
# should still accept "never use OpenAI" content (codex r4 P2).
#
# Word-boundary matching is restricted to negators because applying it to
# all short tokens regressed technical-term fallback: under \b the token
# ``api`` failed to match ``api_key`` (Python treats ``_`` as a word
# character), dropping common queries below the 0.6 threshold.
_NEGATION_TOKENS = frozenset({
    "no", "not", "never", "neither", "nor", "without",
})


def _token_match_score(query_tokens: Sequence[str], content_lower: str) -> float:
    """Return the fraction of *query_tokens* found in *content_lower*.

    Plain substring matching for all tokens, with one semantic-safety
    exception: if any negator appears in the query, the content must carry
    at least one negator from the equivalence class — otherwise the row
    likely carries opposite meaning and the score is forced to 0.0.

    Returns 0.0–1.0.
    """
    if not query_tokens:
        return 0.0

    # Negation equivalence-class gate.
    if any(t in _NEGATION_TOKENS for t in query_tokens):
        if not any(
            re.search(r"\b" + re.escape(n) + r"\b", content_lower)
            for n in _NEGATION_TOKENS
        ):
            return 0.0

    hits = sum(1 for t in query_tokens if t in content_lower)
    return hits / len(query_tokens)


_MATCH_SNIPPET_RADIUS = 60


def _build_match_snippet(text: str, query_lower: str, radius: int = _MATCH_SNIPPET_RADIUS) -> str:
    """Return a short excerpt of *text* centered on the first *query_lower* hit.

    Built from the wrapper-stripped projection — the same text the match was
    decided on — so the snippet always contains the highlighted term and never
    leaks ``<retrieved_context>`` transport into the UI.
    """
    stripped = " ".join(_strip_search_wrappers(text).split())
    idx = stripped.lower().find(query_lower)
    if idx < 0:
        # Tokenized-fallback hit: no contiguous substring to center on.
        head = stripped[: radius * 2]
        return head + ("…" if len(stripped) > radius * 2 else "")
    start = max(0, idx - radius)
    end = min(len(stripped), idx + len(query_lower) + radius)
    return ("…" if start > 0 else "") + stripped[start:end].strip() + ("…" if end < len(stripped) else "")


@dataclass(frozen=True)
class SearchTerms:
    """A query compiled once, so every session is asked the same question.

    Search reaches sessions two ways now (#2961) — the grouper, for the paths
    with no table, and the #2959 projection for the active list — and both have
    to apply *identical* matching rules or a session would be findable through
    one and not the other. Compiling the query into a value both feed to
    :func:`session_match_decoration` is what makes that structural instead of a
    pair of loops that look alike.
    """

    #: The query as it is compared: stripped, lowercased.
    query_lower: str
    #: Its tokens with search wrappers removed, for the overlap fallback.
    tokens: Tuple[str, ...]
    #: Whether that fallback applies at all.
    use_token_fallback: bool

    @classmethod
    def compile(cls, query: str) -> Optional["SearchTerms"]:
        """``None`` for a query that can match nothing — an empty one.

        A *wrapper-only* query is not that: it still matches by substring, and
        only the tokenized fallback is withheld (#1554), because a query made
        entirely of transport scaffolding would otherwise score ≥0.6 against
        every sent-form row in the history.
        """
        query_lower = query.strip().lower()
        if not query_lower:
            return None
        query_tokens = _tokenize_for_search(query)
        stripped_query_tokens = _tokenize_for_search(_strip_search_wrappers(query))
        query_is_wrapper_only = bool(query_tokens) and not stripped_query_tokens
        return cls(
            query_lower=query_lower,
            tokens=tuple(stripped_query_tokens),
            use_token_fallback=(
                not query_is_wrapper_only and len(stripped_query_tokens) >= 2
            ),
        )


def session_match_decoration(
    messages: Iterable[Dict[str, Any]],
    name: Optional[str],
    terms: SearchTerms,
) -> Optional[Dict[str, Any]]:
    """One session's ``match_*`` fields, or ``None`` when it does not match.

    ``messages`` are that session's rows, **decrypted and canonicalized**;
    ``name`` its user-assigned title, which is matched too so a renamed
    conversation is findable by the name the user gave it even when nothing in
    its text says so.

    Matching reuses the exact semantics of ``search_history``: substring match
    against the wrapper-stripped projection first (#1537/#1549), then the
    tokenized ≥0.6-overlap fallback for multi-term queries.

    The one function both search paths call. It is given rows rather than
    finding them, because *which* rows a session is made of is the question the
    two paths answer differently — the grouper computes it for a transcript it
    holds, the active path reads it out of the membership map — and that is the
    only part that should differ.
    """
    match_count = 0
    first_hit: Optional[Dict[str, Any]] = None
    best_token: Optional[Tuple[float, Dict[str, Any]]] = None
    for msg in messages:
        content = msg.get("content") or ""
        if not isinstance(content, str):
            continue
        content_lower = _strip_search_wrappers(content).lower()
        if terms.query_lower in content_lower:
            match_count += 1
            if first_hit is None:
                first_hit = msg
        elif terms.use_token_fallback and match_count == 0:
            score = _token_match_score(terms.tokens, content_lower)
            if score >= _TOKEN_MATCH_THRESHOLD and (
                best_token is None or score > best_token[0]
            ):
                best_token = (score, msg)

    name_match = bool(name) and terms.query_lower in name.lower()
    if match_count == 0 and first_hit is None and best_token is not None:
        first_hit = best_token[1]
        match_count = 1
    if match_count == 0 and not name_match:
        return None

    if first_hit is None:
        # Title-only hit: the name itself is the evidence.
        return {"match_count": match_count, "match_role": None, "match_snippet": None}
    return {
        "match_count": match_count,
        "match_role": first_hit.get("role"),
        "match_snippet": _build_match_snippet(
            first_hit.get("content") or "", terms.query_lower
        ),
    }


def _within_row_budget(
    page: Sequence[Dict[str, Any]],
    membership: Dict[str, List[Any]],
    budget: int,
) -> Iterable[List[Dict[str, Any]]]:
    """Split a page of sessions into groups whose rows fit one read.

    A step of the search walk is bounded in *sessions*, and a session is not
    bounded in anything — one long-running conversation can hold more rows than
    the rest of the page together. Reading a step's rows in one go would
    therefore be an unbounded read wearing a bounded number's clothes, so the
    rows are what the batches are counted in.

    A session larger than ``budget`` on its own is still read whole, in a batch
    of one: its rows cannot be matched apart from each other, and splitting it
    would report two partial answers where the session has one.
    """
    batch: List[Dict[str, Any]] = []
    rows = 0
    for session in page:
        size = len(membership.get(session["session_id"], ()))
        if batch and rows + size > budget:
            yield batch
            batch, rows = [], 0
        batch.append(session)
        rows += size
    if batch:
        yield batch


def search_session_summaries(
    normalized_messages: List[Dict[str, Any]],
    query: str,
    names: Optional[Dict[str, str]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Full-text search over a transcript this caller holds, grouped and matched.

    **The path for a membership with no sessions table.** ISOLATED privacy mode
    keeps its conversations in an in-memory buffer, and the archived view is
    disjoint from what the #2959 projection describes (#3062) — neither has a
    row to look a session up in, so both derive the sessions from the rows they
    have. The active view does not come through here any more: it walks the
    table, in :meth:`AsyncConversationStore._search_active_sessions`.

    Callers hand in **decrypted**, oldest-first normalized message dicts
    (``id``/``role``/``content``/``metadata``/``created_at``) and get back
    newest-first session summaries for the sessions whose content — or
    user-assigned title — matches *query*.

    Returns session dicts shaped like ``group_messages_into_sessions`` output
    (``preview_content``/``preview_metadata``/``preview_wake_source`` retained
    for the endpoint's decorator) plus ``name`` (when titled), ``match_count``,
    ``match_role``, and ``match_snippet`` — decrypted plaintext excerpts around
    the first hit.
    """
    names = names or {}
    terms = SearchTerms.compile(query)
    if terms is None:
        return []

    # keep_empty_markers=True to mirror the UI list (#2222): a just-started,
    # renamed-but-still-empty conversation is list-visible, so its title must
    # be title-searchable too — dropping the marker here would make search
    # disagree with the list it filters.
    grouped = coalesce_sessions_by_session_id(
        group_messages_into_sessions(
            normalized_messages, collect_messages=True, keep_empty_markers=True
        )
    )

    results: List[Dict[str, Any]] = []
    for session in grouped:
        messages = session.pop("messages", [])
        sid = session.get("session_id")
        name = names.get(sid)
        if name is not None:
            session["name"] = name

        decoration = session_match_decoration(messages, name, terms)
        if decoration is None:
            continue
        session.update(decoration)
        results.append(session)

    # The same order the list and the #2959 projection page by, not a third
    # spelling of "newest first". Ties are ordinary and `limit` truncates, so
    # sorting on `last_message_at` alone let search and the list disagree about
    # WHICH session made the page (round-8/9 review).
    sort_sessions(results)
    return results[:limit]

class AsyncConversationStore:
    """Async conversation history storage with per-agent encryption."""

    def __init__(
        self,
        db: AsyncDatabase,
        agent_id: str = "",
        llm_service: Optional[Any] = None,
        destructive_audit: Optional[DestructiveAuditLog] = None,
    ):
        """Initialize the conversation store.

        Args:
            db: Underlying :class:`AsyncDatabase` (SQLite or Postgres).
            agent_id: Scope every read / write to this agent. Empty
                string = global / unscoped (legacy).
            llm_service: Optional :class:`~kestrel_sovereign.llm.service.LLMService`.
                When provided, conversation embeddings are sourced from
                the active chat provider's embedding capability (OpenAI →
                text-embedding-3-small, Vertex/Google → text-embedding-004,
                Ollama → nomic-embed-text). When ``None`` the store falls
                through to ``get_provider_embedding_service(None)`` which
                resolves the global default. Set
                ``KESTREL_DISABLE_CONVERSATION_EMBEDDINGS=true`` to opt
                out entirely (e.g. when the deployment can't reach an
                embedding provider).

                Embeddings are computed against PLAINTEXT content BEFORE
                encryption. The threat model already exposes semantic-
                similarity signal through retrieval results, so the
                column doesn't widen the surface.

                Tests inject by monkey-patching
                :meth:`_lazy_embedding_service` (a method on the class so
                the override sticks per-instance).
        """
        self.db = db
        self.agent_id = agent_id
        self._llm_service = llm_service
        self._destructive_audit = destructive_audit
        # Global key for backward compatibility
        self._global_fernet = get_fernet()
        # Per-agent key (recommended, used for new data)
        self._agent_fernet = get_agent_fernet(agent_id) if agent_id else None
        # Auto-migration on read (can be disabled via env var)
        self._migrate_on_read = os.environ.get("KESTREL_DISABLE_MIGRATION") != "true"
        self._lexical_index = ConversationLexicalIndex(db, agent_id)
        self._last_lexical_bridge_stats: Dict[str, Any] = {}
        self._lexical_coverage_index_available: Optional[bool] = None

    async def _audit_destructive_operation(
        self,
        *,
        operation_type: str,
        rows: list[dict[str, Any]],
        scope: dict[str, Any],
        reason: str,
    ) -> None:
        """Write the fail-closed pre-operation destructive audit row."""

        if self._destructive_audit is None:
            return

        await self._destructive_audit.append(
            DestructiveAuditEvent(
                agent_id=self.agent_id,
                operation_type=operation_type,
                row_count=len(rows),
                pre_operation_hash=hash_rows(rows),
                snapshot_reference="inline:sha256",
                scope=scope,
                reason=reason,
            )
        )

    async def _conversation_lexical_purge_capability(self) -> tuple[str, bool]:
        """Return the safe purge projection and whether token cleanup exists.

        The #2339 lexical migration is deliberately non-fatal at startup.  A
        database may therefore have neither additive schema piece while still
        supporting ordinary conversation reads and hard deletion.  In that
        state there cannot be lexical-token residue, so the purge snapshot uses
        a typed NULL owner column instead of referencing a column that is not
        present.

        The inverse partial state is unsafe: if the token table exists without
        ``conversation_history.lexical_index_id``, ownership cannot be proven.
        Fail before audit or deletion rather than destroying history while
        leaving possibly-sensitive token rows behind.
        """
        if self.db.backend_type == "postgres":
            lexical_tokens_available = bool(
                await self.db.fetchval(
                    "SELECT to_regclass(?) IS NOT NULL",
                    ("conversation_lexical_tokens",),
                )
            )
            lexical_owner_column_available = bool(
                await self.db.fetchval(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM pg_attribute "
                    "WHERE attrelid = to_regclass(?) AND attname = ? "
                    "AND attnum > 0 AND NOT attisdropped)",
                    ("conversation_history", "lexical_index_id"),
                )
            )
        else:
            lexical_tokens_available = await self.db.table_exists(
                "conversation_lexical_tokens"
            )
            lexical_owner_column_available = bool(
                await self.db.fetchone(
                    "SELECT 1 FROM pragma_table_info('conversation_history') "
                    "WHERE name = ? LIMIT 1",
                    ("lexical_index_id",),
                )
            )

        if lexical_tokens_available and not lexical_owner_column_available:
            raise ConversationLexicalSchemaError(
                "conversation_lexical_tokens exists but "
                "conversation_history.lexical_index_id is absent; refusing "
                "hard purge because lexical-token ownership cannot be proven"
            )

        if not lexical_tokens_available:
            return _PURGE_NO_LEXICAL_SELECT_COLUMNS, False
        return _PURGE_LEXICAL_SELECT_COLUMNS, True

    def _exact_id_purge_queries(
        self,
        message_ids: Sequence[int],
    ) -> list[Tuple[str, tuple[Any, ...]]]:
        """Build bounded, globally-ascending selectors for exact IDs."""
        ordered_ids = sorted({int(message_id) for message_id in message_ids})
        queries: list[Tuple[str, tuple[Any, ...]]] = []
        for start in range(0, len(ordered_ids), _EXACT_PURGE_BATCH_SIZE):
            batch = ordered_ids[start : start + _EXACT_PURGE_BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            queries.append(
                (
                    f"SELECT {_PURGE_SELECT_PLACEHOLDER} "
                    "FROM conversation_history "
                    f"WHERE agent_id = ? AND id IN ({placeholders}) "
                    "ORDER BY id ASC",
                    (self.agent_id, *batch),
                )
            )
        return queries

    @staticmethod
    def _render_purge_queries(
        selection_queries: Sequence[Tuple[str, tuple[Any, ...]]],
        projection: str,
    ) -> list[Tuple[str, tuple[Any, ...]]]:
        """Materialize capability-aware purge projections exactly once."""
        rendered = []
        for query, params in selection_queries:
            if query.count(_PURGE_SELECT_PLACEHOLDER) != 1:
                raise ValueError(
                    "Hard-purge selectors must contain exactly one canonical "
                    "projection placeholder"
                )
            rendered.append(
                (query.replace(_PURGE_SELECT_PLACEHOLDER, projection), params)
            )
        return rendered

    async def _purge_conversation_rows(
        self,
        selection_queries: Optional[Sequence[Tuple[str, tuple[Any, ...]]]],
        *,
        operation_type: str,
        scope: dict[str, Any],
        reason: str,
        resolve_message_ids: Optional[
            Callable[[], Awaitable[Sequence[int]]]
        ] = None,
    ) -> int:
        """Audit and atomically destroy one immutable message-id snapshot.

        The selectors run exactly once.  Their result is the complete set the
        audit covers and the only set the destructive transaction may touch.
        This matters for broad retention/privacy predicates: a row inserted
        after the snapshot must survive rather than being deleted unaudited by
        a second evaluation of the predicate.

        Blind-index token rows are removed in the same transaction as their
        owning conversation rows.  ``conversation_lexical_tokens`` has no
        foreign-key cascade, so every hard-purge API must come through this
        primitive or it will leave privacy-sensitive residue.

        PostgreSQL selectors must yield rows in ascending message-id order,
        and multi-query selectors must partition that same order.  This is the
        global row-lock order shared with lexical backfill; violating it can
        deadlock an overlapping privacy purge.
        """
        if (selection_queries is None) == (resolve_message_ids is None):
            raise ValueError(
                "Provide either hard-purge selection queries or one exact-ID "
                "resolver"
            )

        # Surface a static partial schema as the domain-specific error rather
        # than letting the database transaction wrapper obscure it. The state
        # is re-read under the destructive transaction's writer/DDL boundary
        # below so a concurrent startup migration cannot stale this decision.
        await self._conversation_lexical_purge_capability()
        deferred_error: Optional[Exception] = None
        purged = 0
        async with self.db.transaction():
            if self.db.backend_type == "postgres":
                # The lexical migration ALTERs conversation_history before it
                # creates token storage. Hold its DDL boundary while resolving
                # capabilities and deleting the immutable row snapshot.
                await self.db.execute(
                    "LOCK TABLE conversation_history IN ACCESS SHARE MODE"
                )
            else:
                # SQLite's default BEGIN is deferred.  Reserve the writer slot
                # before reading the audit snapshot so another connection
                # cannot replace a selected row's bytes while the external
                # audit sink is appending their hash.
                await self.db.execute(
                    "UPDATE conversation_history SET id = id WHERE 0"
                )

            projection, lexical_tokens_available = (
                await self._conversation_lexical_purge_capability()
            )
            # Same question as the lexical one, asked the same way and inside
            # the same writer/DDL boundary: does this database HAVE the table
            # whose rows must die alongside the messages? A database that never
            # ran #3009's migration has no record table and therefore no
            # records — nothing to leak — so the purge proceeds. Refusing a
            # hard purge because a record of "we could not date this row" is
            # unavailable would trade a real deletion for a bookkeeping one,
            # which is the wrong way round for a privacy primitive
            # (test_conversation_purge_schema_compat pins that a degraded
            # schema keeps hard purge AVAILABLE).
            #
            # Existence, not exceptions: a DELETE that fails for any other
            # reason must still take the transaction down with it.
            undated_records_available = await self.db.table_exists(UNDATED_TABLE)

            active_scope = dict(scope)
            active_queries = selection_queries
            if resolve_message_ids is not None:
                try:
                    resolved_ids = sorted(
                        {
                            int(message_id)
                            for message_id in await resolve_message_ids()
                        }
                    )
                except ConversationSessionTimestampError as error:
                    # Leave the transaction without touching history or the
                    # audit sink, then preserve the domain error rather than
                    # obscuring it behind the backend's TransactionError.
                    deferred_error = error
                    resolved_ids = []
                if deferred_error is None and not resolved_ids:
                    return 0
                if deferred_error is None:
                    active_scope["message_ids"] = resolved_ids
                    active_queries = self._exact_id_purge_queries(resolved_ids)
                else:
                    active_queries = ()

            rendered_queries = self._render_purge_queries(
                active_queries or (), projection
            )

            rows_by_id: dict[int, tuple[Any, ...]] = {}
            for query, params in rendered_queries:
                if self.db.backend_type == "postgres":
                    query = f"{query.rstrip()} FOR UPDATE"
                for row in await self.db.fetchall(query, params):
                    rows_by_id.setdefault(int(row[0]), row)

            rows = list(rows_by_id.values())
            snapshot = [
                {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "metadata": row[3],
                    "created_at": row[4],
                    "deleted_at": row[5],
                }
                for row in rows
            ]
            if deferred_error is None:
                try:
                    await self._audit_destructive_operation(
                        operation_type=operation_type,
                        rows=snapshot,
                        scope=active_scope,
                        reason=reason,
                    )
                except Exception as error:
                    # Exit the database transaction without deleting anything,
                    # then restore the audit sink's historical exception contract
                    # instead of letting the backend wrap it as TransactionError.
                    deferred_error = error

            if deferred_error is None:
                message_ids = list(rows_by_id)
                lexical_index_ids = list(
                    dict.fromkeys(
                        str(row[6]) for row in rows if row[6] is not None
                    )
                )
            else:
                message_ids = []
                lexical_index_ids = []
            for start in range(0, len(message_ids), _EXACT_PURGE_BATCH_SIZE):
                batch = message_ids[start : start + _EXACT_PURGE_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                affected = await self.db.execute(
                    "DELETE FROM conversation_history "
                    f"WHERE agent_id = ? AND id IN ({placeholders})",
                    (self.agent_id, *batch),
                )
                purged += _rows_affected(affected)
                # In the SAME statement batch, and for the same reason the
                # lexical tokens are: ``conversation_history_undated`` has no
                # cascade either, and it holds the ORIGINAL text of a stamp
                # #3009's migration could not read. That text came out of the
                # agent's own history and can be anything at all, so a row left
                # behind here is exactly the residue this primitive exists to
                # prevent — a permanent purge that leaves the message's
                # agent_id and a fragment of its content addressable.
                if undated_records_available:
                    await self.db.execute(
                        f"DELETE FROM {UNDATED_TABLE} "
                        f"WHERE agent_id = ? AND message_id IN ({placeholders})",
                        (self.agent_id, *batch),
                    )

            # Delete a key only after its selected owners are gone, and only
            # when no surviving row still owns it.  Legacy databases did not
            # enforce lexical-key uniqueness, so deleting tokens first could
            # silently break recall for a surviving row that shared the key.
            if lexical_tokens_available:
                async with self._lexical_index.serialized_token_cleanup(
                    lexical_index_ids
                ) as cleanup_keys:
                    for start in range(
                        0, len(cleanup_keys), _EXACT_PURGE_BATCH_SIZE
                    ):
                        batch = cleanup_keys[
                            start : start + _EXACT_PURGE_BATCH_SIZE
                        ]
                        placeholders = ",".join("?" for _ in batch)
                        await self.db.execute(
                            "DELETE FROM conversation_lexical_tokens "
                            "WHERE agent_id = ? "
                            f"AND lexical_index_id IN ({placeholders}) "
                            "AND NOT EXISTS ("
                            "SELECT 1 FROM conversation_history "
                            "WHERE agent_id = ? "
                            "AND conversation_history.lexical_index_id = "
                            "conversation_lexical_tokens.lexical_index_id)",
                            (self.agent_id, *batch, self.agent_id),
                        )

        if deferred_error is not None:
            raise deferred_error
        return purged

    def _lazy_embedding_service(self) -> Optional[Any]:
        """Return the active chat provider's embedding service when available.

        Conversation-history embedding writes live in PR-B (this PR).
        Sourcing from :func:`get_provider_embedding_service` (introduced
        in #1471) keeps saved_items, RAG, and conversation_history all
        on the same provider-backed embedding stack — switching the
        chat provider switches embeddings everywhere.

        Returns ``None`` when the provider doesn't expose embeddings
        (e.g. Anthropic, which has no embedding API) or when
        ``KESTREL_DISABLE_CONVERSATION_EMBEDDINGS=true`` opts out.
        """
        if os.environ.get(
            "KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", ""
        ).lower() == "true":
            return None
        try:
            from kestrel_sovereign.llm.embedding_service import (
                get_provider_embedding_service,
            )
        except Exception:
            return None
        try:
            return get_provider_embedding_service(self._llm_service)
        except Exception as e:
            logger.info(
                "Provider embedding service unavailable for "
                "conversation_history writes (%s); falling back to legacy "
                "column set.", e,
            )
            return None

    @property
    def embedding_service(self) -> Optional[Any]:
        """Expose the active embedding service (read-only).

        :class:`MemoryRetriever` reads this to embed retrieval queries
        with the SAME model that wrote the row embeddings — using a
        different model would silently destroy cosine similarity.
        Returns ``None`` if the active chat provider doesn't expose
        embeddings or if the opt-out env var is set.
        """
        return self._lazy_embedding_service()

    def _now_sql(self) -> str:
        """Get SQL expression for current timestamp based on backend type."""
        if self.db.backend_type == "postgres":
            return "NOW()"
        return "datetime('now')"

    def _timestamp_query_param(self, value: Any) -> Any:
        """Adapt public timestamps through the shared backend boundary."""
        return timestamp_query_param(self.db.backend_type, value)

    def _canonical_timestamp_sql(self, expression: str) -> str:
        """Normalize one timestamp SQL expression for the active backend."""
        return canonical_timestamp_sql(self.db.backend_type, expression)

    def _timestamp_predicate(self, column: str, operator: str) -> str:
        """Compare timestamps canonically across supported storage formats."""
        return timestamp_predicate(self.db.backend_type, column, operator)

    @property
    def encryption_enabled(self) -> bool:
        """Check if encryption at rest is enabled."""
        return self._agent_fernet is not None or self._global_fernet is not None

    @property
    def _fernet(self):
        """Backward compatibility - return agent fernet or global."""
        return self._agent_fernet or self._global_fernet

    # Session boundary constant: see kestrel_sdk.config.constants
    @property
    def _IMPLICIT_SESSION_GAP_MINUTES(self) -> int:
        from kestrel_sdk.config.constants import SESSION_GAP_MINUTES
        return SESSION_GAP_MINUTES

    async def _derive_implicit_session_id(self) -> Optional[str]:
        """
        Derive an implicit session_id from the time-gap heuristic.

        Returns the previous message's session_id if it was within
        the last 30 minutes; otherwise mints a new UUID for a new
        implicit session.

        This makes the implicit session boundaries already used by
        MemoryConsolidator and wellness metrics observable in metadata,
        so callers that filter by session_id get sensible groupings
        even when no explicit session_id is provided.
        """
        try:
            row = await self.db.fetchone(
                "SELECT metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (self.agent_id,),
            )
            if not row:
                # First message ever — start a new session
                return self._new_session_id()

            prev_metadata_str, prev_created_at = row
            prev_meta = json.loads(prev_metadata_str) if prev_metadata_str else {}

            # If the previous message has no session_id (legacy data), start fresh
            prev_sid = prev_meta.get("session_id")
            if not prev_sid:
                return self._new_session_id()

            # Compare gap through the same canonical parser used by display and
            # destructive session grouping. PostgreSQL returns ``datetime``
            # objects while legacy SQLite rows may use SQL, ISO, Z, or offset
            # strings; all normalize to one naive-UTC arithmetic domain.
            prev_dt = coerce_session_timestamp(prev_created_at)
            if prev_dt is None:
                return self._new_session_id()

            now = datetime.now(timezone.utc).replace(tzinfo=None)
            gap = now - prev_dt
            if gap < timedelta(minutes=self._IMPLICIT_SESSION_GAP_MINUTES):
                return prev_sid
            return self._new_session_id()
        except Exception as e:
            # Never let implicit-session derivation block the write
            logger.warning(f"Implicit session derivation failed: {e}")
            return None

    @staticmethod
    def _new_session_id() -> str:
        """Mint a new implicit session_id (UUID4)."""
        import uuid
        return str(uuid.uuid4())

    async def _canonicalize_session_id(
        self, session_id: Optional[str]
    ) -> Optional[str]:
        """Normalize an incoming session_id to its canonical UUID (#2012).

        The conversation-list endpoint historically keyed each session by the
        row-id of its first message, so the UI would round-trip a bare integer
        (e.g. ``"1314"``) as the session_id on the next turn. Stamping that
        integer onto continued messages splits one conversation across two keys
        (the integer here vs. the UUID on the session's own ``new_session``
        marker) — the messages-gone-on-refresh bug.

        If ``session_id`` is integer-shaped AND names a ``new_session`` marker
        row that genuinely OWNS its UUID ``session_id``, return that UUID so the
        continued turn is filed under the canonical key. A genuinely legacy
        time-gap session (row-id anchor with no marker UUID) is returned
        unchanged. Non-integer ids (already UUIDs) and ``None`` pass through.

        Ownership matters: a legacy ``new_session`` marker written without an
        explicit id could INHERIT the previous still-active session's UUID via
        the time-gap heuristic. Canonicalizing to an inherited UUID would merge
        the new conversation into the old one, so we only canonicalize when no
        earlier row already carries that UUID (#2012).
        """
        if not session_id or not str(session_id).isdigit():
            return session_id

        row_id = coerce_persistent_message_id(session_id)
        if row_id is None:
            return session_id

        try:
            row = await self.db.fetchone(
                "SELECT metadata FROM conversation_history "
                "WHERE id = ? AND agent_id = ?",
                (row_id, self.agent_id),
            )
        except Exception as e:
            logger.warning(f"Session canonicalization lookup failed: {e}")
            return session_id

        if not row or not row[0]:
            return session_id
        try:
            meta = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return session_id

        marker_uuid = meta.get("session_id")
        if (
            meta.get("new_session")
            and marker_uuid
            and not str(marker_uuid).isdigit()
            and await self._marker_owns_uuid(row_id, marker_uuid)
        ):
            logger.info(
                "Canonicalized integer session_id %s -> marker UUID %s",
                session_id, marker_uuid,
            )
            return marker_uuid
        return session_id

    async def _marker_owns_uuid(self, row_id: int, uuid: str) -> bool:
        """True if it is safe to canonicalize an integer key to this marker's
        ``uuid``. Safe requires BOTH:

        1. Exactly ONE ``new_session`` marker carries the UUID. Otherwise the
           inheritance bug stamped it on several markers, each potentially
           anchoring a DISTINCT conversation keyed by its own integer row-id —
           consolidating could merge them.
        2. No EARLIER (id < ``row_id``) content row already carries the UUID.
           Such a row is a prior conversation that owns the UUID directly; the
           marker merely inherited it, so filing new turns under it would merge
           into that prior conversation.

        The live path is only a transition safety net — the startup migration
        does the real, fully-analyzable consolidation of multi-marker UUIDs, so
        here we take the provably-safe position and decline anything ambiguous.
        This marker's OWN later turns (id > row_id) never block (#2012).

        Conservative: on any lookup error, return False (do not canonicalize)
        so an ambiguous marker is never merged into another conversation.
        """
        try:
            esc = _escape_like_session_value(str(uuid))
            rows = await self.db.fetchall(
                "SELECT id, metadata FROM conversation_history "
                "WHERE agent_id = ? "
                "AND (metadata LIKE ? ESCAPE '\\' OR metadata LIKE ? ESCAPE '\\')",
                (
                    self.agent_id,
                    f'%"session_id": "{esc}"%',
                    f'%"session_id":"{esc}"%',
                ),
            )
            marker_count = 0
            for rid, meta_json in rows:
                if not meta_json:
                    continue
                try:
                    meta = json.loads(meta_json)
                except (json.JSONDecodeError, TypeError):
                    return False
                # Exact match (LIKE can substring-collide).
                if meta.get("session_id") != str(uuid):
                    continue
                if meta.get("new_session"):
                    marker_count += 1
                    if marker_count > 1:
                        return False
                elif rid < row_id:
                    # Earlier content owns the UUID → prior conversation.
                    return False
            return marker_count == 1
        except Exception as e:
            logger.warning(f"Marker ownership check failed: {e}")
            return False

    async def get_message_embeddings(
        self,
        message_ids: List[int],
        *,
        embedding_profile_id: Optional[str] = None,
    ) -> Dict[int, List[float]]:
        """Load embeddings for the given message ids.

        Returns ``{id: [v0, v1, …]}`` for every row that has a
        non-NULL ``embedding_vec``. Rows without an embedding are
        absent from the result — caller treats absence as
        "fall back to keyword overlap."

        Args:
            message_ids: Conversation_history row ids to fetch.
            embedding_profile_id: When provided (#1477), only rows
                stamped with this exact profile id are returned —
                rows from a different model living in a different
                semantic coordinate space are filtered out at the
                SQL layer rather than blended into cosine. ``None``
                means "no profile filter" (preserves legacy
                behaviour for pre-migration / mixed-corpus
                deployments).

        Decode handling:

        - SQLite: ``embedding_vec`` is a ``BLOB``; bytes come back
          raw and we ``struct.unpack`` little-endian float32 (same
          shape ``_serialize_embedding`` wrote).
        - Postgres: ``embedding_vec`` is ``vector(N)``. asyncpg
          returns it via pgvector's adapter as a list of floats (or
          a numpy array). We accept either via ``list(value)``.

        Empty input → empty dict (no query). Failure paths log at
        info-level and return whatever we managed to decode —
        retrieval is best-effort, not load-bearing.
        """
        if not message_ids:
            return {}

        # SQLite's default ``SQLITE_MAX_VARIABLE_NUMBER`` is 999 on
        # many builds, including the Python stdlib's bundled sqlite.
        # ``MemoryRetriever.retrieve`` passes up to 1000 ids here, plus
        # one for ``agent_id`` — that would raise ``too many SQL
        # variables`` and silently disable cosine recall for long
        # conversations. Chunk the IN clause well under the limit so
        # the lookup keeps working regardless of slice size. PG's
        # ``$N`` placeholder cap is 32K so this chunk size is
        # comfortable for it too. (Codex P2 on PR-C.)
        CHUNK = 500
        rows: List[Tuple[Any, Any]] = []
        for start in range(0, len(message_ids), CHUNK):
            chunk = message_ids[start:start + CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            profile_clause = ""
            profile_params: Tuple[Any, ...] = ()
            if embedding_profile_id is not None:
                # #1477 — cross-profile rows stay out of cosine. We
                # add the predicate at the SQL layer so the row never
                # round-trips just to be discarded by the caller.
                profile_clause = " AND embedding_profile_id = ?"
                profile_params = (embedding_profile_id,)
            sql = (
                f"SELECT id, embedding_vec FROM conversation_history "
                f"WHERE agent_id = ? AND id IN ({placeholders}) "
                f"AND embedding_vec IS NOT NULL{profile_clause}"
            )
            try:
                rows.extend(
                    await self.db.fetchall(
                        sql, (self.agent_id, *chunk, *profile_params)
                    )
                )
            except Exception as e:
                # Most likely cause: the Phase-2 migration hasn't run
                # yet so the column doesn't exist. Don't crash
                # retrieval — the legacy keyword-overlap path still
                # works. Bail the whole batch rather than partial-
                # decoding what we have; mixed-result rankings are
                # worse than a clean keyword fallback.
                logger.info(
                    "Could not load conversation_history.embedding_vec "
                    "for agent %s (%s); MemoryRetriever falls back to "
                    "keyword overlap.", self.agent_id, e,
                )
                return {}

        out: Dict[int, List[float]] = {}
        for row in rows:
            row_id = row[0]
            raw = row[1]
            if raw is None:
                continue
            try:
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    payload = bytes(raw)
                    if len(payload) % 4 != 0:
                        # Wrong-length BLOB — would unpack to noise.
                        # Skip rather than corrupt the score.
                        logger.warning(
                            "Skipping embedding for message %s: %d bytes "
                            "not a multiple of 4.", row_id, len(payload),
                        )
                        continue
                    floats = struct.unpack(
                        f"<{len(payload) // 4}f", payload
                    )
                    out[row_id] = list(floats)
                elif isinstance(raw, str):
                    # pgvector text shape: ``[v0,v1,...]``. The raw
                    # asyncpg path doesn't register a pgvector codec, so
                    # ``SELECT embedding_vec`` on PG comes back as a
                    # string the asyncpg driver decodes from the wire.
                    # Iterating the string character-by-character (the
                    # original fall-through) would call ``float('[')``
                    # and crash, silently dropping every PG row.
                    # (Caught by codex review on PR-C.)
                    stripped = raw.strip()
                    if not (stripped.startswith("[") and stripped.endswith("]")):
                        logger.warning(
                            "Unexpected pgvector text shape for message "
                            "%s: %r", row_id, stripped[:40],
                        )
                        continue
                    inner = stripped[1:-1].strip()
                    if not inner:
                        continue
                    out[row_id] = [float(p) for p in inner.split(",")]
                else:
                    # pgvector adapter / list-like / numpy: trust
                    # iteration + ``float()`` to produce a flat float
                    # list. ``isinstance(value, list)`` would miss
                    # numpy arrays an adapter may produce.
                    out[row_id] = [float(v) for v in raw]
            except Exception as e:
                logger.warning(
                    "Could not decode embedding for message %s: %s",
                    row_id, e,
                )
                continue
        return out

    async def resolve_session_id(self, provided: Optional[str]) -> Optional[str]:
        """Resolve the effective session_id for an incoming turn.

        If the caller provided one, use it. Otherwise apply the same
        time-gap heuristic ``add_conversation`` uses internally — but
        return the resolved value so the streaming/invoke endpoint can
        echo it back to the client. Without this, the pane's
        ``sessionId`` stays ``null`` forever because the implicit UUID
        derived inside ``add_conversation`` is invisible to the caller.
        """
        if provided:
            return await self._canonicalize_session_id(provided)
        return await self._derive_implicit_session_id()

    async def add_conversation(self, role: str, content: str,
                               metadata: Optional[Dict] = None,
                               session_id: Optional[str] = None,
                               rendered_content: Optional[str] = None,
                               model: Optional[str] = None,
                               provider: Optional[str] = None) -> None:
        """Prepare and persist a conversation message with per-agent encryption."""
        prepared = await self._prepare_conversation_write(
            role,
            content,
            metadata,
            session_id,
            rendered_content,
            model,
            provider,
        )
        await self._persist_prepared_conversation(prepared)

    async def _prepare_conversation_write(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict],
        session_id: Optional[str],
        rendered_content: Optional[str],
        model: Optional[str],
        provider: Optional[str],
    ) -> _PreparedConversationWrite:
        """Do provider/lexical prework before any semantic lifecycle fence.

        The returned object is not visible to readers.  Its final insertion is
        intentionally separate so semantic recall can fence only the brief
        canonical-liveness check plus INSERT, rather than a network embedding
        request or token-index write.
        """
        meta = dict(metadata) if metadata else {}

        # Resolve session_id: explicit wins; otherwise derive from time gap.
        # Canonicalize an explicit id first so a row-id echoed back by an older
        # UI client is re-linked to its session's UUID rather than splitting the
        # conversation across two keys (#2012).
        if session_id:
            session_id = await self._canonicalize_session_id(session_id)
        if not session_id:
            if meta.get('new_session'):
                # A new_session marker anchors a NEW session, so it must MINT
                # and own a fresh UUID — never inherit the previous still-active
                # session's id from the time-gap heuristic. Inheriting it would
                # merge the new conversation into the old one once continued
                # turns canonicalize to the marker's id (#2012).
                session_id = self._new_session_id()
            else:
                session_id = await self._derive_implicit_session_id()

        if session_id:
            meta['session_id'] = session_id

        # Use per-agent key for new messages.
        fernet_to_use = self._agent_fernet or self._global_fernet
        to_store, was_encrypted = encrypt_string(content, fernet_to_use)
        if was_encrypted:
            meta['enc'] = True
            meta['key_version'] = CURRENT_KEY_VERSION

        rendered_to_store: Optional[str] = None
        if rendered_content is not None:
            rendered_to_store, rendered_was_encrypted = encrypt_string(
                rendered_content, fernet_to_use
            )
            # rendered_content shares the same key/version as content; the
            # ``enc`` flag covers both. If content was empty and skipped
            # encryption but rendered_content didn't, surface that.
            if rendered_was_encrypted and not was_encrypted:
                meta['enc'] = True
                meta['key_version'] = CURRENT_KEY_VERSION

        # Compute the embedding from plaintext before the final INSERT so it
        # can still be co-written without an autoincrement-id round trip.
        embedding_vec_val = await self._maybe_embed(content)
        profile_id: Optional[str] = None
        embedding_service: Optional[Any] = None
        if embedding_vec_val is not None:
            embedding_service = self._lazy_embedding_service()
            if embedding_service is not None and hasattr(
                embedding_service, "current_profile_id"
            ):
                try:
                    profile_id = embedding_service.current_profile_id()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "current_profile_id() failed for agent %s: %s",
                        self.agent_id, exc,
                    )

        lexical_index_id: Optional[str] = uuid.uuid4().hex
        lexical_index_version: Optional[str] = self._lexical_index.version
        try:
            # Completion protocol: token rows commit first, then the message
            # row and its coverage marker commit together.  A crash can leave
            # harmless orphan tokens, but never a falsely-covered message.
            await self._lexical_index.index_message(
                lexical_index_id,
                _tokenize_for_search(_strip_search_wrappers(content)),
            )
        except Exception as exc:  # noqa: BLE001 - recall falls back safely
            logger.error(
                "Blind lexical index write failed for agent %s: %s. "
                "Persisting the message on the correctness fallback.",
                self.agent_id, exc,
            )
            lexical_index_id = None
            lexical_index_version = None

        return _PreparedConversationWrite(
            role=role,
            content=to_store,
            rendered_content=rendered_to_store,
            metadata=meta,
            embedding=embedding_vec_val,
            embedding_profile_id=profile_id,
            embedding_service=embedding_service,
            model=model,
            provider=provider,
            lexical_index_id=lexical_index_id,
            lexical_index_version=lexical_index_version,
        )

    async def _exclude_prepared_conversation_from_retrieval(
        self,
        prepared: _PreparedConversationWrite,
    ) -> None:
        """Drop precomputed retrieval residue before an excluded write lands."""
        if prepared.lexical_index_id is not None:
            await self._discard_lexical_tokens(prepared.lexical_index_id)
            prepared.lexical_index_id = None
            prepared.lexical_index_version = None
        # An excluded derivative remains an auditable transcript row but must
        # not retain a vector path while it has no retrievable lexical path.
        prepared.embedding = None
        prepared.embedding_profile_id = None
        prepared.embedding_service = None

    async def _persist_prepared_conversation(
        self,
        prepared: _PreparedConversationWrite,
    ) -> None:
        """Insert precomputed bytes and clean token-first work on failure."""
        try:
            lexical_columns_written = await self._insert_message(
                role=prepared.role,
                content=prepared.content,
                rendered_content=prepared.rendered_content,
                metadata=json.dumps(prepared.metadata) if prepared.metadata else None,
                # Derived from the metadata this same INSERT is about to
                # store, rather than from a variable that may have moved on
                # (#2958). The rule is deliberately narrower than session
                # grouping's, so the column stays NULL for ids grouping still
                # honours: it may be silent, never wrong.
                session_id=column_session_id(prepared.metadata),
                embedding=prepared.embedding,
                embedding_profile_id=prepared.embedding_profile_id,
                model=prepared.model,
                provider=prepared.provider,
                lexical_index_id=prepared.lexical_index_id,
                lexical_index_version=prepared.lexical_index_version,
            )
        except Exception:
            if prepared.lexical_index_id:
                await self._discard_lexical_tokens(prepared.lexical_index_id)
                prepared.lexical_index_id = None
            raise
        if not lexical_columns_written and prepared.lexical_index_id:
            await self._discard_lexical_tokens(prepared.lexical_index_id)
            prepared.lexical_index_id = None

        # Upsert the profile descriptor into the registry table so
        # ``kestrel-sovereign embeddings audit`` can map id →
        # human-readable fields. Best-effort: registry write must never block
        # the already-persisted message.
        if (
            prepared.embedding_profile_id is not None
            and prepared.embedding_service is not None
        ):
            try:
                await _upsert_embedding_profile(
                    self.db,
                    prepared.embedding_service,
                    prepared.embedding_profile_id,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to upsert embedding_profiles row for agent %s: %s",
                    self.agent_id, exc,
                )

    async def _maybe_embed(self, content: str) -> Optional[List[float]]:
        """Compute an embedding for ``content`` if a service is wired.

        Failures + empty content + None service all return ``None``,
        which downstream renders as a NULL ``embedding_vec`` — the
        legacy keyword-overlap retriever path still works for these
        rows. Crucially, a provider / network outage during a chat
        turn must NOT block writing the message.

        Validates the returned embedding's dimension against the
        column dim chosen at boot time. A mismatch (e.g. provider
        switched from Ollama-768 to OpenAI-1536 after migration) is
        treated as "no embedding" — better to fall back to keyword
        overlap than persist data the retriever will silently reject.
        The first mismatch per store instance logs a clear error
        pointing operators at the fix (re-migrate the column).
        (Codex P2 on PR-B.)
        """
        if not content:
            # Empty content (rare — guardrails would normally reject
            # earlier) would produce a zero-norm embedding that the
            # vector backends explicitly short-circuit; skip the
            # embedding service call entirely.
            return None
        service = self._lazy_embedding_service()
        if service is None:
            return None
        try:
            embedding = await service.aembed(content)
        except Exception as e:
            # Embedding generation must NEVER block message persistence.
            # The retriever falls back to keyword overlap for this row.
            logger.warning(
                "Embedding generation failed for agent %s; "
                "row will be searchable via keyword overlap only: %s",
                self.agent_id, e,
            )
            return None
        if not embedding:
            return None

        # Defend against provider/column dim mismatch. The vector
        # column was created at ``CONVERSATION_MESSAGE_EMBEDDING_DIM``
        # at boot; if the active provider returns a different dim
        # (e.g. config drift between agent restart and migration run),
        # writing it would either crash on PG or persist bytes that
        # the retriever can't decode on SQLite. Skip the embedding,
        # log once per store instance, and keep persisting the row.
        from .sqla.conversation_message import (
            CONVERSATION_MESSAGE_EMBEDDING_DIM,
        )
        if len(embedding) != CONVERSATION_MESSAGE_EMBEDDING_DIM:
            if not getattr(self, "_warned_dim_mismatch", False):
                logger.error(
                    "Embedding dim mismatch for agent %s: provider returned "
                    "%d, column is %d. Vector recall disabled for this "
                    "agent until the column is re-migrated at the new dim. "
                    "Set KESTREL_EMBEDDING_DIM=%d + drop "
                    "conversation_history.embedding_vec to re-migrate, OR "
                    "switch the active provider back to one that emits "
                    "%d-dim embeddings.",
                    self.agent_id, len(embedding),
                    CONVERSATION_MESSAGE_EMBEDDING_DIM,
                    len(embedding),
                    CONVERSATION_MESSAGE_EMBEDDING_DIM,
                )
                # Mark on the instance so we don't flood the log on
                # every turn. Operators get one error per agent boot.
                self._warned_dim_mismatch = True
            return None
        return list(embedding)

    async def _discard_lexical_tokens(self, lexical_index_id: str) -> None:
        """Best-effort cleanup for a token-first write whose row did not land."""
        try:
            await self.db.execute(
                "DELETE FROM conversation_lexical_tokens "
                "WHERE agent_id = ? AND lexical_index_id = ?",
                (self.agent_id, lexical_index_id),
            )
        except Exception as exc:  # noqa: BLE001 - original write error wins
            logger.debug("Could not clean orphan lexical tokens: %s", exc)

    async def _insert_message(
        self,
        *,
        role: str,
        content: str,
        rendered_content: Optional[str],
        metadata: Optional[str],
        embedding: Optional[List[float]],
        embedding_profile_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        session_id: Optional[str] = None,
        lexical_index_id: Optional[str] = None,
        lexical_index_version: Optional[str] = None,
    ) -> bool:
        """Persist a conversation row, optionally co-writing
        ``embedding_vec`` and ``embedding_profile_id``.

        Dual SQL paths because the ``embedding_vec`` bind shape is
        dialect-specific:

        - PG: bind a ``[v1,v2,…]`` text literal with ``::vector`` cast
          so asyncpg can hand the value to pgvector. Mirrors
          :meth:`SavedItemsStore._write_embedding_vec`.
        - SQLite (and unknown dialects): bind float32 little-endian
          bytes to the ``BLOB`` column — same shape the
          ``PurePythonBackend`` reads back.

        When ``embedding`` is ``None`` (no service / failure / empty
        content) we fall back to the non-vector column list. The
        model/provider columns stay in that base list so every
        assistant write path can stamp the generator identity even
        when embeddings are unavailable. ``embedding_profile_id`` is
        always written alongside ``embedding_vec`` — they share the
        same row state (#1477).

        ``session_id`` duplicates the id already inside ``metadata`` into its
        own indexed column, or is ``None`` where that id is outside the
        column's contract (#2958). It rides in the base AND legacy column
        lists because its migration raises unless the column exists, so no
        schema this method can reach is without it — unlike the
        lexical/embedding columns, whose migrations are non-fatal and whose
        absence the fallbacks below exist to survive.
        """
        base_cols = (
            "agent_id, role, content, rendered_content, model, provider, "
            "metadata, session_id, lexical_index_id, lexical_index_version, "
            "created_at"
        )
        base_vals_suffix = f", {self._now_sql()}"
        base_params = (
            self.agent_id, role, content, rendered_content, model, provider,
            metadata, session_id, lexical_index_id, lexical_index_version,
        )
        legacy_cols = (
            "agent_id, role, content, rendered_content, model, provider, "
            "metadata, session_id, created_at"
        )
        legacy_params = base_params[:8]
        # Generated, not hand-counted: these lists are spelled into six
        # statements below and a miscount would only surface at runtime.
        base_binds = ", ".join(["?"] * len(base_params))
        legacy_binds = ", ".join(["?"] * len(legacy_params))

        if embedding is None:
            sql = (
                f"INSERT INTO conversation_history ({base_cols}) "
                f"VALUES ({base_binds}{base_vals_suffix})"
            )
            try:
                await self.db.execute_commit(sql, base_params)
                return True
            except Exception as exc:
                if not _is_missing_lexical_column_error(exc):
                    raise
                logger.info(
                    "Lexical-index columns unavailable for agent %s (%s); "
                    "persisting without the blind index.", self.agent_id, exc,
                )
                await self.db.execute_commit(
                    f"INSERT INTO conversation_history ({legacy_cols}) "
                    f"VALUES ({legacy_binds}{base_vals_suffix})",
                    legacy_params,
                )
                return False

        backend_type = getattr(self.db, "backend_type", None)
        try:
            if backend_type == "postgres":
                emb_bind: Any = _format_pgvector_text(embedding)
                emb_placeholder = "?::vector"
            else:
                emb_bind = _serialize_embedding(embedding)
                emb_placeholder = "?"

            # Co-write profile id when we have one; the column may
            # be NULL (pre-#1477 deployments without the migration)
            # so we keep the legacy two-column path as the fallback
            # below.
            sql = (
                f"INSERT INTO conversation_history "
                f"({base_cols}, embedding_vec, embedding_profile_id) "
                f"VALUES ({base_binds}{base_vals_suffix}, "
                f"{emb_placeholder}, ?)"
            )
            await self.db.execute_commit(
                sql, base_params + (emb_bind, embedding_profile_id),
            )
        except Exception as e:
            if _is_missing_lexical_column_error(e):
                # The lexical migration is additive/non-fatal. Preserve the
                # independently-migrated vector + profile stamp before trying
                # older schema shapes; otherwise failed lexical DDL silently
                # removes the row from profile-filtered kNN.
                try:
                    await self.db.execute_commit(
                        f"INSERT INTO conversation_history "
                        f"({legacy_cols}, embedding_vec, embedding_profile_id) "
                        f"VALUES ({legacy_binds}{base_vals_suffix}, "
                        f"{emb_placeholder}, ?)",
                        legacy_params + (emb_bind, embedding_profile_id),
                    )
                    return False
                except Exception as legacy_exc:
                    if not _is_missing_column_error(
                        legacy_exc, "embedding_profile_id", "embedding_vec"
                    ):
                        raise
            # The most informative thing to try first is "drop the
            # NEW column but keep the embedding_vec" — that catches
            # the partial-migration shape where Phase-2 ran (vec
            # column exists) but #1477 hasn't (profile_id column
            # doesn't). Without this middle step we would regress
            # those deployments from storing vectors to dropping
            # them entirely. (Codex P2 on #1477.)
            logger.info(
                "Could not write conversation_history.embedding_vec + "
                "embedding_profile_id for agent %s (%s); retrying "
                "without embedding_profile_id.", self.agent_id, e,
            )
            try:
                sql = (
                    f"INSERT INTO conversation_history "
                    f"({base_cols}, embedding_vec) "
                    f"VALUES ({base_binds}{base_vals_suffix}, "
                    f"{emb_placeholder})"
                )
                await self.db.execute_commit(sql, base_params + (emb_bind,))
                return True
            except Exception as e2:
                logger.info(
                    "New lexical-index INSERT paths failed for agent %s (%s); "
                    "retrying the legacy embedding_vec path.", self.agent_id, e2,
                )
                try:
                    await self.db.execute_commit(
                        f"INSERT INTO conversation_history "
                        f"({legacy_cols}, embedding_vec) "
                        f"VALUES ({legacy_binds}{base_vals_suffix}, "
                        f"{emb_placeholder})",
                        legacy_params + (emb_bind,),
                    )
                    return False
                except Exception as e3:
                    logger.info(
                        "Legacy embedding_vec INSERT also failed for agent %s "
                        "(%s); persisting without an embedding.", self.agent_id, e3,
                    )
                    await self.db.execute_commit(
                        f"INSERT INTO conversation_history ({legacy_cols}) "
                        f"VALUES ({legacy_binds}{base_vals_suffix})",
                        legacy_params,
                    )
                    return False
        return True

    def decrypt_stored_content(self, content: str, meta: Optional[Dict]) -> str:
        """Public: decrypt one stored row's content to plaintext.

        The conversation store owns message decryption. Consumers that read
        ``conversation_history`` with their own SQL (the memory consolidator
        does, for clustering and episode synthesis) previously had no
        supported way to reach it, so they treated the at-rest envelope as
        text — which is how ciphertext ended up tokenized into episode topics
        (#2850). This is that supported way; it deliberately does NOT
        opportunistically migrate, because a read-only consumer must not
        rewrite rows.

        Raises ``DecryptionError`` when the row is marked encrypted and no
        key can open it — callers decide whether to skip or fail.
        """
        plaintext, _needs_migration = self._decrypt_with_fallback(content, meta)
        return plaintext

    def _decrypt_with_fallback(self, content: str, meta: Optional[Dict]) -> tuple[str, bool]:
        """Decrypt content, trying per-agent key first then global.

        Returns:
            Tuple of (decrypted_content, needs_migration)

        Raises:
            DecryptionError: If content is marked as encrypted but all
                           decryption attempts fail (wrong key)
        """
        if not meta or not meta.get("enc"):
            return content, False

        key_version = meta.get("key_version", 0)
        last_error: Optional[DecryptionError] = None

        # Version 1+: use per-agent key directly
        if key_version >= 1 and self._agent_fernet:
            try:
                return decrypt_string(content, meta, self._agent_fernet), False
            except DecryptionError as e:
                last_error = e  # Fall through to global key

        # Version 0 or fallback: try global key
        if self._global_fernet:
            try:
                decrypted = decrypt_string(content, meta, self._global_fernet)
                # If we decrypted with global key but have agent key, needs migration
                needs_migration = (
                    key_version == 0 and
                    self._agent_fernet is not None and
                    decrypted != content
                )
                return decrypted, needs_migration
            except DecryptionError as e:
                last_error = e

        # Last resort: try agent key even for old version (maybe re-encrypted)
        if self._agent_fernet:
            try:
                return decrypt_string(content, meta, self._agent_fernet), False
            except DecryptionError as e:
                last_error = e

        # All decryption attempts failed - raise error
        error_msg = f"Failed to decrypt message for agent {self.agent_id}"
        logger.error(error_msg)
        if last_error:
            raise DecryptionError(error_msg) from last_error
        else:
            raise DecryptionError(f"{error_msg}: No encryption keys available")

    async def _migrate_message(
        self, row_id: int, decrypted_content: str, meta: Optional[Dict] = None
    ) -> None:
        """Re-encrypt a message with per-agent key.

        Preserves existing row metadata (``session_id``, ``sent_form``,
        ``excluded_from_context``, ``privacy_mode``, …); only ``enc`` and
        ``key_version`` are added/updated, mirroring the one-shot backfill in
        ``security/encryption_backfill.py``.
        """
        if not self._agent_fernet:
            return

        new_content, _ = encrypt_string(decrypted_content, self._agent_fernet)
        new_meta = {**(meta or {}), "enc": True, "key_version": CURRENT_KEY_VERSION}

        await self.db.execute_commit(
            "UPDATE conversation_history SET content = ?, metadata = ? WHERE id = ?",
            (new_content, json.dumps(new_meta), row_id)
        )
        logger.debug(f"Migrated message {row_id} to per-agent encryption")

    async def _resolve_canonical(
        self,
        row_id: int,
        role: str,
        meta: Optional[Dict],
        content: str,
        rendered_raw: Optional[str],
        migrate: bool = True,
    ) -> tuple[str, Optional[str]]:
        """Apply the canonical/transport split (#1402) for a single row.

        ``migrate=False`` withholds the opportunistic backfill below without
        changing what is returned. Search passes it — see
        :meth:`_normalized_search_row` for why a scan must not write.

        Returns ``(canonical_content, rendered_content)`` where
        ``canonical_content`` is the raw user turn (or original content for
        non-user rows) and ``rendered_content`` is the byte-stable
        transport form for sent_form user turns (or ``None``).

        On legacy ``sent_form`` rows (``rendered_raw is None``) this splits
        the rendered bytes into the new shape in-memory and triggers an
        opportunistic DB UPDATE to persist the split (gated on
        ``_migrate_on_read``). Already-split rows pass through unchanged.

        Handles decryption of ``rendered_raw`` via the same fernet
        fallback path as ``content``. Decryption failures hard-fail (not
        silenced) because a sent_form row we can't decrypt would silently
        regress cache stability.
        """
        rendered_content: Optional[str] = None
        if rendered_raw is not None:
            try:
                rendered_content, _ = self._decrypt_with_fallback(rendered_raw, meta)
            except DecryptionError as e:
                logger.error(
                    "Failed to decrypt rendered_content for message %s: %s",
                    row_id, e,
                )
                raise

        if (
            role == 'user'
            and meta
            and meta.get('sent_form')
            and rendered_content is None
        ):
            # Legacy sent_form row: content currently holds the rendered
            # form. Move it into rendered_content and strip wrappers.
            rendered_content = content
            content = extract_raw_user_content(content)
            if migrate and self._migrate_on_read:
                try:
                    await self._migrate_split_sent_form(
                        row_id, content, rendered_content, meta
                    )
                except Exception as e:
                    logger.warning(
                        "Split-migration failed for message %s: %s",
                        row_id, e,
                    )

        return content, rendered_content

    async def _migrate_split_sent_form(
        self,
        row_id: int,
        raw_content: str,
        rendered_content: str,
        meta: Optional[Dict],
    ) -> None:
        """Backfill the canonical/transport split (#1402) for a legacy row.

        Legacy ``sent_form=True`` rows store the rendered transport form in
        ``content`` with ``rendered_content IS NULL``. Move the original
        bytes into ``rendered_content`` (preserving them byte-for-byte so
        the cache prefix continues to hit through deploy) and replace
        ``content`` with the stripped raw user turn. The shared ``enc``
        flag/key_version cover both columns.

        Idempotent — already-split rows have ``rendered_content IS NOT
        NULL`` and never reach this method.
        """
        fernet_to_use = self._agent_fernet or self._global_fernet
        new_content, was_encrypted_c = encrypt_string(raw_content, fernet_to_use)
        new_rendered, was_encrypted_r = encrypt_string(
            rendered_content, fernet_to_use
        )

        # Preserve everything else in metadata; just refresh the encryption
        # marker if a fernet was available so legacy plaintext rows pick up
        # encryption-at-rest for both columns at the same time.
        new_meta = dict(meta) if meta else {}
        if was_encrypted_c or was_encrypted_r:
            new_meta['enc'] = True
            new_meta['key_version'] = CURRENT_KEY_VERSION

        await self.db.execute_commit(
            "UPDATE conversation_history "
            "SET content = ?, rendered_content = ?, metadata = ? WHERE id = ?",
            (new_content, new_rendered, json.dumps(new_meta), row_id)
        )
        logger.debug(
            "Split-migrated sent_form message %s into canonical/transport columns",
            row_id,
        )

    async def get_conversation_history(
        self, limit: int = 100, session_id: str = None
    ) -> List[Dict[str, Any]]:
        """Get recent conversation history with automatic decryption and migration.

        Args:
            limit: Maximum number of messages to return
            session_id: If provided, get messages from this session only (using time-based grouping)
        """
        if session_id:
            # Session-aware retrieval: get messages from the specified session
            rows = await self._get_session_messages(
                session_id, limit, include_archived=False
            )
        else:
            # Default behavior: get most recent live messages.
            # rendered_content (#1402) appended at row[5] so existing
            # positional accesses on metadata/created_at don't shift.
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at, rendered_content, "
                "model, provider "
                "FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL AND archived_at IS NULL "
                "ORDER BY id DESC LIMIT ?",
                (self.agent_id, limit)
            )
        history = []
        for row in reversed(rows):  # Return in chronological order
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else None

            # Skip messages excluded from context (compacted, summarized, etc.)
            if meta and meta.get("excluded_from_context"):
                continue

            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Opportunistic migration to per-agent key
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content, meta)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id}: {e}")

            # rendered_content (#1402): decrypt + apply canonical/transport
            # split (legacy ``sent_form`` rows are split in-memory and
            # opportunistically migrated). Defensive ``len(row) > 5``
            # covers callers that pass legacy 5-tuple rows (e.g. soft-
            # delete restore paths that haven't been migrated to the new
            # SELECT shape yet).
            rendered_raw = row[5] if len(row) > 5 else None
            content, rendered_content = await self._resolve_canonical(
                row_id, row[1], meta, content, rendered_raw
            )

            entry = {
                'id': row_id,
                'role': row[1],
                'content': content,
                'created_at': row[4]
            }
            if len(row) > 6:
                entry['model'] = row[6]
            if len(row) > 7:
                entry['provider'] = row[7]
            if rendered_content is not None:
                entry['rendered_content'] = rendered_content
            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                # Remove internal key_version from external metadata
                cleaned_meta.pop('key_version', None)
                if cleaned_meta:
                    entry['metadata'] = cleaned_meta
            history.append(entry)
        return history

    @staticmethod
    def _deleted_filter_clause(deleted_filter: str) -> str:
        """Return the SQL fragment that filters by deleted_at state.

        ``live``    → only ``deleted_at IS NULL`` (default for reads).
        ``deleted`` → only ``deleted_at IS NOT NULL`` (Trash view, restore).
        ``all``     → no filter (purge needs every row, regardless of state).

        Returns the leading ``AND`` so it can be appended to a WHERE
        clause that already has at least one condition. ``all`` returns
        an empty string.
        """
        if deleted_filter == "live":
            return " AND deleted_at IS NULL"
        if deleted_filter == "deleted":
            return " AND deleted_at IS NOT NULL"
        if deleted_filter == "all":
            return ""
        raise ValueError(
            f"Invalid deleted_filter={deleted_filter!r}; "
            "expected 'live', 'deleted', or 'all'"
        )

    async def get_session_message_rows(
        self, session_id: str, limit: int = 100
    ) -> List[tuple]:
        """Public entry point to the canonical dual-scheme session resolver.

        Thin wrapper over :meth:`_get_session_messages` so callers outside
        this module (the ``AsyncStorage`` facade, the privacy wrapper, the
        conversation endpoints) can resolve a session by either a UUID or a
        legacy row-id without reaching into a private method (#2012).
        """
        return await self._get_session_messages(
            session_id, limit, include_archived=False
        )

    @staticmethod
    def _filter_session_rows(
        rows: Sequence[tuple],
        session_id: str,
        *,
        limit: Optional[int],
        include_markers: bool,
        metadata_index: int = 3,
        created_at_index: int = 4,
        reject_invalid_timestamps: bool = False,
    ) -> List[tuple]:
        """Apply the canonical time-gap/resumption rules to candidate rows.

        Display reads and exact lifecycle snapshots deliberately share this
        one filter.  The exact-ID path selects only ``id``/``metadata``/
        ``created_at`` and supplies the corresponding indexes so unbounded
        privacy purges do not materialize encrypted message bodies merely to
        resolve membership.
        """
        from kestrel_sovereign.kestrel_config.constants import SESSION_GAP_MINUTES

        seen_ids: set[int] = set()
        candidates: list[tuple[datetime, int, tuple, dict[str, Any]]] = []
        # NOT a clock. An unreadable timestamp dated with `now()` sorts to the
        # end of the candidates and is gap-filtered out of the session the
        # conversation list says it belongs to — and it does so differently
        # depending on when the read happens, so the same session opens with
        # different contents on two consecutive requests. `group_messages_into_
        # sessions` stopped consulting a clock for #2959 (a grouping that reads
        # one cannot be cached); this is the same constant it falls back to.
        #
        # This does not make the two agree about which session an unreadable row
        # joins: the grouper inherits the row's predecessor, and the candidates
        # here are a subset that need not contain it. That is one rule with two
        # implementations, which is #2961's subject.
        fallback_timestamp = UNDATABLE_ROW_FALLBACK
        session_id_str = str(session_id)
        for row in rows:
            row_id = int(row[0])
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            metadata_json = row[metadata_index]
            meta: dict[str, Any] = {}
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                except json.JSONDecodeError as error:
                    logger.warning(
                        "Failed to parse metadata for message in session %s: %s",
                        session_id,
                        error,
                    )
            timestamp = coerce_session_timestamp(row[created_at_index])
            if timestamp is None:
                # Fail closed only when membership would depend on gap
                # chronology. A row whose metadata.session_id matches the
                # requested session EXACTLY is a proven member regardless of
                # its timestamp — refusing it would leave explicit
                # (UUID/resumed) sessions containing one malformed
                # created_at undeletable through count/guard/purge.
                if (
                    reject_invalid_timestamps
                    and meta.get("session_id") != session_id_str
                ):
                    raise ConversationSessionTimestampError(
                        "Refusing exact conversation-session resolution because "
                        f"message {row_id} has an invalid created_at timestamp"
                    )
                timestamp = fallback_timestamp
            candidates.append(
                (timestamp, row_id, row, meta)
            )
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))

        session_rows = []
        last_timestamp: Optional[datetime] = None
        is_first = True
        # Whether the requested session's implicit run is still taking rows.
        #
        # A row filed under a DIFFERENT canonical id ends it, and ending it is
        # not the same as skipping the row: an unlabeled row after it inherits
        # THAT session under ``group_messages_into_sessions``, so it must not
        # be able to rejoin this one by merely falling inside the gap. Only a
        # row carrying the requested id re-opens the run — that is a resumption,
        # and the grouper coalesces those back together by id.
        #
        # Until #3098 this walk did not stop, so a legacy numeric cluster
        # absorbed a following stamped row. The grouper had to absorb it too,
        # or deleting the cluster the list showed would have destroyed the
        # stamped session beside it — and that absorption is what made Phase
        # A's per-row column disagree with the grouping and drop the whole
        # conversation from the list.
        run_open = True

        for timestamp, _row_id, row, meta in candidates:
            is_resumed_message = meta.get("session_id") == session_id_str

            if not is_first and not is_resumed_message and meta.get("new_session"):
                break

            if is_resumed_message:
                run_open = True
            elif not is_first and canonical_session_id(meta) not in (
                None,
                session_id_str,
            ):
                run_open = False
            if not run_open:
                # Deliberately WITHOUT advancing ``last_timestamp``: nothing
                # else can join this session until a resumption re-opens it,
                # and a resumption is admitted regardless of the gap.
                continue

            if last_timestamp and not is_resumed_message:
                gap_minutes = (
                    timestamp - last_timestamp
                ).total_seconds() / 60
                if gap_minutes > SESSION_GAP_MINUTES:
                    # It belongs to a later implicit session. Keep scanning for
                    # explicit resumptions of the requested session.
                    continue

            if not include_markers and meta.get("type") == "session_marker":
                last_timestamp = timestamp
                is_first = False
                continue

            session_rows.append(row)
            last_timestamp = timestamp
            is_first = False

            if limit is not None and len(session_rows) >= limit:
                break

        # Match the historical newest-first raw-row contract.
        return list(reversed(session_rows))

    async def _get_session_messages(
        self,
        session_id: str,
        limit: int,
        deleted_filter: str = "live",
        include_markers: bool = False,
        include_archived: bool = True,
    ) -> List[tuple]:
        """Get messages belonging to a specific session.

        Sessions are determined by:
        1. Time-based grouping (30-minute gaps end a session)
        2. Explicit session_id in metadata (for resumed conversations)

        The session_id is the message ID that starts that session.

        Args:
            session_id: The message ID that marks the session start
            limit: Maximum messages to return
            deleted_filter: ``live`` (default — for reads),
                ``deleted`` (for restore / Trash view), or
                ``all`` (for purge — finds rows in any state).
            include_markers: When True, include the session's ``new_session``
                marker row(s) in the result. Reads/display keep the default
                (markers are structural, not displayable), but session
                LIFECYCLE ops (delete/restore/purge) pass True so the marker —
                the session's live anchor — is acted on too. Otherwise deleting
                a session's content leaves a live orphan marker that keeps the
                session in the active list yet unresolvable by a later delete
                (#2027).
            include_archived: Include rows with ``archived_at`` set. Lifecycle
                operations keep the default so archived sessions can be
                unarchived/purged; normal conversation replay passes False.

        Returns:
            List of raw rows
            (id, role, content, metadata, created_at, rendered_content,
            model, provider) — rendered_content/model/provider are appended
            so existing positional accesses below don't shift.
        """
        del_clause = self._deleted_filter_clause(deleted_filter)
        archive_clause = "" if include_archived else " AND archived_at IS NULL"

        # Try to interpret session_id as a message ID for time-based grouping.
        # If it isn't (e.g. a UUID-based implicit session_id), skip this path
        # and fall through to the metadata-based lookup below.
        all_rows = []
        row_id = coerce_persistent_message_id(session_id)
        if row_id is not None:
            # The anchor row itself is looked up regardless of state — we
            # need its timestamp even if it's been soft-deleted, otherwise
            # we can't restore the session that owned it.
            start_row = await self.db.fetchone(
                "SELECT created_at FROM conversation_history WHERE id = ? AND agent_id = ?",
                (row_id, self.agent_id)
            )

            # If session_id is a message ID, get messages from that timestamp forward
            # rendered_content (#1402) appended at row[5] so existing
            # positional accesses on metadata/created_at don't shift.
            if start_row:
                start_time = start_row[0]
                # Canonicalize the timestamp prefilter/order: SQLite history
                # mixes ``YYYY-MM-DD HH:MM:SS`` and ISO-8601 ``T`` forms, and
                # raw TEXT comparison drops later rows whose stored form sorts
                # below the anchor's.  The exact purge/count resolver already
                # compares via julianday; display must see the same membership
                # or hard purge could destroy rows this path never returned.
                created_at_predicate = self._timestamp_predicate(
                    "created_at", ">="
                )
                created_at_order = self._canonical_timestamp_sql("created_at")
                all_rows = await self.db.fetchall(
                    f"""SELECT id, role, content, metadata, created_at,
                              rendered_content, model, provider
                       FROM conversation_history
                       WHERE agent_id = ? AND {created_at_predicate}{del_clause}{archive_clause}
                       ORDER BY {created_at_order} ASC
                       LIMIT ?""",
                    (
                        self.agent_id,
                        self._timestamp_query_param(start_time),
                        limit * 2,  # Fetch extra in case of filtering
                    ),
                )

        # Also get messages that explicitly belong to this session (resumed conversations)
        # These are messages with session_id in metadata that may come after a time gap.
        # Escape LIKE wildcards so a `%`/`_` in session_id can't broaden the match
        # to EVERY row (#1729); ESCAPE '\' makes the backslash the escape char. For
        # an ordinary UUID this is a no-op.
        esc = _escape_like_session_value(session_id)
        resumed_rows = await self.db.fetchall(
            f"""SELECT id, role, content, metadata, created_at, rendered_content,
                      model, provider
               FROM conversation_history
               WHERE agent_id = ? AND metadata LIKE ? ESCAPE '\\'{del_clause}{archive_clause}
               ORDER BY created_at ASC
               LIMIT ?""",
            (self.agent_id, f'%"session_id": "{esc}"%', limit)
        )

        # Also try without space after colon (JSON formatting varies)
        resumed_rows_alt = await self.db.fetchall(
            f"""SELECT id, role, content, metadata, created_at, rendered_content,
                      model, provider
               FROM conversation_history
               WHERE agent_id = ? AND metadata LIKE ? ESCAPE '\\'{del_clause}{archive_clause}
               ORDER BY created_at ASC
               LIMIT ?""",
            (self.agent_id, f'%"session_id":"{esc}"%', limit)
        )

        return self._filter_session_rows(
            [*all_rows, *resumed_rows, *resumed_rows_alt],
            session_id,
            limit=limit,
            include_markers=include_markers,
        )

    async def _get_complete_session_message_ids(
        self,
        session_id: str,
        *,
        deleted_filter: str = "all",
        include_markers: bool = True,
        include_archived: bool = True,
    ) -> list[int]:
        """Resolve one uncapped session-membership snapshot in one statement.

        A one-statement candidate read gives PostgreSQL one READ COMMITTED
        snapshot; splitting the spaced/minified metadata forms across separate
        queries could otherwise admit an insert between them.  Hard purge calls
        this inside its destructive transaction, then locks the immutable ID
        set in bounded ascending batches.  SQLite reserves its writer slot
        before this query, so its snapshot cannot race a concurrent writer.

        Only fields needed by the shared grouping algorithm are materialized;
        encrypted message bodies remain untouched even for very large sessions.
        """
        del_clause = self._deleted_filter_clause(deleted_filter).replace(
            "deleted_at", "c.deleted_at"
        )
        archive_clause = (
            "" if include_archived else " AND c.archived_at IS NULL"
        )
        escaped_session_id = _escape_like_session_value(session_id)
        spaced_pattern = f'%"session_id": "{escaped_session_id}"%'
        compact_pattern = f'%"session_id":"{escaped_session_id}"%'
        row_id = coerce_persistent_message_id(session_id)

        if row_id is None:
            query_prefix = ""
            candidate_source = "conversation_history c"
            membership_predicate = (
                "(c.metadata LIKE ? ESCAPE '\\' "
                "OR c.metadata LIKE ? ESCAPE '\\')"
            )
            params: tuple[Any, ...] = (
                self.agent_id,
                spaced_pattern,
                compact_pattern,
            )
        else:
            query_prefix = (
                "WITH anchor AS ("
                "SELECT created_at FROM conversation_history "
                "WHERE id = ? AND agent_id = ?"
                ") "
            )
            # LEFT JOIN (not CROSS JOIN): a numeric session id can be
            # metadata-only — the client supplied it explicitly, or the legacy
            # anchor row was already hard-deleted.  An empty anchor CTE must
            # drop only the time-grouping branch, never the metadata branch,
            # or purge/count would miss rows the display resolver still finds.
            candidate_source = "conversation_history c LEFT JOIN anchor ON 1=1"
            candidate_timestamp = self._canonical_timestamp_sql("c.created_at")
            anchor_timestamp = self._canonical_timestamp_sql("anchor.created_at")
            membership_predicate = (
                "(EXISTS (SELECT 1 FROM anchor) "
                f"AND ({candidate_timestamp} >= {anchor_timestamp} "
                f"OR {candidate_timestamp} IS NULL "
                f"OR {anchor_timestamp} IS NULL) "
                "OR c.metadata LIKE ? ESCAPE '\\' "
                "OR c.metadata LIKE ? ESCAPE '\\')"
            )
            params = (
                row_id,
                self.agent_id,
                self.agent_id,
                spaced_pattern,
                compact_pattern,
            )

        candidates = await self.db.fetchall(
            f"{query_prefix}SELECT c.id, c.metadata, c.created_at "
            f"FROM {candidate_source} WHERE c.agent_id = ? AND "
            f"{membership_predicate}{del_clause}{archive_clause} "
            "ORDER BY c.id ASC",
            params,
        )
        session_rows = self._filter_session_rows(
            candidates,
            session_id,
            limit=None,
            include_markers=include_markers,
            metadata_index=1,
            created_at_index=2,
            reject_invalid_timestamps=True,
        )
        return sorted({int(row[0]) for row in session_rows})

    async def get_full_history(self) -> List[Dict[str, Any]]:
        """Get complete live conversation history with automatic decryption.

        Soft-deleted and archived rows are filtered out — use
        ``get_full_history_with_ids(include_deleted=True)`` if you need
        to see Trash too.
        """
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata, rendered_content "
            "FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL AND archived_at IS NULL "
            "ORDER BY id ASC",
            (self.agent_id,)
        )
        history = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else None
            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Opportunistic per-agent key migration
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content, meta)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id} in get_full_history: {e}")

            # Canonical/transport split (#1402)
            content, rendered_content = await self._resolve_canonical(
                row_id, row[1], meta, content, row[4] if len(row) > 4 else None
            )

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            entry = {
                'role': row[1],
                'content': content,
                'metadata': cleaned_meta if cleaned_meta else None
            }
            if rendered_content is not None:
                entry['rendered_content'] = rendered_content
            history.append(entry)
        return history

    async def search_history(
        self,
        query: str,
        limit: int = 20,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search conversation history.

        Fetches and decrypts messages, then filters client-side.
        This approach works correctly with encrypted storage.

        Args:
            query: Substring to search for (case-insensitive).
            limit: Maximum results to return.
            session_id: If provided, restrict search to messages tagged
                with this session_id in metadata. Useful for "what did
                we discuss in this session" queries.
        """
        # SQL pre-filter when session_id is provided. We can match against
        # the metadata JSON because session_id is plaintext (not encrypted).
        # Falls back to full scan when no session_id is given.
        if session_id:
            # Match both `"session_id": "X"` and `"session_id":"X"` formats.
            # The stored metadata is JSON, so the value appears JSON-escaped
            # (e.g. a literal backslash is stored as `\\`); match against that
            # exact form via json.dumps, then escape LIKE wildcards so a
            # `%`/`_`/`\` in the id can't broaden the match. ESCAPE '\' makes
            # the backslash the LIKE escape char (#1653). Order matters:
            # JSON-escape first, then LIKE-escape (backslash first so the
            # wildcards added after aren't doubled). For an ordinary UUID
            # session id both passes are no-ops, so the common case is
            # unchanged.
            esc = _escape_like_session_value(session_id)
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, rendered_content "
                "FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL AND archived_at IS NULL "
                "AND (metadata LIKE ? ESCAPE '\\' OR metadata LIKE ? ESCAPE '\\') "
                "ORDER BY id DESC LIMIT 5000",
                (
                    self.agent_id,
                    f'%"session_id": "{esc}"%',
                    f'%"session_id":"{esc}"%',
                ),
            )
        else:
            # Fetch all live messages (up to 5000) and search client-side after decryption
            # SQL LIKE doesn't work on encrypted content, so we must decrypt first
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, rendered_content "
                "FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL AND archived_at IS NULL "
                "ORDER BY id DESC LIMIT 5000",
                (self.agent_id,)
            )

        query_lower = query.lower()
        query_tokens = _tokenize_for_search(query)

        # Query-side wrapper projection (#1554). #1550 stripped transport
        # wrappers from candidate CONTENT before matching, but a
        # wrapper-only query string (e.g. ``--- END MEMORIES ---`` or
        # ``RELEVANT MEMORIES from past conversations``) was still treated
        # as an ordinary tokenized search. Once candidate content is
        # wrapper-stripped, ordinary canonical words like ``memory`` /
        # ``memories`` / ``end`` still satisfy the 0.6 tokenized threshold,
        # so a query that is itself pure retrieved-context transport syntax
        # matched unrelated rows. Project the query through the same
        # stripper: if no meaningful tokens survive, the query is pure
        # transport syntax and must not drive the tokenized fallback at
        # all. The exact-substring path below runs against the
        # wrapper-stripped candidate content, so such a query can only hit
        # when the literal string survives stripping in genuine canonical
        # user/assistant text (not baked-in transport).
        stripped_query_tokens = _tokenize_for_search(_strip_search_wrappers(query))
        query_is_wrapper_only = bool(query_tokens) and not stripped_query_tokens
        use_token_fallback = (
            not query_is_wrapper_only and len(stripped_query_tokens) >= 2
        )

        exact_results = []
        # Candidates for the tokenized fallback: (score, dict)
        token_candidates: List[tuple] = []

        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else None
            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Opportunistic per-agent key migration
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content, meta)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id} in search_history: {e}")

            # Canonical/transport split (#1402): search must match against
            # the raw user turn, not the rendered transport bytes — a
            # search for what the user said should not false-match against
            # stamped <retrieved_context> from memories/RAG.
            content, _ = await self._resolve_canonical(
                row_id, row[1], meta, content, row[4] if len(row) > 4 else None
            )

            # Match against a wrapper-stripped projection (#1537): even
            # after the canonical/transport split, a row whose ``sent_form``
            # flag is missing (or whose anchored grammar didn't fully
            # strip) can still carry generated <retrieved_context> /
            # <memories> / <documents> blocks and the "RELEVANT MEMORIES"
            # heading. Those are transport, not conversation content —
            # matching on them would resurface the #1500 false positive
            # where wrapper-only query terms make an unrelated row a hit.
            # The returned ``content`` stays canonical; only the matching
            # text is stripped.
            content_lower = _strip_search_wrappers(content).lower()

            cleaned_meta = None  # lazily computed

            def _make_entry():
                nonlocal cleaned_meta
                if cleaned_meta is None:
                    cleaned_meta = remove_enc_flag(meta)
                    if cleaned_meta:
                        cleaned_meta.pop('key_version', None)
                return {
                    'role': row[1],
                    'content': content,
                    'metadata': cleaned_meta if cleaned_meta else None,
                }

            # Exact substring match (original behaviour)
            if query_lower in content_lower:
                exact_results.append(_make_entry())
                if len(exact_results) >= limit:
                    break
                continue

            # Tokenized fallback: score by fraction of query terms present.
            # Scored against the wrapper-stripped query tokens (#1554) so a
            # pure-transport query can never reach this path — it is gated
            # out above via ``query_is_wrapper_only``.
            if use_token_fallback and len(exact_results) < limit:
                score = _token_match_score(stripped_query_tokens, content_lower)
                if score >= _TOKEN_MATCH_THRESHOLD:
                    token_candidates.append((score, _make_entry()))

        # If exact matches filled the limit, return them directly.
        if len(exact_results) >= limit:
            return exact_results

        # Merge: exact matches first, then token-fallback candidates ranked
        # by descending score, up to the requested limit.
        remaining = limit - len(exact_results)
        if token_candidates:
            token_candidates.sort(key=lambda pair: pair[0], reverse=True)
            for _score, entry in token_candidates[:remaining]:
                exact_results.append(entry)

        return exact_results

    #: How many sessions one step of a search walk takes off the table.
    #:
    #: Not the caller's ``limit``: the walk stops when it has that many
    #: *matching* sessions, and how many it must look at to find them is a
    #: property of the query rather than of the request. This is only how far
    #: ahead it reads while looking — big enough that an ordinary query is
    #: answered from one step, small enough that a query matching nothing does
    #: not have to materialize the whole table to discover that.
    #:
    #: It is not a horizon. What stood here was: ``SEARCH_SESSIONS_SCAN_LIMIT``
    #: bounded the *rows* a search could see at 5,000, so a conversation whose
    #: messages had all aged past that was findable by no query at any limit —
    #: the same defect #2960 removed from the list, one control across. The walk
    #: below has no such bound; it stops on an answer, not on a row count.
    SEARCH_SESSION_STEP = 200

    #: How many rows one content read pulls at a time.
    #:
    #: Matching has to decrypt, so the rows a search examines are held in memory
    #: while it does. This bounds that to a batch rather than to "every row of
    #: every session on this step", which nothing bounds — one conversation may
    #: be arbitrarily long.
    SEARCH_CONTENT_BATCH = 500

    async def search_sessions(
        self,
        query: str,
        view: str = "active",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Full-text search across this agent's conversations (#2019, #2961).

        Backs the conversations pane's server-side search. Matching necessarily
        decrypts client-side — SQL ``LIKE`` cannot see encrypted content, the
        same constraint ``search_history`` works under — and resolves canonical
        content (#1402) so a hit is against the raw user turn rather than the
        transport bytes it was replayed as.

        Returns newest-first session summaries, at most ``limit`` of them,
        decorated with ``match_count`` / ``match_role`` / ``match_snippet``.

        **Search is the list, filtered.** For the active view the sessions come
        out of the #2959 projection in the order the list pages them, shaped by
        the same function — so a session that search finds is one paging
        reaches, and the counts and previews beside it are the list's own rather
        than a second derivation's. Before this they were two derivations with
        two different horizons, 1,000 rows and 5,000, neither a retention policy
        anyone chose; a session outside either was simply absent from that
        surface with nothing to say so.

        The archived view has no table yet (#3062), so it still derives its
        sessions by grouping — over *every* archived row, with no cap, exactly
        as the archived list does.
        """
        bounded = max(1, int(limit))
        if view == "archived":
            return await self._search_archived_sessions(query, bounded)
        terms = SearchTerms.compile(query)
        if terms is None:
            return []
        return await self._search_active_sessions(terms, bounded)

    async def _search_archived_sessions(
        self, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Search the archived view, by grouping — the membership with no table.

        No row cap, and that is the point: the archived list reads its rows
        uncapped for the same reason (#2960), so capping here would put a
        conversation in the archive list that no query could find. The cost is
        proportional to what the user chose to archive, which is a set they
        control, rather than to history, which grows on its own.
        """
        from .conversation_sessions import archived_history_predicate, canonical_order

        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata, created_at, rendered_content "
            "FROM conversation_history "
            f"WHERE agent_id = ? AND {archived_history_predicate()} "
            f"{canonical_order(self.db.backend_type)}",
            (self.agent_id,),
        )
        normalized = [await self._normalized_search_row(row) for row in rows]
        return search_session_summaries(
            normalized, query, names=await self._search_names(), limit=limit
        )

    async def _search_active_sessions(
        self, terms: SearchTerms, limit: int
    ) -> List[Dict[str, Any]]:
        """Search the active view by walking the sessions table (#2961).

        Three reads, each answering the part it is the authority on:

        * the **projection**, paged newest-first exactly as the list pages it,
          which decides both which sessions exist and what order they come in;
        * the **membership map**, which says which rows each of those sessions
          is made of — the grouper's answer, because 473 of Emma's 1,522 live
          rows carry no ``session_id`` and only gap arithmetic can place them;
        * the **rows themselves**, decrypted, but only for the sessions the walk
          actually reaches.

        **What it costs, measured rather than hoped.** The decryption is
        proportional to the answer — the walk stops at ``limit`` matches, so an
        ordinary query reads a step's worth of rows rather than 5,000 — but the
        map is not: it is the whole live history, every time, at about 5.4
        microseconds a row on SQLite (measured: 81 ms at 15,000 live rows,
        323 ms at 60,000, 671 ms at 120,000 — linear).

        So this is NOT cheaper than the 5,000-row scan it replaces on a large
        history, and saying otherwise would be the kind of claim this epic keeps
        finding underneath its bugs. That scan was cheap because it was bounded,
        and bounded is what made 34% of a real agent's conversations unfindable.
        The cheapest complete alternative — scan every row WITH its content and
        group once — costs strictly more than this, because it adds the content
        and the decryption to the same pass. #3075 is what would make the map
        proportional to the page instead of to history.

        The walk is retried, not resumed, when the projection moves under it.
        See :meth:`_walk_for_matches` for why a search cannot inherit the list's
        tolerance for that.
        """
        from .conversation_sessions import ConversationSessionProjection

        projection = ConversationSessionProjection(self.db, self.agent_id)
        names = await self._search_names()
        for _ in range(_PAGE_ATTEMPTS):
            walked = await self._walk_for_matches(projection, terms, names, limit)
            if walked is not None:
                return walked
        raise ProjectionNotReady(
            f"{self.agent_id}'s conversation index was rebuilt underneath "
            f"{_PAGE_ATTEMPTS} attempts to search it"
        )

    async def _walk_for_matches(
        self,
        projection,
        terms: SearchTerms,
        names: Dict[str, str],
        limit: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """One walk of the table, or ``None`` if the projection moved under it.

        **A walk of several pages has to be a walk of ONE projection.** The list
        tolerates a repair between its pages and says so: a session whose
        activity moves across the cursor mid-walk can be shown twice or not at
        all, which is a visible oddity in a list the user is scrolling. Search
        cannot inherit that tolerance, because the same event produces a
        different kind of wrong answer. Measured, not reasoned about: an old
        session holding the only match receives a new, non-matching message and
        another request repairs the projection between two pages. Its
        ``last_message_at`` moves AHEAD of this walk's cursor, so no later page
        can reach it — and the search returns *no matches* for a conversation
        that plainly matches, with nothing in the response to say a page was
        skipped. Silence is the answer search gives when it finds nothing, so
        there is no shape left for "and also I lost your cursor".

        So every page is required to come from the revision the first page
        established, and the walk starts again from the top when it is not. The
        revision moves only when a repair WRITES it — ordinary appends do not
        touch it — so this is rare, and starting again is honest where resuming
        would be a guess about where the cursor now points.

        **What the fence covers, and what it does not.** It covers what the
        projection describes — which rows exist, in which session, live or not.
        It does not cover a row's BODY: ``content`` and ``rendered_content`` are
        deliberately outside the change stamp (see ``conversation_sessions``'
        ``_DERIVED_FROM``), because watching them would make every
        re-encryption rebuild the projection, which is the cost that table
        exists to remove.

        That is sound rather than convenient, and it rests on a fact about the
        writers rather than on the fence. Three paths rewrite a body in place,
        and none can move a query term BETWEEN sessions: ``_migrate_message``
        re-encrypts the same plaintext; ``_migrate_split_sent_form`` moves the
        transport bytes into their own column and leaves the canonical text this
        matches against unchanged; ``salvage._mark_durable_folded`` writes a
        summary into one marker row, in place, in its own session. A body edit
        can therefore make a session start or stop matching mid-walk, which is
        an ordinary later truth about a live corpus and the same thing an append
        does — it cannot make a session that matched throughout come back
        unmatched. **A writer that edits bodies across sessions in one statement
        would break that, and there is no fence here that would notice.**

        The map is read AFTER the first page, which repairs, and re-read on
        every restart for the same reason. The projection is then no newer than
        the map, so every session the walk can return has its rows in it. The
        reverse order would leave a conversation written between the two reads
        listed but unsearchable. A session the map has no entry for — one
        appended under the walk, or one that exists only as its ``new_session``
        marker — can still match on its title, and matches on nothing else.
        """
        from .conversation_sessions import live_history_predicate

        snapshot = None
        membership: Optional[Dict[str, List[Any]]] = None
        results: List[Dict[str, Any]] = []
        after: Optional[Tuple[Any, ...]] = None

        while len(results) < limit:
            page, standing = await self._page_a_whole_projection(
                projection,
                limit=self.SEARCH_SESSION_STEP,
                after=after,
                refresh=snapshot is None,
            )
            if snapshot is None:
                snapshot = standing
            elif standing.fence != snapshot.fence:
                return None
            if not page:
                break
            # After the page, not before it: an agent whose projection is empty
            # has nothing to search, and reading its whole history to discover
            # that would be the one cost this walk can avoid paying.
            if membership is None:
                membership = await self._live_membership(snapshot.target)

            matched: List[Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]] = []
            for batch in _within_row_budget(
                page, membership, self.SEARCH_CONTENT_BATCH
            ):
                bodies = await self._search_rows(
                    [
                        row_id
                        for row in batch
                        for row_id in membership.get(row["session_id"], ())
                    ]
                )
                for row in batch:
                    name = names.get(row["session_id"])
                    decoration = session_match_decoration(
                        [
                            bodies[row_id]
                            for row_id in membership.get(row["session_id"], ())
                            if row_id in bodies
                        ],
                        name,
                        terms,
                    )
                    if decoration is not None:
                        matched.append((row, name, decoration))
                if len(results) + len(matched) >= limit:
                    break

            # Previews for the matches only. The list resolves one per listed
            # session; here most of what the walk looked at is not listed, and
            # fetching a preview for it would pay the list's cost for rows
            # nobody is going to see.
            previews = await self._preview_rows(
                [
                    row["first_user_message_id"]
                    for row, _, _ in matched
                    if row["first_user_message_id"] is not None
                ],
                live_history_predicate(),
            )
            for row, name, decoration in matched:
                session = self._as_listed_session(row, previews)
                if name is not None:
                    session["name"] = name
                session.update(decoration)
                results.append(session)
                if len(results) >= limit:
                    break

            if len(page) < self.SEARCH_SESSION_STEP:
                break
            # No `sort_sessions` on the way out, and that is deliberate: these
            # arrive in the projection's own order, which IS the list's order.
            # Re-sorting would be a second spelling of it, and a second spelling
            # is what let search and the list disagree about which session made
            # the page before (round-8/9 review of #2960).
            after = session_cursor_values(self._as_listed_session(page[-1], {}))

        # The other end of the same boundary. The frontier above keeps rows that
        # ARRIVED after the snapshot out of the answer; this asks whether any row
        # already inside it MOVED — archived, trashed, restored, re-grouped — in
        # which case the summaries and the rows no longer describe one history
        # and the walk starts again. It is one question asked once at the end
        # rather than per read, because holding for the whole walk is what has to
        # be true, and three primary-key reads is what asking costs.
        if membership is not None and not await projection.unchanged_below(snapshot):
            return None
        return results

    async def _live_membership(self, through: int) -> Dict[str, List[Any]]:
        """``{session_id: [row id, ...]}`` for this agent's live history.

        ``through`` is the ``conversation_history.id`` frontier the projection
        has accounted for, and this read stops there. **Not an optimization — a
        correctness requirement.** The summaries come out of the projection and
        the matching comes out of these rows, so if the two describe different
        frontiers one result carries both: measured, a session returned with
        ``message_count: 1`` and a snippet from the second message, because the
        append landed between the page read and this one (codex R2 P2). Bounded
        here, the rows a query is matched against are exactly the rows the count
        beside it was computed from.

        A message appended DURING a search is therefore not found by that
        search. That is the same lag the list already has — the projection is a
        cache and may be behind — and the next query repairs before it walks, so
        it is one query wide.

        Unbounded in the other direction on purpose, and it has to be: a row
        carrying no ``session_id`` belongs to whichever cluster it falls next
        to, and that is decided by the rows before it. A read windowed at the
        RECENT end would attribute the rows at its edge to sessions that merely
        look like theirs — which is the failure the 5,000-row scan this replaces
        had, silently. A frontier is not that window: it is the same cut the
        projection made, so both sides stop in the same place.

        The read is the light one — no ``content``, no decryption — and it is
        the same statement and the same derivation the projection's own
        transcript pass uses, so a row is in the session here that the
        projection describes there.
        """
        from .conversation_sessions import (
            TRANSCRIPT_PASS_NOISY_ROWS,
            live_transcript_sql,
            transcript_membership,
        )

        rows = await self.db.fetchall(
            live_transcript_sql(self.db.backend_type, through=True),
            (self.agent_id, through),
        )
        # Said out loud for the same reason the projection's transcript pass
        # says it, and against the same threshold — it is the same read over the
        # same rows, so "how large is worth mentioning" is one question, not two.
        #
        # The message says only what is true of EVERY search: the map covers the
        # whole history. It deliberately does not blame unstamped rows the way
        # the projection's pass does, because this read is taken whether or not
        # any exist — the attribution has to be the grouper's, and the grouper
        # needs the transcript. #3075 is the ticket for making it proportional
        # to the page.
        if len(rows) >= TRANSCRIPT_PASS_NOISY_ROWS:
            logger.warning(
                "search_sessions: %s's search attributed all %d of its live "
                "rows to sessions, because which session a row belongs to is "
                "decided by the rows before it and the map is built for the "
                "whole history rather than for the page (#3075). Search "
                "latency for this agent grows with its history.",
                self.agent_id,
                len(rows),
            )
        return transcript_membership(rows)

    async def _search_rows(
        self, message_ids: Sequence[Any]
    ) -> Dict[Any, Dict[str, Any]]:
        """``{id: normalized row}`` for the rows a search step has to read.

        By primary key, in batches, scoped to this agent and to the membership
        the projection describes — so a map entry pointing at a row that has
        since been deleted or archived resolves to nothing rather than letting a
        search match content the list no longer shows.
        """
        from .conversation_sessions import live_history_predicate

        rows: Dict[Any, Dict[str, Any]] = {}
        ids = list(message_ids)
        for start in range(0, len(ids), self.SEARCH_CONTENT_BATCH):
            batch = ids[start : start + self.SEARCH_CONTENT_BATCH]
            placeholders = ",".join("?" for _ in batch)
            raw = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at, rendered_content "
                "FROM conversation_history "
                f"WHERE agent_id = ? AND id IN ({placeholders}) "
                f"AND {live_history_predicate()}",
                (self.agent_id, *batch),
            )
            for row in raw:
                rows[row[0]] = await self._normalized_search_row(row)
        return rows

    async def _normalized_search_row(self, row: Sequence[Any]) -> Dict[str, Any]:
        """One history row decrypted and canonicalized, as search matches it.

        ``row`` is ``(id, role, content, metadata, created_at,
        rendered_content)``. Both search paths normalize through here, so the
        text a query is compared against is the same text whichever path found
        the row.

        **This one does not migrate**, and the other read paths still do. They
        read what a user asked to see; this reads whatever it has to in order to
        answer whether a word appears, which since #2961 is bounded by the query
        rather than by 5,000 rows. Opportunistically rewriting every row it
        touched would make one search of a legacy history a bulk UPDATE of it —
        tens of thousands of serialized writes on the corpus #3075 was filed
        against — inside a request the user made by typing in a box.

        It would also be a write inside this walk's own consistency fence. A
        rewritten row whose metadata is malformed compares unequal on the raw
        text (measured: valid JSON is unchanged, ``not json at all`` bumps the
        stamp), so the scan would move the change stamp it is about to check and
        restart itself. A read that invalidates its own snapshot is not a
        contract worth keeping.

        The rows still migrate: the one-shot backfill in
        ``security/encryption_backfill.py`` does it deliberately, and every path
        that shows a message to someone does it on the way past.
        """
        row_id, role = row[0], row[1]
        meta = parse_message_metadata(row[3])
        content, _needs_migration = self._decrypt_with_fallback(row[2], meta)

        # Canonical/transport split (#1402): match and snippet against the
        # raw user turn, never the rendered transport bytes.
        content, _ = await self._resolve_canonical(
            row_id, role, meta, content, row[5], migrate=False
        )

        cleaned_meta = remove_enc_flag(meta)
        if cleaned_meta:
            cleaned_meta.pop('key_version', None)
        return {
            "id": row_id,
            "role": role,
            "content": content,
            "metadata": cleaned_meta if cleaned_meta else {},
            "created_at": row[4],
        }

    async def _search_names(self) -> Dict[str, str]:
        """User-assigned titles, best-effort — they decorate, they never gate."""
        try:
            return await self.get_conversation_names()
        except Exception as e:
            logger.warning(f"search_sessions: name lookup failed: {e}")
            return {}
    async def clear_history(self) -> None:
        """Soft-delete every live message for this agent (#763).

        Stamps ``deleted_at`` instead of issuing a SQL DELETE so the rows
        remain recoverable from Trash until the retention janitor (#764)
        sweeps them. Already-deleted rows are left alone — re-stamping
        would extend their retention window.

        Use ``purge_all`` when you genuinely need to destroy the rows
        (administrative wipe, EPHEMERAL session close, restore-from-CAR).
        """
        await self.db.execute_commit(
            f"UPDATE conversation_history SET deleted_at = {self._now_sql()} "
            "WHERE agent_id = ? AND deleted_at IS NULL",
            (self.agent_id,)
        )

    # ------------------------------------------------------------------
    # Conversation titles (user-assigned rename support — issue #716).
    # ------------------------------------------------------------------
    #
    # Stored out-of-band in ``conversation_titles`` rather than on
    # conversation_history rows because:
    #   * message rows are frequently encrypted and their metadata JSON
    #     carries unrelated bookkeeping
    #   * rename is a single-row upsert here, vs. having to find-and-edit
    #     the first message of a session
    #   * deleting a session is a separate concern (#715) that wipes
    #     messages; the title row is stale-but-harmless until the user
    #     does something explicit to address it

    MAX_CONVERSATION_NAME_LENGTH = 120

    async def set_conversation_name(
        self, session_id: str, name: Optional[str]
    ) -> Optional[str]:
        """Upsert the user-chosen display name for a conversation.

        Args:
            session_id: The session whose name we're setting.
            name: New name.  ``None`` or empty-after-strip clears the
                  override (the UI will fall back to the computed
                  preview).  Non-empty values are trimmed and capped to
                  ``MAX_CONVERSATION_NAME_LENGTH``.

        Returns:
            The final stored value (trimmed), or ``None`` when cleared.
        """
        # Normalize inputs.  Empty strings and whitespace-only strings
        # collapse to "clear the override" so UI callers can just blur
        # an empty text input and get the expected behavior.
        if name is None:
            stored: Optional[str] = None
        else:
            trimmed = name.strip()
            if not trimmed:
                stored = None
            else:
                stored = trimmed[: self.MAX_CONVERSATION_NAME_LENGTH]

        if stored is None:
            await self.db.execute_commit(
                "DELETE FROM conversation_titles "
                "WHERE agent_id = ? AND session_id = ?",
                (self.agent_id, session_id),
            )
            return None

        # Upsert.  SQLite and Postgres both accept the ON CONFLICT syntax.
        await self.db.execute_commit(
            "INSERT INTO conversation_titles (agent_id, session_id, name, updated_at) "
            f"VALUES (?, ?, ?, {self._now_sql()}) "
            "ON CONFLICT (agent_id, session_id) DO UPDATE SET "
            f"  name = excluded.name, updated_at = {self._now_sql()}",
            (self.agent_id, session_id, stored),
        )
        return stored

    async def get_conversation_name(self, session_id: str) -> Optional[str]:
        """Return the user-assigned name for a conversation, or None."""
        row = await self.db.fetchone(
            "SELECT name FROM conversation_titles "
            "WHERE agent_id = ? AND session_id = ?",
            (self.agent_id, session_id),
        )
        if not row:
            return None
        return row[0]

    async def get_conversation_names(self) -> Dict[str, str]:
        """Return {session_id: name} for every titled session owned by
        this agent.  Used by the list endpoint to decorate the response
        in a single round-trip instead of querying per row.
        """
        rows = await self.db.fetchall(
            "SELECT session_id, name FROM conversation_titles "
            "WHERE agent_id = ? AND name IS NOT NULL",
            (self.agent_id,),
        )
        return {row[0]: row[1] for row in rows if row[1]}

    async def delete_message(self, message_id: int) -> bool:
        """Soft-delete a specific message by ID (#763).

        Stamps ``deleted_at`` so the row survives in Trash until purged
        explicitly or aged out by the retention janitor (#764). A row
        that's already soft-deleted reports ``False`` (no-op) rather than
        re-stamping its deleted_at.

        Returns:
            True if a live row was soft-deleted, False if not found or
            already in trash.
        """
        affected = await self.db.execute_commit(
            f"UPDATE conversation_history SET deleted_at = {self._now_sql()} "
            "WHERE id = ? AND agent_id = ? AND deleted_at IS NULL",
            (message_id, self.agent_id)
        )
        return _rows_affected(affected) > 0

    async def delete_conversation_session(self, session_id: str) -> int:
        """Soft-delete every live message in the given session (#763).

        Resolves which messages belong to `session_id` using the same
        logic `_get_session_messages` uses for loading — which covers
        both explicit UUID-based sessions (session_id stored in message
        metadata JSON) and legacy time-gap-based sessions (session_id
        is the row id of the first message in the cluster, and cluster
        members are discovered by time-gap walking).

        Stamps ``deleted_at`` rather than issuing a hard DELETE so the
        session is recoverable from Trash. Use ``purge_session`` for
        permanent removal.

        Args:
            session_id: The session to soft-delete.  Accepts either a
                UUID string (for metadata-based sessions) or a numeric
                message ID (for legacy time-gap sessions).

        Returns:
            Number of live messages stamped.  Returns 0 if the session
            doesn't exist, isn't owned by this agent, or is empty / all
            already soft-deleted.

        Notes:
            Per-agent scoped via the `agent_id = ?` filter in the final
            UPDATE.  Already-soft-deleted rows are not re-stamped (their
            existing deleted_at controls retention).  Ephemeral-mode
            callers must be rejected at the privacy wrapper above —
            this method does not read the privacy config.
        """
        # include_markers=True so soft-deleting a session also trashes its
        # new_session marker — otherwise the live orphan marker keeps the
        # session in the active list yet unresolvable by a later delete (#2027).
        rows = await self._get_session_messages(
            session_id, limit=10_000, include_markers=True
        )
        if not rows:
            return 0

        ids = [row[0] for row in rows]
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        params = [*ids, self.agent_id]
        affected = await self.db.execute_commit(
            f"UPDATE conversation_history "
            f"SET deleted_at = {self._now_sql()} "
            f"WHERE id IN ({placeholders}) AND agent_id = ? "
            f"AND deleted_at IS NULL",
            tuple(params),
        )
        return _rows_affected(affected)

    async def archive_conversation_session(self, session_id: str) -> int:
        """Archive every live message in the given session (#2149).

        Mirror image of ``delete_conversation_session`` but stamps
        ``archived_at`` instead of ``deleted_at``. Resolves the session's
        LIVE messages (deleted_filter='live') and stamps them, leaving the
        rows fully intact and un-trashed — archiving simply moves a session
        out of the active list into the archived view. Use
        ``unarchive_conversation_session`` to reverse.

        Returns:
            Number of live messages stamped. Returns 0 if the session
            doesn't exist, isn't owned by this agent, or is already
            archived / soft-deleted.
        """
        # include_markers=True so archiving a session also stamps its
        # new_session marker, keeping the marker in lock-step with content
        # (mirrors the delete path, #2027).
        rows = await self._get_session_messages(
            session_id, limit=10_000, deleted_filter="live", include_markers=True
        )
        if not rows:
            return 0

        ids = [row[0] for row in rows]
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        params = [*ids, self.agent_id]
        affected = await self.db.execute_commit(
            f"UPDATE conversation_history "
            f"SET archived_at = {self._now_sql()} "
            f"WHERE id IN ({placeholders}) AND agent_id = ? "
            f"AND deleted_at IS NULL AND archived_at IS NULL",
            tuple(params),
        )
        return _rows_affected(affected)

    async def unarchive_conversation_session(self, session_id: str) -> int:
        """Clear archived_at on every archived message in a session (#2149).

        Mirror image of ``restore_conversation_session``. Resolves the
        session's live messages (archived rows are still live — they were
        never soft-deleted) and clears ``archived_at`` so the session
        reappears in the active list.

        Returns:
            Number of rows unarchived. Zero if the session has no archived
            rows or doesn't exist.
        """
        rows = await self._get_session_messages(
            session_id, limit=10_000, deleted_filter="live", include_markers=True
        )
        if not rows:
            return 0

        ids = [row[0] for row in rows]
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        params = [*ids, self.agent_id]
        affected = await self.db.execute_commit(
            f"UPDATE conversation_history SET archived_at = NULL "
            f"WHERE id IN ({placeholders}) AND agent_id = ? "
            f"AND archived_at IS NOT NULL",
            tuple(params),
        )
        return _rows_affected(affected)

    # ------------------------------------------------------------------
    # Restore primitives (#763 / #765)
    # ------------------------------------------------------------------
    #
    # Mirror image of the soft-delete methods. Clear ``deleted_at`` so
    # the row reappears in normal reads. A row that was never soft-
    # deleted is a no-op (rowcount=0).

    async def restore_message(self, message_id: int) -> bool:
        """Clear deleted_at on a soft-deleted message (#763).

        Returns:
            True if a soft-deleted row was restored, False if the row
            doesn't exist, isn't owned by this agent, or wasn't actually
            in trash.
        """
        affected = await self.db.execute_commit(
            "UPDATE conversation_history SET deleted_at = NULL "
            "WHERE id = ? AND agent_id = ? AND deleted_at IS NOT NULL",
            (message_id, self.agent_id),
        )
        return _rows_affected(affected) > 0

    async def restore_conversation_session(self, session_id: str) -> int:
        """Clear deleted_at on every soft-deleted message in a session.

        Uses the same session-resolution logic as soft-delete but with
        ``deleted_filter='deleted'`` so we find messages that are in
        trash, not the live ones.

        Returns:
            Number of rows restored. Zero if the session has no soft-
            deleted rows or doesn't exist.
        """
        # include_markers=True so a marker trashed alongside its session
        # (#2027) is restored too, keeping delete/restore symmetric.
        rows = await self._get_session_messages(
            session_id, limit=10_000, deleted_filter="deleted", include_markers=True
        )
        if not rows:
            return 0

        ids = [row[0] for row in rows]
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        params = [*ids, self.agent_id]
        affected = await self.db.execute_commit(
            f"UPDATE conversation_history SET deleted_at = NULL "
            f"WHERE id IN ({placeholders}) AND agent_id = ? "
            f"AND deleted_at IS NOT NULL",
            tuple(params),
        )
        return _rows_affected(affected)

    # ------------------------------------------------------------------
    # Purge primitives (#763)
    # ------------------------------------------------------------------
    #
    # Hard SQL DELETE — the row is gone, no recovery. Callers must
    # supply a reason string for the audit trail (#750). Privacy mode
    # enforcement happens at the wrapper layer above.

    async def purge_message(
        self, message_id: int, reason: str = "user-initiated"
    ) -> bool:
        """Hard-delete a single message (#763).

        Removes the row regardless of whether it's currently live or
        already soft-deleted. The ``reason`` argument is recorded in the
        fail-closed destructive audit before this method performs the DELETE.

        Returns:
            True if a row was destroyed, False if not found.
        """
        purged = await self._purge_conversation_rows(
            [
                (
                    f"SELECT {_PURGE_SELECT_PLACEHOLDER} FROM conversation_history "
                    "WHERE id = ? AND agent_id = ?",
                    (message_id, self.agent_id),
                )
            ],
            operation_type="purge_message",
            scope={"table": "conversation_history", "message_id": message_id},
            reason=reason,
        )
        deleted = purged > 0
        if deleted:
            logger.info(
                "purge_message id=%s agent=%s reason=%s",
                message_id,
                self.agent_id,
                reason,
            )
        return deleted

    async def purge_conversation_session(
        self, session_id: str, reason: str = "user-initiated"
    ) -> int:
        """Hard-delete every message in a session, live or soft-deleted (#763).

        Uses ``deleted_filter='all'`` so we find both live messages and
        ones that previously soft-deleted into trash. The whole session
        is destroyed in one transaction.

        Returns:
            Number of rows destroyed.
        """
        # Resolve the complete membership set inside the destructive transaction.
        # The resolver is one statement (one immutable membership snapshot), then
        # _purge_conversation_rows locks its exact IDs in globally ascending,
        # backend-neutral batches.  A post-snapshot insert therefore survives,
        # while a >10k session is never silently truncated.
        purged = await self._purge_conversation_rows(
            None,
            operation_type="purge_conversation_session",
            scope={
                "table": "conversation_history",
                "session_id": session_id,
            },
            reason=reason,
            resolve_message_ids=lambda: self._get_complete_session_message_ids(
                session_id,
                deleted_filter="all",
                include_markers=True,
            ),
        )
        if purged:
            logger.info(
                "purge_conversation_session sid=%s agent=%s reason=%s rows=%d",
                session_id,
                self.agent_id,
                reason,
                purged,
            )
        return purged

    async def purge_all(self, reason: str = "administrative") -> int:
        """Hard-delete every conversation row for this agent (#763).

        Reserved for restore-from-CAR and explicit administrative wipe.
        NOT the user-facing 'clear history' button — that goes through
        ``clear_history``.  NOT the EPHEMERAL leak-purge — that path
        calls :meth:`purge_all_since` with the timestamp the agent
        entered EPHEMERAL so it can only destroy rows written *during*
        the EPHEMERAL stint.  Calling this method on a long-lived agent
        wipes the entire history regardless of when rows were authored
        (#867).
        """
        purged = await self._purge_conversation_rows(
            [
                (
                    f"SELECT {_PURGE_SELECT_PLACEHOLDER} FROM conversation_history "
                    "WHERE agent_id = ? ORDER BY id ASC",
                    (self.agent_id,),
                )
            ],
            operation_type="purge_all",
            scope={"table": "conversation_history"},
            reason=reason,
        )
        logger.info(
            "purge_all agent=%s reason=%s rows=%d",
            self.agent_id,
            reason,
            purged,
        )
        return purged

    async def purge_all_since(
        self,
        since_iso: str,
        *,
        reason: str = "ephemeral-leak",
    ) -> int:
        """Hard-delete conversation rows authored on/after ``since_iso``.

        Scoped variant of :meth:`purge_all` for the EPHEMERAL leak-purge
        (#867).  EPHEMERAL is "leave no trace," so anything written
        *during* the stint is a privacy-layer leak — but rows authored
        before the agent entered EPHEMERAL are preexisting NORMAL data
        the user explicitly wanted persisted.  Only rows whose
        ``created_at >= since_iso`` are destroyed.  ``since_iso`` is
        captured at the moment the wrapper sees the transition INTO
        EPHEMERAL.

        If ``since_iso`` is empty/None, returns 0 without running the
        DELETE — the absence of a timestamp means we can't safely scope,
        and the original wipe-on-shutdown bug is precisely what this
        method exists to prevent.
        """
        if not since_iso:
            logger.warning(
                "purge_all_since called without since_iso — refusing to purge "
                "(agent=%s, reason=%s)",
                self.agent_id,
                reason,
            )
            return 0
        created_at_predicate = self._timestamp_predicate("created_at", ">=")
        purged = await self._purge_conversation_rows(
            [
                (
                    f"SELECT {_PURGE_SELECT_PLACEHOLDER} FROM conversation_history "
                    f"WHERE agent_id = ? AND {created_at_predicate} "
                    "ORDER BY id ASC",
                    (self.agent_id, self._timestamp_query_param(since_iso)),
                )
            ],
            operation_type="purge_all_since",
            scope={
                "table": "conversation_history",
                "since_iso": since_iso,
            },
            reason=reason,
        )
        logger.info(
            "purge_all_since agent=%s since=%s reason=%s rows=%d",
            self.agent_id,
            since_iso,
            reason,
            purged,
        )
        return purged

    async def purge_trash_older_than(
        self,
        cutoff_iso: str,
        *,
        max_rows: int = 10_000,
        reason: str = "retention-janitor",
    ) -> int:
        """Hard-delete soft-deleted rows older than ``cutoff_iso`` (#764).

        The retention janitor calls this on a periodic tick to enforce
        the per-agent retention window. Three safety rails layered into
        one query:

        1. ``deleted_at IS NOT NULL`` — live rows are NEVER touched, no
           matter how old. The janitor's job is to age out trash, not
           data the user is still using.
        2. ``deleted_at < ?`` — the cutoff. Caller computes
           ``now - retention_days`` once per sweep so all rows in a
           batch use the same threshold.
        3. ``LIMIT ?`` on the one-time snapshot — prevents a runaway sweep
           from stalling the writer thread for minutes if the agent suddenly
           has 500k aged rows. The janitor calls back on the next tick to drain
           the rest.

        Args:
            cutoff_iso: ISO-8601 timestamp string. Rows whose
                ``deleted_at`` is strictly less than this are eligible
                for purge.
            max_rows: Hard cap on rows destroyed in a single call.
                Defaults to 10k. Set lower for tests.
            reason: Audit reason; lands in the operator log.

        Returns:
            Number of rows actually destroyed.
        """
        if not cutoff_iso:
            return 0
        if max_rows <= 0:
            return 0

        deleted_at_predicate = self._timestamp_predicate("deleted_at", "<")
        deleted_at_order = self._canonical_timestamp_sql("deleted_at")
        purged = await self._purge_conversation_rows(
            [
                (
                    f"SELECT {_PURGE_SELECT_PLACEHOLDER} FROM conversation_history "
                    "WHERE agent_id = ? AND deleted_at IS NOT NULL "
                    f"AND {deleted_at_predicate} "
                    "AND id IN ("
                    "SELECT id FROM conversation_history "
                    "WHERE agent_id = ? AND deleted_at IS NOT NULL "
                    f"AND {deleted_at_predicate} "
                    f"ORDER BY {deleted_at_order} ASC, id ASC LIMIT ?"
                    ") ORDER BY id ASC",
                    (
                        self.agent_id,
                        self._timestamp_query_param(cutoff_iso),
                        self.agent_id,
                        self._timestamp_query_param(cutoff_iso),
                        max_rows,
                    ),
                )
            ],
            operation_type="purge_trash_older_than",
            scope={
                "table": "conversation_history",
                "cutoff_iso": cutoff_iso,
                "max_rows": max_rows,
            },
            reason=reason,
        )
        if purged:
            logger.info(
                "purge_trash_older_than agent=%s cutoff=%s reason=%s rows=%d",
                self.agent_id,
                cutoff_iso,
                reason,
                purged,
            )
        return purged

    async def list_conversation_sessions(
        self, limit: int = 50, include_trashed: bool = False
    ) -> List[Dict[str, Any]]:
        """Return lightweight session summaries for this agent (#2019).

        Shares the session-boundary algorithm with the
        ``GET /api/conversations`` endpoint via
        :func:`group_messages_into_sessions`, so the agent's
        ``list_conversations`` tool and the UI never disagree on where one
        conversation ends and the next begins.

        **Audited under #2961 and deliberately left on the grouper.** It has no
        table to read, for two independent reasons, and neither is fixed by
        pointing it at one:

        * its membership is not the projection's. The read below excludes only
          *deleted* rows, so an archived conversation is in this list and is not
          in ``conversation_sessions`` — and with ``include_trashed`` the
          membership is Trash, which nothing projects.
        * it has no horizon to remove. The read is unbounded (``limit`` bounds
          the sessions returned, not the rows examined), so the defect #2948 was
          filed about — a fixed row window putting 34% of conversations out of
          reach — was never present here.

        That the agent's list mixes archived conversations in with active ones
        while the UI's does not is a real difference between the two surfaces.
        It is a question about what this tool should show, not about where it
        should read from, so it is named here rather than changed in passing.

        Args:
            limit: maximum number of sessions to return, most-recent first.
            include_trashed: when True, list soft-deleted (Trash) sessions
                instead of live ones — used to find a session to restore.

        Returns:
            list of ``{session_id, name, message_count, user_message_count,
            started_at, last_message_at, preview, is_trashed}`` dicts ordered
            most-recent first. ``name`` is omitted for untitled sessions.
        """
        history = await self.get_full_history_with_ids(
            include_excluded=True,
            include_stashed=True,
            only_deleted=include_trashed,
        )
        try:
            names = await self.get_conversation_names()
        except Exception as e:  # titles are best-effort decoration
            logger.warning(f"list_conversation_sessions: name lookup failed: {e}")
            names = {}

        # get_full_history_with_ids returns oldest-first, exactly what the
        # shared summarizer expects; it groups, coalesces same-UUID clusters
        # into unique delete targets (#2019), and shapes the newest-first list.
        return summarize_sessions(
            history,
            names=names,
            limit=limit,
            include_trashed=include_trashed,
            # Unwrap sent-form so the preview is the raw user text, not the
            # <retrieved_context>.../<user_input>... replay wrappers.
            preview_transform=extract_raw_user_content,
        )

    async def list_session_page(
        self,
        *,
        agent_id: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One page of the active conversation list, from the sessions table.

        Phase C of #2948, and the read the epic exists for. What this replaces
        fetched a fixed window of ``conversation_history`` — ``limit * 20``
        rows, capped at 1000 — and grouped whatever fell inside it. Measured on
        Emma's live database, 34% of her conversations were outside that window
        and no ``limit`` could reach them, because the cap was a constant rather
        than a function of the request. Here the bound is the page and the
        continuation is a key, so every session is reachable by asking again.

        Returns ``{"sessions": [...], "next_cursor": str | None}``. The session
        dicts carry exactly the fields the shared grouper emits — including the
        undecorated ``preview_content`` / ``preview_metadata`` /
        ``preview_wake_source`` triple — so the endpoint decorates one shape
        whichever path produced it.

        **The preview is a pointer, resolved here, never a stored copy.**
        ``content`` is ciphertext, so a preview column would be a plaintext copy
        of encrypted text and a record outliving what it describes (#2948).
        The projection stores which row the picker chose; this reads that row's
        body at request time, and a pointer at a row that has since gone
        resolves to nothing rather than to a stale excerpt.

        **Repair runs on the first page only.** The list must be current, and
        the projection is a cache that knows how far behind it is — three
        primary-key reads answer that, at any size of history. Repeating it per
        page would be work for no answer, and worse: a repair between two pages
        can move ``last_message_at`` under the cursor, so a page sequence is
        consistent to the extent that it is read from one repair's output.
        Keyset pagination over a table the agent is still writing to can still
        show a session twice or not at all if that session's activity moves
        across the cursor mid-walk — that is inherent to paging live data, and
        it is a far smaller window than the one this replaces, in which 34% of
        sessions could not be shown at all.
        """
        from .conversation_sessions import (
            ConversationSessionProjection,
            decode_session_cursor,
            encode_session_cursor,
            live_history_predicate,
        )


        # Named by the caller, defaulting to the one this store was built for.
        # Every other method here is bound to ``self.agent_id``, and this one
        # would be too if its caller were not the privacy wrapper, whose read
        # methods take the agent as an argument. Ignoring that argument and
        # answering for a different agent is the one behaviour that must not
        # happen — the read this replaces scoped its SQL to the value it was
        # given, so silently substituting another agent's would report an empty
        # list rather than refusing.
        agent = agent_id or self.agent_id
        bounded = max(1, int(limit))
        after = decode_session_cursor(cursor, "active") if cursor else None
        projection = ConversationSessionProjection(self.db, agent)
        # One more than asked for, which is how "is there another page" is
        # answered without a second query and without a COUNT over the whole
        # table — the cost this table exists to remove.
        rows, _ = await self._page_a_whole_projection(
            projection, limit=bounded + 1, after=after, refresh=after is None
        )
        has_more = len(rows) > bounded
        rows = rows[:bounded]

        previews = await self._preview_rows(
            [
                row["first_user_message_id"]
                for row in rows
                if row["first_user_message_id"] is not None
            ],
            live_history_predicate(),
            agent_id=agent,
        )
        sessions = [self._as_listed_session(row, previews) for row in rows]
        next_cursor = (
            encode_session_cursor(sessions[-1], "active")
            if has_more and sessions
            else None
        )
        return {"sessions": sessions, "next_cursor": next_cursor}

    async def _whole_watermark(self, projection):
        """This projection's watermark if it describes the WHOLE history, else None.

        Three questions, and all three have to hold before a page read means
        anything:

        * **valid** — something has been accounted for at all. All-zero is both
          "never built" and "built over an empty history", and only the flag
          separates them.
        * **complete** — the walk reached its target. A part-walked projection
          holds the OLDEST sessions (the walk goes forward by row id) while the
          list is ordered by most recent activity, so it is missing the FRONT of
          the list, not its tail.
        * **the same generation** — the counters this watermark was compared
          against are the ones standing now. A recreated cache rotates the
          generation and leaves the watermark, which is then ``valid`` and
          ``complete`` about a table that no longer exists; without this the
          continuation would page an empty cache and call it the end.
        """
        accounted = await projection.accounted()
        if not accounted.valid or not accounted.complete:
            return None
        if accounted.generation != await projection.observed_generation():
            return None
        return accounted

    async def _page_a_whole_projection(
        self, projection, *, limit: int, after, refresh: bool
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """Read one page, having established the projection was whole THROUGHOUT.

        Returns ``(rows, watermark)`` — the page, and what the projection it came
        out of had accounted for. The watermark is returned rather than left
        inside because a page carries two things a caller cannot ask for
        afterwards without a gap in which the answer changes:

        * its ``fence``, so a caller walking further than one page can require
          that every page came from the same projection — one page being
          self-consistent does not make a SEQUENCE of pages one read;
        * its ``target``, which is the ``conversation_history.id`` frontier
          these summaries describe. A caller that then reads ROWS — search does
          — has to read the same rows the summaries were computed from, or it
          reports one snapshot's count beside another snapshot's content.

        The check and the read are one observation, not two. A repair started by
        another request between them clears the table and commits chunks as it
        goes, so a page read in that window is a partial one — and its last row
        comes back with ``next_cursor: null``, which is a truncated list wearing
        the shape of a complete one. So the watermark is read before and after
        and required to be identical: it moves only when a repair writes it, and
        that is exactly the concurrency this cannot tolerate. Ordinary appends
        do not move it, so a page read beside a live conversation is unaffected.

        ``refresh`` is page one, which repairs first so the list reflects
        messages written since the last request. A continuation does not, on
        purpose: repairing moves ``last_message_at``, and a keyset cursor
        resumed against moved keys skips rows. It DOES repair when the
        projection is not whole — the cursor is meaningless over a partial table
        anyway, and refusing without repairing would make the 503 unclearable,
        since only page-one requests would ever advance the walk.
        """
        for _ in range(_PAGE_ATTEMPTS):
            before = await self._whole_watermark(projection)
            if refresh or before is None:
                await self._repair_until_whole(projection)
                before = await self._whole_watermark(projection)
                if before is None:
                    # Whole a moment ago and not whole now: another repair got
                    # in between. Try again rather than raise — the loop below
                    # is where giving up lives, and a repair that has just
                    # started will have finished by the next attempt.
                    continue
            rows = await projection.page(limit=limit, after=after)
            # The SAME question afterwards, and it turns on the REVISION rather
            # than on the watermark's fields. Two reasons, and the second is why
            # the column exists:
            #
            # * a generation rotation leaves the watermark row untouched, so a
            #   cache recreated under the read would pass a field comparison;
            # * `rebuild()` over an unchanged history invalidates, discards
            #   every row, walks, and lands on a watermark identical to the one
            #   it replaced — field for field. A page read during that walk is
            #   partial, and a field comparison certifies it. ABA.
            #
            # The revision only ever increases, so neither can hide.
            after_read = await self._whole_watermark(projection)
            if after_read is not None and after_read.fence == before.fence:
                return rows, before
        raise ProjectionNotReady(
            f"{projection.agent_id}'s conversation index was rebuilt underneath "
            f"{_PAGE_ATTEMPTS} attempts to read a page of it"
        )

    async def _repair_until_whole(
        self, projection, *, passes: int = REPAIR_PASSES
    ) -> None:
        """Repair until the walk has reached its target, or refuse to serve.

        A repair is budgeted — ``STEP_BUDGET`` chunks of ``CHUNK_ROWS`` — and
        may legitimately stop part-way on a history bigger than that product.
        The projection is then *partial*, and partial is not "a bit behind":
        the walk goes forward by row id, so what it has written is the OLDEST
        sessions, and the list is ordered by most recent activity. A partial
        projection is therefore missing the FRONT of the list — the
        conversations the user is most likely looking for — and it says so
        nowhere: the last page comes back with ``next_cursor: null``, which
        reads as the end of a list that is missing its beginning.

        So this loops rather than serving what happens to be there, and if the
        walk still has not landed it refuses. That is a worse answer for the
        request and the only honest one: an error says the list is not ready,
        while a truncated list says these are your conversations.

        ``accounted().complete`` is the question, not ``RepairOutcome.current``.
        They differ in the case that matters: a row arriving DURING a repair
        makes ``current`` false while the walk has reached its target perfectly
        well, and that is the ordinary state of an agent being written to. Only
        an unfinished WALK leaves a hole.
        """
        for _ in range(max(1, int(passes))):
            await projection.repair()
            if await self._whole_watermark(projection) is not None:
                return
        raise ProjectionNotReady(
            f"{projection.agent_id}'s conversation index is still being built; "
            "serving it now would list the oldest conversations and omit the "
            "newest, with nothing to say so"
        )

    async def _preview_rows(
        self,
        message_ids: List[Any],
        membership: str,
        *,
        agent_id: Optional[str] = None,
    ) -> Dict[Any, Tuple[Any, Dict[str, Any]]]:
        """``{id: (content, metadata)}`` for the rows a page previews.

        One statement for the whole page rather than one per row: a list of 50
        is 50 round trips otherwise, which on the read path is the cost the
        projection just removed reappearing one layer up.

        Scoped to this agent AND to the membership the projection describes, so
        a pointer at a row that has since been deleted or archived resolves to
        nothing. That is the designed direction — the pointer is a *detectable*
        stale state, and the change stamp has already moved, so the next repair
        replaces it.
        """
        if not message_ids:
            return {}
        placeholders = ",".join("?" for _ in message_ids)
        rows = await self.db.fetchall(
            "SELECT id, content, metadata FROM conversation_history "
            f"WHERE agent_id = ? AND id IN ({placeholders}) AND {membership}",
            (agent_id or self.agent_id, *message_ids),
        )
        return {
            row[0]: (row[1], parse_message_metadata(row[2])) for row in rows
        }

    @staticmethod
    def _as_listed_session(
        row: Dict[str, Any], previews: Dict[Any, Tuple[Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """One projection row shaped exactly as the grouper shapes a session.

        The timestamps are re-spelled with ``isoformat()`` because that is what
        the grouper emits and therefore what every consumer of this endpoint
        has always received — the browser parses it, and the projection holds
        the column's own spelling instead (canonical text on SQLite, a
        ``datetime`` on PostgreSQL, which would serialize as neither). A value
        that cannot be read is passed through as text rather than dropped: it is
        wrong to show, but it is what the row says, and a ``None`` here would
        move the session to the end of the list on the strength of a spelling.
        """
        content, metadata = previews.get(row["first_user_message_id"], (None, {}))
        return {
            "session_id": row["session_id"],
            "started_at": iso_session_timestamp(row["started_at"]),
            "last_message_at": iso_session_timestamp(row["last_message_at"]),
            "message_count": row["message_count"],
            "user_message_count": row["user_message_count"],
            "preview_content": content,
            "preview_metadata": metadata,
            "preview_wake_source": row["wake_source"],
        }

    async def count_session_messages(
        self, session_id: str, deleted_filter: str = "all"
    ) -> int:
        """Count messages a session resolves to (#2019).

        Uses the same uncapped exact-membership resolver as hard purge, so a
        permanent-deletion preview counts exactly what that operation will
        touch — including legacy row-id sessions only partially in Trash,
        whose deleted subset a grouped summary would mis-key.

        Args:
            session_id: UUID or legacy numeric session id.
            deleted_filter: ``live`` / ``deleted`` / ``all`` (default ``all``,
                matching ``purge_conversation_session``).
        """
        # The preview and permanent purge share the same uncapped membership
        # snapshot, including structural marker rows (#2027).
        message_ids = await self._get_complete_session_message_ids(
            session_id,
            deleted_filter=deleted_filter,
            include_markers=True,
        )
        return len(message_ids)

    async def message_belongs_to_session(
        self, message_id: Any, session_id: str
    ) -> bool:
        """True if ``message_id`` resolves within ``session_id`` (#2022).

        Uses the same uncapped exact-membership resolver as the hard-purge
        preview and operation, so a ``session_id`` guard agrees across live and
        trashed rows (a restore guard must match a message already in Trash).
        Identity, never content.
        """
        target = coerce_persistent_message_id(message_id)
        if target is None:
            return False
        # Include markers and use the uncapped resolver so a guard cannot deny a
        # valid message merely because 10k earlier rows share its session.
        message_ids = await self._get_complete_session_message_ids(
            session_id,
            deleted_filter="all",
            include_markers=True,
        )
        return target in message_ids

    async def find_messages_matching(
        self, content_pattern: str, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return live messages whose content matches a pattern (#2019).

        Shared by the delete preview and the delete itself so the two always
        agree on which rows are in scope.

        WARNING: searches decrypted content, so it loads all messages first.
        Use carefully on large histories.

        Args:
            content_pattern: case-insensitive substring to match.
            session_id: when provided, only messages belonging to this session
                are considered — the pattern can no longer reach across
                unrelated conversations. Accepts a UUID session_id or a legacy
                numeric message-id, same as the session lifecycle primitives.

        Returns:
            list of matching message dicts (id, role, content, metadata, ...).
        """
        history = await self.get_full_history_with_ids(
            include_excluded=True, include_stashed=True
        )

        # Confine the search to one session when asked, using the same
        # session-resolution logic the soft-delete primitives use.
        allowed_ids: Optional[set] = None
        if session_id is not None:
            rows = await self._get_session_messages(session_id, limit=10_000)
            allowed_ids = {row[0] for row in rows}

        pattern_lower = content_pattern.lower()
        matches = []
        for msg in history:
            if allowed_ids is not None and msg["id"] not in allowed_ids:
                continue
            if pattern_lower in msg.get("content", "").lower():
                matches.append(msg)
        return matches

    async def delete_messages_matching(
        self, content_pattern: str, session_id: Optional[str] = None
    ) -> int:
        """Soft-delete messages containing a pattern (#2019 adds session scope).

        Args:
            content_pattern: Text pattern to match in message content.
            session_id: when provided, confine deletion to that session so the
                pattern cannot reach across unrelated conversations.

        Returns:
            Number of messages deleted.
        """
        matches = await self.find_messages_matching(content_pattern, session_id)
        for msg in matches:
            await self.delete_message(msg["id"])
        return len(matches)

    async def get_full_history_with_ids(
        self,
        include_excluded: bool = False,
        include_stashed: bool = False,
        include_deleted: bool = False,
        only_deleted: bool = False,
        only_archived: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get conversation history with message IDs, optionally bounded.

        Args:
            include_excluded: If True, include messages marked as excluded from context.
            include_stashed: If True, include messages that are stashed.
            include_deleted: If True, include soft-deleted rows alongside
                live rows. Default False — Trash stays hidden.
            only_deleted: If True, return ONLY soft-deleted rows. Used by
                the Trash UI (#765). Implies ``include_deleted``.
            only_archived: If True, return ONLY archived rows that are not
                soft-deleted. Used by the Archive UI (#2149).
            limit: When provided, fetch only the newest N matching database
                rows while still returning them oldest-first. Filtering
                excluded/stashed rows are filtered in SQL before the bound.

        Returns:
            List of message dicts with 'id', 'role', 'content', 'metadata',
            'created_at', 'deleted_at' (None for live rows), and
            'archived_at' (None for non-archived rows).
        """
        if only_archived:
            del_clause = " AND archived_at IS NOT NULL AND deleted_at IS NULL"
        elif only_deleted:
            del_clause = " AND deleted_at IS NOT NULL"
        elif include_deleted:
            del_clause = ""
        else:
            del_clause = " AND deleted_at IS NULL"

        visibility_clause = ""
        if not include_excluded or not include_stashed:
            if self.db.backend_type == "postgres":
                if not include_excluded:
                    visibility_clause += (
                        " AND COALESCE((metadata::jsonb ->> "
                        "'excluded_from_context')::boolean, false) = false"
                    )
                if not include_stashed:
                    visibility_clause += (
                        " AND COALESCE((metadata::jsonb ->> "
                        "'stashed')::boolean, false) = false"
                    )
            else:
                if not include_excluded:
                    visibility_clause += (
                        " AND COALESCE(json_extract(metadata, "
                        "'$.excluded_from_context'), 0) != 1"
                    )
                if not include_stashed:
                    visibility_clause += (
                        " AND COALESCE(json_extract(metadata, '$.stashed'), 0) != 1"
                    )

        columns = (
            "id, role, content, metadata, created_at, deleted_at, archived_at"
        )
        if limit is None:
            sql = (
                f"SELECT {columns} FROM conversation_history "
                f"WHERE agent_id = ?{del_clause}{visibility_clause} ORDER BY id ASC"
            )
            params: Tuple[Any, ...] = (self.agent_id,)
        else:
            bounded_limit = max(1, int(limit))
            sql = (
                f"SELECT {columns} FROM ("
                f"SELECT {columns} FROM conversation_history "
                f"WHERE agent_id = ?{del_clause}{visibility_clause} "
                f"ORDER BY id DESC LIMIT ?"
                f") AS recent_history ORDER BY id ASC"
            )
            params = (self.agent_id, bounded_limit)
        rows = await self.db.fetchall(sql, params)
        history = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else {}
            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Filter out excluded messages unless requested
            if not include_excluded and meta.get("excluded_from_context"):
                continue

            # Filter out stashed messages unless requested
            if not include_stashed and meta.get("stashed"):
                continue

            # Opportunistic migration
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content, meta)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id}: {e}")

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            entry = {
                'id': row_id,
                'role': row[1],
                'content': content,
                'metadata': cleaned_meta if cleaned_meta else {},
                'created_at': row[4],
                'deleted_at': row[5],
                'archived_at': row[6],
            }
            history.append(entry)
        return history

    async def update_message_metadata(
        self,
        message_id: int,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """Update metadata for a specific message using atomic JSON merge.

        Uses one SQL JSON merge statement on both backends so concurrent
        rehearsal, reflection, tagging, and routing writes cannot be erased by
        a stale Python read-modify-write cycle.

        The key set is the caller's, so this is the one door through which
        ``metadata.session_id`` can be rewritten after insertion. When it is,
        the derived column moves in the SAME statement (#2958) — a merge that
        left the column behind would put the row in the single state the column
        is not allowed to occupy: naming a session its metadata no longer does.
        Every other key leaves the column untouched, including the merge that
        deletes nothing and the empty update.

        The column follows the metadata down to NULL where the merge cannot
        speak for the whole document, and "cannot" is wider than it looks. On
        BOTH backends a metadata root that is not an object — ``42``, ``[]``,
        ``"text"``, JSON ``null``, all shapes free-text legacy metadata holds —
        declines the merge while reporting success: SQLite's ``json_set``
        returns the document unchanged and PostgreSQL's ``||`` concatenates
        into an array instead of merging. On SQLite a legacy row carrying the
        key twice additionally stays ambiguous after ``json_set``. None of
        those rows has an id the column may claim. See
        :func:`~kestrel_sovereign.storage.session_id_column.merged_column_assignment`
        for the measurements and why the metadata is left as it is.

        Args:
            message_id: The message ID to update
            metadata_updates: Dict of metadata fields to update (merged with existing)

        Returns:
            True if message was found and updated, False otherwise
        """
        # The value is derived from the update rather than re-read from the
        # row, because the merge is last-writer-wins on this key and re-reading
        # would reintroduce the read-modify-write race this method exists to
        # avoid. Whether that value may be stamped at all is a question about
        # the DOCUMENT, though, and the two dialects merge documents
        # differently, so the clause comes from the shared contract rather than
        # being spelled here (#2958).
        if SESSION_ID_KEY in metadata_updates:
            column_assignment = ", " + merged_column_assignment(
                self.db.backend_type
            )
            column_params: tuple = (column_session_id(metadata_updates),)
        else:
            column_assignment = ""
            column_params = ()

        if self.db.backend_type == "postgres":
            updates_json = json.dumps(metadata_updates)
            # PostgreSQL: atomic JSON merge via || operator
            # COALESCE handles NULL metadata columns gracefully
            result = await self.db.execute_commit(
                "UPDATE conversation_history "
                "SET metadata = COALESCE(metadata::jsonb, '{}'::jsonb) || ?::jsonb"
                f"{column_assignment} "
                "WHERE id = ? AND agent_id = ?",
                (updates_json, *column_params, message_id, self.agent_id)
            )
            updated = result.rowcount > 0 if hasattr(result, 'rowcount') else True
            if not updated:
                logger.warning(f"Message {message_id} not found for agent {self.agent_id}")
            return updated
        else:
            # json_patch implements MergePatch, where a JSON null DELETES a
            # key. Python dict.update (and PostgreSQL ||) stores null, so use
            # variadic json_set with parsed JSON values to preserve the public
            # contract for None/list/dict/bool as well as scalar values.
            if metadata_updates:
                assignments = ", ".join("?, json(?)" for _ in metadata_updates)
                merge_params: List[Any] = []
                for key, value in metadata_updates.items():
                    escaped_key = str(key).replace('"', '\\"')
                    merge_params.extend((f'$."{escaped_key}"', json.dumps(value)))
                sql = (
                    "UPDATE conversation_history SET metadata = "
                    f"json_set(COALESCE(metadata, '{{}}'), {assignments})"
                    f"{column_assignment} "
                    "WHERE id = ? AND agent_id = ?"
                )
                params = (*merge_params, *column_params,
                          message_id, self.agent_id)
            else:
                # No keys, so no session_id among them — the column cannot be
                # stale after a merge that changes nothing.
                sql = (
                    "UPDATE conversation_history SET metadata = "
                    "COALESCE(metadata, '{}') WHERE id = ? AND agent_id = ?"
                )
                params = (message_id, self.agent_id)
            result = await self.db.execute_commit(sql, params)
            updated = _rows_affected(result) > 0
            if not updated:
                logger.warning(f"Message {message_id} not found for agent {self.agent_id}")
            return updated

    async def _semantic_recall_dependency_rows(
        self,
        *,
        assertion_ids: Iterable[str] = (),
        revision_ids: Iterable[str] = (),
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Return this agent's rows linked to exact canonical identities.

        The lookup is an exact JSON identity match, never a scan over message
        content.  It includes archived and trashed artifacts deliberately: a
        later restore must not make a forgotten fact reappear in context.

        Physical erasure can remove a historical revision while preserving its
        assertion identity with a fresh direct current revision.  Both identity
        dimensions therefore participate in the match: assertion IDs cover a
        fully erased assertion; revision IDs cover erased historical lineage.
        """
        normalized_assertion_ids = tuple(
            sorted({value for value in assertion_ids if isinstance(value, str) and value})
        )
        normalized_revision_ids = tuple(
            sorted({value for value in revision_ids if isinstance(value, str) and value})
        )
        if not normalized_assertion_ids and not normalized_revision_ids:
            return []

        predicates: list[str] = []
        params: list[Any] = [self.agent_id]
        if normalized_assertion_ids:
            predicates.append(
                "dependency ->> 'assertion_id' IN ("
                + ", ".join("?" for _ in normalized_assertion_ids)
                + ")"
            )
            params.extend(normalized_assertion_ids)
        if normalized_revision_ids:
            predicates.append(
                "dependency ->> 'revision_id' IN ("
                + ", ".join("?" for _ in normalized_revision_ids)
                + ")"
            )
            params.extend(normalized_revision_ids)
        postgres_predicate = " OR ".join(predicates)

        if self.db.backend_type == "postgres":
            postgres_sql = (
                "SELECT id, metadata FROM conversation_history "
                "WHERE agent_id = ? AND EXISTS ("
                "SELECT 1 FROM jsonb_array_elements("
                "CASE jsonb_typeof(metadata::jsonb -> "
                "'semantic_recall_dependencies') "
                "WHEN 'array' THEN metadata::jsonb -> "
                "'semantic_recall_dependencies' ELSE '[]'::jsonb END"
                ") AS dependency WHERE "
                + postgres_predicate
                + ") ORDER BY id ASC"
            )
            rows = await self.db.fetchall(
                postgres_sql,
                tuple(params),
            )
        else:
            sqlite_predicates: list[str] = []
            sqlite_params: list[Any] = [self.agent_id]
            if normalized_assertion_ids:
                sqlite_predicates.append(
                    "json_extract(CASE WHEN json_valid(dependency.value) "
                    "THEN dependency.value ELSE '{}' END, '$.assertion_id') IN ("
                    + ", ".join("?" for _ in normalized_assertion_ids)
                    + ")"
                )
                sqlite_params.extend(normalized_assertion_ids)
            if normalized_revision_ids:
                sqlite_predicates.append(
                    "json_extract(CASE WHEN json_valid(dependency.value) "
                    "THEN dependency.value ELSE '{}' END, '$.revision_id') IN ("
                    + ", ".join("?" for _ in normalized_revision_ids)
                    + ")"
                )
                sqlite_params.extend(normalized_revision_ids)
            sqlite_sql = (
                "SELECT id, metadata FROM conversation_history "
                "WHERE agent_id = ? AND EXISTS ("
                "SELECT 1 FROM json_each(CASE "
                "WHEN json_type(CASE WHEN json_valid(COALESCE(metadata, '{}')) "
                "THEN metadata ELSE '{}' END, "
                "'$.semantic_recall_dependencies') = 'array' THEN json_extract("
                "CASE WHEN json_valid(COALESCE(metadata, '{}')) THEN metadata "
                "ELSE '{}' END, '$.semantic_recall_dependencies') "
                "ELSE '[]' END) AS dependency WHERE "
                + " OR ".join(sqlite_predicates)
                + ") ORDER BY id ASC"
            )
            rows = await self.db.fetchall(
                sqlite_sql,
                tuple(sqlite_params),
            )

        matched: List[Tuple[int, Dict[str, Any]]] = []
        for row_id, raw_metadata in rows:
            try:
                metadata = (
                    dict(raw_metadata)
                    if isinstance(raw_metadata, dict)
                    else json.loads(raw_metadata) if raw_metadata else {}
                )
            except (TypeError, json.JSONDecodeError):
                # The SQL predicate only admits valid JSON in SQLite; keep the
                # defensive guard for old/manual PostgreSQL rows.
                continue
            if isinstance(metadata, dict):
                matched.append((int(row_id), metadata))
        return matched

    async def exclude_semantic_recall_dependencies(
        self,
        *,
        assertion_ids: Iterable[str] = (),
        revision_ids: Iterable[str] = (),
    ) -> Tuple[int, ...]:
        """Exclude every conversation artifact linked to an exact identity.

        This is intentionally a reversible context exclusion rather than a
        string-based deletion.  The privacy wrapper invokes it in the same
        transaction as canonical fact deletion; callers receive exact message
        IDs only so dependent episode summaries can be excluded too.
        """
        rows = await self._semantic_recall_dependency_rows(
            assertion_ids=assertion_ids,
            revision_ids=revision_ids,
        )
        message_ids: list[int] = []
        for message_id, metadata in rows:
            updates: Dict[str, Any] = {"excluded_from_context": True}
            if not metadata.get("excluded_reason"):
                updates["excluded_reason"] = "semantic_assertion_deleted"
            if await self.update_message_metadata(message_id, updates):
                message_ids.append(message_id)
        return tuple(message_ids)

    async def scrub_semantic_recall_dependencies(
        self,
        *,
        assertion_ids: Iterable[str] = (),
        revision_ids: Iterable[str] = (),
    ) -> int:
        """Remove erased assertion/revision IDs from excluded artifacts.

        Physical canonical erasure must not leave a durable reference to the
        erased identifier.  ``excluded_from_context`` is deliberately sticky:
        scrubbing lineage never re-admits the derivative content.
        """
        normalized_assertion_ids = {
            value for value in assertion_ids if isinstance(value, str) and value
        }
        normalized_revision_ids = {
            value for value in revision_ids if isinstance(value, str) and value
        }
        rows = await self._semantic_recall_dependency_rows(
            assertion_ids=normalized_assertion_ids,
            revision_ids=normalized_revision_ids,
        )
        scrubbed = 0
        for message_id, metadata in rows:
            dependencies = metadata.get("semantic_recall_dependencies")
            if not isinstance(dependencies, list):
                continue
            retained = [
                dependency
                for dependency in dependencies
                if not (
                    isinstance(dependency, dict)
                    and (
                        dependency.get("assertion_id") in normalized_assertion_ids
                        or dependency.get("revision_id") in normalized_revision_ids
                    )
                )
            ]
            if len(retained) == len(dependencies):
                continue
            updated = await self.update_message_metadata(
                message_id,
                {
                    "semantic_recall_dependencies": retained,
                    "excluded_from_context": True,
                },
            )
            if updated:
                scrubbed += 1
        return scrubbed

    async def atomic_increment_metadata_counter(
        self,
        message_id: int,
        counter_field: str,
        timestamp_field: Optional[str] = None,
    ) -> bool:
        """Atomically increment a numeric counter in the metadata JSON.

        Wraps the read-modify-write of an integer field into a single
        SQL statement so concurrent callers (rehearsal-effect updates,
        reflection ``mark_applied`` writes, etc.) can't lose increments
        by both reading the same value and writing the same successor.

        ``counter_field``  — the JSON key holding the integer to bump
                             (e.g. ``"access_count"``, ``"applied_count"``).
        ``timestamp_field`` — optional JSON key to overwrite with the
                             current UTC ISO timestamp in the same
                             statement (e.g. ``"last_accessed"``,
                             ``"last_applied"``).  Pass ``None`` to skip.

        Returns ``True`` when the message was found and updated,
        ``False`` if no row matched (unknown message id, or the row
        belongs to a different agent).  Use the return value to
        verify the bookkeeping landed — the caller (retriever or
        reflection hook) swallows exceptions but a False return is
        the only signal that the write was a no-op rather than a
        success.

        Backend-aware: both SQLite (3.38+) and PostgreSQL (jsonb)
        support ``json_set`` + ``json_extract`` natively, so the
        statement is a single atomic UPDATE on either.

        Both field names come from the caller, which makes this the second door
        onto ``metadata.session_id`` — and unlike
        :meth:`update_message_metadata` it declines rather than keeping the
        derived column in step (#2958). Not squeamishness: this writes a
        counter or a timestamp, and neither is ever a session identity. A
        caller asking for one has a bug that a silently-synchronized column
        would hide.
        """
        if SESSION_ID_KEY in (counter_field, timestamp_field):
            raise ValueError(
                f"atomic_increment_metadata_counter cannot write metadata."
                f"{SESSION_ID_KEY}: it writes counters and timestamps, and "
                f"neither is a session identity. Session identity is moved "
                f"through update_message_metadata, which keeps the indexed "
                f"conversation_history.{SESSION_ID_KEY} column in step (#2958)."
            )

        now_iso = datetime.now(timezone.utc).isoformat() if timestamp_field else None

        if self.db.backend_type == "postgres":
            # PostgreSQL: ``conversation_history.metadata`` is declared
            # TEXT (see async_database.py).  Cast to ``jsonb`` consistently
            # for both the extract and the set — using ``metadata->>?``
            # on a TEXT column raises an operator-not-found error at
            # runtime, and the caller (``update_access`` /
            # ``update_applied``) swallows the exception, so a typo here
            # would silently drop every bookkeeping write.  Mirrors the
            # existing ``update_message_metadata`` PG pattern that uses
            # ``metadata::jsonb`` everywhere it reads metadata.
            #
            # Every ``?`` used as a JSON key carries an explicit ``::text``
            # cast — ``jsonb ->>`` is overloaded between (jsonb, text) and
            # (jsonb, int), and ``ARRAY[?]`` feeding ``jsonb_set`` needs
            # the element to be unambiguously text for asyncpg's
            # statement-prep type inference.  Without these casts the
            # bookkeeping UPDATE fails to prepare on Postgres and the
            # caller's exception-swallow silently drops the write.
            if timestamp_field is None:
                sql = (
                    "UPDATE conversation_history SET metadata = "
                    "  jsonb_set("
                    "    COALESCE(metadata::jsonb, '{}'::jsonb),"
                    "    ARRAY[?::text],"
                    "    to_jsonb(COALESCE((metadata::jsonb->>(?::text))::int, 0) + 1)"
                    "  ) "
                    "WHERE id = ? AND agent_id = ?"
                )
                params: tuple = (counter_field, counter_field, message_id, self.agent_id)
            else:
                sql = (
                    "UPDATE conversation_history SET metadata = "
                    "  jsonb_set("
                    "    jsonb_set("
                    "      COALESCE(metadata::jsonb, '{}'::jsonb),"
                    "      ARRAY[?::text],"
                    "      to_jsonb(COALESCE((metadata::jsonb->>(?::text))::int, 0) + 1)"
                    "    ),"
                    "    ARRAY[?::text],"
                    "    to_jsonb(?::text)"
                    "  ) "
                    "WHERE id = ? AND agent_id = ?"
                )
                params = (
                    counter_field, counter_field,
                    timestamp_field, now_iso,
                    message_id, self.agent_id,
                )
            rows_affected = await self.db.execute_commit(sql, params)
            return rows_affected > 0
        else:
            # SQLite: json_set + json_extract in one statement.  Both
            # functions are core (3.38+) so no extension load required.
            if timestamp_field is None:
                rows_affected = await self.db.execute_commit(
                    "UPDATE conversation_history SET metadata = "
                    "  json_set("
                    "    COALESCE(metadata, '{}'),"
                    "    '$.' || ?,"
                    "    COALESCE(json_extract(metadata, '$.' || ?), 0) + 1"
                    "  ) "
                    "WHERE id = ? AND agent_id = ?",
                    (counter_field, counter_field, message_id, self.agent_id),
                )
            else:
                rows_affected = await self.db.execute_commit(
                    "UPDATE conversation_history SET metadata = "
                    "  json_set("
                    "    json_set("
                    "      COALESCE(metadata, '{}'),"
                    "      '$.' || ?,"
                    "      COALESCE(json_extract(metadata, '$.' || ?), 0) + 1"
                    "    ),"
                    "    '$.' || ?,"
                    "    ?"
                    "  ) "
                    "WHERE id = ? AND agent_id = ?",
                    (
                        counter_field, counter_field,
                        timestamp_field, now_iso,
                        message_id, self.agent_id,
                    ),
                )
            return rows_affected > 0

    async def update_messages_metadata(
        self,
        message_ids: List[int],
        metadata_updates: Dict[str, Any]
    ) -> int:
        """Update metadata for multiple messages.

        One :meth:`update_message_metadata` per id, so the derived
        ``session_id`` column moves with the metadata here too (#2958) — the
        per-row statement stays atomic; only the batch is not.

        Args:
            message_ids: List of message IDs to update
            metadata_updates: Dict of metadata fields to update (merged with existing)

        Returns:
            Number of messages successfully updated
        """
        updated_count = 0
        for msg_id in message_ids:
            if await self.update_message_metadata(msg_id, metadata_updates):
                updated_count += 1
        return updated_count

    async def get_messages_by_ids(
        self,
        message_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Get specific messages by their IDs.

        Args:
            message_ids: List of message IDs to retrieve

        Returns:
            List of message dicts with 'id', 'role', 'content', 'metadata', 'created_at'
        """
        if not message_ids:
            return []

        placeholders = ",".join("?" * len(message_ids))
        rows = await self.db.fetchall(
            f"SELECT id, role, content, metadata, created_at, rendered_content "
            f"FROM conversation_history "
            f"WHERE id IN ({placeholders}) AND agent_id = ? "
            f"AND deleted_at IS NULL AND archived_at IS NULL ORDER BY id ASC",
            (*message_ids, self.agent_id)
        )

        history = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else {}
            content, _ = self._decrypt_with_fallback(row[2], meta)
            content, _ = await self._resolve_canonical(
                row_id, row[1], meta, content, row[5]
            )

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            entry = {
                'id': row_id,
                'role': row[1],
                'content': content,
                'metadata': cleaned_meta if cleaned_meta else {},
                'created_at': row[4]
            }
            history.append(entry)
        return history

    async def get_salient_memory_candidates(
        self, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return pinned/high-importance rows from across the live corpus."""
        if self.db.backend_type == "postgres":
            salience_order = """
                 CASE WHEN (metadata::jsonb ->> 'decay_protected')::boolean
                      THEN 1 ELSE 0 END DESC,
                 COALESCE((metadata::jsonb ->> 'importance')::double precision, 0) DESC,"""
        else:
            salience_order = """
                 CASE WHEN json_extract(metadata, '$.decay_protected') = 1
                      THEN 1 ELSE 0 END DESC,
                 COALESCE(CAST(json_extract(metadata, '$.importance') AS REAL), 0) DESC,"""
        rows = await self.db.fetchall(
            f"""SELECT id, role, content, metadata, created_at, rendered_content
               FROM conversation_history
               WHERE agent_id = ? AND deleted_at IS NULL AND archived_at IS NULL
               ORDER BY {salience_order}
                 id DESC
               LIMIT ?""",
            (self.agent_id, max(1, int(limit))),
        )
        candidates: List[Dict[str, Any]] = []
        for row_id, role, encrypted_content, raw_meta, created_at, rendered in rows:
            try:
                meta = json.loads(raw_meta) if raw_meta else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            content, _ = self._decrypt_with_fallback(encrypted_content, meta)
            content, _ = await self._resolve_canonical(
                row_id, role, meta, content, rendered
            )
            cleaned_meta = remove_enc_flag(meta)
            cleaned_meta.pop("key_version", None)
            candidates.append({
                "id": row_id,
                "role": role,
                "content": content,
                "metadata": cleaned_meta,
                "created_at": created_at,
            })
        return candidates

    async def get_lexical_memory_candidates(
        self,
        query: str,
        limit: int = 100,
        page_size: int = 1000,
        excluded_embedding_profile_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return bounded lexical top-k while scanning eligible live rows.

        When ``excluded_embedding_profile_id`` is supplied, scan only rows
        outside that vector space (NULL/legacy or a different profile). This
        is the mixed-corpus bridge: active-profile rows arrive through kNN,
        while exact legacy matches remain recallable during backfill/migration.
        """
        started = perf_counter()
        query_tokens = set(_tokenize_for_search(_strip_search_wrappers(query)))
        if not query_tokens:
            return []
        ranked: List[Tuple[float, int, Dict[str, Any]]] = []
        indexed_candidates = 0
        fallback_rows_scanned = 0
        fallback_reason: Optional[str] = None

        def rank_candidate(candidate: Dict[str, Any]) -> None:
            content = candidate.get("content") or ""
            candidate_tokens = set(
                _tokenize_for_search(_strip_search_wrappers(content))
            )
            overlap = len(query_tokens & candidate_tokens) / len(query_tokens)
            if overlap <= 0:
                return
            ranked.append((overlap, int(candidate["id"]), candidate))

        index_available = True
        try:
            indexed_ids = await self._lexical_index.candidate_message_ids(
                sorted(query_tokens),
                limit=max(1, int(limit)),
                excluded_embedding_profile_id=excluded_embedding_profile_id,
            )
            indexed_rows = await self.get_messages_by_ids(indexed_ids)
            indexed_candidates = len(indexed_rows)
            for candidate in indexed_rows:
                rank_candidate(candidate)
        except ValueError as exc:
            fallback_reason = "query_token_budget"
            logger.info(
                "Blind lexical index budget exceeded for agent %s (%s); "
                "using the complete decrypt-scan fallback.", self.agent_id, exc,
            )
            index_available = False
        except Exception as exc:  # noqa: BLE001 - pre-migration compatibility
            fallback_reason = "index_unavailable"
            logger.warning(
                "Blind lexical candidate index unavailable for agent %s (%s); "
                "using the complete decrypt-scan fallback.", self.agent_id, exc,
            )
            index_available = False

        before_id: Optional[int] = None
        use_coverage_hint = index_available and await self._has_lexical_coverage_index()
        while True:
            cursor_clause = " AND id < ?" if before_id is not None else ""
            profile_clause = ""
            params_list: List[Any] = [self.agent_id]
            if excluded_embedding_profile_id is not None:
                profile_clause = (
                    " AND (embedding_profile_id IS NULL "
                    "OR embedding_profile_id != ?)"
                )
                params_list.append(excluded_embedding_profile_id)
            coverage_hint = (
                " INDEXED BY idx_conversation_lexical_coverage"
                if use_coverage_hint
                else ""
            )
            base_sql = (
                "SELECT id, role, content, metadata, created_at, rendered_content "
                f"FROM conversation_history{coverage_hint} WHERE agent_id = ? "
                "AND deleted_at IS NULL AND archived_at IS NULL"
                f"{profile_clause}"
            )
            if index_available:
                # Four disjoint range probes keep the complete-coverage path
                # on idx_conversation_lexical_coverage. A single OR predicate
                # made SQLite scan every live row even when there were no gaps.
                gap_rows: Dict[int, Any] = {}
                for gap, gap_params in self._lexical_index.coverage_gap_predicates(
                    "conversation_history"
                ):
                    gap_query_params = [*params_list, *gap_params]
                    if before_id is not None:
                        gap_query_params.append(before_id)
                    gap_query_params.append(page_size)
                    for row in await self.db.fetchall(
                        f"{base_sql} AND {gap}{cursor_clause} "
                        "ORDER BY id DESC LIMIT ?",
                        tuple(gap_query_params),
                    ):
                        gap_rows[int(row[0])] = row
                rows = sorted(
                    gap_rows.values(), key=lambda row: int(row[0]), reverse=True
                )[:page_size]
            else:
                if before_id is not None:
                    params_list.append(before_id)
                params_list.append(page_size)
                rows = await self.db.fetchall(
                    f"{base_sql}{cursor_clause} ORDER BY id DESC LIMIT ?",
                    tuple(params_list),
                )
            if not rows:
                break
            fallback_rows_scanned += len(rows)
            for row_id, role, encrypted_content, raw_meta, created_at, rendered in rows:
                try:
                    meta = json.loads(raw_meta) if raw_meta else {}
                except (json.JSONDecodeError, TypeError):
                    meta = {}
                content, _ = self._decrypt_with_fallback(encrypted_content, meta)
                content, _ = await self._resolve_canonical(
                    row_id, role, meta, content, rendered
                )
                cleaned_meta = remove_enc_flag(meta) or {}
                cleaned_meta.pop("key_version", None)
                rank_candidate({
                    "id": row_id,
                    "role": role,
                    "content": content,
                    "metadata": cleaned_meta,
                    "created_at": created_at,
                })
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            del ranked[max(1, int(limit)):]
            before_id = int(rows[-1][0])
            if len(rows) < page_size:
                break
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        del ranked[max(1, int(limit)):]
        elapsed_ms = (perf_counter() - started) * 1000.0
        self._last_lexical_bridge_stats = {
            "elapsed_ms": round(elapsed_ms, 3),
            "indexed_candidates": indexed_candidates,
            "fallback_rows_scanned": fallback_rows_scanned,
            "index_available": index_available,
            "fallback_reason": fallback_reason,
            "query_token_count": len(query_tokens),
        }
        return [item[2] for item in ranked]

    async def _has_lexical_coverage_index(self) -> bool:
        """Verify SQLite's optional planner hint before naming the index.

        The migration is intentionally non-fatal.  A partially migrated DB
        must therefore degrade to the slower complete scan, never fail recall
        with ``no such index``.
        """
        if getattr(self.db, "backend_type", None) != "sqlite":
            return False
        if self._lexical_coverage_index_available is not None:
            return self._lexical_coverage_index_available
        try:
            row = await self.db.fetchone(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
                ("idx_conversation_lexical_coverage",),
            )
            self._lexical_coverage_index_available = bool(row and row[0])
        except Exception as exc:  # noqa: BLE001 - performance hint only
            logger.debug("Could not inspect lexical coverage index: %s", exc)
            self._lexical_coverage_index_available = False
        return self._lexical_coverage_index_available

    async def get_lexical_index_health(self) -> Dict[str, Any]:
        """Return durable coverage plus the most recent bridge cost."""
        health = (await self._lexical_index.health()).as_dict()
        health["last_bridge"] = dict(self._last_lexical_bridge_stats)
        return health

    async def get_embedding_profile_health(
        self, current_profile_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return active/null/stale vector coverage without materializing rows."""
        if current_profile_id is None:
            service = self._lazy_embedding_service()
            if service is not None and hasattr(service, "current_profile_id"):
                try:
                    current_profile_id = service.current_profile_id()
                except Exception as exc:  # pragma: no cover - provider defensive
                    logger.debug("Could not resolve current embedding profile: %s", exc)
        row = await self.db.fetchone(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN embedding_profile_id IS NULL THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN embedding_profile_id = ? "
            "AND embedding_vec IS NOT NULL THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN embedding_profile_id IS NOT NULL "
            "AND embedding_profile_id != ? THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN embedding_profile_id = ? "
            "AND embedding_vec IS NULL THEN 1 ELSE 0 END) "
            "FROM conversation_history WHERE agent_id = ? "
            "AND deleted_at IS NULL AND archived_at IS NULL",
            (current_profile_id, current_profile_id, current_profile_id, self.agent_id),
        )
        values = tuple(row or (0, 0, 0, 0, 0))
        return {
            "current_profile_id": current_profile_id,
            "total_live": int(values[0] or 0),
            "null_profile": int(values[1] or 0),
            "active_profile_vectors": int(values[2] or 0),
            "other_profile": int(values[3] or 0),
            "active_profile_missing_vector": int(values[4] or 0),
        }

    async def backfill_lexical_index(
        self,
        *,
        batch_size: int = 500,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Resumably blind-index every current live conversation row.

        Backfill revalidates each hydrated owner and commits its token rows and
        coverage marker atomically.  A concurrent hard purge therefore wins
        either before or after the whole replacement; it can never leave a
        token-only remnant from stale hydrated data. Garbage collection is
        deliberately limited to obsolete keys from successfully replaced
        owners: new-message writers commit tokens before their history row, so
        a broad ownerless-token sweep can mistake an in-flight write for trash.
        """
        started = perf_counter()
        initial_health = await self._lexical_index.health()
        scanned = attempted = failed = garbage_collected = 0
        after_id: Optional[int] = None
        page_size = max(1, int(batch_size))
        while max_rows is None or scanned < max_rows:
            gap, gap_params = self._lexical_index.coverage_gap_sql("c")
            cursor_clause = " AND c.id > ?" if after_id is not None else ""
            params: List[Any] = [*gap_params, self.agent_id]
            if after_id is not None:
                params.append(after_id)
            remaining = (
                page_size if max_rows is None
                else min(page_size, max_rows - scanned)
            )
            params.append(remaining)
            rows = await self.db.fetchall(
                f"SELECT c.id, c.lexical_index_id FROM conversation_history c WHERE {gap} "
                "AND c.agent_id = ? AND c.deleted_at IS NULL "
                f"AND c.archived_at IS NULL{cursor_clause} "
                "ORDER BY c.id LIMIT ?",
                tuple(params),
            )
            if not rows:
                break
            ids = [int(row[0]) for row in rows]
            old_keys_by_id = {
                int(row[0]): str(row[1]) if row[1] is not None else None
                for row in rows
            }
            after_id = ids[-1]
            scanned += len(ids)
            hydrated = {
                int(row["id"]): row
                for row in await self.get_messages_by_ids(ids)
            }
            replacement_entries: List[LexicalIndexReplacement] = []
            for row_id in ids:
                row = hydrated.get(row_id)
                if row is None:
                    failed += 1
                    continue
                content = row.get("content") or ""
                tokens = _tokenize_for_search(_strip_search_wrappers(content))
                message_key = self._lexical_index.backfill_message_key(row_id)
                replacement_entries.append(
                    LexicalIndexReplacement(
                        message_id=row_id,
                        expected_key=old_keys_by_id[row_id],
                        replacement_key=message_key,
                        tokens=tokens,
                    )
                )
            try:
                replacement_result = (
                    await self._lexical_index.replace_existing_messages(
                        replacement_entries
                    )
                )
                attempted += len(replacement_entries)
                garbage_collected += replacement_result.garbage_collected
            except Exception as exc:  # noqa: BLE001 - continue resumable sweep
                logger.warning(
                    "Lexical index backfill batch ending at message %s failed: %s",
                    after_id, exc,
                )
                failed += len(replacement_entries)
            if len(rows) < remaining:
                break
        health = await self.get_lexical_index_health()
        # asyncpg.executemany does not report affected rows.  Derive the
        # truthful durable gain from coverage instead of treating attempts as
        # successes (concurrent deletion is harmless and not a failed row).
        indexed = max(
            0,
            min(scanned, int(health["indexed_current"]) - initial_health.indexed_current),
        )
        return {
            "scanned": scanned,
            "attempted": attempted,
            "indexed": indexed,
            "failed": failed,
            "garbage_collected": garbage_collected,
            "remaining": health["unindexed"],
            "coverage": health["coverage"],
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "index_version": health["index_version"],
        }

    async def search_messages_by_content(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search messages and return with IDs (for context management).

        Note: For encrypted content, this searches the encrypted text.
        For reliable search with encryption, use get_full_history_with_ids
        and filter client-side.

        Args:
            query: Search query string
            limit: Maximum results

        Returns:
            List of matching messages with IDs
        """
        # Get all messages and search client-side (handles encryption)
        all_messages = await self.get_full_history_with_ids(include_excluded=True)
        query_lower = query.lower()

        matches = []
        for msg in all_messages:
            if query_lower in msg.get("content", "").lower():
                matches.append(msg)
                if len(matches) >= limit:
                    break

        return matches

    async def get_excluded_messages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages that have been excluded from context.

        Args:
            limit: Maximum messages to return

        Returns:
            List of excluded messages with their metadata
        """
        excl_spaced, excl_packed = _metadata_like_forms(
            "excluded_from_context", "true"
        )
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata, created_at FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "AND (metadata LIKE ? OR metadata LIKE ?) "
            "ORDER BY id DESC LIMIT ?",
            (self.agent_id, excl_spaced, excl_packed, limit)
        )

        results = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else {}
            content, _ = self._decrypt_with_fallback(row[2], meta)

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            results.append({
                'id': row_id,
                'role': row[1],
                'content': content,
                'metadata': cleaned_meta if cleaned_meta else {},
                'created_at': row[4]
            })
        return results

    async def get_stashed_messages(
        self,
        stash_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get messages that have been stashed.

        Args:
            stash_id: Optional specific stash ID to retrieve
            limit: Maximum messages to return

        Returns:
            List of stashed messages with their metadata
        """
        if stash_id:
            # Get specific stash
            sid_spaced, sid_packed = _metadata_like_forms(
                "stash_id", f'"{stash_id}"'
            )
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "AND (metadata LIKE ? OR metadata LIKE ?) "
                "ORDER BY id ASC LIMIT ?",
                (self.agent_id, sid_spaced, sid_packed, limit)
            )
        else:
            # Get all stashed messages
            stashed_spaced, stashed_packed = _metadata_like_forms("stashed", "true")
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "AND (metadata LIKE ? OR metadata LIKE ?) "
                "ORDER BY id DESC LIMIT ?",
                (self.agent_id, stashed_spaced, stashed_packed, limit)
            )

        results = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else {}
            content, _ = self._decrypt_with_fallback(row[2], meta)

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            results.append({
                'id': row_id,
                'role': row[1],
                'content': content,
                'metadata': cleaned_meta if cleaned_meta else {},
                'created_at': row[4]
            })
        return results

    async def list_stashes(self) -> List[Dict[str, Any]]:
        """Get a list of all stashes with summary info.

        Returns:
            List of stash summaries with id, name, message_count, stashed_at
        """
        # Get all stashed messages
        stashed_spaced, stashed_packed = _metadata_like_forms("stashed", "true")
        rows = await self.db.fetchall(
            "SELECT metadata FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "AND (metadata LIKE ? OR metadata LIKE ?)",
            (self.agent_id, stashed_spaced, stashed_packed)
        )

        # Group by stash_id
        stashes: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            meta = json.loads(row[0]) if row[0] else {}
            stash_id = meta.get("stash_id")
            if not stash_id:
                continue

            if stash_id not in stashes:
                stashes[stash_id] = {
                    "stash_id": stash_id,
                    "name": meta.get("stash_name", "unnamed"),
                    "message_count": 0,
                    "stashed_at": meta.get("stashed_at")
                }
            stashes[stash_id]["message_count"] += 1

        # Sort by stashed_at descending
        return sorted(
            stashes.values(),
            key=lambda x: x.get("stashed_at", ""),
            reverse=True
        )

    async def get_all_audit_failures(self) -> List[Dict[str, Any]]:
        """
        Retrieves all conversation entries that are marked as audit failures.
        Automatically decrypts content if encryption was enabled.
        """
        af_spaced, af_packed = _metadata_like_forms("audit_failure", "true")
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "AND (metadata LIKE ? OR metadata LIKE ?)",
            (self.agent_id, af_spaced, af_packed)
        )
        results = []
        for row in rows:
            if not row[3]:
                continue
            row_id = row[0]
            meta = json.loads(row[3])
            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Opportunistic migration
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content, meta)
                except Exception as e:
                    logger.warning(f"Migration failed for message {row_id} in get_all_audit_failures: {e}")

            cleaned_meta = remove_enc_flag(meta)
            if cleaned_meta:
                cleaned_meta.pop('key_version', None)

            results.append({
                "role": row[1],
                "content": content,
                "metadata": cleaned_meta if cleaned_meta else None
            })
        return results
