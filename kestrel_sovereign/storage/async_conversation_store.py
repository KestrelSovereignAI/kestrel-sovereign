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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .async_database import AsyncDatabase
from .conversation_ids import coerce_persistent_message_id
from .sqla.embedding_profile import upsert_embedding_profile as _upsert_embedding_profile
from .encryption import (
    get_fernet, get_agent_fernet, encrypt_string, decrypt_string, remove_enc_flag,
    DecryptionError
)
from kestrel_sovereign.security.input_guardrails import extract_raw_user_content

logger = logging.getLogger(__name__)

# Current key version for new encryptions
CURRENT_KEY_VERSION = 1


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


def _token_match_score(query_tokens: List[str], content_lower: str) -> float:
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


class AsyncConversationStore:
    """Async conversation history storage with per-agent encryption."""

    def __init__(
        self,
        db: AsyncDatabase,
        agent_id: str = "",
        llm_service: Optional[Any] = None,
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
        # Global key for backward compatibility
        self._global_fernet = get_fernet()
        # Per-agent key (recommended, used for new data)
        self._agent_fernet = get_agent_fernet(agent_id) if agent_id else None
        # Auto-migration on read (can be disabled via env var)
        self._migrate_on_read = os.environ.get("KESTREL_DISABLE_MIGRATION") != "true"

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

            # Compare gap; reuse if within window
            from datetime import datetime, timezone, timedelta
            if isinstance(prev_created_at, str):
                try:
                    prev_dt = datetime.fromisoformat(prev_created_at.replace("Z", "+00:00"))
                except ValueError:
                    return self._new_session_id()
            else:
                return self._new_session_id()

            if prev_dt.tzinfo is None:
                prev_dt = prev_dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
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
            return provided
        return await self._derive_implicit_session_id()

    async def add_conversation(self, role: str, content: str,
                               metadata: Optional[Dict] = None,
                               session_id: Optional[str] = None,
                               rendered_content: Optional[str] = None) -> None:
        """Add a conversation message with per-agent encryption.

        Args:
            role: Message role (user, assistant, system)
            content: Canonical message content. For user turns this is the
                raw user speech (typically ``wrap_user_input(raw)``) — never
                the rendered-with-retrieval transport form.
            metadata: Optional metadata dict
            session_id: If provided, link this message to a specific session.
                       This allows resuming old conversations beyond the 30-min gap.
                       If not provided, an implicit session_id is derived from
                       the time-gap heuristic (30 min inactivity = new session).
            rendered_content: Write-once transport bytes for byte-stable cache
                replay (#1402). Carries memories + RAG baked into the user
                template. The history-load path emits this verbatim so the
                prefix bytes match what the LLM saw at send time. Encrypted
                with the same per-agent key as ``content``.
        """
        meta = dict(metadata) if metadata else {}

        # Resolve session_id: explicit wins; otherwise derive from time gap
        if not session_id:
            session_id = await self._derive_implicit_session_id()

        if session_id:
            meta['session_id'] = session_id

        # Use per-agent key for new messages
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

        # Compute the embedding from plaintext content BEFORE the
        # INSERT so we can co-write ``embedding_vec`` in a single
        # statement (no follow-up UPDATE → no autoincrement-id
        # round-trip needed). The embedding service is optional;
        # absence + per-call failure both fall back to the legacy
        # column set with no behavioural change.
        embedding_vec_val: Optional[List[float]] = await self._maybe_embed(content)

        # #1477 — derive the active embedding profile id so the row
        # can be filtered out of kNN reads that don't share the
        # same semantic coordinate space. Never blocks the write —
        # if the service can't describe itself the row's profile id
        # stays NULL (= invisible to profile-filtered kNN, harmless).
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

        await self._insert_message(
            role=role,
            content=to_store,
            rendered_content=rendered_to_store,
            metadata=json.dumps(meta) if meta else None,
            embedding=embedding_vec_val,
            embedding_profile_id=profile_id,
        )

        # Upsert the profile descriptor into the registry table so
        # ``kestrel-sovereign embeddings audit`` can map id →
        # human-readable fields. Best-effort: registry write must
        # NEVER block message persistence. Cached in-process to
        # avoid an UPSERT per turn.
        if profile_id is not None and embedding_service is not None:
            try:
                await _upsert_embedding_profile(
                    self.db, embedding_service, profile_id,
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

    async def _insert_message(
        self,
        *,
        role: str,
        content: str,
        rendered_content: Optional[str],
        metadata: Optional[str],
        embedding: Optional[List[float]],
        embedding_profile_id: Optional[str] = None,
    ) -> None:
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
        content) we fall back to the legacy column list — preserving
        bit-identical INSERT shape with prior versions so any tooling
        sniffing the SQL surface keeps working. ``embedding_profile_id``
        is always written alongside ``embedding_vec`` — they share
        the same row state (#1477).
        """
        base_cols = "agent_id, role, content, rendered_content, metadata, created_at"
        base_vals_suffix = f", {self._now_sql()}"
        base_params = (
            self.agent_id, role, content, rendered_content, metadata,
        )

        if embedding is None:
            sql = (
                f"INSERT INTO conversation_history ({base_cols}) "
                f"VALUES (?, ?, ?, ?, ?{base_vals_suffix})"
            )
            await self.db.execute_commit(sql, base_params)
            return

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
                f"VALUES (?, ?, ?, ?, ?{base_vals_suffix}, {emb_placeholder}, ?)"
            )
            await self.db.execute_commit(
                sql, base_params + (emb_bind, embedding_profile_id),
            )
        except Exception as e:
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
                    f"VALUES (?, ?, ?, ?, ?{base_vals_suffix}, {emb_placeholder})"
                )
                await self.db.execute_commit(sql, base_params + (emb_bind,))
            except Exception as e2:
                # Even the legacy embedding_vec column is missing
                # (Phase-2 migration hasn't run either). Fall all
                # the way back to the original column list so the
                # message still persists; the retriever uses
                # keyword overlap until the next boot finishes the
                # migrations.
                logger.info(
                    "Legacy embedding_vec-only INSERT also failed for "
                    "agent %s (%s); persisting the message without any "
                    "embedding column.", self.agent_id, e2,
                )
                sql = (
                    f"INSERT INTO conversation_history ({base_cols}) "
                    f"VALUES (?, ?, ?, ?, ?{base_vals_suffix})"
                )
                await self.db.execute_commit(sql, base_params)

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

    async def _migrate_message(self, row_id: int, decrypted_content: str) -> None:
        """Re-encrypt a message with per-agent key."""
        if not self._agent_fernet:
            return

        new_content, _ = encrypt_string(decrypted_content, self._agent_fernet)
        new_meta = {"enc": True, "key_version": CURRENT_KEY_VERSION}

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
    ) -> tuple[str, Optional[str]]:
        """Apply the canonical/transport split (#1402) for a single row.

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
            if self._migrate_on_read:
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
            rows = await self._get_session_messages(session_id, limit)
        else:
            # Default behavior: get most recent live messages.
            # rendered_content (#1402) appended at row[5] so existing
            # positional accesses on metadata/created_at don't shift.
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at, rendered_content "
                "FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "ORDER BY id DESC LIMIT ?",
                (self.agent_id, limit)
            )
        history = []
        for row in reversed(rows):  # Return in chronological order
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else None

            # Skip messages excluded from context (compressed, summarized, etc.)
            if meta and meta.get("excluded_from_context"):
                continue

            content, needs_migration = self._decrypt_with_fallback(row[2], meta)

            # Opportunistic migration to per-agent key
            if needs_migration and self._migrate_on_read:
                try:
                    await self._migrate_message(row_id, content)
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

    async def _get_session_messages(
        self,
        session_id: str,
        limit: int,
        deleted_filter: str = "live",
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

        Returns:
            List of raw rows
            (id, role, content, metadata, created_at, rendered_content)
            — rendered_content (#1402) appended so existing positional
            accesses below don't shift.
        """
        del_clause = self._deleted_filter_clause(deleted_filter)
        from datetime import datetime
        from kestrel_sdk.config.constants import SESSION_GAP_MINUTES

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
                all_rows = await self.db.fetchall(
                    f"""SELECT id, role, content, metadata, created_at, rendered_content
                       FROM conversation_history
                       WHERE agent_id = ? AND created_at >= ?{del_clause}
                       ORDER BY created_at ASC
                       LIMIT ?""",
                    (self.agent_id, start_time, limit * 2)  # Fetch extra in case of filtering
                )

        # Also get messages that explicitly belong to this session (resumed conversations)
        # These are messages with session_id in metadata that may come after a time gap
        resumed_rows = await self.db.fetchall(
            f"""SELECT id, role, content, metadata, created_at, rendered_content
               FROM conversation_history
               WHERE agent_id = ? AND metadata LIKE ?{del_clause}
               ORDER BY created_at ASC
               LIMIT ?""",
            (self.agent_id, f'%"session_id": "{session_id}"%', limit)
        )

        # Also try without space after colon (JSON formatting varies)
        resumed_rows_alt = await self.db.fetchall(
            f"""SELECT id, role, content, metadata, created_at, rendered_content
               FROM conversation_history
               WHERE agent_id = ? AND metadata LIKE ?{del_clause}
               ORDER BY created_at ASC
               LIMIT ?""",
            (self.agent_id, f'%"session_id":"{session_id}"%', limit)
        )

        # Merge resumed rows (dedupe by id)
        seen_ids = set()
        merged_rows = []
        for row in all_rows + resumed_rows + resumed_rows_alt:
            if row[0] not in seen_ids:
                seen_ids.add(row[0])
                merged_rows.append(row)

        # Sort by created_at
        merged_rows.sort(key=lambda r: r[4] or '')

        # Filter to only include messages in this session
        session_rows = []
        last_timestamp = None
        is_first = True
        session_id_str = str(session_id)

        for row in merged_rows:
            created_at = row[4]
            metadata_json = row[3]

            # Parse timestamp
            if isinstance(created_at, str):
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                    try:
                        timestamp = datetime.strptime(created_at, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    timestamp = datetime.now()
            elif created_at:
                timestamp = created_at
            else:
                timestamp = datetime.now()

            # Check if this message explicitly belongs to this session (resumed)
            is_resumed_message = False
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                    if meta.get('session_id') == session_id_str:
                        is_resumed_message = True
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for message in session {session_id}: {e}")

            # Check for new_session marker (skip after first message, but not for resumed)
            if not is_first and not is_resumed_message and metadata_json:
                try:
                    meta = json.loads(metadata_json)
                    if meta.get('new_session'):
                        break
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for new_session check: {e}")

            # Check time gap (only for non-resumed messages)
            if last_timestamp and not is_resumed_message:
                gap_minutes = (timestamp - last_timestamp).total_seconds() / 60
                if gap_minutes > SESSION_GAP_MINUTES:
                    # Skip this non-resumed message (it belongs to a different session)
                    # Continue looking for resumed messages that explicitly belong to this session
                    continue

            # Skip session markers from results
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                    if meta.get('type') == 'session_marker':
                        last_timestamp = timestamp
                        is_first = False
                        continue
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for session_marker check: {e}")

            session_rows.append(row)
            last_timestamp = timestamp
            is_first = False

            if len(session_rows) >= limit:
                break

        # Return in DESC order to match the non-session query format
        # (will be reversed by caller in get_conversation_history)
        return list(reversed(session_rows))

    async def get_full_history(self) -> List[Dict[str, Any]]:
        """Get complete live conversation history with automatic decryption.

        Soft-deleted rows (#763) are filtered out — use
        ``get_full_history_with_ids(include_deleted=True)`` if you need
        to see Trash too.
        """
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata, rendered_content "
            "FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
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
                    await self._migrate_message(row_id, content)
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
            # Match both `"session_id": "X"` and `"session_id":"X"` formats
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, rendered_content "
                "FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "AND (metadata LIKE ? OR metadata LIKE ?) "
                "ORDER BY id DESC LIMIT 5000",
                (
                    self.agent_id,
                    f'%"session_id": "{session_id}"%',
                    f'%"session_id":"{session_id}"%',
                ),
            )
        else:
            # Fetch all live messages (up to 5000) and search client-side after decryption
            # SQL LIKE doesn't work on encrypted content, so we must decrypt first
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, rendered_content "
                "FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "ORDER BY id DESC LIMIT 5000",
                (self.agent_id,)
            )

        query_lower = query.lower()
        query_tokens = _tokenize_for_search(query)
        use_token_fallback = len(query_tokens) >= 2

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
                    await self._migrate_message(row_id, content)
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

            # Tokenized fallback: score by fraction of query terms present
            if use_token_fallback and len(exact_results) < limit:
                score = _token_match_score(query_tokens, content_lower)
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
        rows = await self._get_session_messages(session_id, limit=10_000)
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
        rows = await self._get_session_messages(
            session_id, limit=10_000, deleted_filter="deleted"
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
        already soft-deleted. The ``reason`` argument is recorded by
        the caller in the audit log; this method just performs the
        DELETE.

        Returns:
            True if a row was destroyed, False if not found.
        """
        affected = await self.db.execute_commit(
            "DELETE FROM conversation_history WHERE id = ? AND agent_id = ?",
            (message_id, self.agent_id),
        )
        deleted = _rows_affected(affected) > 0
        if deleted:
            logger.info(
                "purge_message id=%s agent=%s reason=%s",
                message_id, self.agent_id, reason,
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
        rows = await self._get_session_messages(
            session_id, limit=10_000, deleted_filter="all"
        )
        if not rows:
            return 0

        ids = [row[0] for row in rows]
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        params = [*ids, self.agent_id]
        affected = await self.db.execute_commit(
            f"DELETE FROM conversation_history "
            f"WHERE id IN ({placeholders}) AND agent_id = ?",
            tuple(params),
        )
        purged = _rows_affected(affected)
        if purged:
            logger.info(
                "purge_conversation_session sid=%s agent=%s reason=%s rows=%d",
                session_id, self.agent_id, reason, purged,
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
        affected = await self.db.execute_commit(
            "DELETE FROM conversation_history WHERE agent_id = ?",
            (self.agent_id,),
        )
        purged = _rows_affected(affected)
        logger.info(
            "purge_all agent=%s reason=%s rows=%d",
            self.agent_id, reason, purged,
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
                self.agent_id, reason,
            )
            return 0
        affected = await self.db.execute_commit(
            "DELETE FROM conversation_history "
            "WHERE agent_id = ? AND created_at >= ?",
            (self.agent_id, since_iso),
        )
        purged = _rows_affected(affected)
        logger.info(
            "purge_all_since agent=%s since=%s reason=%s rows=%d",
            self.agent_id, since_iso, reason, purged,
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
        3. ``LIMIT ?`` (via ``IN (subquery)``) — prevents a runaway
           sweep from stalling the writer thread for minutes if the
           agent suddenly has 500k aged rows. The janitor calls back
           on the next tick to drain the rest.

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

        # SQLite doesn't support LIMIT directly inside DELETE on every
        # build path, and even when it does the syntax differs from
        # PostgreSQL. The IN (SELECT ... LIMIT ...) form is portable.
        affected = await self.db.execute_commit(
            "DELETE FROM conversation_history "
            "WHERE id IN ("
            "  SELECT id FROM conversation_history "
            "  WHERE agent_id = ? "
            "    AND deleted_at IS NOT NULL "
            "    AND deleted_at < ? "
            "  ORDER BY deleted_at ASC "
            "  LIMIT ?"
            ")",
            (self.agent_id, cutoff_iso, max_rows),
        )
        purged = _rows_affected(affected)
        if purged:
            logger.info(
                "purge_trash_older_than agent=%s cutoff=%s reason=%s rows=%d",
                self.agent_id, cutoff_iso, reason, purged,
            )
        return purged

    async def delete_messages_matching(self, content_pattern: str) -> int:
        """Delete messages containing a specific pattern (case-insensitive).

        WARNING: This searches decrypted content, so it loads all messages first.
        Use carefully on large histories.

        Args:
            content_pattern: Text pattern to match in message content

        Returns:
            Number of messages deleted
        """
        # Get all messages with IDs
        history = await self.get_full_history_with_ids(include_excluded=True, include_stashed=True)

        # Find matching IDs
        pattern_lower = content_pattern.lower()
        ids_to_delete = []
        for msg in history:
            if pattern_lower in msg.get("content", "").lower():
                ids_to_delete.append(msg["id"])

        # Delete them
        for msg_id in ids_to_delete:
            await self.delete_message(msg_id)

        return len(ids_to_delete)

    async def get_full_history_with_ids(
        self,
        include_excluded: bool = False,
        include_stashed: bool = False,
        include_deleted: bool = False,
        only_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get complete conversation history with message IDs.

        Args:
            include_excluded: If True, include messages marked as excluded from context.
            include_stashed: If True, include messages that are stashed.
            include_deleted: If True, include soft-deleted rows alongside
                live rows. Default False — Trash stays hidden.
            only_deleted: If True, return ONLY soft-deleted rows. Used by
                the Trash UI (#765). Implies ``include_deleted``.

        Returns:
            List of message dicts with 'id', 'role', 'content', 'metadata',
            'created_at', and 'deleted_at' (None for live rows).
        """
        if only_deleted:
            del_clause = " AND deleted_at IS NOT NULL"
        elif include_deleted:
            del_clause = ""
        else:
            del_clause = " AND deleted_at IS NULL"

        rows = await self.db.fetchall(
            f"SELECT id, role, content, metadata, created_at, deleted_at "
            f"FROM conversation_history "
            f"WHERE agent_id = ?{del_clause} ORDER BY id ASC",
            (self.agent_id,)
        )
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
                    await self._migrate_message(row_id, content)
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
            }
            history.append(entry)
        return history

    async def update_message_metadata(
        self,
        message_id: int,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """Update metadata for a specific message using atomic JSON merge.

        Uses SQL-level json_patch (PostgreSQL) or a SELECT-then-UPDATE with
        optimistic locking (SQLite) to avoid race conditions when multiple
        callers update metadata on the same message concurrently.

        Args:
            message_id: The message ID to update
            metadata_updates: Dict of metadata fields to update (merged with existing)

        Returns:
            True if message was found and updated, False otherwise
        """
        updates_json = json.dumps(metadata_updates)

        if self.db.backend_type == "postgres":
            # PostgreSQL: atomic JSON merge via || operator
            # COALESCE handles NULL metadata columns gracefully
            result = await self.db.execute_commit(
                "UPDATE conversation_history "
                "SET metadata = COALESCE(metadata::jsonb, '{}'::jsonb) || ?::jsonb "
                "WHERE id = ? AND agent_id = ?",
                (updates_json, message_id, self.agent_id)
            )
            updated = result.rowcount > 0 if hasattr(result, 'rowcount') else True
            if not updated:
                logger.warning(f"Message {message_id} not found for agent {self.agent_id}")
            return updated
        else:
            # SQLite: SELECT-then-UPDATE (single-writer, no true race condition)
            row = await self.db.fetchone(
                "SELECT metadata FROM conversation_history WHERE id = ? AND agent_id = ?",
                (message_id, self.agent_id)
            )
            if not row:
                logger.warning(f"Message {message_id} not found for agent {self.agent_id}")
                return False

            current_meta = json.loads(row[0]) if row[0] else {}
            current_meta.update(metadata_updates)

            await self.db.execute_commit(
                "UPDATE conversation_history SET metadata = ? WHERE id = ? AND agent_id = ?",
                (json.dumps(current_meta), message_id, self.agent_id)
            )
            return True

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
        """
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
            f"SELECT id, role, content, metadata, created_at FROM conversation_history "
            f"WHERE id IN ({placeholders}) AND agent_id = ? "
            f"AND deleted_at IS NULL ORDER BY id ASC",
            (*message_ids, self.agent_id)
        )

        history = []
        for row in rows:
            row_id = row[0]
            meta = json.loads(row[3]) if row[3] else {}
            content, _ = self._decrypt_with_fallback(row[2], meta)

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
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata, created_at FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "AND metadata LIKE '%\"excluded_from_context\": true%' "
            "ORDER BY id DESC LIMIT ?",
            (self.agent_id, limit)
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
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL AND metadata LIKE ? "
                "ORDER BY id ASC LIMIT ?",
                (self.agent_id, f'%"stash_id": "{stash_id}"%', limit)
            )
        else:
            # Get all stashed messages
            rows = await self.db.fetchall(
                "SELECT id, role, content, metadata, created_at FROM conversation_history "
                "WHERE agent_id = ? AND deleted_at IS NULL "
                "AND metadata LIKE '%\"stashed\": true%' "
                "ORDER BY id DESC LIMIT ?",
                (self.agent_id, limit)
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
        rows = await self.db.fetchall(
            "SELECT metadata FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "AND metadata LIKE '%\"stashed\": true%'",
            (self.agent_id,)
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
        rows = await self.db.fetchall(
            "SELECT id, role, content, metadata FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "AND metadata LIKE '%\"audit_failure\": true%'",
            (self.agent_id,)
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
                    await self._migrate_message(row_id, content)
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
