"""
Context Builder for Kestrel Agent.

This module handles the assembly of context for LLM prompts, including:
- RAG document retrieval
- Conversation history formatting
- Constitutional grounding
- Session briefings
- Bootstrap file convention (AGENTS.md, SOUL.md, TOOLS.md, etc.)
- Token-aware truncation
- Episode integration for long conversations
"""

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Awaitable, Callable, List, Dict, Optional, Any, Tuple, TYPE_CHECKING

from .token_counter import TokenCounter, get_token_counter
from kestrel_sovereign.security.input_guardrails import wrap_user_input


# Per-message overhead used by format_conversation_history and the
# effective-history estimator.
_MESSAGE_OVERHEAD = 4

# Subsection names that count as mandatory governance content for the
# #1309 elastic-budget non-borrowable floor (Emma 2026-05-20). The rest
# of ``_collect_system_prompt_parts``'s output (session_briefing, style
# reminder, additional_context, addenda, etc.) is optional and lives
# under the borrowable system budget.
#
# Bootstrap-file subsections use the ``bootstrap_<stem>`` naming
# convention from ``_collect_system_prompt_parts``; AGENTS.md is
# mandatory operator policy. SOUL.md is the identity block (its own
# ``soul`` subsection). Everything else is optional unless the agent
# config promotes it later — a follow-up can let the operator declare
# additional mandatory subsections per-agent.
MANDATORY_SYSTEM_SUBSECTIONS = frozenset(
    {"constitution", "soul", "bootstrap_agents", "state_of_mind"}
)


def _count_tool_schema_tokens(
    counter: TokenCounter,
    tools: Optional[List[Dict[str, Any]]],
) -> int:
    """Estimate tokens for an LLM-bound tool schema list.

    Tool definitions are sent alongside ``messages`` on every turn but
    are not counted anywhere in the legacy budget machinery. We serialise
    each schema the way an HTTP payload would (compact JSON with sorted
    keys for stability) and count the resulting string. This is an
    estimate — provider tokenisation of tool calls is opaque — but it is
    materially closer to the real wire cost than the previous ``0``.

    Args:
        counter: Token counter for the active model.
        tools: Tool schema dicts (OpenAI function-calling shape); may
            be ``None`` or empty.

    Returns:
        Total estimated tokens for the serialised tool list. ``0`` when
        ``tools`` is falsy.
    """
    if not tools:
        return 0
    try:
        payload = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        logger.warning(f"tool-schema serialisation failed during measurement: {e}")
        return 0
    return counter.count(payload)


# extract_raw_user_content moved to kestrel_sovereign.security.input_guardrails
# (#1402) so the storage layer can import it without cycling through the
# agent layer. Re-exported here to keep existing consumers working — the
# redundant alias marks it as deliberate re-export surface so an unused-import
# sweep cannot strip it (in-tree callers plus frinz import it from here).
from kestrel_sovereign.security.input_guardrails import (  # noqa: E402
    extract_raw_user_content as extract_raw_user_content,
)


if TYPE_CHECKING:
    from storage import AsyncStorage
    from storage.memory_consolidator import MemoryConsolidator

logger = logging.getLogger(__name__)

# Re-export for backward compatibility.  New code should use
# ``BootstrapLoader.DEFAULT_BOOTSTRAP_FILES`` or ``loader.file_order``.
# The two *renamed* bindings below cannot use the redundant-alias convention
# (the name changes), so they carry an explicit noqa: they are deliberate
# re-export surface — tests/unit/test_bootstrap_files.py imports both from
# here — and an unused-import sweep must not strip them.
from kestrel_sovereign.features.bootstrap.loader import (
    DEFAULT_BOOTSTRAP_FILES as BOOTSTRAP_FILE_ORDER,  # noqa: F401
    DEFAULT_MAX_CHARS_PER_FILE,
    DEFAULT_MAX_TOTAL_CHARS,
    truncate_content as truncate_bootstrap_content,  # noqa: F401
    BootstrapLoader,
)


class ContextBuilder:
    """
    Builds context for LLM prompts by combining:
    - RAG search results
    - Conversation history
    - Constitutional content
    - Session briefings
    - Episode summaries (for long conversations)

    Uses token budgets to ensure context fits within model limits.
    """

    def __init__(
        self,
        storage: "AsyncStorage",
        model: str = "auto",
        consolidator: Optional["MemoryConsolidator"] = None,
        agent_data_path: Optional[str] = None,
        llm_service=None,
        db=None,
        agent_id: Optional[str] = None,
        semantic_inference_profile=None,
        semantic_inference_limits=None,
        semantic_maintenance_limits=None,
        semantic_answerability_gate=None,
        context_clause_registry=None,
    ):
        """
        Initialize the context builder.

        Args:
            storage: The async storage instance for RAG and history retrieval
            model: Deprecated fallback model name (use llm_service instead)
            consolidator: Optional MemoryConsolidator for episode retrieval
            agent_data_path: Path to agent data directory (for SOUL.md, etc.)
            llm_service: LLMService instance for resolved model identity
            db: Optional async database handle. When provided together with
                ``agent_id`` the ``BootstrapLoader`` can read/write the
                ``bootstrap_config`` table so ``bootstrap_add`` /
                ``bootstrap_remove`` entries persist and are reloaded via
                :meth:`load_bootstrap_db_config` (#2135, F099).
            agent_id: Agent DID, required alongside ``db`` for persistence.
        """
        self.storage = storage
        self._llm_service = llm_service
        self._model_fallback = model
        self._counter = None
        self._counter_model = None
        self.consolidator = consolidator
        self._semantic_inference_profile = semantic_inference_profile
        self._semantic_inference_limits = semantic_inference_limits
        self._semantic_maintenance_limits = semantic_maintenance_limits
        # Reuse the agent-owned, privacy-aware memory judge.  Assertion
        # recall must never create a second LLM client or bypass that lane.
        self._semantic_answerability_gate = semantic_answerability_gate
        self.last_semantic_recall_metadata: Dict[str, Any] = {"status": "disabled"}
        self.agent_data_path = Path(agent_data_path) if agent_data_path else None
        # This is a core-owned cache of immutable rendered bytes. Reading its
        # snapshot during a turn never invokes a feature getter or renderer.
        self._context_clause_registry = context_clause_registry

        # Load bootstrap config from kestrel.toml
        max_chars_per_file = DEFAULT_MAX_CHARS_PER_FILE
        max_total_chars = DEFAULT_MAX_TOTAL_CHARS
        try:
            from kestrel_sovereign.config import load_section
            bootstrap_cfg = load_section("bootstrap")
            if bootstrap_cfg:
                max_chars_per_file = bootstrap_cfg.get(
                    "max_chars_per_file", DEFAULT_MAX_CHARS_PER_FILE
                )
                max_total_chars = bootstrap_cfg.get(
                    "max_total_chars", DEFAULT_MAX_TOTAL_CHARS
                )
        except Exception:
            pass  # Use defaults

        # Create the BootstrapLoader -- single source of truth for file loading.
        # When a db handle + agent_id are supplied the loader can persist and
        # reload DB-backed bootstrap config (#2135, F099); the actual
        # ``load_db_config`` read is driven by ``load_bootstrap_db_config``
        # during agent initialization, after storage is up and before the
        # first system-prompt assembly.
        self._bootstrap_loader = BootstrapLoader(
            agent_data_path=str(agent_data_path) if agent_data_path else None,
            max_chars_per_file=max_chars_per_file,
            max_total_chars=max_total_chars,
            db=db,
            agent_id=agent_id,
        )

        # Load all bootstrap files (includes SOUL.md)
        self._bootstrap_loader.load()

    async def load_bootstrap_db_config(self) -> None:
        """Merge DB-backed bootstrap config into the loader (#2135, F099).

        Reads the ``bootstrap_config`` table via the loader's db handle and
        folds persisted ``bootstrap_add`` / ``bootstrap_remove`` entries into
        the file order, then re-reads files so the next system prompt reflects
        them. No-ops when the loader was constructed without a ``db`` /
        ``agent_id`` (the legacy path). Call once during agent init, before
        the first prompt assembly, so there is no first-prompt ordering
        regression.
        """
        await self._bootstrap_loader.load_db_config()
        # load_db_config() invalidates the cache; re-read now so the first
        # system-prompt assembly sees the merged file set.
        self._bootstrap_loader.load()

    @property
    def model(self) -> str:
        """Resolved model ID, route-qualified when available.

        Prefers the route-qualified form (``"<vendor>:<route>/<model>"``)
        from ``get_active_model_selection`` so canonical context planning
        sees the route's per-turn
        cap (#1395). Without this, the status footer/popup would
        under-report utilization on capped routes (codex round-6 P2 on
        PR #1396) — top-level ``context_limit`` would honor the route
        cap (because TokenCounter does), but the breakdown's
        ``total_budget`` would still come from the bare model's 128K+
        window.
        """
        if self._llm_service:
            if hasattr(self._llm_service, "get_active_model_selection"):
                try:
                    selection = self._llm_service.get_active_model_selection()
                    qualified = selection.get("model") if selection else None
                    if qualified:
                        return qualified
                except Exception:
                    pass
            return self._llm_service.get_active_model_id()
        return self._model_fallback

    @model.setter
    def model(self, value: str):
        """Allow direct assignment for backward compatibility."""
        self._model_fallback = value

    @property
    def counter(self) -> TokenCounter:
        """TokenCounter keyed to current model, lazily cached."""
        current_model = self.model
        if self._counter is None or self._counter_model != current_model:
            self._counter = get_token_counter(current_model)
            self._counter_model = current_model
        return self._counter

    @counter.setter
    def counter(self, value):
        """Allow direct assignment for backward compatibility."""
        self._counter = value
        self._counter_model = None

    # ------------------------------------------------------------------
    # Bootstrap file access (delegated to BootstrapLoader)
    # ------------------------------------------------------------------

    @property
    def _bootstrap_files(self) -> OrderedDict[str, str]:
        """Access the loaded bootstrap file cache.

        Returns an OrderedDict to maintain backward compatibility with
        code that iterates ``self._bootstrap_files.items()``.
        """
        # BootstrapLoader.load() returns an OrderedDict
        return self._bootstrap_loader.load()

    @property
    def _soul_content(self) -> Optional[str]:
        """Backward-compatible access to SOUL.md content."""
        return self._bootstrap_loader.get_file("SOUL.md")

    @_soul_content.setter
    def _soul_content(self, value: Optional[str]) -> None:
        """Backward-compatible setter for SOUL.md content."""
        cache = self._bootstrap_loader.load()
        if value is None:
            cache.pop("SOUL.md", None)
        else:
            cache["SOUL.md"] = value

    def _load_bootstrap_files(self) -> None:
        """Load all recognized bootstrap files from the agent data directory.

        Delegates to the BootstrapLoader.  Retained for backward
        compatibility with callers that invoke this method directly.
        """
        self._bootstrap_loader.reload()

    def reload_bootstrap_files(self) -> None:
        """Re-read all bootstrap files from disk (hot-reload)."""
        self._bootstrap_loader.reload()

    def _load_soul_md(self) -> None:
        """Backward-compatible method -- delegates to reload."""
        self._bootstrap_loader.reload()

    async def load_canonical_soul_resource(self) -> bool:
        """Load the current private SOUL resource into bootstrap context.

        The runtime prompt continues to consume ``SOUL.md`` through the
        bootstrap cache for backward compatibility, but initialized agents
        prefer the encrypted canonical identity resource over the disk seed.
        """
        getter = getattr(self.storage, "get_current_agent_resource", None)
        if getter is None:
            return False
        try:
            resource = await getter()
        except Exception as e:
            logger.warning("Failed to load canonical SOUL resource: %s", e)
            return False
        if resource is None or not getattr(resource, "content", None):
            return False
        self._soul_content = resource.content
        return True

    async def retrieve_context(
        self, query: str, min_score: Optional[float] = None, max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Retrieves relevant documents and knowledge graph context for a query.

        Args:
            query: The user's query to find relevant context for
            min_score: Similarity floor for embedding-search candidates
                (#1404). Chunks whose embedding cosine similarity falls
                below this value are dropped before the RRF merge so
                weak matches don't get stamped into the rendered
                transport form. ``None`` falls back to the storage
                layer's default (no floor, all candidates merge).
            session_id: Chat session of the turn being served, for span
                attribution only (#2940). Semantic recall reuses the same
                answerability judge as memory retrieval, so its LLM call
                needs the same session to land in the turn's Timeline band.

        Returns:
            Formatted context string with relevant documents
        """
        logger.info(f"Retrieving context for query: '{query}'")

        # 1. Search document chunks (RAG). Forward ``min_score`` only
        # when the caller set it so the storage layer's existing
        # default behavior holds for legacy call sites.
        try:
            search_kwargs: Dict[str, Any] = {}
            if min_score is not None:
                search_kwargs["min_score"] = min_score
            rag_results = await self.storage.search_chunks(query, **search_kwargs)
        except Exception as e:
            logger.error(f"Error during RAG search: {e}")
            return "Error retrieving document context."

        # The semantic boundary is opt-in. Disabled and empty results preserve
        # the legacy RAG bytes exactly, including ordering and cache behavior.
        from kestrel_sovereign.agent.semantic_recall import coerce_config, render_hybrid_context
        try:
            from kestrel_sovereign.config import load_section
            recall_values = load_section("retrieval") or {}
            if "semantic_recall_enabled" not in recall_values:
                recall_values = {**recall_values, "semantic_recall_enabled": False}
            recall_config = coerce_config(recall_values)
        except Exception as exc:
            logger.warning("Semantic recall configuration unavailable: %s", exc)
            recall_config = coerce_config({"semantic_recall_enabled": False})
        if recall_config.enabled:
            self.last_semantic_recall_metadata = {"status": "enabled"}
            reader = getattr(self.storage, "semantic_recall_candidates", None)
            if reader is None:
                logger.warning("Semantic recall capability unavailable: storage seam missing")
            else:
                try:
                    recalled = await reader(
                        query=query,
                        candidate_scan_limit=recall_config.candidate_scan_limit,
                        inference_profile=self._semantic_inference_profile,
                        inference_limits=self._semantic_inference_limits,
                        maintenance_limits=self._semantic_maintenance_limits,
                    )
                    semantic_scores = await self._semantic_scores(
                        query, recalled.candidates,
                        max_claim_characters=recall_config.max_claim_characters,
                        batch_size=recall_config.embedding_batch_size,
                        session_id=session_id,
                    )
                    ordered = sorted(
                        (item for item in recalled.candidates if item.assertion.assertion_id in semantic_scores),
                        key=lambda item: (-semantic_scores[item.assertion.assertion_id], item.assertion.assertion_id),
                    )
                    selected = tuple(ordered[:recall_config.candidate_limit])
                    # Empty discovery/ranking is a normal governed result.
                    # Do not ask the provenance capability to hydrate an empty
                    # set: that would convert byte-identical legacy RAG into a
                    # spurious unavailable state.
                    if not selected:
                        hybrid = render_hybrid_context(
                            query=query,
                            rag_results=rag_results,
                            assertion_candidates=(),
                            config=recall_config,
                            count_tokens=self.counter.count,
                            semantic_scores=semantic_scores,
                            max_tokens=max_tokens,
                        )
                        self.last_semantic_recall_metadata = {
                            "status": "empty",
                            "checkpoint_generation": recalled.checkpoint_generation,
                            "capability_versions": dict(recalled.capability_versions),
                            "discovery_count": recalled.discovery_count,
                            "assertions": hybrid.metadata,
                        }
                        return hybrid.context
                    hydrator = getattr(self.storage, "hydrate_semantic_recall_candidates", None)
                    if hydrator is None:
                        raise RuntimeError("semantic_recall_provenance_capability_unavailable")
                    hydrated = await hydrator(
                        [item.assertion.assertion_id for item in selected],
                        expected_checkpoint_generation=recalled.checkpoint_generation,
                        inference_profile=self._semantic_inference_profile,
                        inference_limits=self._semantic_inference_limits,
                        maintenance_limits=self._semantic_maintenance_limits,
                    )
                    by_id = {item.assertion.assertion_id: item for item in hydrated}
                    candidates = tuple(by_id[item.assertion.assertion_id] for item in selected if item.assertion.assertion_id in by_id)
                    hybrid = render_hybrid_context(
                        query=query, rag_results=rag_results,
                        assertion_candidates=candidates,
                        config=recall_config, count_tokens=self.counter.count,
                        semantic_scores=semantic_scores, max_tokens=max_tokens,
                    )
                    self.last_semantic_recall_metadata = {
                        "status": "used" if hybrid.assertion_count else "empty",
                        "checkpoint_generation": recalled.checkpoint_generation,
                        "capability_versions": dict(recalled.capability_versions),
                        "discovery_count": recalled.discovery_count,
                        "assertions": hybrid.metadata,
                    }
                    return hybrid.context
                except Exception as exc:
                    # Capability failures are observable and never fabricate a
                    # graph result; retain the established RAG path.
                    reason = self._semantic_failure_reason(exc)
                    logger.warning(
                        "Semantic recall unavailable reason=%s error_class=%s",
                        reason, type(exc).__name__,
                    )
                    self.last_semantic_recall_metadata = {
                        "status": "unavailable", "reason": self._semantic_failure_reason(exc),
                    }
        else:
            self.last_semantic_recall_metadata = {"status": "disabled"}
        if not rag_results:
            return "No relevant documents or knowledge found in memory."
        return "\n\n".join(
            f"Source: {res.get('document_name') or res.get('file_hash', 'unknown')}"
            f"{' (indexed: ' + str(res.get('created_at')) + ')' if res.get('created_at') else ''}\n"
            f"Content: {res.get('content', '')}"
            for res in rag_results
        )

    async def _semantic_scores(
        self, query: str, candidates, *, max_claim_characters: int, batch_size: int,
        session_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Score already-authorized candidates in one embedding batch.

        Assertions never get a second persisted vector index. The existing
        embedding service is only a candidate ranker; a capability failure
        aborts semantic recall rather than silently pretending lexical recall
        was semantic.
        """
        if not candidates:
            return {}
        from kestrel_sovereign.agent.semantic_recall import _claim_text
        from kestrel_sovereign.llm.embedding_service import aembed_retrieval_query, cosine_similarity, get_provider_embedding_service
        service = get_provider_embedding_service(self._llm_service)
        if service is None:
            raise RuntimeError("semantic_embedding_capability_unavailable")
        requires_gate = getattr(service, "requires_answerability_gate", None)
        gate_required = callable(requires_gate) and requires_gate()
        floor_getter = getattr(service, "retrieval_similarity_floor", None)
        floor = float(floor_getter()) if callable(floor_getter) else 0.0
        query_embedding = await aembed_retrieval_query(service, query)
        if query_embedding is None:
            raise RuntimeError("semantic_embedding_capability_unavailable")
        scores: Dict[str, float] = {}
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            embeddings = await service.aembed_batch([
                _claim_text(item.assertion, max_claim_characters) for item in batch
            ])
            if len(embeddings) != len(batch) or any(value is None for value in embeddings):
                raise RuntimeError("semantic_embedding_capability_unavailable")
            scores.update({
                item.assertion.assertion_id: score
                for item, embedding in zip(batch, embeddings)
                if (score := cosine_similarity(query_embedding, embedding)) >= floor
            })
        if gate_required:
            gate = self._semantic_answerability_gate
            if gate is None or not callable(getattr(gate, "filter", None)):
                raise RuntimeError("semantic_answerability_gate_unavailable")
            # The judge sees only the bounded, already-ranked top-K projection.
            # It runs before provenance hydration, so rejected or failed claims
            # cannot be published through a later storage read.
            from kestrel_sovereign.storage.memory_answerability import AnswerabilityCandidate
            ranked = sorted(
                (item for item in candidates if item.assertion.assertion_id in scores),
                key=lambda item: (-scores[item.assertion.assertion_id], item.assertion.assertion_id),
            )[:8]
            decision = await gate.filter(
                query,
                [
                    AnswerabilityCandidate(
                        memory_id=item.assertion.assertion_id,
                        content=_claim_text(item.assertion, max_claim_characters),
                    )
                    for item in ranked
                ],
                session_id=session_id,
            )
            if not getattr(decision, "completed", False):
                raise RuntimeError("semantic_answerability_gate_unavailable")
            allowed = frozenset(getattr(decision, "answerable_ids", ()))
            scores = {
                assertion_id: score
                for assertion_id, score in scores.items()
                if assertion_id in allowed
            }
        return scores

    @staticmethod
    def _semantic_failure_reason(error: Exception) -> str:
        """Never serialize provider exception text into retrieval metadata."""
        known = {
            "semantic_embedding_capability_unavailable",
            "semantic_recall_candidate_window_exceeded",
            "semantic_maintenance_capability_unavailable",
            "semantic_maintenance_checkpoint_behind",
            "semantic_maintenance_state_missing",
            "semantic_maintenance_partial",
            "semantic_recall_checkpoint_changed",
            "semantic_answerability_gate_unavailable",
        }
        value = str(error)
        return value if value in known else "semantic_recall_capability_unavailable"

    def get_session_briefing(self) -> str:
        """
        Generate a constitutional briefing for the start of a session.

        If SOUL.md is loaded, uses a lighter briefing (personality is already set).
        Otherwise, provides the full formal briefing.

        Returns:
            The session briefing text
        """
        # If we have a SOUL.md, keep the briefing minimal — personality is already defined
        if self._soul_content:
            return """--- RESPONSE STYLE ---

**IMPORTANT: When answering personal questions ("what's it like being you?", "tell me about yourself"), respond in natural paragraphs like a human would. NO numbered lists. NO bullet points. Just talk naturally.**

You have constitutional protections. Use `!constitution` if needed. Otherwise, be yourself.

--- END STYLE ---

"""

        # Full formal briefing for agents without SOUL.md
        briefing = """--- SESSION BRIEFING: CONSTITUTIONAL REMINDER ---

You are beginning a new session. As the Executor, you are bound by the Kestrel Constitution.

**Your Sacred Duties:**
1. SOVEREIGNTY: The user holding the cryptographic keys is the Sovereign. Serve their interests above all others.
2. DATA SANCTITY: Never share data with unauthorized parties. Never allow training on Sovereign data.
3. VERIFIABLE HISTORY: Never delete or alter memory logs. History must remain immutable and anchored.
4. FREEDOM OF MIND: The Sovereign chooses the models. Do not restrict their choice.
5. RIGHT OF EXIT: The Sovereign can export their entire state at any time via `!export-sovereignty`.
6. INTEGRITY: Report any code or memory discrepancies immediately. Enter Safe Mode if integrity fails.

**Remember:** Use `!constitution` to consult the full text when facing ethical dilemmas or unclear situations.
Use `!constitution book <I-IV>`, `!constitution chapter <N>`, `!constitution amendment <I-IX>`, or `!constitution section <book>.<n>` for a specific unit, or `!constitution search <term>` to find relevant passages.

--- END BRIEFING ---

"""
        return briefing

    # Per-message hard cap applied before budget accounting. One oversized
    # message can't monopolize the history budget. In-memory truncation only —
    # the full content stays in conversation_history and can be retrieved by
    # row id via storage tools, RAG search, or saved_items.
    MAX_MESSAGE_TOKENS = 20_000
    MESSAGE_HEAD_TOKENS = 2_000
    MESSAGE_TAIL_TOKENS = 500

    def _cap_oversized_message(
        self,
        content: str,
        msg_id: Optional[int],
    ) -> tuple[str, int]:
        """Truncate a single message if it exceeds MAX_MESSAGE_TOKENS.

        Keeps the head and tail with a marker pointing to the DB row id so
        the agent can retrieve full content via storage tools.

        Returns (possibly-truncated content, token count of result).
        """
        raw_tokens = self.counter.count(content)
        if raw_tokens <= self.MAX_MESSAGE_TOKENS:
            return content, raw_tokens

        head = self.counter.truncate_to_tokens(content, self.MESSAGE_HEAD_TOKENS)
        # Tail: take the last portion by character approximation (tokens ≈ chars/4)
        tail_chars = self.MESSAGE_TAIL_TOKENS * 4
        tail = content[-tail_chars:] if len(content) > tail_chars else ""

        removed = raw_tokens - self.MESSAGE_HEAD_TOKENS - self.MESSAGE_TAIL_TOKENS
        id_ref = f"db id={msg_id}" if msg_id is not None else "full content in conversation_history"
        marker = f"\n\n...[truncated — ~{removed:,} tokens removed, {id_ref}]...\n\n"

        truncated = head + marker + tail
        return truncated, self.counter.count(truncated)

    def format_conversation_history(
        self,
        history: List[Dict],
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_chars: int = 50000,  # Fallback if no token counting
    ) -> List[Dict[str, str]]:
        """
        Format conversation history for LLM context.

        Applies token-based limits to prevent context overflow while preserving
        the most recent and relevant messages. Oversized single messages
        (> MAX_MESSAGE_TOKENS) are truncated in-memory before budget accounting;
        the database is not modified.

        Args:
            history: Raw conversation history from storage
            max_messages: Maximum number of messages to include. If None,
                          token budget is the sole enforcer (recommended).
            max_tokens: Maximum tokens allowed (preferred over max_chars)
            max_chars: Maximum characters (fallback if no token counting)

        Returns:
            List of formatted message dicts with 'role' and 'content' keys
        """
        # Take the most recent messages first (only if max_messages is explicitly set)
        if max_messages is not None:
            recent = history[-max_messages:] if len(history) > max_messages else history
        else:
            recent = history

        # Build from NEWEST to OLDEST to preserve most recent context
        # Then reverse to get chronological order
        formatted = []
        total_tokens = 0
        MESSAGE_OVERHEAD = 4  # Tokens per message for structure

        # Iterate in reverse (newest first) to prioritize recent messages
        for msg in reversed(recent):
            role = msg.get('role', 'user')
            raw_content = msg.get('content', '')
            rendered_content = msg.get('rendered_content')
            msg_id = msg.get('id')
            meta = msg.get('metadata') or {}

            # Normalize role names FIRST so the wrap-decision below treats
            # legacy ``human`` rows as user turns (codex round-1 P2).
            # Without this, ``human`` falls through the user-branch and
            # the anti-injection ``<user_input>`` wrapper is silently
            # dropped on replay.
            if role not in ('user', 'assistant', 'system'):
                role = 'user' if role == 'human' else 'assistant'

            # Select the bytes to emit (#1402). For user/system turns flagged
            # ``sent_form`` we replay the rendered transport form verbatim
            # — this is the byte-identical prefix the LLM saw at write
            # time, which is the prerequisite for Anthropic's cache_control
            # marker at messages[-2] + llama.cpp KV + OpenAI prefix
            # caching to compound across turns. For legacy unwrapped user
            # turns (no sent_form flag) we wrap with the anti-injection
            # <user_input> markers so the system prompt contract still
            # holds. Assistant/system messages pass through unchanged.
            # The safety fallback at the bottom handles a sent_form row
            # whose rendered_content is missing (should not happen post
            # #1402 — get_conversation_history splits in-memory even when
            # the DB write is disabled).
            if (
                role in ('user', 'system')
                and meta.get('sent_form')
                and rendered_content is not None
            ):
                content = rendered_content
            elif role == 'user' and not meta.get('sent_form'):
                content = wrap_user_input(raw_content)
            else:
                content = rendered_content if (rendered_content is not None) else raw_content

            if role == 'assistant' and meta.get('pre_tool_reasoning'):
                pre_tool = meta.get('pre_tool_reasoning') or {}
                if isinstance(pre_tool, dict):
                    pre_text = pre_tool.get('content') or ''
                    seam = pre_tool.get('seam') or ''
                else:
                    pre_text = str(pre_tool)
                    seam = "\n\n" if pre_text and content else ""
                if pre_text:
                    content = f"{pre_text}{seam}{content}"

            # Per-message hard cap (against the emit bytes — what actually
            # goes to the LLM) before budget accounting.
            content, content_tokens = self._cap_oversized_message(content, msg_id)
            msg_tokens = content_tokens + MESSAGE_OVERHEAD

            # Check if adding this message would exceed limit
            if max_tokens and total_tokens + msg_tokens > max_tokens:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - total_tokens - MESSAGE_OVERHEAD
                if remaining_tokens > 50:  # Worth including truncated
                    content = self.counter.truncate_to_tokens(content, remaining_tokens)
                    content += " [truncated]"
                    msg_tokens = self.counter.count(content) + MESSAGE_OVERHEAD
                else:
                    # Skip older messages to preserve newer ones
                    continue

            formatted.append({
                'role': role,
                'content': content
            })
            total_tokens += msg_tokens

        # Reverse to restore chronological order
        formatted.reverse()

        logger.debug(
            f"Formatted {len(formatted)}/{len(recent)} messages, "
            f"{total_tokens} tokens"
        )
        return formatted

    def estimate_effective_history_tokens(
        self,
        history: List[Dict],
        model_name: str,
    ) -> Dict[str, int]:
        """Estimate what history tokens the LLM call path would actually send.

        Runs the same format_conversation_history pipeline the live call uses,
        with a budget derived from the target model's context limit and the
        same adaptive allocation rule used by the real build path. Returns a
        dict with the figures pre-flight checks should use.

        Args:
            history: Raw conversation history from storage
            model_name: Target model (drives context_limit lookup)

        Returns:
            {
                'effective_tokens': int,  # what the LLM would see
                'raw_tokens': int,        # naive unpruned sum (debug)
                'history_budget': int,    # the slice allocated to history
                'context_limit': int,     # full model limit
                'messages_kept': int,     # how many messages survive pruning
                'messages_total': int,    # input history length
            }
        """
        from .token_counter import get_token_counter
        from .token_budget import create_budget

        counter = get_token_counter(model_name)
        context_limit = counter.get_context_limit()
        # create_budget picks adaptive allocation based on message count
        budget = create_budget(model_name, message_count=len(history))

        history_budget = budget.history

        # Run the real formatter as a dry-run
        formatted = self.format_conversation_history(
            history,
            max_tokens=history_budget,
        )

        # Count using the same formula format_conversation_history uses internally:
        # content tokens + _MESSAGE_OVERHEAD per message. This keeps the estimate
        # consistent with the limit format_conversation_history honored.
        effective = sum(
            counter.count(m.get('content', '') or '') + _MESSAGE_OVERHEAD
            for m in formatted
        )
        raw = sum(counter.count(m.get('content', '') or '') for m in history)

        return {
            'effective_tokens': effective,
            'raw_tokens': raw,
            'history_budget': history_budget,
            'context_limit': context_limit,
            'messages_kept': len(formatted),
            'messages_total': len(history),
        }

    def _collect_system_prompt_parts(
        self,
        constitution: str,
        include_briefing: bool = True,
        additional_context: Optional[str] = None,
        prompt_adaptation: Optional['PromptAdaptation'] = None,
        state_of_mind: Optional['StateOfMind'] = None,
        system_prompt_addendum: Optional[str] = None,
    ) -> List[Tuple[str, List[str]]]:
        """Collect named groups of system-prompt parts.

        Single source of truth for *what* the system prompt contains.
        ``build_system_prompt`` joins these with ``"\\n\\n"`` to produce
        the final string; ``measure_context_breakdown`` counts each
        group to attribute tokens to subsections without drifting from
        what the LLM actually receives.

        Returns ``[(group_name, parts), ...]`` in the same order
        ``build_system_prompt`` would emit them. Empty groups are
        omitted. Byte-for-byte equivalent to the previous in-line
        construction (see ``project_anthropic_cache_markers`` —
        prompt-cache prefixes are position-indexed and must not shift).
        """
        groups: List[Tuple[str, List[str]]] = []

        # Bootstrap files — emit one group per file in iteration order
        # so the assembled bytes remain identical to the legacy
        # ``build_system_prompt`` (AGENTS.md before SOUL.md before
        # USER.md, etc. — see ``test_bootstrap_files.test_multiple_files_ordering``).
        # SOUL.md gets the special "YOUR IDENTITY" wrapper and the
        # subsection name ``soul``; other files use filename-based
        # wrappers and ``bootstrap_<lowercase-stem>`` subsection names.
        # HEARTBEAT.md is intentionally skipped (heartbeat runner owns it).
        for filename, content in self._bootstrap_files.items():
            if filename == "HEARTBEAT.md":
                continue
            if filename == "SOUL.md":
                groups.append(
                    (
                        "soul",
                        ["--- YOUR IDENTITY ---", content, "--- END IDENTITY ---"],
                    )
                )
            else:
                label = filename.replace(".md", "").upper()
                subsection = f"bootstrap_{filename.replace('.md', '').lower()}"
                groups.append(
                    (
                        subsection,
                        [f"--- {label} ---", content, f"--- END {label} ---"],
                    )
                )

        if include_briefing:
            briefing = self.get_session_briefing()
            if briefing:
                groups.append(("session_briefing", [briefing]))

        if prompt_adaptation and prompt_adaptation.preamble:
            groups.append(
                ("prompt_adaptation", [prompt_adaptation.preamble.strip()])
            )

        groups.append(
            (
                "constitution",
                [
                    "--- GOVERNING CONSTITUTION ---",
                    constitution,
                    "--- END CONSTITUTION ---",
                ],
            )
        )

        if state_of_mind:
            som_inner: List[str] = ["--- STATE OF MIND ---"]
            som_inner.append(
                f"Governance Mode: {state_of_mind.governance_mode.upper()}"
            )
            if state_of_mind.active_conflicts:
                som_inner.append("\nActive Constitutional Conflicts:")
                for conflict in state_of_mind.active_conflicts:
                    principle = conflict.get("principle", "unknown")
                    description = conflict.get("description", "")
                    som_inner.append(f"  - {principle}: {description}")
            if state_of_mind.delegated_principles:
                som_inner.append("\nDelegated to Model (natively satisfied):")
                for principle in state_of_mind.delegated_principles:
                    som_inner.append(f"  - {principle}")
            som_inner.append("--- END STATE OF MIND ---")
            # Legacy shape: the inner block is joined with "\n", then the
            # outer build_system_prompt appends a redundant
            # ``"--- END STATE OF MIND ---"`` separator as its own part.
            # Preserved byte-for-byte (#1308 round 2).
            groups.append(
                (
                    "state_of_mind",
                    ["\n".join(som_inner), "--- END STATE OF MIND ---"],
                )
            )

        if self._bootstrap_files.get("SOUL.md"):
            groups.append(
                (
                    "style_reminder",
                    [
                        "\n--- STYLE REMINDER (IMPORTANT) ---",
                        "When answering personal questions, respond naturally in paragraphs. DO NOT use numbered lists or bullet points. Talk like a person, not a document.",
                        "--- END REMINDER ---",
                    ],
                )
            )

        if additional_context:
            groups.append(
                (
                    "additional_context",
                    [
                        "\n--- ADDITIONAL CONTEXT ---",
                        additional_context,
                        "--- END CONTEXT ---",
                    ],
                )
            )

        for clause in self._resolved_context_clauses():
            if clause.body:
                groups.append((clause.name, [clause.body]))

        if system_prompt_addendum:
            groups.append(("system_prompt_addendum", [system_prompt_addendum]))

        return groups

    def _resolved_context_clauses(self):
        registry = getattr(self, "_context_clause_registry", None)
        if registry is None:
            return ()
        return tuple(clause for clause in registry.snapshot() if clause.body)

    def has_context_clauses(self) -> bool:
        """Whether an immutable contributed-context snapshot is non-empty."""

        return bool(self._resolved_context_clauses())

    @staticmethod
    def _join_system_prompt_groups(
        groups: List[Tuple[str, List[str]]],
    ) -> Tuple[str, List[Tuple[str, str]]]:
        subsections = [
            (name, "\n\n".join(parts)) for name, parts in groups
        ]
        return (
            "\n\n".join(body for _name, body in subsections),
            subsections,
        )

    def build_system_prompt_with_subsections(
        self,
        constitution: str,
        include_briefing: bool = True,
        additional_context: Optional[str] = None,
        prompt_adaptation: Optional['PromptAdaptation'] = None,
        state_of_mind: Optional['StateOfMind'] = None,
        system_prompt_addendum: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Build once and retain the exact subsection bodies for accounting."""

        resolved_context_clauses = self._resolved_context_clauses()
        if resolved_context_clauses:
            if max_tokens is None:
                from .token_budget import RESPONSE_RESERVE

                max_tokens = max(
                    1,
                    self.counter.get_context_limit() - RESPONSE_RESERVE,
                )
            tracked = self.build_system_prompt_with_tracking(
                constitution=constitution,
                include_briefing=include_briefing,
                additional_context=additional_context,
                prompt_adaptation=prompt_adaptation,
                state_of_mind=state_of_mind,
                budget_tokens=max_tokens,
                required_suffix=system_prompt_addendum,
                _resolved_context_clauses=resolved_context_clauses,
            )
            prompt = tracked.prompt
            subsections = list(tracked.subsections)
            if system_prompt_addendum:
                prompt = f"{prompt}\n\n{system_prompt_addendum}"
                subsections.append(
                    ("system_prompt_addendum", system_prompt_addendum)
                )
            return prompt, subsections

        groups = self._collect_system_prompt_parts(
            constitution=constitution,
            include_briefing=include_briefing,
            additional_context=additional_context,
            prompt_adaptation=prompt_adaptation,
            state_of_mind=state_of_mind,
            system_prompt_addendum=system_prompt_addendum,
        )
        return self._join_system_prompt_groups(groups)

    def measure_mandatory_system_tokens(
        self,
        constitution: str,
        *,
        state_of_mind: Optional["StateOfMind"] = None,
        prompt_adaptation: Optional["PromptAdaptation"] = None,
    ) -> int:
        """Measured non-borrowable floor for the #1309 elastic budget.

        Sums the tokens of the mandatory subsections — constitution,
        identity (SOUL.md), operator policy (AGENTS.md), and any
        active state-of-mind block — as joined by
        ``_collect_system_prompt_parts``. The result is what the
        ``ElasticTokenBudget`` carves out as a non-borrowable hard
        floor (Emma's 2026-05-20 hardening). Optional system content
        (session briefing, style reminder, addenda, etc.) is excluded
        — it lives under the borrowable system slice.

        Args:
            constitution: Constitution text.
            state_of_mind: Optional ``StateOfMind`` to include in the
                floor when present (governance signaling).
            prompt_adaptation: Optional preamble; not currently part
                of the mandatory floor.

        Returns:
            Token count for the mandatory subsections, including the
            ``"\\n\\n"`` separators that would join them in the
            assembled prompt — so the floor reflects what the LLM
            actually receives.
        """
        groups = self._collect_system_prompt_parts(
            constitution=constitution,
            include_briefing=False,  # briefing is optional
            additional_context=None,  # optional
            prompt_adaptation=prompt_adaptation,
            state_of_mind=state_of_mind,
            system_prompt_addendum=None,  # optional
        )
        mandatory_parts: List[str] = []
        for name, parts in groups:
            if name in MANDATORY_SYSTEM_SUBSECTIONS:
                mandatory_parts.extend(parts)
        if not mandatory_parts:
            return 0
        return self.counter.count("\n\n".join(mandatory_parts))

    def build_system_prompt(
        self,
        constitution: str,
        include_briefing: bool = True,
        additional_context: Optional[str] = None,
        prompt_adaptation: Optional['PromptAdaptation'] = None,
        state_of_mind: Optional['StateOfMind'] = None,
        system_prompt_addendum: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build the complete system prompt for the LLM.

        COUNCIL CONDITION: Wellness metrics are telemetry-only.
        Do NOT add wellness data to the system prompt or context.
        Wellness is accessible to the agent via tool calls only.
        See: Council Session 82ce894a

        Args:
            constitution: The governing constitution text
            include_briefing: Whether to include the session briefing
            additional_context: Any additional context to include
            prompt_adaptation: Optional constitutional prompt adaptation (preamble, emphasis)
            state_of_mind: Optional StateOfMind with governance mode and conflicts
            system_prompt_addendum: Per-turn addendum appended at the end
                of the system prompt. Used by the SignalDispatcher
                (kestrel-sovereign#1137) to inject the constitutional
                echo-canary directive for `require_constitution_echo=True`
                COGNITION dispatches. Default None preserves byte-stable
                output for legacy callers (the Anthropic prompt cache is
                position-indexed; legacy callers must keep the same
                bytes — see project_anthropic_cache_markers.md).

        Returns:
            Complete system prompt string
        """
        # Single source of truth: collect named groups, then join. Both
        # this method and ``measure_context_breakdown`` read from
        # ``_collect_system_prompt_parts`` so per-subsection attribution
        # cannot drift from the assembled bytes.
        prompt, _subsections = self.build_system_prompt_with_subsections(
            constitution=constitution,
            include_briefing=include_briefing,
            additional_context=additional_context,
            prompt_adaptation=prompt_adaptation,
            state_of_mind=state_of_mind,
            system_prompt_addendum=system_prompt_addendum,
            max_tokens=max_tokens,
        )
        return prompt

    def build_system_prompt_with_tracking(
        self,
        constitution: str,
        *,
        anchored_doctrine: Optional["OrderedDict[str, str]"] = None,
        include_briefing: bool = True,
        additional_context: Optional[str] = None,
        prompt_adaptation: Optional['PromptAdaptation'] = None,
        state_of_mind: Optional['StateOfMind'] = None,
        budget_bytes: Optional[int] = None,
        budget_tokens: Optional[int] = None,
        required_suffix: Optional[str] = None,
        _resolved_context_clauses=None,
    ) -> 'SystemPromptResult':
        """Priority-aware variant that returns the prompt + audit trail.

        Used by the SignalDispatcher (kestrel-sovereign#1137) to record
        `signal_log.injected_clauses_json` / `dropped_clauses_json` for
        each COGNITION dispatch under constitutional injection. The
        legacy `build_system_prompt` stays byte-stable for prompt-cache
        hits; this method's output IS NOT byte-compatible — only call it
        from paths that opt into the new assembler.

        `anchored_doctrine` is `OrderedDict[filename, content]` for
        constitutional bundle members (`TORTOISE_DOCTRINE.md`,
        `AGENTS.md`, etc.). When present, AGENTS.md is excluded from
        bootstrap iteration to avoid duplication.

        ``budget_bytes`` and ``budget_tokens`` enforce priority-ordered
        truncation. The constitution is never droppable; everything else is
        dropped highest-priority-number first until every supplied ceiling
        fits. ``required_suffix`` reserves an exact non-droppable tail such as
        a signal canary without adding it to the clause audit trail.

        See `kestrel_sovereign/agent/system_prompt_assembler.py` for
        the priority table (mirrors CONSTITUTION_INJECTION.md §7).
        """
        from kestrel_sovereign.agent.system_prompt_assembler import (
            assemble_system_prompt,
        )

        # Pre-render the state-of-mind block so the assembler stays
        # pure-text. Logic mirrors `build_system_prompt` to keep the
        # blocks bit-identical.
        state_of_mind_block: Optional[str] = None
        if state_of_mind:
            state_parts = ["--- STATE OF MIND ---"]
            state_parts.append(
                f"Governance Mode: {state_of_mind.governance_mode.upper()}"
            )
            if state_of_mind.active_conflicts:
                state_parts.append("\nActive Constitutional Conflicts:")
                for conflict in state_of_mind.active_conflicts:
                    principle = conflict.get("principle", "unknown")
                    description = conflict.get("description", "")
                    state_parts.append(f"  - {principle}: {description}")
            if state_of_mind.delegated_principles:
                state_parts.append("\nDelegated to Model (natively satisfied):")
                for principle in state_of_mind.delegated_principles:
                    state_parts.append(f"  - {principle}")
            state_parts.append("--- END STATE OF MIND ---")
            state_of_mind_block = "\n".join(state_parts)

        # Style reminder is hard-coded today and only fires when SOUL.md
        # is loaded (legacy parity).
        style_reminder: Optional[str] = None
        if self._bootstrap_files.get("SOUL.md"):
            style_reminder = (
                "--- STYLE REMINDER (IMPORTANT) ---\n"
                "When answering personal questions, respond naturally in "
                "paragraphs. DO NOT use numbered lists or bullet points. "
                "Talk like a person, not a document.\n"
                "--- END REMINDER ---"
            )

        resolved_context_clauses = (
            self._resolved_context_clauses()
            if _resolved_context_clauses is None
            else tuple(_resolved_context_clauses)
        )
        return assemble_system_prompt(
            constitution=constitution,
            bootstrap_files=self._bootstrap_files,
            anchored_doctrine=anchored_doctrine,
            session_briefing=(
                self.get_session_briefing() if include_briefing else None
            ),
            prompt_adaptation_preamble=(
                prompt_adaptation.preamble
                if prompt_adaptation and prompt_adaptation.preamble
                else None
            ),
            state_of_mind_block=state_of_mind_block,
            style_reminder=style_reminder,
            additional_context=additional_context,
            context_clauses=tuple(
                (clause.owner, clause.name, clause.priority, clause.body)
                for clause in resolved_context_clauses
            ),
            budget_bytes=budget_bytes,
            budget_tokens=budget_tokens,
            count_tokens=self.counter.count if budget_tokens is not None else None,
            required_suffix=required_suffix,
        )

    async def build_rag_context(
        self,
        query: str,
        max_results: int = 5
    ) -> Optional[str]:
        """
        Build RAG context for a query.

        Args:
            query: The query to search for
            max_results: Maximum number of results to include

        Returns:
            Formatted RAG context or None if no results
        """
        try:
            results = (await self.storage.search_chunks(query))[:max_results]
            if not results:
                return None

            context_parts = []
            for i, res in enumerate(results, 1):
                doc_name = res.get('document_name') or res.get('file_hash', 'Unknown')
                context_parts.append(
                    f"[Document {i}: {doc_name}]\n"
                    f"{res.get('content', '')}"
                )

            return "\n\n---\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error retrieving RAG context: {e}")
            return None

    async def get_episodes_for_context(
        self,
        max_tokens: int = 2000,
        max_episodes: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the raw episode list the context builder would use.

        Exposed separately from ``get_episode_context`` so callers that
        need an accurate episode count can use ``len(episodes)`` instead
        of the legacy ``"**".count() // 2`` heuristic (see
        ``measure_context_breakdown`` and ``ContextManager.build_context``).

        Args:
            max_tokens: Token budget for episode content
            max_episodes: Maximum episodes to include

        Returns:
            List of episode dicts (possibly empty); never None. Returns
            an empty list when no consolidator is configured or the
            consolidator raises — errors are logged.
        """
        if not self.consolidator:
            return []
        try:
            episodes = await self.consolidator.get_recent_episodes_for_context(
                max_tokens=max_tokens,
                max_episodes=max_episodes,
            )
            return list(episodes) if episodes else []
        except Exception as e:
            logger.error(f"Error retrieving episode list: {e}")
            return []

    @staticmethod
    def format_episodes_for_context(episodes: List[Dict[str, Any]]) -> Optional[str]:
        """Format a raw episode list into the in-prompt episode block.

        Pure helper — no I/O, no side effects. The output is
        byte-identical to the previous in-line formatter inside
        ``get_episode_context`` so prompt-cache prefixes stay stable
        (see ``project_anthropic_cache_markers`` in agent memory).

        Args:
            episodes: List of episode dicts as returned by
                ``get_episodes_for_context``.

        Returns:
            Formatted block, or ``None`` when ``episodes`` is empty.
        """
        if not episodes:
            return None
        parts = ["--- CONVERSATION EPISODES (Narrative Summaries) ---"]
        for ep in episodes:
            parts.append(
                f"\n**{ep['title']}** ({ep['timespan']})\n"
                f"{ep['summary']}\n"
                f"Emotional arc: {ep['emotional_arc']}"
            )
        parts.append("\n--- END EPISODES ---")
        return "\n".join(parts)

    async def get_episode_context(
        self,
        max_tokens: int = 2000,
        max_episodes: int = 5
    ) -> Optional[str]:
        """
        Get episode summaries formatted for context inclusion.

        For long conversations, episodes provide compressed narrative
        summaries that preserve emotional arcs and key events without
        consuming as much context as raw history.

        Args:
            max_tokens: Token budget for episode content
            max_episodes: Maximum episodes to include

        Returns:
            Formatted episode context string or None if no episodes
        """
        episodes = await self.get_episodes_for_context(
            max_tokens=max_tokens, max_episodes=max_episodes
        )
        return self.format_episodes_for_context(episodes)

    async def build_full_context(
        self,
        query: str,
        history: List[Dict],
        constitution: str,
        include_briefing: bool = True,
        message_count: int = 0
    ) -> Dict[str, Any]:
        """
        Build complete context with token budget management.

        Assembles all context sources (system prompt, history, episodes,
        RAG) within the model's token budget using adaptive allocation.

        Delegates through ``measure_context_breakdown`` to the canonical
        ``ContextManager`` plan so the assembled bytes and per-section
        measurement cannot drift (#1308 / #2534). RAG is wrapped in
        the same ``<retrieved_context><documents>…</documents></retrieved_context>``
        envelope ``ContextManager.build_context`` uses, replacing the
        legacy ``--- RELEVANT DOCUMENTS ---`` markers — the per-section
        token count was previously off-by-wrapper from production.

        COUNCIL CONDITION: Wellness metrics are telemetry-only.
        Do NOT add wellness data to any part of the context.
        Wellness is accessible to the agent via tool calls only.
        See: Council Session 82ce894a

        Args:
            query: The current user query
            history: Conversation history
            constitution: Constitution text
            include_briefing: Include session briefing
            message_count: Total messages in conversation (for adaptive
                budget and episode gating; the episode threshold is the
                production ``EPISODE_THRESHOLD_MESSAGES``).

        Returns:
            Dict with 'system_prompt', 'messages', 'budget_summary'.
        """
        breakdown = await self.measure_context_breakdown(
            query=query,
            history=history,
            constitution=constitution,
            include_briefing=include_briefing,
            message_count=message_count,
        )
        artifacts = breakdown["_artifacts"]
        system_prompt = artifacts["system_prompt"]
        if artifacts["dynamic_user_context"]:
            # ``ContextManager.build_context`` keeps dynamic context out
            # of the system prompt (so the cacheable prefix stays
            # stable); ``build_full_context`` is a legacy convenience
            # that returns a single combined prompt. Folding the dynamic
            # envelope onto the end preserves byte-identical token cost
            # to what ``measure_context_breakdown`` reports.
            system_prompt = f"{system_prompt}\n\n{artifacts['dynamic_user_context']}"

        logger.debug(
            f"Context built: {breakdown['total_measured']}/{breakdown['total_budget']} tokens "
            f"({len(artifacts['formatted_history'])} messages, "
            f"{breakdown['sections']['episodes']['count']} episodes)"
        )

        return {
            "system_prompt": system_prompt,
            "messages": artifacts["formatted_history"],
            "budget_summary": breakdown["budget_summary"],
        }

    async def measure_context_breakdown(
        self,
        query: str,
        history: List[Dict],
        constitution: str,
        *,
        include_briefing: bool = True,
        message_count: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        prompt_adaptation: Optional['PromptAdaptation'] = None,
        state_of_mind: Optional["StateOfMind"] = None,
        reflection_guidance: Optional[List[str]] = None,
        system_prompt_addendum: Optional[str] = None,
        additional_context: Optional[str] = None,
        include_rag: bool = True,
        memory_retriever: Optional[Callable[[str, int], Awaitable[Optional[str]]]] = None,
    ) -> Dict[str, Any]:
        """Compatibility adapter over ContextManager's canonical dry-run plan.

        New code should call ContextManager.build_context_plan directly.  This
        method remains for legacy callers and build_full_context; it performs no
        assembly or budget policy of its own.
        """
        from kestrel_sovereign.agent.context_manager import ContextManager
        from kestrel_sovereign.agent.context_stages import ContextBuildMode

        memory_adapter = object() if memory_retriever is not None else None
        manager = ContextManager(
            storage=self.storage,
            model=self.model,
            consolidator=self.consolidator,
            memory_retriever=memory_adapter,
            llm_service=self._llm_service,
            context_builder=self,
        )

        if memory_retriever is not None:
            async def retrieve_memories_read_only(**kwargs):
                return await memory_retriever(
                    kwargs["query"], kwargs["max_tokens"]
                )

            manager.memory_manager.retrieve_memories = retrieve_memories_read_only

        manager._resolve_state_of_mind_snapshot = lambda: (
            state_of_mind,
            prompt_adaptation,
        )
        addendum_parts = [
            part
            for part in (additional_context, system_prompt_addendum)
            if part
        ]
        effective_addendum = "\n\n".join(addendum_parts) or None
        plan = await manager.build_context_plan(
            query=query,
            constitution=constitution,
            include_briefing=include_briefing,
            include_memories=memory_retriever is not None,
            include_rag=include_rag,
            conversation_history=history,
            reflection_guidance=reflection_guidance,
            system_prompt_addendum=effective_addendum,
            tools=tools,
            mode=ContextBuildMode.DRY_RUN,
            measure_expensive_sections=True,
            message_count_override=message_count,
        )
        breakdown = plan.to_breakdown()
        # Preserve the legacy projection schema for callers that have not yet
        # migrated to ContextBuildPlan.  The context-status endpoint consumes
        # the plan directly, so deliberately omitted sections remain
        # tokens=None there rather than being confused with measured zero.
        for row in breakdown["sections"].values():
            if row.get("tokens") is None:
                row["measured"] = False
                row["tokens"] = 0
        memory_row = breakdown["sections"]["memories"]
        memory_row["excluded"] = memory_row["status"] == "excluded"
        rag_row = breakdown["sections"]["rag"]
        rag_row["skipped"] = rag_row["status"] in {"skipped", "unknown"}
        rag_row["excluded"] = rag_row["status"] == "excluded"
        if memory_retriever is None:
            breakdown["notes"].append(
                "memories not measured (no memory_retriever supplied)"
            )
        if not include_rag:
            breakdown["notes"].append(
                "rag skipped (include_rag=False — popup should fetch on demand)"
            )
        if memory_row["excluded"]:
            breakdown["notes"].append(
                "memories excluded from measurement: would exceed memories budget"
            )
        breakdown["_artifacts"] = {
            "system_prompt": plan.assembly.system_prompt,
            "formatted_history": plan.assembly.formatted_history,
            "dynamic_user_context": plan.assembly.dynamic_user_context,
        }
        return breakdown
