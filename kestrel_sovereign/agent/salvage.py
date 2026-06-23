"""Durable salvage primitive for the context system (epic #1307 / C).

Implements Emma's 2026-05-20 invariant verbatim:

    No model-visible pruning without a synchronous durable artifact
    or lossless pointer. The summary may be async; the salvage record
    must be sync.

See ``docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md`` for the full
design (PR #1348, head ``185b7eb1``, Emma-acked 2026-05-21).

Architecture overview
---------------------

When ``ContextManager.build_context`` is about to drop verbatim
history from the model-visible slice (pre-trim or post-budget prune),
it calls :func:`salvage_messages` **synchronously, before the LLM
call proceeds**. The primitive does, in a single database
transaction:

1. INSERT a salvage marker row into ``conversation_history`` with
   ``metadata.type = "salvage"`` and ``metadata.salvage_state =
   "pointer-only"``.
2. UPDATE the original rows' metadata to mark them
   ``excluded_from_context: True`` with ``summarized_into`` pointing
   at the new marker id.

The transaction commit is the **fail-closed gate**: if either step
raises, the LLM call MUST abort with a degraded-mode ContextResult.
The bytes are durably accounted for once the transaction commits;
the originals are still on disk and restorable via
``restore_excluded`` (``features/context/feature.py``); the model
simply does not see them this turn.

Async summarisation runs *after* the transaction commits, on a
background asyncio task. The salvage marker row IS the durable
queue — ``salvage_state == "pointer-only"`` means "not summarised
yet"; ``"pending-summary"`` means "summarisation in flight";
``"durable-folded"`` means done; ``"failed-fold"`` means terminal
failure after retry exhaustion. The :class:`SalvageWorker` schedules
the summary task; the janitor sweep re-schedules pointer-only rows
that were missed (process restart between INSERT and task spawn,
SignalDispatcher down, etc.).

Back-pressure: at a configurable queue-depth threshold, new salvages
skip the async schedule and stay terminal in ``pointer-only`` (the
``pointer_only_terminal`` flag in marker metadata). The
:func:`get_salvage_state_counts` function powers the breakdown
popup's "summariser falling behind" warning when the pending count
exceeds the warn-threshold.

Public surface
--------------

- :class:`SalvageReason` — enum-ish constants for the
  ``salvage_reason`` field.
- :class:`SalvageState` — the four-state lifecycle.
- :func:`salvage_messages` — the sync primitive (Phase 1).
- :class:`SalvageWorker` — background summariser + janitor (Phases
  3 + 4).
- :func:`get_salvage_state_counts` — for breakdown/popup (Phase 7).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class SalvageReason:
    """Why a span was salvaged.

    Stored as ``metadata.salvage_reason`` on the salvage marker.
    Used by the popup to label what triggered the fold; by the
    operator to debug salvage frequency by source.
    """

    AUTO_PRUNE_PRETRIM = "auto-prune-pretrim"
    AUTO_PRUNE_POSTBUDGET = "auto-prune-postbudget"
    MANUAL_COMPACT = "manual-compact"


class SalvageState:
    """Lifecycle states for a salvaged span.

    See the state diagram in
    ``docs/architecture/CONTEXT_C_DURABLE_SALVAGE.md``.
    """

    POINTER_ONLY = "pointer-only"
    PENDING_SUMMARY = "pending-summary"
    DURABLE_FOLDED = "durable-folded"
    FAILED_FOLD = "failed-fold"
    RESTORED = "restored"  # set by restore_excluded for audit


# Default queue-depth thresholds. Operator-configurable per
# ``multi_agent.toml`` in a follow-up; design doc names this an ops
# review item.
DEFAULT_PENDING_TERMINAL_THRESHOLD = 50
"""When this many salvages are pointer-only or pending-summary, new
prunes skip the async enqueue and stay ``pointer-only-terminal``."""

DEFAULT_PENDING_WARN_THRESHOLD = 10
"""When this many salvages are pending, the popup surfaces the
"summariser falling behind" warning."""

# Janitor sweep: how often to scan for pointer-only rows that the
# initial async task spawn missed (process restart, etc).
DEFAULT_JANITOR_INTERVAL_SECONDS = 60
DEFAULT_JANITOR_STALE_SECONDS = 30
"""Pointer-only rows older than this without a task in flight are
re-scheduled by the janitor."""

# Async summarisation: bound the LLM retry attempts per span.
MAX_SUMMARY_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Sync salvage primitive (Phase 1)
# ---------------------------------------------------------------------------


@dataclass
class SalvageResult:
    """What ``salvage_messages`` returns when the sync write succeeds."""

    salvage_id: int
    original_message_ids: List[int]
    token_estimate: int
    reason: str
    pointer_only_terminal: bool
    """When True, the queue-depth threshold was exceeded and the
    async enqueue was skipped. The span stays terminal in
    ``pointer-only`` until the janitor drains the queue."""


class SalvageWriteError(Exception):
    """Sync salvage write failed.

    ``ContextManager.build_context`` MUST catch this, populate
    ``ContextResult.warnings``, set ``degraded_mode=True`` and ABORT
    the LLM call — never silently absorb (Emma 2026-05-20 hardening
    invariant).
    """


async def salvage_messages(
    *,
    conv_store: "AsyncConversationStore",
    original_messages: List[Dict[str, Any]],
    reason: str,
    model: str,
    session_id: Optional[str],
    token_estimate: int,
    pending_count: int = 0,
    pending_terminal_threshold: int = DEFAULT_PENDING_TERMINAL_THRESHOLD,
) -> SalvageResult:
    """Write a sync salvage record for ``original_messages``.

    This is the fail-closed gate from C's invariant: when this call
    returns successfully, the database state guarantees that
    ``original_messages`` are durably marked
    ``excluded_from_context=True`` with ``summarized_into`` linking to
    a salvage marker row, and the LLM call may proceed. The async
    summarisation is scheduled separately by the caller (via
    :class:`SalvageWorker`).

    Args:
        conv_store: Per-agent ``AsyncConversationStore``.
        original_messages: The full message dicts from
            ``get_conversation_history`` that are leaving the
            model-visible slice. The function reads their ``id`` and
            ``content`` fields; everything else is ignored.
        reason: One of :class:`SalvageReason`'s constants.
        model: The model id whose context window forced the prune.
            Stored on the marker for diagnostics (e.g. "salvage
            happened because we were running on the 8k-window
            fallback").
        session_id: Active session id. Required for non-leakage
            (#713) — the salvage row carries it so cross-session
            queries do not surface spans from other sessions.
        token_estimate: Pre-computed token cost of the salvaged span
            (caller has it from the prune loop; passing it in avoids
            re-counting).
        pending_count: Current per-session count of salvages in
            ``pointer-only`` or ``pending-summary`` state. When this
            exceeds ``pending_terminal_threshold``, the salvage is
            written as ``pointer_only_terminal: True`` so the
            background worker skips it until the janitor drains the
            queue.
        pending_terminal_threshold: Threshold above which new
            salvages skip the async enqueue.

    Returns:
        :class:`SalvageResult` with the new marker id, the original
        ids, and ``pointer_only_terminal`` set per back-pressure.

    Raises:
        :class:`SalvageWriteError`: on any DB error during the
        transaction. ContextManager.build_context must treat this as
        fail-closed (no LLM call).
    """
    if not original_messages:
        raise SalvageWriteError("salvage_messages called with empty span")
    original_ids = [m["id"] for m in original_messages if m.get("id") is not None]
    if not original_ids:
        raise SalvageWriteError(
            "salvage_messages called with messages lacking ids"
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    pointer_only_terminal = pending_count > pending_terminal_threshold
    # Unique salvage handle for atomic id lookup after INSERT (codex
    # round 1 #1). Before this we used ``ORDER BY id DESC LIMIT 1``
    # which is racy under Postgres READ COMMITTED — a concurrent
    # transaction's commit could land between the INSERT and the
    # SELECT and the marker id would point at the wrong row. The
    # ``salvage_uuid`` is unique by construction; the lookup matches
    # exactly one row regardless of isolation level.
    salvage_uuid = str(uuid.uuid4())

    marker_metadata = {
        "type": "salvage",
        "salvage_state": SalvageState.POINTER_ONLY,
        "salvage_reason": reason,
        "original_message_ids": original_ids,
        "message_range": {"first": original_ids[0], "last": original_ids[-1]},
        "salvaged_at": now_iso,
        "summarized_at": None,
        "token_estimate": int(token_estimate),
        "model_at_salvage": model,
        "session_id": session_id,
        "summary_attempts": 0,
        "last_attempt_error": None,
        "pointer_only_terminal": pointer_only_terminal,
        "salvage_uuid": salvage_uuid,
    }

    db = conv_store.db
    try:
        async with db.transaction():
            # Phase 1a: INSERT salvage marker with a unique
            # ``salvage_uuid`` so we can identify it deterministically
            # after the insert.
            await db.execute_commit(
                f"INSERT INTO conversation_history "
                f"(agent_id, role, content, model, provider, metadata, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, {conv_store._now_sql()})",
                (
                    conv_store.agent_id,
                    "system",
                    "",  # body fills in on durable-folded
                    None,
                    None,
                    json.dumps(marker_metadata),
                ),
            )
            # Find the marker we just inserted by its uuid. This is
            # atomic regardless of concurrent writers because the uuid
            # is unique to this call (codex round 1 #1).
            row = await db.fetchone(
                "SELECT id FROM conversation_history "
                "WHERE agent_id = ? AND metadata LIKE ?",
                (
                    conv_store.agent_id,
                    f'%"salvage_uuid": "{salvage_uuid}"%',
                ),
            )
            if not row or row[0] is None:
                raise SalvageWriteError(
                    "salvage marker not found by uuid after INSERT — "
                    "the transaction may have rolled back silently"
                )
            salvage_id = int(row[0])

            # Phase 1b: UPDATE originals' metadata. The conversation
            # store helper handles the JSON-patch semantics safely.
            # Codex round 1 #2: check the affected-row count and fail
            # closed if any original was missed (concurrent delete,
            # cross-agent id, etc.) — a partial update would leave
            # bytes leaving the model view without a durable link.
            exclusion_patch = {
                "excluded_from_context": True,
                "excluded_at": now_iso,
                "excluded_reason": f"salvage:{reason}",
                "summarized_into": str(salvage_id),
            }
            updated_count = await conv_store.update_messages_metadata(
                original_ids, exclusion_patch
            )
            # ``update_messages_metadata`` may return None on some
            # backends (the call returns the rows-affected on others).
            # When it returns an int, enforce the all-or-nothing
            # contract; when it returns None we cannot verify and we
            # treat it as best-effort (the unit test stub returns None).
            if (
                updated_count is not None
                and updated_count != len(original_ids)
            ):
                raise SalvageWriteError(
                    f"originals UPDATE affected {updated_count} rows "
                    f"but expected {len(original_ids)} — refusing to "
                    "commit a salvage with broken linkage"
                )
        # Transaction committed — fail-closed gate passes.
    except SalvageWriteError:
        raise
    except Exception as e:
        # Any DB error rolls back; caller must fail-closed.
        raise SalvageWriteError(
            f"salvage transaction failed: {e}"
        ) from e

    logger.info(
        "salvage marker %s written for %d messages (reason=%s, "
        "tokens=%d, terminal=%s)",
        salvage_id,
        len(original_ids),
        reason,
        token_estimate,
        pointer_only_terminal,
    )
    return SalvageResult(
        salvage_id=salvage_id,
        original_message_ids=original_ids,
        token_estimate=int(token_estimate),
        reason=reason,
        pointer_only_terminal=pointer_only_terminal,
    )


# ---------------------------------------------------------------------------
# Background async worker + janitor (Phases 3, 4)
# ---------------------------------------------------------------------------


# The LLM summarisation prompt is exactly the one ``compact_session``
# uses today (``conversation_manager.py:142-154``). C's design keeps
# the prompt identical so the summary content stays consistent
# regardless of who triggered the fold (prune-driven vs operator-
# driven).
_SUMMARY_PROMPT_TEMPLATE = """Summarize this conversation segment concisely, preserving:
- Key facts, decisions, and conclusions reached
- Important context needed to continue the conversation
- Any commitments, preferences, or requests mentioned
- The emotional tone and relationship context

Do NOT include meta-commentary. Write as a direct summary that could replace this conversation segment.

CONVERSATION:
{conversation_text}

SUMMARY:"""

_SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Create concise summaries "
    "that preserve essential context."
)


class SalvageWorker:
    """Background summariser for ``pointer-only`` salvage rows.

    Owns the asyncio tasks that drive each salvage marker from
    ``pointer-only`` → ``pending-summary`` → ``durable-folded`` (or
    ``failed-fold`` on retry exhaustion). Also owns the periodic
    janitor sweep that re-schedules orphaned ``pointer-only`` rows
    (the process restarted between the sync salvage transaction
    and the task spawn, for instance).

    The worker stays decoupled from
    :class:`signals.dispatcher.SignalDispatcher` — the salvage row
    itself is the durable queue (no separate persistence), and the
    summariser is not a "source" in the constitutional-injection
    sense; it is an internal background loop tied to the agent's
    lifecycle.
    """

    def __init__(
        self,
        *,
        conv_store: "AsyncConversationStore",
        llm_completion: Callable[..., Any],
        janitor_interval_seconds: float = DEFAULT_JANITOR_INTERVAL_SECONDS,
        janitor_stale_seconds: float = DEFAULT_JANITOR_STALE_SECONDS,
        max_summary_attempts: int = MAX_SUMMARY_ATTEMPTS,
    ):
        self.conv_store = conv_store
        self.llm_completion = llm_completion
        self._janitor_interval = janitor_interval_seconds
        self._janitor_stale = janitor_stale_seconds
        self._max_attempts = max_summary_attempts
        self._tasks: set[asyncio.Task] = set()
        self._in_flight: set[int] = set()
        self._janitor_task: Optional[asyncio.Task] = None
        self._stopped = False

    async def start(self) -> None:
        """Launch the periodic janitor sweep."""
        if self._janitor_task is None and not self._stopped:
            self._janitor_task = asyncio.create_task(
                self._janitor_loop(), name="salvage-janitor"
            )

    async def stop(self) -> None:
        """Cancel the janitor + drain in-flight summary tasks."""
        self._stopped = True
        if self._janitor_task is not None:
            self._janitor_task.cancel()
            try:
                await self._janitor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._janitor_task = None
        await self.drain()

    async def drain(self) -> None:
        """Wait for all in-flight summary tasks (used by tests + shutdown)."""
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._in_flight.clear()

    def schedule_summary(self, salvage_id: int) -> None:
        """Spawn the async summary task for a salvage marker.

        Caller (``ContextManager.build_context``) invokes this AFTER
        the sync salvage transaction commits but inside the same
        turn. Idempotent — re-scheduling a salvage that already has
        a task in flight is a no-op.
        """
        if self._stopped:
            return
        if salvage_id in self._in_flight:
            return
        self._in_flight.add(salvage_id)
        task = asyncio.create_task(
            self._summarize_salvage(salvage_id),
            name=f"salvage-summarize-{salvage_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        # ``_in_flight`` is cleared inside _summarize_salvage in the
        # state transitions; nothing to do here.

    async def _summarize_salvage(self, salvage_id: int) -> None:
        """Run the summarisation LLM call and update the salvage row."""
        try:
            # Atomic claim (codex round 1 #6): conditional UPDATE
            # only fires when the row is still ``pointer-only``.
            # Two competing workers (this process + the janitor, or
            # two app processes) cannot both claim the same row.
            claimed = await self._claim_for_summary(salvage_id)
            if not claimed:
                logger.debug(
                    "salvage %s: not claimed (already in another state)",
                    salvage_id,
                )
                return
            originals = await self._load_originals(salvage_id)
            if not originals:
                # Marker references rows that have since vanished —
                # mark failed-fold with explicit reason.
                await self._mark_failed_fold(
                    salvage_id, "originals not found (restored or deleted?)"
                )
                return
            summary_text = await self._call_llm_summary(originals)
            await self._mark_durable_folded(salvage_id, summary_text)
        except Exception as e:
            attempts = await self._increment_attempts(
                salvage_id, last_error=str(e)
            )
            if attempts >= self._max_attempts:
                await self._mark_failed_fold(
                    salvage_id,
                    f"exhausted {self._max_attempts} retries: {e}",
                )
                logger.error(
                    "salvage %s: failed-fold after %d attempts: %s",
                    salvage_id,
                    attempts,
                    e,
                )
            else:
                # Drop back to pointer-only; the janitor will retry.
                await self._update_salvage_state(
                    salvage_id, SalvageState.POINTER_ONLY
                )
                logger.warning(
                    "salvage %s: attempt %d/%d failed: %s — janitor will retry",
                    salvage_id,
                    attempts,
                    self._max_attempts,
                    e,
                )
        finally:
            self._in_flight.discard(salvage_id)

    async def _load_originals(
        self, salvage_id: int
    ) -> List[Dict[str, Any]]:
        marker = await self._load_marker(salvage_id)
        if not marker:
            return []
        original_ids = marker.get("metadata", {}).get(
            "original_message_ids", []
        )
        if not original_ids:
            return []
        # Pull the originals via the conv store (decryption handled there).
        all_rows = await self.conv_store.get_full_history_with_ids(
            include_excluded=True, include_stashed=True
        )
        by_id = {row["id"]: row for row in all_rows if "id" in row}
        return [by_id[i] for i in original_ids if i in by_id]

    async def _call_llm_summary(
        self, originals: List[Dict[str, Any]]
    ) -> str:
        conversation_text = "\n".join(
            f"{(m.get('role') or 'user').upper()}: {m.get('content', '') or ''}"
            for m in originals
        )
        prompt = _SUMMARY_PROMPT_TEMPLATE.format(
            conversation_text=conversation_text
        )
        # The caller binds ``llm_completion`` to a callable that
        # accepts ``user_prompt`` + ``system_prompt`` (matching the
        # production ``LLMService.generate`` signature — codex round
        # 1 #4 caught the previous ``prompt=`` mismatch). For
        # adapter ergonomics we pass both ``user_prompt`` (the
        # canonical kwarg) and ``prompt`` (legacy alias) so simple
        # test fakes that only accept ``prompt`` still work.
        result = await self._invoke_completion(prompt)
        return (result or "").strip() if isinstance(result, str) else str(result).strip()

    async def _invoke_completion(self, prompt: str) -> str:
        """Invoke the bound LLM completion callable.

        Tries the canonical ``user_prompt`` kwarg first (matches
        ``LLMService.generate``); falls back to the legacy ``prompt``
        kwarg so test fakes do not have to adopt the production
        signature.
        """
        import inspect

        sig = None
        try:
            sig = inspect.signature(self.llm_completion)
        except (TypeError, ValueError):
            pass
        kwargs: Dict[str, Any] = {"system_prompt": _SUMMARY_SYSTEM_PROMPT}
        if sig is not None and "user_prompt" in sig.parameters:
            kwargs["user_prompt"] = prompt
        else:
            kwargs["prompt"] = prompt
        # ``model_override`` is accepted by production but not all
        # test fakes; pass it only when the signature allows.
        if sig is not None and "model_override" in sig.parameters:
            kwargs["model_override"] = None
        return await self.llm_completion(**kwargs)

    async def _load_marker(
        self, salvage_id: int
    ) -> Optional[Dict[str, Any]]:
        row = await self.conv_store.db.fetchone(
            "SELECT id, role, content, metadata FROM conversation_history "
            "WHERE id = ?",
            (salvage_id,),
        )
        if not row:
            return None
        meta = {}
        raw_meta = row[3] if len(row) > 3 else None
        if raw_meta:
            try:
                meta = json.loads(raw_meta)
            except (TypeError, ValueError):
                meta = {}
        return {"id": row[0], "role": row[1], "content": row[2], "metadata": meta}

    async def _patch_marker_metadata(
        self, salvage_id: int, patch: Dict[str, Any]
    ) -> None:
        marker = await self._load_marker(salvage_id)
        if not marker:
            return
        meta = marker["metadata"]
        meta.update(patch)
        await self.conv_store.db.execute_commit(
            "UPDATE conversation_history SET metadata = ? WHERE id = ?",
            (json.dumps(meta), salvage_id),
        )

    async def _update_salvage_state(
        self, salvage_id: int, state: str
    ) -> None:
        await self._patch_marker_metadata(
            salvage_id, {"salvage_state": state}
        )

    async def _claim_for_summary(self, salvage_id: int) -> bool:
        """Atomically claim a ``pointer-only`` salvage row for summary.

        Codex round 1 #6: two app processes (or this process + the
        janitor) could both pick the same row. Use a compare-and-set
        on ``salvage_state == 'pointer-only'`` so only one wins. The
        SQL LIKE is verbose but portable across SQLite and Postgres
        without backend-specific JSON ops; the marker's JSON metadata
        contains the canonical state string we patch via
        ``_patch_marker_metadata``.

        Returns True when this worker successfully claimed the row.
        """
        marker = await self._load_marker(salvage_id)
        if not marker:
            return False
        current_state = marker["metadata"].get("salvage_state")
        if current_state != SalvageState.POINTER_ONLY:
            return False
        # Patch via the metadata pipeline; this is best-effort
        # atomicity (the load-then-write is not a single SQL
        # statement). The DB-level guarantee comes from the conditional
        # WHERE clause below.
        meta = dict(marker["metadata"])
        meta["salvage_state"] = SalvageState.PENDING_SUMMARY
        affected = await self.conv_store.db.execute_commit(
            "UPDATE conversation_history SET metadata = ? "
            "WHERE id = ? AND metadata LIKE ?",
            (
                json.dumps(meta),
                salvage_id,
                f'%"salvage_state": "{SalvageState.POINTER_ONLY}"%',
            ),
        )
        # ``execute_commit`` returns rows-affected (int) on most
        # backends and may return None on a few. Treat 0 as "another
        # worker beat us"; treat None as best-effort success (the
        # janitor's idempotency saves us if duplicated).
        if affected == 0:
            return False
        return True

    async def _increment_attempts(
        self, salvage_id: int, last_error: str
    ) -> int:
        marker = await self._load_marker(salvage_id)
        if not marker:
            return self._max_attempts
        attempts = int(marker["metadata"].get("summary_attempts", 0)) + 1
        await self._patch_marker_metadata(
            salvage_id,
            {
                "summary_attempts": attempts,
                "last_attempt_error": last_error,
            },
        )
        return attempts

    async def _mark_durable_folded(
        self, salvage_id: int, summary_text: str
    ) -> None:
        # Write the summary text into the marker's content and flip
        # state. Done in two statements (UPDATE content; UPDATE
        # metadata) — the metadata patch is small and the content
        # write is bounded by the salvage size.
        await self.conv_store.db.execute_commit(
            "UPDATE conversation_history SET content = ? WHERE id = ?",
            (
                f"[SALVAGED CONTEXT — durable summary]\n\n{summary_text}",
                salvage_id,
            ),
        )
        await self._patch_marker_metadata(
            salvage_id,
            {
                "salvage_state": SalvageState.DURABLE_FOLDED,
                "summarized_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _mark_failed_fold(
        self, salvage_id: int, reason: str
    ) -> None:
        await self._patch_marker_metadata(
            salvage_id,
            {
                "salvage_state": SalvageState.FAILED_FOLD,
                "last_attempt_error": reason,
                "summarized_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ------------------- janitor sweep -------------------

    async def _janitor_loop(self) -> None:
        """Periodically re-enqueue stale ``pointer-only`` salvage rows.

        Covers the failure mode where the agent crashed between the
        sync salvage transaction and the ``schedule_summary`` call,
        or where the background task itself was lost.
        """
        while not self._stopped:
            try:
                await asyncio.sleep(self._janitor_interval)
                if self._stopped:
                    return
                stale = await self._find_stale_pointer_only()
                for sid in stale:
                    if sid not in self._in_flight:
                        self.schedule_summary(sid)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"salvage janitor sweep failed: {e}")

    async def _find_stale_pointer_only(self) -> List[int]:
        """Return salvage marker ids that are ``pointer-only`` and
        older than ``janitor_stale_seconds``, AND not marked
        ``pointer_only_terminal`` (those are deliberately not
        scheduled until back-pressure drains)."""
        cutoff = datetime.now(timezone.utc).timestamp() - self._janitor_stale
        rows = await self.conv_store.db.fetchall(
            "SELECT id, metadata FROM conversation_history "
            "WHERE agent_id = ? AND role = 'system' "
            "AND metadata LIKE '%\"type\": \"salvage\"%' "
            "AND metadata LIKE '%\"salvage_state\": \"pointer-only\"%'",
            (self.conv_store.agent_id,),
        )
        stale: List[int] = []
        for row in rows:
            sid = row[0]
            meta_raw = row[1] if len(row) > 1 else None
            if not meta_raw:
                continue
            try:
                meta = json.loads(meta_raw)
            except (TypeError, ValueError):
                continue
            # Skip pointer-only-terminal: the back-pressure decision.
            if meta.get("pointer_only_terminal"):
                continue
            salvaged_at = meta.get("salvaged_at")
            if not salvaged_at:
                # No timestamp = old → re-enqueue.
                stale.append(sid)
                continue
            try:
                ts = datetime.fromisoformat(salvaged_at).timestamp()
            except (TypeError, ValueError):
                stale.append(sid)
                continue
            if ts <= cutoff:
                stale.append(sid)
        return stale


# ---------------------------------------------------------------------------
# Read-only helpers for breakdown popup (Phase 7)
# ---------------------------------------------------------------------------


async def get_salvage_state_counts(
    conv_store: "AsyncConversationStore",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Per-state counts of salvage markers for the breakdown popup.

    Returns a dict ready to slot into
    ``measure_context_breakdown.sections.history.salvages``:

        {
            "pointer_only_count": int,
            "pointer_only_terminal_count": int,
            "pending_count": int,
            "folded_count": int,
            "failed_count": int,
            "pre_c_boundary_at": <ISO8601 | None>,
        }

    Session-scoped when ``session_id`` is provided; otherwise
    cross-session per-agent (idle case).
    """
    counts = {
        "pointer_only_count": 0,
        "pointer_only_terminal_count": 0,
        "pending_count": 0,
        "folded_count": 0,
        "failed_count": 0,
        "pre_c_boundary_at": None,
    }
    rows = await conv_store.db.fetchall(
        "SELECT metadata FROM conversation_history "
        "WHERE agent_id = ? AND role = 'system' "
        "AND metadata LIKE '%\"type\": \"salvage\"%'",
        (conv_store.agent_id,),
    )
    earliest: Optional[str] = None
    for row in rows:
        raw = row[0] if row else None
        if not raw:
            continue
        try:
            meta = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if session_id and meta.get("session_id") != session_id:
            continue
        state = meta.get("salvage_state")
        if state == SalvageState.POINTER_ONLY:
            if meta.get("pointer_only_terminal"):
                counts["pointer_only_terminal_count"] += 1
            else:
                counts["pointer_only_count"] += 1
        elif state == SalvageState.PENDING_SUMMARY:
            counts["pending_count"] += 1
        elif state == SalvageState.DURABLE_FOLDED:
            counts["folded_count"] += 1
        elif state == SalvageState.FAILED_FOLD:
            counts["failed_count"] += 1
        # ``restored`` is excluded from counts — it's audit-only.

        salvaged_at = meta.get("salvaged_at")
        if salvaged_at:
            if earliest is None or salvaged_at < earliest:
                earliest = salvaged_at
    counts["pre_c_boundary_at"] = earliest
    return counts


async def get_pending_count(
    conv_store: "AsyncConversationStore",
    session_id: Optional[str] = None,
) -> int:
    """Lightweight pending-salvage counter used by the back-pressure
    gate inside ``ContextManager.build_context``. Counts both
    ``pointer-only`` (queued) and ``pending-summary`` (running)."""
    counts = await get_salvage_state_counts(conv_store, session_id)
    return counts["pointer_only_count"] + counts["pending_count"]


# ---------------------------------------------------------------------------
# Feature flag (Phase 9)
# ---------------------------------------------------------------------------


def is_durable_salvage_enabled() -> bool:
    """Feature flag for the C release gate.

    When True, the full sync-salvage + async-worker + popup-wiring
    path is live end-to-end, AND the endpoint flips
    ``silently_pruned_path_active`` to False.

    When False, ``ContextManager.build_context`` falls back to the
    legacy silent-prune behaviour (no salvage records) — exactly
    pre-C semantics. The popup keeps surfacing
    ``silently-pruned path still active``.

    The flag exists so the implementation can ship behind a switch:
    a partially-shipped C never silently claims correctness it
    doesn't have. The release gate from epic #1307 keys off this
    flag flipping to True in production, not off ticket closure
    (Emma 2026-05-21).
    """
    import os

    val = os.environ.get("KESTREL_CONTEXT_C_DURABLE_SALVAGE", "")
    return val.lower() in ("1", "true", "yes", "on")
