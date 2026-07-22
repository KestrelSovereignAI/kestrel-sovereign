"""
Privacy-Enforcing Storage Wrapper for Kestrel.

This module provides a storage wrapper that enforces privacy mode restrictions
at the storage layer itself, preventing data leakage by design.

The wrapper intercepts all storage operations and:
1. EPHEMERAL mode: Blocks all write operations (raises PrivacyViolationError)
2. ISOLATED mode: Redirects to in-memory session storage
3. ANONYMOUS mode: Applies PII scrubbing before storage
4. NORMAL/PUBLIC mode: Passes through to underlying storage

This is a defense-in-depth measure - even if application code forgets to
check privacy mode, the storage layer will enforce it.
"""

import asyncio
import contextvars
import json
import logging
import re
import sys
import warnings
from contextlib import asynccontextmanager, contextmanager

from kestrel_sovereign.storage.session_grouping import summarize_sessions
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING, Union
from enum import Enum
from dataclasses import dataclass

from kestrel_sovereign.privacy import (
    PrivacyMode,
    PrivacyConfig,
    get_privacy_preset,
    privacy_mode_to_config,
    privacy_config_to_mode,
)
from kestrel_sovereign.storage.conversation_ids import coerce_persistent_message_id
from kestrel_sovereign.storage.async_conversation_store import (
    _escape_like_session_value,
    search_session_summaries,
)
from kestrel_sovereign.storage.async_graph_store import NodeSwapResult
from kestrel_sovereign.storage.agent_resource_store import (
    SOUL_MARKDOWN_RESOURCE_TYPE,
)

# Lazy import to avoid circular dependency with features.privacy
# Note: This global cache is shared across all instances and async contexts.
# The anonymize_text function is stateless and thread-safe, so this is acceptable.
_anonymize_text = None

def get_anonymize_text():
    """Lazy-load the anonymize_text function to avoid circular imports."""
    global _anonymize_text
    if _anonymize_text is None:
        from kestrel_sovereign.features.privacy.pii_detector import anonymize_text
        _anonymize_text = anonymize_text
    return _anonymize_text

logger = logging.getLogger(__name__)


# Stable id bucketing ISOLATED in-memory messages that were stored without an
# explicit session_id, so list_conversation_sessions never returns a synthetic
# index that delete_conversation_session can't resolve (#2019).
_ISOLATED_UNLABELED_SESSION_ID = "session-local"


def _conv_session_id(conv: Dict[str, Any]) -> Optional[str]:
    """The session id of an in-memory ISOLATED message.

    ISOLATED entries carry ``session_id`` as a top-level field (see
    ``add_conversation``); fall back to ``metadata.session_id`` for safety.
    """
    sid = conv.get("session_id")
    if sid is None:
        sid = (conv.get("metadata") or {}).get("session_id")
    return sid


def _in_session(conv: Dict[str, Any], session_id: Optional[str]) -> bool:
    """True if an in-memory ISOLATED message belongs to ``session_id``.

    ``session_id=None`` means unscoped (every message qualifies). When a
    session is named, a message qualifies only if its own ``session_id``
    matches — so scoped deletes can't reach across isolated conversations that
    happen to share text, and deleting one listed isolated session never wipes
    the others (#2019).
    """
    if session_id is None:
        return True
    sid = _conv_session_id(conv)
    if sid is None:
        # Unlabeled isolated entries are bucketed under the sentinel id that
        # list_conversation_sessions surfaces, so they stay deletable.
        return str(session_id) == _ISOLATED_UNLABELED_SESSION_ID
    return str(sid) == str(session_id)


class PrivacyViolationError(Exception):
    """Raised when a storage operation violates the current privacy mode."""
    pass


class OperationType(Enum):
    """Types of storage operations for permission checking."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"


# ── Cross-task reentry token for the privacy-transition lock (#2672 review P1) ──
#
# Same-task reentry (below) fixes the ANTHROPIC path, where a durable-identity
# write dispatched inline runs on the SAME asyncio task that holds the turn's
# transition lock. It does NOT fix the CODEX app-server path: the long-lived
# app-server dispatches each ``item/tool/call`` handler on its own reader-spawned
# task (``agent/orchestrator_engine.py`` — the reader loop spawns a per-call
# task), NOT the turn task. That handler task calls ``execute_named_tool`` →
# ``rename_agent_core`` (or description / discovery history / user name / SOUL),
# which re-acquires the SAME lock. By task identity it is a DIFFERENT task, so it
# would block waiting for the turn task to release — while the turn task is itself
# blocked awaiting the app-server's tool response. That is a deadlock.
#
# The fix, per the review: give inline tool callbacks a private, captured
# per-turn ownership token that permits re-entry into the lock for THAT specific
# active turn. The executor captures the owning turn's token on the turn task
# (where the lock is held — see ``ReentrantTransitionLock.current_reentry_token``)
# and re-presents it via ``bind_transition_lock_reentry`` around the tool call.
# The lock then admits that one turn's nested write cross-task, while a genuinely
# concurrent ``set_privacy_mode`` from an UNRELATED task (which captured no token)
# still serializes — cross-turn exclusion, the only exclusion the #2672
# check-then-write race needs, is preserved.
#
# The token authorizes re-entry but does NOT wave the reentrant write straight
# through: because the app-server can dispatch MULTIPLE inline tools of one turn
# concurrently — each on its own reader task, all carrying the SAME token — the
# lock funnels cross-task reentrants through a per-span reentry mutex
# (``_reentry_lock``) so two durable identity writes cannot interleave. See the
# ``ReentrantTransitionLock`` docstring and ``__aenter__`` for the serialization.
_transition_lock_reentry_token: "contextvars.ContextVar[Optional[object]]" = (
    contextvars.ContextVar("kestrel_transition_lock_reentry_token", default=None)
)


@contextmanager
def bind_transition_lock_reentry(token: Optional[object]):
    """Re-present a captured transition-lock reentry token on the current task.

    Companion to :meth:`ReentrantTransitionLock.current_reentry_token`. An
    inline-tool executor built inside a streamed turn captures that turn's token
    (on the turn task, which holds the lock) and wraps the actual tool call in
    ``with bind_transition_lock_reentry(token):`` so a nested durable-identity
    write (rename / description / discovery history / user name / SOUL) dispatched
    on the codex app-server's SEPARATE reader task can re-enter the lock its
    OWNING turn holds instead of deadlocking (#2672 review P1). The token is the
    identity of ONE held span on ONE lock instance, so it authorizes re-entry into
    that span only — a different lock (another agent) or a later span sees no
    match. Binding ``None`` is a no-op (the anthropic path runs the tool on the
    turn task, where reentry is by task identity), preserving prior behaviour.
    """
    if token is None:
        yield
        return
    reset = _transition_lock_reentry_token.set(token)
    try:
        yield
    finally:
        _transition_lock_reentry_token.reset(reset)


def current_bound_reentry_token() -> Optional[object]:
    """Return the transition-lock reentry token currently BOUND on this task, or ``None``.

    Companion to :func:`bind_transition_lock_reentry`. Where
    :meth:`ReentrantTransitionLock.current_reentry_token` reads the token off the
    LOCK (meaningful only on the task that OWNS the lock), this reads it off the
    CONTEXTVAR — i.e. whatever a surrounding :func:`bind_transition_lock_reentry`
    put there. This is what a NESTED inline executor needs (#2672 review P1
    follow-up): the feature-subagent inline executor is BUILT on the parent inline
    executor's reader task, INSIDE that executor's ``bind_transition_lock_reentry``
    scope, so the owning turn's token is visible here even though this task does not
    own the lock. The nested executor captures it and re-presents it around its OWN
    cross-task tool dispatch, so a durable-identity write invoked by a subagent
    re-enters the owning turn's span instead of deadlocking. Returns ``None`` when no
    reentry token is bound (the anthropic path, or a subagent not nested under a held
    turn lock) — so an unrelated background task, which captured no token, still
    serializes.
    """
    return _transition_lock_reentry_token.get()


class ReentrantTransitionLock:
    """Task-reentrant async lock for the privacy-transition boundary (#2672 P1).

    ``asyncio.Lock`` is NOT reentrant. A streamed turn holds the privacy-transition
    lock across the ENTIRE turn — including feature/tool execution
    (``agent/streaming.py`` acquires it before dispatching tools) — so a durable
    identity write invoked as a tool WITHIN that turn (rename / description /
    discovery history / user name / SOUL) that re-acquires the SAME lock would wait
    forever on a lock its own turn already holds, hanging the stream. This lock
    admits such a nested write via TWO scoped-to-the-held-span signals:

      * SAME-TASK reentry — the task that acquired the lock re-acquires it. This
        covers the anthropic path, where the inline write runs on the turn task
        itself. A nested acquire returns immediately and bumps a depth counter;
        only the OUTERMOST acquire releases to other tasks.

      * CROSS-TASK reentry via a captured token — the codex app-server dispatches
        each inline tool on its own reader-spawned task, NOT the turn task, so
        same-task reentry cannot help it and it would DEADLOCK (the tool waits on
        the lock the turn holds; the turn waits on the app-server's tool result).
        The inline executor captures this span's token on the turn task (via
        :meth:`current_reentry_token`) and re-presents it via
        :func:`bind_transition_lock_reentry`; a caller carrying THIS span's token
        re-enters cross-task — but through a SEPARATE per-span reentry mutex, not
        by bypassing the lock outright. That mutex serializes the app-server's
        concurrent reader tasks against each other: all inline tools of one turn
        share the same token, so two durable identity writes (e.g. two
        ``rename_agent`` calls) that would otherwise interleave their
        metadata/graph/memory/SOUL writes are forced one-at-a-time, the second
        blocked until the first's critical section completes (#2672 review P1).
        The mutex is distinct from the base lock the turn owner holds, so the
        cross-task write still never deadlocks; a nested write on the SAME reader
        task re-enters the mutex by task identity.

    Cross-turn exclusion is unchanged — a writer in task A and a concurrent
    ``set_privacy_mode`` in task B (which captured no token) still serialize —
    which is the only exclusion the #2672 check-then-write race needs. Both reentry
    signals are scoped to the ONE currently-held span (the token is a fresh object
    identity minted on the outermost acquire and cleared on the outermost release),
    so a stale token from a prior span never re-enters a later one. Exposes the
    ``asyncio.Lock``-compatible surface (``locked()`` + async context manager) the
    call sites use.

    CANCELLATION DRAIN (#2672 review P1). A cross-task reentry admitted via the
    token is an ACTIVE LEASE on the outer span, not merely a hold of the reentry
    mutex: the codex app-server runs each inline tool on a DETACHED reader task, so
    if the owning turn is cancelled/disconnected while such a reader is mid
    durable-write, the naive release would clear the token and free ``_lock``
    immediately — and a concurrent privacy transition could then acquire ``_lock``
    and flip to a volatile mode while the already-admitted NORMAL-mode write is
    still persisting. To prevent that, the outermost owner exit first INVALIDATES
    the token (rejecting any new reentry) and then WAITS for every active
    token-bearing writer to finish before releasing ``_lock`` — cancellation-safely,
    re-raising the turn's own cancellation only after the base lock is released. A
    reader that is itself cancelled unblocks the drain via the same idle signal, so
    the drain always terminates.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: Optional["asyncio.Task"] = None
        self._depth = 0
        # Fresh identity per HELD span, minted on the outermost acquire and
        # cleared on the outermost release. Handed to inline tool callbacks that
        # run on a foreign task so they can re-enter THIS span cross-task; a token
        # from a prior span never matches a later span's fresh identity.
        self._token: Optional[object] = None
        # Serializes CROSS-TASK token reentries within a span (#2672 review P1).
        # The codex app-server dispatches every inline tool RPC of one turn on
        # its OWN reader-spawned task, and all of them carry the SAME span token.
        # Admitting them purely on token match (the prior bug) let two durable
        # identity writes — e.g. two ``rename_agent`` calls in one turn — run
        # their metadata/graph/memory/SOUL writes concurrently and interleave,
        # leaving the durable identity sources inconsistent. Cross-task
        # reentrants therefore acquire THIS mutex — distinct from the base
        # ``_lock`` the turn owner holds, so there is no deadlock — which makes
        # the second reentrant write wait for the first to complete. The owner's
        # own (same-task) reentry never touches it; a single task cannot run two
        # critical sections at once, so a depth bump is already exclusive there.
        self._reentry_lock = asyncio.Lock()
        # The reader task currently inside the reentry mutex, plus its nesting
        # depth — so a nested durable write on that SAME reader task re-enters by
        # task identity instead of self-deadlocking on the non-reentrant mutex.
        self._reentry_owner: Optional["asyncio.Task"] = None
        self._reentry_depth = 0
        # Set when NO cross-task reentrant writer holds this span, cleared while one
        # is active. The outermost owner AWAITS this before releasing ``_lock`` so a
        # detached reader task's ALREADY-ADMITTED durable write completes INSIDE the
        # owner's lock span even when the owning turn is cancelled/disconnected
        # mid-flight (#2672 review P1 cancellation). Without the drain, the owner's
        # ``__aexit__`` would clear the token and release ``_lock`` the instant the
        # turn is cancelled — while the codex app-server's detached tool task is
        # still persisting — and a concurrent ``set_privacy_mode`` could then
        # acquire ``_lock`` and flip to a volatile mode under that in-flight
        # NORMAL-mode write. Starts set (no reentrant writer at construction).
        self._reentry_idle = asyncio.Event()
        self._reentry_idle.set()

    def locked(self) -> bool:
        return self._lock.locked()

    def current_reentry_token(self) -> Optional[object]:
        """The token that permits cross-task re-entry into the CURRENTLY-held span.

        Returns the active span's token ONLY when the current task owns the lock —
        i.e. when called on the turn task at the point the inline-tool executor
        closure is built (``_make_inline_tool_executor`` runs inside the streamed
        turn's ``async with transition_lock``). The executor captures this and
        re-presents it via :func:`bind_transition_lock_reentry` when its tool runs
        on the codex app-server's separate reader task, so a nested durable-identity
        write re-enters the lock instead of deadlocking (#2672 review P1). Returns
        ``None`` when the current task does not own the lock — there is no span to
        authorize re-entry into, so an unrelated caller gets no token and still
        serializes.
        """
        task = asyncio.current_task()
        if task is not None and self._owner is task and self._token is not None:
            return self._token
        return None

    def _token_matches_current_span(self) -> bool:
        """Whether the current task carries THIS span's cross-task reentry token."""
        token = _transition_lock_reentry_token.get()
        return token is not None and self._token is not None and token is self._token

    async def __aenter__(self) -> "ReentrantTransitionLock":
        task = asyncio.current_task()
        # (1) Same-task reentry by the span owner (the anthropic inline path runs
        #     the write on the turn task itself; also any nested acquire). A single
        #     task cannot run two critical sections at once, so a depth bump is
        #     both sufficient and exclusive — no reentry mutex needed.
        if task is not None and self._owner is task:
            self._depth += 1
            return self
        # (2) Cross-task reentry by a reader task that ALREADY holds the reentry
        #     mutex — a nested durable write on that SAME reader task. Reentrant by
        #     task identity so it can't self-deadlock on the non-reentrant mutex.
        if task is not None and self._reentry_owner is task:
            self._reentry_depth += 1
            return self
        # (3) First cross-task reentry via a captured span token (the codex
        #     app-server's per-tool reader task). Multiple inline tools in ONE turn
        #     all carry the SAME span token; serialize concurrent token-bearers on
        #     the per-span reentry mutex so the second durable identity write cannot
        #     enter until the first's critical section completes — while STILL
        #     bypassing the base lock the turn owner holds, so there is no deadlock
        #     (#2672 review P1). The owner never reaches here: branch (1) caught it.
        if self._token_matches_current_span():
            await self._reentry_lock.acquire()
            # Re-validate after awaiting: the owning span may have ended while we
            # waited (owner released the base lock, token now stale). If so this is
            # no longer a reentry — release the mutex and fall through to a normal
            # acquire so a genuinely-after-span write serializes as its own turn.
            if self._token_matches_current_span():
                self._reentry_owner = task
                self._reentry_depth = 1
                # A cross-task reentrant writer is now ACTIVE in this span: mark the
                # span non-idle so the owner's outermost exit drains it before
                # releasing ``_lock`` (#2672 review P1 cancellation).
                self._reentry_idle.clear()
                return self
            self._reentry_lock.release()
        # (4) Normal acquire — an unrelated task (cross-turn), or the stale-token
        #     fallthrough above. This is the cross-turn exclusion the #2672
        #     check-then-write race depends on, unchanged.
        await self._lock.acquire()
        self._owner = asyncio.current_task()
        self._token = object()
        self._depth = 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        task = asyncio.current_task()
        # Unwind a cross-task reentry frame (reader task holding the reentry mutex)
        # before the base-lock frame — the two are always distinct tasks, so task
        # identity disambiguates which frame this ``__aexit__`` pairs with.
        if self._reentry_owner is not None and self._reentry_owner is task:
            self._reentry_depth -= 1
            if self._reentry_depth <= 0:
                self._reentry_depth = 0
                self._reentry_owner = None
                self._reentry_lock.release()
                # This cross-task reentrant write has finished its critical
                # section; signal any owner blocked in the drain that it may now
                # release the base lock (#2672 review P1 cancellation).
                self._reentry_idle.set()
            return False
        self._depth -= 1
        if self._depth <= 0:
            self._depth = 0
            self._owner = None
            # (a) Invalidate the span token FIRST so no NEW cross-task reentry is
            #     admitted past this point — a foreign task presenting the now-stale
            #     token re-validates in ``__aenter__`` branch (3) and falls through
            #     to a normal acquire, blocking on ``_lock`` until we release it.
            self._token = None
            # (b) DRAIN any ALREADY-admitted cross-task reentrant writer before
            #     releasing the base lock. On the normal path the reader's write has
            #     already completed (the app-server returned its tool result, which
            #     is what unblocked this turn), so the event is set and this is a
            #     no-op. On the cancellation path a detached reader may still be mid
            #     durable-write; hold ``_lock`` until it finishes (or is itself
            #     cancelled — its ``__aexit__`` sets the event either way) so a
            #     concurrent privacy transition cannot flip the mode under it
            #     (#2672 review P1 cancellation).
            pending_cancel = await self._drain_active_reentrant()
            self._lock.release()
            # Re-raise a cancellation that arrived DURING the drain only after the
            # lock is safely released, so the owning turn's cancellation still
            # propagates but never at the cost of skipping the drain.
            if pending_cancel is not None:
                raise pending_cancel
        return False

    async def _drain_active_reentrant(self) -> Optional[BaseException]:
        """Block until no cross-task reentrant writer holds this span.

        Called by the outermost owner exit AFTER the span token is invalidated and
        BEFORE ``_lock`` is released. Returns immediately on the normal path (the
        detached reader's write already finished, so ``_reentry_idle`` is set). On
        the cancellation path a detached codex app-server tool task may still be mid
        durable-write; wait for it to signal idle — its own ``__aexit__`` sets the
        event when its critical section completes OR when the reader task is itself
        cancelled, so the drain always terminates. The wait is SHIELDED so cancelling
        the owning turn cannot skip the drain (and cannot busy-spin under repeated
        cancellation — the shielded waiter keeps the reader schedulable); a caught
        cancellation is returned for the caller to re-raise AFTER releasing the base
        lock (#2672 review P1 cancellation).
        """
        if self._reentry_idle.is_set():
            return None
        cancelled: Optional[BaseException] = None
        waiter = asyncio.ensure_future(self._reentry_idle.wait())
        try:
            while not self._reentry_idle.is_set():
                try:
                    await asyncio.shield(waiter)
                except asyncio.CancelledError as exc:
                    cancelled = exc
        finally:
            if not waiter.done():
                waiter.cancel()
        return cancelled


@asynccontextmanager
async def optional_transition_lock(lock):
    """Hold ``lock`` for the block if provided, else run unguarded (#2672 review P1).

    ``lock`` is the agent's privacy-transition lock
    (``KestrelAgent._privacy_transition_lock``), the same mutex a privacy-mode
    transition holds while it flips the mode across every state holder. A direct
    durable user-content writer (rename, description, discovery history, user name,
    SOUL) that checks the privacy mode and THEN ``await``s its persistence must
    hold this lock across BOTH steps, or a transition can land the mode change in
    the ``await`` gap — the writer passes the NORMAL-mode check, the mode flips to
    EPHEMERAL, and the write persists anyway. Holding the lock makes the writer and
    the transition mutually exclusive, so the writer either completes fully before
    the flip or re-checks the mode after it and skips. ``None`` (raw storage /
    offline CLI paths with no running agent) runs unguarded, preserving prior
    behaviour.
    """
    if lock is None:
        yield
    else:
        async with lock:
            yield


def _resolve_transition_lock(holder):
    """Best-effort resolve an agent's privacy-transition lock, or ``None``.

    Accepts an object exposing ``_get_privacy_transition_lock()`` (the agent) or a
    plain lock. Returns ``None`` when neither is available (test/CLI shapes), so
    the writer runs unguarded rather than failing.
    """
    if holder is None:
        return None
    getter = getattr(holder, "_get_privacy_transition_lock", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001 - never let lock resolution block a write
            return None
    # Already a lock (has async context-manager protocol).
    if hasattr(holder, "__aenter__"):
        return holder
    return None


# ── Privacy-aware graph write policy (#2672) ─────────────────────────────────
#
# The knowledge graph is durable storage. In a volatile privacy mode —
# EPHEMERAL ("leave no trace"), ISOLATED ("session buffer only"), and
# DEIDENTIFIED (fail-closed until the Safe Harbor pipeline lands): every mode
# whose policy disallows persistent writes — a durable graph write is a real
# privacy leak. Facts, todos, decisions, concepts, and consolidated episodes
# are all derived from user conversation input. The pre-#2672 wrapper waved
# every graph write through as "structural, not PII", which let those
# user-derived nodes reach the backend directly, *outside* the #1760
# post-response gate, and let feature code bypass enforcement entirely by
# reaching through the raw ``.graph`` / ``.graph_store`` surfaces.
#
# A prior revision admitted a small allowlist by node-type/key/label plus a
# generic "is this value content-free-shaped?" check, and used a boolean
# ``control_plane=True`` marker for the value-bearing types. The third Terra
# review rejected both halves:
#
#   * P1 — a generic "short single-line string" check treats any secret under
#     ~512 chars as content-free, so a secret fits inside a ``hash`` / ``type`` /
#     ``source_path`` field. Shape-of-a-string is not semantics-of-a-field.
#   * P2 — ``control_plane=True`` is a publicly forgeable boolean on a public
#     method signature: ANY feature/tool passes it to self-authorize.
#
# The policy is now a TWO-tier default-deny where each tier closes one finding:
#
#   1. Content-free structural types — ``document`` (constitution byte-anchor)
#      and ``audit_anchor`` (tamper-evidence receipt), plus the ``governed_by``
#      governance edge. Their fields are hashes / counts / timestamps / an enum
#      literal BY CONSTRUCTION, so they are admitted on the ordinary (untrusted)
#      path — but every field is now validated by a PER-FIELD SEMANTIC validator
#      (a ``hash`` must be hex of a hash's length, a ``*_at`` must parse as an
#      ISO-8601 timestamp, an ``entries_count`` must be a non-negative int, a
#      ``type`` must be a known literal, ``document.hash`` must equal the node's
#      own content-hash id). A short secret is not a 64-hex digest, not a
#      timestamp, not an int, not the literal ``"Constitution"`` — so it fails
#      closed in EVERY field (finding P1). Labels are pinned to exact literals /
#      a fixed regex.
#
#   2. The ``agent`` control-plane identity node (DID, constitution / doctrine
#      anchors, genesis audit, bootstrap + lifecycle state). It carries the
#      control-plane CAPABILITY marker (``acquire_control_plane_capability()`` —
#      see below). Its content-free fields are per-field validated, and — the
#      load-bearing part — its FREE-TEXT fields (``name`` / ``description`` /
#      ``expected_duration`` identity text AND the governance-receipt blobs
#      ``genesis_audit`` / ``genesis_audit_history`` / ``emancipation_contract`` /
#      ``constitution_reanchor`` / ``doctrine_bundle_reanchor``) are admitted in a
#      volatile mode ONLY when CARRIED ALONG UNCHANGED from the stored node (or the
#      CAS ``expected`` snapshot). That CONTENT check, not the capability, is what
#      closes the free-text leak the review reproduced.
#
#      ``constitution_amendment_artifact`` (the signed reanchor receipt) is NOT a
#      tier-2 type: it is ALWAYS a fresh node (its id is its own content hash), so
#      its free-text ``source_path`` / ``signer`` / ``verification`` could never be
#      carried along — admitting it would be exactly the fresh-free-text channel
#      the review reproduced. It is default-denied. No boot path needs it (see
#      finding P3 below).
#
# IMPORTANT — the capability is same-process defense-in-depth, NOT an unforgeable
# authorization boundary. The third Terra review proved the earlier "unforgeable"
# claim false: the governance writers are mixin methods ON THE AGENT, and feature
# code holds the agent, so ANY in-process signal the writers use (caller-frame
# provenance, a held token, a module singleton) is reachable/forgeable by feature
# code running in the same interpreter — a feature can ``exec`` a helper into a
# real trusted module's ``__dict__`` and obtain the token. The privacy guarantee
# therefore CANNOT rest on caller identity; it rests on CONTENT (per-field
# validation + the carried-along free-text boundary). The genuinely hard boundary
# for untrusted feature code is the separate-process isolated feature runtime
# (``features/isolated_runtime.py``); the capability only keeps FIRST-PARTY
# governance code from accidentally tripping the gate and documents intent.
#
# The two user-derived surfaces the review told us NOT to blanket-trust —
# ``agent.description`` (free text) and ``feature_config.config`` (an arbitrary
# settings dict) — are gated at their SINGLE SOURCE OF TRUTH:
# ``bootstrap.service.persist_agent_description`` and ``Feature.persist_config``
# skip their durable writes entirely in a volatile mode. ``feature_config`` is
# therefore NOT in the allowlist at all, and ``agent.description`` reaching the
# node directly is refused by the carried-along boundary above unless unchanged
# (finding P3).
#
# Finding P3 — preserve volatile boot/governance through the SMALLEST EXPLICIT
# source-of-truth path, never a wrapper free-text channel. The one governance
# receipt that MUST persist fresh in a volatile mode is the first-cognition
# GENESIS AUDIT receipt (it gates cognition and runs regardless of privacy mode).
# It cannot be carried along (it is, by definition, fresh), so admitting it via
# this wrapper would reopen the free-text channel. Instead the genesis-audit
# writers persist it to the RAW store directly
# (``ConstitutionMixin._governance_graph_store`` → ``agent._raw_storage``), the
# same low-level store inception uses for the initial agent node — a first-party,
# content-addressed governance write that never traverses the feature-facing
# wrapper. The other governance ceremonies do not need a fresh wrapper free-text
# write: ``mark_stale_bootstrap`` / the doctrine BOOT anchor copy the node and
# mutate only content-free state (free-text carried unchanged → admitted); the
# runtime ``!reanchor-constitution`` command is blocked earlier by the volatile
# ``store_file`` gate; a runtime doctrine reanchor's fresh ``doctrine_bundle_reanchor``
# receipt fails closed in a volatile mode (a governance ceremony that mutates
# durable state belongs in a persistent mode). NORMAL / PUBLIC / ANONYMOUS are
# unaffected (the wrapper never governs a persistent-write mode).
#
# Residual (documented, not hidden): closing the free-text wrapper channel does
# NOT close the raw store — an in-process feature can still reach
# ``agent._raw_storage`` and write arbitrary nodes/free-text there, exactly like
# the genesis-audit writer does. That is the SAME residual as the raw-store note
# below and requires process isolation to close; this fix only guarantees the
# SANCTIONED wrapper surface is no longer a free-text channel.
#
# The SAME residual covers the ungoverned raw store, ``agent._raw_storage`` (and
# its ``.db``), which every in-process feature can reach off the agent object. It
# is a privileged, first-party control-plane handle — the store BELOW this wrapper
# that core paths use to persist identity/anchor state regardless of privacy mode
# — NOT part of the feature-facing storage API. This wrapper is the governed API
# surface for the SANCTIONED path (features persist via ``agent.storage``); it is
# not, and cannot be, an in-process sandbox, because first-party code sharing this
# interpreter can already reach anything (the capability-forgery note above is the
# same point). What this wrapper DOES guarantee is that the sanctioned surface has
# no accidental leak: ``.graph`` / ``.graph_store`` return a governing proxy that
# refuses the raw ``db`` handle and any un-allowlisted attribute (#2672 review P1),
# so ``storage.graph.db`` / ``storage.graph.<write>`` cannot slip a write past the
# boundary. The hard boundary against UNTRUSTED extension code is process
# isolation (``features/isolated_runtime.py``), which never receives ``_raw_storage``.
#
# Everything else — user-derived types, unknown types (now including
# ``constitution_amendment_artifact``), payload-bearing structural edges, and a
# fresh/changed free-text field — fails closed with a ``PrivacyViolationError``
# the tool caller can see. NORMAL / PUBLIC / ANONYMOUS (any mode that allows
# persistent writes) is unaffected.
#
# The ``agent`` key set is the COMPLETE governance vocabulary the production
# writers set (inception_service, agent/constitution, agent/doctrine_bundle,
# bootstrap/service, features/bootstrap, graduate_service, kestrel_agent,
# setup/constitution_reanchor, setup/overlay_anchor). A field missing here makes
# a born-volatile agent fail closed on a legitimate governance write, so a NEW
# governance field must be added consciously with a validator. Adding a type,
# key, or trusted module REQUIRES a test proving the written node carries no user
# content (see ``tests/unit/test_privacy_graph_boundary.py``).


# ── Control-plane write capability — same-process defense-in-depth (#2672) ────
#
# HONESTY NOTE (corrects an earlier false claim). A prior revision presented this
# capability as an "unforgeable" authorization boundary issued by caller-frame
# module-dict provenance. The third Terra review proved that false: any in-process
# code can ``import`` a trusted module and ``exec`` a helper into its real
# ``__dict__``; that helper's frame then has the exact ``f_globals`` identity the
# issuer checks, so it obtains the genuine token. There is NO same-interpreter
# provenance (module-dict identity, ``__name__``, a code-object check) and NO
# held/module token that feature code cannot reach, because the governance writers
# are mixin methods on the agent and features hold the agent.
#
# So this capability is NOT relied on for the privacy guarantee. The guarantee is
# CONTENT-based: per-field semantic validation for content-free fields, and the
# carried-along free-text boundary (``_StructuralNodeShape.free_text_carry_along``)
# for the ``agent`` node's user-derived free-text — a fresh/changed ``description`` is
# refused even to a caller holding a genuine (forged) capability. The capability
# is kept only as a coarse type gate and to document which writes are governance;
# ``acquire_control_plane_capability()`` still verifies module-dict provenance so
# a stray non-governance import can't casually pass it, but that is convenience,
# not security. The hard boundary for untrusted feature code is process isolation
# (``features/isolated_runtime.py``). This is stated plainly so no future reader
# mistakes the capability for an authorization boundary again.
#
#   * The sole instance is closure-private (no importable module attribute), and
#     the wrapper accepts it by object IDENTITY (``is``) — a boolean, a lookalike,
#     or a fresh ``_ControlPlaneCapability()`` are refused. This stops ACCIDENTAL
#     passes and trivially-forged markers, not a determined in-process attacker.


class _ControlPlaneCapability:
    """Marker that a durable graph write is a first-party governance write.

    Same-process defense-in-depth, NOT an authorization boundary (see the header
    HONESTY NOTE). The sole instance is closure-private and the wrapper checks it
    by ``is`` identity, so a boolean or a fresh ``_ControlPlaneCapability()`` won't
    stand in — this stops accidental/trivially-forged passes, not a determined
    in-process attacker. The privacy guarantee is carried by content validation
    and the carried-along identity boundary, not by this marker.
    """

    __slots__ = ()


# Fully-qualified module names whose namespace may obtain the control-plane
# marker. Each is a first-party governance / identity / bootstrap writer. The
# ``__dict__``-identity check below only distinguishes a casual import from a
# genuine governance write — it is NOT tamper-proof (any in-process code can
# ``exec`` into these modules' dicts; see the header note), which is exactly why
# the privacy guarantee does not rest on it. Adding a module REQUIRES tracing its
# write's input to a governance/identity source of truth and a test.
_TRUSTED_CONTROL_PLANE_MODULES = frozenset({
    "kestrel_sovereign.agent.constitution",
    "kestrel_sovereign.agent.doctrine_bundle",
    "kestrel_sovereign.bootstrap.service",
    "kestrel_sovereign.features.bootstrap.feature",
    "kestrel_sovereign.kestrel_agent",
})


def _build_control_plane_gate():
    """Build the ``(acquire, has)`` pair over a closure-private capability marker.

    The instance is closure-private (bound to no importable module attribute) so
    it can't be casually imported. This is defense-in-depth, not a hard boundary —
    a determined in-process caller can still obtain it (see the header note), which
    is why the privacy guarantee rests on content validation and the carried-along
    identity boundary instead.
    """
    _singleton = _ControlPlaneCapability()

    def acquire_control_plane_capability() -> "_ControlPlaneCapability":
        """Return the control-plane graph-write marker to a first-party writer.

        Verifies MODULE-DICT provenance: the immediate caller's ``f_globals`` must
        BE the ``__dict__`` of a module in :data:`_TRUSTED_CONTROL_PLANE_MODULES`.
        This distinguishes a genuine governance write from a casual import; it is
        NOT tamper-proof (in-process code can ``exec`` into those dicts), so it is
        defense-in-depth, not authorization — the privacy guarantee is enforced by
        content validation and the carried-along identity boundary regardless of
        who holds this marker (#2672 review P1). Call it inline at the write site.
        """
        try:
            caller_globals = sys._getframe(1).f_globals
        except Exception:  # pragma: no cover - frame access should always work
            caller_globals = None
        if caller_globals is not None:
            for module_name in _TRUSTED_CONTROL_PLANE_MODULES:
                module = sys.modules.get(module_name)
                if module is not None and module.__dict__ is caller_globals:
                    return _singleton
        raise PrivacyViolationError(
            "Control-plane graph-write marker refused: it is returned only to code "
            "running in the namespace of a first-party governance/identity/"
            "bootstrap module. This is a defense-in-depth convenience, not an "
            "authorization boundary — the privacy guarantee is enforced by content "
            "validation and the carried-along identity boundary regardless (#2672)."
        )

    def _has_control_plane_capability(capability: Any) -> bool:
        """Whether ``capability`` is the one true control-plane token (by identity)."""
        return capability is _singleton

    return acquire_control_plane_capability, _has_control_plane_capability


acquire_control_plane_capability, _has_control_plane_capability = _build_control_plane_gate()


# ── Per-field semantic validators (#2672 review P1) ──────────────────────────
#
# Each validator answers "is this value a valid, content-free instance of THIS
# field's semantic class?" — not the old "is it a short string?". A short secret
# is not a hash, not a timestamp, not a non-negative int, not a known enum
# literal, so it fails closed in every content-free field surface.

# A hash digest as written by ``store_file`` (SHA-256 → 64 lowercase hex) and by
# the identity/audit writers; the range tolerates other digest widths without
# admitting free text.
_HEX_HASH_RE = re.compile(r"^[0-9a-fA-F]{32,128}$")
# A short structural enum/token: a bootstrap state, a document ``type``, an
# artifact ``type`` — alphanumerics with ``_ . -`` only, no spaces, bounded.
_ENUM_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
# An identity token: DID / agent_id / test cycle id — allows the ``:`` and ``%``
# a DID method carries, single-line, bounded.
_ID_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:%#\-]{0,255}$")

_MAX_IDENTITY_TEXT = 100_000  # name / description (capability-gated identity)
_MAX_BOUNDED_TEXT = 8_192     # a bounded free-form receipt/path/reason string
_MAX_PATH = 4_096


def _is_hex_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX_HASH_RE.fullmatch(value))


def _is_hex_hash_or_none(value: Any) -> bool:
    return value is None or _is_hex_hash(value)


def _is_iso_timestamp(value: Any) -> bool:
    """True if ``value`` parses as an ISO-8601 / SQLite-datetime timestamp.

    Accepts the ISO form the identity writers emit (``datetime.isoformat()``,
    optionally ``Z``-terminated) and the ``YYYY-MM-DD HH:MM:SS`` shape SQLite
    writes. A free-text secret does not parse, so it fails closed.
    """
    if not isinstance(value, str) or not value or len(value) > 64 or "\n" in value:
        return False
    from datetime import datetime
    candidate = value.strip()
    normalized = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
    try:
        datetime.fromisoformat(normalized)
        return True
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            datetime.strptime(candidate, fmt)
            return True
        except ValueError:
            continue
    return False


def _is_iso_timestamp_or_none(value: Any) -> bool:
    return value is None or _is_iso_timestamp(value)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_enum_token(value: Any) -> bool:
    return isinstance(value, str) and bool(_ENUM_TOKEN_RE.fullmatch(value))


def _is_enum_token_or_none(value: Any) -> bool:
    return value is None or _is_enum_token(value)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_id_token(value: Any) -> bool:
    return isinstance(value, str) and bool(_ID_TOKEN_RE.fullmatch(value))


def _is_id_token_or_none(value: Any) -> bool:
    return value is None or _is_id_token(value)


def _is_numeric_str_or_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str) or not value or len(value) > 64:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_bounded_text_or_none(value: Any) -> bool:
    """A bounded, single-line free-form string (or None).

    Used for the agent node's ``expected_duration`` operator free-text. It CANNOT
    be proven content-free by shape, which is exactly why it is a
    ``free_text_carry_along`` field (admitted in a volatile mode only carried along
    unchanged) — this check only bounds length / rejects multi-line blobs as
    defense-in-depth.
    """
    if value is None:
        return True
    return isinstance(value, str) and len(value) <= _MAX_BOUNDED_TEXT and "\n" not in value


#: An opaque, provider-issued credential hash — e.g. the OpenRouter child-key
#: hash the host-owned ``payer_resolver`` writes to the agent node so
#: ``retirement_service`` can revoke the remote key. It is CONTENT-FREE billing
#: metadata (a credential handle), not user text: a bounded, single-line token
#: from the character class hash encodings use (hex plus ``: . _ -`` separators).
#: Whitespace, newlines, and oversized values are refused so it cannot become a
#: smuggling channel (defense-in-depth on the capability-gated ``agent`` node).
_CREDENTIAL_HASH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._\-]{7,255}$")


def _is_credential_hash_or_none(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and bool(_CREDENTIAL_HASH_RE.fullmatch(value))
    )


def _is_identity_text_or_none(value: Any) -> bool:
    """A bounded identity string (agent name / description / label).

    User-derived, so accepted only on the capability-gated ``agent`` node whose
    single introducer (``persist_agent_description``) is separately source-gated
    in volatile modes — this validator only bounds length and rejects non-strings
    / structured smuggling, it is not the trust boundary (#2672 review P3).
    """
    return value is None or (isinstance(value, str) and len(value) <= _MAX_IDENTITY_TEXT)


def _is_path_list_or_none(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, (list, tuple)):
        return False
    return all(
        isinstance(item, str) and len(item) <= _MAX_PATH and "\n" not in item
        for item in value
    )


def _is_governance_receipt(value: Any, _depth: int = 0) -> bool:
    """A bounded, shallow governance-receipt container (or scalar / None).

    The ``genesis_audit`` / ``emancipation_contract`` / ``*_reanchor`` fields are
    structured receipts (hashes, timestamps, statuses, provenance strings, signed
    contracts) written only by trusted governance code on the capability-gated
    ``agent`` node. This bounds their shape (no oversized blob, no deep nesting)
    as defense-in-depth; the capability is the trust boundary.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, str):
        return len(value) <= _MAX_BOUNDED_TEXT
    if _depth >= 6:
        return False
    if isinstance(value, (list, tuple)):
        return all(_is_governance_receipt(item, _depth + 1) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and len(key) <= 128
            and _is_governance_receipt(val, _depth + 1)
            for key, val in value.items()
        )
    return False


@dataclass(frozen=True)
class _StructuralNodeShape:
    """The canonical shape of one allowlisted structural node type.

    ``field_validators`` maps every permitted property key to its per-field
    semantic validator (the allowed key set is exactly its keys). ``requires_capability``
    marks the ``agent`` control-plane identity node, which carries the
    control-plane capability marker in a volatile mode (a same-process
    defense-in-depth signal, not an authorization boundary — see the header
    note); content-free types leave it ``False`` and are admitted on the ordinary
    path purely on strict per-field validation.

    ``free_text_carry_along`` names the node's user-derived / free-text fields
    (identity free-text AND the governance-receipt blobs). No per-field regex can
    prove these content-free, so in a volatile mode they are admitted ONLY when
    CARRIED ALONG UNCHANGED from the stored node (or the CAS ``expected``
    snapshot). A FRESH or CHANGED free-text value is refused even to a caller
    holding the (same-process, forgeable) capability — that CONTENT boundary, not
    the capability, is the load-bearing privacy gate, and it is what closes the
    reproduced free-text forgery leak (#2672 review P1). A genuinely-fresh
    governance receipt (e.g. the genesis-audit receipt at first cognition) is
    therefore NOT written through this wrapper in a volatile mode; it goes to the
    RAW store as the smallest explicit source-of-truth path (see the header note
    and ``ConstitutionMixin._governance_graph_store``).

    ``hash_equals_node_id``, when set, names a property that must equal the node's
    own id (the content hash) — closing the "arbitrary hex in a hash field"
    channel for the constitution ``document`` node.
    """
    field_validators: Dict[str, Any]
    label_literals: Optional[frozenset] = None
    label_regex: Optional[Any] = None  # compiled ``re.Pattern`` or None
    requires_capability: bool = False
    hash_equals_node_id: Optional[str] = None
    free_text_carry_along: frozenset = frozenset()

    @property
    def keys(self) -> frozenset:
        return frozenset(self.field_validators)

    @property
    def label_content_free(self) -> bool:
        """Whether this type's LABEL is validated content-free by shape.

        ``True`` when the label is pinned to a literal set or a strict regex
        (``document`` / ``audit_anchor``): such a label cannot carry user text,
        so it is admitted on content alone. ``False`` when the label is an
        identity-derived free-text string (the ``agent`` node's name) that no
        regex can prove content-free — that label is a top-level free-text field
        and, exactly like the free-text PROPERTIES, is admitted in a volatile
        mode ONLY carried along UNCHANGED from the stored node. This split is the
        per-type label audit the P1 review demanded: EVERY allowlisted type is
        either content-free-labeled or free-text-carried, so no shape can smuggle
        durable user text through ``GraphNode.label`` (#2672 review P1).
        """
        return self.label_literals is not None or self.label_regex is not None


_STRUCTURAL_NODE_SHAPES: Dict[str, _StructuralNodeShape] = {
    # Control-plane identity node — CAPABILITY REQUIRED in volatile modes.
    # Accretes governance/lifecycle metadata over the agent's life. Every
    # content-free governance field is per-field validated (defense-in-depth);
    # the user-derived identity fields ``name`` / ``description`` / label are
    # accepted only because (a) the capability restricts writers to first-party
    # governance code and (b) the ONLY writer that introduces a new
    # ``description`` (``persist_agent_description``) is source-gated to skip in
    # volatile modes, so on this node ``description`` is a carried-along value,
    # never freshly-introduced content (#2672 review P3). The key set is the
    # complete production governance vocabulary; a NEW field must be added here
    # consciously with a validator or a volatile-mode agent fails closed on it.
    "agent": _StructuralNodeShape(
        field_validators={
            "agent_id": _is_id_token,
            "did": _is_id_token,
            "created_at": _is_iso_timestamp,
            "constitution_hash": _is_hex_hash,
            "constitution_overlay_hash": _is_hex_hash_or_none,
            "initialBalance": _is_numeric_str_or_none,
            "name": _is_identity_text_or_none,
            "description": _is_identity_text_or_none,
            # ``avatar_hash`` is the ``store_file`` content hash of the avatar
            # image (hex), never free text — validate it as a hash so it cannot be
            # a smuggling channel on the ordinary path (the image bytes themselves
            # are blocked by the volatile-mode ``store_file`` gate) (#2672 review P1).
            "avatar_hash": _is_hex_hash_or_none,
            "bootstrap_state": _is_enum_token_or_none,
            "bootstrap_status": _is_enum_token_or_none,
            "bootstrap_stale_at": _is_iso_timestamp_or_none,
            "bootstrap_pending_age_seconds": _is_count,
            "genesis_audit": _is_governance_receipt,
            # ``supersede_genesis_audit`` archives the prior receipt here on every
            # signed reanchor, so a volatile-mode reanchor writes this property on
            # the agent node — it MUST be a canonical governance field or the
            # reanchor's agent-node write fails closed and the whole ceremony rolls
            # back (#2672 review: volatile reanchor after audit history).
            "genesis_audit_history": _is_governance_receipt,
            "emancipation_contract": _is_governance_receipt,
            "constitution_reanchor": _is_governance_receipt,
            "doctrine_bundle_hash": _is_hex_hash_or_none,
            "doctrine_bundle_files": _is_path_list_or_none,
            "doctrine_bundle_anchored_at": _is_iso_timestamp_or_none,
            "doctrine_bundle_reanchor": _is_governance_receipt,
            "doctrine_anchored_paths": _is_path_list_or_none,
            "graduated_at": _is_iso_timestamp_or_none,
            # ``openrouter_key_hash`` is the provider-issued hash of the agent's
            # delegated OpenRouter child key, persisted onto the agent node by the
            # host-owned ``payer_resolver`` so ``retirement_service`` can revoke
            # the remote key at teardown. It is content-free billing/control-plane
            # metadata (a credential handle, never user content), so it is a
            # canonical ``agent`` field validated as an opaque credential hash.
            # WITHOUT it, any agent that already carries the field fails a full
            # agent-node governance upsert closed once volatile — breaking
            # doctrine/bootstrap/audit persistence for delegated-OpenRouter agents
            # (#2672 review P2).
            "openrouter_key_hash": _is_credential_hash_or_none,
            "is_test_instance": _is_bool,
            "test_cycle_id": _is_id_token_or_none,
            # ``expected_duration`` is operator free text ("1 hour", "unspecified").
            "expected_duration": _is_bounded_text_or_none,
            "is_demo": _is_bool,
        },
        requires_capability=True,
        # Every user-derived / free-text field on the agent node. In a volatile
        # mode each is admitted ONLY carried along UNCHANGED from the stored node
        # (the CONTENT boundary that closes the free-text forgery leak, #2672
        # review P1): identity free-text (name / description / expected_duration)
        # AND the governance-receipt blobs (genesis audit, emancipation contract,
        # constitution / doctrine reanchor provenance), none of which a per-field
        # regex can prove content-free. A governance write that copies the node
        # and mutates only content-free state (mark-stale, doctrine boot-anchor)
        # carries these unchanged and passes; a genuinely-fresh receipt (the
        # first-cognition genesis audit) is written to the RAW store instead, so
        # it never needs a fresh free-text admit here.
        free_text_carry_along=frozenset({
            "name",
            "description",
            "expected_duration",
            "genesis_audit",
            "genesis_audit_history",
            "emancipation_contract",
            "constitution_reanchor",
            "doctrine_bundle_reanchor",
        }),
        # label = the agent's own name, or ``f"Agent {did}"`` — identity.
    ),
    # Content-free constitution byte-anchor — ORDINARY PATH, strict per-field.
    # ``hash`` must equal the node id (both are the constitution content hash),
    # ``type`` must be the literal ``"Constitution"``, ``created_at`` an ISO
    # timestamp — no field can carry a secret.
    "document": _StructuralNodeShape(
        field_validators={
            "hash": _is_hex_hash,
            "type": lambda v: v == "Constitution",
            "created_at": _is_iso_timestamp,
        },
        label_literals=frozenset({"KESTREL_CONSTITUTION"}),
        hash_equals_node_id="hash",
    ),
    # NOTE: ``constitution_amendment_artifact`` (the signed constitution-reanchor
    # receipt) is DELIBERATELY NOT allowlisted here. Its ``source_path`` /
    # ``signer`` / ``verification`` are free-form governance strings no per-field
    # regex can prove content-free, and — unlike the ``agent`` node's free-text —
    # the artifact is ALWAYS a fresh node (its id is its own content hash, so
    # there is never a stored value to carry along). Admitting it in a volatile
    # mode is therefore an unavoidable fresh-free-text channel, which the third
    # Terra review reproduced (a forged capability persisted arbitrary text in
    # ``source_path`` during EPHEMERAL). It is default-denied instead. No boot
    # path needs it: the runtime ``!reanchor-constitution`` command is already
    # blocked earlier by the volatile-mode ``store_file`` gate, and the offline
    # setup reanchor writes the artifact through the RAW store, outside this
    # wrapper (#2672 review P1).
    #
    # Content-free tamper-evidence anchor — ORDINARY PATH, strict per-field.
    # ``storage_ref`` is a ``store_file`` content hash (hex) or None, never text.
    "audit_anchor": _StructuralNodeShape(
        field_validators={
            "anchor_hash": _is_hex_hash,
            "storage_ref": _is_hex_hash_or_none,
            "entries_count": _is_count,
            "first_entry_at": _is_iso_timestamp_or_none,
            "last_entry_at": _is_iso_timestamp_or_none,
            "created_at": _is_iso_timestamp,
        },
        label_regex=re.compile(r"^Audit Anchor \(\d+ entries\)$"),
    ),
}

# Derived from the shape table so the two can never drift out of sync.
STRUCTURAL_GRAPH_NODE_TYPES = frozenset(_STRUCTURAL_NODE_SHAPES)

# The structural node types the control-plane capability marks (they carry
# free-text fields not provable content-free by shape). Now just the ``agent``
# identity node — ``constitution_amendment_artifact`` was dropped from the
# allowlist entirely (it is always fresh free-text). See the carried-along content
# boundary below and the header note on why the capability is same-process
# defense-in-depth rather than an unforgeable authorization boundary (#2672).
CONTROL_PLANE_ONLY_NODE_TYPES = frozenset(
    node_type
    for node_type, shape in _STRUCTURAL_NODE_SHAPES.items()
    if shape.requires_capability
)

# ── Carried-along free-text boundary (#2672 review P1 — the load-bearing gate) ──
#
# The reproduced review leak was: a feature obtains the (same-process, forgeable)
# capability and persists FRESH free-text — an ``agent.description``, or arbitrary
# text in ``constitution_amendment_artifact.source_path`` — in EPHEMERAL. Caller
# identity cannot stop this: the governance writers ARE mixin methods on the
# agent, and feature code holds the agent, so any in-process token/provenance the
# writers use is reachable by feature code too (the true boundary for untrusted
# code is the separate-process isolated feature runtime).
#
# So the free-text leak is closed by CONTENT, independent of the capability: a
# control-plane node's free-text fields (declared per-type in
# ``_StructuralNodeShape.free_text_carry_along``) may be admitted in a volatile
# mode ONLY when CARRIED ALONG UNCHANGED from the stored node (or the CAS
# ``expected`` snapshot). A governance write that copies the existing node and
# mutates only content-free state (mark-stale, doctrine boot-anchor) keeps these
# equal and passes; a FRESH or CHANGED value is refused REGARDLESS of any
# capability. Because that means a genuinely-fresh governance receipt cannot be
# admitted here, the writers that MUST persist one in a volatile mode (the
# first-cognition genesis audit) write to the RAW store instead — the smallest
# explicit source-of-truth path (#2672 review finding P3). ``avatar_hash`` and the
# other hashes/timestamps/counts are NOT carry-along fields because they are
# validated content-free by shape; ``constitution_amendment_artifact`` is not
# allowlisted at all because it is always fresh (see the note above its former
# entry).


def _free_text_carry_along_fields(node_type) -> frozenset:
    """The declared free-text carry-along fields for a node type (empty if none)."""
    shape = _STRUCTURAL_NODE_SHAPES.get(node_type)
    return shape.free_text_carry_along if shape is not None else frozenset()

# The only structural edge that must survive a volatile-mode write is the
# governance binding ``agent --governed_by--> constitution``, written during the
# startup constitution audit regardless of the agent's configured mode. Every
# content edge (knows / records_action / records_decision / concept
# co-occurrence / memory / todo links) fails closed.
STRUCTURAL_GRAPH_EDGE_LABELS = frozenset({
    "governed_by",
})


@dataclass
class PrivacyPolicy:
    """Defines what operations are allowed in each privacy mode."""
    allow_persistent_write: bool
    allow_persistent_read: bool
    require_anonymization: bool
    use_session_storage: bool
    allow_cloud_backup: bool
    
    @staticmethod
    def for_mode(mode: Union[PrivacyMode, PrivacyConfig, str]) -> "PrivacyPolicy":
        """Get the policy for a given privacy mode or config."""
        # Convert to PrivacyConfig
        if isinstance(mode, PrivacyConfig):
            config = mode
        elif isinstance(mode, PrivacyMode):
            config = privacy_mode_to_config(mode)
        elif isinstance(mode, str):
            config = get_privacy_preset(mode)
        else:
            raise TypeError(f"Expected PrivacyMode, PrivacyConfig, or str, got {type(mode)}")
        
        # Build policy from config flags. ``deidentified`` persistence is
        # fail-closed until the Safe Harbor / Expert Determination evidence
        # pipeline is in place; it must not silently degrade to full storage.
        # When that pipeline enables writes, this branch must be replaced with
        # evidence-backed de-identification rather than plain PII redaction.
        if config.requires_deidentification():
            return PrivacyPolicy(
                allow_persistent_write=False,
                allow_persistent_read=True,
                require_anonymization=False,
                use_session_storage=False,
                allow_cloud_backup=False,
            )
        return PrivacyPolicy(
            allow_persistent_write=config.allows_persistent_storage(),
            allow_persistent_read=True,  # Reading existing data is always allowed
            require_anonymization=config.requires_anonymization(),
            use_session_storage=config.uses_temp_storage(),
            allow_cloud_backup=config.allows_persistent_storage() and not config.is_ephemeral()
        )
    
    @staticmethod
    def from_config(config: PrivacyConfig) -> "PrivacyPolicy":
        """Build policy directly from PrivacyConfig."""
        if config.requires_deidentification():
            return PrivacyPolicy(
                allow_persistent_write=False,
                allow_persistent_read=True,
                require_anonymization=False,
                use_session_storage=False,
                allow_cloud_backup=False,
            )
        return PrivacyPolicy(
            allow_persistent_write=config.allows_persistent_storage(),
            allow_persistent_read=True,
            require_anonymization=config.requires_anonymization(),
            use_session_storage=config.uses_temp_storage(),
            allow_cloud_backup=config.allows_persistent_storage()
        )


class PrivacyEnforcingStorage:
    """
    A storage wrapper that enforces privacy mode at the storage layer.
    
    This is a decorator pattern that wraps AsyncStorage and
    intercepts all operations to enforce privacy policies.
    
    Usage:
        async with AsyncStorage(db_path) as storage:
            privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.ANONYMOUS)
            await privacy_storage.add_conversation("user", "Hello")
    """
    
    def __init__(self, underlying_storage, privacy_mode: Union[PrivacyMode, PrivacyConfig, str] = PrivacyMode.NORMAL):
        """
        Initialize the privacy-enforcing wrapper.
        
        Args:
            underlying_storage: The real AsyncStorage instance to wrap
            privacy_mode: Initial privacy mode (PrivacyMode, PrivacyConfig, or preset name)
        """
        self._storage = underlying_storage
        self._privacy_config = self._to_config(privacy_mode)
        self._policy = PrivacyPolicy.from_config(self._privacy_config)
        self._session_conversations: List[Dict] = []
        self._session_files: Dict[str, bytes] = {}
        # ISO-8601 timestamp recorded at the moment the wrapper transitions
        # INTO EPHEMERAL.  The leak-purge (#867) uses this to scope its
        # DELETE so that flipping a long-lived agent into EPHEMERAL for 30
        # seconds doesn't destroy the months of NORMAL history that
        # preceded it — only rows authored on/after this timestamp can be
        # leaks.  None when the agent was never in EPHEMERAL during this
        # process; refreshed each time we re-enter EPHEMERAL.
        self._entered_ephemeral_at: Optional[str] = None
        if self._privacy_config.is_ephemeral():
            self._entered_ephemeral_at = self._now_iso()
        # Optional safety-net sweep for the A2A observability sink (F076). The
        # observability store lives in the TaskManager, not the storage facade,
        # so the agent binds it after construction via
        # ``set_observability_purge``. It receives the entered-ephemeral
        # watermark and returns ``{table: rows_deleted}`` which is merged into
        # the purge breakdown. ``None`` = not wired (no observability sweep).
        self._observability_purge = None
        logger.info(f"PrivacyEnforcingStorage initialized with config: storage={self._privacy_config.storage}, llm={self._privacy_config.llm_location}")

    def set_observability_purge(self, purge_callable) -> None:
        """Bind the observability safety-net sweep (F076).

        ``purge_callable`` is an async callable ``(since_iso) -> dict`` that
        deletes ``a2a_tool_dispatches`` / ``a2a_observability`` rows authored
        since the watermark. Wired by the agent once its observability store
        exists; called from :meth:`purge_ephemeral_session`.
        """
        self._observability_purge = purge_callable

    @staticmethod
    def _now_iso() -> str:
        """Watermark format used to scope the EPHEMERAL leak-purge.

        Matches SQLite's ``datetime('now')`` shape (``YYYY-MM-DD HH:MM:SS``,
        UTC, no offset, no microseconds) so a lexicographic ``>=`` compares
        cleanly against the values stored in ``conversation_history.created_at``.
        ISO-8601 with a ``T`` separator and microseconds compares as
        strictly greater than every value the DB writes — that's the bug
        the original implementation hit, where ``2026-04-26T13:24:05.5``
        (watermark) was lexicographically *higher* than
        ``2026-04-26 13:24:06`` (row), so no rows ever matched.
        """
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def agent_id(self) -> str:
        """Get the agent_id from underlying storage for multi-tenant isolation."""
        return getattr(self._storage, 'agent_id', '')
    
    def _to_config(self, mode: Union[PrivacyMode, PrivacyConfig, str]) -> PrivacyConfig:
        """Convert various mode representations to PrivacyConfig."""
        if isinstance(mode, PrivacyConfig):
            return mode
        elif isinstance(mode, PrivacyMode):
            return privacy_mode_to_config(mode)
        elif isinstance(mode, str):
            return get_privacy_preset(mode)
        else:
            raise TypeError(f"Expected PrivacyMode, PrivacyConfig, or str, got {type(mode)}")

    @property
    def privacy_mode(self) -> PrivacyMode:
        """Backward compatibility: return PrivacyMode enum."""
        return privacy_config_to_mode(self._privacy_config)
    
    @property
    def privacy_config(self) -> PrivacyConfig:
        """Get the current privacy configuration."""
        return self._privacy_config
    
    # Keep _privacy_mode for backward compatibility in error messages
    @property
    def _privacy_mode(self) -> PrivacyMode:
        return self.privacy_mode
    
    def set_privacy_mode(self, mode: Union[PrivacyMode, PrivacyConfig, str]) -> None:
        """
        Change the privacy mode.

        Note: Changing to a more restrictive mode does NOT delete existing data.
        It only affects future operations.

        Records ``_entered_ephemeral_at`` on every transition INTO
        EPHEMERAL so the leak-purge (#867) can scope its DELETE to rows
        authored *during* the EPHEMERAL stint.  Stale watermarks from a
        prior EPHEMERAL stint are overwritten on re-entry; the watermark
        is preserved across the EPHEMERAL→exit transition because the
        purge needs to read it before clearing.
        """
        old_config = self._privacy_config
        new_config = self._to_config(mode)
        was_ephemeral = old_config.is_ephemeral()
        is_ephemeral = new_config.is_ephemeral()
        if is_ephemeral and not was_ephemeral:
            self._entered_ephemeral_at = self._now_iso()
        self._privacy_config = new_config
        self._policy = PrivacyPolicy.from_config(self._privacy_config)
        logger.info(f"Privacy config changed: storage={old_config.storage}->{self._privacy_config.storage}, llm={old_config.llm_location}->{self._privacy_config.llm_location}")

    async def purge_ephemeral_session(
        self, reason: str = "ephemeral-session-close"
    ) -> Dict[str, int]:
        """Hard-purge any data the EPHEMERAL agent may have leaked (#767).

        EPHEMERAL is the strongest privacy guarantee Kestrel offers —
        the contract is "leave no trace." If a write somehow reached
        ``conversation_history``, ``graph_nodes``, or the channels
        feature's ``channel_messages`` table despite the privacy layer
        rejecting persistent writes, this method is the safety net that
        scrubs it.

        Soft-delete (#763) is for *user delete intent* on data the user
        knew was being persisted. EPHEMERAL is the inverse — the user
        explicitly chose "don't persist." Honor that contract by never
        letting EPHEMERAL data live in trash. We bypass the soft-delete
        code path entirely and call the ``purge_*`` primitives.

        If this method actually finds rows, it WARNs and writes a
        security_audit_log entry — that's a bug in the privacy layer,
        and the audit trail is the only way the operator finds out.

        The session-local in-memory buffer (``_session_conversations``)
        is also cleared as a belt-and-braces measure, in case the
        agent flips out of EPHEMERAL while ISOLATED-style buffering
        accumulated something.

        Args:
            reason: Audit reason. Defaults to ``ephemeral-session-close``.

        Returns:
            Dict of ``{table_name: rows_destroyed}`` so callers can log.
            Zero is the happy path; non-zero is a leak.
        """
        agent_id = self.agent_id
        if not agent_id:
            logger.debug("purge_ephemeral_session: no agent_id, skipping")
            return {
                "conversation_history": 0,
                "graph_nodes": 0,
                "channel_messages": 0,
            }

        result: Dict[str, int] = {}

        # Belt-and-braces: clear in-memory ISOLATED buffer too. No row
        # count needed — the buffer never persisted.
        self._session_conversations = []
        self._session_files = {}

        # Scoped to ``_entered_ephemeral_at`` (#867) so the DELETE only
        # touches rows authored *during* the EPHEMERAL stint.  Without
        # this watermark, flipping a long-lived agent into EPHEMERAL for
        # a few seconds and back out destroyed every preexisting NORMAL
        # row — that's the wipe that prompted the scoping fix.  When the
        # watermark is missing (e.g. the agent was already EPHEMERAL
        # before the wrapper was constructed and we never observed the
        # transition), the scoped purge primitives refuse to delete
        # anything rather than fall back to the unbounded behaviour.
        since = self._entered_ephemeral_at
        if not since:
            logger.warning(
                "purge_ephemeral_session: no entered_ephemeral_at watermark "
                "(agent=%s, reason=%s) — refusing to purge to avoid the "
                "wipe-on-shutdown bug fixed in #867",
                agent_id, reason,
            )
            return {
                "conversation_history": 0,
                "graph_nodes": 0,
                "channel_messages": 0,
            }

        try:
            convs = await self._storage.purge_conversations_since(
                since, reason=reason,
            )
        except Exception as e:
            logger.warning(
                "purge_ephemeral_session: conversation purge failed: %s", e
            )
            convs = 0
        result["conversation_history"] = convs

        try:
            nodes = await self._storage.purge_agent_graph_nodes(
                since_iso=since,
            )
        except Exception as e:
            logger.warning(
                "purge_ephemeral_session: graph_nodes purge failed: %s", e
            )
            nodes = 0
        result["graph_nodes"] = nodes

        # Defense-in-depth for the channels feature (#2096 / F112): a
        # leaked channel_messages row must be swept on EPHEMERAL exit,
        # scoped to the same watermark. Tolerates the table being absent
        # when the channels feature was never loaded.
        try:
            channel_msgs = await self._storage.purge_channel_messages_since(
                since, reason=reason,
            )
        except Exception as e:
            logger.warning(
                "purge_ephemeral_session: channel_messages purge failed: %s", e
            )
            channel_msgs = 0
        result["channel_messages"] = channel_msgs

        # Safety net: sweep the A2A observability sink (a2a_tool_dispatches /
        # a2a_observability) for rows authored during the EPHEMERAL stint
        # (F076). Content-free counts that were metered are acceptable losses;
        # the contract is "leave no trace" of user content, and tool-call args
        # are user content.
        if self._observability_purge is not None:
            try:
                obs_counts = await self._observability_purge(since)
            except Exception as e:
                logger.warning(
                    "purge_ephemeral_session: observability sweep failed: %s", e
                )
                obs_counts = {}
            for table, count in (obs_counts or {}).items():
                result[table] = int(count or 0)

        # Leak accounting is scoped to the tables EPHEMERAL should NEVER write
        # user content to — conversation_history, graph_nodes, and channel_messages
        # (all three hold user text and are real leaks). The observability sink is
        # *expected* to hold content-free metric rows during an EPHEMERAL stint
        # (F076 permits counts/latency to remain), so sweeping those is routine
        # hygiene, not a privacy-layer leak — counting them would fire a spurious
        # security audit on every ephemeral session that ran a tool.
        leaked = (
            result.get("conversation_history", 0)
            + result.get("graph_nodes", 0)
            + result.get("channel_messages", 0)
        )
        if leaked > 0:
            logger.warning(
                "[privacy] WARNING: EPHEMERAL session leaked %d row(s) "
                "into persistent storage (agent=%s, since=%s, breakdown=%s); "
                "hard-purged with reason=%s",
                leaked, agent_id, since, result, reason,
            )
        else:
            logger.debug(
                "purge_ephemeral_session: clean (no leaks) for agent %s "
                "since %s",
                agent_id, since,
            )

        # Clear the watermark — the EPHEMERAL stint is over.  Re-entering
        # EPHEMERAL refreshes it via :meth:`set_privacy_mode`.
        self._entered_ephemeral_at = None

        # Audit-log emission is the caller's responsibility — the agent
        # has natural access to its SecurityFeature; the storage wrapper
        # doesn't and shouldn't try to reach back through layers.
        return result
    
    async def _check_write_permission(self, operation_name: str) -> None:
        """Check if write operations are allowed in current mode."""
        if not self._policy.allow_persistent_write and not self._policy.use_session_storage:
            raise PrivacyViolationError(
                f"Operation '{operation_name}' blocked: persistent writes are disabled in "
                f"current privacy config (storage={self._privacy_config.storage})"
            )
    
    def _anonymize_if_required(self, content: str) -> str:
        """Anonymize content if required by current policy."""
        if self._policy.require_anonymization:
            anonymize_text = get_anonymize_text()
            return anonymize_text(content)
        return content
    
    # === Conversation Storage (Privacy-Sensitive) ===
    
    async def add_conversation(self, role: str, content: str, metadata: Optional[Dict] = None,
                               session_id: Optional[str] = None,
                               rendered_content: Optional[str] = None,
                               model: Optional[str] = None,
                               provider: Optional[str] = None) -> None:
        """
        Add a conversation entry, respecting privacy mode.

        - EPHEMERAL: Raises PrivacyViolationError (use in-memory buffer instead)
        - ISOLATED: Stores in session-local list
        - ANONYMOUS: Anonymizes content before storing
        - NORMAL/PUBLIC: Stores as-is

        Args:
            rendered_content: Write-once transport bytes for byte-stable
                cache replay (#1402); anonymized identically to ``content``
                so the redacted bytes match what was actually sent.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot store conversations in ephemeral mode. "
                "Use EphemeralSession for in-memory buffering."
            )
        if not self._policy.allow_persistent_write and not self._policy.use_session_storage:
            raise PrivacyViolationError(
                f"Cannot store conversations in current privacy config "
                f"(storage={self._privacy_config.storage})."
            )

        processed_content = self._anonymize_if_required(content)
        processed_rendered = (
            self._anonymize_if_required(rendered_content)
            if rendered_content is not None else None
        )

        if metadata is None:
            metadata = {}
        metadata["privacy_mode"] = self._privacy_mode.value

        if self._policy.use_session_storage:
            # Store in session-local list (ISOLATED mode). The session
            # buffer IS replayed across turns within the same isolated
            # session, so preserve rendered_content for cache stability
            # (#1402 codex round-1 P2).
            entry = {
                "role": role,
                "content": processed_content,
                "metadata": metadata,
                "session_id": session_id,
                "model": model,
                "provider": provider,
            }
            if processed_rendered is not None:
                entry["rendered_content"] = processed_rendered
            self._session_conversations.append(entry)
            logger.debug(f"Conversation stored in session ({len(self._session_conversations)} total)")
        else:
            # Store in persistent storage
            await self._storage.add_conversation(
                role, processed_content, metadata, session_id,
                rendered_content=processed_rendered,
                model=model,
                provider=provider,
            )
    
    async def resolve_session_id(self, provided: Optional[str]) -> Optional[str]:
        """Surface the effective session_id to the caller.

        EPHEMERAL: no persistence, return whatever was provided (or None).
        ISOLATED: session-local; the in-memory buffer doesn't expose
        time-gap heuristics, so an explicit value passes through and
        ``None`` stays ``None``.
        NORMAL/PUBLIC: delegate to the persistent store, which applies
        the 30-min-gap heuristic.
        """
        if self._privacy_config.is_ephemeral() or self._policy.use_session_storage:
            return provided
        return await self._storage.resolve_session_id(provided)

    async def get_conversation_history(
        self, limit: int = 100, session_id: str = None
    ) -> List[Dict]:
        """
        Get conversation history respecting privacy mode.

        Args:
            limit: Maximum number of messages to return
            session_id: If provided, get messages from this session only

        - ISOLATED: Returns session-local history only (ignores session_id)
        - Others: Returns from persistent storage
        """
        if self._policy.use_session_storage:
            return self._session_conversations[-limit:]
        return await self._storage.get_conversation_history(limit, session_id=session_id)
    
    def clear_session(self) -> int:
        """
        Clear session-local storage (for ISOLATED mode).
        
        Returns:
            Number of items cleared
        """
        count = len(self._session_conversations)
        self._session_conversations.clear()
        self._session_files.clear()
        logger.info(f"Session cleared: {count} conversations")
        return count
    
    async def save_session_to_persistent(self) -> int:
        """
        Save session-local storage to persistent storage (promote ISOLATED to NORMAL).
        
        Returns:
            Number of items saved
        """
        count = 0
        for conv in self._session_conversations:
            await self._storage.add_conversation(
                conv["role"],
                conv["content"],
                conv.get("metadata"),
                conv.get("session_id"),
                rendered_content=conv.get("rendered_content"),
                model=conv.get("model"),
                provider=conv.get("provider"),
            )
            count += 1
        self._session_conversations.clear()
        logger.info(f"Session saved to persistent storage: {count} conversations")
        return count
    
    # === File Storage (Privacy-Sensitive) ===
    
    async def store_file(self, content: bytes, original_name: str, metadata: Optional[Dict] = None) -> str:
        """
        Store a file, respecting privacy mode.
        
        - EPHEMERAL: Raises PrivacyViolationError
        - ISOLATED: Stores in session-local dict
        - Others: Stores in persistent storage
        """
        await self._check_write_permission("store_file")
        
        if self._policy.use_session_storage:
            # Generate hash for session storage
            import hashlib
            content_hash = hashlib.sha256(content).hexdigest()
            self._session_files[content_hash] = content
            logger.debug(f"File stored in session: {original_name} ({content_hash[:8]}...)")
            return content_hash
        
        return await self._storage.store_file(content, original_name, metadata)
    
    async def retrieve_file(self, content_hash: str) -> bytes:
        """
        Retrieve a file, checking session storage first in ISOLATED mode.
        """
        if self._policy.use_session_storage and content_hash in self._session_files:
            return self._session_files[content_hash]
        return await self._storage.retrieve_file(content_hash)
    
    # === Graph Storage (privacy-governed durable writes — #2672) ===
    #
    # The knowledge graph is durable storage, so its writes are governed by the
    # same privacy contract as conversation history. In a volatile privacy mode
    # (EPHEMERAL / ISOLATED / DEIDENTIFIED — every mode whose policy disallows
    # persistent writes) node, edge, and compare-and-swap writes default-deny.
    # Only these reach the backend: (1) content-free structural types
    # (``document`` / ``audit_anchor`` and the ``governed_by`` edge), validated by
    # strict per-field semantic validators so no user text can ride in a ``hash`` /
    # ``type`` / ``*_at`` field; and (2) the ``agent`` control-plane identity node,
    # whose content-free fields are per-field validated, and whose free-text fields
    # (identity text + governance receipts) are admitted ONLY carried along
    # unchanged (``_enforce_free_text_carry_along``) — the CONTENT check that
    # actually closes the free-text leak, independent of the (same-process,
    # defense-in-depth) capability marker. ``constitution_amendment_artifact`` is
    # no longer admitted at all (it is always fresh free-text). Everything else —
    # user-derived and unknown writes, a fresh/changed free-text field — raises
    # ``PrivacyViolationError`` so the tool caller sees the rejection instead of a
    # silent "success". Reads and deletes are never gated. NORMAL / PUBLIC /
    # ANONYMOUS pass through unchanged.

    @property
    def _graph_writes_governed(self) -> bool:
        """True when durable graph writes must be privacy-governed.

        Governed in every mode whose policy disallows persistent writes —
        EPHEMERAL, ISOLATED, and DEIDENTIFIED. NORMAL / PUBLIC / ANONYMOUS
        (persistent-write modes) pass through unchanged, preserving the
        pre-#2672 durable-mode behaviour. ISOLATED is governed too: unlike
        conversation history it has no in-memory graph buffer, so a durable
        graph write there is a real leak, not a session-local write.
        """
        return not self._policy.allow_persistent_write

    def allows_persistent_writes(self) -> bool:
        """Whether durable, user-derived writes are permitted in the current mode.

        ``True`` in persistent modes (NORMAL / PUBLIC / ANONYMOUS); ``False`` in
        the volatile modes (EPHEMERAL / ISOLATED / DEIDENTIFIED) whose contract
        forbids persisting user-derived content. The memory consolidator consults
        this before persisting a durable episode summary, so manual / scheduled
        consolidation cannot leak a user-derived episode into ``memory_episodes``
        while the agent is volatile (#2672).
        """
        return self._policy.allow_persistent_write

    def _assert_graph_node_write_allowed(
        self, node, operation: str, *, validate_label: bool = True,
        capability: Any = None,
    ) -> None:
        """Fail closed on a durable graph node write in a volatile mode.

        Two admit tiers, each closing one Terra finding:

        * Content-free structural types (``document`` / ``audit_anchor``) are
          admitted on the ordinary (untrusted) path, but every property is
          validated by its per-field SEMANTIC validator (a ``hash`` must be hex of
          a digest's length, a ``*_at`` must parse as a timestamp, an
          ``entries_count`` must be a non-negative int, a ``type`` must be the
          expected literal, ``document.hash`` must equal the node's own id). A
          short secret is none of those, so it fails closed in every field
          (finding P1) — this replaces the old "any short single-line string is
          content-free" check.
        * The ``agent`` control-plane node carries free-text fields no per-field
          regex can prove content-free. ``capability`` is a same-process
          defense-in-depth marker (see the header note), NOT an authorization
          boundary — it is checked here so an absent/obviously-forged marker is
          refused, but the ACTUAL privacy gate for the ``agent`` node is the
          carried-along free-text boundary applied by the async callers
          (:meth:`_enforce_free_text_carry_along` / the CAS ``expected`` check),
          which refuses a fresh/changed free-text field even to a caller holding a
          genuine marker. Its content-free fields are per-field validated.

        In every case the node must match its type's COMPLETE canonical shape —
        a known ``node_type``, property keys drawn only from that type's key set,
        and (when ``validate_label``) a label matching that type's rule. Unknown
        types, unrecognized keys, and non-canonical labels all raise.

        ``validate_label`` is ``False`` on a compare-and-swap *update*, where the
        primitive writes only ``properties`` and leaves the stored label
        untouched, so ``new_node.label`` is never persisted and must not be
        judged.
        """
        if not self._graph_writes_governed:
            return
        node_type = getattr(node, "node_type", None)
        shape = _STRUCTURAL_NODE_SHAPES.get(node_type)
        if shape is None:
            raise PrivacyViolationError(
                f"Graph write '{operation}' blocked: node_type={node_type!r} is "
                f"not a content-free structural type, and durable graph writes "
                f"are disabled in the current privacy config "
                f"(storage={self._privacy_config.storage}). User-derived and "
                f"unknown graph nodes are default-denied in volatile privacy "
                f"modes (#2672)."
            )
        if shape.requires_capability and not _has_control_plane_capability(capability):
            raise PrivacyViolationError(
                f"Graph write '{operation}' blocked: node_type={node_type!r} is a "
                f"control-plane node (it carries identity / governance state no "
                f"per-field check can prove content-free), so a durable write to it "
                f"in the current privacy config (storage={self._privacy_config.storage}) "
                f"must carry the control-plane capability marker. That marker is "
                f"same-process defense-in-depth; the ACTUAL identity-content gate is "
                f"the carried-along boundary applied by the async caller (#2672)."
            )
        properties = getattr(node, "properties", None) or {}
        if not isinstance(properties, dict):
            raise PrivacyViolationError(
                f"Graph write '{operation}' blocked: {node_type!r} node "
                f"properties must be a mapping of content-free fields, got "
                f"{type(properties).__name__} (#2672)."
            )
        extra_keys = set(properties) - shape.keys
        if extra_keys:
            raise PrivacyViolationError(
                f"Graph write '{operation}' blocked: {node_type!r} node carries "
                f"non-canonical propert{'y' if len(extra_keys) == 1 else 'ies'} "
                f"{sorted(extra_keys)!r}. A structural node may hold only its "
                f"content-free identity/governance fields in the current privacy "
                f"config (storage={self._privacy_config.storage}); any other key "
                f"could carry user content and is default-denied (#2672)."
            )
        # Per-field semantic validation. For content-free types this is the
        # primary guard (finding P1); for control-plane types it is
        # defense-in-depth behind the capability. A field whose value fails its
        # semantic validator fails the whole write closed.
        for key, value in properties.items():
            validator = shape.field_validators.get(key)
            if validator is None or not validator(value):
                raise PrivacyViolationError(
                    f"Graph write '{operation}' blocked: {node_type!r} node "
                    f"property {key!r} is not a valid content-free value for its "
                    f"field (expected a hash / timestamp / count / enum / bounded "
                    f"identity as the field's semantics require). A free-text or "
                    f"otherwise-shaped value could carry user content and is "
                    f"default-denied in the current privacy config "
                    f"(storage={self._privacy_config.storage}) (#2672)."
                )
        if shape.hash_equals_node_id is not None:
            # The hash field must equal the node's own content-hash id, so a
            # caller cannot use an otherwise-valid 64-hex field as a channel for
            # 32 arbitrary bytes — the value is pinned to the id (#2672 P1).
            pinned = shape.hash_equals_node_id
            if properties.get(pinned) != getattr(node, "node_id", None):
                raise PrivacyViolationError(
                    f"Graph write '{operation}' blocked: {node_type!r} node "
                    f"property {pinned!r} must equal the node's own content-hash "
                    f"id; a hash field that is not the node's own id could carry "
                    f"arbitrary bytes and is default-denied (#2672)."
                )
        if validate_label:
            self._assert_structural_label_allowed(
                node_type, getattr(node, "label", None), shape, operation
            )

    def _assert_structural_label_allowed(
        self, node_type, label, shape: "_StructuralNodeShape", operation: str
    ) -> None:
        """Fail closed unless a structural node's label matches its type's rule."""
        if not shape.label_content_free:
            # Identity-derived label (the agent's own name): agent state, not
            # conversation content, but STILL user-derived free-text that no regex
            # can prove content-free. The load-bearing gate for it is the
            # carried-along boundary applied by the async caller
            # (:meth:`_enforce_free_text_carry_along` /
            # :meth:`_governed_compare_and_swap`), which admits it only UNCHANGED
            # from the stored trusted label — the same treatment as the free-text
            # properties (#2672 review P1). Here we only enforce the shape: it must
            # be a plain string, never a structured payload smuggling content.
            if label is not None and not isinstance(label, str):
                raise PrivacyViolationError(
                    f"Graph write '{operation}' blocked: {node_type!r} node label "
                    f"must be a string identity, got {type(label).__name__} "
                    f"(#2672)."
                )
            return
        if isinstance(label, str):
            if shape.label_literals is not None and label in shape.label_literals:
                return
            if shape.label_regex is not None and shape.label_regex.match(label):
                return
        raise PrivacyViolationError(
            f"Graph write '{operation}' blocked: {node_type!r} node label "
            f"{label!r} is not the canonical content-free label for this "
            f"structural type. A non-canonical label could carry user content "
            f"and is default-denied in the current privacy config "
            f"(storage={self._privacy_config.storage}) (#2672)."
        )

    def _free_text_carry_violation(
        self, node_type, new_props, stored_props
    ) -> Optional[str]:
        """First free-text field on a control-plane write that is freshly
        introduced or changed vs. the stored/expected node, or ``None``.

        This is the CONTENT boundary that actually closes the free-text leak
        (#2672 review P1): in a volatile mode a node type's declared free-text
        fields (identity free-text AND governance-receipt blobs) are admitted only
        carried along unchanged, so a fresh/changed value is refused whether or not
        the caller presents the (same-process, defense-in-depth) capability.
        """
        stored_props = stored_props or {}
        for field in _free_text_carry_along_fields(node_type):
            if field in new_props and new_props.get(field) != stored_props.get(field):
                return field
        return None

    def _raise_free_text_carry(self, node_type, field, operation) -> None:
        raise PrivacyViolationError(
            f"Graph write '{operation}' blocked: {node_type!r} node property "
            f"{field!r} is user-derived free-text (identity text or a governance "
            f"receipt) being introduced or changed (its value differs from the "
            f"stored node). In the current privacy config "
            f"(storage={self._privacy_config.storage}) a control-plane node admits "
            f"its free-text fields ONLY carried along UNCHANGED — rename / "
            f"description edits skip their durable writes while volatile, and a "
            f"genuinely-fresh governance receipt is written to the raw store, so a "
            f"fresh/changed value here is user content and is default-denied "
            f"REGARDLESS of any control-plane capability (#2672)."
        )

    @staticmethod
    def _label_is_free_text(node_type) -> bool:
        """Whether a structural type's LABEL is identity-derived free-text.

        ``True`` only for a control-plane node whose label is the agent's own
        name (no literal/regex proves it content-free); such a label is a
        top-level free-text field guarded by carry-along (#2672 review P1).
        """
        shape = _STRUCTURAL_NODE_SHAPES.get(node_type)
        return shape is not None and not shape.label_content_free

    @staticmethod
    def _is_content_free_identity_label(node) -> bool:
        """Whether ``node``'s label is the DID-derived ``Agent {node_id}`` form.

        That form is fully determined by the node's own id, so it carries NO user
        text and is content-free by construction — admitted even on a fresh create
        with no stored label to carry from. This is exactly what the born-agent
        boot path writes when it first materialises the identity node in a volatile
        mode, and it is the ONLY fresh identity label a volatile-mode create admits
        (a real user-authored name is refused) (#2672 review P1)."""
        label = getattr(node, "label", None)
        node_id = getattr(node, "node_id", None)
        return isinstance(node_id, str) and label == f"Agent {node_id}"

    def _label_carry_violation(self, node, stored) -> bool:
        """True when an identity-labeled control-plane node's LABEL would introduce
        or MUTATE durable user text vs. the stored node.

        The top-level label is the agent's name — user-derived free-text exactly
        like ``properties['name']``. ``add_node`` is a whole-row upsert, so on an
        EXISTING node any label other than the stored one is a durable mutation:
        ``None`` would CLEAR the stored name and the DID-derived ``Agent {node_id}``
        form would OVERWRITE a stored user name. So the admitted set depends on
        whether a stored row exists (#2672 review P2):

        * Existing node (``stored is not None``) — carry-along means EXACTLY the
          stored label, nothing else. Require ``new_label == stored_label`` (which
          also permits ``None`` only when the stored label is itself ``None``).
        * Fresh create (``stored is None``) — no trusted label to carry, so admit
          only a label that carries no user text: ``None`` (writes nothing) or the
          content-free DID-derived ``Agent {node_id}`` form. Any other non-null
          label is fresh user content and is refused.
        """
        new_label = getattr(node, "label", None)
        if stored is not None:
            # Carry-along on an existing node = identical to the stored label only.
            return new_label != getattr(stored, "label", None)
        # Fresh create: no stored label to carry from.
        if new_label is None or self._is_content_free_identity_label(node):
            return False
        return True

    def _raise_label_carry(self, node_type, operation) -> None:
        raise PrivacyViolationError(
            f"Graph write '{operation}' blocked: {node_type!r} node label is "
            f"user-derived identity free-text being introduced or changed (it "
            f"differs from the stored node's label). In the current privacy config "
            f"(storage={self._privacy_config.storage}) a control-plane node admits "
            f"its identity label ONLY carried along UNCHANGED — a volatile rename "
            f"updates the live session name but skips the durable node — so a "
            f"fresh/changed label is user content and is default-denied REGARDLESS "
            f"of any control-plane capability, closing the label smuggling channel "
            f"(#2672 review P1)."
        )

    async def _enforce_free_text_carry_along(self, node, store, operation) -> None:
        """Read the stored node and refuse a fresh/changed free-text field — in a
        property OR in the top-level LABEL — on a control-plane write in a volatile
        mode. No-op otherwise.

        The stored read is only reached when the write actually carries one of the
        guarded free-text properties OR targets a free-text-labeled control-plane
        type, so ordinary content-free governance writes on content-free-labeled
        types pay nothing.
        """
        if not self._graph_writes_governed:
            return
        node_type = getattr(node, "node_type", None)
        carry_fields = _free_text_carry_along_fields(node_type)
        props = getattr(node, "properties", None) or {}
        has_prop_carry = bool(carry_fields) and any(f in props for f in carry_fields)
        label_free_text = self._label_is_free_text(node_type)
        if not has_prop_carry and not label_free_text:
            return
        stored = await store.get_node(getattr(node, "node_id", None))
        stored_props = (getattr(stored, "properties", None) or {}) if stored else {}
        field = self._free_text_carry_violation(node_type, props, stored_props)
        if field is not None:
            self._raise_free_text_carry(node_type, field, operation)
        # The label is a top-level free-text field on the identity-labeled agent
        # node — guard it with the same carried-along boundary as the properties,
        # or a forged capability could persist arbitrary durable user text through
        # ``GraphNode.label`` while the property gate looked closed (#2672 P1).
        if label_free_text and self._label_carry_violation(node, stored):
            self._raise_label_carry(node_type, operation)

    def _assert_graph_edge_write_allowed(
        self, label, operation: str, properties: Optional[Dict] = None,
        *, capability: Any = None,
    ) -> None:
        """Fail closed on a durable graph edge write in a volatile mode.

        Allows only the structural governance relationship(s), AND only as a pure
        binding: a structural edge must carry no properties, so user content
        cannot ride in the edge payload. Every content edge, and any structural
        edge with a non-empty payload, raises ``PrivacyViolationError``.
        ``governed_by`` is a content-free binding admitted on the ordinary path;
        ``capability`` is accepted for call-site uniformity but the edge allowlist
        does not depend on it.
        """
        if not self._graph_writes_governed:
            return
        if label not in STRUCTURAL_GRAPH_EDGE_LABELS:
            raise PrivacyViolationError(
                f"Graph write '{operation}' blocked: edge label={label!r} is not "
                f"a content-free structural relationship, and durable graph "
                f"writes are disabled in the current privacy config "
                f"(storage={self._privacy_config.storage}). User-derived and "
                f"unknown graph edges are default-denied in volatile privacy "
                f"modes (#2672)."
            )
        if properties:
            raise PrivacyViolationError(
                f"Graph write '{operation}' blocked: structural edge {label!r} "
                f"must be a pure binding with no properties, but got keys "
                f"{sorted(properties)!r}. An edge payload could carry user "
                f"content and is default-denied in volatile privacy modes "
                f"(#2672)."
            )

    async def _governed_compare_and_swap(
        self, store, node_id, expected, new_node, operation: str,
        *, capability: Any = None,
    ):
        """Shared, TOCTOU-free CAS governance for the wrapper and the ``.graph``
        proxy.

        In a durable-write mode this is a straight passthrough. In a volatile
        mode it (1) validates ``new_node`` against the structural canonical
        shape — the label only on a compare-and-create, since a swap never
        writes it, and gated by ``capability`` for the control-plane types —
        and (2) pins the operation to that exact node type via the primitive's
        ``allowed_node_types`` predicate. That pin is what stops the
        relabel exploit: a swap ignores ``new_node.node_type`` and writes
        ``properties`` onto whatever row exists, so authorizing on
        ``new_node.node_type`` alone would let a caller rewrite an existing
        user-derived node (e.g. a ``concept``) under a spoofed structural type.
        The storage layer instead refuses the swap unless the STORED row is of
        the validated structural type, atomically, without a wrapper pre-read.
        """
        if not self._graph_writes_governed:
            return await store.compare_and_swap_node(node_id, expected, new_node)
        self._assert_graph_node_write_allowed(
            new_node, operation, validate_label=(expected is None),
            capability=capability,
        )
        # Carried-along free-text boundary, atomically: compare the new node's
        # free-text fields against the caller's ``expected`` snapshot (no extra
        # read — the atomic CAS itself fails if ``expected`` is a lie). A
        # compare-and-create (``expected is None``) means fresh free-text → refused.
        new_type = getattr(new_node, "node_type", None)
        if _free_text_carry_along_fields(new_type):
            field = self._free_text_carry_violation(
                new_type, getattr(new_node, "properties", None) or {}, expected or {}
            )
            if field is not None:
                self._raise_free_text_carry(new_type, field, operation)
        # A swap NEVER writes ``label`` (the primitive is properties-only), so the
        # label carry-along only bites on a compare-and-CREATE (``expected is
        # None``), where the full node — including a fresh label — WOULD be
        # inserted. With no stored trusted label to carry from, only the
        # content-free ``Agent {node_id}`` form is admitted; a fresh user-authored
        # label is refused, closing the label smuggling channel on the CAS path too
        # (#2672 P1).
        if (
            expected is None
            and self._label_is_free_text(new_type)
            and self._label_carry_violation(new_node, None)
        ):
            self._raise_label_carry(new_type, operation)
        result = await store.compare_and_swap_node(
            node_id,
            expected,
            new_node,
            allowed_node_types=frozenset({getattr(new_node, "node_type", None)}),
        )
        if result == NodeSwapResult.TYPE_NOT_ALLOWED:
            node_type = getattr(new_node, "node_type", None)
            raise PrivacyViolationError(
                f"Graph write '{operation}' blocked: the existing node at "
                f"{node_id!r} is not a {node_type!r} structural node, so a "
                f"durable properties swap onto it is a user-derived write and is "
                f"default-denied in the current privacy config "
                f"(storage={self._privacy_config.storage}). CAS cannot relabel a "
                f"user-derived node as structural to smuggle content through "
                f"(#2672)."
            )
        return result

    async def add_node(self, node, *, capability: Any = None) -> None:
        """Add a graph node, governed by the volatile-mode write policy.

        In a durable-write mode this is a straight pass-through. In a volatile
        mode it default-denies: only content-free structural nodes (strict
        per-field validation) are admitted, and the ``agent`` control-plane node
        carries the ``capability`` marker. The load-bearing privacy check for the
        ``agent`` node is the
        carried-along free-text boundary applied here after the structural check
        (:meth:`_enforce_free_text_carry_along`), which refuses a fresh/changed
        free-text field regardless of the marker. User-derived (facts, concepts,
        todos, decisions, episodes), unknown types, non-canonical fields, and fresh
        free-text content raise ``PrivacyViolationError`` before any row is
        written.

        ``capability`` is the same-process defense-in-depth marker from
        :func:`acquire_control_plane_capability` (NOT an authorization boundary —
        see the module header note); the privacy guarantee does not depend on it.
        """
        self._assert_graph_node_write_allowed(
            node, "add_node", capability=capability
        )
        await self._enforce_free_text_carry_along(node, self._storage, "add_node")
        await self._storage.add_node(node)

    async def compare_and_swap_node(
        self, node_id, expected, new_node, *, capability: Any = None
    ):
        """Atomically compare-and-swap a graph node, governed by the policy.

        The write intent is governed *before* delegating — the primitive itself
        stays a single atomic passthrough. The wrapper never decomposes it into
        ``get_node`` + ``add_node``: that would reintroduce exactly the TOCTOU
        race the caller is trying to close. See :meth:`_governed_compare_and_swap`.
        """
        return await self._governed_compare_and_swap(
            self._storage, node_id, expected, new_node, "compare_and_swap_node",
            capability=capability,
        )

    async def get_node(self, node_id: str):
        """Get a graph node. Read — never privacy-gated."""
        return await self._storage.get_node(node_id)

    async def add_edge(self, source_id: str, target_id: str, label: str, properties: Optional[Dict] = None,
                        *, capability: Any = None):
        """Add a graph edge, governed by the volatile-mode write policy.

        In a volatile mode only structural governance edges (``governed_by``),
        carrying no properties, are admitted; content edges and payload-bearing
        edges raise ``PrivacyViolationError``. ``capability`` is accepted for
        call-site uniformity with the node writers.
        """
        self._assert_graph_edge_write_allowed(
            label, "add_edge", properties, capability=capability
        )
        await self._storage.add_edge(source_id, target_id, label, properties)

    async def delete_edge(self, source_id: str, target_id: str, label: str) -> None:
        """Remove a graph edge. Delete — never privacy-gated (removal is not a leak)."""
        await self._storage.delete_edge(source_id, target_id, label)

    @asynccontextmanager
    async def transaction(self):
        """Atomic write unit over the underlying storage.

        The transaction scope itself grants no extra write permission —
        every operation inside it still goes through this wrapper's
        privacy checks individually. It only guarantees that the writes
        which ARE permitted commit or roll back together.
        """
        async with self._storage.transaction():
            yield

    async def delete_node(self, node_id: str) -> None:
        """Delete a graph node and its edges. Structural operation."""
        await self._storage.delete_node(node_id)

    async def get_edges_from(self, node_id: str) -> List:
        """Get outgoing edges from a node."""
        return await self._storage.get_edges_from(node_id)

    async def get_edges_to(self, node_id: str) -> List:
        """Get incoming edges to a node."""
        return await self._storage.get_edges_to(node_id)

    # === Private Agent Identity Resources (privacy-governed durable writes) ===
    #
    # A private identity resource (the SOUL body most notably) is user-derived
    # identity content: ``create_version`` writes the encrypted body into
    # ``agent_identity_resources`` AND records a durable ``agent_identity_resource``
    # graph node + ``has_private_identity_resource`` edge on the RAW store (bypassing
    # the graph proxy). So in a volatile privacy mode these writes are a real leak
    # and must default-deny — this is the storage-layer backstop; the SOUL callers
    # (``save_soul_md`` / startup seed promotion / rename) also skip at their source
    # so the raise only fires on an un-gated path (#2672 review P1). Reads pass
    # through unchanged.

    def _assert_private_resource_write_allowed(
        self, operation: str, resource_type: Optional[str]
    ) -> None:
        """Fail closed on a durable private identity-resource write while volatile."""
        if not self._graph_writes_governed:
            return
        raise PrivacyViolationError(
            f"Private identity-resource write '{operation}' "
            f"(resource_type={resource_type!r}) blocked: the encrypted resource "
            f"body and its durable graph reference are user-derived identity "
            f"content, default-denied in the current privacy config "
            f"(storage={self._privacy_config.storage}). The SOUL and other private "
            f"resources are not persisted in volatile privacy modes (#2672)."
        )

    async def create_agent_resource_version(self, *args, **kwargs):
        """Create a private identity-resource version, governed by the write policy."""
        resource_type = kwargs.get("resource_type") or (args[0] if args else None)
        self._assert_private_resource_write_allowed(
            "create_agent_resource_version", resource_type
        )
        return await self._storage.create_agent_resource_version(*args, **kwargs)

    async def get_current_agent_resource(self, *args, **kwargs):
        """Read the current private identity resource."""
        return await self._storage.get_current_agent_resource(*args, **kwargs)

    async def get_agent_resource_public_metadata(self, *args, **kwargs):
        """Read public, body-free identity-resource metadata."""
        return await self._storage.get_agent_resource_public_metadata(*args, **kwargs)

    async def promote_soul_seed(self, *args, **kwargs):
        """Promote local SOUL.md seed/cache content into canonical storage."""
        self._assert_private_resource_write_allowed(
            "promote_soul_seed", SOUL_MARKDOWN_RESOURCE_TYPE
        )
        return await self._storage.promote_soul_seed(*args, **kwargs)

    # === RAG Storage ===
    
    async def chunk_document(self, content_hash: str) -> int:
        """Chunk a document for RAG. Respects privacy mode."""
        await self._check_write_permission("chunk_document")
        return await self._storage.chunk_document(content_hash)
    
    async def search_chunks(
        self, query: str, limit: int = 5, min_score: float = 0.0,
    ) -> List[Dict]:
        """Search document chunks. Read-only, always allowed.

        ``min_score`` (#1404) forwarded to the underlying RAG store.
        """
        return await self._storage.search_chunks(query, limit, min_score=min_score)
    
    # === Backup Operations (Privacy-Sensitive) ===
    
    async def create_backup_blob(self, include_db: bool = True) -> bytes:
        """
        Create a backup blob.
        
        - EPHEMERAL/ISOLATED: Raises PrivacyViolationError
        - Others: Creates backup
        """
        if not self._policy.allow_cloud_backup:
            raise PrivacyViolationError(
                f"Backups are disabled in current privacy config (storage={self._privacy_config.storage})"
            )
        return await self._storage.create_backup_blob(include_db)
    
    async def restore_from_backup_blob(self, backup_blob: bytes) -> Dict:
        """Restore from a backup blob."""
        await self._check_write_permission("restore_from_backup_blob")
        return await self._storage.restore_from_backup_blob(backup_blob)
    
    async def record_backup_artifact(self, agent_id: str, result: Any) -> str:
        """Record a backup artifact in the graph store."""
        await self._check_write_permission("record_backup_artifact")
        return await self._storage.record_backup_artifact(agent_id, result)
    
    async def get_nodes_by_type(self, node_type: str) -> List:
        """Get all nodes of a specific type."""
        return await self._storage.get_nodes_by_type(node_type)
    
    async def get_file_metadata(self, content_hash: str) -> Optional[Dict]:
        """Get file metadata."""
        return await self._storage.get_file_metadata(content_hash)
    
    async def search_case_law(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search case law (constitutional RAG)."""
        return await self._storage.search_case_law(query, top_k)
    
    @property
    def encryption_enabled(self) -> bool:
        """Check if conversation encryption at rest is enabled.

        This provides a safe way to check encryption status without
        accessing the conversation store directly.
        """
        conv_store = getattr(self._storage, 'conversation', None)
        if conv_store and hasattr(conv_store, 'encryption_enabled'):
            return conv_store.encryption_enabled
        return False

    # === Privacy-Aware Query Methods ===
    #
    # These methods provide privacy-respecting access to the database for
    # operations that were previously done via direct storage.db access.
    # Use these instead of accessing .db, .conversation, or .files directly.

    async def query_conversations(
        self, agent_id: str, limit: int = 50, view: str = "active"
    ) -> List[Tuple]:
        """
        Query conversation history rows respecting privacy mode.

        In EPHEMERAL mode, returns an empty list (no persistent data exposed).
        In ISOLATED mode, returns session-local conversations as tuple rows.
        In other modes, queries the persistent database.

        Args:
            view: ``active`` (default) returns live, non-archived rows;
                ``archived`` returns live rows with ``archived_at`` set
                (#2149). Any other value falls back to ``active``.

        Returns rows as tuples:
        (id, role, content, metadata, created_at, model, provider)
        """
        bounded_limit = max(1, min(int(limit), 1000))
        if view not in ("active", "archived"):
            view = "active"

        if self._privacy_config.is_ephemeral():
            logger.debug("query_conversations blocked: ephemeral mode returns no data")
            return []

        if self._policy.use_session_storage:
            # In-memory session storage has no archive concept: the archived
            # view is always empty, the active view returns the buffer.
            if view == "archived":
                return []
            # Return session conversations formatted as tuple rows. Surface the
            # resolved session_id in the metadata (real id, else the sentinel)
            # so the /api/conversations grouping labels each session with an id
            # that delete_conversation_session can actually resolve — the UI and
            # the agent tools share one identity (#2019).
            rows = []
            for i, conv in enumerate(self._session_conversations):
                meta = dict(conv.get("metadata") or {})
                sid = _conv_session_id(conv)
                meta["session_id"] = (
                    sid if sid is not None else _ISOLATED_UNLABELED_SESSION_ID
                )
                rows.append((
                    i,  # synthetic row id (message id, not session id)
                    conv.get("role", ""),
                    conv.get("content", ""),
                    json.dumps(meta),
                    conv.get("created_at", None),
                    conv.get("model"),
                    conv.get("provider"),
                ))
            return rows

        if view == "archived":
            archive_clause = "archived_at IS NOT NULL"
        else:
            archive_clause = "archived_at IS NULL"

        return await self._storage.db.fetchall(f"""
            SELECT id, role, content, metadata, created_at, model, provider
            FROM conversation_history
            WHERE agent_id = ? AND deleted_at IS NULL AND {archive_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """, (agent_id, bounded_limit))

    async def search_conversations(
        self, agent_id: str, query: str, limit: int = 20, view: str = "active"
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across conversations, respecting privacy mode.

        In EPHEMERAL mode, returns an empty list (no persistent data exposed).
        In ISOLATED mode, searches the in-memory session buffer.
        In other modes, delegates to the conversation store, which decrypts
        client-side and groups hits into session summaries (#2019).

        Returns newest-first session summary dicts as produced by
        :func:`~kestrel_sovereign.storage.async_conversation_store.search_session_summaries`
        (``match_count`` / ``match_role`` / ``match_snippet`` decorated).
        """
        bounded_limit = max(1, min(int(limit), 500))
        if view not in ("active", "archived"):
            view = "active"

        if self._privacy_config.is_ephemeral():
            logger.debug("search_conversations blocked: ephemeral mode returns no data")
            return []

        if self._policy.use_session_storage:
            # In-memory session storage has no archive concept.
            if view == "archived":
                return []
            normalized = []
            for i, conv in enumerate(self._session_conversations):
                meta = dict(conv.get("metadata") or {})
                sid = _conv_session_id(conv)
                meta["session_id"] = (
                    sid if sid is not None else _ISOLATED_UNLABELED_SESSION_ID
                )
                normalized.append({
                    "id": i,
                    "role": conv.get("role", ""),
                    "content": conv.get("content", ""),
                    "metadata": meta,
                    "created_at": conv.get("created_at", None),
                })
            # Titles via the wrapper's OWN privacy-aware accessor — reading
            # self._storage directly would surface persisted names against
            # in-memory rows, leaking durable metadata into an isolated
            # session (codex r3 P1).
            try:
                names = await self.get_conversation_names() or {}
            except Exception:
                names = {}
            return search_session_summaries(
                normalized, query, names=names, limit=bounded_limit
            )

        conv_store = getattr(self._storage, "conversation", None)
        if conv_store is None:
            return []
        return await conv_store.search_sessions(
            query, view=view, limit=bounded_limit
        )

    async def query_conversation_start(
        self, message_id: str, agent_id: str
    ) -> Optional[Tuple]:
        """
        Get the created_at timestamp for a specific message, respecting privacy.

        In EPHEMERAL mode, returns None.
        In ISOLATED mode, returns from session storage.
        Otherwise queries the persistent database.

        Returns a single-element tuple (created_at,) or None.
        """
        if self._privacy_config.is_ephemeral():
            return None

        if self._policy.use_session_storage:
            try:
                idx = int(message_id)
                if 0 <= idx < len(self._session_conversations):
                    return (self._session_conversations[idx].get("created_at"),)
            except (ValueError, IndexError):
                pass
            return None

        row_id = coerce_persistent_message_id(message_id)
        if row_id is None:
            return None

        # Filter out soft-deleted anchors so detail-view requests for
        # trashed sessions return 404 from the higher layer instead of
        # silently loading their content.
        return await self._storage.db.fetchone(
            "SELECT created_at FROM conversation_history "
            "WHERE id = ? AND agent_id = ? AND deleted_at IS NULL",
            (row_id, agent_id)
        )

    async def query_conversation_messages(
        self, agent_id: str, start_time: Any, limit: int = 100
    ) -> List[Tuple]:
        """
        Get conversation messages starting from a given time, respecting privacy.

        Returns rows as tuples:
        (id, role, content, metadata, created_at, model, provider)
        """
        if self._privacy_config.is_ephemeral():
            return []

        if self._policy.use_session_storage:
            rows = []
            for i, conv in enumerate(self._session_conversations):
                rows.append((
                    i,
                    conv.get("role", ""),
                    conv.get("content", ""),
                    json.dumps(conv.get("metadata", {})) if conv.get("metadata") else None,
                    conv.get("created_at", None),
                    conv.get("model"),
                    conv.get("provider"),
                ))
            return rows[:limit]

        return await self._storage.db.fetchall("""
            SELECT id, role, content, metadata, created_at, model, provider
            FROM conversation_history
            WHERE agent_id = ? AND created_at >= ? AND deleted_at IS NULL
            ORDER BY created_at ASC
            LIMIT ?
        """, (agent_id, start_time, limit))

    async def query_session_rows(
        self, session_id: str, limit: int = 100
    ) -> List[Tuple]:
        """Resolve every message belonging to ``session_id``, respecting privacy.

        Unlike :meth:`query_conversation_messages` (which only time-gap walks
        forward from a row-id anchor), this delegates to the store's canonical
        dual-scheme resolver ``_get_session_messages`` — so it links messages
        both by time-gap clustering AND by explicit ``metadata.session_id``
        (UUID) membership. This is what lets a conversation whose continued
        turns were mis-filed under a different key still load completely on a
        hard refresh (#2012).

        Returns rows as tuples in the same shape the conversation endpoints
        expect: ``(id, role, content, metadata, created_at, model, provider)``,
        in chronological (ASC) order. The store resolver yields an 8-tuple with
        ``rendered_content`` at index 5; we drop it so positional model/provider
        accesses in the endpoint stay at indices 5/6.
        """
        if self._privacy_config.is_ephemeral():
            return []

        if self._policy.use_session_storage:
            # Isolated/temp storage keeps conversations in memory. Scope to the
            # requested session_id with the SAME matcher the delete path uses,
            # so opening one listed isolated session never shows another's
            # messages (#2019).
            rows = []
            for i, conv in enumerate(self._session_conversations):
                if not _in_session(conv, session_id):
                    continue
                rows.append((
                    i,
                    conv.get("role", ""),
                    conv.get("content", ""),
                    json.dumps(conv.get("metadata", {})) if conv.get("metadata") else None,
                    conv.get("created_at", None),
                    conv.get("model"),
                    conv.get("provider"),
                ))
            return rows[:limit]

        # Preserve the live-anchor guard the previous detail-read path had
        # (via query_conversation_start's `deleted_at IS NULL` filter): for a
        # numeric session_id, _get_session_messages looks up the anchor row
        # IGNORING deleted_at (it needs the trashed anchor's timestamp for
        # restore workflows), so a soft-deleted anchor whose cluster siblings
        # are still live would otherwise leak rows instead of 404ing. UUID ids
        # have no single anchor row — their live filtering is intrinsic.
        row_id = coerce_persistent_message_id(session_id)
        if row_id is not None:
            anchor = await self._storage.db.fetchone(
                "SELECT 1 FROM conversation_history "
                "WHERE id = ? AND agent_id = ? AND deleted_at IS NULL",
                (row_id, self._storage.agent_id),
            )
            if not anchor:
                return []

        raw_rows = await self._storage.get_session_message_rows(session_id, limit)
        # get_session_message_rows returns rows DESC; the endpoint walks ASC.
        normalized = []
        for row in reversed(raw_rows):
            # (id, role, content, metadata, created_at, rendered_content,
            #  model, provider) -> drop rendered_content at index 5.
            normalized.append((
                row[0], row[1], row[2], row[3], row[4],
                row[6] if len(row) > 6 else None,
                row[7] if len(row) > 7 else None,
            ))
        return normalized

    async def session_exists(self, session_id: str) -> bool:
        """Whether a session exists at all, respecting privacy.

        ``query_session_rows`` strips ``session_marker`` rows, so a freshly
        started (marker-only) session resolves to zero rows. Callers use this
        to tell "session exists but has no displayable messages yet" (→ 200
        with an empty list) from "session not found" (→ 404), preserving the
        old ``query_conversation_start`` contract (#2012).

        - numeric session_id: existence is a LIVE anchor row (so a
          soft-deleted anchor still 404s, matching the trash semantics).
        - UUID session_id: any LIVE row carrying it in ``metadata.session_id``
          (the marker alone is enough).
        """
        if self._privacy_config.is_ephemeral():
            return False
        if self._policy.use_session_storage:
            # Scope to the requested session_id (same matcher as the delete and
            # detail-read paths) so existence is per-session, not "any isolated
            # message at all" (#2019).
            return any(
                _in_session(conv, session_id) for conv in self._session_conversations
            )

        row_id = coerce_persistent_message_id(session_id)
        if row_id is not None:
            anchor = await self._storage.db.fetchone(
                "SELECT 1 FROM conversation_history "
                "WHERE id = ? AND agent_id = ? AND deleted_at IS NULL",
                (row_id, self._storage.agent_id),
            )
            return anchor is not None

        esc = _escape_like_session_value(str(session_id))
        row = await self._storage.db.fetchone(
            "SELECT 1 FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL "
            "AND (metadata LIKE ? ESCAPE '\\' OR metadata LIKE ? ESCAPE '\\') "
            "LIMIT 1",
            (
                self._storage.agent_id,
                f'%"session_id": "{esc}"%',
                f'%"session_id":"{esc}"%',
            ),
        )
        return row is not None

    async def query_last_conversation_row(
        self, agent_id: str
    ) -> Optional[Tuple]:
        """
        Get the most recent conversation row for an agent, respecting privacy.

        Returns a tuple (id, created_at) or None.
        """
        if self._privacy_config.is_ephemeral():
            return None

        if self._policy.use_session_storage:
            if self._session_conversations:
                idx = len(self._session_conversations) - 1
                return (idx, self._session_conversations[idx].get("created_at"))
            return None

        return await self._storage.db.fetchone("""
            SELECT id, created_at FROM conversation_history
            WHERE agent_id = ? AND deleted_at IS NULL
            ORDER BY id DESC LIMIT 1
        """, (agent_id,))

    async def delete_conversation_message(
        self, message_id: int, agent_id: str
    ) -> bool:
        """
        Soft-delete a conversation message by ID, respecting privacy mode (#763).

        In EPHEMERAL mode, raises PrivacyViolationError (nothing to delete).
        In ISOLATED mode, removes from in-memory session storage (which has
        no soft/hard distinction — the row never persisted).
        Otherwise stamps ``deleted_at`` on the persistent row so it can be
        restored from Trash. The matching memory_pin is hard-deleted to
        preserve the sovereign invariant that pins cannot block, delay, or
        resurrect erased content (#750).

        Returns True if a row was soft-deleted, False if not found or
        already in trash.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot delete conversations in ephemeral mode (no persistent data)."
            )

        if self._policy.use_session_storage:
            try:
                idx = int(message_id)
                if 0 <= idx < len(self._session_conversations):
                    self._session_conversations.pop(idx)
                    return True
            except (ValueError, IndexError):
                pass
            return False

        row_id = coerce_persistent_message_id(message_id)
        if row_id is None:
            return False

        await self._check_write_permission("delete_conversation_message")
        deleted = await self._storage.delete_message(row_id)

        # Sovereign override: pins cannot point into Trash. Hard-delete
        # the matching pin so the user can't navigate from a pin into a
        # soft-deleted message. If the user later restores the message,
        # they can re-pin it explicitly.
        if deleted:
            await self._delete_pin_for_message(row_id, agent_id)

        return deleted

    async def _delete_pin_for_message(
        self, row_id: int, agent_id: str
    ) -> None:
        """Best-effort drop of any pin pointing at this message id.

        Tolerates a missing ``memory_pins`` table (see
        ``_delete_orphaned_pins`` for rationale).
        """
        try:
            await self._storage.db.execute_commit(
                "DELETE FROM memory_pins WHERE message_id = ? AND agent_id = ?",
                (row_id, agent_id)
            )
        except Exception as e:
            logger.debug(
                "Pin cleanup skipped (memory_pins likely absent): %s", e
            )

    async def restore_conversation_message(
        self, message_id: int, agent_id: str
    ) -> bool:
        """Clear deleted_at on a soft-deleted message (#763 / #765).

        EPHEMERAL has nothing to restore (raises). ISOLATED has no
        persistent state, so restore is a no-op (returns False).
        Otherwise delegates to the conversation store.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot restore conversations in ephemeral mode (no persistent data)."
            )
        if self._policy.use_session_storage:
            return False

        row_id = coerce_persistent_message_id(message_id)
        if row_id is None:
            return False

        await self._check_write_permission("restore_conversation_message")
        return await self._storage.restore_message(row_id)

    async def restore_conversation_session(
        self, session_id: str, agent_id: str
    ) -> int:
        """Clear deleted_at on every soft-deleted message in a session.

        EPHEMERAL raises (no persistent data). ISOLATED returns 0 (the
        in-memory list has no Trash distinction). Otherwise delegates to
        the conversation store.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot restore conversations in ephemeral mode (no persistent data)."
            )
        if self._policy.use_session_storage:
            return 0

        await self._check_write_permission("restore_conversation_session")
        return await self._storage.restore_conversation_session(session_id)

    async def archive_conversation_session(
        self, session_id: str, agent_id: str
    ) -> int:
        """Stamp archived_at on every live message in a session (#2149).

        EPHEMERAL raises (no persistent data). ISOLATED returns 0 (the
        in-memory list has no archive distinction). Otherwise delegates to
        the conversation store.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot archive conversations in ephemeral mode (no persistent data)."
            )
        if self._policy.use_session_storage:
            return 0

        await self._check_write_permission("archive_conversation_session")
        return await self._storage.archive_conversation_session(session_id)

    async def unarchive_conversation_session(
        self, session_id: str, agent_id: str
    ) -> int:
        """Clear archived_at on every archived message in a session (#2149).

        EPHEMERAL raises (no persistent data). ISOLATED returns 0. Otherwise
        delegates to the conversation store.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot unarchive conversations in ephemeral mode (no persistent data)."
            )
        if self._policy.use_session_storage:
            return 0

        await self._check_write_permission("unarchive_conversation_session")
        return await self._storage.unarchive_conversation_session(session_id)

    async def purge_conversation_message(
        self, message_id: int, agent_id: str, reason: str = "user-initiated"
    ) -> bool:
        """Hard-delete a single message (#763).

        Permanent — bypasses Trash. EPHEMERAL raises (nothing to purge).
        ISOLATED falls through to the soft-delete path because the row
        never persisted in the first place.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot purge conversations in ephemeral mode (no persistent data)."
            )

        if self._policy.use_session_storage:
            return await self.delete_conversation_message(message_id, agent_id)

        row_id = coerce_persistent_message_id(message_id)
        if row_id is None:
            return False

        await self._check_write_permission("purge_conversation_message")
        purged = await self._storage.purge_message(row_id, reason=reason)

        if purged:
            await self._delete_pin_for_message(row_id, agent_id)

        return purged

    async def purge_conversation_session(
        self, session_id: str, agent_id: str, reason: str = "user-initiated"
    ) -> int:
        """Hard-delete every message in a session (#763).

        Wipes both live and soft-deleted rows. EPHEMERAL raises.
        ISOLATED falls through to the soft-delete equivalent.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot purge conversations in ephemeral mode (no persistent data)."
            )

        if self._policy.use_session_storage:
            return await self.delete_conversation_session(session_id, agent_id)

        await self._check_write_permission("purge_conversation_session")
        purged = await self._storage.purge_conversation_session(
            session_id, reason=reason
        )

        if purged:
            await self._delete_orphaned_pins(agent_id)

        return purged

    async def purge_trash_older_than(
        self,
        cutoff_iso: str,
        *,
        max_rows: int = 10_000,
        reason: str = "retention-janitor",
    ) -> int:
        """Retention-janitor primitive — wrapper delegator (#764).

        The privacy wrapper has to expose this method because the cron
        handler reads ``agent.storage.purge_trash_older_than`` and
        ``agent.storage`` is the wrapper, not the raw facade. Smoke
        testing caught the omission — the task skipped silently with
        "storage facade missing purge_trash_older_than" on every tick.

        No privacy gating needed: the rail only purges rows that were
        already soft-deleted (``deleted_at IS NOT NULL``). Live data is
        never touched. Even in EPHEMERAL mode, where the wrapper
        rejects new persistent writes, aging out already-trashed rows
        from a prior NORMAL stint is the right thing to do.
        """
        return await self._storage.purge_trash_older_than(
            cutoff_iso, max_rows=max_rows, reason=reason,
        )

    async def purge_decayed_episodes(
        self,
        *,
        delete_threshold: float,
        grace_days: int,
        max_rows: int = 10_000,
        half_life_days: int = 30,
        reason: str = "forgetting",
    ) -> int:
        """Forgetting primitive — wrapper delegator (#1674).

        Exposed on the wrapper because the consolidation pass reads
        ``agent.storage.purge_decayed_episodes`` and ``agent.storage`` is
        the wrapper, not the raw facade (same reason as
        ``purge_trash_older_than``). No extra privacy gating: the deletion tier
        only removes episodes whose importance-scaled decay has fallen below the
        operator-configured threshold and are past the grace window.
        """
        return await self._storage.purge_decayed_episodes(
            delete_threshold=delete_threshold,
            grace_days=grace_days,
            max_rows=max_rows,
            half_life_days=half_life_days,
            reason=reason,
        )

    async def delete_conversation_session(
        self, session_id: str, agent_id: str
    ) -> int:
        """
        Delete an entire conversation session by ID, respecting privacy mode.

        In EPHEMERAL mode, raises PrivacyViolationError (no persistent data).
        In ISOLATED mode, filters the in-memory session conversations.
        Otherwise delegates to the underlying storage which removes every
        message belonging to the session (metadata-based OR time-gap-based
        resolution — see AsyncConversationStore.delete_conversation_session).

        Returns the number of messages removed (0 when the session didn't
        exist or was already empty).
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot delete conversations in ephemeral mode (no persistent data)."
            )

        if self._policy.use_session_storage:
            # ISOLATED conversations live in an in-memory list. Scope the delete
            # to the requested session_id so removing one listed isolated
            # conversation never wipes the others (#2019). Use clear_session()
            # to drop the whole buffer.
            before = len(self._session_conversations)
            self._session_conversations = [
                conv
                for conv in self._session_conversations
                if not _in_session(conv, session_id)
            ]
            return before - len(self._session_conversations)

        await self._check_write_permission("delete_conversation_session")
        count = await self._storage.delete_conversation_session(session_id)

        # Sovereign override: clean up any memory pins that pointed at
        # messages we just soft-deleted. Subquery filters on
        # ``deleted_at IS NULL`` so pins on rows that just moved into
        # Trash are caught here — without that filter the subquery
        # would still find the trashed rows and the NOT IN would skip
        # them, leaving dangling pins (#763 regression).
        if count > 0:
            await self._delete_orphaned_pins(agent_id)

        return count

    async def _delete_orphaned_pins(self, agent_id: str) -> None:
        """Drop pins whose message is no longer live (deleted or purged).

        Tolerates a missing ``memory_pins`` table — the table is created
        by the memory_agency feature, which may not be loaded in slim
        startup paths or constrained tests. Production runs always have
        it; the guard exists so pin cleanup never blocks a legitimate
        delete in those edge cases.
        """
        try:
            await self._storage.db.execute_commit(
                "DELETE FROM memory_pins "
                "WHERE agent_id = ? AND message_id NOT IN "
                "(SELECT id FROM conversation_history "
                " WHERE agent_id = ? AND deleted_at IS NULL)",
                (agent_id, agent_id),
            )
        except Exception as e:
            logger.debug(
                "Pin cleanup skipped (memory_pins likely absent): %s", e
            )

    async def list_conversation_sessions(
        self, limit: int = 50, include_trashed: bool = False
    ) -> List[Dict[str, Any]]:
        """List session summaries for navigation, respecting privacy mode (#2019).

        EPHEMERAL exposes no persistent data, so returns ``[]``. ISOLATED
        summarizes the in-memory session buffer (which has no Trash
        distinction, so ``include_trashed`` yields ``[]``). Otherwise delegates
        to the persistent store.
        """
        if self._privacy_config.is_ephemeral():
            return []

        if self._policy.use_session_storage:
            if include_trashed:
                return []
            messages = []
            for i, conv in enumerate(self._session_conversations):
                meta = dict(conv.get("metadata") or {})
                # ISOLATED entries hold session_id at the top level; surface it
                # in metadata so the grouper labels each summary with the real
                # session_id — the same value delete_conversation_session scopes
                # on (#2019).
                # Always assign a resolvable id: the real session_id when
                # present, else the sentinel bucket — never the grouper's
                # synthetic index, which delete can't match (#2019).
                sid = _conv_session_id(conv)
                meta["session_id"] = sid if sid is not None else _ISOLATED_UNLABELED_SESSION_ID
                messages.append({
                    "id": i,
                    "role": conv.get("role"),
                    "content": conv.get("content"),
                    "metadata": meta,
                    "created_at": conv.get("created_at"),
                })
            return summarize_sessions(messages, limit=limit)

        return await self._storage.list_conversation_sessions(
            limit=limit, include_trashed=include_trashed
        )

    async def count_session_messages(
        self, session_id: str, deleted_filter: str = "all"
    ) -> int:
        """Count a session's messages, respecting privacy mode (#2019).

        EPHEMERAL has no persistent data → 0. ISOLATED counts matching in-memory
        rows (which have no Trash, so ``deleted_filter='deleted'`` → 0).
        Otherwise delegates to the resolver-based store count.
        """
        if self._privacy_config.is_ephemeral():
            return 0

        if self._policy.use_session_storage:
            if deleted_filter == "deleted":
                return 0
            return sum(
                1
                for conv in self._session_conversations
                if _in_session(conv, session_id)
            )

        return await self._storage.count_session_messages(
            session_id, deleted_filter=deleted_filter
        )

    async def message_belongs_to_session(
        self, message_id: Any, session_id: str
    ) -> bool:
        """Whether a message resolves within a session, respecting privacy (#2022).

        EPHEMERAL has no persistent data → False. ISOLATED checks the in-memory
        entry at ``message_id`` (its buffer index) against the session. Otherwise
        delegates to the resolver-based store check.
        """
        if self._privacy_config.is_ephemeral():
            return False

        if self._policy.use_session_storage:
            try:
                idx = int(message_id)
            except (TypeError, ValueError):
                return False
            if 0 <= idx < len(self._session_conversations):
                return _in_session(self._session_conversations[idx], session_id)
            return False

        return await self._storage.message_belongs_to_session(
            message_id, session_id
        )

    async def find_messages_matching(
        self, content_pattern: str, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find messages matching a pattern, respecting privacy mode (#2019).

        EPHEMERAL returns ``[]`` (no persistent data). ISOLATED searches the
        in-memory buffer. Otherwise delegates to the persistent store.
        """
        if self._privacy_config.is_ephemeral():
            return []

        if self._policy.use_session_storage:
            pattern_lower = content_pattern.lower()
            return [
                {
                    "id": i,
                    "role": conv.get("role"),
                    "content": conv.get("content"),
                    "metadata": conv.get("metadata") or {},
                }
                for i, conv in enumerate(self._session_conversations)
                if pattern_lower in (conv.get("content") or "").lower()
                and _in_session(conv, session_id)
            ]

        return await self._storage.find_messages_matching(
            content_pattern, session_id=session_id
        )

    async def delete_messages_matching(
        self, content_pattern: str, session_id: Optional[str] = None
    ) -> int:
        """Soft-delete messages matching a pattern, respecting privacy mode (#2019).

        EPHEMERAL raises (no persistent data). ISOLATED removes matches from the
        in-memory buffer. Otherwise delegates to the persistent store.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot delete conversations in ephemeral mode (no persistent data)."
            )

        if self._policy.use_session_storage:
            pattern_lower = content_pattern.lower()
            before = len(self._session_conversations)
            # Drop only messages that match the pattern AND (when a session is
            # named) belong to that session — never reach across isolated
            # conversations that happen to share text (#2019).
            self._session_conversations = [
                conv
                for conv in self._session_conversations
                if not (
                    pattern_lower in (conv.get("content") or "").lower()
                    and _in_session(conv, session_id)
                )
            ]
            return before - len(self._session_conversations)

        await self._check_write_permission("delete_messages_matching")
        return await self._storage.delete_messages_matching(
            content_pattern, session_id=session_id
        )

    async def list_trashed_conversations(
        self, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """List soft-deleted messages for the Trash UI (#763 / #765).

        Returns rows where ``deleted_at IS NOT NULL`` for this agent,
        sorted most-recently-trashed first. EPHEMERAL and ISOLATED modes
        return an empty list — neither has a persistent Trash store.

        Structural ``session_marker`` rows are excluded: lifecycle deletes now
        trash a session's marker alongside its content (#2027), but the marker
        is not a displayable message — surfacing it would show a blank system
        row and inflate the deleted-message total.
        """
        if self._privacy_config.is_ephemeral():
            return []
        if self._policy.use_session_storage:
            return []

        history = await self._storage.conversation.get_full_history_with_ids(
            include_excluded=True,
            include_stashed=True,
            only_deleted=True,
        )
        history = [
            m for m in history
            if (m.get("metadata") or {}).get("type") != "session_marker"
        ]
        history.sort(key=lambda m: m.get("deleted_at") or "", reverse=True)
        return history[:limit]

    async def list_archived_conversations(
        self, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """List archived messages for the Archive UI (#2149).

        Mirror image of ``list_trashed_conversations``: returns rows where
        ``archived_at IS NOT NULL`` (and not soft-deleted) for this agent,
        sorted most-recently-archived first. EPHEMERAL and ISOLATED modes
        return an empty list — neither has a persistent archive store.

        Structural ``session_marker`` rows are excluded, matching the Trash
        listing — the marker is not a displayable message.
        """
        if self._privacy_config.is_ephemeral():
            return []
        if self._policy.use_session_storage:
            return []

        history = await self._storage.conversation.get_full_history_with_ids(
            include_excluded=True,
            include_stashed=True,
            only_archived=True,
        )
        history = [
            m for m in history
            if (m.get("metadata") or {}).get("type") != "session_marker"
        ]
        history.sort(key=lambda m: m.get("archived_at") or "", reverse=True)
        return history[:limit]

    async def set_conversation_name(
        self, session_id: str, name: Optional[str]
    ) -> Optional[str]:
        """Upsert a user-chosen display name for a session (issue #716).

        EPHEMERAL raises (no durable data); ISOLATED has no persistent
        store so the wrapper echoes the normalized value without writing;
        NORMAL delegates to the conversation store.
        """
        if self._privacy_config.is_ephemeral():
            raise PrivacyViolationError(
                "Cannot rename conversations in ephemeral mode (no persistent data)."
            )
        if self._policy.use_session_storage:
            if name is None:
                return None
            trimmed = name.strip()
            return trimmed or None

        await self._check_write_permission("set_conversation_name")
        return await self._storage.set_conversation_name(session_id, name)

    async def get_conversation_name(self, session_id: str) -> Optional[str]:
        """Read the user-assigned display name for a session."""
        if self._privacy_config.is_ephemeral():
            return None
        if self._policy.use_session_storage:
            return None
        return await self._storage.get_conversation_name(session_id)

    async def get_conversation_names(self) -> Dict[str, str]:
        """Bulk read of user-assigned conversation names for this agent."""
        if self._privacy_config.is_ephemeral():
            return {}
        if self._policy.use_session_storage:
            return {}
        return await self._storage.get_conversation_names()

    # === Pass-through properties (with deprecation warnings) ===
    #
    # These properties expose the underlying storage objects directly,
    # which bypasses privacy mode enforcement. They are deprecated and
    # callers should migrate to the privacy-aware methods above.
    # They remain for backward compatibility with internal agent code.

    def _warn_direct_access(self, property_name: str) -> None:
        """Log a deprecation warning when a raw storage property is accessed."""
        warnings.warn(
            f"Direct access to PrivacyEnforcingStorage.{property_name} bypasses "
            f"privacy enforcement. Use privacy-aware methods instead "
            f"(e.g., query_conversations, get_conversation_history).",
            DeprecationWarning,
            stacklevel=3,
        )
        logger.warning(
            f"Privacy bypass: direct access to .{property_name} property "
            f"(current mode: {self._privacy_mode.value})"
        )

    @property
    def db(self):
        """Access to underlying database. DEPRECATED: bypasses privacy enforcement."""
        self._warn_direct_access("db")
        return self._storage.db

    @property
    def db_path(self) -> str:
        """Get the database file path from underlying storage."""
        return self._storage.db_path

    @property
    def llm_service(self):
        """Get the agent-scoped LLM service from underlying storage."""
        return getattr(self._storage, "llm_service", None)

    @property
    def graph_store(self):
        """Privacy-governing view of the graph store (#2672).

        Returns a proxy that applies the SAME volatile-mode graph-write policy
        as this wrapper's own :meth:`add_node` / :meth:`add_edge` /
        :meth:`compare_and_swap_node` — so feature code reaching through
        ``.graph_store`` can no longer bypass privacy enforcement (pre-#2672
        this returned the raw store). Reads, deletes, and ``bind_agent`` forward
        straight through; the raw ``db`` handle and any other un-allowlisted
        attribute fail closed so the proxy cannot be used to reach an ungoverned
        write path (#2672 review P1).
        """
        return _PrivacyGoverningGraphStore(self, self._storage.graph)

    @property
    def graph(self):
        """Privacy-governing view of the graph store (alias for graph_store)."""
        return _PrivacyGoverningGraphStore(self, self._storage.graph)

    @property
    def conversation(self):
        """Access to conversation store. DEPRECATED: bypasses privacy enforcement."""
        self._warn_direct_access("conversation")
        return self._storage.conversation

    @property
    def files(self):
        """Access to file store. DEPRECATED: bypasses privacy enforcement."""
        self._warn_direct_access("files")
        return self._storage.files

    @property
    def rag(self):
        """Access to RAG store."""
        return self._storage.rag
    
    async def close(self):
        """Close the underlying storage."""
        await self._storage.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()



class _PrivacyGoverningGraphStore:
    """Privacy-governing view over the underlying ``AsyncGraphStore`` (#2672).

    Returned by :attr:`PrivacyEnforcingStorage.graph` /
    :attr:`~PrivacyEnforcingStorage.graph_store` so that feature code reaching
    through those surfaces is subject to the SAME default-deny graph-write
    policy as the wrapper's own ``add_node`` / ``add_edge`` /
    ``compare_and_swap_node`` methods — closing the bypass where ``.graph``
    returned the raw store.

    The four write entry points are governed methods ON this proxy. Everything
    else is handled by :meth:`__getattr__`, which forwards ONLY a fixed allowlist
    of non-write surfaces (reads, deletes, ``bind_agent``, read-only metadata) and
    FAILS CLOSED on anything else. In particular it refuses the raw ``db`` handle
    (and any other connection/backend handle) so ``storage.graph.db.execute(...)``
    can no longer smuggle an ungoverned SQL write past the boundary, and a future
    write method added to ``AsyncGraphStore`` is refused by default rather than
    silently forwarded ungoverned (#2672 review P1).
    """

    __slots__ = ("_wrapper", "_store")

    #: Non-write attributes safe to forward to the wrapped ``AsyncGraphStore``.
    #: Reads, removals (a delete is not a durable user-content WRITE — it takes
    #: content away, which volatile modes never forbid), agent-scope binding, and
    #: read-only metadata. The four durable WRITE entry points are governed
    #: methods on this proxy and never reach ``__getattr__``. Any name NOT here —
    #: notably the raw ``db`` handle and any newly-added write method — fails
    #: closed (#2672 review P1). Extending this set REQUIRES confirming the target
    #: cannot perform a durable user-content graph write.
    _FORWARDED_ATTRS = frozenset({
        "get_node",
        "get_nodes_by_type",
        "query_nodes_by_type_and_property",
        "get_edges",
        "delete_node",
        "delete_edge",
        "purge_agent_nodes",
        "bind_agent",
        "agent_id",
    })

    def __init__(self, wrapper: "PrivacyEnforcingStorage", store) -> None:
        object.__setattr__(self, "_wrapper", wrapper)
        object.__setattr__(self, "_store", store)

    async def add_node(self, node, *, capability: Any = None) -> None:
        self._wrapper._assert_graph_node_write_allowed(
            node, "graph.add_node", capability=capability
        )
        await self._wrapper._enforce_free_text_carry_along(
            node, self._store, "graph.add_node"
        )
        return await self._store.add_node(node)

    async def compare_and_swap_node(
        self, node_id, expected, new_node, *, capability: Any = None
    ):
        # Govern the write intent without decomposing the atomic primitive: the
        # shared helper validates new_node's structural shape and pins the swap
        # to the stored node's type via the primitive's allowed_node_types
        # predicate, then delegates the single atomic CAS on THIS store.
        return await self._wrapper._governed_compare_and_swap(
            self._store, node_id, expected, new_node,
            "graph.compare_and_swap_node", capability=capability,
        )

    async def add_edge(self, source_id, target_id, label, properties=None,
                       *, capability: Any = None):
        self._wrapper._assert_graph_edge_write_allowed(
            label, "graph.add_edge", properties, capability=capability
        )
        return await self._store.add_edge(source_id, target_id, label, properties)

    async def add_trusted_cross_agent_edge(
        self, source_id, target_id, label, properties=None,
        *, capability: Any = None,
    ):
        self._wrapper._assert_graph_edge_write_allowed(
            label, "graph.add_trusted_cross_agent_edge", properties,
            capability=capability,
        )
        return await self._store.add_trusted_cross_agent_edge(
            source_id, target_id, label, properties
        )

    def __getattr__(self, name):
        # Reached for any attribute not defined on this proxy (i.e. anything but
        # the four governed writers). Forward ONLY the allowlisted non-write
        # surface; fail closed on everything else so a caller cannot reach the raw
        # ``db`` handle — or any un-vetted / future write method — and bypass the
        # volatile-mode graph-write policy through the ``.graph`` surface
        # (#2672 review P1).
        if name in _PrivacyGoverningGraphStore._FORWARDED_ATTRS:
            return getattr(self._store, name)
        if name.startswith("__") and name.endswith("__"):
            # Let normal Python attribute/dunder probing (copy, hasattr on
            # dunders, etc.) behave as "absent" rather than raising a privacy
            # error the interpreter would surface in confusing places.
            raise AttributeError(name)
        raise PrivacyViolationError(
            f"Graph proxy refuses to forward {name!r}: the privacy-governing "
            f"graph view exposes only its four governed writers (add_node / "
            f"add_edge / compare_and_swap_node / add_trusted_cross_agent_edge) "
            f"plus a fixed allowlist of reads/deletes/bind_agent. Raw handles "
            f"such as 'db' and any other attribute are refused so a caller cannot "
            f"bypass the volatile-mode graph-write policy through the '.graph' "
            f"surface. Use the raw store deliberately if an ungoverned write is "
            f"truly intended (#2672)."
        )


def wrap_storage_with_privacy(storage, privacy_mode: Union[PrivacyMode, PrivacyConfig, str]) -> PrivacyEnforcingStorage:
    """
    Factory function to wrap a Storage instance with privacy enforcement.
    
    Args:
        storage: The Storage instance to wrap
        privacy_mode: Initial privacy mode (PrivacyMode, PrivacyConfig, or preset name)
        
    Returns:
        PrivacyEnforcingStorage wrapper
    """
    return PrivacyEnforcingStorage(storage, privacy_mode)
