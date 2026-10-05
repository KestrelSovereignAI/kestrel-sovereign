"""The generic wait poll loop and provider registry.

See :mod:`kestrel_sovereign.waits` for the why. This module has two
public surfaces:

``run_wait_loop`` is the engine. Give it a :class:`Waitable` provider, a
handle, and timing bounds; it polls until the provider reports a terminal
:class:`Outcome` or the timeout expires, then returns a canonical
:class:`ToolResult`. It holds no reference to the agent, so it is callable
directly (and unit-testable) with a provider constructed in isolation; in
production the :class:`WaitRegistry` calls it.

``WaitRegistry`` is the per-agent dispatch table behind the SINGLE generic
``wait`` tool. There are no per-feature wait tools — each feature registers
one provider per handle kind in ``post_all_features_loaded``, and
``wait("<kind>:<handle>")`` resolves the ``kind`` prefix here to reach the
owning feature's provider. The reconciler cron also enumerates the registry
to drive the signal-resume path.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

from kestrel_sdk.tools import Outcome, ToolResult, Waitable

logger = logging.getLogger(__name__)

# Held-turn ceiling for a handle wait. A handle wait polls (cheap) rather
# than sleeping idle, so it inherits the larger of the legacy caps
# (talon_wait's 3600s) rather than the dumb-sleep cap. Waits that need to
# run longer than this must not hold a turn at all: they park and resume on
# the handle's durable wake via
# ``kestrel_sovereign.waits.reconciler.register_wait_resume_consumer``.
# Raising this ceiling only moves the cliff (#3295).
MAX_HANDLE_WAIT_SECONDS = 3600
DEFAULT_POLL_INTERVAL_SECONDS = 5

# The ``WaitStatus.data`` key through which a provider names the terminal
# EVENT a terminal poll describes (#3399). The reconciler dedups wakes on it
# when present, so a provider that sets it gets one wake per distinct terminal
# event rather than one per outcome class: a CI re-run that fails again is a
# new event even though its outcome is the same ``failed``.
#
# The value is a non-empty string, and it must be:
#
# * stable — the same event polled again, at any later time, yields the same
#   string. Anything that varies between polls of one event (a fetch time, an
#   ``updated_at`` touched by an unrelated edit) re-fires the wake.
# * read from the provider's raw record, never from its classification. The
#   reconciler compares identities without the outcome, so a classifier change
#   that re-labels an old event does not make it news and replay it (#3390).
# * distinct for distinct events, including ones with the same outcome — a
#   re-run, a new head commit, a corrected terminal record.
# * the same through every read path the provider has. A provider that can
#   read one event through several paths that name its records differently
#   puts only what every path names alike here (for CI, the head SHA), and
#   the rest in ``TERMINAL_EVENT_DETAIL_KEY``.
# * written by the provider itself. A provider that spreads third-party data
#   into ``WaitStatus.data`` must drop this key, and the four below, from
#   that data, or the third party decides when the agent is woken.
#
# A provider that does not set it keeps the legacy ``"<outcome>"`` /
# ``"<outcome>:<native status>"`` token unchanged.
TERMINAL_EVENT_KEY = "terminal_event"

# Optional, and always set as a pair: the part of the event's identity that
# only one read path can see (the DETAIL — for CI, the execution set), and
# that path's name (the VIEW). GitHub CI is the case (#3399): the Checks API
# names a re-run by its new check-run ids, the Actions API fallback by its
# workflow run's ``run_attempt``, and nothing maps one onto the other. A watch
# fires on a new execution, never on a change of view, so with the event
# unchanged:
#
# * same view: a new event if and only if the details differ.
# * different view: NOT a new event, whatever the outcome — one execution can
#   be DONE through one view and PARTIAL through a narrower one. The
#   reconciler does not wake; it re-baselines the delivered token, and any
#   watch armed over it, to the new view's identity and logs the switch, so a
#   genuine re-run seen later through that view still fires.
#
# Accepted limit: a new execution whose first terminal read is also the read
# where the view switches is absorbed into that re-baseline. A view switch is
# rare (for CI, a credential gaining or losing the Checks API), and the next
# distinct execution still fires.
#
# A provider whose read path never changes does not need these keys: put the
# whole identity in ``TERMINAL_EVENT_KEY``.
TERMINAL_EVENT_DETAIL_KEY = "terminal_event_detail"
TERMINAL_EVENT_VIEW_KEY = "terminal_event_view"

# Optional: ``True`` when the handle can produce no later terminal event — a
# merged PR, a finished local task, an A2A task whose terminal state this
# agent has stamped. A re-registered watch waits for an event other than the one
# already delivered, so over a final event it can never fire; the reconciler
# disarms it when it polls that event again rather than polling it every tick
# forever. Leave it unset for anything that can still change (a closed PR can
# be reopened). Like the keys above, only the provider may write it.
TERMINAL_EVENT_FINAL_KEY = "terminal_event_final"

# Optional: when the terminal event happened, read from the provider's raw
# record (a job's exit time, not the time it was polled or classified), as an
# ISO 8601 string (a ``datetime`` is accepted too, and re-written as one). A
# value without a timezone is read as UTC.
#
# The reconciler compares it with the handle's last delivered wake (#3390).
# A wake whose event is older than that delivery is a REPLAY: the event
# happened before the agent was last woken for this handle, so whatever the
# provider's own wake would tell the agent to do in that turn may already be
# done or superseded. The reconciler announces a replay on the generic
# ``wait.replay`` source, which states that it is a replay and carries none of
# the provider's act-now instructions, instead of on the provider's signal.
# A provider that does not set it never has a wake labelled a replay. Like the
# keys above, only the provider may write it: a forged earlier time would
# strip a fresh wake of its instructions.
TERMINAL_EVENT_AT_KEY = "terminal_event_at"


def parse_ref(ref: str) -> Tuple[str, str]:
    """Split a ``"<kind>:<handle>"`` wait reference into its parts.

    The handle may itself contain ``:`` (e.g. a URL); only the first
    colon is the separator. Raises ``ValueError`` on a malformed ref.
    """
    if not isinstance(ref, str) or ":" not in ref:
        raise ValueError(
            f"wait reference must be '<kind>:<handle>', got {ref!r}"
        )
    kind, handle = ref.split(":", 1)
    kind = kind.strip()
    handle = handle.strip()
    if not kind or not handle:
        raise ValueError(
            f"wait reference must be '<kind>:<handle>' with both parts "
            f"non-empty, got {ref!r}"
        )
    return kind, handle


async def run_wait_loop(
    provider: Waitable,
    handle: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    max_seconds: int = MAX_HANDLE_WAIT_SECONDS,
    label: Optional[str] = None,
) -> ToolResult:
    """Poll ``provider`` for ``handle`` until terminal or timeout.

    Args:
        provider: A :class:`Waitable`; ``provider.poll(handle)`` is called
            once per iteration and must return a :class:`WaitStatus`.
        handle: The handle to poll (the part after the ``kind:`` prefix).
        timeout_seconds: Maximum seconds to wait before returning a
            still-pending PARTIAL. Capped at ``max_seconds``.
        poll_interval_seconds: Seconds to sleep between polls (> 0).
        max_seconds: The held-turn ceiling enforced on ``timeout_seconds``.
        label: Display label for messages; defaults to ``"<kind>:<handle>"``.

    Returns:
        ``ToolResult.ok`` on a terminal DONE, ``.failed`` on terminal
        FAILED, ``.partial`` on a terminal PARTIAL or on timeout. The
        ``data`` payload always carries ``waited_seconds`` and ``ref``;
        the provider's own ``WaitStatus.data`` is merged underneath.
    """
    label = label or f"{provider.kind}:{handle}"

    try:
        timeout_val = int(timeout_seconds)
        poll_val = int(poll_interval_seconds)
    except (TypeError, ValueError):
        return ToolResult.failed(
            "timeout_seconds and poll_interval_seconds must be integers, "
            f"got {timeout_seconds!r}, {poll_interval_seconds!r}"
        )
    if timeout_val < 0 or poll_val <= 0:
        return ToolResult.failed(
            "timeout_seconds must be >= 0 and poll_interval_seconds must be > 0"
        )
    if timeout_val > max_seconds:
        return ToolResult.failed(
            f"timeout_seconds {timeout_val} exceeds the maximum "
            f"{max_seconds}s for a held wait; rely on the signal-resume "
            f"path to wake the agent instead",
            data={
                "ref": label,
                "requested_seconds": timeout_val,
                "max_seconds": max_seconds,
            },
        )

    start = time.monotonic()
    while True:
        try:
            status = await provider.poll(handle)
        except Exception as exc:  # provider bug or transport failure
            logger.exception("wait provider %s.poll(%r) raised", provider.kind, handle)
            return ToolResult.failed(
                f"wait on {label} failed: {exc}",
                data={"ref": label, "waited_seconds": int(time.monotonic() - start)},
            )
        elapsed = int(time.monotonic() - start)

        if status.outcome.is_terminal():
            data = dict(status.data or {})
            data.update({"ref": label, "waited_seconds": elapsed, "timed_out": False})
            if status.outcome is Outcome.DONE:
                return ToolResult.ok(confirmation=status.summary, data=data)
            if status.outcome is Outcome.FAILED:
                return ToolResult.failed(status.summary, data=data)
            # PARTIAL: a mixed terminal state. Surface the summary as both
            # halves so the honesty layer sees the caveat (a richer split
            # can ride on data["caveat"] when a provider needs it).
            return ToolResult.partial(
                confirmation=status.summary,
                error=str(data.get("caveat") or status.summary),
                data=data,
            )

        if elapsed >= timeout_val:
            data = dict(status.data or {})
            data.update({
                "ref": label,
                "waited_seconds": elapsed,
                "timeout_seconds": timeout_val,
                "timed_out": True,
            })
            return ToolResult.partial(
                confirmation=f"{label} still pending after {elapsed}s ({status.summary})",
                error=f"Timeout after {timeout_val}s; {label} not terminal",
                data=data,
            )

        await asyncio.sleep(poll_val)


class WaitRegistry:
    """Per-agent dispatch table of :class:`Waitable` providers.

    Lives at ``agent.wait_registry`` (mirrors ``agent.signal_registry``).
    Features register one provider per handle kind; the generic ``wait``
    tool and the Wave-2 reconciler resolve kinds here.

    Ownership per kind is a **stack**, not a single slot plus a saved
    "previous" provider (#2522 P3 redesign). Each ``register`` pushes; the
    effective provider for a kind is the top of its stack. A displaced
    predecessor sits *beneath* the provider that displaced it and becomes
    effective again only when every provider above it is torn down. This is
    what makes teardown restore the nearest **still-live** predecessor: a
    provider removed from the *middle* of the stack (e.g. a soft-disabled
    feature that a newer provider already superseded) is gone for good and can
    never be resurrected by a later teardown. The old single-slot design saved
    each owner's ``previous`` at registration time, so a three-deep chain
    ``host → A → B`` would, on ``disable A`` then ``disable B``, restore B's
    saved ``previous`` (== A) and resurrect the disabled A — the exact bug this
    redesign fixes.
    """

    def __init__(self) -> None:
        # kind -> ownership stack, oldest (host) first, current (effective) last.
        self._stacks: Dict[str, List[Waitable]] = {}

    @staticmethod
    def _validate_kind(provider: Waitable) -> str:
        kind = getattr(provider, "kind", None)
        if not kind or not isinstance(kind, str) or ":" in kind:
            raise ValueError(
                f"Waitable.kind must be a non-empty ':'-free string, got {kind!r}"
            )
        return kind

    def register(self, provider: Waitable, *, replace: bool = False) -> None:
        """Push ``provider`` onto its ``kind``'s ownership stack.

        Raises ``ValueError`` on a malformed kind or a duplicate kind
        (unless ``replace=True``) — a silent overwrite would mask two features
        fighting over the same namespace. Re-registering the SAME provider
        object is idempotent: it is moved to the top of the stack rather than
        stacked as a duplicate, so a feature whose ``initialize()`` /
        ``post_all_features_loaded`` re-runs in one live cycle never buries a
        second copy of itself.
        """
        kind = self._validate_kind(provider)
        stack = self._stacks.setdefault(kind, [])

        # Idempotent re-registration: drop any existing occurrence of THIS exact
        # provider so it ends up on top exactly once (no duplicate stack entry).
        existing = [i for i, p in enumerate(stack) if p is provider]
        for index in reversed(existing):
            del stack[index]

        if stack and not replace and not existing:
            raise ValueError(
                f"a Waitable provider for kind {kind!r} is already registered"
            )
        stack.append(provider)
        logger.debug(
            "registered wait provider kind=%s (%s), depth=%d",
            kind, type(provider).__name__, len(stack),
        )

    def unregister(self, kind: str) -> bool:
        """Pop the CURRENT (top) provider off ``kind``'s stack. Returns True if present.

        The deliberate inverse of a bare :meth:`register`. Feature teardown /
        boot rollback should instead use :meth:`deregister` (identity-aware) so
        a feature only ever removes *its own* provider; this bare pop is kept
        for callers that just want to drop the current provider for a kind.
        Idempotent: popping an absent/empty kind is a benign ``False``.
        """
        stack = self._stacks.get(kind)
        if not stack:
            return False
        stack.pop()
        if not stack:
            self._stacks.pop(kind, None)
        return True

    def deregister(self, kind: str, provider: Waitable) -> bool:
        """Remove ``provider`` from ``kind``'s stack by object identity, wherever
        it sits, and let the nearest still-live predecessor become effective
        (#2522 P3).

        This is the identity-aware teardown primitive feature shutdown / boot
        rollback use. Removing the CURRENT (top) provider restores whatever is
        beneath it — the nearest predecessor that is still on the stack, never a
        provider some earlier teardown already removed. Removing a MIDDLE
        provider (one a newer owner already superseded) simply drops it without
        disturbing the current owner. Identity is checked with ``is``: two
        distinct providers of the same kind are different owners.

        Returns True iff ``provider`` was the current (top) provider before
        removal — i.e. its removal changed which provider is effective.
        """
        stack = self._stacks.get(kind)
        if not stack:
            return False
        was_current = stack[-1] is provider
        removed = False
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is provider:
                del stack[index]
                removed = True
                break
        if not stack:
            self._stacks.pop(kind, None)
        return removed and was_current

    def restore_if_current(
        self,
        kind: str,
        expected: Waitable,
        previous: Optional[Waitable] = None,
    ) -> bool:
        """Back-compat teardown shim over :meth:`deregister` (#2522 P3).

        Removes ``expected`` from ``kind``'s stack so the nearest still-live
        predecessor becomes effective. ``previous`` is retained only for call
        compatibility and is IGNORED: the per-kind stack is now the single
        source of truth for what gets restored, so teardown can never resurrect
        a disabled predecessor by trusting a stale saved value. Returns True
        when ``expected`` was the current provider (its removal restored a
        predecessor).
        """
        return self.deregister(kind, expected)

    def get(self, kind: str) -> Optional[Waitable]:
        stack = self._stacks.get(kind)
        return stack[-1] if stack else None

    def contains(self, kind: str, provider: Waitable) -> bool:
        """Whether this exact provider is registered anywhere in ``kind``.

        Lifecycle teardown uses this identity-aware query before removing a
        provider that may sit beneath a newer, explicitly replacing owner.
        """
        return any(item is provider for item in self._stacks.get(kind, ()))

    def kinds(self) -> List[str]:
        return sorted(kind for kind, stack in self._stacks.items() if stack)

    async def wait(
        self,
        ref: str,
        *,
        timeout_seconds: int,
        poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
        max_seconds: int = MAX_HANDLE_WAIT_SECONDS,
    ) -> ToolResult:
        """Resolve a ``"<kind>:<handle>"`` ref and run the poll loop."""
        try:
            kind, handle = parse_ref(ref)
        except ValueError as exc:
            return ToolResult.failed(str(exc))
        provider = self.get(kind)
        if provider is None:
            known = ", ".join(self.kinds()) or "(none registered)"
            return ToolResult.failed(
                f"no wait provider for kind {kind!r}; known kinds: {known}",
                data={"ref": ref, "known_kinds": self.kinds()},
            )
        return await run_wait_loop(
            provider,
            handle,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            max_seconds=max_seconds,
            label=ref,
        )
