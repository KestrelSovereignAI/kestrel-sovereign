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

from kestrel_sdk.tools import Outcome, ToolResult, WaitStatus, Waitable

logger = logging.getLogger(__name__)

# Held-turn ceiling for a handle wait. A handle wait polls (cheap) rather
# than sleeping idle, so it inherits the larger of the legacy caps
# (talon_wait's 3600s) rather than the dumb-sleep cap. Waits that need to
# run longer than this should use the signal-resume path (Wave 2), not a
# held turn.
MAX_HANDLE_WAIT_SECONDS = 3600
DEFAULT_POLL_INTERVAL_SECONDS = 5


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
    last: Optional[WaitStatus] = None
    while True:
        try:
            status = await provider.poll(handle)
        except Exception as exc:  # provider bug or transport failure
            logger.exception("wait provider %s.poll(%r) raised", provider.kind, handle)
            return ToolResult.failed(
                f"wait on {label} failed: {exc}",
                data={"ref": label, "waited_seconds": int(time.monotonic() - start)},
            )
        last = status
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
