"""The generic wait poll loop and provider registry.

See :mod:`kestrel_sovereign.waits` for the why. This module has two
public surfaces:

``run_wait_loop`` is the engine. Give it a :class:`Waitable` provider, a
handle, and timing bounds; it polls until the provider reports a terminal
:class:`Outcome` or the timeout expires, then returns a canonical
:class:`ToolResult`. It holds no reference to the agent, so a feature's
own ``wait_for_task`` / ``talon_wait`` tools call it directly with a
provider they construct — which keeps them unit-testable without a live
agent.

``WaitRegistry`` is the per-agent dispatch table. Features register one
provider per handle kind in ``post_all_features_loaded``; the generic
``wait("talon:job_42")`` tool resolves the ``kind`` prefix here so it can
reach a provider owned by a *different* feature. The Wave-2 reconciler
cron will also enumerate the registry to drive the signal-resume path.
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
    """

    def __init__(self) -> None:
        self._providers: Dict[str, Waitable] = {}

    def register(self, provider: Waitable, *, replace: bool = False) -> None:
        """Register ``provider`` under its ``kind``.

        Raises ``ValueError`` on a malformed kind or a duplicate kind
        (unless ``replace=True``) — a silent overwrite would mask two
        features fighting over the same namespace.
        """
        kind = getattr(provider, "kind", None)
        if not kind or not isinstance(kind, str) or ":" in kind:
            raise ValueError(
                f"Waitable.kind must be a non-empty ':'-free string, got {kind!r}"
            )
        if kind in self._providers and not replace:
            raise ValueError(
                f"a Waitable provider for kind {kind!r} is already registered"
            )
        self._providers[kind] = provider
        logger.debug("registered wait provider kind=%s (%s)", kind, type(provider).__name__)

    def get(self, kind: str) -> Optional[Waitable]:
        return self._providers.get(kind)

    def kinds(self) -> List[str]:
        return sorted(self._providers)

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
        provider = self._providers.get(kind)
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
