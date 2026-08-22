"""WaitFeature — the single generic ``wait`` tool.

The wait *engine* and provider *registry* are core infrastructure
(:mod:`kestrel_sovereign.waits`); individual features register a
:class:`~kestrel_sdk.tools.Waitable` provider per handle kind. This
feature exposes the one agent-facing surface over that machinery: the
generic ``wait`` tool.

It lives in its own MANDATORY feature (not on TaskFeature) on purpose:
``wait`` is generic — it dispatches ``"<kind>:<handle>"`` to whatever
provider a loaded feature registered, and also serves the plain bounded
sleep. Tying it to an optional feature (e.g. tasks) would mean an agent
profile that loads another async-work feature but not Tasks could have
registered provider-owned waitables but no ``wait`` tool to reach them. Being mandatory, ``wait``
is always present wherever any waitable is.
"""

from __future__ import annotations

import asyncio
import logging
import time

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.waits.reconciler import register_wait_watch
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)


class WaitFeature(Feature):
    """Provides the single generic ``wait`` tool over the wait registry."""

    # Conservative ceiling on a single bounded (no-target) sleep. A pause
    # longer than this should be a scheduled/cron resume, not a held turn.
    _MAX_WAIT_SECONDS = 1800

    # Fallback reconcile driver cadence (#2729). The scheduler's
    # ``wait_reconcile`` cron is the PRIMARY driver at 60s; this loop only
    # drives a tick when that cron is NOT keeping the reconciler fresh (an
    # agent profile without SchedulerFeature). WaitFeature is mandatory, so
    # this makes wait reconciliation a core runtime service that is available
    # wherever a `mode="signal"` watch can be armed — otherwise a durable
    # signal watch registered in a scheduler-less profile would never wake.
    _FALLBACK_RECONCILE_POLL_SECONDS = 60
    # Drive only when the last tick is older than this. Held above the cron's
    # 60s cadence so a live scheduler keeps the reconciler fresh and this loop
    # stays dormant (no double-driving in the common case).
    _FALLBACK_RECONCILE_STALE_SECONDS = 90

    def __init__(self, agent=None):
        if agent is not None:
            super().__init__(agent)
        else:
            # Standalone mode for external integration / tests.
            self.agent = None
            self.name = self.__class__.__name__

    @property
    def tool_description(self) -> str:
        return (
            "The one generic wait — block on any feature's async work via "
            "wait('<kind>:<handle>'), or pause for a bounded duration"
        )

    @property
    def promote_tools_on_startup(self) -> bool:
        return True

    async def initialize(self):
        self.enabled = True

    async def post_all_features_loaded(self, agent):
        """Start the fallback wait-reconcile driver (#2729).

        The wait reconciler (which turns durable ``mode="signal"`` watches into
        ``wait.complete`` cognition wakes) has historically been driven ONLY by
        the ``wait_reconcile`` cron seeded by :class:`SchedulerFeature`. That
        feature is optional, so a valid minimal profile (Peers + Wait, no
        Scheduler) could accept a signal watch that then NEVER wakes. Since
        WaitFeature is mandatory, owning the fallback driver here makes
        reconciliation a core runtime service present wherever a signal watch
        can be armed. The loop stands down while the scheduler cron keeps the
        reconciler fresh (see :meth:`_fallback_reconcile_due`), so an agent that
        has a scheduler is not double-driven.
        """
        if agent is None or getattr(agent, "wait_registry", None) is None:
            # Standalone / no wait engine — nothing to reconcile.
            return
        self._track_owned_background_task(
            self._fallback_reconcile_loop(agent),
            name="wait_fallback_reconcile",
        )

    def _fallback_reconcile_due(self, reconciler) -> bool:
        """Whether the fallback loop should drive a reconcile tick now.

        ``True`` when no reconciler has run yet (or the singleton is not built),
        or the last tick is older than :attr:`_FALLBACK_RECONCILE_STALE_SECONDS`.
        A scheduler cron running every 60s keeps ``seconds_since_last_reconcile``
        below the threshold, so this returns ``False`` and the fallback stays
        dormant — the two drivers never overlap in steady state.
        """
        if reconciler is None:
            return True
        since = reconciler.seconds_since_last_reconcile()
        return since is None or since >= self._FALLBACK_RECONCILE_STALE_SECONDS

    async def _fallback_reconcile_loop(self, agent):
        """Periodically drive the wait reconciler when nothing else does.

        Feature-owned (cancelled on shutdown / disable via
        :meth:`Feature.shutdown`). Each iteration is gated by
        :meth:`_fallback_reconcile_due` so it defers to the scheduler cron when
        present; the reconciler's own lock makes an accidental overlap safe.
        """
        from kestrel_sovereign.waits.reconciler import run_wait_reconcile

        while True:
            try:
                await asyncio.sleep(self._FALLBACK_RECONCILE_POLL_SECONDS)
                reconciler = getattr(agent, "_wait_reconciler", None)
                if self._fallback_reconcile_due(reconciler):
                    await run_wait_reconcile(agent)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - loop must never die
                logger.warning(
                    "wait fallback reconcile tick failed: %s", exc,
                    exc_info=True,
                )

    @tool(
        name="wait",
        description=(
            "The ONE generic wait — works across EVERY feature. There is no "
            "per-feature wait tool; whatever async work a loaded feature "
            "exposes, you wait on it here with `target=\"<kind>:<handle>\"`.\n"
            "Known handle kinds (each contributed by a feature; more may be "
            "registered by whatever features are loaded):\n"
            "• `task:<task_id>` — a LOCAL Kestrel background task (this "
            "agent's own store)\n"
            "• `a2a:<task_id>` — an OUTBOUND A2A TASK you sent a peer via "
            "send_a2a_task (route it here, NOT `task:` — a `task:` on an "
            "outbound A2A id is a provider mismatch and is rejected at "
            "registration). A2A QUESTIONS are NOT watched here: "
            "send_a2a_question already wakes you via its own "
            "`a2a.question_answered` signal, so an `a2a:<question-id>` watch is "
            "rejected to avoid waking you twice for one answer.\n"
            "• `ci:<owner/repo#N>` — a GitHub PR's merge/CI-check state\n"
            "• `lora_train:<...>`, `tx:<...>`, `workflow:<run_id>` and others "
            "when those features are present.\n"
            "A kind being LISTED here is documentation, not a guarantee it is "
            "AVAILABLE: a provider is only reachable when its feature is "
            "loaded. If you pass an unknown/unavailable kind, the error lists "
            "the kinds currently registered. A registered kind's signal-mode "
            "watch is durable and RE-ARMS across restart; availability "
            "(is the provider loaded?) and re-arming (does a live watch "
            "resume?) are separate — a documented kind whose feature is not "
            "loaded neither registers nor re-arms.\n"
            "\n"
            "Three ways to call it:\n"
            "• `target=\"<kind>:<handle>\"` (default `mode=\"block\"`) — hold "
            "the turn, polling until that thing reaches a terminal state or "
            "the timeout expires; returns the terminal outcome (or a still-"
            "pending result on timeout).\n"
            "• `target=\"<kind>:<handle>\", mode=\"signal\"` — register a watch "
            "and return IMMEDIATELY; the wait reconciler wakes you with a "
            "`wait.complete` cognition signal once it finishes. Use this for "
            "long/unattended waits so you don't hold a turn.\n"
            "• `duration_seconds=N` (no target) — a plain bounded pause, the "
            "native alternative to shelling out to `sleep` between polls in an "
            "autonomous loop."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!wait",
    )
    async def wait(
        self,
        target: str = "",
        duration_seconds: int = 0,
        timeout_seconds: int = 600,
        poll_interval_seconds: int = 5,
        reason: str = "",
        mode: str = "block",
    ) -> ToolResult:
        """
        Block on a handle, or pause for a bounded duration.

        Args:
            target: ``"<kind>:<handle>"`` to wait on (e.g.
                ``"workflow:run_42"``). When set, ``duration_seconds`` is
                ignored and the wait is driven by the registered provider.
            duration_seconds: Seconds to pause when no ``target`` is given
                (0 to the enforced maximum).
            timeout_seconds: Max seconds to block on a ``target`` before
                returning a still-pending result.
            poll_interval_seconds: Seconds between polls of a ``target``.
            reason: Optional human-readable note (recorded in the result).
            mode: ``"block"`` (default) holds the turn until the target is
                terminal; ``"signal"`` registers a watch and returns
                immediately, waking the agent via a ``wait.complete`` signal
                when the target finishes (requires a ``target``).
        """
        mode = str(mode).strip().lower() if mode else "block"
        if mode not in ("block", "signal"):
            return ToolResult.failed(
                f"mode must be 'block' or 'signal', got {mode!r}"
            )

        # The first positional accepts BOTH forms so the interface stays
        # one tool: `!wait 5` (bare number) is a bounded sleep, while
        # `!wait workflow:run_42` is a handle wait. parse_command_args binds
        # positional CLI tokens in signature order, so a numeric target is
        # the legacy `!wait <seconds>` command — route it to the pause.
        target = str(target).strip() if target else ""
        if target and target.lstrip("-").isdigit():
            duration_seconds = int(target)
            target = ""

        if mode == "signal" and not target:
            return ToolResult.failed(
                "mode='signal' requires a target handle (e.g. "
                "'task:<id>'); a bare duration sleep cannot be signalled"
            )

        if target:
            if mode == "signal":
                # Register a watch and return immediately — the reconciler
                # wakes the agent with a wait.complete signal on completion.
                try:
                    await register_wait_watch(self.agent, target)
                except ValueError as exc:
                    return ToolResult.failed(str(exc))
                return ToolResult.ok(
                    confirmation=(
                        f"Watching {target}; will wake on completion via "
                        f"wait.complete"
                    ),
                    data={"ref": target, "mode": "signal", "watching": True},
                )

            registry = getattr(self.agent, "wait_registry", None) if self.agent else None
            if registry is None:
                return ToolResult.failed(
                    "wait engine unavailable: no wait_registry on the agent"
                )
            return await registry.wait(
                target,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )

        # No target: bounded idle pause (native alternative to shelling
        # out to `sleep` in an autonomous work loop).
        try:
            duration = int(duration_seconds)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"duration_seconds must be an integer, got {duration_seconds!r}"
            )
        if duration < 0:
            return ToolResult.failed(
                f"duration_seconds must be >= 0, got {duration}"
            )
        if duration > self._MAX_WAIT_SECONDS:
            return ToolResult.failed(
                f"duration_seconds {duration} exceeds the maximum "
                f"{self._MAX_WAIT_SECONDS}s for a single wait; schedule a "
                f"resume instead of holding the turn",
                data={
                    "requested_seconds": duration,
                    "max_seconds": self._MAX_WAIT_SECONDS,
                },
            )

        start = time.monotonic()
        await asyncio.sleep(duration)
        elapsed = round(time.monotonic() - start, 3)

        confirmation = f"Waited {elapsed}s"
        if reason:
            confirmation += f" ({reason})"
        return ToolResult.ok(
            confirmation=confirmation,
            data={
                "requested_seconds": duration,
                "elapsed_seconds": elapsed,
                "reason": reason,
            },
        )
