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
profile that loads Talon but not Tasks would have registered talon
waitables but no ``wait`` tool to reach them. Being mandatory, ``wait``
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

    @tool(
        name="wait",
        description=(
            "The ONE generic wait — works across EVERY feature. There is no "
            "per-feature wait tool; whatever async work a loaded feature "
            "exposes, you wait on it here with `target=\"<kind>:<handle>\"`.\n"
            "Known handle kinds (each contributed by a feature; more may be "
            "registered by whatever features are loaded):\n"
            "• `task:<task_id>` — a Kestrel background task\n"
            "• `talon:<job_id>` — a Talon coding job\n"
            "• `ci:<...>`, `lora_train:<...>`, `tx:<...>`, `workflow:<run_id>` "
            "and others when those features are present.\n"
            "If you pass an unknown kind, the error lists the kinds currently "
            "registered.\n"
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
                ``"talon:job_42"``). When set, ``duration_seconds`` is
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
        # `!wait talon:job_42` is a handle wait. parse_command_args binds
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
