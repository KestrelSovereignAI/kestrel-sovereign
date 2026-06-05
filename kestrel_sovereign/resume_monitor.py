"""Host sleep/wake (suspend/resume) detection for Kestrel Sovereign (#1545).

A long-running agent process is frozen wholesale when the host machine
sleeps: every ``asyncio.sleep`` loop simply pauses and resumes on wake,
unaware that wall-clock time jumped. The scheduler then fires stale jobs
late, the heartbeat silently drops missed beats, and the dispatcher's
coalescing (wall-clock) and rate-limit (monotonic) windows disagree about
how much time passed. Nothing notices the gap.

This module is the one primitive that *notices*. It compares wall-clock
elapsed against monotonic elapsed across a periodic tick:

    gap = (wall_now - wall_prev) - (mono_now - mono_prev)

``time.monotonic()`` excludes system-suspend time on both macOS and Linux
(``CLOCK_MONOTONIC``), while ``time.time()`` is corrected forward on wake.
So a large positive divergence is exactly the suspend duration — detected
without any OS-specific power API, identically on macOS/Linux/Windows.

When the gap exceeds a threshold the monitor fires a single ``on_resume``
callback carrying the measured gap. The wiring in ``KestrelAgent`` turns
that into one audited ``system.resumed`` signal through the existing
SignalDispatcher (see ``signals/sources/system_resumed.py``), whose ACTION
handler re-anchors the dispatcher's throttling windows. The scheduler and
heartbeat detect staleness locally on their own ticks (so they self-heal
even if the signal is never delivered); this monitor is the observable
spine and the dispatcher's re-anchor trigger.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from kestrel_sovereign.config import load_section, parse_duration

logger = logging.getLogger(__name__)

# Defaults: poll every 30s (cheap — two clock reads and a subtraction),
# treat a divergence over 120s as a genuine suspend. 120s is comfortably
# above any scheduling jitter or GC/event-loop stall yet small enough to
# catch a brief lid-close, and it never false-positives on normal operation
# because two awake ticks diverge by milliseconds.
DEFAULT_TICK_SECONDS = 30.0
DEFAULT_THRESHOLD_SECONDS = 120.0


def suspend_gap_seconds(
    prev_wall: float,
    now_wall: float,
    prev_mono: float,
    now_mono: float,
) -> float:
    """Return the approximate host-suspend duration between two samples.

    ``wall_delta`` includes time the host was asleep; ``mono_delta`` does
    not. Their difference is the suspend gap. Floored at 0 so a backward
    wall-clock correction (NTP step) can never report a negative gap.
    """
    wall_delta = now_wall - prev_wall
    mono_delta = now_mono - prev_mono
    return max(0.0, wall_delta - mono_delta)


@dataclass
class ResumeMonitorConfig:
    """Configuration for the resume monitor, from ``[resume_monitor]`` in
    kestrel.toml. All fields optional; defaults are sensible for a laptop."""

    enabled: bool = True
    tick_seconds: float = DEFAULT_TICK_SECONDS
    threshold_seconds: float = DEFAULT_THRESHOLD_SECONDS

    @classmethod
    def from_config(cls) -> "ResumeMonitorConfig":
        cfg = load_section("resume_monitor")
        if not cfg:
            return cls()

        def _duration(key: str, default: float) -> float:
            raw = cfg.get(key)
            if raw is None:
                return default
            if isinstance(raw, (int, float)):
                return float(raw)
            try:
                return float(parse_duration(str(raw)))
            except ValueError:
                logger.warning(
                    "Invalid resume_monitor.%s=%r, using %s", key, raw, default
                )
                return default

        return cls(
            enabled=bool(cfg.get("enabled", True)),
            tick_seconds=_duration("tick", DEFAULT_TICK_SECONDS),
            threshold_seconds=_duration("threshold", DEFAULT_THRESHOLD_SECONDS),
        )


# Signature: async def on_resume(gap_seconds: float) -> None
ResumeCallback = Callable[[float], Awaitable[None]]


class ResumeMonitor:
    """Background loop that detects a host suspend and fires ``on_resume``.

    The loop itself relies on the very behaviour it detects: ``asyncio.sleep``
    is frozen during suspend, so on wake the elapsed-clock comparison reveals
    the gap. Clocks are injectable for deterministic testing.
    """

    def __init__(
        self,
        *,
        on_resume: ResumeCallback,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
        threshold_seconds: float = DEFAULT_THRESHOLD_SECONDS,
        wall_clock: Callable[[], float] = time.time,
        mono_clock: Callable[[], float] = time.monotonic,
        name: str = "resume-monitor",
    ) -> None:
        self._on_resume = on_resume
        self._tick_seconds = max(1.0, float(tick_seconds))
        self._threshold_seconds = max(1.0, float(threshold_seconds))
        self._wall_clock = wall_clock
        self._mono_clock = mono_clock
        self._name = name
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._prev_wall: Optional[float] = None
        self._prev_mono: Optional[float] = None

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def _baseline(self) -> None:
        self._prev_wall = self._wall_clock()
        self._prev_mono = self._mono_clock()

    async def poll_once(self) -> float:
        """Sample the clocks once, compare against the previous sample, and
        fire ``on_resume`` if the suspend gap exceeds the threshold.

        Returns the measured gap (0.0 on the first call or when no suspend is
        detected). Separated from the loop so tests can drive it with injected
        clocks and no real sleeping.
        """
        wall = self._wall_clock()
        mono = self._mono_clock()
        if self._prev_wall is None or self._prev_mono is None:
            self._prev_wall, self._prev_mono = wall, mono
            return 0.0

        gap = suspend_gap_seconds(self._prev_wall, wall, self._prev_mono, mono)
        self._prev_wall, self._prev_mono = wall, mono

        if gap >= self._threshold_seconds:
            logger.info(
                "Host suspend detected: ~%.0fs wall-clock gap; firing resume",
                gap,
            )
            try:
                await self._on_resume(gap)
            except Exception:
                # The monitor must never die because a consumer raised —
                # the next suspend still needs to be caught.
                logger.exception("ResumeMonitor on_resume callback failed")
            return gap
        return 0.0

    async def _loop(self) -> None:
        self._baseline()
        while self._running:
            try:
                await asyncio.sleep(self._tick_seconds)
            except asyncio.CancelledError:
                break
            if not self._running:
                break
            await self.poll_once()

    async def start(self) -> None:
        if self._running:
            logger.warning("ResumeMonitor already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name=self._name)
        logger.info(
            "ResumeMonitor started (tick=%.0fs, threshold=%.0fs)",
            self._tick_seconds,
            self._threshold_seconds,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
