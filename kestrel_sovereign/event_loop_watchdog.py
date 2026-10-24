"""Event-loop stall watchdog: name the frame that blocks the host's loop (#3806).

Every agent turn, the Distributed Stop owner's lease renewal and the scheduler
share one asyncio event loop. A synchronous section that runs too long on it
delays everything else, and the visible symptom is usually far from the cause:
the Stop owner is fenced because its 2.0 s lease renewal ran late (#3745), and
nothing in the log says what held the loop.

The watchdog records that evidence while the stall is happening. The loop runs
a heartbeat callback every ``heartbeat_interval_seconds``. A daemon thread,
which keeps running while the loop thread is blocked, watches the heartbeat.
Once the next heartbeat is overdue by more than ``stall_threshold_seconds`` the
thread logs one WARNING with how long the loop has stalled so far and the loop
thread's current Python stack, which names the blocking frame (when many short
callbacks add up instead, it names whichever one is running). When the loop
runs again it logs one INFO line with the stall's total. A stall gets one
stack, however long it lasts.

Both lines give two measured numbers: how late the heartbeat is (or ran), and
how long since the previous heartbeat ran. The loop was blocked for at least
the first and at most the second; they differ by one heartbeat interval.

The thread can only sample the stack while it can run. Native code that holds
the GIL through a whole stall keeps it out until the stall is over; the stall
is then reported from the late heartbeat alone, without a stack.

Configuration is the ``[event_loop_watchdog]`` section of ``kestrel.toml``.
Every key is optional and defaults to the constants below. An unknown key or an
invalid value refuses host startup rather than running a watchdog nobody
configured; a ``kestrel.toml`` that does not parse at all is skipped with a
warning, as it is for every section.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

from kestrel_sovereign.config import load_section

logger = logging.getLogger(__name__)

EVENT_LOOP_WATCHDOG_SECTION = "event_loop_watchdog"
DEFAULT_ENABLED = True
DEFAULT_STALL_THRESHOLD_SECONDS = 1.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 0.25
# A shorter heartbeat would make the loop spin on it.
MINIMUM_HEARTBEAT_INTERVAL_SECONDS = 0.01
THREAD_NAME = "kestrel-event-loop-watchdog"

# The watchdog thread sleeps until the moment the current heartbeat would
# cross the threshold. This floor only keeps a wake landing exactly on that
# boundary from becoming a zero-length wait.
_MINIMUM_WAIT_SECONDS = 0.001


def _seconds(
    section: Mapping[str, Any], key: str, default: float, minimum: float = 0.0
) -> float:
    if key not in section:
        return default
    raw = section[key]
    bound = f"at least {minimum}" if minimum else "a positive number of"
    # bool is an int subclass; ``true`` is not a number of seconds.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"[{EVENT_LOOP_WATCHDOG_SECTION}].{key} must be {bound} seconds, "
            f"got {raw!r}"
        )
    value = float(raw)
    if not math.isfinite(value) or value <= 0 or value < minimum:
        raise ValueError(
            f"[{EVENT_LOOP_WATCHDOG_SECTION}].{key} must be {bound} seconds, "
            f"got {raw!r}"
        )
    return value


@dataclass(frozen=True)
class EventLoopWatchdogConfig:
    """The ``[event_loop_watchdog]`` section of ``kestrel.toml``."""

    enabled: bool = DEFAULT_ENABLED
    stall_threshold_seconds: float = DEFAULT_STALL_THRESHOLD_SECONDS
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS

    @classmethod
    def from_section(cls, section: Mapping[str, Any]) -> "EventLoopWatchdogConfig":
        """Validate a section; raise ``ValueError`` naming the bad key."""

        known = {"enabled", "stall_threshold_seconds", "heartbeat_interval_seconds"}
        unknown = sorted(set(section) - known)
        if unknown:
            raise ValueError(
                f"[{EVENT_LOOP_WATCHDOG_SECTION}] has unknown key(s): "
                + ", ".join(unknown)
                + "; expected "
                + ", ".join(sorted(known))
            )
        enabled = section.get("enabled", DEFAULT_ENABLED)
        if not isinstance(enabled, bool):
            raise ValueError(
                f"[{EVENT_LOOP_WATCHDOG_SECTION}].enabled must be true or false, "
                f"got {enabled!r}"
            )
        return cls(
            enabled=enabled,
            stall_threshold_seconds=_seconds(
                section, "stall_threshold_seconds", DEFAULT_STALL_THRESHOLD_SECONDS
            ),
            heartbeat_interval_seconds=_seconds(
                section,
                "heartbeat_interval_seconds",
                DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
                MINIMUM_HEARTBEAT_INTERVAL_SECONDS,
            ),
        )


def resolve_event_loop_watchdog_config() -> EventLoopWatchdogConfig:
    """Read and validate ``[event_loop_watchdog]`` from ``kestrel.toml``."""

    return EventLoopWatchdogConfig.from_section(
        load_section(EVENT_LOOP_WATCHDOG_SECTION)
    )


@dataclass(frozen=True)
class _Stall:
    """A stall the watchdog thread has reported and is waiting to see end."""

    beats: int


class EventLoopStallWatchdog:
    """Log the loop thread's stack when the event loop stops running.

    ``start`` and ``stop`` must be called on the thread that runs the loop.
    """

    def __init__(
        self,
        *,
        stall_threshold_seconds: float,
        heartbeat_interval_seconds: float,
    ) -> None:
        if not (math.isfinite(stall_threshold_seconds) and stall_threshold_seconds > 0):
            raise ValueError(
                "stall_threshold_seconds must be a positive number of seconds"
            )
        if not (
            math.isfinite(heartbeat_interval_seconds)
            and heartbeat_interval_seconds >= MINIMUM_HEARTBEAT_INTERVAL_SECONDS
        ):
            raise ValueError(
                "heartbeat_interval_seconds must be at least "
                f"{MINIMUM_HEARTBEAT_INTERVAL_SECONDS} seconds"
            )
        self._threshold = float(stall_threshold_seconds)
        self._interval = float(heartbeat_interval_seconds)
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread_id: Optional[int] = None
        self._timer: Optional[asyncio.TimerHandle] = None
        # Shared between the loop's heartbeat and the watchdog thread.
        self._lock = threading.Lock()
        self._due = 0.0
        self._last_beat = 0.0
        self._beats = 0
        self._unseen_lateness = 0.0
        self._unseen_gap = 0.0
        # Owned by the watchdog thread.
        self._stall: Optional[_Stall] = None

    @property
    def running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stopped.is_set()
        )

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._thread is not None:
            raise RuntimeError("event-loop stall watchdog was already started")
        self._loop = loop
        self._loop_thread_id = threading.get_ident()
        now = time.monotonic()
        with self._lock:
            self._last_beat = now
            self._due = now + self._interval
        thread = threading.Thread(target=self._watch, name=THREAD_NAME, daemon=True)
        # The thread first: if it cannot start, no heartbeat is left
        # rescheduling itself on the loop, and start() can be tried again.
        thread.start()
        self._thread = thread
        self._timer = loop.call_later(self._interval, self._beat)
        logger.info(
            "Event-loop stall watchdog started: heartbeat every %.2f s, "
            "stall threshold %.2f s",
            self._interval,
            self._threshold,
        )

    def stop(self) -> None:
        if self._thread is None or self._stopped.is_set():
            return
        if threading.get_ident() != self._loop_thread_id:
            raise RuntimeError(
                "event-loop stall watchdog must be stopped on its loop's thread"
            )
        self._stopped.set()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._thread.join()
        if self._stall is not None:
            self._resumed_at_stop()

    def _resumed_at_stop(self) -> None:
        """Close a reported stall whose end the watchdog thread did not see."""

        with self._lock:
            now = time.monotonic()
            due = self._due
            last_beat = self._last_beat
            beats = self._beats
            lateness = self._unseen_lateness
            gap = self._unseen_gap
        stall, self._stall = self._stall, None
        if beats != stall.beats:
            self._log_resumed(lateness, gap)
            return
        # No heartbeat ran after the stall: the loop is running again, but
        # only to stop the watchdog.
        logger.info(
            "Event loop resumed: the watchdog was stopped %.2f s after its "
            "heartbeat was due, %.2f s after the previous one ran",
            now - due,
            now - last_beat,
        )

    def _log_resumed(self, lateness: float, gap: float) -> None:
        logger.info(
            "Event loop resumed: its heartbeat ran %.2f s late, %.2f s after "
            "the previous one",
            lateness,
            gap,
        )

    # -- loop thread -------------------------------------------------------

    def _beat(self) -> None:
        if self._stopped.is_set():
            return
        now = time.monotonic()
        with self._lock:
            lateness = now - self._due
            if lateness > self._unseen_lateness:
                self._unseen_lateness = lateness
                self._unseen_gap = now - self._last_beat
            self._beats += 1
            self._last_beat = now
            self._due = now + self._interval
        self._timer = self._loop.call_later(self._interval, self._beat)

    # -- watchdog thread ---------------------------------------------------

    def _watch(self) -> None:
        timeout = self._interval + self._threshold
        while True:
            expected_wake = time.monotonic() + timeout
            if self._stopped.wait(timeout):
                return
            timeout = self._check(expected_wake)

    def _check(self, expected_wake: float) -> float:
        """Report what the heartbeat shows; return how long to sleep next."""

        with self._lock:
            now = time.monotonic()
            due = self._due
            last_beat = self._last_beat
            beats = self._beats
            lateness = self._unseen_lateness
            gap = self._unseen_gap
            self._unseen_lateness = 0.0
            self._unseen_gap = 0.0
        watcher_delay = now - expected_wake

        if self._stall is not None:
            if beats == self._stall.beats:
                return self._interval
            self._stall = None
            self._log_resumed(lateness, gap)
        elif lateness > self._threshold:
            # The stall began and ended between two looks by this thread.
            logger.warning(
                "Event loop stalled: its heartbeat ran %.2f s late, %.2f s after "
                "the previous one (threshold %.2f s). It resumed before the "
                "watchdog thread could sample its stack.%s",
                lateness,
                gap,
                self._threshold,
                self._watcher_delay_note(watcher_delay),
            )

        overdue = now - due
        if overdue > self._threshold:
            self._stall = _Stall(beats=beats)
            self._report_stall(
                overdue=overdue,
                since_last_beat=now - last_beat,
                watcher_delay=watcher_delay,
            )
            return self._interval
        return max(due + self._threshold - now, _MINIMUM_WAIT_SECONDS)

    def _watcher_delay_note(self, watcher_delay: float) -> str:
        if watcher_delay <= self._interval:
            return ""
        return (
            f" The watchdog thread itself woke {watcher_delay:.2f} s late, "
            "usually because something held the GIL (native code, a long "
            "garbage collection) or the process was starved of CPU."
        )

    def _report_stall(
        self, *, overdue: float, since_last_beat: float, watcher_delay: float
    ) -> None:
        frames = sys._current_frames()
        frame = frames.get(self._loop_thread_id)
        del frames
        if frame is None:
            stack = "  (the loop thread has no Python frame)\n"
        else:
            stack = "".join(traceback.format_stack(frame))
        del frame
        delay_note = self._watcher_delay_note(watcher_delay)
        if delay_note:
            delay_note += (
                " This stack was sampled once it let the watchdog run, so it may "
                "show the frame just past the blocking call."
            )
        logger.warning(
            "Event loop stalled: its heartbeat is %.2f s overdue, %.2f s since "
            "the last one ran (threshold %.2f s, heartbeat every %.2f s).%s "
            "Loop thread stack, most recent call last:\n%s",
            overdue,
            since_last_beat,
            self._threshold,
            self._interval,
            delay_note,
            stack.rstrip("\n"),
        )


@contextmanager
def event_loop_stall_watchdog(
    config: Optional[EventLoopWatchdogConfig] = None,
) -> Iterator[Optional[EventLoopStallWatchdog]]:
    """Run the configured watchdog on the current loop for the ``with`` body.

    Without ``config`` the ``[event_loop_watchdog]`` section is read, and a
    malformed one raises before anything starts.
    """

    if config is None:
        config = resolve_event_loop_watchdog_config()
    if not config.enabled:
        logger.info(
            "Event-loop stall watchdog disabled ([%s].enabled = false)",
            EVENT_LOOP_WATCHDOG_SECTION,
        )
        yield None
        return
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=config.stall_threshold_seconds,
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
    )
    watchdog.start()
    try:
        yield watchdog
    finally:
        watchdog.stop()


__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_STALL_THRESHOLD_SECONDS",
    "EVENT_LOOP_WATCHDOG_SECTION",
    "MINIMUM_HEARTBEAT_INTERVAL_SECONDS",
    "EventLoopStallWatchdog",
    "EventLoopWatchdogConfig",
    "event_loop_stall_watchdog",
    "resolve_event_loop_watchdog_config",
]
