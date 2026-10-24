"""The event-loop stall watchdog names the frame that blocks the loop (#3806)."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import threading
import time
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI

from kestrel_sovereign import paths, server
from kestrel_sovereign.event_loop_watchdog import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_STALL_THRESHOLD_SECONDS,
    MINIMUM_HEARTBEAT_INTERVAL_SECONDS,
    THREAD_NAME,
    EventLoopStallWatchdog,
    EventLoopWatchdogConfig,
    event_loop_stall_watchdog,
    resolve_event_loop_watchdog_config,
)

WATCHDOG_LOGGER = "kestrel_sovereign.event_loop_watchdog"
RESUMED = re.compile(
    r"Event loop resumed: its heartbeat ran (\d+\.\d+) s late, "
    r"(\d+\.\d+) s after the previous one"
)
RESUMED_AT_STOP = re.compile(
    r"Event loop resumed: the watchdog was stopped (\d+\.\d+) s after its "
    r"heartbeat was due, (\d+\.\d+) s after the previous one ran"
)


def _block_the_loop_for_one_and_a_half_seconds() -> None:
    time.sleep(1.5)


def _block_the_loop(seconds: float) -> None:
    time.sleep(seconds)


def _hold_the_gil(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def _warnings(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == WATCHDOG_LOGGER and record.levelno >= logging.WARNING
    ]


def _resumes(caplog) -> list[tuple[float, float]]:
    found = []
    for record in caplog.records:
        if record.name != WATCHDOG_LOGGER:
            continue
        match = RESUMED.search(record.getMessage())
        if match:
            assert record.levelno == logging.INFO
            found.append((float(match.group(1)), float(match.group(2))))
    return found


def _watchdog_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == THREAD_NAME]


@pytest.fixture
def watchdog_log(caplog):
    caplog.set_level(logging.INFO, logger=WATCHDOG_LOGGER)
    return caplog


async def test_a_blocked_loop_logs_one_warning_naming_the_sleeping_frame(watchdog_log):
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=DEFAULT_STALL_THRESHOLD_SECONDS,
        heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    )
    watchdog.start()
    try:
        await asyncio.sleep(0.6)
        _block_the_loop_for_one_and_a_half_seconds()
        await asyncio.sleep(0.6)
    finally:
        watchdog.stop()

    warnings = _warnings(watchdog_log)
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    message = warnings[0].getMessage()
    assert "Event loop stalled: its heartbeat is" in message
    assert "in _block_the_loop_for_one_and_a_half_seconds" in message
    assert "time.sleep(1.5)" in message

    # The loop was blocked for at least how late the heartbeat ran and at most
    # how long it went without one; the two differ by one heartbeat interval.
    [(late, gap)] = _resumes(watchdog_log)
    assert 1.5 - DEFAULT_HEARTBEAT_INTERVAL_SECONDS - 0.01 <= late <= 1.75
    assert 1.49 <= gap <= 2.0


async def test_a_healthy_loop_logs_no_stall(watchdog_log):
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=DEFAULT_STALL_THRESHOLD_SECONDS,
        heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    )
    watchdog.start()
    try:
        for _ in range(30):
            await asyncio.sleep(0.1)
    finally:
        watchdog.stop()

    assert _warnings(watchdog_log) == []
    assert _resumes(watchdog_log) == []


async def test_a_block_shorter_than_the_threshold_is_not_a_stall(watchdog_log):
    # The loop is blocked 96% of the time, so the watchdog thread's looks
    # (about one per threshold) land inside a block, where the heartbeat is
    # overdue but not by more than the threshold.
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=0.8, heartbeat_interval_seconds=0.05
    )
    watchdog.start()
    try:
        for _ in range(6):
            _block_the_loop(0.5)
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
    finally:
        watchdog.stop()

    assert _warnings(watchdog_log) == []
    assert _resumes(watchdog_log) == []


async def test_a_long_stall_gets_one_stack(watchdog_log):
    # The watchdog thread looks again every heartbeat interval while the
    # loop is stalled; it must not log the stack again on each look.
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=0.3, heartbeat_interval_seconds=0.1
    )
    watchdog.start()
    try:
        await asyncio.sleep(0.3)
        _block_the_loop(1.5)
        await asyncio.sleep(0.4)
    finally:
        watchdog.stop()

    assert len(_warnings(watchdog_log)) == 1
    [(late, gap)] = _resumes(watchdog_log)
    assert late >= 1.5 - 0.1 - 0.01
    assert gap >= 1.49


async def test_each_stall_gets_its_own_stack_and_resume_line(watchdog_log):
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=0.3, heartbeat_interval_seconds=0.1
    )
    watchdog.start()
    try:
        await asyncio.sleep(0.3)
        _block_the_loop(0.7)
        await asyncio.sleep(0.4)
        _block_the_loop(0.7)
        await asyncio.sleep(0.4)
    finally:
        watchdog.stop()

    warnings = _warnings(watchdog_log)
    assert len(warnings) == 2
    assert all("in _block_the_loop" in record.getMessage() for record in warnings)
    assert len(_resumes(watchdog_log)) == 2


async def test_a_stall_holding_the_gil_is_reported_with_the_watchdogs_own_delay(
    watchdog_log,
):
    # Native code that never releases the GIL keeps the watchdog thread out
    # until the stall is over. Raising the switch interval makes pure Python
    # do the same, so the thread can only look once the loop is idle again.
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=0.3, heartbeat_interval_seconds=0.1
    )
    switch_interval = sys.getswitchinterval()
    watchdog.start()
    try:
        await asyncio.sleep(0.3)
        sys.setswitchinterval(5.0)
        try:
            _hold_the_gil(1.0)
        finally:
            sys.setswitchinterval(switch_interval)
        await asyncio.sleep(0.4)
    finally:
        watchdog.stop()

    warnings = _warnings(watchdog_log)
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    message = warnings[0].getMessage()
    assert "The watchdog thread itself woke" in message
    assert "held the GIL" in message


async def test_stopping_during_a_reported_stall_still_logs_its_end(watchdog_log):
    # Teardown can block the loop and unwind straight into stop(), so no
    # heartbeat runs after the stall the watchdog reported.
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=0.3, heartbeat_interval_seconds=0.1
    )
    watchdog.start()
    await asyncio.sleep(0.2)
    _block_the_loop(0.8)
    watchdog.stop()

    assert len(_warnings(watchdog_log)) == 1
    assert _resumes(watchdog_log) == []
    ends = [
        RESUMED_AT_STOP.search(record.getMessage())
        for record in watchdog_log.records
        if record.name == WATCHDOG_LOGGER and record.levelno == logging.INFO
    ]
    [end] = [match for match in ends if match]
    late, gap = float(end.group(1)), float(end.group(2))
    assert late >= 0.8 - 0.1 - 0.01
    assert gap >= 0.79


async def test_a_thread_that_cannot_start_leaves_no_heartbeat(monkeypatch):
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=1.0, heartbeat_interval_seconds=0.05
    )
    loop = asyncio.get_running_loop()
    scheduled = []
    real_call_later = loop.call_later

    def recording_call_later(delay, callback, *args, **kwargs):
        if getattr(callback, "__self__", None) is watchdog:
            scheduled.append(callback)
        return real_call_later(delay, callback, *args, **kwargs)

    real_thread_start = threading.Thread.start

    def refusing_start(thread):
        if thread.name == THREAD_NAME:
            raise RuntimeError("can't start new thread")
        return real_thread_start(thread)

    monkeypatch.setattr(loop, "call_later", recording_call_later)
    monkeypatch.setattr(threading.Thread, "start", refusing_start)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        watchdog.start()
    assert scheduled == []

    monkeypatch.setattr(threading.Thread, "start", real_thread_start)
    watchdog.start()
    try:
        assert watchdog.running
        assert len(scheduled) == 1
    finally:
        watchdog.stop()


async def test_stop_joins_the_thread_and_cancels_the_heartbeat():
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=1.0, heartbeat_interval_seconds=0.05
    )
    watchdog.start()
    assert watchdog.running
    assert len(_watchdog_threads()) == 1
    watchdog.stop()
    assert not watchdog.running
    assert _watchdog_threads() == []
    # Idempotent.
    watchdog.stop()


async def test_start_twice_is_refused():
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=1.0, heartbeat_interval_seconds=0.25
    )
    watchdog.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            watchdog.start()
    finally:
        watchdog.stop()


def test_start_needs_a_running_loop():
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=1.0, heartbeat_interval_seconds=0.25
    )
    with pytest.raises(RuntimeError):
        watchdog.start()
    assert _watchdog_threads() == []


async def test_stop_off_the_loop_thread_is_refused():
    watchdog = EventLoopStallWatchdog(
        stall_threshold_seconds=1.0, heartbeat_interval_seconds=0.25
    )
    watchdog.start()
    try:
        with pytest.raises(RuntimeError, match="loop's thread"):
            await asyncio.to_thread(watchdog.stop)
        assert watchdog.running
    finally:
        watchdog.stop()


def test_constructor_refuses_a_heartbeat_the_loop_would_spin_on():
    with pytest.raises(ValueError, match="at least"):
        EventLoopStallWatchdog(
            stall_threshold_seconds=1.0,
            heartbeat_interval_seconds=MINIMUM_HEARTBEAT_INTERVAL_SECONDS / 2,
        )
    EventLoopStallWatchdog(
        stall_threshold_seconds=0.005,
        heartbeat_interval_seconds=MINIMUM_HEARTBEAT_INTERVAL_SECONDS,
    )


@pytest.mark.parametrize("value", [0, -1.0, float("inf"), float("nan")])
def test_constructor_refuses_a_non_positive_duration(value):
    with pytest.raises(ValueError, match="stall_threshold_seconds"):
        EventLoopStallWatchdog(
            stall_threshold_seconds=value, heartbeat_interval_seconds=0.25
        )
    with pytest.raises(ValueError, match="heartbeat_interval_seconds"):
        EventLoopStallWatchdog(
            stall_threshold_seconds=1.0, heartbeat_interval_seconds=value
        )


# -- configuration ------------------------------------------------------------


def test_an_absent_section_gives_the_documented_defaults():
    config = EventLoopWatchdogConfig.from_section({})
    assert config == EventLoopWatchdogConfig(
        enabled=True,
        stall_threshold_seconds=1.0,
        heartbeat_interval_seconds=0.25,
    )


def test_the_section_sets_every_value():
    config = EventLoopWatchdogConfig.from_section(
        {
            "enabled": False,
            "stall_threshold_seconds": 2,
            "heartbeat_interval_seconds": 0.5,
        }
    )
    assert config == EventLoopWatchdogConfig(
        enabled=False,
        stall_threshold_seconds=2.0,
        heartbeat_interval_seconds=0.5,
    )


@pytest.mark.parametrize("key", ["stall_threshold_seconds", "heartbeat_interval_seconds"])
@pytest.mark.parametrize(
    "value", [0, -0.5, True, "1s", float("inf"), float("nan"), None, [1]]
)
def test_a_malformed_duration_is_refused_by_name(key, value):
    with pytest.raises(ValueError, match=rf"\[event_loop_watchdog\]\.{key}"):
        EventLoopWatchdogConfig.from_section({key: value})


def test_a_heartbeat_under_the_minimum_is_refused():
    with pytest.raises(
        ValueError, match=r"\[event_loop_watchdog\]\.heartbeat_interval_seconds"
    ):
        EventLoopWatchdogConfig.from_section({"heartbeat_interval_seconds": 0.005})
    config = EventLoopWatchdogConfig.from_section(
        {"stall_threshold_seconds": 0.005, "heartbeat_interval_seconds": 0.01}
    )
    assert config.stall_threshold_seconds == 0.005
    assert config.heartbeat_interval_seconds == 0.01


@pytest.mark.parametrize("value", ["true", 1, None])
def test_a_non_boolean_enabled_is_refused(value):
    with pytest.raises(ValueError, match=r"\[event_loop_watchdog\]\.enabled"):
        EventLoopWatchdogConfig.from_section({"enabled": value})


def test_an_unknown_key_is_refused_by_name():
    with pytest.raises(ValueError, match="stall_threshold"):
        EventLoopWatchdogConfig.from_section({"stall_threshold": 2.0})


def _write_project_kestrel_toml(text: str) -> None:
    home = paths.project_dir()
    home.mkdir(parents=True, exist_ok=True)
    home.joinpath("kestrel.toml").write_text(text, encoding="utf-8")


def test_the_section_is_read_from_kestrel_toml():
    _write_project_kestrel_toml(
        "[event_loop_watchdog]\n"
        "stall_threshold_seconds = 3.5\n"
        "heartbeat_interval_seconds = 0.1\n"
    )
    assert resolve_event_loop_watchdog_config() == EventLoopWatchdogConfig(
        enabled=True,
        stall_threshold_seconds=3.5,
        heartbeat_interval_seconds=0.1,
    )


# -- lifecycle ---------------------------------------------------------------


async def test_the_context_runs_the_watchdog_for_its_body():
    config = EventLoopWatchdogConfig(heartbeat_interval_seconds=0.05)
    with event_loop_stall_watchdog(config) as watchdog:
        assert watchdog is not None and watchdog.running
        assert len(_watchdog_threads()) == 1
    assert not watchdog.running
    assert _watchdog_threads() == []


async def test_a_disabled_watchdog_starts_nothing():
    with event_loop_stall_watchdog(EventLoopWatchdogConfig(enabled=False)) as watchdog:
        assert watchdog is None
        assert _watchdog_threads() == []


async def test_the_context_stops_the_watchdog_when_its_body_raises():
    config = EventLoopWatchdogConfig(heartbeat_interval_seconds=0.05)
    with pytest.raises(LookupError), event_loop_stall_watchdog(config):
        raise LookupError("body failed")
    assert _watchdog_threads() == []


@pytest.fixture
def stub_server_lifespan_phases(monkeypatch):
    """Replace the server's teardown owner and startup with recording no-ops."""

    entered = []

    @asynccontextmanager
    async def teardown_owner(_app):
        entered.append("teardown-owner")
        yield

    @asynccontextmanager
    async def startup(_app):
        entered.append("startup")
        yield

    monkeypatch.setattr(server, "_lifespan_teardown_owner", teardown_owner)
    monkeypatch.setattr(server, "_lifespan_startup", startup)
    return entered


async def test_the_server_lifespan_runs_the_watchdog_around_startup_and_teardown(
    stub_server_lifespan_phases,
):
    async with server.lifespan(FastAPI()):
        assert stub_server_lifespan_phases == ["teardown-owner", "startup"]
        [thread] = _watchdog_threads()
        assert thread.is_alive()
    assert _watchdog_threads() == []


async def test_a_malformed_section_refuses_server_startup(stub_server_lifespan_phases):
    _write_project_kestrel_toml("[event_loop_watchdog]\nstall_threshold_seconds = 0\n")
    with pytest.raises(ValueError, match="stall_threshold_seconds"):
        async with server.lifespan(FastAPI()):
            pytest.fail("the server started with a malformed watchdog section")
    assert stub_server_lifespan_phases == []
    assert _watchdog_threads() == []
