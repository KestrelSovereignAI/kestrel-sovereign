"""Tests for the generic wait engine (kestrel_sovereign.waits.engine).

The engine owns the poll loop, the cap, the interval, and the
Outcome->ToolResult mapping; providers only classify a single poll.
These pin that contract independent of any feature.
"""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sdk.tools import Outcome, ToolResultStatus, WaitStatus
from kestrel_sovereign.waits import WaitRegistry, run_wait_loop
from kestrel_sovereign.waits.engine import parse_ref


class _ScriptedProvider:
    """Returns a queued sequence of WaitStatus, one per poll."""

    kind = "demo"
    signal = None

    def __init__(self, sequence):
        self._seq = list(sequence)
        self.calls = 0

    async def poll(self, handle):
        self.calls += 1
        # Hold on the last entry once exhausted (simulates "still pending").
        idx = min(self.calls - 1, len(self._seq) - 1)
        return self._seq[idx]


class _BoomProvider:
    kind = "boom"
    signal = None

    async def poll(self, handle):
        raise RuntimeError("transport exploded")


# ---------------------------------------------------------------------------
# parse_ref
# ---------------------------------------------------------------------------


class TestParseRef:
    def test_basic(self):
        assert parse_ref("talon:job_42") == ("talon", "job_42")

    def test_handle_may_contain_colons(self):
        assert parse_ref("ci:https://x/y") == ("ci", "https://x/y")

    @pytest.mark.parametrize("bad", ["", "nohandle", ":h", "k:", "  :  "])
    def test_malformed_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_ref(bad)


# ---------------------------------------------------------------------------
# run_wait_loop — terminal mapping
# ---------------------------------------------------------------------------


class TestRunWaitLoop:
    @pytest.mark.asyncio
    async def test_done_maps_to_ok(self):
        p = _ScriptedProvider([WaitStatus(Outcome.DONE, "all good", data={"rc": 0})])
        r = await run_wait_loop(p, "h", timeout_seconds=10)
        assert r.status is ToolResultStatus.OK
        assert r.confirmation == "all good"
        assert r.data["rc"] == 0
        assert r.data["ref"] == "demo:h"
        assert "waited_seconds" in r.data

    @pytest.mark.asyncio
    async def test_failed_maps_to_error(self):
        p = _ScriptedProvider([WaitStatus(Outcome.FAILED, "it broke")])
        r = await run_wait_loop(p, "h", timeout_seconds=10)
        assert r.status is ToolResultStatus.ERROR
        assert r.error == "it broke"

    @pytest.mark.asyncio
    async def test_partial_terminal_surfaces_both_halves(self):
        p = _ScriptedProvider([
            WaitStatus(Outcome.PARTIAL, "done but degraded", data={"caveat": "no index"})
        ])
        r = await run_wait_loop(p, "h", timeout_seconds=10)
        assert r.status is ToolResultStatus.PARTIAL
        assert r.confirmation == "done but degraded"
        assert r.error == "no index"

    @pytest.mark.asyncio
    async def test_polls_until_terminal(self):
        p = _ScriptedProvider([
            WaitStatus(Outcome.PENDING, "running"),
            WaitStatus(Outcome.PENDING, "running"),
            WaitStatus(Outcome.DONE, "finished"),
        ])
        r = await run_wait_loop(p, "h", timeout_seconds=10, poll_interval_seconds=1)
        assert r.status is ToolResultStatus.OK
        assert p.calls == 3

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_pending(self):
        p = _ScriptedProvider([WaitStatus(Outcome.PENDING, "still running")])
        r = await run_wait_loop(p, "h", timeout_seconds=0, poll_interval_seconds=1)
        assert r.status is ToolResultStatus.PARTIAL
        assert r.data["timed_out"] is True
        assert "Timeout" in r.error

    @pytest.mark.asyncio
    async def test_provider_exception_maps_to_error(self):
        r = await run_wait_loop(_BoomProvider(), "h", timeout_seconds=10)
        assert r.status is ToolResultStatus.ERROR
        assert "transport exploded" in r.error

    @pytest.mark.asyncio
    async def test_timeout_over_cap_rejected(self):
        p = _ScriptedProvider([WaitStatus(Outcome.PENDING, "x")])
        r = await run_wait_loop(p, "h", timeout_seconds=99999, max_seconds=60)
        assert r.status is ToolResultStatus.ERROR
        assert "exceeds the maximum" in r.error

    @pytest.mark.asyncio
    async def test_bad_interval_rejected(self):
        p = _ScriptedProvider([WaitStatus(Outcome.PENDING, "x")])
        r = await run_wait_loop(p, "h", timeout_seconds=10, poll_interval_seconds=0)
        assert r.status is ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# WaitRegistry — dispatch
# ---------------------------------------------------------------------------


class TestWaitRegistry:
    def test_register_and_lookup(self):
        reg = WaitRegistry()
        p = _ScriptedProvider([WaitStatus(Outcome.DONE, "x")])
        reg.register(p)
        assert reg.get("demo") is p
        assert reg.kinds() == ["demo"]

    def test_duplicate_kind_rejected(self):
        reg = WaitRegistry()
        reg.register(_ScriptedProvider([WaitStatus(Outcome.DONE, "x")]))
        with pytest.raises(ValueError):
            reg.register(_ScriptedProvider([WaitStatus(Outcome.DONE, "y")]))

    def test_duplicate_kind_replace_ok(self):
        reg = WaitRegistry()
        reg.register(_ScriptedProvider([WaitStatus(Outcome.DONE, "x")]))
        p2 = _ScriptedProvider([WaitStatus(Outcome.DONE, "y")])
        reg.register(p2, replace=True)
        assert reg.get("demo") is p2

    @pytest.mark.asyncio
    async def test_wait_dispatches_to_provider(self):
        reg = WaitRegistry()
        reg.register(_ScriptedProvider([WaitStatus(Outcome.DONE, "ok!")]))
        r = await reg.wait("demo:h", timeout_seconds=10)
        assert r.status is ToolResultStatus.OK
        assert r.confirmation == "ok!"

    @pytest.mark.asyncio
    async def test_unknown_kind_errors_with_known_list(self):
        reg = WaitRegistry()
        reg.register(_ScriptedProvider([WaitStatus(Outcome.DONE, "x")]))
        r = await reg.wait("nope:h", timeout_seconds=10)
        assert r.status is ToolResultStatus.ERROR
        assert "no wait provider for kind 'nope'" in r.error
        assert r.data["known_kinds"] == ["demo"]

    @pytest.mark.asyncio
    async def test_malformed_ref_errors(self):
        reg = WaitRegistry()
        r = await reg.wait("garbage", timeout_seconds=10)
        assert r.status is ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# WaitRegistry.restore_if_current — identity-aware teardown (#2522 P3)
# ---------------------------------------------------------------------------


class _NamedProvider:
    """Distinct providers of the SAME kind, told apart by object identity."""

    kind = "demo"
    signal = None

    def __init__(self, tag):
        self.tag = tag

    async def poll(self, handle):  # pragma: no cover - never polled here
        raise NotImplementedError


class TestRestoreIfCurrent:
    def test_restores_previous_when_still_current(self):
        reg = WaitRegistry()
        host = _NamedProvider("host")
        reg.register(host)
        mine = _NamedProvider("mine")
        reg.register(mine, replace=True)
        assert reg.get("demo") is mine

        # Our provider is still current → restore the displaced host provider.
        assert reg.restore_if_current("demo", mine, host) is True
        assert reg.get("demo") is host

    def test_removes_kind_when_no_previous(self):
        reg = WaitRegistry()
        mine = _NamedProvider("mine")
        reg.register(mine)

        # We installed into an empty slot → restoring means removing the kind.
        assert reg.restore_if_current("demo", mine, None) is True
        assert reg.get("demo") is None
        assert reg.kinds() == []

    def test_retains_newer_provider_untouched(self):
        reg = WaitRegistry()
        host = _NamedProvider("host")
        reg.register(host)
        mine = _NamedProvider("mine")
        reg.register(mine, replace=True)
        # A NEWER owner replaces ours after we installed.
        newer = _NamedProvider("newer")
        reg.register(newer, replace=True)
        assert reg.get("demo") is newer

        # Our provider is no longer current → leave the newer one in place,
        # do NOT resurrect the stale host provider.
        assert reg.restore_if_current("demo", mine, host) is False
        assert reg.get("demo") is newer

    def test_no_provider_registered_is_a_benign_false(self):
        reg = WaitRegistry()
        mine = _NamedProvider("mine")
        assert reg.restore_if_current("demo", mine, None) is False
        assert reg.get("demo") is None


# ---------------------------------------------------------------------------
# WaitRegistry per-kind ownership stack — the A→B disable-order regression
# that a single-slot "save previous" design gets wrong (#2522 P3 redesign).
# ---------------------------------------------------------------------------


class TestOwnershipStack:
    def test_middle_removal_leaves_current_untouched(self):
        """host → A → B; removing A (middle) must NOT disturb current B."""
        reg = WaitRegistry()
        host = _NamedProvider("host")
        reg.register(host)
        a = _NamedProvider("A")
        reg.register(a, replace=True)
        b = _NamedProvider("B")
        reg.register(b, replace=True)
        assert reg.get("demo") is b

        # A is superseded by B → its removal returns False and B stays current.
        assert reg.deregister("demo", a) is False
        assert reg.get("demo") is b

    def test_disable_order_A_then_B_restores_host_not_disabled_A(self):
        """The core bug: host → A → B, disable A then B must restore HOST.

        The old single-slot design saved B's ``previous`` as A at registration
        time, so tearing B down restored A — resurrecting the already-disabled
        predecessor. The stack removes A when A is disabled, so B's teardown can
        only fall through to the nearest STILL-LIVE predecessor (host).
        """
        reg = WaitRegistry()
        host = _NamedProvider("host")
        reg.register(host)
        a = _NamedProvider("A")
        reg.register(a, replace=True)
        b = _NamedProvider("B")
        reg.register(b, replace=True)

        # Disable A first (it is currently in the MIDDLE, superseded by B).
        assert reg.deregister("demo", a) is False
        assert reg.get("demo") is b

        # Now disable B (the current provider). The nearest live predecessor is
        # host — A is gone and must never come back.
        assert reg.deregister("demo", b) is True
        assert reg.get("demo") is host
        assert reg.kinds() == ["demo"]

    def test_disable_top_then_next_walks_down_the_stack(self):
        """Disabling B then A (reverse order) walks host ← A ← B correctly."""
        reg = WaitRegistry()
        host = _NamedProvider("host")
        reg.register(host)
        a = _NamedProvider("A")
        reg.register(a, replace=True)
        b = _NamedProvider("B")
        reg.register(b, replace=True)

        assert reg.deregister("demo", b) is True
        assert reg.get("demo") is a
        assert reg.deregister("demo", a) is True
        assert reg.get("demo") is host

    def test_reregistering_same_provider_is_idempotent(self):
        """A feature whose init re-runs must not stack a duplicate of itself."""
        reg = WaitRegistry()
        host = _NamedProvider("host")
        reg.register(host)
        mine = _NamedProvider("mine")
        reg.register(mine, replace=True)
        reg.register(mine, replace=True)  # re-run in one live cycle
        assert reg.get("demo") is mine

        # One removal fully clears mine; the host underneath is restored once.
        assert reg.deregister("demo", mine) is True
        assert reg.get("demo") is host

    def test_unregister_pops_only_the_current_provider(self):
        reg = WaitRegistry()
        host = _NamedProvider("host")
        reg.register(host)
        mine = _NamedProvider("mine")
        reg.register(mine, replace=True)
        assert reg.unregister("demo") is True
        assert reg.get("demo") is host
        assert reg.unregister("demo") is True
        assert reg.get("demo") is None
        assert reg.unregister("demo") is False
