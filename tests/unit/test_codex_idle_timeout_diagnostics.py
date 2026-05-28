"""codex app-server idle-timeout diagnostic surface (#1410).

The openai:plan route's most common failure is the codex app-server
opening the upstream websocket and never returning ``response.completed``.
This module pins three behaviors:

1. ``CodexAppServerClient.recent_stderr`` exposes a snapshot of the
   live-captured codex-rs stderr ring buffer.

2. ``CodexAppServerClient.recent_codex_log`` queries
   ``<CODEX_HOME>/logs_2.sqlite`` (codex-rs's structured log DB) and
   returns the last N rows; defensive on every failure mode (missing
   DB, schema drift, locked DB) — never raises.

3. ``iter_turn_events`` augments the idle-timeout ``CodexAppServerError``
   with both tails so the operator sees codex-side root cause instead
   of just "idle for Ns".

4. ``CodexAdapter._iter_with_overflow_hint`` branches the cap-vs-payload
   hint on whether the est payload actually exceeded the route cap —
   "compact or raise cap" advice is correct ONLY when payload > cap.
"""
import asyncio
import sqlite3
from pathlib import Path

import pytest

from kestrel_sovereign.llm.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
)


class TestRecentStderr:
    def test_returns_empty_when_no_stderr_captured(self):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._stderr_tail = []
        assert c.recent_stderr(10) == []

    def test_returns_tail_slice(self):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._stderr_tail = [f"line {i}" for i in range(20)]
        got = c.recent_stderr(5)
        assert got == ["line 15", "line 16", "line 17", "line 18", "line 19"]

    def test_n_larger_than_buffer_returns_all(self):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._stderr_tail = ["a", "b", "c"]
        assert c.recent_stderr(100) == ["a", "b", "c"]


class TestRecentCodexLog:
    """codex-rs writes to ``logs_2.sqlite`` via sqlx with the schema:
    ``logs(id, ts, ts_nanos, level, target, feedback_log_body,
    module_path, file, line, thread_id)``."""

    @staticmethod
    def _make_db(path: Path, rows):
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(
                "CREATE TABLE logs ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "ts INTEGER NOT NULL,"
                "ts_nanos INTEGER NOT NULL,"
                "level TEXT NOT NULL,"
                "target TEXT NOT NULL,"
                "feedback_log_body TEXT,"
                "module_path TEXT,"
                "file TEXT,"
                "line INTEGER,"
                "thread_id TEXT"
                ")"
            )
            for ts, level, target, body in rows:
                conn.execute(
                    "INSERT INTO logs "
                    "(ts, ts_nanos, level, target, feedback_log_body) "
                    "VALUES (?, 0, ?, ?, ?)",
                    (ts, level, target, body),
                )
            conn.commit()
        finally:
            conn.close()

    def test_returns_empty_when_codex_home_unset(self):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._codex_home = None
        assert c.recent_codex_log(30) == []

    def test_returns_empty_when_db_missing(self, tmp_path):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._codex_home = tmp_path  # no logs_2.sqlite in here
        assert c.recent_codex_log(30) == []

    def test_returns_rows_in_chronological_order(self, tmp_path):
        db_path = tmp_path / "logs_2.sqlite"
        self._make_db(db_path, [
            (1000, "INFO", "codex_core", "first event"),
            (2000, "WARN", "codex_app_server", "second event"),
            (3000, "ERROR", "codex_protocol", "third event"),
        ])
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._codex_home = tmp_path
        got = c.recent_codex_log(10)
        # Chronological order: oldest → newest
        assert len(got) == 3
        assert "first event" in got[0]
        assert "second event" in got[1]
        assert "third event" in got[2]
        # Format includes ts, level, target
        assert "1000" in got[0] and "INFO" in got[0] and "codex_core" in got[0]

    def test_limit_returns_only_last_n(self, tmp_path):
        db_path = tmp_path / "logs_2.sqlite"
        rows = [(i, "INFO", "x", f"event {i}") for i in range(20)]
        self._make_db(db_path, rows)
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._codex_home = tmp_path
        got = c.recent_codex_log(3)
        assert len(got) == 3
        # The last 3 events by insertion order are events 17, 18, 19
        assert "event 17" in got[0]
        assert "event 18" in got[1]
        assert "event 19" in got[2]

    def test_null_body_does_not_crash(self, tmp_path):
        db_path = tmp_path / "logs_2.sqlite"
        self._make_db(db_path, [(100, "INFO", "codex_core", None)])
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._codex_home = tmp_path
        got = c.recent_codex_log(10)
        assert len(got) == 1
        assert "INFO" in got[0] and "codex_core" in got[0]

    def test_corrupted_db_returns_empty(self, tmp_path):
        # File exists but isn't a valid sqlite DB — must not raise.
        db_path = tmp_path / "logs_2.sqlite"
        db_path.write_bytes(b"this is not sqlite")
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._codex_home = tmp_path
        assert c.recent_codex_log(30) == []

    def test_line_truncated_to_500_chars(self, tmp_path):
        db_path = tmp_path / "logs_2.sqlite"
        long_body = "x" * 1000
        self._make_db(db_path, [(1, "INFO", "t", long_body)])
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._codex_home = tmp_path
        got = c.recent_codex_log(1)
        assert len(got[0]) <= 500


class TestIterTurnEventsIdleTimeoutSurfacesTails:
    """The idle-timeout error must now carry the stderr ring buffer +
    the codex-rs log tail so the operator sees what codex thought
    happened upstream — not just ``idle for Ns``."""

    @pytest.mark.asyncio
    async def test_idle_timeout_appends_stderr_tail(self):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._closed_error = None
        c._stderr_tail = ["websocket closed", "auth refresh failed"]
        c._codex_home = None  # skip log tail path
        q: asyncio.Queue = asyncio.Queue()  # empty — guaranteed idle

        with pytest.raises(CodexAppServerError) as ei:
            async for _ in c.iter_turn_events(q, idle_timeout=0.05):
                pass
        msg = str(ei.value)
        assert "idle for 0.05s" in msg
        assert "websocket closed" in msg
        assert "auth refresh failed" in msg
        assert "codex stderr" in msg

    @pytest.mark.asyncio
    async def test_idle_timeout_appends_codex_log_tail(self, tmp_path):
        db_path = tmp_path / "logs_2.sqlite"
        TestRecentCodexLog._make_db(db_path, [
            (1, "ERROR", "codex_protocol", "MARKER_FROM_CODEX_LOG"),
        ])
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._closed_error = None
        c._stderr_tail = []
        c._codex_home = tmp_path
        q: asyncio.Queue = asyncio.Queue()

        with pytest.raises(CodexAppServerError) as ei:
            async for _ in c.iter_turn_events(q, idle_timeout=0.05):
                pass
        msg = str(ei.value)
        assert "MARKER_FROM_CODEX_LOG" in msg
        assert "codex-rs log" in msg

    @pytest.mark.asyncio
    async def test_idle_timeout_with_no_tails_keeps_base_message(self):
        c = CodexAppServerClient.__new__(CodexAppServerClient)
        c._closed_error = None
        c._stderr_tail = []
        c._codex_home = None
        q: asyncio.Queue = asyncio.Queue()

        with pytest.raises(CodexAppServerError) as ei:
            async for _ in c.iter_turn_events(q, idle_timeout=0.05):
                pass
        msg = str(ei.value)
        # Base message preserved when no diagnostics to attach
        assert "codex turn idle for 0.05s with no completion" in msg
        # No empty sections leaked
        assert "codex stderr" not in msg
        assert "codex-rs log" not in msg


class TestOverflowHintBranchesOnPayloadVsCap:
    """The cap-vs-payload hint in CodexAdapter._iter_with_overflow_hint
    is now branched (#1410):
      - payload > cap → "compact or raise cap" (correct advice)
      - payload ≤ cap → "transient upstream stall, retry" (honest)
    """

    async def _drive_hint(self, est_payload_tokens: int, cap_value):
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        # Stub the catalog service so cap is deterministic
        from kestrel_sovereign.llm import model_catalog as mc

        original = mc.get_catalog_service

        class _StubCatalog:
            def get_route_context_cap(self, route):
                return cap_value

        mc.get_catalog_service = lambda: _StubCatalog()

        # Stub app.iter_turn_events to raise the base idle-timeout error
        class _StubApp:
            async def iter_turn_events(self, sink, *, idle_timeout=300):
                raise CodexAppServerError(
                    "codex turn idle for 300s with no completion"
                )
                yield  # pragma: no cover — make this an async generator

        a = CodexAdapter.__new__(CodexAdapter)
        sink: asyncio.Queue = asyncio.Queue()

        try:
            with pytest.raises(CodexAppServerError) as ei:
                async for _ in a._iter_with_overflow_hint(
                    _StubApp(), sink, est_payload_tokens
                ):
                    pass
            return str(ei.value)
        finally:
            mc.get_catalog_service = original

    @pytest.mark.asyncio
    async def test_payload_under_cap_says_transient_stall_not_compact(self):
        msg = await self._drive_hint(est_payload_tokens=13_220, cap_value=20_480)
        assert "is within the per-turn cap" in msg
        assert "transient upstream codex" in msg.lower() or "transient upstream" in msg
        # Crucial: must NOT instruct the operator to compact when payload < cap
        assert "compact" not in msg.lower()
        assert "raise the cap" not in msg.lower()
        # Includes payload + cap in the diagnosis
        assert "13220" in msg
        assert "20480" in msg

    @pytest.mark.asyncio
    async def test_payload_over_cap_keeps_compact_advice(self):
        msg = await self._drive_hint(est_payload_tokens=25_000, cap_value=20_480)
        assert "EXCEEDS" in msg
        assert "compact" in msg.lower()
        assert "raise the cap" in msg.lower()
        assert "25000" in msg
        assert "20480" in msg

    @pytest.mark.asyncio
    async def test_cap_unset_treated_as_under_cap(self):
        # When the catalog can't surface a cap, we don't know if payload
        # exceeded it. Default to the "transient stall" branch — safer
        # advice than misdirecting the operator toward compaction.
        msg = await self._drive_hint(est_payload_tokens=13_220, cap_value=None)
        assert "within the per-turn cap" in msg
        assert "unset" in msg
        assert "compact" not in msg.lower()
