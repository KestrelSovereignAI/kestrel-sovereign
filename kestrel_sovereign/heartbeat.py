"""
Heartbeat system for Kestrel Sovereign.

Provides periodic agent self-checks inspired by the OpenClaw heartbeat pattern.
The agent reads HEARTBEAT.md on each tick and acts on its contents.

Configuration via [heartbeat] section in kestrel.toml:

    [heartbeat]
    enabled = true
    interval = "30m"
    active_hours_start = "09:00"
    active_hours_end = "22:00"
    timezone = "America/New_York"
    target = "log"
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kestrel_sovereign.config import load_section, parse_duration

_UTC = timezone.utc

if TYPE_CHECKING:
    from kestrel_sovereign.kestrel_agent import KestrelAgent

logger = logging.getLogger(__name__)

# Default heartbeat prompt sent as user input on each tick.
HEARTBEAT_PROMPT = (
    "[HEARTBEAT] Read HEARTBEAT.md if it exists. Follow it strictly. "
    "Do not infer or repeat old tasks from prior chats. "
    "If nothing needs attention, reply exactly: HEARTBEAT_OK\n"
    "Current time: {timestamp}"
)

# Default HEARTBEAT.md template created when heartbeat is enabled but file doesn't exist.
DEFAULT_HEARTBEAT_TEMPLATE = """\
# Heartbeat Checklist

Review these items on each heartbeat. If nothing needs attention, reply HEARTBEAT_OK.

- [ ] Any pending tasks or reminders?
- [ ] Any scheduled events coming up?
- [ ] System status normal?
"""

# Maximum heartbeat results to keep in history.
MAX_HISTORY = 50

# Regex for detecting HEARTBEAT_OK token (with optional markdown/HTML wrappers).
_OK_PATTERN = re.compile(
    r'(?:<b>)?(?:\*\*)?HEARTBEAT_OK(?:\*\*)?(?:</b>)?[!.]*',
    re.IGNORECASE,
)


@dataclass
class HeartbeatConfig:
    """Configuration for the heartbeat system."""
    enabled: bool = False
    interval_seconds: int = 1800  # 30 minutes
    active_hours_start: Optional[str] = None  # "HH:MM"
    active_hours_end: Optional[str] = None    # "HH:MM"
    timezone: str = "UTC"
    heartbeat_file: str = "HEARTBEAT.md"
    target: str = "log"       # "log", "last_session", "none"
    suppress_ok: bool = True  # Suppress routine HEARTBEAT_OK results

    @classmethod
    def from_config(cls) -> "HeartbeatConfig":
        """Load heartbeat config from kestrel.toml [heartbeat] section."""
        cfg = load_section("heartbeat")
        if not cfg:
            return cls()

        interval_seconds = 1800
        raw_interval = cfg.get("interval", "30m")
        if isinstance(raw_interval, str):
            try:
                interval_seconds = parse_duration(raw_interval)
            except ValueError:
                logger.warning(f"Invalid heartbeat interval '{raw_interval}', using 30m")
        elif isinstance(raw_interval, (int, float)):
            interval_seconds = int(raw_interval) * 60  # treat as minutes

        return cls(
            enabled=cfg.get("enabled", False),
            interval_seconds=interval_seconds,
            active_hours_start=cfg.get("active_hours_start"),
            active_hours_end=cfg.get("active_hours_end"),
            timezone=cfg.get("timezone", "UTC"),
            heartbeat_file=cfg.get("heartbeat_file", "HEARTBEAT.md"),
            target=cfg.get("target", "log"),
            suppress_ok=cfg.get("suppress_ok", True),
        )


@dataclass
class HeartbeatResult:
    """Result of a single heartbeat execution."""
    status: str                    # "ok", "alert", "skipped", "error"
    message: Optional[str] = None  # Alert content if not OK
    timestamp: str = ""
    duration_ms: int = 0
    reason: Optional[str] = None   # Skip/error reason

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HeartbeatRunner:
    """Runs periodic heartbeat checks for a Kestrel agent.

    Uses a simple asyncio.create_task loop — no external scheduler dependency.
    """

    def __init__(self, agent: "KestrelAgent", config: HeartbeatConfig):
        self.agent = agent
        self.config = config
        self._task: Optional[asyncio.Task] = None
        self._history: List[Dict[str, Any]] = []
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    async def start(self) -> None:
        """Start the heartbeat loop."""
        if not self.config.enabled:
            logger.info("Heartbeat disabled in config, not starting")
            return
        if self._running:
            logger.warning("Heartbeat already running")
            return

        # Ensure HEARTBEAT.md exists (create template if missing)
        self._ensure_heartbeat_file()

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"Heartbeat started: interval={self.config.interval_seconds}s, "
            f"target={self.config.target}"
        )

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Heartbeat stopped")

    async def run_once(self) -> HeartbeatResult:
        """Execute a single heartbeat tick (for manual triggers / API)."""
        return await self._tick()

    def get_status(self) -> Dict[str, Any]:
        """Return current heartbeat status for API/commands."""
        return {
            "enabled": self.config.enabled,
            "running": self.is_running,
            "interval_seconds": self.config.interval_seconds,
            "active_hours": {
                "start": self.config.active_hours_start,
                "end": self.config.active_hours_end,
                "timezone": self.config.timezone,
            } if self.config.active_hours_start else None,
            "target": self.config.target,
            "history_count": len(self._history),
            "last_result": self._history[-1] if self._history else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Main heartbeat loop."""
        # Initial delay: wait one interval before first tick
        try:
            await asyncio.sleep(self.config.interval_seconds)
        except asyncio.CancelledError:
            return

        while self._running:
            try:
                result = await self._tick()
                logger.info(
                    f"Heartbeat tick: status={result.status}, "
                    f"duration={result.duration_ms}ms"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat tick error: {e}", exc_info=True)

            try:
                await asyncio.sleep(self.config.interval_seconds)
            except asyncio.CancelledError:
                break

    async def _tick(self) -> HeartbeatResult:
        """Execute a single heartbeat tick.

        Routes through the SignalDispatcher (Phase 3 of #889). The
        dispatcher's attention_policy enforces quiet hours; the result's
        OK/COALESCED/DROPPED_QUIET_HOURS status is translated back to
        heartbeat's ok/skipped/error vocabulary by `_translate_result`.

        In-memory history is preserved as the source of truth for the
        existing `/heartbeat/status` endpoint. The signal_log written by
        the dispatcher provides parallel audit; migrating the endpoint
        to read from signal_log is a follow-up.
        """
        from kestrel_sdk.signals import (
            Signal,
            SignalMode,
            Status,
            Urgency,
            Visibility,
        )
        from kestrel_sovereign.signals.sources.heartbeat import SOURCE_NAME

        start_time = time.monotonic()
        timestamp = datetime.now(tz=_UTC).isoformat()

        heartbeat_content = self._load_heartbeat_file() or ""

        signal = Signal(
            source=SOURCE_NAME,
            kind="tick",
            mode=SignalMode.COGNITION,
            payload={"heartbeat_md": heartbeat_content},
            target_agent=self.agent.did,
            visibility=Visibility.INTERNAL,
            urgency=Urgency.NORMAL,
        )

        try:
            sig_result = await self.agent.dispatcher.dispatch_signal(signal)
            duration_ms = int((time.monotonic() - start_time) * 1000)
            result = self._translate_result(sig_result, timestamp, duration_ms)
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            result = HeartbeatResult(
                status="error",
                message=str(e),
                timestamp=timestamp,
                duration_ms=duration_ms,
                reason=f"dispatch_signal failed: {type(e).__name__}",
            )

        self._record_result(result)
        return result

    def _translate_result(
        self, sig_result, timestamp: str, duration_ms: int
    ) -> HeartbeatResult:
        """Map a dispatcher SignalResult into the heartbeat vocabulary.

        Today's heartbeat statuses are: ok | alert | skipped | error. The
        dispatcher knows: OK | COALESCED | DROPPED_QUIET_HOURS |
        DROPPED_RATE_LIMIT | DROPPED_CYCLE | DROPPED_VALIDATION | FAILED.

        - OK              → run `_normalize_response` on the artifact (LLM text)
        - DROPPED_*       → "skipped" with a reason string
        - COALESCED       → "skipped" (a duplicate fired within the window)
        - FAILED          → "error" with the dispatcher's error message
        """
        from kestrel_sdk.signals import Status

        status = sig_result.status

        if status == Status.OK:
            response_text = sig_result.artifact or ""
            return self._normalize_response(response_text, timestamp, duration_ms)

        if status == Status.FAILED:
            return HeartbeatResult(
                status="error",
                message=sig_result.error or "dispatcher reported FAILED",
                timestamp=timestamp,
                duration_ms=duration_ms,
                reason=f"dispatch failed: {sig_result.error or 'unknown'}",
            )

        # COALESCED, DROPPED_QUIET_HOURS, DROPPED_RATE_LIMIT, DROPPED_CYCLE,
        # DROPPED_VALIDATION → no LLM call happened; record as skipped.
        return HeartbeatResult(
            status="skipped",
            timestamp=timestamp,
            duration_ms=duration_ms,
            reason=f"{status.value}: {sig_result.error or ''}".strip(": "),
        )

    def _normalize_response(
        self, text: str, timestamp: str, duration_ms: int
    ) -> HeartbeatResult:
        """Detect HEARTBEAT_OK token and classify result."""
        if not text:
            return HeartbeatResult(
                status="ok",
                timestamp=timestamp,
                duration_ms=duration_ms,
            )

        # Strip HEARTBEAT_OK token
        stripped = _OK_PATTERN.sub("", text).strip()

        if not stripped:
            # Response was just HEARTBEAT_OK (possibly with formatting)
            return HeartbeatResult(
                status="ok",
                timestamp=timestamp,
                duration_ms=duration_ms,
            )

        # Check if response contains HEARTBEAT_OK alongside other text
        if _OK_PATTERN.search(text):
            # OK token present but with additional content — treat as alert
            # (agent said OK but also had something to report)
            return HeartbeatResult(
                status="ok",
                message=stripped if len(stripped) > 10 else None,
                timestamp=timestamp,
                duration_ms=duration_ms,
            )

        # No OK token — this is an alert
        return HeartbeatResult(
            status="alert",
            message=text,
            timestamp=timestamp,
            duration_ms=duration_ms,
        )

    def _is_within_active_hours(self) -> bool:
        """Check if the current time is within configured active hours."""
        if not self.config.active_hours_start or not self.config.active_hours_end:
            return True  # No active hours configured = always active

        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            # Python < 3.9 fallback
            try:
                from backports.zoneinfo import ZoneInfo
            except ImportError:
                logger.warning("zoneinfo not available, skipping active hours check")
                return True

        try:
            tz = ZoneInfo(self.config.timezone)
            now = datetime.now(tz)
            current_time = now.strftime("%H:%M")
            start = self.config.active_hours_start
            end = self.config.active_hours_end

            if start <= end:
                return start <= current_time <= end
            else:
                # Overnight range (e.g., 22:00 - 06:00)
                return current_time >= start or current_time <= end
        except Exception as e:
            logger.warning(f"Active hours check failed: {e}")
            return True  # Fail open

    def _load_heartbeat_file(self) -> Optional[str]:
        """Read HEARTBEAT.md from the agent data directory."""
        agent_data_dir = self._get_agent_data_dir()
        if not agent_data_dir:
            return None

        filepath = agent_data_dir / self.config.heartbeat_file
        if not filepath.exists():
            return None

        try:
            content = filepath.read_text(encoding="utf-8")
            return content if content.strip() else None
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            return None

    def _ensure_heartbeat_file(self) -> None:
        """Create default HEARTBEAT.md template if it doesn't exist."""
        agent_data_dir = self._get_agent_data_dir()
        if not agent_data_dir:
            return

        filepath = agent_data_dir / self.config.heartbeat_file
        if filepath.exists():
            return

        try:
            filepath.write_text(DEFAULT_HEARTBEAT_TEMPLATE, encoding="utf-8")
            logger.info(f"Created default {self.config.heartbeat_file} at {filepath}")
        except Exception as e:
            logger.warning(f"Failed to create default heartbeat file: {e}")

    def _get_agent_data_dir(self) -> Optional[Path]:
        """Get the agent data directory from the agent's storage path."""
        storage_path = getattr(self.agent, 'storage_path', None)
        if storage_path:
            return Path(storage_path).parent
        return None

    def _record_result(self, result: HeartbeatResult) -> None:
        """Store heartbeat result in history (in-memory, capped)."""
        self._history.append(result.to_dict())
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]

        # Log based on result type
        if result.status == "alert":
            logger.warning(f"Heartbeat ALERT: {result.message[:200] if result.message else 'no message'}")
        elif result.status == "error":
            logger.error(f"Heartbeat ERROR: {result.reason}")
        elif result.status == "skipped":
            logger.debug(f"Heartbeat skipped: {result.reason}")
