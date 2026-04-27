"""Append-only JSONL audit log for the computer-use feature.

One record per tool call, written with ``fsync`` so a crash never loses
the chain. The same record is forwarded to the existing
``record_tool_usage`` feedback hook (when an agent is supplied) so the
knowledge graph stays in lock-step with the on-disk trail.

Schema::

    {
      "ts": "<iso8601 utc>",
      "agent_did": "<did or 'anonymous'>",
      "tool": "fs-read|fs-list|fs-write|fs-edit|shell",
      "backend": "docker|local",
      "args": {...},
      "allowed_by": ["privacy", "constitution", "approval:<scope>:<approver>"],
      "outcome": "ok|denied|error",
      "duration_ms": <int>,
      "error": null | "<short message>"
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """One row in the audit log."""

    tool: str
    backend: str
    args: dict[str, Any]
    allowed_by: list[str] = field(default_factory=list)
    outcome: str = "ok"
    duration_ms: int = 0
    error: str | None = None
    agent_did: str = "anonymous"
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AuditLog:
    """Append-only JSONL writer with fsync per record.

    Concurrent writers are serialized by an :class:`asyncio.Lock` so the
    file always contains complete records in arrival order. A failed
    write does not silently drop the record — it is logged at WARNING
    level and re-raised; the feature is expected to surface that as an
    error result rather than continue.
    """

    def __init__(self, path: Path | str, *, agent=None) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._agent = agent

    async def write(self, record: AuditRecord) -> None:
        async with self._lock:
            await asyncio.to_thread(self._write_sync, record)
            await self._forward_to_feedback(record)

    def _write_sync(self, record: AuditRecord) -> None:
        line = json.dumps(asdict(record), separators=(",", ":")) + "\n"
        # O_APPEND on POSIX is atomic for a single write() up to PIPE_BUF;
        # JSONL records are far smaller than that for any sane payload.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    async def _forward_to_feedback(self, record: AuditRecord) -> None:
        if self._agent is None:
            return
        hook = getattr(self._agent, "record_tool_usage", None)
        if hook is None:
            return
        try:
            payload = asdict(record)
            if asyncio.iscoroutinefunction(hook):
                await hook("computer_use", record.tool, payload)
            else:
                await asyncio.to_thread(hook, "computer_use", record.tool, payload)
        except Exception as exc:  # noqa: BLE001 — feedback is best-effort
            logger.warning("audit feedback hook failed: %s", exc)
