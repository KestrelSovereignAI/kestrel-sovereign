"""
SQLite WAL Listener

Monitors SQLite Write-Ahead Log (WAL) for changes to enable real-time
synchronization. This is the core mechanism for SQLite-first sync.

How it works:
    1. SQLite uses WAL mode for concurrent reads during writes
    2. We monitor the WAL file for new frames (committed transactions)
    3. Changes are extracted and queued for replication

References:
    - https://sqlite.org/wal.html
    - https://litestream.io/how-it-works/
"""

import asyncio
import logging
import os
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, AsyncIterator
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class WALFrame:
    """Represents a single WAL frame (page change)."""
    frame_number: int
    page_number: int
    db_size: int
    salt1: int
    salt2: int
    checksum1: int
    checksum2: int
    data: bytes
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WALChange:
    """Represents a batch of changes from WAL checkpoint."""
    frames: List[WALFrame]
    wal_checksum: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def bytes_changed(self) -> int:
        return sum(len(f.data) for f in self.frames)


class WALListener:
    """
    Monitors SQLite WAL file for changes.

    Provides an async iterator interface for streaming changes
    as they are committed to the database.
    """

    # WAL header size in bytes
    WAL_HEADER_SIZE = 32
    # WAL frame header size in bytes
    FRAME_HEADER_SIZE = 24
    # Standard SQLite page size
    DEFAULT_PAGE_SIZE = 4096

    def __init__(
        self,
        db_path: str,
        poll_interval: float = 0.1,
        on_change: Optional[Callable[[WALChange], None]] = None,
    ):
        """
        Initialize WAL listener.

        Args:
            db_path: Path to SQLite database file
            poll_interval: How often to check for changes (seconds)
            on_change: Optional callback for change notifications
        """
        self.db_path = Path(db_path)
        self.wal_path = Path(f"{db_path}-wal")
        self.poll_interval = poll_interval
        self._on_change = on_change
        self._running = False
        self._last_frame = 0
        self._last_size = 0
        self._page_size = self.DEFAULT_PAGE_SIZE

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start monitoring WAL for changes."""
        if self._running:
            return

        self._running = True
        logger.info(f"WAL listener started for {self.db_path}")

        try:
            async for change in self._watch_wal():
                if self._on_change:
                    self._on_change(change)
        except asyncio.CancelledError:
            logger.info("WAL listener cancelled")
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop monitoring WAL."""
        self._running = False
        logger.info(f"WAL listener stopped for {self.db_path}")

    async def _watch_wal(self) -> AsyncIterator[WALChange]:
        """Watch WAL file and yield changes."""
        while self._running:
            if not self.wal_path.exists():
                await asyncio.sleep(self.poll_interval)
                continue

            try:
                current_size = self.wal_path.stat().st_size
                if current_size > self._last_size:
                    change = await self._read_new_frames()
                    if change and change.frames:
                        self._last_size = current_size
                        yield change
            except (OSError, IOError) as e:
                logger.warning(f"Error reading WAL: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _read_new_frames(self) -> Optional[WALChange]:
        """Read new frames from WAL file."""
        frames = []

        try:
            with open(self.wal_path, "rb") as f:
                # Read and parse header if we haven't yet
                if self._last_frame == 0:
                    header = f.read(self.WAL_HEADER_SIZE)
                    if len(header) < self.WAL_HEADER_SIZE:
                        return None
                    self._parse_header(header)

                # Seek to last read position
                frame_size = self.FRAME_HEADER_SIZE + self._page_size
                start_offset = self.WAL_HEADER_SIZE + (self._last_frame * frame_size)
                f.seek(start_offset)

                # Read new frames
                frame_num = self._last_frame
                while True:
                    frame_header = f.read(self.FRAME_HEADER_SIZE)
                    if len(frame_header) < self.FRAME_HEADER_SIZE:
                        break

                    frame_data = f.read(self._page_size)
                    if len(frame_data) < self._page_size:
                        break

                    frame = self._parse_frame(frame_num, frame_header, frame_data)
                    if frame:
                        frames.append(frame)
                        frame_num += 1

                self._last_frame = frame_num

        except Exception as e:
            logger.error(f"Error reading WAL frames: {e}")
            return None

        if not frames:
            return None

        # Compute checksum of this batch
        checksum = hashlib.sha256()
        for frame in frames:
            checksum.update(frame.data)

        return WALChange(
            frames=frames,
            wal_checksum=checksum.hexdigest(),
        )

    def _parse_header(self, header: bytes) -> None:
        """Parse WAL header to get page size and other metadata."""
        # WAL header format (32 bytes):
        # 0-3: Magic number
        # 4-7: File format version
        # 8-11: Page size
        # 12-15: Checkpoint sequence number
        # 16-19: Salt-1
        # 20-23: Salt-2
        # 24-27: Checksum-1
        # 28-31: Checksum-2

        magic = struct.unpack(">I", header[0:4])[0]
        if magic not in (0x377f0682, 0x377f0683):
            logger.warning(f"Invalid WAL magic number: {magic}")
            return

        self._page_size = struct.unpack(">I", header[8:12])[0]
        logger.debug(f"WAL page size: {self._page_size}")

    def _parse_frame(
        self,
        frame_num: int,
        header: bytes,
        data: bytes,
    ) -> Optional[WALFrame]:
        """Parse a single WAL frame."""
        # Frame header format (24 bytes):
        # 0-3: Page number
        # 4-7: Database size (pages) after commit, or 0 if not commit
        # 8-11: Salt-1
        # 12-15: Salt-2
        # 16-19: Checksum-1
        # 20-23: Checksum-2

        page_num = struct.unpack(">I", header[0:4])[0]
        db_size = struct.unpack(">I", header[4:8])[0]
        salt1 = struct.unpack(">I", header[8:12])[0]
        salt2 = struct.unpack(">I", header[12:16])[0]
        checksum1 = struct.unpack(">I", header[16:20])[0]
        checksum2 = struct.unpack(">I", header[20:24])[0]

        return WALFrame(
            frame_number=frame_num,
            page_number=page_num,
            db_size=db_size,
            salt1=salt1,
            salt2=salt2,
            checksum1=checksum1,
            checksum2=checksum2,
            data=data,
        )

    async def get_current_position(self) -> dict:
        """Get current WAL position for resumable sync."""
        return {
            "db_path": str(self.db_path),
            "last_frame": self._last_frame,
            "last_size": self._last_size,
            "page_size": self._page_size,
        }

    async def set_position(self, position: dict) -> None:
        """Restore WAL position for resuming sync."""
        self._last_frame = position.get("last_frame", 0)
        self._last_size = position.get("last_size", 0)
        self._page_size = position.get("page_size", self.DEFAULT_PAGE_SIZE)
