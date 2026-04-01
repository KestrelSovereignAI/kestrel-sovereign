"""
Bootstrap file loader for Kestrel agents.

Scans configured directories for known bootstrap filenames (SOUL.md, AGENTS.md,
MEMORY.md, etc.) and loads them into an ordered dict for injection into the
agent's system prompt.

Features:
- Per-file size limits (default 10KB, configurable)
- Total content budget (default 50KB)
- Priority ordering: agent-specific > project-level > global
- Head+tail truncation to preserve context at both ends
- Hot-reload support via reload()
- Database-backed custom file configuration
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default bootstrap files loaded in this order (all optional).
# SOUL.md gets special treatment in the system prompt wrapper.
DEFAULT_BOOTSTRAP_FILES = [
    "AGENTS.md",
    "SOUL.md",
    "TOOLS.md",
    "IDENTITY.md",
    "USER.md",
    "HEARTBEAT.md",
    "BOOTSTRAP.md",
    "MEMORY.md",
    "CAPABILITIES.md",
    "GOALS.md",
    "STRATEGY.yaml",
]

DEFAULT_MAX_BYTES_PER_FILE = 10_240      # 10 KB
DEFAULT_MAX_TOTAL_BYTES = 51_200         # 50 KB
DEFAULT_MAX_CHARS_PER_FILE = 20_000      # Backward compat with context_builder
DEFAULT_MAX_TOTAL_CHARS = 150_000        # Backward compat with context_builder

# Minimum remaining budget to bother truncating into (otherwise skip the file)
_MIN_USEFUL_CHARS = 100


def truncate_content(content: str, max_chars: int) -> str:
    """Truncate content keeping head (70%) and tail (20%) with a marker.

    This preserves context at both ends of the file, which is important
    because bootstrap files often have structure at the top (headers,
    identity) and actionable items at the bottom (rules, reminders).

    Args:
        content: The raw file content.
        max_chars: Maximum characters to keep.

    Returns:
        The original content if under limit, otherwise a truncated version
        with a ``[...truncated...]`` marker.
    """
    if len(content) <= max_chars:
        return content
    head_chars = int(max_chars * 0.7)
    tail_chars = int(max_chars * 0.2)
    head = content[:head_chars]
    tail = content[-tail_chars:] if tail_chars > 0 else ""
    return f"{head}\n[...truncated...]\n{tail}"


class BootstrapLoader:
    """Loads and caches bootstrap files for an agent.

    The loader scans one or more directories for recognised filenames,
    reads them subject to per-file and total size budgets, and caches
    the results.  The cache is invalidated explicitly via ``reload()``.

    The file list can be extended at runtime via ``add_file()`` and
    ``remove_file()``.  These changes are ephemeral unless persisted
    to the ``bootstrap_config`` database table by the caller.

    Args:
        agent_data_path: Primary directory to scan for bootstrap files.
            Typically ``agent_data/<agent_id>/``.
        extra_paths: Additional directories to scan, in decreasing
            priority order.  Files found in earlier paths shadow later
            ones with the same filename.
        max_chars_per_file: Per-file character limit before truncation.
        max_total_chars: Total character budget across all loaded files.
        file_order: Ordered list of filenames to look for.  Defaults
            to ``DEFAULT_BOOTSTRAP_FILES``.
        db: Optional async database handle.  If provided, the loader
            will read/write the ``bootstrap_config`` table.
        agent_id: Agent DID, required when *db* is provided.
    """

    def __init__(
        self,
        agent_data_path: Optional[str] = None,
        extra_paths: Optional[List[str]] = None,
        max_chars_per_file: int = DEFAULT_MAX_CHARS_PER_FILE,
        max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
        file_order: Optional[List[str]] = None,
        db=None,
        agent_id: Optional[str] = None,
    ):
        self._agent_data_path = Path(agent_data_path) if agent_data_path else None
        self._extra_paths: List[Path] = [
            Path(p) for p in (extra_paths or [])
        ]
        self._max_chars_per_file = max_chars_per_file
        self._max_total_chars = max_total_chars
        self._file_order: List[str] = list(file_order or DEFAULT_BOOTSTRAP_FILES)
        self._db = db
        self._agent_id = agent_id

        # Cached content: filename -> content (ordered)
        self._cache: OrderedDict[str, str] = OrderedDict()
        # Track resolved paths for reporting
        self._resolved_paths: Dict[str, str] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> OrderedDict[str, str]:
        """Load (or return cached) bootstrap files.

        Returns:
            Ordered dict mapping filename to content string.
        """
        if not self._loaded:
            self._do_load()
        return self._cache

    def reload(self) -> OrderedDict[str, str]:
        """Force re-read of all bootstrap files from disk.

        Returns:
            Ordered dict mapping filename to content string.
        """
        self._loaded = False
        self._cache.clear()
        self._resolved_paths.clear()
        return self.load()

    def get_bootstrap_content(self) -> Dict[str, str]:
        """Return a plain dict of filename -> content.

        Convenience wrapper around :meth:`load`.
        """
        return dict(self.load())

    def get_file(self, filename: str) -> Optional[str]:
        """Get the content of a specific bootstrap file, or None."""
        return self.load().get(filename)

    def list_files(self) -> List[Dict[str, str]]:
        """List all loaded bootstrap files with metadata.

        Returns:
            List of dicts with keys: name, path, chars, status.
        """
        self.load()  # Ensure loaded
        result = []

        # Show all configured files, whether loaded or not
        for filename in self._file_order:
            path = self._resolved_paths.get(filename)
            content = self._cache.get(filename)
            if content is not None:
                result.append({
                    "name": filename,
                    "path": path or "unknown",
                    "chars": len(content),
                    "status": "loaded",
                })
            else:
                # Try to find the file path even if not loaded
                found_path = self._find_file(filename)
                result.append({
                    "name": filename,
                    "path": str(found_path) if found_path else "not found",
                    "chars": 0,
                    "status": "not found" if not found_path else "skipped (budget)",
                })

        return result

    def add_file(self, filename: str, path: Optional[str] = None) -> bool:
        """Add a new bootstrap file to the loading order.

        If *path* is provided, it is used directly instead of scanning
        directories.  The file is appended at the end of the load order.

        Args:
            filename: Name of the file (e.g. ``"NOTES.md"``).
            path: Optional absolute path to the file.

        Returns:
            True if the file was added (not already present).
        """
        if filename in self._file_order:
            return False
        self._file_order.append(filename)
        # Force reload to pick up the new file
        self._loaded = False
        return True

    def remove_file(self, filename: str) -> bool:
        """Remove a bootstrap file from the loading order.

        Args:
            filename: Name of the file to remove.

        Returns:
            True if the file was present and removed.
        """
        if filename not in self._file_order:
            return False
        self._file_order.remove(filename)
        self._cache.pop(filename, None)
        self._resolved_paths.pop(filename, None)
        return True

    @property
    def total_chars(self) -> int:
        """Total characters across all loaded bootstrap files."""
        return sum(len(c) for c in self._cache.values())

    @property
    def file_count(self) -> int:
        """Number of loaded bootstrap files."""
        return len(self._cache)

    @property
    def file_order(self) -> List[str]:
        """Current file loading order."""
        return list(self._file_order)

    # ------------------------------------------------------------------
    # Database persistence (bootstrap_config table)
    # ------------------------------------------------------------------

    async def load_db_config(self) -> None:
        """Load additional file configuration from the bootstrap_config table.

        Merges database-stored entries into the file order.  Files marked
        as ``enabled=0`` are removed from the order.
        """
        if not self._db or not self._agent_id:
            return

        try:
            rows = await self._db.fetchall(
                """
                SELECT file_name, file_path, enabled, priority
                FROM bootstrap_config
                WHERE agent_id = ?
                ORDER BY priority ASC
                """,
                (self._agent_id,),
            )
            if not rows:
                return

            for row in rows:
                file_name = row[0]
                file_path = row[1]
                enabled = row[2]

                if not enabled:
                    # Disabled in DB -- remove from order if present
                    if file_name in self._file_order:
                        self._file_order.remove(file_name)
                    continue

                # Add to order if not already present
                if file_name not in self._file_order:
                    self._file_order.append(file_name)

            # Invalidate cache so next load() picks up changes
            self._loaded = False

        except Exception as e:
            logger.debug(f"bootstrap_config table not available: {e}")

    async def save_db_entry(
        self,
        file_name: str,
        file_path: str = "",
        enabled: bool = True,
        priority: int = 100,
        max_size_bytes: int = DEFAULT_MAX_BYTES_PER_FILE,
    ) -> None:
        """Persist a bootstrap file configuration entry to the database.

        Args:
            file_name: The bootstrap filename.
            file_path: Optional explicit path override.
            enabled: Whether the file should be loaded.
            priority: Sort priority (lower = earlier).
            max_size_bytes: Per-file size limit.
        """
        if not self._db or not self._agent_id:
            return

        import uuid
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        row_id = str(uuid.uuid4())

        await self._db.execute(
            """
            INSERT OR REPLACE INTO bootstrap_config
                (id, agent_id, file_name, file_path, enabled, priority, max_size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (row_id, self._agent_id, file_name, file_path, int(enabled), priority, max_size_bytes, now),
        )

    async def delete_db_entry(self, file_name: str) -> None:
        """Remove a bootstrap file configuration entry from the database."""
        if not self._db or not self._agent_id:
            return

        await self._db.execute(
            "DELETE FROM bootstrap_config WHERE agent_id = ? AND file_name = ?",
            (self._agent_id, file_name),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_load(self) -> None:
        """Scan directories and load bootstrap files into cache."""
        self._cache.clear()
        self._resolved_paths.clear()

        if not self._agent_data_path and not self._extra_paths:
            self._loaded = True
            return

        total_chars = 0

        for filename in self._file_order:
            filepath = self._find_file(filename)
            if filepath is None:
                logger.warning(
                    f"Bootstrap file '{filename}' not found in any search path — "
                    "agent will start without this context"
                )
                continue

            try:
                content = filepath.read_text(encoding="utf-8")
                if not content.strip():
                    continue

                # Per-file truncation
                content = truncate_content(content, self._max_chars_per_file)

                # Total budget check
                if total_chars + len(content) > self._max_total_chars:
                    remaining = self._max_total_chars - total_chars
                    if remaining > _MIN_USEFUL_CHARS:
                        content = truncate_content(content, remaining)
                    else:
                        logger.warning(
                            f"Bootstrap budget exhausted, skipping {filename}"
                        )
                        break

                self._cache[filename] = content
                self._resolved_paths[filename] = str(filepath)
                total_chars += len(content)
                logger.info(
                    f"Loaded bootstrap file: {filename} ({len(content)} chars)"
                )
            except Exception as e:
                logger.warning(f"Failed to load bootstrap file {filename}: {e}")

        if self._cache:
            names = ", ".join(self._cache.keys())
            logger.info(
                f"Bootstrap files loaded: {names} ({total_chars} total chars)"
            )

        self._loaded = True

    def _find_file(self, filename: str) -> Optional[Path]:
        """Find a file across search paths (agent-specific first).

        Returns the first match, giving priority to the agent data path,
        then extra paths in order.
        """
        search_paths: List[Path] = []
        if self._agent_data_path:
            search_paths.append(self._agent_data_path)
        search_paths.extend(self._extra_paths)

        for directory in search_paths:
            candidate = directory / filename
            if candidate.exists():
                return candidate

        return None
