"""Whether the SOUL.md on disk is the SOUL the next boot will serve (#3779).

An agent's canonical SOUL is the current encrypted ``agent.soul.markdown``
version in ``agent_identity_resources``. ``agent_data/<agent>/SOUL.md`` is only
its seed and cache: at boot a canonical version, when one exists, is loaded in
place of the file. An edit made to the file alone, even one loaded into the
running session by ``bootstrap_reload``, is therefore gone after the next
restart. :func:`inspect_soul_file` compares the two so boot, reload and status
report a mismatch instead of resolving it silently. ``!revise-soul`` is how an
operator makes the file the new canonical version.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from kestrel_sovereign.execution_custody import is_execution_control_error

SOUL_FILENAME = "SOUL.md"
SOUL_REVISION_COMMAND = "!revise-soul"


def soul_sha256(content: str) -> str:
    """The digest the resource store records as a SOUL version's content hash."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class SoulFileState(str, Enum):
    """How SOUL.md on disk relates to the canonical SOUL."""

    IN_SYNC = "in_sync"            # the file holds the canonical text
    FILE_DIFFERS = "file_differs"  # the next boot serves canonical, not the file
    FILE_MISSING = "file_missing"  # canonical exists; the file is absent or empty
    NO_CANONICAL = "no_canonical"  # the next boot serves the file, if any
    UNVERIFIED = "unverified"      # the file or the canonical SOUL could not be read


@dataclass(frozen=True)
class SoulFileStatus:
    """The comparison of SOUL.md with the canonical SOUL."""

    state: SoulFileState
    soul_path: Optional[str]
    file_sha256: Optional[str] = None
    canonical_version: Optional[int] = None
    canonical_sha256: Optional[str] = None
    error: Optional[str] = None

    @property
    def file_survives_restart(self) -> bool:
        """Whether the next boot serves what SOUL.md holds now."""
        return self.state in (SoulFileState.IN_SYNC, SoulFileState.NO_CANONICAL)

    def revert_warning(self) -> Optional[str]:
        """Why the next restart will not serve SOUL.md as it is now, if it won't."""
        if self.state is SoulFileState.FILE_DIFFERS:
            return (
                f"SOUL.md differs from the canonical SOUL v{self.canonical_version}, "
                f"and the next restart serves v{self.canonical_version}, not the "
                f"file. To keep the file, the Sovereign runs "
                f"`{SOUL_REVISION_COMMAND} <reason>`."
            )
        if self.state is SoulFileState.FILE_MISSING:
            return (
                "SOUL.md is missing or empty; the next restart serves the canonical "
                f"SOUL v{self.canonical_version}."
            )
        if self.state is SoulFileState.UNVERIFIED:
            return (
                f"Could not compare SOUL.md with the canonical SOUL ({self.error}), "
                "so whether the next restart serves the file is unknown."
            )
        return None

    def reload_caveat(self) -> Optional[str]:
        """Why a SOUL.md just reloaded into the session is not durable, if it isn't."""
        warning = self.revert_warning()
        if warning is None:
            return None
        if self.state is SoulFileState.FILE_DIFFERS:
            return "The reloaded SOUL.md is live for this session only. " + warning
        if self.state is SoulFileState.FILE_MISSING:
            return "The reload left this session without a SOUL. " + warning
        return warning

    def boot_warning(self) -> Optional[str]:
        """What boot reports when it serves canonical over a different file."""
        if self.state is SoulFileState.FILE_DIFFERS:
            return (
                f"SOUL.md at {self.soul_path} differs from the canonical SOUL "
                f"v{self.canonical_version} (file sha256 {self.file_sha256[:12]}, "
                f"canonical sha256 {self.canonical_sha256[:12]}). Serving canonical "
                f"v{self.canonical_version}; the file's edits are NOT in effect. "
                f"Run `{SOUL_REVISION_COMMAND} <reason>` as the Sovereign to make "
                "the file the canonical SOUL."
            )
        if self.state is SoulFileState.UNVERIFIED:
            return (
                f"Could not compare SOUL.md at {self.soul_path} with the canonical "
                f"SOUL: {self.error}"
            )
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "file_survives_restart": self.file_survives_restart,
            "soul_path": self.soul_path,
            "file_sha256": self.file_sha256,
            "canonical_version": self.canonical_version,
            "canonical_sha256": self.canonical_sha256,
            "error": self.error,
        }


async def inspect_soul_file(
    storage: Any, agent_data_path: Optional[Union[str, Path]]
) -> SoulFileStatus:
    """Compare ``<agent_data_path>/SOUL.md`` with the canonical SOUL.

    An empty file counts as missing, as it does for the boot-time promotion.
    A storage without a private-resource reader has no canonical SOUL.
    """
    soul_path = Path(agent_data_path) / SOUL_FILENAME if agent_data_path else None
    path_text = str(soul_path) if soul_path is not None else None

    # The canonical SOUL is read before the file. A writer that publishes the
    # file's text on the strength of this comparison names the canonical
    # version read here as the one it expects, so that version must be no
    # newer than the file text: a canonical write landing between the two
    # reads is then caught as a conflict rather than overwritten with text
    # read before it (#3779).
    reader = getattr(storage, "get_current_agent_resource", None)
    canonical = None
    canonical_error: Optional[Exception] = None
    if reader is not None:
        try:
            canonical = await reader()
        except Exception as exc:
            if is_execution_control_error(exc):
                raise
            canonical_error = exc

    file_content: Optional[str] = None
    if soul_path is not None:
        try:
            file_content = soul_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            file_content = None
        except (OSError, UnicodeDecodeError) as exc:
            return SoulFileStatus(
                SoulFileState.UNVERIFIED, path_text, error=f"SOUL.md read failed: {exc}"
            )
    if file_content is not None and not file_content.strip():
        file_content = None
    file_sha = soul_sha256(file_content) if file_content is not None else None

    if canonical_error is not None:
        return SoulFileStatus(
            SoulFileState.UNVERIFIED,
            path_text,
            file_sha256=file_sha,
            error=f"canonical SOUL read failed: {canonical_error}",
        )
    canonical_content = getattr(canonical, "content", None) if canonical else None
    if not canonical_content:
        # An empty current version is no canonical SOUL to boot, which then
        # promotes the file, but it is still the current version: a write
        # computed from this comparison must name it as the one it expects.
        return SoulFileStatus(
            SoulFileState.NO_CANONICAL,
            path_text,
            file_sha256=file_sha,
            canonical_version=getattr(canonical, "version", None) if canonical else None,
        )
    if not isinstance(canonical_content, str):
        return SoulFileStatus(
            SoulFileState.UNVERIFIED,
            path_text,
            file_sha256=file_sha,
            error=(
                "canonical SOUL body is "
                f"{type(canonical_content).__name__}, not text"
            ),
        )

    if file_content is None:
        state = SoulFileState.FILE_MISSING
    elif file_content == canonical_content:
        state = SoulFileState.IN_SYNC
    else:
        state = SoulFileState.FILE_DIFFERS
    return SoulFileStatus(
        state,
        path_text,
        file_sha256=file_sha,
        canonical_version=getattr(canonical, "version", None),
        canonical_sha256=soul_sha256(canonical_content),
    )
