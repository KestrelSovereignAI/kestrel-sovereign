"""Read-merge-write helpers for ``kestrel.toml``.

Mirrors the safety contract of :mod:`kestrel_sovereign.setup.env_file`:

  - Existing keys outside the merge scope are preserved.
  - Backup is taken to ``<path>.backup-<timestamp>`` before any change.
  - No-op writes (identical content) skip the backup.

Comment-preservation is *not* offered here. Python's ``toml`` library
discards comments on load. Users who care about comments should keep
their hand-written ``kestrel.toml`` and rely on the wizard only for
machine-managed sections (``[llm]``, ``[features]``, ``[agent]``).
The risk of comment loss is bounded by the backup.
"""

from __future__ import annotations

import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import toml


@dataclass(frozen=True)
class TomlWriteResult:
    """Outcome of a write_toml call."""

    path: Path
    backup_path: Path | None
    changed: bool


def read_toml(path: Path) -> dict[str, Any]:
    """Load ``kestrel.toml`` (or any toml). Missing file → empty dict."""
    if not path.exists():
        return {}
    try:
        return toml.load(path)
    except toml.TomlDecodeError:
        return {}


def write_toml(
    path: Path,
    updates: Mapping[str, Any],
    *,
    deep_merge: bool = True,
    backup: bool = True,
) -> TomlWriteResult:
    """Merge ``updates`` into the toml file at ``path``.

    With ``deep_merge=True`` (default), nested tables are merged key by
    key — a partial ``{"llm": {"route_priority": [...]}}`` update will
    not erase an existing ``[llm.vendors.openai]`` table.

    With ``deep_merge=False``, top-level keys in ``updates`` replace
    their counterparts wholesale.
    """
    existing = read_toml(path)
    merged = (
        _deep_merge(existing, updates) if deep_merge else {**existing, **updates}
    )

    if merged == existing:
        return TomlWriteResult(path=path, backup_path=None, changed=False)

    backup_path: Path | None = None
    if backup and path.exists():
        backup_path = _backup_path(path)
        shutil.copy2(path, backup_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(merged, f)
    return TomlWriteResult(path=path, backup_path=backup_path, changed=True)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a deep copy of ``base``."""
    out = deepcopy(base)
    for key, value in overlay.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value) if isinstance(value, (dict, list)) else value
    return out


def _backup_path(path: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.name}.backup-{ts}")
