"""Read-merge-write helpers for ``.env`` files.

The wizard never destroys user secrets. Every write goes through
``write_env`` which:

  1. Backs up the current file to ``.env.backup-<timestamp>`` if its
     content would change.
  2. Preserves the order, comments, and blank lines of existing keys.
  3. Appends new keys at the bottom under a generated section header.
  4. Refuses to write a key whose value is empty or ``None`` unless
     ``allow_empty=True`` (prevents accidental clobbering).

The format we care about is the classic dotenv subset:

    # comment
    KEY=value
    KEY="quoted value"

Multi-line values, ``export`` prefixes, and shell expansion are out of
scope; we leave such lines untouched but never edit them.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_KEY_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


@dataclass(frozen=True)
class EnvWriteResult:
    """Outcome of a write_env call.

    ``backup_path`` is None when nothing was written (no diff).
    ``added`` and ``updated`` are the keys that changed.
    """

    path: Path
    backup_path: Path | None
    added: tuple[str, ...]
    updated: tuple[str, ...]


def read_env(path: Path) -> dict[str, str]:
    """Parse a dotenv file into a plain dict. Missing file → empty dict."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _KEY_LINE_RE.match(raw)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        out[key] = _unquote(value)
    return out


def write_env(
    path: Path,
    updates: Mapping[str, str],
    *,
    allow_empty: bool = False,
    section_header: str = "Added by kestrel setup",
) -> EnvWriteResult:
    """Merge ``updates`` into the dotenv file at ``path``.

    Existing keys are updated in place (line order preserved). New keys
    are appended below a section header. The file is backed up to
    ``<path>.backup-YYYYMMDD-HHMMSS`` before any change.

    Pass ``allow_empty=True`` to allow writing empty-string values
    (use sparingly — the default refuses to clobber a key with "").
    """
    filtered = {
        k: v for k, v in updates.items() if allow_empty or (v not in (None, ""))
    }
    if not filtered:
        return EnvWriteResult(path=path, backup_path=None, added=(), updated=())

    existing_lines: list[str] = (
        path.read_text(encoding="utf-8").splitlines()
        if path.exists()
        else []
    )

    seen_keys: set[str] = set()
    new_lines: list[str] = []
    updated: list[str] = []

    for line in existing_lines:
        m = _KEY_LINE_RE.match(line)
        if not m:
            new_lines.append(line)
            continue
        key = m.group(1)
        seen_keys.add(key)
        if key in filtered:
            new_value = filtered[key]
            old_value = _unquote(m.group(2).strip())
            if new_value != old_value:
                updated.append(key)
                new_lines.append(f"{key}={_quote_if_needed(new_value)}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    added: list[str] = [k for k in filtered if k not in seen_keys]
    if added:
        if new_lines and new_lines[-1] != "":
            new_lines.append("")
        new_lines.append(f"# {section_header}")
        for key in added:
            new_lines.append(f"{key}={_quote_if_needed(filtered[key])}")

    new_text = "\n".join(new_lines)
    if new_lines:
        new_text += "\n"

    old_text = (
        path.read_text(encoding="utf-8") if path.exists() else ""
    )
    if new_text == old_text:
        return EnvWriteResult(path=path, backup_path=None, added=(), updated=())

    backup_path: Path | None = None
    if path.exists():
        backup_path = _backup_path(path)
        shutil.copy2(path, backup_path)
        # The .env (and its backup) can hold KESTREL_DATA_KEY / KESTREL_API_KEY —
        # restrict to owner-only so it isn't world-readable (#1729). copy2
        # preserves the source mode, which may have been lax.
        _chmod_600(backup_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    _chmod_600(path)
    return EnvWriteResult(
        path=path,
        backup_path=backup_path,
        added=tuple(added),
        updated=tuple(updated),
    )


def _chmod_600(path: Path) -> None:
    """Best-effort owner-only (0o600) perms; no-op on platforms without chmod."""
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _backup_path(path: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.name}.backup-{ts}")


def _unquote(value: str) -> str:
    """Strip a single layer of surrounding quotes if present."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _quote_if_needed(value: str) -> str:
    """Quote values that contain whitespace or shell-special characters.

    Plain alphanumeric/underscore/hyphen/dot values are left bare for
    readability; anything else is double-quoted with internal double
    quotes escaped.
    """
    if value == "":
        return ""
    if re.fullmatch(r"[A-Za-z0-9_./:@\-+]+", value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
