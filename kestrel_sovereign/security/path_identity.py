"""Filesystem-aware path identity and containment checks.

``Path.resolve()`` removes symlink aliases but deliberately preserves caller
spelling.  On a case-insensitive filesystem that spelling is not an identity:
``host-data`` and ``HOST-DATA`` can name the same directory.  Custody guards
use this module when a merely lexical comparison could grant write access to
the same inode through another spelling.
"""

from __future__ import annotations

import os
import stat
import unicodedata
from pathlib import Path


def _nearest_existing_path(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _alternate_case(name: str) -> str | None:
    for position, character in enumerate(name):
        swapped = character.swapcase()
        if swapped != character:
            return name[:position] + swapped + name[position + 1 :]
    return None


def _filesystem_is_case_insensitive(path: Path) -> bool:
    """Probe existing ancestors without creating filesystem state."""

    if os.name == "nt":
        return True
    existing = _nearest_existing_path(path)
    for candidate in (existing, *existing.parents):
        alternate_name = _alternate_case(candidate.name)
        if alternate_name is None:
            continue
        alternate = candidate.with_name(alternate_name)
        try:
            if alternate.exists() and candidate.samefile(alternate):
                return True
        except OSError:
            continue
    return False


def _casefolded_parts(path: Path) -> tuple[str, ...]:
    # APFS/HFS aliases may differ by both case and Unicode normalization.
    return tuple(
        unicodedata.normalize("NFD", part).casefold() for part in path.parts
    )


def paths_overlap_by_filesystem_identity(first: Path, second: Path) -> bool:
    """Return whether either custody root contains the other on this volume.

    Existing aliases are compared by inode.  For descendants that do not exist
    yet, the nearest existing ancestor establishes the volume's case behavior;
    component-wise normalized comparison then predicts the identity that a
    later create would receive.
    """

    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    if first == second or first in second.parents or second in first.parents:
        return True
    try:
        if first.exists() and second.exists() and first.samefile(second):
            return True
    except OSError:
        pass
    if not (
        _filesystem_is_case_insensitive(first)
        or _filesystem_is_case_insensitive(second)
    ):
        return False
    first_parts = _casefolded_parts(first)
    second_parts = _casefolded_parts(second)
    shorter = min(len(first_parts), len(second_parts))
    return first_parts[:shorter] == second_parts[:shorter]


def paths_equal_by_filesystem_identity(first: Path, second: Path) -> bool:
    """Return whether two paths name the same leaf on their filesystem."""

    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    if first == second:
        return True
    try:
        if first.exists() and second.exists() and first.samefile(second):
            return True
    except OSError:
        pass
    if not (
        _filesystem_is_case_insensitive(first)
        or _filesystem_is_case_insensitive(second)
    ):
        return False
    return _casefolded_parts(first) == _casefolded_parts(second)


def is_multiply_linked_regular_file(path: Path) -> bool:
    """Return whether a mutation target has an unknowable hard-link owner."""

    try:
        metadata = path.resolve(strict=False).stat()
    except FileNotFoundError:
        return False
    except (OSError, RuntimeError):
        # A mutation boundary must not convert an uninspectable existing path
        # into permission to overwrite a possibly shared inode.
        return True
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1


__all__ = [
    "is_multiply_linked_regular_file",
    "paths_equal_by_filesystem_identity",
    "paths_overlap_by_filesystem_identity",
]
