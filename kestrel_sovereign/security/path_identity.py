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


def _filesystem_is_case_insensitive(path: Path) -> bool | None:
    """Probe the nearest existing directory without creating filesystem state.

    A mount point or Linux casefold-enabled directory may have different case
    semantics from its parent.  Changing the directory's *own* spelling tests
    the parent filesystem and can therefore grant a false case-sensitive
    result.  Probe an entry inside the directory instead.  ``None`` means no
    readable, case-bearing entry proved the directory's behavior; custody
    comparisons preserve that uncertainty and treat a case-folded match as a
    possible alias.
    """

    if os.name == "nt":
        return True
    existing = _nearest_existing_path(path)
    if existing.is_dir():
        try:
            candidates = existing.iterdir()
        except OSError:
            return None
    else:
        # An existing file's spelling is interpreted by its containing
        # directory, which is exactly the filesystem boundary being probed.
        candidates = iter((existing,))
    try:
        for candidate in candidates:
            alternate_name = _alternate_case(candidate.name)
            if alternate_name is None:
                continue
            alternate = candidate.with_name(alternate_name)
            try:
                return candidate.samefile(alternate)
            except FileNotFoundError:
                return False
            except OSError:
                continue
    except OSError:
        # Directory iteration itself can fail after it has begun.
        return None
    return None


def _combined_case_insensitivity(*results: bool | None) -> bool | None:
    """Combine probes without converting an unknown result to permission."""

    if any(result is True for result in results):
        return True
    if all(result is False for result in results):
        return False
    return None


def _normalized_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize names independently of the volume's case behavior."""

    return tuple(unicodedata.normalize("NFD", part) for part in parts)


def _parts_overlap(
    first: tuple[str, ...],
    second: tuple[str, ...],
    *,
    case_insensitive: bool | None,
) -> bool:
    """Compare two unresolved suffixes using conservative file identity."""

    first_normalized = _normalized_parts(first)
    second_normalized = _normalized_parts(second)
    shorter = min(len(first_normalized), len(second_normalized))
    if first_normalized[:shorter] == second_normalized[:shorter]:
        return True
    if case_insensitive is False:
        return False
    return tuple(part.casefold() for part in first_normalized[:shorter]) == tuple(
        part.casefold() for part in second_normalized[:shorter]
    )


def _parts_equal(
    first: tuple[str, ...],
    second: tuple[str, ...],
    *,
    case_insensitive: bool | None,
) -> bool:
    if len(first) != len(second):
        return False
    return _parts_overlap(first, second, case_insensitive=case_insensitive)


def _same_existing_path(first: Path, second: Path) -> bool:
    try:
        return first.samefile(second)
    except OSError:
        return False


def _aliased_ancestor_suffixes(
    first: Path,
    second: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], Path, Path] | None:
    """Return suffixes below physically identical existing ancestors."""

    first_ancestor = _nearest_existing_path(first)
    second_ancestor = _nearest_existing_path(second)
    if not _same_existing_path(first_ancestor, second_ancestor):
        return None
    return (
        first.relative_to(first_ancestor).parts,
        second.relative_to(second_ancestor).parts,
        first_ancestor,
        second_ancestor,
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
    if first.exists() and second.exists() and _same_existing_path(first, second):
        return True
    aliased = _aliased_ancestor_suffixes(first, second)
    if aliased is not None:
        first_suffix, second_suffix, first_ancestor, second_ancestor = aliased
        if _parts_overlap(
            first_suffix,
            second_suffix,
            case_insensitive=_combined_case_insensitivity(
                _filesystem_is_case_insensitive(first_ancestor),
                _filesystem_is_case_insensitive(second_ancestor),
            ),
        ):
            return True
    return _parts_overlap(
        first.parts,
        second.parts,
        case_insensitive=_combined_case_insensitivity(
            _filesystem_is_case_insensitive(first),
            _filesystem_is_case_insensitive(second),
        ),
    )


def paths_equal_by_filesystem_identity(first: Path, second: Path) -> bool:
    """Return whether two paths name the same leaf on their filesystem."""

    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    if first == second:
        return True
    if first.exists() and second.exists() and _same_existing_path(first, second):
        return True
    aliased = _aliased_ancestor_suffixes(first, second)
    if aliased is not None:
        first_suffix, second_suffix, first_ancestor, second_ancestor = aliased
        if _parts_equal(
            first_suffix,
            second_suffix,
            case_insensitive=_combined_case_insensitivity(
                _filesystem_is_case_insensitive(first_ancestor),
                _filesystem_is_case_insensitive(second_ancestor),
            ),
        ):
            return True
    return _parts_equal(
        first.parts,
        second.parts,
        case_insensitive=_combined_case_insensitivity(
            _filesystem_is_case_insensitive(first),
            _filesystem_is_case_insensitive(second),
        ),
    )


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
