"""One-shot migration: ``llm_config.toml`` -> ``kestrel.toml [llm]``.

Phase 1 of #938. Additive and reversible: never deletes the source,
only renames it to ``llm_config.toml.bak``. Re-uses the merge-writer
in :mod:`kestrel_sovereign.setup.toml_file` so the contract matches
what the setup wizard already writes.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import toml

from .toml_file import read_toml, write_toml

Action = Literal[
    "migrated", "no_source", "diverged", "already_clean", "parse_error",
]


@dataclass(frozen=True)
class MigrationResult:
    action: Action
    kestrel_toml_path: Path
    source_path: Path
    bak_path: Path | None = None
    backup_path: Path | None = None
    diff: str | None = None
    error: str | None = None


def migrate_llm_config(
    project_dir: Path,
    *,
    force: bool = False,
) -> MigrationResult:
    """Merge a standalone ``llm_config.toml`` into ``kestrel.toml [llm]``.

    Outcomes:
        ``no_source`` — No standalone file present. Exit 0; nothing to do.
        ``parse_error`` — Source file is malformed TOML. The source is
            **preserved untouched** so the user can fix it; ``kestrel.toml``
            is also untouched. ``error`` carries the parser message.
        ``already_clean`` — Source content already lives in ``[llm]``
            byte-for-byte. Source is renamed to ``.bak``; ``kestrel.toml``
            is left untouched (no backup taken).
        ``migrated`` — Source merged into ``kestrel.toml [llm]``; source
            renamed to ``.bak``; previous ``kestrel.toml`` (if any)
            backed up via the standard ``.backup-<timestamp>`` rule.
        ``diverged`` — ``kestrel.toml`` already has an ``[llm]`` table
            whose contents differ from the source. Without ``force``, the
            file is left alone and a unified diff is returned. With
            ``force``, the source wins via deep-merge.
    """
    source = project_dir / "llm_config.toml"
    kestrel_toml = project_dir / "kestrel.toml"

    if not source.exists():
        return MigrationResult(
            action="no_source",
            kestrel_toml_path=kestrel_toml,
            source_path=source,
        )

    # Strict parse on the source. read_toml() tolerates malformed TOML by
    # returning {}, which is right for runtime config (degrade gracefully
    # rather than refuse to boot) but wrong for a migration: an empty dict
    # would make a corrupted source look identical to a missing [llm]
    # section, the source would get renamed to .bak with a "success"
    # message, and the user's only LLM config would silently vanish.
    try:
        source_data = _read_source_strict(source)
    except _SourceParseError as exc:
        return MigrationResult(
            action="parse_error",
            kestrel_toml_path=kestrel_toml,
            source_path=source,
            error=str(exc),
        )

    existing = read_toml(kestrel_toml)
    existing_llm = existing.get("llm", {}) if isinstance(existing, dict) else {}

    if existing_llm and existing_llm != source_data and not force:
        return MigrationResult(
            action="diverged",
            kestrel_toml_path=kestrel_toml,
            source_path=source,
            diff=_render_diff(existing_llm, source_data),
        )

    if existing_llm == source_data:
        bak_path = _rename_to_bak(source)
        return MigrationResult(
            action="already_clean",
            kestrel_toml_path=kestrel_toml,
            source_path=source,
            bak_path=bak_path,
        )

    write_result = write_toml(kestrel_toml, {"llm": source_data})
    bak_path = _rename_to_bak(source)
    return MigrationResult(
        action="migrated",
        kestrel_toml_path=kestrel_toml,
        source_path=source,
        bak_path=bak_path,
        backup_path=write_result.backup_path,
    )


class _SourceParseError(Exception):
    """Raised when llm_config.toml exists but cannot be parsed as TOML."""


def _read_source_strict(path: Path) -> dict[str, Any]:
    """Parse ``path`` as TOML and raise on failure.

    Distinct from :func:`read_toml`, which is deliberately tolerant for
    runtime config loading. The migration tool needs to know the
    difference between *empty* and *broken*.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _SourceParseError(f"cannot read {path}: {exc}") from exc
    try:
        return toml.loads(text)
    except toml.TomlDecodeError as exc:
        raise _SourceParseError(str(exc)) from exc


def _rename_to_bak(source: Path) -> Path:
    """Rename ``llm_config.toml`` -> ``llm_config.toml.bak``.

    If a ``.bak`` already exists (rerun after a prior force), append a
    timestamp so we never clobber a previous backup.
    """
    bak = source.with_name(f"{source.name}.bak")
    if bak.exists():
        import time
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = source.with_name(f"{source.name}.bak.{ts}")
    source.rename(bak)
    return bak


def _render_diff(existing_llm: dict, source_data: dict) -> str:
    import toml

    a = toml.dumps(existing_llm).splitlines(keepends=True)
    b = toml.dumps(source_data).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            a, b,
            fromfile="kestrel.toml [llm]",
            tofile="llm_config.toml",
            n=3,
        )
    )
