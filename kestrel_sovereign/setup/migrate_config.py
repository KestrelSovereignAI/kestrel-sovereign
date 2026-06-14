"""Migrate legacy model config files into ``kestrel.toml``.

Folds:

  - ``model_mandate.toml`` -> ``[llm.mandate]``
  - ``model_catalog.toml`` -> ``[llm.catalog]``

The migration is additive and idempotent: existing unified sections win,
unrelated ``kestrel.toml`` content is preserved, and the standard TOML
writer takes a backup before changing an existing file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .migrate_llm_config import _SourceParseError, _read_source_strict
from .toml_file import read_toml, write_toml

Action = Literal["migrated", "already_clean", "no_source", "parse_error"]


@dataclass(frozen=True)
class SourceMigration:
    source_path: Path
    section_path: str
    action: Action
    error: str | None = None


@dataclass(frozen=True)
class ConfigMigrationResult:
    action: Action
    kestrel_toml_path: Path
    sources: list[SourceMigration] = field(default_factory=list)
    backup_path: Path | None = None


_MIGRATIONS = {
    "model_mandate.toml": ("llm", "mandate"),
    "model_catalog.toml": ("llm", "catalog"),
}


def migrate_config(project_dir: Path) -> ConfigMigrationResult:
    """Merge legacy model config files into ``kestrel.toml``.

    Existing ``[llm.mandate]`` or ``[llm.catalog]`` sections are never
    overwritten. This keeps reruns safe and avoids silently replacing a
    user's hand-edited unified config with stale standalone content.
    """
    kestrel_toml = project_dir / "kestrel.toml"
    existing = read_toml(kestrel_toml)
    updates: dict[str, Any] = {}
    sources: list[SourceMigration] = []

    for file_name, section_parts in _MIGRATIONS.items():
        source = project_dir / file_name
        section_path = ".".join(section_parts)

        if _nested_section(existing, section_parts):
            sources.append(SourceMigration(source, section_path, "already_clean"))
            continue

        if not source.exists():
            sources.append(SourceMigration(source, section_path, "no_source"))
            continue

        try:
            source_data = _read_source_strict(source)
        except _SourceParseError as exc:
            sources.append(
                SourceMigration(source, section_path, "parse_error", error=str(exc))
            )
            continue

        _set_nested(updates, section_parts, source_data)
        sources.append(SourceMigration(source, section_path, "migrated"))

    if any(source.action == "parse_error" for source in sources):
        return ConfigMigrationResult(
            action="parse_error",
            kestrel_toml_path=kestrel_toml,
            sources=sources,
        )

    if not updates:
        action: Action = (
            "already_clean"
            if any(source.action == "already_clean" for source in sources)
            else "no_source"
        )
        return ConfigMigrationResult(
            action=action,
            kestrel_toml_path=kestrel_toml,
            sources=sources,
        )

    write_result = write_toml(kestrel_toml, updates)
    return ConfigMigrationResult(
        action="migrated",
        kestrel_toml_path=kestrel_toml,
        sources=sources,
        backup_path=write_result.backup_path,
    )


def _nested_section(data: dict[str, Any], section_parts: tuple[str, ...]) -> Any:
    current: Any = data
    for part in section_parts:
        if not isinstance(current, dict):
            return {}
        current = current.get(part, {})
        if not current:
            return {}
    return current


def _set_nested(
    data: dict[str, Any],
    section_parts: tuple[str, ...],
    value: dict[str, Any],
) -> None:
    current = data
    for part in section_parts[:-1]:
        current = current.setdefault(part, {})
    current[section_parts[-1]] = value
