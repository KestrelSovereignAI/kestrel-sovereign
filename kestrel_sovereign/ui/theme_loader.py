"""Theme/locale label resolver for the UI theme + i18n system (epic #986).

Loads `kestrel_sovereign/themes/<theme>/<locale>.toml` and resolves it to a
flat label map. Missing keys fall back to `legacy/<locale>.toml`; missing
locales fall back to `en`. A missing theme is a hard error (404 territory),
not silently filled.

See docs/architecture/ui_theme_schema.md for the file format.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_THEME = "legacy"
DEFAULT_LOCALE = "en"

THEMES_DIR = Path(__file__).resolve().parent.parent / "themes"


class ThemeNotFoundError(LookupError):
    """Raised when the requested theme directory does not exist."""


@dataclass(frozen=True)
class ThemeBundle:
    """Resolved label set for one (theme, locale) pair.

    `labels` is the flat key→value map. `fallback_keys` lists keys that
    were resolved via the `legacy/<locale>` fallback because they were
    missing from the requested theme; the frontend can surface this to
    indicate an incomplete theme.
    """

    theme: str
    locale: str
    labels: dict[str, str]
    fallback_keys: tuple[str, ...] = field(default_factory=tuple)


def _read_theme_file(theme: str, locale: str) -> dict[str, str] | None:
    """Read one theme file and return its `[labels]` table, or None if absent."""
    path = THEMES_DIR / theme / f"{locale}.toml"
    if not path.is_file():
        return None
    with path.open("rb") as fp:
        data = tomllib.load(fp)
    labels = data.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(
            f"theme file {path} is missing or has malformed [labels] table"
        )
    return {str(k): str(v) for k, v in labels.items()}


def _theme_dir_exists(theme: str) -> bool:
    return (THEMES_DIR / theme).is_dir()


@lru_cache(maxsize=64)
def load_theme(theme: str = DEFAULT_THEME, locale: str = DEFAULT_LOCALE) -> ThemeBundle:
    """Resolve the label map for (theme, locale) with fallbacks.

    Resolution order:
      1. requested theme + requested locale
      2. requested theme + DEFAULT_LOCALE  (if requested locale missing)
      3. DEFAULT_THEME + chosen locale     (per-key fill for missing keys)

    A missing theme directory raises `ThemeNotFoundError` — there is no
    fallback for unknown themes because that would mask configuration
    errors silently.

    Cached: themes are static files and the loader is a hot path on UI
    boot. Restart to pick up theme-file edits.
    """
    if not _theme_dir_exists(theme):
        raise ThemeNotFoundError(f"theme not found: {theme!r}")

    requested_labels = _read_theme_file(theme, locale)
    effective_locale = locale
    if requested_labels is None:
        if locale != DEFAULT_LOCALE:
            logger.warning(
                "theme.locale_fallback theme=%s requested_locale=%s effective_locale=%s",
                theme, locale, DEFAULT_LOCALE,
            )
            requested_labels = _read_theme_file(theme, DEFAULT_LOCALE)
            effective_locale = DEFAULT_LOCALE
        if requested_labels is None:
            raise ThemeNotFoundError(
                f"theme {theme!r} has no file for locale {locale!r} or {DEFAULT_LOCALE!r}"
            )

    if theme == DEFAULT_THEME:
        return ThemeBundle(
            theme=DEFAULT_THEME,
            locale=effective_locale,
            labels=requested_labels,
            fallback_keys=(),
        )

    legacy_labels = _read_theme_file(DEFAULT_THEME, effective_locale)
    if legacy_labels is None:
        legacy_labels = _read_theme_file(DEFAULT_THEME, DEFAULT_LOCALE) or {}

    merged = dict(legacy_labels)
    merged.update(requested_labels)

    fallback_keys = tuple(sorted(set(legacy_labels) - set(requested_labels)))
    if fallback_keys:
        logger.warning(
            "theme.key_fallback theme=%s locale=%s fallback_count=%d sample=%s",
            theme, effective_locale, len(fallback_keys), fallback_keys[:5],
        )

    return ThemeBundle(
        theme=theme,
        locale=effective_locale,
        labels=merged,
        fallback_keys=fallback_keys,
    )


def list_available_themes() -> list[str]:
    """Return the list of theme names that have a directory under THEMES_DIR."""
    if not THEMES_DIR.is_dir():
        return []
    return sorted(
        p.name for p in THEMES_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def clear_cache() -> None:
    """Drop the load_theme LRU cache. Tests use this; production restarts."""
    load_theme.cache_clear()
