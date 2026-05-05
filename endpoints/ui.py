"""UI theme + i18n endpoint.

GET /api/ui/theme?theme=<theme>&locale=<locale>
  → JSON {theme, locale, labels: {...}, fallback_keys: [...]}

Defaults: theme=legacy, locale=en (matches what renders on `main` today).
Unknown theme → 404. Missing locale falls back to en. Missing keys in a
non-legacy theme are filled from legacy and reported in fallback_keys.

See kestrel_sovereign/ui/theme_loader.py and docs/architecture/ui_theme_schema.md.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from kestrel_sovereign.ui.theme_loader import (
    DEFAULT_LOCALE,
    DEFAULT_THEME,
    ThemeNotFoundError,
    list_available_themes,
    load_theme,
)

router = APIRouter(prefix="/api/ui", tags=["ui"])


@router.get("/theme")
async def get_theme(
    theme: str = Query(default=DEFAULT_THEME, min_length=1, max_length=64),
    locale: str = Query(default=DEFAULT_LOCALE, min_length=1, max_length=16),
) -> dict:
    """Resolve a (theme, locale) pair to its flat label map.

    The frontend calls this once on init and again whenever the user
    switches themes. Response is suitable for direct use as the
    label-key→text dictionary for `data-label-key` hydration.
    """
    try:
        bundle = load_theme(theme=theme, locale=locale)
    except ThemeNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "theme_not_found",
                "message": str(exc),
                "available_themes": list_available_themes(),
            },
        )

    return {
        "theme": bundle.theme,
        "locale": bundle.locale,
        "labels": bundle.labels,
        "fallback_keys": list(bundle.fallback_keys),
    }


@router.get("/themes")
async def list_themes() -> dict:
    """List the theme names available on this server."""
    return {"themes": list_available_themes()}
