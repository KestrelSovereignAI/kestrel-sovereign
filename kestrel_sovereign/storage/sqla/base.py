"""Declarative base for sovereign-core SQLAlchemy entities.

Lives separately from the feature-pkg ``EntityBase`` in
``kestrel-feature-entities`` so sovereign-core doesn't pull a
feature-pkg dep just to host its own models.

Per-feature models that need to read sovereign-owned tables should
import the entity class directly (e.g. ``from
kestrel_sovereign.storage.sqla import SavedItem``); they don't need to
share a metadata namespace because each side manages its own
migrations.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class SovereignBase(DeclarativeBase):
    """Declarative base for tables owned by sovereign-core.

    Currently has no shared columns or mixins — sovereign-core's tables
    pre-date the SQLAlchemy story by years and each carries its own
    legacy schema (``saved_items.id`` is TEXT, no created_at/updated_at
    convention, etc.). Adding mixins here would force schema migrations
    on every existing table, so we defer that.
    """
