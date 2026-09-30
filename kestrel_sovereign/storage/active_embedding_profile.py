"""The embedding profile a running agent resolves, recorded for offline tools (#3420).

Vector search filters by ``embedding_profile_id``, so a row is findable only
while it is stamped with the profile the agent resolves. That resolution
depends on state an offline tool cannot fully reproduce: config, the
persisted embedding settings, the shared spaces whose parity was verified,
discovery, and the privacy mode. ``kestrel embeddings reindex`` once resolved
a different profile for the same model and rewrote every row onto it, leaving
none on the profile the agent searches (#3420).

The agent therefore records the profile it resolves in ``agent_metadata``
under :data:`ACTIVE_EMBEDDING_PROFILE_KEY`: at boot, once its embedding
config is loaded, and whenever an operator changes that config.
``embeddings reindex`` refuses to target any other profile, and
``embeddings verify`` counts the stored vectors that are not on it.

An agent that resolves no profile leaves the previous record in place: that is
still the last profile its vectors were written in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from kestrel_sovereign.storage.timestamps import utc_timestamp_parameter

ACTIVE_EMBEDDING_PROFILE_KEY = "active_embedding_profile"


class ActiveEmbeddingProfileError(ValueError):
    """A stored active-profile record is not one this module wrote."""


@dataclass(frozen=True)
class RecordedEmbeddingProfile:
    """One agent's recorded active profile."""

    agent_id: str
    profile_id: str
    provider: Optional[str]
    model: Optional[str]
    dim: Optional[int]
    recorded_at: Optional[str]

    def describe(self) -> str:
        """``provider/model dim=N``, the form ``embeddings audit`` prints."""
        return f"{self.provider}/{self.model} dim={self.dim}"


@dataclass(frozen=True)
class ActiveProfileLookup:
    """Every active-profile record in scope, and the one profile they name."""

    records: Tuple[RecordedEmbeddingProfile, ...]

    @property
    def profile_ids(self) -> Tuple[str, ...]:
        return tuple(sorted({record.profile_id for record in self.records}))

    @property
    def ambiguous(self) -> bool:
        """Agents sharing this database record different profiles."""
        return len(self.profile_ids) > 1

    @property
    def profile_id(self) -> Optional[str]:
        """The recorded profile, or ``None`` when none or several are recorded."""
        ids = self.profile_ids
        return ids[0] if len(ids) == 1 else None

    def record_for(self, profile_id: str) -> Optional[RecordedEmbeddingProfile]:
        return next(
            (r for r in self.records if r.profile_id == profile_id), None
        )


async def record_active_embedding_profile(
    db: Any, agent_id: str, embedding_service: Any
) -> Optional[str]:
    """Record the profile *embedding_service* stamps as *agent_id*'s active one.

    Returns the recorded profile id, or ``None`` (writing nothing) when there
    is no service or it cannot describe itself.
    """
    profile = embedding_service.describe() if embedding_service else None
    if profile is None:
        return None
    recorded_at = datetime.now(timezone.utc)
    value = json.dumps({
        "profile_id": profile.profile_id,
        "provider": profile.provider,
        "model": profile.model,
        "dim": profile.dim,
        "space_id": profile.space_id,
        "normalized": profile.normalized,
        "recorded_at": recorded_at.isoformat(),
    })
    await db.execute(
        "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (
            agent_id,
            ACTIVE_EMBEDDING_PROFILE_KEY,
            value,
            utc_timestamp_parameter(db.backend_type, recorded_at),
        ),
    )
    return profile.profile_id


async def load_active_embedding_profiles(
    db: Any, agent_id: Optional[str] = None
) -> ActiveProfileLookup:
    """Read the recorded active profiles, for *agent_id* or every agent.

    Raises :class:`ActiveEmbeddingProfileError` for a record without a profile
    id; a database error propagates. Neither is evidence of which profile is
    active, so a caller must not treat it as "nothing recorded".
    """
    if agent_id:
        rows = await db.fetchall(
            "SELECT agent_id, value FROM agent_metadata "
            "WHERE key = ? AND agent_id = ?",
            (ACTIVE_EMBEDDING_PROFILE_KEY, agent_id),
        )
    else:
        rows = await db.fetchall(
            "SELECT agent_id, value FROM agent_metadata WHERE key = ?",
            (ACTIVE_EMBEDDING_PROFILE_KEY,),
        )
    records = []
    for owner, raw in rows or []:
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ActiveEmbeddingProfileError(
                f"the active embedding profile recorded for {owner} is not "
                f"JSON: {exc}"
            ) from exc
        profile_id = value.get("profile_id") if isinstance(value, dict) else None
        if not isinstance(profile_id, str) or not profile_id:
            raise ActiveEmbeddingProfileError(
                f"the active embedding profile recorded for {owner} names no "
                "profile id"
            )
        dim = value.get("dim")
        records.append(RecordedEmbeddingProfile(
            agent_id=str(owner),
            profile_id=profile_id,
            provider=value.get("provider"),
            model=value.get("model"),
            dim=int(dim) if isinstance(dim, int) else None,
            recorded_at=value.get("recorded_at"),
        ))
    return ActiveProfileLookup(records=tuple(records))
