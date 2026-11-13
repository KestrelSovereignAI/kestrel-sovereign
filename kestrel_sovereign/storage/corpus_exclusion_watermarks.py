"""The highest corpus exclusion manifest each agent's corpus builds have admitted (#3893).

The Sovereign-signed exclusion manifest
(:mod:`kestrel_sovereign.knowledge.corpus_exclusions`) is a file an agent can
replace, though not forge. Every manifest the Sovereign ever signed verifies,
so an agent could put back an older one that excluded less. Its ``sequence``
only moves forward, and this table records, per agent, the highest sequence a
governed corpus build or consumption has admitted together with that
manifest's digest:

* a lower sequence is refused: the manifest moved back;
* the same sequence with another digest is refused: two different signed
  manifests cannot share one place in the order;
* a higher sequence is admitted and becomes the record.

The record only moves forward. Every query is filtered by ``agent_id``, so a
shared backend cannot read or move another agent's record.
"""

from __future__ import annotations

from datetime import datetime, timezone

from kestrel_sovereign.knowledge.corpus import CorpusExclusionReceipt
from kestrel_sovereign.knowledge.corpus_exclusions import CorpusExclusionManifestError

TABLE = "corpus_exclusion_watermarks"


async def admit_manifest(db, *, agent_id: str, receipt: CorpusExclusionReceipt) -> None:
    """Admit ``receipt`` for ``agent_id``'s next corpus build, or refuse it.

    The conditional upsert moves the record only to a higher sequence, so
    concurrent builds under different manifests leave the highest; each then
    compares its own manifest with what the record holds.
    """
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("a corpus exclusion watermark is recorded per agent")
    if not isinstance(receipt, CorpusExclusionReceipt):
        raise TypeError("receipt must be a CorpusExclusionReceipt")
    await db.execute(
        f"INSERT INTO {TABLE} (agent_id, sequence, manifest_sha256, recorded_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT (agent_id) DO UPDATE SET "
        "sequence = excluded.sequence, "
        "manifest_sha256 = excluded.manifest_sha256, "
        "recorded_at = excluded.recorded_at "
        f"WHERE excluded.sequence > {TABLE}.sequence",
        (
            agent_id,
            receipt.sequence,
            receipt.manifest_sha256,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    row = await db.fetchone(
        f"SELECT sequence, manifest_sha256 FROM {TABLE} WHERE agent_id = ?",
        (agent_id,),
    )
    if row is None:
        raise CorpusExclusionManifestError(
            "corpus exclusion manifest refused: its sequence could not be recorded"
        )
    admitted_sequence, admitted_sha256 = int(row[0]), str(row[1])
    if admitted_sequence > receipt.sequence:
        raise CorpusExclusionManifestError(
            "corpus exclusion manifest refused: its sequence "
            f"{receipt.sequence} is lower than sequence {admitted_sequence}, "
            "which this agent's corpus builds already admitted. A manifest "
            "cannot move back; the Sovereign must sign one with a higher sequence."
        )
    if admitted_sha256 != receipt.manifest_sha256:
        raise CorpusExclusionManifestError(
            "corpus exclusion manifest refused: sequence "
            f"{receipt.sequence} was already admitted for a manifest with other "
            "content. A changed manifest needs a higher sequence."
        )


async def admitted_manifest(db, *, agent_id: str) -> tuple[str, int] | None:
    """``(manifest_sha256, sequence)`` of the manifest ``agent_id`` last admitted.

    ``None`` when nothing has been admitted. Read inside a corpus consumption's
    own transaction, so a manifest a concurrent build or consumption admitted
    after this one read the file is seen before a corpus is served.
    """
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("a corpus exclusion watermark is recorded per agent")
    row = await db.fetchone(
        f"SELECT manifest_sha256, sequence FROM {TABLE} WHERE agent_id = ?",
        (agent_id,),
    )
    return None if row is None else (str(row[0]), int(row[1]))
