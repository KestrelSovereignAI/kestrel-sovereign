"""The central doctrine keeps causation separate from authority (#3138)."""

from pathlib import Path


def test_doctrine_names_the_two_relationship_axes_and_their_boundary() -> None:
    doctrine = (
        Path(__file__).resolve().parents[2] / "docs" / "TORTOISE_DOCTRINE.md"
    ).read_text(encoding="utf-8")

    required = {
        "Causation never confers authority.",
        "`CausationFrame`",
        "`spawned_by`",
        "parent-signed durable lineage receipt",
        "unsigned attribution today",
        "not authority unless a verified receipt corroborates it",
        "`kestrel.orchestrator`",
        "A capability available to every agent is not a third axis.",
        "Peer Stop",
        "Durable Hold",
    }
    for phrase in required:
        assert phrase in doctrine
