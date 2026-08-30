"""The ecosystem index keeps Talon outside agent hierarchy (#3140)."""

from pathlib import Path


def test_talon_run_orchestration_is_explicitly_outside_both_agent_axes() -> None:
    ecosystem = (
        Path(__file__).resolve().parents[2] / "docs" / "ECOSYSTEM.md"
    ).read_text(encoding="utf-8")

    required = {
        "Repository-run orchestration is not agent hierarchy.",
        "`kestrel-talon` coordinates issue-processing runs",
        "no agent DID or `CausationFrame`",
        "no `spawned_by` lineage graph edge",
        "no verified parent-signed spawn mandate or durable lineage receipt",
        "gives an agent a governed control surface",
        "does not turn Talon runs into agents",
        "bridge Talon",
        "process relationships onto either agent relationship axis",
        "outside agent causation and authority",
    }
    for phrase in required:
        assert phrase in ecosystem
    assert "`spawned_by` authority" not in ecosystem
