"""The ecosystem index keeps Talon outside agent hierarchy (#3140)."""

from pathlib import Path


def test_talon_run_orchestration_is_explicitly_outside_both_agent_axes() -> None:
    ecosystem_text = (
        Path(__file__).resolve().parents[2] / "docs" / "ECOSYSTEM.md"
    ).read_text(encoding="utf-8")
    ecosystem = " ".join(ecosystem_text.split())

    required = {
        "Repository-run orchestration is not agent hierarchy.",
        "`kestrel-talon` coordinates issue-processing runs",
        "may publish a self-asserted `did:pkh` instrument identity",
        "not a Kestrel agent lineage node",
        "`talon.job_complete` enters the target agent's causation chain",
        "receives a `CausationFrame`",
        "frame attributes the agent wake, not the Talon process",
        "no `spawned_by` lineage graph edge",
        "no verified parent-signed spawn mandate or durable lineage receipt",
        "gives an agent a governed control surface",
        "does not turn Talon runs into agents",
        "bridge Talon",
        "process relationships onto either agent relationship axis",
        "does not create an agent relationship",
    }
    for phrase in required:
        assert phrase in ecosystem
    assert "`spawned_by` authority" not in ecosystem
    assert "no agent DID" not in ecosystem
    assert "outside agent causation and authority" not in ecosystem
