"""The ecosystem index keeps Talon outside agent hierarchy (#3140)."""

from pathlib import Path


def test_talon_run_orchestration_is_explicitly_outside_both_agent_axes() -> None:
    ecosystem = (
        Path(__file__).resolve().parents[2] / "docs" / "ECOSYSTEM.md"
    ).read_text(encoding="utf-8")

    required = {
        "Repository-run orchestration is not agent hierarchy.",
        "`kestrel-talon` coordinates issue-processing runs",
        "no agent DID, `CausationFrame`, verified parent-signed spawn",
        "does not turn Talon runs into agents",
        "outside agent causation and authority",
    }
    for phrase in required:
        assert phrase in ecosystem
