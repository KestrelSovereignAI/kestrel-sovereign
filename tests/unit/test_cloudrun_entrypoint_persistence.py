"""Container bootstrap persistence drift guards (#2472)."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_single_agent_entrypoint_never_incepts_durable_identity():
    script = (REPO_ROOT / "docker/cloudrun_entrypoint.sh").read_text()
    durable = script.split("durable_sovereign)", 1)[1].split(
        ";;", 1
    )[0]

    assert "custody_bundle" in durable
    assert "KESTREL_EXPECTED_DID" in durable
    assert "KESTREL_DATABASE_URL" in durable
    assert "KESTREL_HOLD_EVIDENCE_DATABASE_URL" in durable
    assert "unset KESTREL_IDENTITY_BUNDLE" in durable
    assert "create_kestrel_identity" not in durable


def test_multi_agent_entrypoint_refuses_durable_cloudrun():
    script = (REPO_ROOT / "docker/multi_agent_entrypoint.sh").read_text()

    refusal = script.index('if [ "$PERSISTENCE_MODE" = "durable_sovereign" ]')
    inception = script.index("create_kestrel_identity")
    assert refusal < inception
    assert "per agent; refusing local inception" in script
    assert "is_test_instance=True" in script
    assert "is_demo=True" in script
