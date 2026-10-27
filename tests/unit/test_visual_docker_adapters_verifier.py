"""Evidence/cleanup guard tests; only the opt-in daemon run proves isolation."""
import json

import pytest
from kestrel_sdk.resources import ResourceNotFound, ResourceOutcomeUnknown
from scripts import verify_visual_docker_adapters as verifier


@pytest.mark.asyncio
async def test_existing_evidence_is_not_reused(tmp_path):
    (tmp_path / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        await verifier.run(tmp_path, "/must-not-connect")
    assert (tmp_path / "keep").read_text() == "keep"


@pytest.mark.asyncio
async def test_unknown_build_does_not_claim_cleanup_even_if_tag_is_now_absent(tmp_path, monkeypatch):
    from kestrel_sovereign.visual_computer import docker
    async def invoke(self, arguments, **options):
        if arguments[0] == "build":
            raise ResourceOutcomeUnknown("build_unknown")
        if arguments[:2] == ["image", "inspect"]:
            raise ResourceNotFound("absent_now")
        raise AssertionError(arguments)
    monkeypatch.setattr(docker.DockerBoundary, "invoke", invoke)
    output = tmp_path / "evidence"
    with pytest.raises(ResourceOutcomeUnknown, match="build_unknown"):
        await verifier.run(output, "/fixture.sock")
    custody = json.loads((output / "custody.json").read_text())
    assert custody["cleanup_confirmed"] is False
    assert custody["outcome_unknown"] is True
