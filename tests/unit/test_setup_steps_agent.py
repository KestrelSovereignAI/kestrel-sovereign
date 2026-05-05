"""Unit tests for the agent step.

These mock out :func:`create_kestrel_identity_async` since real inception
takes ~1s of crypto + DB writes; we only care that the step orchestrates
the call correctly and that multi_agent registration is right.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import toml

from kestrel_sovereign.multi_agent.config import (
    DEFAULT_AGENT_START_PORT,
    LocalAgentConfig,
    MULTI_AGENT_CONFIG_FILENAME,
    HostConfig,
    MultiAgentConfig,
)
from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.prompts import StubPrompter
from kestrel_sovereign.setup.steps import agent


class _FakeCreds:
    def __init__(self, did: str, db_path: str):
        self.agent_did = did
        self.db_path = db_path


def _make_ctx(tmp_path: Path, flow: Flow, *, answers=None) -> SetupContext:
    return SetupContext(
        project_dir=tmp_path,
        agent_data_root=tmp_path / "agent_data",
        flow=flow,
        prompter=StubPrompter(answers=list(answers or [])),
    )


def _fake_inception_factory(did: str = "did:pkh:eip155:1:0xFakeFAKEfake"):
    """Build an async mock that creates a stub kestrel_prime.db and returns creds."""

    async def _inception(*, output_dir: str, agent_name: str, **_kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        db = out / "kestrel_prime.db"
        db.write_bytes(b"")
        return _FakeCreds(did=did, db_path=str(db))

    return _inception


def test_agent_quickstart_creates_kestrel_when_multi_agent_empty(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_fake_inception_factory(),
    ):
        agent.run(ctx)

    multi_agent = MultiAgentConfig.load(tmp_path / MULTI_AGENT_CONFIG_FILENAME)
    assert "Kestrel" in multi_agent.get_local_agents()
    cfg = multi_agent.get_local_agents()["Kestrel"]
    assert cfg.port == DEFAULT_AGENT_START_PORT
    assert cfg.autostart is True


def test_agent_quickstart_skips_when_multi_agent_has_agent(tmp_path):
    multi_agent = MultiAgentConfig(
        host=HostConfig(),
        agents={
            "Existing": LocalAgentConfig(
                data_dir=Path("agent_data/existing"), port=8801, autostart=True
            )
        },
    )
    multi_agent.save(tmp_path / MULTI_AGENT_CONFIG_FILENAME)
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)

    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async"
    ) as mock_inc:
        agent.run(ctx)
        mock_inc.assert_not_called()

    # Agent map unchanged.
    multi_agent_after = MultiAgentConfig.load(tmp_path / MULTI_AGENT_CONFIG_FILENAME)
    assert list(multi_agent_after.get_local_agents()) == ["Existing"]


def test_agent_interactive_with_existing_can_decline_more(tmp_path):
    multi_agent = MultiAgentConfig(
        host=HostConfig(),
        agents={
            "First": LocalAgentConfig(
                data_dir=Path("agent_data/first"), port=8801, autostart=True
            )
        },
    )
    multi_agent.save(tmp_path / MULTI_AGENT_CONFIG_FILENAME)
    # Only one bool answer needed: "Add another?" → False
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=[False])

    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async"
    ) as mock_inc:
        agent.run(ctx)
        mock_inc.assert_not_called()


def test_agent_interactive_creates_with_custom_name(tmp_path):
    answers = [
        False,  # Don't keep existing? (no existing, this answer ignored — but OK to omit)
    ]
    # No existing multi_agent → goes straight to name prompt
    ctx = _make_ctx(
        tmp_path, Flow.INTERACTIVE, answers=["Falcon", True]
    )  # name="Falcon", autostart=True
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_fake_inception_factory("did:pkh:eip155:1:0xFalcon"),
    ):
        agent.run(ctx)

    multi_agent = MultiAgentConfig.load(tmp_path / MULTI_AGENT_CONFIG_FILENAME)
    assert "Falcon" in multi_agent.get_local_agents()


def test_agent_does_not_re_incept_existing_db(tmp_path):
    """If kestrel_prime.db already exists for that name, skip inception."""
    agent_dir = tmp_path / "agent_data" / "Kestrel"
    agent_dir.mkdir(parents=True)
    (agent_dir / "kestrel_prime.db").write_bytes(b"placeholder")

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async"
    ) as mock_inc:
        agent.run(ctx)
        mock_inc.assert_not_called()

    multi_agent = MultiAgentConfig.load(tmp_path / MULTI_AGENT_CONFIG_FILENAME)
    assert "Kestrel" in multi_agent.get_local_agents()


def test_create_agent_helper_is_idempotent(tmp_path):
    """Re-running create_agent for the same name re-registers without re-incepting."""
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_fake_inception_factory("did:test"),
    ):
        first = agent.create_agent(
            name="Solo",
            project_dir=tmp_path,
            agent_data_root=tmp_path / "agent_data",
        )
        second = agent.create_agent(
            name="Solo",
            project_dir=tmp_path,
            agent_data_root=tmp_path / "agent_data",
        )

    assert first.already_existed is False
    assert first.did == "did:test"
    assert second.already_existed is True
    assert second.did is None  # No re-inception
    assert first.port == second.port  # Same port re-allocated


def test_create_agent_avoids_port_collisions(tmp_path):
    """Allocating ports for multiple agents must give distinct ports."""
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_fake_inception_factory(),
    ):
        a = agent.create_agent(
            name="A", project_dir=tmp_path, agent_data_root=tmp_path / "agent_data"
        )
        b = agent.create_agent(
            name="B", project_dir=tmp_path, agent_data_root=tmp_path / "agent_data"
        )

    assert a.port != b.port
    multi_agent = MultiAgentConfig.load(tmp_path / MULTI_AGENT_CONFIG_FILENAME)
    ports = {cfg.port for cfg in multi_agent.get_local_agents().values()}
    assert ports == {a.port, b.port}


def test_create_agent_respects_explicit_port(tmp_path):
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_fake_inception_factory(),
    ):
        result = agent.create_agent(
            name="Explicit",
            project_dir=tmp_path,
            agent_data_root=tmp_path / "agent_data",
            port=9999,
        )
    assert result.port == 9999
