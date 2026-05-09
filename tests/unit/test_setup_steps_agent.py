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


# --- --test plumbing: wizard surfaces is_test_instance to inception ----------

def test_create_agent_propagates_is_test_instance_to_inception(tmp_path):
    """``is_test_instance=True`` reaches ``create_kestrel_identity_async``.

    Catches the bug where the wizard accepted ``--test`` but the keyword
    argument never made it past ``setup/steps/agent.py::_run_inception``.
    """
    captured: dict = {}

    async def _capturing_inception(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "kestrel_prime.db").write_bytes(b"")
        return _FakeCreds(did="did:test", db_path=str(out / "kestrel_prime.db"))

    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_capturing_inception,
    ):
        agent.create_agent(
            name="CIAgent",
            project_dir=tmp_path,
            agent_data_root=tmp_path / "agent_data",
            is_test_instance=True,
        )

    assert captured.get("is_test_instance") is True


def test_create_agent_default_is_not_test_instance(tmp_path):
    """Without an explicit ``is_test_instance``, the flag is False.

    Locks in the default. A regression here would silently mark every
    user's first ``kestrel setup`` agent as a test instance.
    """
    captured: dict = {}

    async def _capturing_inception(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "kestrel_prime.db").write_bytes(b"")
        return _FakeCreds(did="did:test", db_path=str(out / "kestrel_prime.db"))

    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_capturing_inception,
    ):
        agent.create_agent(
            name="ProdAgent",
            project_dir=tmp_path,
            agent_data_root=tmp_path / "agent_data",
        )

    assert captured.get("is_test_instance") is False


def test_wizard_run_propagates_ctx_is_test_instance(tmp_path):
    """``SetupContext.is_test_instance`` reaches inception via ``run()``.

    Verifies the full wizard chain: a context flagged as a test instance
    surfaces the flag through ``agent.run()`` → ``create_agent()`` →
    ``_run_inception()`` → ``create_kestrel_identity_async``.
    """
    captured: dict = {}

    async def _capturing_inception(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "kestrel_prime.db").write_bytes(b"")
        return _FakeCreds(did="did:test", db_path=str(out / "kestrel_prime.db"))

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    ctx.is_test_instance = True

    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_capturing_inception,
    ):
        agent.run(ctx)

    assert captured.get("is_test_instance") is True
    assert any("test instance" in change for change in ctx.changes), (
        f"expected the changes list to mention test instance; got {ctx.changes}"
    )


# --- #1109: malformed [emancipation] block aborts before inception -----------

def test_invalid_emancipation_block_blocks_run_without_inception(tmp_path):
    """A malformed [emancipation] block must surface as a wizard
    blocker AND prevent inception — never raise an unhandled exception
    that crashes the wizard."""
    (tmp_path / "kestrel.toml").write_text(
        # enabled=true but no terms — parse_emancipation_block raises
        '[emancipation]\nenabled = true\n',
        encoding="utf-8",
    )
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async"
    ) as mock_inc:
        agent.run(ctx)
        mock_inc.assert_not_called()

    assert any("[emancipation]" in b and "invalid" in b for b in ctx.blockers)
    # Wizard returned cleanly — no traceback escaped.
    assert not (tmp_path / "agent_data").exists() or not list(
        (tmp_path / "agent_data").iterdir()
    )
