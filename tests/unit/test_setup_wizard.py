"""End-to-end tests for the wizard orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env
from kestrel_sovereign.setup.prompts import StubPrompter
from kestrel_sovereign.setup.toml_file import read_toml
from kestrel_sovereign.setup.wizard import run_wizard


class _FakeCreds:
    def __init__(self, did: str = "did:pkh:eip155:1:0xFakeFAKEfake"):
        self.agent_did = did


def _stub_inception_factory():
    async def _inception(*, output_dir, agent_name, **_kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "kestrel_prime.db").write_bytes(b"")
        return _FakeCreds()

    return _inception


def _make_ctx(
    tmp_path: Path, flow: Flow, *, reset: bool = False, answers=None
) -> SetupContext:
    return SetupContext(
        project_dir=tmp_path,
        agent_data_root=tmp_path / "agent_data",
        flow=flow,
        prompter=StubPrompter(answers=list(answers or [])),
        reset=reset,
    )


def test_wizard_quickstart_full_run(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_stub_inception_factory(),
    ):
        rc = run_wizard(ctx)

    assert rc == 0, f"blockers={ctx.blockers}"
    env = read_env(tmp_path / ".env")
    config = read_toml(tmp_path / "kestrel.toml")
    assert "KESTREL_DATA_KEY" in env
    assert config["llm"]["route_priority"] == ["ollama:local"]
    # Agent created and registered
    rookery_path = tmp_path / "rookery.toml"
    assert rookery_path.exists()


def test_wizard_idempotent_second_run_is_noop(tmp_path):
    """A second quickstart run with everything already set must produce no diffs."""
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_stub_inception_factory(),
    ):
        run_wizard(_make_ctx(tmp_path, Flow.QUICKSTART))

    env_text_1 = (tmp_path / ".env").read_text()
    toml_text_1 = (tmp_path / "kestrel.toml").read_text()
    rookery_text_1 = (tmp_path / "rookery.toml").read_text()

    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_stub_inception_factory(),
    ):
        run_wizard(_make_ctx(tmp_path, Flow.QUICKSTART))

    env_text_2 = (tmp_path / ".env").read_text()
    toml_text_2 = (tmp_path / "kestrel.toml").read_text()
    rookery_text_2 = (tmp_path / "rookery.toml").read_text()

    assert env_text_1 == env_text_2
    assert toml_text_1 == toml_text_2
    assert rookery_text_1 == rookery_text_2

    # No new backups should have been created.
    assert list(tmp_path.glob(".env.backup-*")) == []
    assert list(tmp_path.glob("kestrel.toml.backup-*")) == []


def test_wizard_check_mode_never_writes(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    rc = run_wizard(ctx)
    assert rc != 0  # Empty project = blockers
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / "kestrel.toml").exists()
    assert not (tmp_path / "rookery.toml").exists()


def test_wizard_check_mode_returns_zero_when_ready(tmp_path):
    """A fully-configured project should pass --check."""
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_stub_inception_factory(),
    ):
        run_wizard(_make_ctx(tmp_path, Flow.QUICKSTART))

    rc = run_wizard(_make_ctx(tmp_path, Flow.CHECK))
    assert rc == 0


def test_wizard_check_with_reset_does_not_move_files(tmp_path):
    """`--check --reset` is forbidden combo. CLI rejects it; this guards
    direct ``run_wizard`` callers (tests / embedders) from accidentally
    moving files when --check should be read-only.

    Reproducer: before this guard, the wizard moved .env and kestrel.toml
    to .backup-<ts> *before* the check flow noticed it should not write.
    The check then reported the originals as missing — silent corruption
    of a "read-only" command.
    """
    write_env(
        tmp_path / ".env",
        {"KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii")},
    )
    (tmp_path / "kestrel.toml").write_text(
        "[llm]\nroute_priority = []\n", encoding="utf-8"
    )
    env_text_before = (tmp_path / ".env").read_text()
    toml_text_before = (tmp_path / "kestrel.toml").read_text()

    ctx = _make_ctx(tmp_path, Flow.CHECK, reset=True)
    rc = run_wizard(ctx)

    # Originals must still be there, content unchanged.
    assert (tmp_path / ".env").exists()
    assert (tmp_path / "kestrel.toml").exists()
    assert (tmp_path / ".env").read_text() == env_text_before
    assert (tmp_path / "kestrel.toml").read_text() == toml_text_before
    # No backup files created.
    assert list(tmp_path.glob(".env.backup-*")) == []
    assert list(tmp_path.glob("kestrel.toml.backup-*")) == []
    # And the wizard must have flagged the misuse as a blocker.
    assert rc != 0
    assert any("refused to reset in --check" in b for b in ctx.blockers)


def test_wizard_reset_moves_existing_files_aside(tmp_path):
    write_env(
        tmp_path / ".env",
        {"KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii")},
    )
    (tmp_path / "kestrel.toml").write_text(
        "[llm]\nroute_priority = []\n", encoding="utf-8"
    )

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART, reset=True)
    with patch(
        "kestrel_sovereign.inception_service.create_kestrel_identity_async",
        side_effect=_stub_inception_factory(),
    ):
        run_wizard(ctx)

    # Backups exist
    env_backups = list(tmp_path.glob(".env.backup-*"))
    toml_backups = list(tmp_path.glob("kestrel.toml.backup-*"))
    assert len(env_backups) >= 1
    assert len(toml_backups) >= 1
    # Originals were regenerated
    assert (tmp_path / ".env").exists()
    assert (tmp_path / "kestrel.toml").exists()


def test_wizard_only_step_runs_just_that_step(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    rc = run_wizard(ctx, only_step="keys")
    assert rc == 0  # No verify step → no blockers
    env = read_env(tmp_path / ".env")
    assert "KESTREL_DATA_KEY" in env
    # kestrel.toml should NOT have been written (LLM step skipped)
    assert not (tmp_path / "kestrel.toml").exists()


def test_wizard_unknown_step_returns_error(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    rc = run_wizard(ctx, only_step="not-a-real-step")
    assert rc == 1


def test_wizard_blockers_cause_nonzero_exit(tmp_path):
    """If verify finds anything wrong, the wizard exits non-zero."""
    # Run keys only — agent step never ran, so verify will block on missing agents.
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    rc = run_wizard(ctx)  # Full run, but no agents → blocker
    # Actually a full quickstart creates an agent, so this should pass.
    # Force a blocker by running everything except agent:
    # (Easier: re-test by removing agent registration after running)
    ctx2 = _make_ctx(tmp_path, Flow.CHECK)
    # Wipe the rookery to break verify
    (tmp_path / "rookery.toml").unlink(missing_ok=True)
    # The check run should now block on missing rookery
    rc2 = run_wizard(ctx2)
    assert rc2 != 0
