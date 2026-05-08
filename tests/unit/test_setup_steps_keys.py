"""Unit tests for the keys step."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env
from kestrel_sovereign.setup.prompts import StubPrompter
from kestrel_sovereign.setup.steps import keys


def _make_ctx(tmp_path: Path, flow: Flow, *, answers=None) -> SetupContext:
    return SetupContext(
        project_dir=tmp_path,
        agent_data_root=tmp_path / "agent_data",
        flow=flow,
        prompter=StubPrompter(answers=list(answers or [])),
    )


def test_keys_generates_data_key_when_absent(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)
    env = read_env(tmp_path / ".env")
    assert "KESTREL_DATA_KEY" in env
    # The generated key must be a valid Fernet key.
    Fernet(env["KESTREL_DATA_KEY"].encode("ascii"))


def test_keys_never_regenerates_existing_data_key(tmp_path):
    """Critical: regenerating KESTREL_DATA_KEY would brick existing DBs."""
    p = tmp_path / ".env"
    original_key = Fernet.generate_key().decode("ascii")
    write_env(p, {"KESTREL_DATA_KEY": original_key})

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    env = read_env(p)
    assert env["KESTREL_DATA_KEY"] == original_key


def test_keys_quickstart_generates_api_key_too(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)
    env = read_env(tmp_path / ".env")
    assert env.get("KESTREL_API_KEY")


def test_keys_check_mode_blocks_when_data_key_missing(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    keys.run(ctx)
    assert any("KESTREL_DATA_KEY missing" in b for b in ctx.blockers)
    # Check mode never writes.
    assert not (tmp_path / ".env").exists()


def test_keys_check_mode_silent_when_data_key_present(tmp_path):
    write_env(
        tmp_path / ".env",
        {"KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii")},
    )
    ctx = _make_ctx(tmp_path, Flow.CHECK)
    keys.run(ctx)
    assert ctx.blockers == []


def test_keys_interactive_asks_about_api_key(tmp_path):
    """Interactive flow prompts a yes/no for API key generation."""
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=[True])
    keys.run(ctx)
    env = read_env(tmp_path / ".env")
    assert env.get("KESTREL_API_KEY")


def test_keys_interactive_skips_api_key_when_declined(tmp_path):
    ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, answers=[False])
    keys.run(ctx)
    env = read_env(tmp_path / ".env")
    assert "KESTREL_API_KEY" not in env


def test_keys_propagates_generated_data_key_to_os_environ(tmp_path, monkeypatch):
    """Quickstart's KESTREL_DATA_KEY must reach the live process before the
    agent step runs inception.

    Without this, ``inception_service.save_kestrel_identity`` reads
    ``os.environ["KESTREL_DATA_KEY"]``, sees nothing, and silently writes a
    plaintext PEM. The key is on disk in ``.env`` but no one in this
    process has reloaded it.
    """
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    monkeypatch.delenv("KESTREL_API_KEY", raising=False)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    env_file_value = read_env(tmp_path / ".env")["KESTREL_DATA_KEY"]
    assert os.environ.get("KESTREL_DATA_KEY") == env_file_value
    # API key (also generated in quickstart) should propagate too.
    assert os.environ.get("KESTREL_API_KEY") == read_env(tmp_path / ".env")["KESTREL_API_KEY"]


def test_quickstart_inception_writes_encrypted_pem_not_plaintext(tmp_path, monkeypatch):
    """End-to-end check: after ``keys.run``, an inception-style key save
    should land on the encrypted ``.key.enc`` path — not the plaintext
    ``.pem`` fallback.

    This is the regression guard for the bug where ``--quickstart`` wrote
    a fresh ``KESTREL_DATA_KEY`` into ``.env`` but never told ``os.environ``,
    so ``save_kestrel_identity`` saw an empty master key and silently fell
    back to plaintext PEM."""
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    monkeypatch.delenv("KESTREL_API_KEY", raising=False)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    # Now run an inception-style key save in the same process, mimicking
    # what ``agent.run`` would do later in the wizard.
    from cryptography.hazmat.primitives.asymmetric import ec

    from kestrel_sovereign.inception_service import save_kestrel_identity

    private_key = ec.generate_private_key(ec.SECP256K1())
    out_dir = tmp_path / "agent_data" / "Kestrel"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kestrel_identity(
        did_document={"id": "did:pkh:eip155:1:0xtest"},
        keys={
            "private_key_obj": private_key,
            "public_key_hex": "00" * 64,
            "address": "0xtest",
        },
        key_id="kestrel_0xtest",
        output_dir=out_dir,
    )

    encrypted = list(out_dir.glob("*.key.enc"))
    plaintext = list(out_dir.glob("*.pem"))
    assert encrypted, f"expected encrypted .key.enc; found {[p.name for p in out_dir.iterdir()]}"
    assert not plaintext, (
        f"plaintext PEM written despite quickstart generating KESTREL_DATA_KEY: "
        f"{[p.name for p in plaintext]}"
    )


def test_keys_does_not_clobber_preexisting_os_environ_value(tmp_path, monkeypatch):
    """If the operator already has ``KESTREL_DATA_KEY`` exported in their
    shell, the keys step should not overwrite it in ``os.environ`` even if
    ``.env`` was empty. (Their shell value is what inception will use, and
    overwriting silently would be surprising.)"""
    user_supplied = "user-shell-key-do-not-touch"
    monkeypatch.setenv("KESTREL_DATA_KEY", user_supplied)
    monkeypatch.delenv("KESTREL_API_KEY", raising=False)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    # The .env got a generated value (the keys step doesn't read os.environ
    # to decide whether to generate — only ``read_env``), but os.environ
    # keeps the user's pre-existing export.
    assert os.environ["KESTREL_DATA_KEY"] == user_supplied


def test_keys_idempotent_when_everything_set(tmp_path):
    """Re-running with all keys set must not produce a new backup."""
    write_env(
        tmp_path / ".env",
        {
            "KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii"),
            "KESTREL_API_KEY": "stable-key",
        },
    )
    backups_before = list(tmp_path.glob(".env.backup-*"))

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    backups_after = list(tmp_path.glob(".env.backup-*"))
    assert backups_before == backups_after
