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


@pytest.fixture(autouse=True)
def _clean_data_key_env(monkeypatch):
    """Start each keys-step test from a clean process environment.

    The suite-wide ``_born_hybrid_inception_env`` fixture exports a
    ``KESTREL_DATA_KEY`` so inception works elsewhere. The keys step now
    resolves key authority *from the process environment as well as*
    ``.env`` (#2468), so tests that exercise the generate / target-key paths
    must not inherit that ambient export — each test opts back in explicitly
    when it wants an exported key.
    """
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    monkeypatch.delenv("KESTREL_API_KEY", raising=False)


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


def test_keys_adopts_exported_key_into_env_no_split_brain(tmp_path, monkeypatch):
    """True exported-key semantics (#2468): if the operator has a valid
    ``KESTREL_DATA_KEY`` exported in their shell and the target ``.env`` has
    none, the keys step must persist *that same* key to the target ``.env``.

    The old behaviour generated a *different* value into ``.env`` while
    keeping the exported value effective — a split brain: inception encrypts
    with the exported key while ``.env`` persists another, so an immediate
    restart cannot decrypt the identity. There must be exactly one effective,
    persisted key.
    """
    user_supplied = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("KESTREL_DATA_KEY", user_supplied)
    monkeypatch.delenv("KESTREL_API_KEY", raising=False)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    env = read_env(tmp_path / ".env")
    # The exported key is what gets persisted — not a freshly generated one.
    assert env["KESTREL_DATA_KEY"] == user_supplied
    # ...and it stays the effective process key. Encrypt-key == persist-key.
    assert os.environ["KESTREL_DATA_KEY"] == user_supplied


def test_keys_conflict_exported_vs_target_blocks_before_inception(tmp_path, monkeypatch):
    """A persisted target key and a *different* exported key is an
    unresolvable conflict (#2468): encrypting with one while persisting the
    other is the loss-of-custody defect. The keys step must block instead of
    silently picking a winner — and never regenerate the target key."""
    target = Fernet.generate_key().decode("ascii")
    write_env(tmp_path / ".env", {"KESTREL_DATA_KEY": target})
    monkeypatch.setenv("KESTREL_DATA_KEY", "a-different-exported-value")

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    assert any("KESTREL_DATA_KEY conflict" in b for b in ctx.blockers)
    # Target key never regenerated.
    assert read_env(tmp_path / ".env")["KESTREL_DATA_KEY"] == target


def test_keys_existing_target_key_becomes_effective(tmp_path, monkeypatch):
    """An existing target ``.env`` key is authoritative and is propagated to
    ``os.environ`` so inception encrypts with exactly the persisted key —
    even if the process env started without it."""
    target = Fernet.generate_key().decode("ascii")
    write_env(tmp_path / ".env", {"KESTREL_DATA_KEY": target})
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    assert os.environ["KESTREL_DATA_KEY"] == target
    assert read_env(tmp_path / ".env")["KESTREL_DATA_KEY"] == target


def test_keys_invalid_target_key_blocks(tmp_path):
    """Corrupted persisted key material must fail before identity creation.

    A passphrase is accepted (PBKDF2), so "invalid" here means material that
    embeds whitespace/control characters — a corruption signal that would also
    break the single-line ``.env`` format.
    """
    write_env(tmp_path / ".env", {"KESTREL_DATA_KEY": "broken key with spaces"})

    ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
    keys.run(ctx)

    assert any("not a valid master key" in b for b in ctx.blockers)
    # Never regenerates a present (even if corrupted) key.
    assert read_env(tmp_path / ".env")["KESTREL_DATA_KEY"] == "broken key with spaces"


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
