"""Tests for the wizard 'payments' step.

Phase 4 of the PayerPolicy foundation work.

Coverage:
- CHECK flow: validates an existing policy, blocks on malformed.
- QUICKSTART flow: writes default policy to kestrel.toml when missing,
  no prompts.
- INTERACTIVE flow: walks each ResourceClass, only offers READY
  combinations from the support matrix, persists the result.
- The step is wired into ORDERED in the right place (after keys,
  before llm).
- Idempotent re-run with no changes does not rewrite kestrel.toml.

A FakePrompter records prompts and returns canned answers so tests
don't require an interactive terminal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import pytest

from kestrel_sdk.payer_policy import (
    PayerKind,
    PayerPolicy,
    PayerSpec,
    ResourceClass,
)

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.steps import ORDERED, payments
from kestrel_sovereign.setup.toml_file import read_toml, write_toml


# =============================================================================
# Fake prompter
# =============================================================================


@dataclass
class _FakePrompter:
    """Records prompts; returns canned answers in FIFO order.

    Each method pulls from its dedicated answer queue. If a queue is
    empty when the step prompts, the test fails loudly — that means the
    step asked something the test didn't predict.
    """

    select_answers: list[str] = field(default_factory=list)
    text_answers: list[str] = field(default_factory=list)
    confirm_answers: list[bool] = field(default_factory=list)
    info_log: list[str] = field(default_factory=list)
    asked_select: list[tuple[str, list[str]]] = field(default_factory=list)
    asked_text: list[str] = field(default_factory=list)

    def text(self, message: str, *, default: str = "") -> str:
        self.asked_text.append(message)
        if not self.text_answers:
            return default
        return self.text_answers.pop(0)

    def secret(self, message: str, *, default: str = "") -> str:
        return self.text(message, default=default)

    def confirm(self, message: str, *, default: bool = True) -> bool:
        if not self.confirm_answers:
            return default
        return self.confirm_answers.pop(0)

    def select(
        self,
        message: str,
        *,
        choices: Iterable[str],
        default: str | None = None,
    ) -> str:
        choice_list = list(choices)
        self.asked_select.append((message, choice_list))
        if not self.select_answers:
            assert default is not None, (
                f"FakePrompter ran out of answers and no default for: {message}"
            )
            return default
        return self.select_answers.pop(0)

    def info(self, message: str) -> None:
        self.info_log.append(message)


def _make_ctx(
    tmp_path: Path,
    flow: Flow,
    prompter: _FakePrompter | None = None,
) -> SetupContext:
    return SetupContext(
        project_dir=tmp_path,
        agent_data_root=tmp_path / "agent_data",
        flow=flow,
        prompter=prompter or _FakePrompter(),
    )


@pytest.fixture(autouse=True)
def _kestrel_data_key(monkeypatch) -> Iterator[None]:
    monkeypatch.setenv(
        "KESTREL_DATA_KEY",
        "test-master-key-32-bytes-fixed--",
    )
    yield


# =============================================================================
# Wiring
# =============================================================================


class TestWiring:
    def test_payments_is_in_ordered_between_keys_and_llm(self) -> None:
        names = [name for name, _ in ORDERED]
        assert "keys" in names
        assert "payments" in names
        assert "llm" in names
        # Order: keys → payments → llm. payments must follow keys (so
        # KESTREL_DATA_KEY is in place before HostKeyStorage uses it)
        # and precede llm (so per-agent OpenRouter provisioning can
        # find the master).
        assert names.index("keys") < names.index("payments")
        assert names.index("payments") < names.index("llm")


# =============================================================================
# CHECK flow
# =============================================================================


class TestCheckFlow:
    def test_check_no_section_records_default_message(
        self, tmp_path: Path
    ) -> None:
        prompter = _FakePrompter()
        ctx = _make_ctx(tmp_path, Flow.CHECK, prompter)
        payments.run(ctx)
        assert any("host_env_default" in m for m in prompter.info_log)
        assert ctx.blockers == []

    def test_check_valid_policy_passes(self, tmp_path: Path) -> None:
        write_toml(
            tmp_path / "kestrel.toml",
            {"payments": PayerPolicy.host_env_default().to_toml_section()},
        )
        prompter = _FakePrompter()
        ctx = _make_ctx(tmp_path, Flow.CHECK, prompter)
        payments.run(ctx)
        assert any("validates" in m for m in prompter.info_log)
        assert ctx.blockers == []

    def test_check_malformed_policy_blocks(self, tmp_path: Path) -> None:
        # Unknown key should fail validation (extra='forbid' on PayerSpec).
        write_toml(
            tmp_path / "kestrel.toml",
            {"payments": {"llm": {"vendor": "openrouter", "kind": "host_env",
                                   "billing_provider": "stripe"}}},
        )
        prompter = _FakePrompter()
        ctx = _make_ctx(tmp_path, Flow.CHECK, prompter)
        payments.run(ctx)
        assert ctx.blockers, "expected a blocker for malformed [payments]"


# =============================================================================
# QUICKSTART flow
# =============================================================================


class TestQuickstartFlow:
    def test_quickstart_writes_default_when_missing(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
        payments.run(ctx)
        toml = read_toml(tmp_path / "kestrel.toml")
        assert "payments" in toml
        # Default is host_env_default — every kind should be HOST_ENV.
        for slot in toml["payments"].values():
            assert slot["kind"] == "host_env"
        assert any("Wrote default PayerPolicy" in c for c in ctx.changes)

    def test_quickstart_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        write_toml(
            tmp_path / "kestrel.toml",
            {"payments": PayerPolicy.host_env_default().to_toml_section()},
        )
        original = (tmp_path / "kestrel.toml").read_text()
        ctx = _make_ctx(tmp_path, Flow.QUICKSTART)
        payments.run(ctx)
        # File content unchanged (modulo whitespace, but write_toml is idempotent).
        assert (tmp_path / "kestrel.toml").read_text() == original


# =============================================================================
# INTERACTIVE flow
# =============================================================================


class TestInteractiveFlow:
    def test_interactive_only_offers_ready_kinds(self, tmp_path: Path) -> None:
        # User accepts vendor defaults and HOST_ENV for every slot.
        prompter = _FakePrompter(
            # Per slot the wizard asks: vendor (when multiple), then kind.
            # llm has 2 vendors (openrouter, local) → asks vendor.
            # storage has 2 vendors → asks vendor.
            # compute / tools / comms have only "*" → skips vendor question.
            select_answers=[
                # llm
                "openrouter",
                "host_env — host env var (today's behavior)",
                # storage
                "lighthouse",
                "host_env — host env var (today's behavior)",
                # compute (no vendor prompt; only "*")
                "host_env — host env var (today's behavior)",
                # tools
                "host_env — host env var (today's behavior)",
                # comms
                "host_env — host env var (today's behavior)",
            ],
        )
        ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, prompter)
        payments.run(ctx)

        toml = read_toml(tmp_path / "kestrel.toml")
        # Round-trip parses cleanly.
        policy = PayerPolicy.from_toml_section(toml["payments"])
        assert policy.llm.vendor == "openrouter"
        assert policy.llm.kind is PayerKind.HOST_ENV

        # CRITICAL: every kind shown to the user came from the matrix.
        # For lighthouse storage, HOST_ENV, SELF_WALLET, and NONE should
        # appear. Delegated-master kinds remain deferred until there is a
        # payer-wallet custody/consent path for host/user/sponsor wallets.
        kind_question_pairs = [
            (msg, choices) for msg, choices in prompter.asked_select
            if "how is it paid" in msg
        ]
        assert kind_question_pairs, "wizard never asked about kind"
        # Find the storage/lighthouse one specifically.
        lighthouse_kinds = next(
            choices for msg, choices in kind_question_pairs
            if "storage/lighthouse" in msg
        )
        # The matrix says HOST_ENV + SELF_WALLET + NONE are READY.
        assert any("host_env" in c for c in lighthouse_kinds)
        assert any("self_wallet" in c for c in lighthouse_kinds)
        assert any("none" in c for c in lighthouse_kinds)
        # Must NOT offer the delegated-master kinds yet.
        assert not any("host_master_provisioned" in c for c in lighthouse_kinds)

    def test_interactive_idempotent_no_change(self, tmp_path: Path) -> None:
        # Pre-write the same default policy.
        write_toml(
            tmp_path / "kestrel.toml",
            {"payments": PayerPolicy.host_env_default().to_toml_section()},
        )
        # User accepts every default — wizard should detect no change.
        prompter = _FakePrompter(
            select_answers=[
                "openrouter", "host_env — host env var (today's behavior)",
                "lighthouse", "host_env — host env var (today's behavior)",
                "host_env — host env var (today's behavior)",
                "host_env — host env var (today's behavior)",
                "host_env — host env var (today's behavior)",
            ],
        )
        ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, prompter)
        payments.run(ctx)
        # No "Updated" change recorded; "unchanged" info logged.
        assert not any("Updated" in c for c in ctx.changes)
        assert any("unchanged" in m for m in prompter.info_log)


# =============================================================================
# Sponsor / user-master require master_did
# =============================================================================


class TestStaleKeysOnKindChange:
    """Codex Phase 4 round 1: changing a slot from a kind that uses
    master_did (USER_MASTER_PROVISIONED / SPONSOR) to one that doesn't
    (HOST_ENV / NONE) must NOT leave the old master_did sitting in the
    TOML — that produces a malformed PayerSpec on the next read.

    The fix is write_toml(deep_merge=False) on the [payments] table.
    """

    def test_kind_change_drops_master_did_from_toml(
        self, tmp_path: Path
    ) -> None:
        # Pre-write a policy with sponsor + master_did on LLM.
        sponsor_policy = PayerPolicy(
            llm=PayerSpec(
                vendor="openrouter",
                kind=PayerKind.SPONSOR,
                master_did="did:test:original-sponsor",
            ),
            storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
            compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        )
        write_toml(
            tmp_path / "kestrel.toml",
            {"payments": sponsor_policy.to_toml_section()},
        )
        # Sanity: master_did is in the file.
        before = read_toml(tmp_path / "kestrel.toml")
        assert before["payments"]["llm"]["master_did"] == "did:test:original-sponsor"

        # Now run the wizard: user changes LLM from sponsor → host_env.
        prompter = _FakePrompter(
            select_answers=[
                # llm
                "openrouter",
                "host_env — host env var (today's behavior)",
                # storage / compute / tools / comms — accept defaults.
                "lighthouse",
                "host_env — host env var (today's behavior)",
                "host_env — host env var (today's behavior)",
                "host_env — host env var (today's behavior)",
                "host_env — host env var (today's behavior)",
            ],
        )
        ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, prompter)
        payments.run(ctx)

        # Post-condition: master_did is gone from kestrel.toml.
        after = read_toml(tmp_path / "kestrel.toml")
        assert "master_did" not in after["payments"]["llm"], (
            "master_did persisted after kind change — deep-merge bug"
        )
        # And the policy parses cleanly (no host_env-with-master_did
        # combination that would fail PayerSpec validation).
        rebuilt = PayerPolicy.from_toml_section(after["payments"])
        assert rebuilt.llm.kind is PayerKind.HOST_ENV
        assert rebuilt.llm.master_did is None


class TestMasterKeyMasked:
    """Codex Phase 4 round 1: the OpenRouter master API key prompt
    must use Prompter.secret (masked) not Prompter.text (echoed).
    """

    def test_master_key_uses_secret_prompt(self, tmp_path: Path) -> None:
        # Track which prompter method got the master-key prompt.
        secret_calls: list[str] = []
        text_calls: list[str] = []

        class _Tracking:
            def text(self, message: str, *, default: str = "") -> str:
                text_calls.append(message)
                # Return defaults from a queue if needed.
                if "DID" in message or "master_did" in message.lower():
                    return ""
                return default
            def secret(self, message: str, *, default: str = "") -> str:
                secret_calls.append(message)
                return ""  # decline to enter — wizard handles gracefully
            def confirm(self, message: str, *, default: bool = True) -> bool:
                return False  # don't rotate
            def select(self, message: str, *, choices, default=None) -> str:
                # Walk select prompts; LLM gets host_master_provisioned.
                if "vendor" in message:
                    return "openrouter" if "llm" in message else "lighthouse"
                if "llm/openrouter" in message and "paid for" in message:
                    return "host_master_provisioned — host master account, child credential per agent"
                return next(
                    c for c in choices if c.startswith("host_env")
                )
            def info(self, message: str) -> None:
                pass

        ctx = SetupContext(
            project_dir=tmp_path,
            agent_data_root=tmp_path / "agent_data",
            flow=Flow.INTERACTIVE,
            prompter=_Tracking(),
        )
        payments.run(ctx)

        # The master API key prompt must have gone through secret().
        master_prompts = [m for m in secret_calls if "master API key" in m]
        assert master_prompts, (
            f"OpenRouter master key was never prompted via secret(). "
            f"text() calls: {text_calls}; secret() calls: {secret_calls}"
        )
        # And it must NOT have been prompted via text().
        text_master = [m for m in text_calls if "master API key" in m]
        assert not text_master, (
            f"OpenRouter master key was prompted via text() (echoed): {text_master}"
        )


class TestSponsorMasterDid:
    def test_user_master_provisioned_prompts_for_master_did(
        self, tmp_path: Path
    ) -> None:
        # User picks user_master_provisioned for LLM/openrouter; wizard
        # then prompts for master_did.
        prompter = _FakePrompter(
            select_answers=[
                "openrouter",
                "user_master_provisioned — user's master account",
                # storage / compute / tools / comms — accept defaults.
                "lighthouse", "host_env — host env var (today's behavior)",
                "host_env — host env var (today's behavior)",
                "host_env — host env var (today's behavior)",
                "host_env — host env var (today's behavior)",
            ],
            text_answers=["did:test:user-master-1234"],
        )
        ctx = _make_ctx(tmp_path, Flow.INTERACTIVE, prompter)
        payments.run(ctx)

        toml = read_toml(tmp_path / "kestrel.toml")
        policy = PayerPolicy.from_toml_section(toml["payments"])
        assert policy.llm.kind is PayerKind.USER_MASTER_PROVISIONED
        assert policy.llm.master_did == "did:test:user-master-1234"
        # A text prompt was actually asked for the master_did.
        assert any("master_did" in t.lower() or "DID" in t
                   for t in prompter.asked_text)
