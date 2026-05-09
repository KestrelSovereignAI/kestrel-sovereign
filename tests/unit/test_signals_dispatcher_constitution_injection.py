"""Dispatcher integration tests for constitutional injection.

Pin the wiring kestrel-sovereign#1137 chunk 1G adds:

- For sources with `constitution_injection="full"`, the dispatcher
  consults optional agent hooks (`get_constitution_hash`,
  `get_anchored_doctrine_bundle_hash`,
  `compute_live_doctrine_bundle_hash`) and stamps signal_log.
- Bundle drift (anchored vs live mismatch) → DROPPED_VALIDATION with
  `error="doctrine_bundle_drift"` and the hashes recorded.
- For `require_constitution_echo=True`, the dispatcher derives a
  canary and asks `agent.verify_constitution_echo`. MISSING flips
  the dispatch to FAILED with `error="constitution_not_received"`.
- Agents that don't expose the optional hooks fall back to safe
  defaults (NULL hashes, MISSING canary).
- ACTION/ARTIFACT signals are unaffected — the audit defaults to
  NOT_REQUIRED with all-NULL signal_log fields.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
    Trust,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.constitution_canary import CanaryStatus
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _redaction() -> RedactionPolicy:
    return RedactionPolicy(summarize=lambda p: "<redacted>")


class _AuditingAgent:
    """Stand-in agent supporting the optional constitutional-injection
    hooks. Tests configure each hook's return value explicitly so the
    dispatcher's behavior under partial / missing data is exercised."""

    def __init__(
        self,
        *,
        did: str = "agent-test",
        constitution_hash: Optional[str] = None,
        anchored_bundle_hash: Optional[str] = None,
        live_bundle_hash: Optional[str] = None,
        echo_status: Optional[CanaryStatus] = None,
        echo_raises: bool = False,
        process_input_return: object = "ok",
    ):
        self._did = did
        self._const = constitution_hash
        self._anchored = anchored_bundle_hash
        self._live = live_bundle_hash
        self._echo_status = echo_status
        self._echo_raises = echo_raises
        self.background_tasks: list[asyncio.Task] = []
        self.process_input_calls: list[str] = []
        self.process_input_return = process_input_return
        self.verify_calls: list[dict] = []

    @property
    def did(self) -> str:
        return self._did

    async def process_input(self, prompt: str):
        self.process_input_calls.append(prompt)
        return self.process_input_return

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task

    # Optional constitutional-injection hooks.
    def get_constitution_hash(self) -> Optional[str]:
        return self._const

    def get_anchored_doctrine_bundle_hash(self) -> Optional[str]:
        return self._anchored

    async def compute_live_doctrine_bundle_hash(self) -> Optional[str]:
        return self._live

    async def verify_constitution_echo(
        self,
        *,
        canary: str,
        prompt_template_format: str,
        signal_id: str,
    ) -> Optional[CanaryStatus]:
        self.verify_calls.append(
            {
                "canary": canary,
                "format": prompt_template_format,
                "signal_id": signal_id,
            }
        )
        if self._echo_raises:
            raise RuntimeError("verify boom")
        return self._echo_status


class _MinimalAgent:
    """Agent that does NOT implement the optional hooks. Exercises
    the safe-default paths: NULL hashes + MISSING canary when echo
    is required."""

    def __init__(self):
        self._did = "minimal"
        self.background_tasks: list[asyncio.Task] = []

    @property
    def did(self) -> str:
        return self._did

    async def process_input(self, prompt: str):
        return "ok"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.fixture
def template_path(tmp_path) -> Path:
    p = tmp_path / "tpl.md"
    p.write_text("source={source} kind={kind} target={target_agent} payload={payload} urgency={urgency} arrived={arrived_at}", encoding="utf-8")
    return p


async def _make_dispatcher(tmp_path, agent) -> SimpleNamespace:
    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    locks = OrderedLockManager()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=locks, store=store
    )
    return SimpleNamespace(
        dispatcher=dispatcher,
        agent=agent,
        registry=registry,
        store=store,
        backend=backend,
    )


def _signal(source: str, *, target="agent-test", mode=SignalMode.COGNITION) -> Signal:
    return Signal(
        source=source,
        kind="tick",
        mode=mode,
        payload={},
        target_agent=target,
    )


def _cognition_reg(
    template_path: Path, name: str = "cog_src", **overrides
) -> SourceRegistration:
    base = dict(
        name=name,
        schema=dict,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=template_path,
        log_redaction=_redaction(),
    )
    base.update(overrides)
    return SourceRegistration(**base)


async def _drain(env: SimpleNamespace) -> None:
    pending = [t for t in env.agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# constitution_injection="none" — the legacy COGNITION path is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_cognition_signals_skip_audit_entirely(
    tmp_path, template_path
):
    agent = _AuditingAgent(
        constitution_hash="should_not_be_recorded",
        anchored_bundle_hash="bundle_a",
        live_bundle_hash="bundle_a",
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(template_path, name="legacy_cog")
    )

    result = await env.dispatcher.dispatch_signal(_signal("legacy_cog"))
    await _drain(env)

    assert result.status == Status.OK

    # Verify signal_log was stamped with all-NULL constitutional fields:
    # legacy sources don't go through the audit at all.
    rows = await env.backend.fetch_all(
        "SELECT constitution_hash, doctrine_bundle_hash, "
        "echo_canary_status FROM signal_log"
    )
    assert rows == [(None, None, "not_required")]
    await env.backend.close()


# ---------------------------------------------------------------------------
# constitution_injection="full" — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_injection_records_hashes_when_no_drift(
    tmp_path, template_path
):
    agent = _AuditingAgent(
        constitution_hash="con_abc",
        anchored_bundle_hash="bundle_xyz",
        live_bundle_hash="bundle_xyz",
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path, name="full_cog", constitution_injection="full"
        )
    )

    result = await env.dispatcher.dispatch_signal(_signal("full_cog"))
    await _drain(env)

    assert result.status == Status.OK
    rows = await env.backend.fetch_all(
        "SELECT constitution_hash, doctrine_bundle_hash, "
        "echo_canary_status FROM signal_log"
    )
    assert rows == [("con_abc", "bundle_xyz", "not_required")]
    await env.backend.close()


# ---------------------------------------------------------------------------
# Doctrine-bundle drift detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_injection_drift_returns_dropped_validation(
    tmp_path, template_path
):
    agent = _AuditingAgent(
        constitution_hash="con_abc",
        anchored_bundle_hash="bundle_anchored",
        live_bundle_hash="bundle_DRIFTED",
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="drift_cog",
            constitution_injection="full",
        )
    )

    result = await env.dispatcher.dispatch_signal(_signal("drift_cog"))
    await _drain(env)

    assert result.status == Status.DROPPED_VALIDATION
    assert "doctrine_bundle_drift" in (result.error or "")
    # process_input must NOT have been called — the dispatcher
    # refuses the dispatch BEFORE the LLM turn.
    assert agent.process_input_calls == []

    # signal_log records the anchored hash (what was expected) so an
    # auditor sees what diverged.
    rows = await env.backend.fetch_all(
        "SELECT constitution_hash, doctrine_bundle_hash, "
        "echo_canary_status FROM signal_log"
    )
    assert rows == [("con_abc", "bundle_anchored", "not_required")]
    await env.backend.close()


@pytest.mark.asyncio
async def test_full_injection_no_drift_when_only_one_hash_resolvable(
    tmp_path, template_path
):
    """If only ONE of (anchored, live) is available the dispatcher
    cannot conclude drift — record what it has, run the dispatch."""
    agent = _AuditingAgent(
        constitution_hash="con_abc",
        anchored_bundle_hash=None,  # not anchored yet (first run)
        live_bundle_hash="bundle_live",
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="first_run_cog",
            constitution_injection="full",
        )
    )

    result = await env.dispatcher.dispatch_signal(_signal("first_run_cog"))
    await _drain(env)

    assert result.status == Status.OK
    rows = await env.backend.fetch_all(
        "SELECT doctrine_bundle_hash FROM signal_log"
    )
    # The LIVE hash gets recorded (audit reflects what would have been
    # injected).
    assert rows == [("bundle_live",)]
    await env.backend.close()


# ---------------------------------------------------------------------------
# Echo verification — require_constitution_echo=True branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_echo_required_verified_status_succeeds(
    tmp_path, template_path
):
    agent = _AuditingAgent(
        constitution_hash="con_abc",
        anchored_bundle_hash="bundle",
        live_bundle_hash="bundle",
        echo_status=CanaryStatus.VERIFIED,
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="echo_ok_cog",
            constitution_injection="full",
            require_constitution_echo=True,
            prompt_template_format="codex",
        )
    )

    result = await env.dispatcher.dispatch_signal(_signal("echo_ok_cog"))
    await _drain(env)

    assert result.status == Status.OK
    # The verifier was called exactly once with the right format.
    assert len(agent.verify_calls) == 1
    call = agent.verify_calls[0]
    assert call["format"] == "codex"
    # Canary is well-formed: 16 lowercase hex.
    assert len(call["canary"]) == 16
    assert all(c in "0123456789abcdef" for c in call["canary"])

    # Codex P1 fix: the canary must have been injected into the prompt
    # the model saw — verify it appears in the rendered prompt that
    # process_input received, AND it equals what the verifier was
    # asked about (same token in both places).
    assert len(agent.process_input_calls) == 1
    rendered_prompt = agent.process_input_calls[0]
    assert call["canary"] in rendered_prompt
    # Codex format instruction is recognizable.
    assert "constitution_canary" in rendered_prompt

    rows = await env.backend.fetch_all(
        "SELECT echo_canary_status FROM signal_log"
    )
    assert rows == [("verified",)]
    await env.backend.close()


@pytest.mark.asyncio
async def test_canary_injected_pre_dispatch_matches_verifier_input(
    tmp_path, template_path
):
    """Codex round-1 P1 regression guard: the canary token sent to
    the verifier MUST be the same one embedded in the prompt that
    `process_input` received. Otherwise the model can't satisfy a
    request it never saw."""
    agent = _AuditingAgent(
        constitution_hash="con_abc",
        anchored_bundle_hash="bundle",
        live_bundle_hash="bundle",
        echo_status=CanaryStatus.VERIFIED,
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="cross_check",
            constitution_injection="full",
            require_constitution_echo=True,
            prompt_template_format="local",
        )
    )

    await env.dispatcher.dispatch_signal(_signal("cross_check"))
    await _drain(env)

    rendered_prompt = agent.process_input_calls[0]
    canary_sent_to_verifier = agent.verify_calls[0]["canary"]
    # Same token in both places.
    assert canary_sent_to_verifier in rendered_prompt
    # Local-format instruction (JSON requirement) appears.
    assert "_canary" in rendered_prompt
    await env.backend.close()


@pytest.mark.asyncio
async def test_echo_required_missing_status_fails_dispatch(
    tmp_path, template_path
):
    agent = _AuditingAgent(
        constitution_hash="con_abc",
        anchored_bundle_hash="bundle",
        live_bundle_hash="bundle",
        echo_status=CanaryStatus.MISSING,
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="echo_missing_cog",
            constitution_injection="full",
            require_constitution_echo=True,
            prompt_template_format="codex",
        )
    )

    result = await env.dispatcher.dispatch_signal(
        _signal("echo_missing_cog")
    )
    await _drain(env)

    assert result.status == Status.FAILED
    assert result.error == "constitution_not_received"

    rows = await env.backend.fetch_all(
        "SELECT echo_canary_status FROM signal_log"
    )
    assert rows == [("missing",)]
    await env.backend.close()


@pytest.mark.asyncio
async def test_echo_required_verifier_raises_records_missing(
    tmp_path, template_path
):
    """Agent-side bug in verify_constitution_echo must not crash the
    dispatch — degrade to MISSING (the safe default)."""
    agent = _AuditingAgent(
        constitution_hash="con_abc",
        anchored_bundle_hash="bundle",
        live_bundle_hash="bundle",
        echo_raises=True,
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="echo_boom_cog",
            constitution_injection="full",
            require_constitution_echo=True,
            prompt_template_format="codex",
        )
    )

    result = await env.dispatcher.dispatch_signal(_signal("echo_boom_cog"))
    await _drain(env)

    assert result.status == Status.FAILED
    assert result.error == "constitution_not_received"
    await env.backend.close()


@pytest.mark.asyncio
async def test_echo_required_no_constitution_hash_records_missing(
    tmp_path, template_path
):
    """Without an anchored constitution we can't derive a stable
    canary; the dispatch still runs (model gets the user prompt
    without an injected directive) but the verifier is NOT called
    and signal_log records MISSING. Failing here surfaces the
    anchoring gap rather than fabricating a token the model never
    saw."""
    agent = _AuditingAgent(
        constitution_hash=None,
        anchored_bundle_hash="bundle",
        live_bundle_hash="bundle",
        echo_status=CanaryStatus.VERIFIED,  # would have been verified
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="echo_no_const_cog",
            constitution_injection="full",
            require_constitution_echo=True,
            prompt_template_format="codex",
        )
    )

    result = await env.dispatcher.dispatch_signal(
        _signal("echo_no_const_cog")
    )
    await _drain(env)

    assert result.status == Status.FAILED
    assert result.error == "constitution_not_received"
    # Verifier was NOT called because canary derivation was skipped.
    assert agent.verify_calls == []
    # process_input still ran — the dispatch didn't drop pre-LLM —
    # but the rendered prompt has NO injected canary directive.
    assert len(agent.process_input_calls) == 1
    assert "constitution_canary" not in agent.process_input_calls[0]
    await env.backend.close()


# ---------------------------------------------------------------------------
# Minimal agent fallback — no optional hooks at all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_minimal_agent_full_injection_records_nulls(
    tmp_path, template_path
):
    """Agent without get_constitution_hash / bundle hooks: the
    dispatcher records NULLs and runs the dispatch. No drift is
    detected because both sides are None."""
    agent = _MinimalAgent()
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="min_cog",
            constitution_injection="full",
        )
    )

    result = await env.dispatcher.dispatch_signal(_signal("min_cog"))
    await _drain(env)

    assert result.status == Status.OK
    rows = await env.backend.fetch_all(
        "SELECT constitution_hash, doctrine_bundle_hash, "
        "echo_canary_status FROM signal_log"
    )
    assert rows == [(None, None, "not_required")]
    await env.backend.close()


@pytest.mark.asyncio
async def test_minimal_agent_echo_required_fails(tmp_path, template_path):
    """Agent without verify_constitution_echo + echo required: the
    dispatcher cannot verify and stamps MISSING → FAILED."""
    agent = _MinimalAgent()
    env = await _make_dispatcher(tmp_path, agent)
    # Need to bypass the docstring-rationale warning by setting it
    # explicitly via the registration, not opt out — that's the path
    # operators take when they really mean it.
    env.registry.register(
        _cognition_reg(
            template_path,
            name="min_echo_cog",
            constitution_injection="full",
            require_constitution_echo=True,
            prompt_template_format="codex",  # codex requires echo
        )
    )

    result = await env.dispatcher.dispatch_signal(_signal("min_echo_cog"))
    await _drain(env)

    assert result.status == Status.FAILED
    assert result.error == "constitution_not_received"
    await env.backend.close()


# ---------------------------------------------------------------------------
# ACTION / ARTIFACT signals are unaffected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_echo_required_treats_not_required_as_failure(
    tmp_path, template_path
):
    """Codex round-3 P2: when echo IS required, a verifier that
    returns NOT_REQUIRED is a contract violation, not a pass. The
    dispatch must fail with `constitution_not_received`."""
    agent = _AuditingAgent(
        constitution_hash="con_abc",
        anchored_bundle_hash="bundle",
        live_bundle_hash="bundle",
        echo_status=CanaryStatus.NOT_REQUIRED,  # bogus answer
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="echo_bogus_cog",
            constitution_injection="full",
            require_constitution_echo=True,
            prompt_template_format="codex",
        )
    )

    result = await env.dispatcher.dispatch_signal(_signal("echo_bogus_cog"))
    await _drain(env)

    assert result.status == Status.FAILED
    assert result.error == "constitution_not_received"
    rows = await env.backend.fetch_all(
        "SELECT echo_canary_status FROM signal_log"
    )
    # Stamp reflects what the verifier said — auditor sees the
    # protocol violation rather than a stomp to MISSING.
    assert rows == [("not_required",)]
    await env.backend.close()


@pytest.mark.asyncio
async def test_audit_preserved_when_process_input_raises(
    tmp_path, template_path
):
    """Codex round-3 P2: if process_input raises after the audit was
    built, the audit fields (constitution_hash, doctrine_bundle_hash)
    must still land in signal_log so the per-dispatch forensic trail
    survives LLM/API failures."""

    class _RaisingAgent(_AuditingAgent):
        async def process_input(self, prompt: str):
            self.process_input_calls.append(prompt)
            raise RuntimeError("openai 503")

    agent = _RaisingAgent(
        constitution_hash="con_traceable",
        anchored_bundle_hash="bundle_anchored",
        live_bundle_hash="bundle_anchored",
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="will_raise_cog",
            constitution_injection="full",
        )
    )

    result = await env.dispatcher.dispatch_signal(
        _signal("will_raise_cog")
    )
    await _drain(env)

    assert result.status == Status.FAILED
    # Outer message includes the raising cause.
    assert "openai 503" in (result.error or "")

    rows = await env.backend.fetch_all(
        "SELECT constitution_hash, doctrine_bundle_hash, "
        "echo_canary_status FROM signal_log"
    )
    # Audit preserved even though process_input failed.
    assert rows == [("con_traceable", "bundle_anchored", "not_required")]
    await env.backend.close()


@pytest.mark.asyncio
async def test_full_injection_inlines_constitution_into_template(
    tmp_path,
):
    """Codex round-4 P2 mitigation: the dispatcher exposes
    `{constitution}` as a render variable for full-injection sources.
    Source authors writing codex/local templates can include it where
    they want; the canary then proves the model honored a prompt that
    actually carried the constitution. Phase 2 will move to
    system-prompt injection for claude_code; Phase 1 ships the
    user-prompt-level path."""

    class _GovernedAgent(_AuditingAgent):
        async def _get_governing_constitution(self) -> str:
            return "ARTICLE I — sovereignty over data"

    template_path = tmp_path / "tpl_with_const.md"
    template_path.write_text(
        "Reviewer prompt:\n\n{constitution}\n\nDispatch payload: {payload}",
        encoding="utf-8",
    )

    agent = _GovernedAgent(
        constitution_hash="con_abc",
        anchored_bundle_hash="bundle",
        live_bundle_hash="bundle",
        echo_status=CanaryStatus.VERIFIED,
    )
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="codex_full_inject",
            constitution_injection="full",
            require_constitution_echo=True,
            prompt_template_format="codex",
        )
    )

    await env.dispatcher.dispatch_signal(_signal("codex_full_inject"))
    await _drain(env)

    rendered = agent.process_input_calls[0]
    # Constitution body is inline in the rendered prompt (the canary
    # then verifies the model honored a constitution-bearing prompt,
    # not an empty directive).
    assert "ARTICLE I — sovereignty over data" in rendered
    # Canary directive still appended.
    assert "constitution_canary" in rendered
    await env.backend.close()


@pytest.mark.asyncio
async def test_render_omits_constitution_when_injection_is_none(tmp_path):
    """For `constitution_injection="none"` the dispatcher does NOT
    fetch the constitution. Templates referencing `{constitution}`
    get an empty string, matching legacy behavior."""

    class _GovernedAgent(_AuditingAgent):
        constitution_calls = 0

        async def _get_governing_constitution(self) -> str:
            type(self).constitution_calls += 1
            return "should not appear"

    template_path = tmp_path / "tpl.md"
    template_path.write_text(
        "X{constitution}Y", encoding="utf-8"
    )

    agent = _GovernedAgent()
    env = await _make_dispatcher(tmp_path, agent)
    env.registry.register(
        _cognition_reg(
            template_path,
            name="legacy_cog_render",
        )
    )

    await env.dispatcher.dispatch_signal(_signal("legacy_cog_render"))
    await _drain(env)

    rendered = agent.process_input_calls[0]
    assert rendered == "XY"  # placeholder substituted with empty
    assert _GovernedAgent.constitution_calls == 0
    await env.backend.close()


@pytest.mark.asyncio
async def test_constitution_mixin_get_constitution_hash_reads_node():
    """The default ConstitutionMixin hook returns
    agent_node.properties['constitution_hash'] — the trivial wiring
    that lets real agents stop silently null-stamping signal_log."""
    from unittest.mock import AsyncMock, MagicMock

    from kestrel_sovereign.agent.constitution import ConstitutionMixin

    class _Agent(ConstitutionMixin):
        def __init__(self, hash_value):
            self.agent_id = "agent-x"
            node = MagicMock()
            node.properties = {"constitution_hash": hash_value}
            self.storage = MagicMock()
            self.storage.get_node = AsyncMock(return_value=node)

    agent = _Agent("con_real_anchored")
    assert await agent.get_constitution_hash() == "con_real_anchored"

    # Missing node → None (no crash)
    class _NoNodeAgent(ConstitutionMixin):
        def __init__(self):
            self.agent_id = "missing"
            self.storage = MagicMock()
            self.storage.get_node = AsyncMock(return_value=None)

    assert await _NoNodeAgent().get_constitution_hash() is None


@pytest.mark.asyncio
async def test_constitution_mixin_compute_live_returns_none_default():
    """Phase 1: default implementation returns None. Phase 2 wires
    project_root + bootstrap loader to produce a real hash."""
    from kestrel_sovereign.agent.constitution import ConstitutionMixin

    class _Agent(ConstitutionMixin):
        pass

    assert await _Agent().compute_live_doctrine_bundle_hash() is None


@pytest.mark.asyncio
async def test_action_signals_bypass_constitutional_audit(tmp_path):
    agent = _AuditingAgent(
        constitution_hash="should_not_be_used",
        anchored_bundle_hash="bundle_a",
        live_bundle_hash="bundle_b",  # would be drift, but ACTION skips
    )
    env = await _make_dispatcher(tmp_path, agent)

    async def _h(payload):
        return {"ok": True}

    env.registry.register(
        SourceRegistration(
            name="act_src",
            schema=dict,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=_h,
            log_redaction=_redaction(),
        )
    )

    result = await env.dispatcher.dispatch_signal(
        _signal("act_src", mode=SignalMode.ACTION)
    )
    await _drain(env)

    assert result.status == Status.OK  # drift would have failed COGNITION
    rows = await env.backend.fetch_all(
        "SELECT constitution_hash, doctrine_bundle_hash, "
        "echo_canary_status FROM signal_log"
    )
    # ACTION signals leave all constitutional fields NULL — chunk 1C
    # pinned the same in `test_constitution_columns_default_to_null`
    # and the dispatcher's COGNITION-only audit preserves that.
    assert rows == [(None, None, None)]
    await env.backend.close()
