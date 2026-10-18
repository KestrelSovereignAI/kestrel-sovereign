"""Actual dispatcher constitution consumers must not normalize native control.

Agent governance/receipt/cognition hooks are explicit component inputs, not
full Agent genesis/READY or an LLM answer. Actual PG SQL and the actual retired
advisory lease produce the control original; no SQL result/Task/ACK is faked.
"""

import asyncio
import contextvars
import secrets

import pytest
from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
)

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.constitution_canary import (
    PHANTOM_RECEIPT_TOOL_NAME,
    CanaryStatus,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg

DOORS = (
    "audit-anchor",
    "audit-constitution",
    "audit-anchored",
    "audit-live",
    "governing",
    "post-anchor",
    "inline-doctrine",
    "turn-doctrine",
    "receipt-registration",
    "process-input",
    "verify",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize(
    "kind",
    ["native-control", "native-cancel", "caller-withdrawal", "ordinary", "healthy"],
)
async def test_actual_constitution_consumers_keep_original_native_control(
    native_pg, tmp_path, door, kind
):
    backend, observer, schema = native_pg
    store = SignalLogStore(backend)
    await store.initialize()
    await backend.execute("CREATE TABLE doctrine_turns (id INTEGER PRIMARY KEY)")
    async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
        lease.require_live()
    assert lease._state == "retired"
    reached, originals, tasks, process_calls = [], [], [], []
    chain = contextvars.ContextVar("component-doctrine-chain", default=None)
    chain_at_exit = []
    receipt = []
    hash_calls = 0

    async def boundary(selected):
        # Native read completes before this original lease receipt is delivered.
        assert await backend.fetch_val("SELECT 1") == 1
        if selected != door:
            return
        reached.append(selected)
        if kind == "caller-withdrawal":
            asyncio.current_task().cancel("actual governance observer withdrawn")
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as original:
                originals.append(original)
                raise
        if kind.startswith("native"):
            try:
                lease.require_live()
            except BaseException as original:
                originals.append(original)
                if kind == "native-cancel":
                    raise asyncio.CancelledError(
                        "native governance observation withdrawn"
                    ) from original
                raise
        if kind == "ordinary":
            raise OSError("ordinary governance observation unavailable")

    class GovernanceInput:
        did = "did:test:constitution-consumer"

        def __init__(self):
            self._direct_tools = {"existing-safe-tool": None}
            self._tool_to_feature = {"existing-safe-tool": "existing"}
            self._direct_tool_defs = [{"function": {"name": "existing-safe-tool"}}]
            self._constitution_receipt_expected = None
            self._constitution_receipt_tool_calls = []
            self._tool_registry_generation = 0

        async def ensure_doctrine_bundle_anchored(self):
            await boundary("audit-anchor")

        async def get_constitution_hash(self):
            nonlocal hash_calls
            hash_calls += 1
            if door == "post-anchor" and hash_calls == 1:
                return None
            await boundary(
                "post-anchor" if door == "post-anchor" else "audit-constitution"
            )
            return "a" * 64

        async def get_anchored_doctrine_bundle_hash(self):
            await boundary("audit-anchored")
            return "b" * 64

        async def compute_live_doctrine_bundle_hash(self):
            await boundary("audit-live")
            return "b" * 64

        async def _get_governing_constitution(self):
            await boundary("governing")
            return "ARTICLE I — declared component constitution"

        async def get_anchored_doctrine_files(self):
            await boundary(
                "inline-doctrine" if door == "inline-doctrine" else "turn-doctrine"
            )
            return {"AGENTS.md": "Declared component doctrine"}

        async def register_constitution_receipt_tool(self, **kwargs):
            KestrelAgent.register_constitution_receipt_tool(self, **kwargs)
            receipt.append(kwargs)
            await boundary("receipt-registration")

        def clear_constitution_receipt_tool(self):
            KestrelAgent.clear_constitution_receipt_tool(self)
            receipt.clear()

        def _set_current_chain(self, value):
            return chain.set(value)

        def _clear_current_chain(self, token):
            chain.reset(token)

        async def verify_constitution_echo(self, **kwargs):
            if door in {"receipt-registration", "process-input", "verify"}:
                if door == "verify":
                    await boundary("verify")
                return ConstitutionMixin.verify_constitution_echo(self, **kwargs)
            return CanaryStatus.VERIFIED

        def _track_background_task(self, coro, *, name):
            task = create_owned_task(coro, name=name, retain_originals=True)
            tasks.append(task)
            return task

        async def process_input(self, prompt, **kwargs):
            # Local component cognition body, NOT a fabricated provider reply.
            process_calls.append((prompt, kwargs))
            await backend.execute("INSERT INTO doctrine_turns VALUES (1)")
            if door == "process-input":
                assert PHANTOM_RECEIPT_TOOL_NAME in self._direct_tools
                assert self._constitution_receipt_expected is not None
                await boundary("process-input")
            if self._constitution_receipt_expected is not None:
                await KestrelAgent._handle_constitution_receipt_tool(
                    self,
                    canary=self._constitution_receipt_expected["canary"],
                )
            return "local component body completed"

    agent = GovernanceInput()
    registry = SourceRegistry()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=store
    )
    template = tmp_path / "native-doctrine.md"
    template.write_text("source={source} payload={payload}", encoding="utf-8")
    registration = SourceRegistration(
        name="native_doctrine",
        schema=dict,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=template,
        log_redaction=RedactionPolicy(summarize=lambda p: "<redacted>"),
        constitution_injection="full",
        prompt_template_format="local" if door == "inline-doctrine" else "claude_code",
        require_constitution_echo=door
        in {
            "inline-doctrine",
            "turn-doctrine",
            "receipt-registration",
            "process-input",
            "verify",
        },
    )
    registry.register(registration)
    signal = Signal(
        source=registration.name,
        kind="tick",
        mode=SignalMode.COGNITION,
        payload={},
        target_agent=agent.did,
    )
    caught = result = None
    try:

        async def dispatch():
            try:
                return await dispatcher.dispatch_signal(signal)
            finally:
                # Inspect the actual dispatch Task's binding, not its parent's
                # necessarily different ContextVar snapshot.
                chain_at_exit.append(chain.get())

        operation = create_owned_task(
            dispatch(), name="actual-doctrine-dispatch", retain_originals=True
        )
        outcome = await await_owned_task(operation)
        caught, result = outcome.error, outcome.result
        assert reached, (
            f"declared native constitution boundary not reached: {door} {caught!r} {result!r}"
        )
        if kind.startswith("native") or kind == "caller-withdrawal":
            assert (
                originals
                and caught is not None
                and originals[0] in tuple(_explicit_control_chain(caught))
            ), f"native constitution control normalized: {door} {caught!r} {result!r}"
            assert is_execution_control_error(caught) == kind.startswith("native")
            assert len(process_calls) == (
                1 if door in {"process-input", "verify"} else 0
            )
            assert await observer.fetchval(
                f'SELECT count(*) FROM "{schema}".doctrine_turns'
            ) == (1 if door in {"process-input", "verify"} else 0)
        else:
            assert caught is None
            assert result.status is (
                Status.DROPPED_VALIDATION
                if kind == "ordinary" and door == "governing"
                else Status.FAILED
                if kind == "ordinary"
                and door in {"receipt-registration", "process-input", "verify"}
                else Status.OK
            )
            assert len(process_calls) == (
                0
                if kind == "ordinary" and door in {"governing", "receipt-registration"}
                else 1
            )
        assert not receipt, (
            "pre-turn native refusal leaked ephemeral receipt capability"
        )
        assert PHANTOM_RECEIPT_TOOL_NAME not in agent._direct_tools
        assert PHANTOM_RECEIPT_TOOL_NAME not in agent._tool_to_feature
        assert all(
            item.get("function", {}).get("name") != PHANTOM_RECEIPT_TOOL_NAME
            for item in agent._direct_tool_defs
        )
        assert agent._constitution_receipt_expected is None
        assert "existing-safe-tool" in agent._direct_tools
        assert chain_at_exit == [None], (
            "pre-turn native refusal leaked causation binding"
        )
    finally:
        for task in tasks:
            outcome = await await_owned_task(task)
            assert outcome.error is None and outcome.cancellation is None
