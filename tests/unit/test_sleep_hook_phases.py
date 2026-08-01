"""Focused post-consolidation phase and dependency coverage (#2749)."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import pytest

from kestrel_sovereign.agent.sleep import (
    PrerequisiteFailurePolicy,
    SleepHookContract,
    SleepHookPhase,
    SleepMixin,
)


class _Agent(SleepMixin):
    def __init__(self, hooks, *, consolidation_result=None):
        self.sleep_hooks = hooks
        self._consolidation_result = (
            {"episodes_created": 1}
            if consolidation_result is None
            else consolidation_result
        )

    async def _consolidate_memories(self):
        return self._consolidation_result


class _PostHook:
    def __init__(
        self,
        name,
        calls,
        *,
        contract=None,
        legacy_id=None,
        result=None,
        error=None,
    ):
        self.name = name
        self.calls = calls
        self.sleep_hook_contract = contract
        if legacy_id is not None:
            self.sleep_hook_id = legacy_id
        self.result = {"success": True} if result is None else result
        self.error = error

    async def on_post_consolidation(self, agent, consolidation_result):
        self.calls.append(self.name)
        if self.error is not None:
            raise self.error
        return self.result


class _PreHook:
    def __init__(self, name, calls, *, result):
        self.name = name
        self.calls = calls
        self.result = result

    async def on_pre_sleep(self, agent):
        self.calls.append(self.name)
        return self.result


def _contract(hook_id, phase, **kwargs):
    return SleepHookContract(hook_id=hook_id, phase=phase, **kwargs)


async def _run(hooks, *, consolidation_result=None):
    return await _Agent(
        hooks,
        consolidation_result=consolidation_result,
    ).sleep(skip_export=True)


@pytest.mark.asyncio
async def test_artifact_sweep_failure_reports_failure_but_runs_other_sleep_phases():
    calls = []

    class FailingArtifactStorage:
        async def sweep_expired_governed_semantic_artifacts(self):
            calls.append("artifact_sweep")
            raise RuntimeError("private backend detail")

    class SweepFailureAgent(_Agent):
        def __init__(self):
            super().__init__([])
            self.storage = FailingArtifactStorage()

        async def _consolidate_memories(self):
            calls.append("consolidation")
            return {"episodes_created": 1}

        async def _export_sovereignty(self, storage_tier):
            del storage_tier
            calls.append("export")
            return {"cid": "cid:sweep-failure", "shards_exported": 1}

    report = await SweepFailureAgent().sleep()

    assert calls == ["artifact_sweep", "consolidation", "export"]
    assert report.episodes_created == 1
    assert report.cid == "cid:sweep-failure"
    assert report.shards_exported == 1
    assert report.success is False
    assert report.error == "semantic_artifact_expiry_sweep_failed"
    assert str(report) == "Sleep failed: semantic_artifact_expiry_sweep_failed"
    assert "private backend detail" not in report.to_dict()["error"]


@pytest.mark.asyncio
async def test_storage_without_artifact_sweep_capability_skips_it_as_unconfigured():
    class LegacyStorage:
        pass

    agent = _Agent([])
    agent.storage = LegacyStorage()

    report = await agent.sleep(skip_export=True)

    assert report.success is True
    assert report.episodes_created == 1
    assert report.error is None


class _ReflectionMemory:
    def __init__(self, db):
        self.storage = type("Storage", (), {"db": db})()
        self.marked_ids = []

    async def mark_applied(self, message_id, *, reason):
        self.marked_ids.append((message_id, reason))


class _ReflectionDb:
    def __init__(self, rows):
        self.rows = rows

    async def fetchall(self, query, params):
        return self.rows


def _reflection_agent(hooks, rows=(), *, consolidation_result=None):
    agent = _Agent(hooks, consolidation_result=consolidation_result)
    agent.agent_id = "did:test:reflection-phase"
    agent.memory_system = _ReflectionMemory(_ReflectionDb(rows))
    return agent


@pytest.mark.asyncio
async def test_annotated_hooks_are_ordered_by_phase_and_identity_not_registration():
    first_calls = []
    second_calls = []
    contracts = {
        "extract": _contract("extract", SleepHookPhase.KNOWLEDGE_EXTRACTION),
        "maintain": _contract("maintain", SleepHookPhase.SEMANTIC_MAINTENANCE),
        "prepare": _contract("prepare", SleepHookPhase.TRAINING_PREPARATION),
        "train": _contract("train", SleepHookPhase.TRAINING),
    }

    await _run([
        _PostHook("train", first_calls, contract=contracts["train"]),
        _PostHook("maintain", first_calls, contract=contracts["maintain"]),
        _PostHook("extract", first_calls, contract=contracts["extract"]),
        _PostHook("prepare", first_calls, contract=contracts["prepare"]),
    ])
    await _run([
        _PostHook("prepare", second_calls, contract=contracts["prepare"]),
        _PostHook("extract", second_calls, contract=contracts["extract"]),
        _PostHook("train", second_calls, contract=contracts["train"]),
        _PostHook("maintain", second_calls, contract=contracts["maintain"]),
    ])

    assert first_calls == ["extract", "maintain", "prepare", "train"]
    assert second_calls == first_calls


@pytest.mark.asyncio
async def test_before_dependency_reorders_hooks_within_one_phase():
    calls = []
    await _run([
        _PostHook(
            "consumer",
            calls,
            contract=_contract("consumer", SleepHookPhase.SEMANTIC_MAINTENANCE),
        ),
        _PostHook(
            "producer",
            calls,
            contract=_contract(
                "producer",
                SleepHookPhase.SEMANTIC_MAINTENANCE,
                before=("consumer",),
            ),
        ),
    ])

    assert calls == ["producer", "consumer"]


@pytest.mark.asyncio
async def test_legacy_post_hooks_keep_registration_order_and_continue_after_failure():
    calls = []
    report = await _run([
        _PostHook("first", calls, error=RuntimeError("private user content")),
        _PostHook("second", calls),
    ])

    assert calls == ["first", "second"]
    assert [result.status.value for result in report.hook_results] == [
        "failed", "success"
    ]
    assert [result.phase for result in report.hook_results] == ["legacy", "legacy"]


@pytest.mark.asyncio
async def test_legacy_registration_does_not_constrain_annotated_phase_order():
    calls = []
    await _run([
        _PostHook("legacy", calls),
        _PostHook(
            "annotated",
            calls,
            contract=_contract("annotated", SleepHookPhase.SEMANTIC_MAINTENANCE),
        ),
    ])

    assert calls == ["annotated", "legacy"]


@pytest.mark.asyncio
async def test_mixed_legacy_registration_cannot_create_a_phase_cycle():
    calls = []
    report = await _run([
        _PostHook(
            "training",
            calls,
            contract=_contract("training", SleepHookPhase.TRAINING),
        ),
        _PostHook("legacy", calls),
        _PostHook(
            "semantic",
            calls,
            contract=_contract("semantic", SleepHookPhase.SEMANTIC_MAINTENANCE),
        ),
    ])

    assert calls == ["semantic", "training", "legacy"]
    assert {result.status.value for result in report.hook_results} == {"success"}


@pytest.mark.asyncio
async def test_invalid_post_hook_result_does_not_abort_later_hooks(caplog):
    calls = []
    caplog.set_level(logging.WARNING)
    report = await _run([
        _PostHook(
            "invalid",
            calls,
            legacy_id="invalid",
            result={"success": True, "insights_generated": "private user content"},
        ),
        _PostHook(
            "later",
            calls,
            legacy_id="later",
            result={"success": True, "insights_generated": 2},
        ),
    ])

    assert calls == ["invalid", "later"]
    outcomes = {result.hook_id: result for result in report.hook_results}
    assert outcomes["invalid"].status.value == "failed"
    assert outcomes["invalid"].reason == "invalid_hook_result"
    assert outcomes["later"].status.value == "success"
    assert report.insights_generated == 2
    assert "private user content" not in caplog.text


@pytest.mark.asyncio
async def test_invalid_pre_sleep_result_does_not_abort_later_hooks(caplog):
    calls = []
    caplog.set_level(logging.WARNING)
    report = await _run([
        _PreHook(
            "invalid",
            calls,
            result={"success": True, "insights_generated": "private user content"},
        ),
        _PreHook(
            "later",
            calls,
            result={"success": True, "insights_generated": 3},
        ),
    ])

    assert calls == ["invalid", "later"]
    outcomes = [
        result for result in report.hook_results if result.stage == "pre_sleep"
    ]
    assert outcomes[0].reason == "invalid_hook_result"
    assert outcomes[1].status.value == "success"
    assert report.insights_generated == 3
    assert "private user content" not in caplog.text


@pytest.mark.asyncio
async def test_annotated_hook_can_require_success_from_stable_legacy_provider():
    calls = []
    report = await _run([
        _PostHook("legacy", calls, legacy_id="legacy-provider"),
        _PostHook(
            "consumer",
            calls,
            contract=_contract(
                "consumer",
                SleepHookPhase.SEMANTIC_MAINTENANCE,
                after="legacy-provider",
            ),
        ),
    ])

    assert calls == ["legacy", "consumer"]
    consumer = next(result for result in report.hook_results if result.hook_id == "consumer")
    assert consumer.status.value == "success"


@pytest.mark.asyncio
async def test_failed_legacy_prerequisite_blocks_annotated_consumer():
    calls = []
    report = await _run([
        _PostHook(
            "legacy",
            calls,
            legacy_id="legacy-provider",
            result={"success": False},
        ),
        _PostHook(
            "consumer",
            calls,
            contract=_contract(
                "consumer",
                SleepHookPhase.SEMANTIC_MAINTENANCE,
                after="legacy-provider",
            ),
        ),
    ])

    assert calls == ["legacy"]
    consumer = next(result for result in report.hook_results if result.hook_id == "consumer")
    assert consumer.status.value == "blocked"
    assert consumer.reason == "required_prerequisite_not_successful:legacy-provider:failed"


@pytest.mark.asyncio
async def test_cycle_and_duplicate_are_blocked_without_stopping_unrelated_hook():
    calls = []
    report = await _run([
        _PostHook(
            "a",
            calls,
            contract=_contract("a", SleepHookPhase.SEMANTIC_MAINTENANCE, after=("b",)),
        ),
        _PostHook(
            "b",
            calls,
            contract=_contract("b", SleepHookPhase.SEMANTIC_MAINTENANCE, after=("a",)),
        ),
        _PostHook(
            "duplicate-1",
            calls,
            contract=_contract("duplicate", SleepHookPhase.SEMANTIC_MAINTENANCE),
        ),
        _PostHook(
            "duplicate-2",
            calls,
            contract=_contract("duplicate", SleepHookPhase.SEMANTIC_MAINTENANCE),
        ),
        _PostHook(
            "unrelated",
            calls,
            contract=_contract("unrelated", SleepHookPhase.SEMANTIC_MAINTENANCE),
        ),
    ])

    assert calls == ["unrelated"]
    outcomes = {result.hook_id: result for result in report.hook_results}
    assert outcomes["a"].reason == "dependency_cycle"
    assert outcomes["b"].reason == "dependency_cycle"
    duplicate_results = [
        result for result in report.hook_results if result.hook_id == "duplicate"
    ]
    assert len(duplicate_results) == 2
    assert all(result.reason == "duplicate_hook_id:duplicate" for result in duplicate_results)


@pytest.mark.asyncio
async def test_duplicate_semantic_hook_blocks_later_training_phase():
    """A rejected semantic duplicate must never let training consume stale data."""
    calls = []
    report = await _run([
        _PostHook(
            "semantic-one",
            calls,
            contract=_contract("semantic", SleepHookPhase.SEMANTIC_MAINTENANCE),
        ),
        _PostHook(
            "semantic-two",
            calls,
            contract=_contract("semantic", SleepHookPhase.SEMANTIC_MAINTENANCE),
        ),
        _PostHook(
            "training",
            calls,
            contract=_contract("training", SleepHookPhase.TRAINING),
        ),
    ])

    assert calls == []
    training = next(result for result in report.hook_results if result.hook_id == "training")
    assert training.status.value == "blocked"
    assert training.reason == "required_prerequisite_not_successful:semantic:blocked"


@pytest.mark.asyncio
async def test_missing_required_dependency_blocks_but_optional_dependency_does_not():
    calls = []
    report = await _run([
        _PostHook(
            "required",
            calls,
            contract=_contract(
                "required",
                SleepHookPhase.TRAINING,
                after=("not-installed",),
            ),
        ),
        _PostHook(
            "optional",
            calls,
            contract=_contract(
                "optional",
                SleepHookPhase.TRAINING,
                optional_after=("not-installed",),
            ),
        ),
    ])

    assert calls == ["optional"]
    required = next(result for result in report.hook_results if result.hook_id == "required")
    assert required.status.value == "blocked"
    assert required.reason == "missing_required_dependency:not-installed"


@pytest.mark.asyncio
async def test_failed_required_prerequisite_blocks_consumer_while_unrelated_hook_runs():
    calls = []
    report = await _run([
        _PostHook(
            "training",
            calls,
            contract=_contract(
                "training",
                SleepHookPhase.TRAINING,
                after=("semantic",),
            ),
        ),
        _PostHook(
            "semantic",
            calls,
            contract=_contract("semantic", SleepHookPhase.SEMANTIC_MAINTENANCE),
            result={"success": False},
        ),
        _PostHook(
            "unrelated",
            calls,
            contract=_contract("unrelated", SleepHookPhase.SEMANTIC_MAINTENANCE),
        ),
    ])

    assert calls == ["semantic", "unrelated"]
    training = next(result for result in report.hook_results if result.hook_id == "training")
    assert training.status.value == "blocked"
    assert training.reason == "required_prerequisite_not_successful:semantic:failed"


@pytest.mark.asyncio
async def test_failed_semantic_maintenance_blocks_training_without_explicit_after():
    calls = []
    report = await _run([
        _PostHook(
            "preparation",
            calls,
            contract=_contract("preparation", SleepHookPhase.TRAINING_PREPARATION),
        ),
        _PostHook(
            "training",
            calls,
            contract=_contract("training", SleepHookPhase.TRAINING),
        ),
        _PostHook(
            "semantic",
            calls,
            contract=_contract("semantic", SleepHookPhase.SEMANTIC_MAINTENANCE),
            result={"success": False},
        ),
    ])

    assert calls == ["semantic"]
    blocked = {
        result.hook_id: result
        for result in report.hook_results
        if result.hook_id in {"preparation", "training"}
    }
    assert {hook_id: result.status.value for hook_id, result in blocked.items()} == {
        "preparation": "blocked",
        "training": "blocked",
    }
    assert all(
        result.reason == "required_prerequisite_not_successful:semantic:failed"
        for result in blocked.values()
    )


@pytest.mark.parametrize(
    "attribute",
    ("before", "after", "optional_before", "optional_after"),
)
def test_singleton_string_dependency_is_normalized(attribute):
    contract = _contract(
        "consumer",
        SleepHookPhase.SEMANTIC_MAINTENANCE,
        **{attribute: "provider"},
    )

    assert getattr(contract, attribute) == ("provider",)


@pytest.mark.asyncio
async def test_optional_prerequisite_continues_and_required_policy_can_skip():
    calls = []
    report = await _run([
        _PostHook(
            "optional-consumer",
            calls,
            contract=_contract(
                "optional-consumer",
                SleepHookPhase.SEMANTIC_MAINTENANCE,
                optional_after=("failed",),
            ),
        ),
        _PostHook(
            "skip-consumer",
            calls,
            contract=_contract(
                "skip-consumer",
                SleepHookPhase.SEMANTIC_MAINTENANCE,
                after=("failed",),
                prerequisite_failure_policy=PrerequisiteFailurePolicy.SKIP,
            ),
        ),
        _PostHook(
            "failed",
            calls,
            contract=_contract("failed", SleepHookPhase.SEMANTIC_MAINTENANCE),
            result={"success": False},
        ),
    ])

    assert calls == ["failed", "optional-consumer"]
    optional = next(
        result for result in report.hook_results if result.hook_id == "optional-consumer"
    )
    skipped = next(
        result for result in report.hook_results if result.hook_id == "skip-consumer"
    )
    assert optional.status.value == "success"
    assert skipped.status.value == "skipped"
    assert skipped.reason == "required_prerequisite_not_successful:failed:failed"


@pytest.mark.asyncio
async def test_hook_report_serialization_is_additive_and_content_free(caplog):
    calls = []
    caplog.set_level(logging.WARNING)
    report = await _run([
        _PostHook(
            "fails",
            calls,
            contract=_contract("fails", SleepHookPhase.SEMANTIC_MAINTENANCE),
            error=RuntimeError("do not disclose this user sentence"),
        )
    ])

    payload = report.to_dict()
    assert "reflection" in payload  # Existing JSON shape remains available.
    assert payload["reflection"]["post_reflection"] == []
    assert payload["hook_results"] == [{
        "hook_id": "fails",
        "phase": "semantic_maintenance",
        "status": "failed",
        "duration_ms": report.hook_results[0].duration_ms,
        "stage": "post_consolidation",
        "reason": "hook_exception",
    }]
    assert json.loads(json.dumps(payload)) == payload
    assert report.hook_results[0].duration_ms >= 0
    assert "do not disclose this user sentence" not in caplog.text


def test_reflection_hook_declares_knowledge_extraction_contract():
    """Memory's installed hook opts into the shared phase protocol."""
    from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook

    contract = ReflectionSleepHook.sleep_hook_contract
    assert contract.hook_id == "kestrel_sovereign.memory.reflection"
    assert contract.phase is SleepHookPhase.KNOWLEDGE_EXTRACTION


@pytest.mark.asyncio
async def test_reflection_post_boundary_allows_a_real_declared_dependent():
    """Reflection's advertised identity is a satisfiable post-stage barrier."""
    from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook

    calls = []
    semantic_hook = _PostHook(
        "semantic",
        calls,
        contract=_contract(
            "semantic",
            SleepHookPhase.SEMANTIC_MAINTENANCE,
            after=("kestrel_sovereign.memory.reflection",),
        ),
    )

    report = await _reflection_agent([semantic_hook, ReflectionSleepHook()]).sleep(
        skip_export=True
    )

    post_results = [
        result
        for result in report.hook_results
        if result.stage == "post_consolidation"
    ]
    assert calls == ["semantic"]
    assert [result.hook_id for result in post_results] == [
        "kestrel_sovereign.memory.reflection",
        "semantic",
    ]
    assert [result.status.value for result in post_results] == ["success", "success"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("consolidation_result", "expected_reason"),
    [
        ({"error": "consolidator unavailable"}, "consolidation_failed"),
        ({"skipped": True, "privacy_blocked": True}, "consolidation_skipped"),
    ],
)
async def test_unavailable_consolidation_never_dispatches_post_dependencies(
    consolidation_result,
    expected_reason,
):
    """Structured consolidation failures are not a post-hook data boundary."""
    from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook

    calls = []
    semantic_hook = _PostHook(
        "semantic",
        calls,
        contract=_contract(
            "semantic",
            SleepHookPhase.SEMANTIC_MAINTENANCE,
            after=("kestrel_sovereign.memory.reflection",),
        ),
    )

    report = await _reflection_agent(
        [semantic_hook, ReflectionSleepHook()],
        consolidation_result=consolidation_result,
    ).sleep(skip_export=True)

    pre_reflection = next(
        result
        for result in report.hook_results
        if result.hook_id == "kestrel_sovereign.memory.reflection"
        and result.stage == "pre_sleep"
    )
    post_results = [
        result
        for result in report.hook_results
        if result.stage == "post_consolidation"
    ]
    assert pre_reflection.status.value == "success"
    assert calls == []
    assert report.success is False
    assert report.error == expected_reason
    assert report.post_reflection == []
    assert {result.hook_id for result in post_results} == {
        "kestrel_sovereign.memory.reflection",
        "semantic",
    }
    assert {result.status.value for result in post_results} == {"skipped"}
    assert {result.reason for result in post_results} == {expected_reason}


@pytest.mark.asyncio
async def test_failed_reflection_pre_stage_blocks_its_post_dependent():
    """A real provider failure cannot be acknowledged by the post barrier."""
    from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook

    class _FailingReflectionSleepHook(ReflectionSleepHook):
        async def _attest_application(self, agent, *, candidate, session_context):
            raise RuntimeError("attestation failed")

    calls = []
    semantic_hook = _PostHook(
        "semantic",
        calls,
        contract=_contract(
            "semantic",
            SleepHookPhase.SEMANTIC_MAINTENANCE,
            after=("kestrel_sovereign.memory.reflection",),
        ),
    )

    recently_retrieved = datetime.now(timezone.utc).isoformat()
    report = await _reflection_agent(
        [semantic_hook, _FailingReflectionSleepHook()],
        rows=[
            (
                1,
                "assistant",
                "private memory content",
                json.dumps({"last_accessed": recently_retrieved}),
                recently_retrieved,
            )
        ],
    ).sleep(skip_export=True)

    reflection = next(
        result
        for result in report.hook_results
        if result.hook_id == "kestrel_sovereign.memory.reflection"
        and result.stage == "post_consolidation"
    )
    semantic = next(
        result for result in report.hook_results if result.hook_id == "semantic"
    )
    assert calls == []
    assert reflection.status.value == "failed"
    assert semantic.status.value == "blocked"
    assert semantic.reason == (
        "required_prerequisite_not_successful:"
        "kestrel_sovereign.memory.reflection:failed"
    )


@pytest.mark.asyncio
async def test_skipped_reflection_pre_stage_blocks_its_post_dependent():
    """An unavailable reflection prerequisite is not a successful corpus boundary."""
    from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook

    calls = []
    semantic_hook = _PostHook(
        "semantic",
        calls,
        contract=_contract(
            "semantic",
            SleepHookPhase.SEMANTIC_MAINTENANCE,
            after=("kestrel_sovereign.memory.reflection",),
        ),
    )

    report = await _run([semantic_hook, ReflectionSleepHook()])

    reflection = next(
        result
        for result in report.hook_results
        if result.hook_id == "kestrel_sovereign.memory.reflection"
        and result.stage == "post_consolidation"
    )
    semantic = next(
        result for result in report.hook_results if result.hook_id == "semantic"
    )
    assert calls == []
    assert reflection.status.value == "skipped"
    assert semantic.status.value == "blocked"
    assert semantic.reason == (
        "required_prerequisite_not_successful:"
        "kestrel_sovereign.memory.reflection:skipped"
    )
    serialized_pre_reflection = report.to_dict()["reflection"]["pre_reflection"]
    assert serialized_pre_reflection == [{
        "success": True,
        "skipped": True,
        "reason": "memory_system_unavailable",
        "insights_generated": 0,
        "candidates": 0,
        "applied_count": 0,
    }]


@pytest.mark.asyncio
async def test_overlapping_sleep_cycles_keep_reflection_attestations_cycle_scoped():
    """A scheduled and manual sleep cannot consume each other's pre-stage result."""
    from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook

    class _InterleavingAgent(_Agent):
        def __init__(self, hook):
            super().__init__([hook])
            self.agent_id = "did:test:reflection-overlap"
            self.memory_system = _ReflectionMemory(_ReflectionDb(()))
            self.first_consolidation_started = asyncio.Event()
            self.release_first_consolidation = asyncio.Event()
            self._consolidation_calls = 0

        async def _consolidate_memories(self):
            self._consolidation_calls += 1
            if self._consolidation_calls == 1:
                self.first_consolidation_started.set()
                await self.release_first_consolidation.wait()
            return {"episodes_created": 1}

    agent = _InterleavingAgent(ReflectionSleepHook())
    scheduled_sleep = asyncio.create_task(agent.sleep(skip_export=True))
    await agent.first_consolidation_started.wait()

    manual_sleep = asyncio.create_task(agent.sleep(skip_export=True))
    manual_report = await manual_sleep
    agent.release_first_consolidation.set()
    scheduled_report = await scheduled_sleep

    for report in (scheduled_report, manual_report):
        reflection_post = next(
            result
            for result in report.hook_results
            if result.hook_id == "kestrel_sovereign.memory.reflection"
            and result.stage == "post_consolidation"
        )
        assert reflection_post.status.value == "success"


@pytest.mark.asyncio
async def test_unannotated_training_consumer_remains_legacy_until_it_adopts_contract():
    """Core never identifies optional feature classes to impose phase safety.

    Parametric-self's current hook is intentionally unannotated. Its migration
    to the training phase and semantic-maintenance prerequisite belongs to its
    own feature repository; until then it remains a loadable legacy hook with
    the established continue-on-error semantics.
    """
    calls = []
    semantic_hook = _PostHook(
        "semantic",
        calls,
        contract=_contract("semantic", SleepHookPhase.SEMANTIC_MAINTENANCE),
        result={"success": False},
    )
    legacy_training_hook = _PostHook(
        "legacy-training",
        calls,
        result={"success": True},
    )

    report = await _run([legacy_training_hook, semantic_hook])

    legacy_training = next(
        result
        for result in report.hook_results
        if result.hook_id.startswith("legacy:")
        and result.stage == "post_consolidation"
    )
    assert calls == ["semantic", "legacy-training"]
    assert legacy_training.phase == "legacy"
    assert legacy_training.status.value == "success"
