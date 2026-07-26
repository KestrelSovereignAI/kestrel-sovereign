"""Focused post-consolidation phase and dependency coverage (#2749)."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.sleep import (
    PrerequisiteFailurePolicy,
    SleepHookContract,
    SleepHookPhase,
    SleepMixin,
)


class _Agent(SleepMixin):
    def __init__(self, hooks):
        self.sleep_hooks = hooks

    async def _consolidate_memories(self):
        return {"episodes_created": 1}


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


async def _run(hooks):
    return await _Agent(hooks).sleep(skip_export=True)


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
async def test_installed_feature_hooks_block_training_after_semantic_failure():
    """The real feature wrappers honor the phase contract across packages."""
    sleep_hook_module = pytest.importorskip(
        "kestrel_feature_parametric_self.sleep_hook",
        reason="parametric-self is an optional extracted feature",
    )
    from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook

    training_feature = MagicMock()
    training_feature.on_post_consolidation = AsyncMock(
        return_value={"success": True}
    )
    training_hook = sleep_hook_module.ParametricSelfSleepHook(training_feature)
    semantic_hook = _PostHook(
        "semantic",
        [],
        contract=_contract("semantic", SleepHookPhase.SEMANTIC_MAINTENANCE),
        result={"success": False},
    )

    report = await _run([training_hook, ReflectionSleepHook(), semantic_hook])

    training_feature.on_post_consolidation.assert_not_awaited()
    training = next(
        result
        for result in report.hook_results
        if result.hook_id == "kestrel_feature_parametric_self.training"
        and result.stage == "post_consolidation"
    )
    assert training.status.value == "blocked"
    assert training.reason == (
        "required_prerequisite_not_successful:semantic:failed"
    )
