"""Action items and decisions must be commitments, not conversation (#2852).

The two action items that existed in production were both false positives:

    "be precise about why"
    "a claude code agent to enable your heartbeat or what"

Every extraction pattern captures with ``[^.!?\\n]+``, which stops *before*
terminal punctuation — so an interrogative is indistinguishable from a
commitment by the time the capture is made. The guard has to look at the
enclosing sentence.
"""

import pytest

from kestrel_sovereign.storage.schema_router import (
    ActionItemExtractor,
    DecisionExtractor,
    DEFAULT_CLAIM_CONFIDENCE,
    STRONG_CLAIM_CONFIDENCE,
    claim_confidence,
)


def _items(content):
    return ActionItemExtractor().extract(content)


class TestInterrogativesAreNotCommitments:
    """The production false-positive shape."""

    @pytest.mark.parametrize(
        "content",
        [
            "Do I need a claude code agent to enable your heartbeat or what?",
            "Should I will the scheduler into durable leases?",
            "Why do I have to restart the host every time?",
            "I'll do what, exactly?",
        ],
    )
    def test_question_yields_no_action_item(self, content):
        assert _items(content) == []

    def test_statement_in_the_same_shape_still_extracts(self):
        """The guard must not silence real commitments."""
        assert _items("I need to restart the host after the migration.") == [
            "restart the host after the migration"
        ]


class TestNegationIsNotCommitment:
    @pytest.mark.parametrize(
        "content",
        [
            "I won't restart the host tonight.",
            "I don't need to restart the host.",
            "I'm not going to migrate the scheduler.",
        ],
    )
    def test_negated_clause_yields_no_action_item(self, content):
        assert _items(content) == []


class TestFillerTailIsNotCommitment:
    @pytest.mark.parametrize(
        "tail", ["or what", "or something", "or whatever", "I guess", "maybe"]
    )
    def test_trailing_filler_rejects(self, tail):
        assert _items(f"I need to restart the host {tail}") == []

    def test_same_clause_without_filler_extracts(self):
        assert _items("I need to restart the host") == ["restart the host"]


class TestDecisionsUseTheSameGuard:
    def test_question_yields_no_decision(self):
        assert DecisionExtractor().extract("Have we decided to use leases?") == []

    def test_statement_yields_a_decision(self):
        assert DecisionExtractor().extract("We decided to use durable leases.") == [
            "use durable leases"
        ]


class TestConfidenceCarriesInformation:
    """Previously the literal 0.7 on every claim — the field said nothing."""

    def test_explicit_todo_is_stronger_than_a_bare_promise(self):
        strong = claim_confidence("ship the fix", source="TODO: ship the fix")
        weak = claim_confidence("ship the fix", source="I'll ship the fix")
        assert strong == STRONG_CLAIM_CONFIDENCE
        assert weak == DEFAULT_CLAIM_CONFIDENCE
        assert strong > weak

    def test_reminder_phrasing_is_strong_evidence(self):
        assert claim_confidence("remind me to call Gabi") == STRONG_CLAIM_CONFIDENCE

    def test_confidence_is_not_a_constant(self):
        values = {
            claim_confidence("x", source="TODO: x"),
            claim_confidence("x", source="I'll x"),
        }
        assert len(values) > 1, "confidence must discriminate, not be a literal"


class TestProductionFalsePositivesAreGone:
    """Regression: the exact two rows found in Emma's graph."""

    def test_heartbeat_fragment_is_rejected(self):
        content = "Do I need a claude code agent to enable your heartbeat or what?"
        assert not any("heartbeat" in i for i in _items(content))

    def test_be_precise_fragment_is_rejected(self):
        content = "Can you be precise about why? I'd like to understand."
        assert not any("precise about why" in i for i in _items(content))


# ---------------------------------------------------------------------------
# codex review r1
# ---------------------------------------------------------------------------

class TestNegationVetoIsScopedToTheCommitment:
    """A sentence-wide negation veto broke more than it fixed."""

    def test_dont_forget_to_survives_its_own_negation_token(self):
        """The reminder cue literally contains 'don't'."""
        assert _items("Don't forget to submit the report.") == ["submit the report"]

    def test_unrelated_negated_clause_does_not_veto_a_real_commitment(self):
        items = _items("I don't need to deploy, but I will restart the host.")
        assert any("restart the host" in i for i in items), items

    def test_the_negated_clause_itself_is_still_rejected(self):
        assert not any("deploy" in i for i in _items("I don't need to deploy."))


class TestConfidenceOnTheProductionPath:
    """The earlier test passed `source=` by hand and never exercised routing.

    Extraction strips the cue ("TODO: ship it" -> "ship it"), so scoring the
    capture alone can never see the strongest evidence — the persisted value
    stayed 0.7 no matter what.
    """

    def test_extractor_carries_the_matched_span(self):
        pairs = ActionItemExtractor().extract_with_evidence("TODO: ship the fix")
        assert pairs, "nothing extracted"
        text, evidence = pairs[0]
        assert text == "ship the fix"
        assert "TODO" in evidence, "cue lost before confidence can see it"

    def test_strong_cue_scores_strong_via_the_carried_evidence(self):
        pairs = ActionItemExtractor().extract_with_evidence("TODO: ship the fix")
        text, evidence = pairs[0]
        assert claim_confidence(text, evidence) == STRONG_CLAIM_CONFIDENCE

    def test_bare_promise_stays_default_via_the_carried_evidence(self):
        pairs = ActionItemExtractor().extract_with_evidence("I'll ship the fix")
        text, evidence = pairs[0]
        assert claim_confidence(text, evidence) == DEFAULT_CLAIM_CONFIDENCE

    def test_reminder_cue_scores_strong_end_to_end(self):
        pairs = ActionItemExtractor().extract_with_evidence("Remind me to call Gabi")
        text, evidence = pairs[0]
        assert claim_confidence(text, evidence) == STRONG_CLAIM_CONFIDENCE

    def test_extract_still_returns_plain_strings(self):
        """The legacy shape stays intact for existing callers."""
        assert ActionItemExtractor().extract("TODO: ship the fix") == ["ship the fix"]


@pytest.mark.asyncio
class TestPersistedConfidenceIsReal:
    """Drive the actual persistence path.

    The tests above check `extract_with_evidence` and `claim_confidence`
    separately, which a mutation proved is not enough: reverting the persist
    call to `claim_confidence(text)` left every one of them green while the
    stored value silently went back to a constant. Assert what lands on the
    node.
    """

    async def _persisted(self, content):
        from unittest.mock import AsyncMock, MagicMock

        from kestrel_sovereign.storage.schema_router import SchemaRouter

        router = SchemaRouter.__new__(SchemaRouter)
        router.agent_id = "did:test:agent"
        router.graph = MagicMock()
        router.graph.get_node = AsyncMock(return_value=None)
        router.graph.add_node = AsyncMock()
        router.graph.add_edge = AsyncMock()

        items = ActionItemExtractor().extract_with_evidence(content)
        await router._persist_action_items(items, message_id=1, epistemic=None)
        return [c.args[0] for c in router.graph.add_node.await_args_list]

    async def test_explicit_todo_persists_strong_confidence(self):
        nodes = await self._persisted("TODO: ship the fix")
        assert nodes, "nothing persisted"
        assert nodes[0].properties["confidence"] == STRONG_CLAIM_CONFIDENCE

    async def test_bare_promise_persists_default_confidence(self):
        nodes = await self._persisted("I'll ship the fix")
        assert nodes, "nothing persisted"
        assert nodes[0].properties["confidence"] == DEFAULT_CLAIM_CONFIDENCE

    async def test_persisted_confidence_is_not_a_constant(self):
        strong = (await self._persisted("TODO: ship the fix"))[0]
        weak = (await self._persisted("I'll ship the fix"))[0]
        assert (
            strong.properties["confidence"] != weak.properties["confidence"]
        ), "persisted confidence carries no information"


# ---------------------------------------------------------------------------
# codex review r2 — negation belongs to the operator, not the action
# ---------------------------------------------------------------------------

class TestNegationInTheActionIsNotNegationOfTheCommitment:
    def test_negative_condition_inside_a_real_action_survives(self):
        items = _items("I need to ensure the backup never expires.")
        assert any("backup never expires" in i for i in items), items

    def test_negated_operator_is_still_rejected(self):
        assert not any("deploy" in i for i in _items("I don't need to deploy."))

    def test_negative_decision_is_still_a_decision(self):
        """Choosing NOT to do something is an explicit decision."""
        assert DecisionExtractor().extract("We've decided not to deploy.") == [
            "not to deploy"
        ]


class TestDedupKeepsStrongestEvidence:
    def test_later_todo_upgrades_an_earlier_weak_match(self):
        pairs = ActionItemExtractor().extract_with_evidence("I will ship. TODO: ship.")
        by_text = {t: e for t, e in pairs}
        assert "ship" in by_text, by_text
        assert claim_confidence("ship", by_text["ship"]) == STRONG_CLAIM_CONFIDENCE

    def test_weak_only_stays_weak(self):
        pairs = ActionItemExtractor().extract_with_evidence("I will ship.")
        text, evidence = pairs[0]
        assert claim_confidence(text, evidence) == DEFAULT_CLAIM_CONFIDENCE


# ---------------------------------------------------------------------------
# codex review r3
# ---------------------------------------------------------------------------

class TestExplicitCuesSurviveQuestionShapedText:
    """An explicit directive is a task even when its action is a question."""

    def test_todo_with_a_question_action_survives(self):
        assert _items("TODO: determine why CI is failing?") == [
            "determine why CI is failing"
        ]

    def test_reminder_with_a_question_action_survives(self):
        items = _items("Remind me to ask whether the deploy succeeded?")
        assert any("deploy succeeded" in i for i in items), items

    def test_speculative_question_is_still_rejected(self):
        assert _items("Do I need a heartbeat agent or what?") == []


class TestNegationBeyondTheOperator:
    def test_negation_at_the_head_of_the_action_is_caught(self):
        """'I should not restart' — the capture itself starts at 'not'."""
        assert not any("restart" in i for i in _items("I should not restart."))

    def test_negation_scoping_an_inner_cue_is_caught(self):
        assert not any(
            "restart" in i for i in _items("I don't think I need to restart.")
        )

    def test_unrelated_negated_clause_still_does_not_veto(self):
        items = _items("I don't need to deploy, but I will restart the host.")
        assert any("restart the host" in i for i in items), items

    def test_negative_condition_deep_in_the_action_still_survives(self):
        items = _items("I need to ensure the backup never expires.")
        assert any("backup never expires" in i for i in items), items
