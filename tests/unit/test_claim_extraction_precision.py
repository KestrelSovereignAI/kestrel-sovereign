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
