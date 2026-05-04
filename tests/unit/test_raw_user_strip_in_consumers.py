"""
Regression: raw-user content consumers (personality calibration,
wellness depth metrics) must strip the sent-form wrappers from
user-role rows before measuring/exporting.

User turns are persisted in fully-rendered prompt form
(``<retrieved_context>...</retrieved_context>\\n<user_input>\\n{raw}\\n</user_input>``)
for prompt-cache stability. Consumers that treat that text as "what
the user said" will:

  - export retrieved-context blocks as personality calibration input
    (skewing the agent's identity fingerprint)
  - count the wrapper bytes as substantive engagement, inflating
    wellness depth scores when the user actually typed "hi"

Both of these have been observed in practice. Reviewer flagged the
broader strip as "use the helper everywhere raw user text matters."
"""
import pytest
from unittest.mock import AsyncMock, MagicMock


SENT_FORM = (
    "\n<retrieved_context>\n<memories>previous turn's retrieval</memories>\n"
    "</retrieved_context>\n<user_input>\nhi\n</user_input>"
)
RAW_USER = "hi"


@pytest.mark.asyncio
async def test_personality_calibration_strips_sent_form_from_user_input():
    """``_get_calibration_examples`` exports user/assistant pairs as
    the agent's personality fingerprint. Sent-form leaking into the
    user-side input would make the export look like the user is
    speaking in <retrieved_context> blocks."""
    from kestrel_sovereign.identity.personality_analyzer import PersonalityAnalyzer

    analyzer = PersonalityAnalyzer.__new__(PersonalityAnalyzer)
    analyzer.agent_id = "test-agent"
    analyzer.db = MagicMock()
    analyzer.db.fetchall = AsyncMock(return_value=[
        # Two pairs. Both user-side rows are sent-form; the export
        # must show only the raw user text.
        (SENT_FORM, "I see, here's my response."),
        (SENT_FORM, "Another reply that's long enough to count."),
    ])

    examples = await analyzer._get_calibration_examples(num_examples=10)

    assert len(examples) > 0, "should produce at least one example"
    for ex in examples:
        assert "<retrieved_context>" not in ex["input"], (
            "personality export must NOT contain retrieval wrappers"
        )
        assert "<user_input>" not in ex["input"], (
            "personality export must NOT contain user-input tags"
        )
        # Output (assistant) is unchanged because assistant rows are
        # persisted raw.
        assert "I see," in ex["output"] or "Another reply" in ex["output"]


@pytest.mark.asyncio
async def test_wellness_depth_metric_strips_user_sent_form_for_length():
    """The depth metric measures avg_length / substantive_rate /
    depth_score. A user typing "hi" but having that wrapped to a
    2000-char sent-form stored row would falsely register as a
    substantive message."""
    from kestrel_sovereign.features.wellness.metrics import (
        InteractionDepthCalculator,
    )

    db = MagicMock()
    db.table_exists = AsyncMock(return_value=True)
    # Three short user turns persisted as sent-form (~120 chars each)
    # plus one short assistant reply. Without the strip, all three
    # user rows would clear the SUBSTANTIVE_THRESHOLD; with the strip,
    # they reduce to "hi" (2 chars) and don't count as substantive.
    db.fetchall = AsyncMock(return_value=[
        ("user", SENT_FORM, None),
        ("user", SENT_FORM, None),
        ("user", SENT_FORM, None),
        ("assistant", "ok", None),
    ])

    calc = InteractionDepthCalculator()
    result = await calc.measure(db, "test-agent")

    assert result["message_count"] == 4
    # All rows are short after stripping → 0 substantive.
    assert result["substantive_rate"] == 0.0, (
        "post-strip substantive_rate must be 0; pre-strip would be 0.75"
    )


@pytest.mark.asyncio
async def test_wellness_depth_metric_does_not_strip_assistant_content():
    """Assistant content is persisted raw — the strip must not touch
    it. Defense-in-depth against an over-eager strip rule."""
    from kestrel_sovereign.features.wellness.metrics import (
        InteractionDepthCalculator,
    )

    long_assistant_reply = "x" * 250
    db = MagicMock()
    db.table_exists = AsyncMock(return_value=True)
    db.fetchall = AsyncMock(return_value=[
        ("assistant", long_assistant_reply, None),
        ("assistant", long_assistant_reply, None),
    ])

    calc = InteractionDepthCalculator()
    result = await calc.measure(db, "test-agent")

    # Both assistant messages are above SUBSTANTIVE_THRESHOLD.
    assert result["substantive_rate"] == 1.0, (
        "assistant rows must NOT have their content stripped"
    )
    assert result["avg_length"] == 250.0
