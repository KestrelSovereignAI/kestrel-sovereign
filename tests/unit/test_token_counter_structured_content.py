"""Contract tests for counting STRUCTURED message content.

Anthropic message content is frequently a block array (``text`` /
``tool_use`` / ``tool_result``) rather than a plain string. Those blocks
reach the provider as JSON and cost real tokens, so the counter that feeds
the context budget has to measure them.

Regression: a non-string fell through to the character estimate and was
measured as ``len(obj) // 4`` — the count of BLOCKS, not their size. A
single ``tool_result`` holding 400 KB measured as 7 tokens, leaving any
budget fed block-form content blind to essentially all of it.
"""

import pytest

from kestrel_sovereign.agent.token_counter import get_token_counter

PAYLOAD = "x" * 400_000
# The old implementation returned 7 for every case below. A floor well above
# that — but still far under the true count — fails loudly on a regression
# without pinning the exact tokenisation.
SUBSTANTIAL = 40_000


@pytest.fixture
def counter():
    return get_token_counter("claude-opus-5")


def test_plain_string_content_is_counted(counter):
    """Control: the shape that always worked must still work, so a failure
    below is attributable to the structured path and not the counter."""
    assert counter.count_messages(
        [{"role": "user", "content": PAYLOAD}]
    ) > SUBSTANTIAL


def test_text_block_content_is_counted(counter):
    assert counter.count_messages(
        [{"role": "user", "content": [{"type": "text", "text": PAYLOAD}]}]
    ) > SUBSTANTIAL


def test_tool_result_block_content_is_counted(counter):
    """The shape Emma's turns are mostly made of."""
    assert counter.count_messages(
        [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": PAYLOAD}
        ]}]
    ) > SUBSTANTIAL


def test_tool_use_input_is_counted(counter):
    assert counter.count_messages(
        [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "f", "input": {"q": PAYLOAD}}
        ]}]
    ) > SUBSTANTIAL


def test_block_and_string_forms_agree(counter):
    """Same payload, two encodings: the budget must not depend on which one
    the caller happens to hold."""
    as_string = counter.count(PAYLOAD)
    as_block = counter.count([{"type": "text", "text": PAYLOAD}])
    assert as_block == pytest.approx(as_string, rel=0.02)


def test_many_small_blocks_are_not_collapsed(counter):
    """``len(obj) // 4`` scaled with the number of blocks, so this case was
    the one it got least wrong — it must still be measured by size."""
    blocks = [{"type": "text", "text": "y" * 1000} for _ in range(200)]
    assert counter.count_messages([{"role": "user", "content": blocks}]) > SUBSTANTIAL


@pytest.mark.parametrize("empty", [None, "", [], {}, 0])
def test_empty_content_is_zero(counter, empty):
    assert counter.count(empty) == 0


def test_unserialisable_content_does_not_raise(counter):
    """Measurement must never be the thing that breaks a turn."""
    class Opaque:
        __slots__ = ()

    assert counter.count([{"type": "text", "text": Opaque()}]) >= 0
