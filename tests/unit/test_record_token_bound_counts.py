"""The #4085 recording script builds the record row 14 reads, with no live call."""

import json

import pytest

from kestrel_sovereign.agent import token_counter as tc
from scripts import record_token_bound_counts as script

MODEL = "claude-fable-5-2"


@pytest.fixture(autouse=True)
def fresh_ratchet(monkeypatch):
    monkeypatch.setattr(tc, "_ratchets", {})


def test_every_fixture_class_has_a_committed_text():
    for name in script.FIXTURE_CLASSES:
        request = script.fixture_request(name)
        if name == "tool_schemas":
            assert request["tools"] and request["messages"][0]["content"] == script.TOOL_CLASS_PROMPT
        else:
            assert request["tools"] is None
            assert len(request["messages"][0]["content"].encode("utf-8")) > 10_000


@pytest.mark.asyncio
async def test_the_record_pairs_each_class_bytes_with_the_provider_count_and_no_content():
    counted = []

    async def count(request):
        counted.append(request)
        return 1_000 + len(counted)

    record = await script.record_counts(MODEL, count)
    assert record["model"] == MODEL
    assert set(record["classes"]) == set(script.FIXTURE_CLASSES)
    counter = tc.get_token_counter(MODEL)
    for index, name in enumerate(script.FIXTURE_CLASSES, start=1):
        request = script.fixture_request(name)
        bound = counter.upper_bound(request["system"], request["messages"], request["tools"])
        assert record["classes"][name] == {"wire_bytes": bound.wire_bytes, "provider_count": 1_000 + index}
    text = (script.FIXTURE_DIR / "hex_and_ids.txt").read_text(encoding="utf-8")
    assert text.splitlines()[0] not in json.dumps(record)


@pytest.mark.asyncio
async def test_a_class_the_provider_did_not_count_is_an_error_not_a_record():
    async def count(request):
        return None

    with pytest.raises(RuntimeError, match="no count"):
        await script.record_counts(MODEL, count)
