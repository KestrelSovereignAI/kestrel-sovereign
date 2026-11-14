"""A tool result pruned for context pressure must say it is gone (#4076).

``_prune_orchestrator_messages`` replaces an oversized tool result with a stub
once shedding replayed history is not enough. The stub used to read "Tool
completed successfully", so a read whose content was dropped looked like a
complete one (#3016). It must state the size that was dropped, that the content
is not retained and cannot be reread, and claim no outcome beyond the status
the result recorded.

Every test here is pure: no host, no model, no subprocess.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.agent import orchestrator_engine as oe
from kestrel_sovereign.agent.orchestrator_engine import (
    PRUNED_TOOL_RESULT_PREFIX,
    OrchestratorEngineMixin,
)

# ~4,480 message chars at the 0.2 reserve: a 40,000-char result cannot fit,
# but the conversation fits again once that one result is dropped.
_CONTEXT_LIMIT = 1600


def _prune(messages, context_limit=_CONTEXT_LIMIT):
    agent = MagicMock()
    agent.llm_service = MagicMock(spec=[])
    prune = OrchestratorEngineMixin._prune_orchestrator_messages.__get__(agent)
    return prune(messages, [], context_limit=context_limit)


def _turn(*tool_contents):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": "", "tool_calls": []},
    ]
    for index, content in enumerate(tool_contents):
        messages.append(
            {"role": "tool", "tool_call_id": f"call_{index}", "content": content}
        )
    return messages


def _ok_read(chars):
    return json.dumps({
        "status": "ok",
        "confirmation": "Read docs/audit/REPO_MAP.md",
        "data": {"content": "x" * chars},
    })


def test_pruned_result_stub_states_size_dropped_and_not_retained():
    content = _ok_read(40000)
    original = _turn(content)

    result = _prune(original)

    stub = result[-1]
    assert stub["role"] == "tool"
    assert stub["tool_call_id"] == "call_0"
    text = stub["content"]
    assert f"{len(content):,} chars" in text
    assert "dropped for context pressure" in text
    assert "not retained" in text
    assert "cannot be reread" in text
    assert "completed successfully" not in text.lower()
    # "not retained" is literal: nothing of the content reaches the provider.
    assert "x" * 100 not in json.dumps(result)


def test_result_that_is_not_pruned_is_unchanged():
    small = json.dumps({"status": "ok", "confirmation": "listed", "data": [1]})
    later = _ok_read(1000)
    original = _turn(_ok_read(40000), small, later)
    expected_untouched = [dict(m) for m in original[-2:]]

    result = _prune([dict(m) for m in original])

    assert result[3]["content"].startswith(PRUNED_TOOL_RESULT_PREFIX)
    # Dropping the first result was enough, so neither the small result nor the
    # later one is touched.
    assert result[-2:] == expected_untouched


def test_result_that_fits_is_returned_unchanged():
    original = _turn(_ok_read(40000))

    result = _prune([dict(m) for m in original], context_limit=200000)

    assert result == original


@pytest.mark.parametrize(
    "envelope",
    [
        {"status": "error", "error": "permission denied: " + "e" * 400},
        {"success": False, "error": "file not found: " + "e" * 400},
        {"error": "upstream 502: " + "e" * 400},
    ],
    ids=["tool-result-error", "legacy-success-false", "bare-error"],
)
def test_pruned_result_with_recorded_error_does_not_claim_success(envelope):
    result = _prune(_turn(json.dumps(envelope), _ok_read(40000)))

    text = result[3]["content"]
    assert text.startswith(PRUNED_TOOL_RESULT_PREFIX)
    assert "Recorded status: error." in text
    assert "success" not in text.lower()
    assert "completed" not in text.lower()


def test_pruned_partial_result_reports_partial():
    envelope = {
        "status": "partial",
        "confirmation": "read part of it",
        "error": "stopped at byte 4096",
        "data": {"content": "p" * 40000},
    }
    result = _prune(_turn(json.dumps(envelope)))

    assert "Recorded status: partial." in result[-1]["content"]


@pytest.mark.parametrize(
    ("content", "status"),
    [
        (json.dumps({"status": "ok", "confirmation": "done"}), "ok"),
        (json.dumps({"success": True, "data": "d"}), "ok"),
        (json.dumps({"success": False}), "error"),
        (json.dumps({"success": "false"}), "unknown"),
        (json.dumps({"status": "open", "title": "an issue"}), "unknown"),
        (json.dumps({"status": {"state": "open"}}), "unknown"),
        (json.dumps({"rows": [1, 2, 3]}), "unknown"),
        (json.dumps("a raw string result"), "unknown"),
        (
            oe._build_persisted_preview(_ok_read(20000), "fs_read", 20000),
            "unknown",
        ),
    ],
    ids=[
        "tool-result-ok", "legacy-success-true", "legacy-success-false",
        "stringly-success",
        "foreign-status-field", "unhashable-status-field", "no-status",
        "raw-string", "persisted-preview",
    ],
)
def test_recorded_status_claims_only_what_the_result_recorded(content, status):
    assert oe._recorded_tool_result_status(content) == status


def test_unparseable_deep_result_reads_as_unknown_rather_than_failing_the_turn():
    # Python 3.11's json.loads raises RecursionError, not ValueError, on deep
    # nesting; patched so the case is exercised on every interpreter.
    with patch.object(oe.json, "loads", side_effect=RecursionError("deep")):
        assert oe._recorded_tool_result_status("[" * 2000) == "unknown"


def test_a_later_prune_pass_keeps_the_original_size():
    """Each tool iteration prunes the accumulated array again.

    A stub long enough to be pruned itself would be replaced by one stating the
    stub's own length, and the original size would be lost.
    """
    content = _ok_read(40000)
    first = _prune(_turn(content), context_limit=10)
    stub = first[-1]["content"]

    second = _prune([dict(m) for m in first], context_limit=10)

    assert second[-1]["content"] == stub
    assert f"{len(content):,} chars" in second[-1]["content"]
