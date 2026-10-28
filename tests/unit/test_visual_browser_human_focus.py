"""Original worker engine with its existing bounded page double; no browser claim."""
import asyncio
import json
import os
from pathlib import Path

import pytest

from tests.unit.test_visual_browser_human import opened, action, Refused


def facts_path(tmp_path):
    return Path(os.environ.get("CURRENT_ROW_PROOF_DIR", tmp_path)) / "focus-facts.json"


@pytest.mark.asyncio
async def test_focus_failure_before_intent_has_original_durable_denial(tmp_path, monkeypatch):
    controller, page, session, human, _, lease = await opened(tmp_path)
    request = await action(controller, session, human, lease)
    failure = RuntimeError("owned browser focus context destroyed before input")

    async def fail_focus(_):
        raise failure

    monkeypatch.setattr(page, "evaluate", fail_focus)
    caught = None
    try:
        try:
            await human.act(session, lease, request)
        except Exception as error:
            caught = error
        evidence = {
            "exception_type": None if caught is None else type(caught).__name__,
            "typed_refusal": isinstance(caught, Refused),
            "original_untyped_failure": caught is failure,
            "effect_count": len(page.effects),
            "human_intent_present": request["action"]["action_id"] in human.actions,
            "controller_receipt_present": request["action"]["action_id"] in controller._actions,
            "page_closed": page.closed,
        }
        facts_path(tmp_path).write_text(json.dumps(evidence, indent=2) + "\n")
        assert isinstance(caught, Refused), evidence
        assert not page.effects and not page.closed
        receipt = human.action_status(session, lease, request["action"]["action_id"])
        assert receipt["receipt"]["outcome"] == "denied"
        assert receipt["receipt"]["action_id"] == request["action"]["action_id"]
        assert receipt["receipt"]["observation_id"] == request["action"]["observation_id"]
        assert human.actions[request["action"]["action_id"]]["receipt"] == receipt
        persisted = json.loads(human.journal.read_text())["payload"]["actions"]
        assert persisted[request["action"]["action_id"]]["receipt"] == receipt
        assert "PRIVATE-MANUAL-SECRET" not in human.journal.read_text()
    finally:
        await controller.retire()


@pytest.mark.asyncio
async def test_focus_cancellation_keeps_original_exception_and_no_intent(tmp_path, monkeypatch):
    controller, page, session, human, _, lease = await opened(tmp_path)
    request = await action(controller, session, human, lease)
    cancelled = asyncio.CancelledError("owned focus cancellation")

    async def cancel_focus(_):
        raise cancelled

    monkeypatch.setattr(page, "evaluate", cancel_focus)
    try:
        with pytest.raises(asyncio.CancelledError) as caught:
            await human.act(session, lease, request)
        assert caught.value is cancelled
        assert not page.effects
        assert request["action"]["action_id"] not in human.actions
        assert request["action"]["action_id"] not in controller._actions
    finally:
        await controller.retire()
