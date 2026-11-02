"""Genuine registered host/native encrypted SQLite; controlled SDK worker only.

The worker double reports an uncertain open without accepting a lease. No input,
Chromium, physical browser, external transport or cloud conformance is claimed.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest
from kestrel_sdk.computer_use import ComputerSession
from kestrel_sdk.computer_use.human_control import HumanControlRevocationTombstone
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign.visual_computer.human import HumanComputerRuntime, HumanRecord
from tests.unit.visual_computer.test_host import AUTH, intent, path
from tests.unit.visual_computer.test_host import visual_host as visual_host
from tests.unit.visual_computer.test_human import Extension


class UnknownOpenBeforeLease(Extension):
    """Existing guarded SDK test extension, one deliberate unknown outcome."""
    def __init__(self, runtime):
        super().__init__(runtime)
        self.original_requests = []
        self.original_open_children = []

    async def open(self, session, request):
        def run():
            request.require_session(session, at=self.r._now())
            self.original_requests.append(request)
            self.original_open_children.append(asyncio.current_task())
            raise ResourceOutcomeUnknown("controlled original open has no accepted lease")
        return await self._call(session, request.control_id, "open", run)


async def _paused_human(visual_host, harness):
    client, agents, _ = visual_host
    feature = agents["kite"].features["VisualComputerFeature"]
    # Exactly the original constructor/configuration used by test_human_http;
    # no constructor, authenticated resolver, current checker or grant replaced.
    feature.runtime.human = HumanComputerRuntime(feature.runtime)
    extension = UnknownOpenBeforeLease(feature.runtime)
    harness.adapter.human_control = lambda: extension
    admitted = await client.post(path() + "/requests", headers=AUTH, json=intent())
    assert admitted.status_code == 200, admitted.text
    sid = admitted.json()["session_id"]
    base = path() + "/sessions/" + sid
    takeover = await client.post(base + "/takeover", headers=AUTH, json={
        "request_id": "original-takeover", "session_id": sid,
        "expected_control_epoch": 0, "operation": "takeover"})
    assert takeover.status_code == 200, takeover.text
    session = ComputerSession.model_validate_json(json.dumps(takeover.json()))
    assert session.controller == "human" and session.control_epoch == 1
    assert session.scope.tenant_id == "did:test:kite"
    assert session.scope.principal_id == "host-sovereign"
    assert feature.runtime.repository.db is harness.db
    assert harness.db.backend_type == "sqlite"
    return client, feature, extension, session, base


async def _observed_row(repository, session, cid):
    # Native read-only observation cannot manufacture a request/fence or repair
    # custody through another status call. All writes are original host methods.
    owner = repository._owner(session.scope)
    key = repository._address("private", session.session_id, "human", cid)
    row = await repository.db.fetchone(
        "SELECT payload FROM visual_computer_private_records WHERE owner = ? AND record_key = ?",
        (owner, key))
    assert row is not None
    payload = bytes(row[0])
    value = repository._open(owner, key, payload, HumanRecord)
    assert b'"request"' not in payload
    return value, payload


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", ["same-id", "new-id"])
async def test_original_unknown_open_tombstone_reconciles_same_target_revoke(
    visual_host, harness, tmp_path, retry,
):
    client, feature, extension, session, base = await _paused_human(visual_host, harness)
    cid = "unknown-open-control"
    hp = base + "/human/" + cid
    out = Path(os.environ.get("CURRENT_ROW_PROOF_DIR", tmp_path))
    facts = {"controlled_sdk_only": True, "physical_browser_proof": False, "retry": retry}
    def persist():
        (out / ("tombstone-" + retry + ".json")).write_text(json.dumps(facts, indent=2) + "\n")
    try:
        opened = await client.post(hp, headers=AUTH)
        assert opened.status_code == 503, opened.text
        assert opened.json()["error"]["code"] == "visual_outcome_unknown"
        original, _ = await _observed_row(feature.runtime.repository, session, cid)
        assert original.request == extension.original_requests[0]
        assert original.lease is None and original.tombstone is None
        assert original.request.scope == session.scope
        assert original.request.session_id == session.session_id
        assert len(extension.original_requests) == 1 and extension.effects == 0
        first_request = {"request_id": "original-revoke", "control_epoch": session.control_epoch}
        first = await client.post(hp + "/revoke", headers=AUTH, json=first_request)
        assert first.status_code == 200 and first.json() == {"revoked": True}, first.text
        terminal, before = await _observed_row(feature.runtime.repository, session, cid)
        assert terminal.request == original.request and terminal.lease is None
        assert isinstance(terminal.tombstone, HumanControlRevocationTombstone)
        terminal.tombstone.require_revocation(terminal.revocation, session)
        assert terminal.tombstone.request.request_id == "original-revoke"
        assert terminal.tombstone == extension.tombstones[cid]
        assert terminal.frame is None
        repeated_request = dict(first_request)
        if retry == "new-id":
            repeated_request["request_id"] = "confirm-original-target"
        response = await client.post(hp + "/revoke", headers=AUTH, json=repeated_request)
        after, payload = await _observed_row(feature.runtime.repository, session, cid)
        facts.update(open_http=opened.status_code, first_revoke_http=first.status_code,
            repeated_revoke_http=response.status_code, repeated_revoke_body=response.json(),
            request_preserved=after.request == original.request,
            lease_is_none=after.lease is None, original_tombstone_retained=after.tombstone == terminal.tombstone,
            original_revocation_retained=after.revocation == terminal.revocation,
            fenced_at_unchanged=after.tombstone.fenced_at == terminal.tombstone.fenced_at,
            ciphertext_before=hashlib.sha256(before).hexdigest(),
            ciphertext_after=hashlib.sha256(payload).hexdigest(),
            ciphertext_unchanged=before == payload, worker_open_calls=len(extension.original_requests),
            worker_effects=extension.effects, egress_calls=len(feature.runtime.egress._calls),
            native_transaction_open=harness.db.backend._in_transaction,
            original_result_holds=len(harness.db.backend._original_result_holds))
        persist()
        assert after.request == original.request and after.lease is None
        assert extension.effects == 0 and len(extension.original_requests) == 1
        assert not feature.runtime.egress._calls
        # Same-target reconciliation retains the original accepted request
        # while confirming the original fence without reopening the channel.
        assert response.status_code == 200, response.text
        assert response.json() == {"revoked": True}
        assert after.tombstone.request.request_id == repeated_request["request_id"]
        assert after.tombstone.fenced_at == terminal.tombstone.fenced_at
    finally:
        await asyncio.gather(*extension.original_open_children, return_exceptions=True)
        facts["original_open_children_awaited"] = len(extension.original_open_children)
        facts["original_open_children_done"] = all(t.done() for t in extension.original_open_children)
        persist()


@pytest.mark.asyncio
async def test_original_preopen_tombstone_still_refuses_new_request(visual_host, harness, tmp_path):
    client, feature, extension, session, base = await _paused_human(visual_host, harness)
    cid = "never-opened-control"
    hp = base + "/human/" + cid
    first = await client.post(hp + "/revoke", headers=AUTH,
        json={"request_id": "preopen-first", "control_epoch": session.control_epoch})
    assert first.status_code == 200, first.text
    original, before = await _observed_row(feature.runtime.repository, session, cid)
    assert original.request is None and original.lease is None
    original.tombstone.require_revocation(original.revocation, session)
    # The genuinely authenticated request must not install an accepted request
    # or reach even the controlled worker after this original pre-open fence.
    response = await client.post(hp, headers=AUTH)
    after, payload = await _observed_row(feature.runtime.repository, session, cid)
    facts = {"controlled_sdk_only": True, "physical_browser_proof": False,
        "first_revoke_http": first.status_code, "late_open_http": response.status_code,
        "request_remains_none": after.request is None, "lease_remains_none": after.lease is None,
        "original_tombstone_retained": after.tombstone == original.tombstone,
        "ciphertext_unchanged": before == payload, "worker_open_calls": len(extension.original_requests),
        "worker_effects": extension.effects, "egress_calls": len(feature.runtime.egress._calls)}
    out = Path(os.environ.get("CURRENT_ROW_PROOF_DIR", tmp_path))
    (out / "preopen-inverse.json").write_text(json.dumps(facts, indent=2) + "\n")
    assert response.status_code == 409, response.text
    assert after == original and before == payload
    assert not extension.original_requests and extension.effects == 0
    assert not feature.runtime.egress._calls
