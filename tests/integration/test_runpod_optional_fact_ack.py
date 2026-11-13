"""Known native create IDs survive optional validation and late payload changes.

Only the owned local TLS peer replies are controlled. The original Core
custody, encrypted SQLite, native HTTP and client/socket joins all execute.
"""
import json

import pytest

from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionStorage
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.visual_computer.acquisition_native import (
    MAX_ACQUISITION_RESPONSE_BYTES,
)
from tests.integration.test_visual_runpod_acquisition import (
    SCOPE,
    custody as custody,  # noqa: PLC0414 - pytest fixture re-export.
    invoke,
    original_acquisition,
    peer as peer,  # noqa: PLC0414 - pytest fixture re-export.
    vendor as vendor,  # noqa: PLC0414 - pytest fixture re-export.
)


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_payload", [False, True], ids=["original", "late-repair"])
@pytest.mark.parametrize("optional_fact", [
    "env_integer", "malformed_created_at", "empty_status",
    "optional_text_over64k", "optional_nodes_over4096", "optional_depth_over16",
])
async def test_successful_original_create_persists_known_id_before_rejecting_optional_facts(
    custody, vendor, peer, monkeypatch, optional_fact, repair_payload,
):
    from kestrel_cloud_runpod.models import RunPodAmbiguousResultError

    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        if optional_fact == "env_integer":
            peer.reply["env"] = {"OPTIONAL_FACT": 7}
        elif optional_fact == "malformed_created_at":
            peer.reply["createdAt"] = "not-a-provider-timestamp"
        elif optional_fact == "empty_status":
            peer.reply["status"] = ""
        elif optional_fact == "optional_text_over64k":
            peer.reply["env"] = {"OPTIONAL_TEXT": "x" * (64 * 1024 + 1)}
        elif optional_fact == "optional_nodes_over4096":
            peer.reply["args"] = [None] * 4097
        else:
            nested = "optional-leaf"
            for _ in range(18):
                nested = [nested]
            peer.reply["args"] = nested
        expected_id, expected_name = peer.reply["id"], peer.reply["name"]
        reply_json = json.dumps(peer.reply).encode()
        assert MAX_ACQUISITION_RESPONSE_BYTES == 1024 * 1024
        assert len(reply_json) <= MAX_ACQUISITION_RESPONSE_BYTES
        assert json.loads(reply_json)["id"] == expected_id
        assert json.loads(reply_json)["name"] == expected_name
        assert type(expected_id) is type(expected_name) is str
        assert 0 < len(expected_id) <= 1024 and 0 < len(expected_name) <= 1024
        original_body = original.prepared.exact_body()
        assert await original.attempt.begin_dispatch() is True
        phase = native_error = None
        try:
            phase = await invoke(original)
        except RunPodAmbiguousResultError as error:
            if is_execution_control_error(error):
                raise
            native_error = error
        checkpoint = await original.handle.persist_minimal()
        durable = await original.store.get(
            SCOPE, request_id=original.request.request_id, guard=original.check,
        )
        receipt = original.handle._receipt
        assert receipt is original.attempt._result._registration.native_receipt
        assert receipt.raw_bytes == len(reply_json)
        assert receipt.exhausted and receipt.response_closed and receipt.client_closed
        assert original.native.is_closed and len(original.closes) == 1
        assert len(peer.requests) == 1
        assert peer.requests[0][0] == "/v2/pods" and peer.requests[0][1] == original_body
        assert peer.requests[0][2]["Accept-Encoding"] == "identity"
        assert peer.requests[0][3] == "TLSv1.3"
        assert checkpoint.state == "unknown"
        assert checkpoint.lease is not None and checkpoint.lease.provider_resource_id == expected_id, (
            "201 acknowledged an exact original Pod ID, but optional nonidentity validation discarded its minimal custody"
        )
        assert checkpoint.lease.actual_cost_usd is None
        assert durable.lease == checkpoint.lease and durable.acquisition.native_result is not None
        assert native_error is None
        assert phase is receipt.phase and phase.id == expected_id and phase.name == expected_name
        captured_facts = phase.facts
        if optional_fact.startswith("optional_"):
            assert captured_facts is None, "over-budget facts must remain permanently unavailable"
        else:
            assert type(captured_facts) is tuple
        producer = original.producer
        assert not producer._finished and producer._public_result is None
        if repair_payload:
            # Change the actual original continuation input before its FIRST
            # call. A consumed-call refusal cannot mask a missing facts guard.
            payload = producer._cost_payload
            assert type(payload) is dict
            if optional_fact == "env_integer":
                payload["env"] = {"OPTIONAL_FACT": "valid"}
            elif optional_fact == "malformed_created_at":
                payload["createdAt"] = "2026-11-13T00:00:00Z"
            elif optional_fact == "empty_status":
                payload["status"] = "RUNNING"
            elif optional_fact == "optional_text_over64k":
                payload["env"] = {"OPTIONAL_TEXT": "valid"}
            else:
                payload["args"] = []
            assert payload["id"] == expected_id and payload["name"] == expected_name
        assert phase.facts is captured_facts and not producer._finished
        with pytest.raises(RunPodAmbiguousResultError) as invalid:
            producer.finish_optional_cost(phase)
        assert invalid.value.status_code == 201
        assert invalid.value.title == "Unverified Pod creation response"
        assert producer._finished and producer._public_result is None
        assert phase.facts is captured_facts
        after = await original.store.get(
            SCOPE, request_id=original.request.request_id, guard=original.check,
        )
        assert after == durable
        saved = EnvironmentAcquisitionStorage.deserialize_private(after.acquisition.storage)
        assert saved.checkpoint == checkpoint and saved.checkpoint.state == "unknown"
        assert after.lease.provider_resource_id == expected_id
        assert after.lease.actual_cost_usd is None and after.lease.released_at is None
        assert len(peer.requests) == len(original.closes) == 1
