"""Complete native request digests and callback-free executable projections."""

import hashlib

import pytest
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, SendMessageRequest
from a2a.utils.errors import InvalidParamsError
from google.protobuf.json_format import MessageToDict

from kestrel_sovereign.a2a.native_input import (
    native_request_digest,
    prepare_native_task_input,
)


def request(*, callback=True):
    value = SendMessageRequest(
        tenant="tenant",
        message=new_text_message("Original work", role=Role.ROLE_USER),
        metadata={"nested": {"keep": [False, 0, None, {}, []]}},
    )
    value.configuration.accepted_output_modes.append("application/json")
    value.configuration.history_length = 0
    if callback:
        config = value.configuration.task_push_notification_config
        config.url = "https://callback.example/private-path?secret=do-not-persist"
        config.token = "callback-validation-secret"
        config.authentication.scheme = "Bearer"
        config.authentication.credentials = "callback-authorization-secret"
    return value


def test_inline_secrets_never_enter_executable_input_json_or_representation():
    original = request()
    before = original.SerializeToString(deterministic=True)
    prepared = prepare_native_task_input(original)
    assert original.SerializeToString(deterministic=True) == before
    assert prepared.wire_request_digest == native_request_digest(
        original, "SendMessage"
    )
    execution = prepared.request
    assert not execution.configuration.HasField("task_push_notification_config")
    assert execution.configuration.HasField("history_length")
    assert execution.configuration.history_length == 0
    assert list(execution.configuration.accepted_output_modes) == ["application/json"]
    assert execution.message == original.message
    assert execution.metadata == original.metadata
    assert (
        prepared.execution_digest
        == hashlib.sha256(execution.SerializeToString(deterministic=True)).hexdigest()
    )
    assert prepared.execution_digest != hashlib.sha256(before).hexdigest()
    for secret in [
        "callback.example",
        "private-path",
        "do-not-persist",
        "callback-validation-secret",
        "callback-authorization-secret",
    ]:
        assert secret not in prepared.request_json
        assert secret not in repr(prepared)
    assert "taskPushNotificationConfig" not in prepared.request_json
    assert (
        prepared.push_configuration.authentication.credentials
        == "callback-authorization-secret"
    )
    assert prepared.push_configuration.tenant == original.tenant
    assert not original.configuration.task_push_notification_config.tenant


def test_readers_cannot_mutate_the_canonical_execution_or_callback_projection():
    prepared = prepare_native_task_input(request())
    execution, callback = prepared.request, prepared.push_configuration
    execution.message.parts[0].text = "Mutated work"
    callback.authentication.credentials = "Mutated credential"
    assert prepared.request.message.parts[0].text == "Original work"
    assert (
        prepared.push_configuration.authentication.credentials
        == "callback-authorization-secret"
    )


def test_callback_changes_preserve_executable_input_but_change_wire_conflict_identity():
    original = request()
    first = prepare_native_task_input(original)
    original.configuration.task_push_notification_config.authentication.credentials = (
        "changed-secret"
    )
    changed = prepare_native_task_input(original)
    assert changed.request == first.request
    assert changed.execution_digest == first.execution_digest
    assert changed.wire_request_digest != first.wire_request_digest


def test_non_callback_input_retains_every_official_field_and_optional_presence():
    original = request(callback=False)
    prepared = prepare_native_task_input(original)
    assert prepared.request == original
    assert prepared.push_configuration is None
    assert MessageToDict(prepared.request)["configuration"]["historyLength"] == 0
    assert (
        prepared.execution_digest
        == hashlib.sha256(original.SerializeToString(deterministic=True)).hexdigest()
    )


@pytest.mark.parametrize("field", ["url", "token", "credentials"])
def test_every_callback_change_is_bound_by_the_complete_request_digest(field):
    original = request()
    first = prepare_native_task_input(original)
    config = original.configuration.task_push_notification_config
    if field == "credentials":
        config.authentication.credentials += "changed"
    else:
        setattr(config, field, getattr(config, field) + "changed")
    second = prepare_native_task_input(original)
    assert second.wire_request_digest != first.wire_request_digest
    assert second.execution_digest == first.execution_digest


def test_inline_config_cannot_select_an_existing_task_or_another_tenant():
    original = request()
    original.configuration.task_push_notification_config.task_id = "other-task"
    with pytest.raises(InvalidParamsError, match="taskId"):
        prepare_native_task_input(original)
    original.configuration.task_push_notification_config.task_id = ""
    original.configuration.task_push_notification_config.tenant = "other-tenant"
    with pytest.raises(InvalidParamsError, match="tenant"):
        prepare_native_task_input(original)


def test_matching_inline_tenant_is_retained_in_secret_projection():
    original = request()
    original.configuration.task_push_notification_config.tenant = "tenant"
    prepared = prepare_native_task_input(original)
    assert prepared.push_configuration.tenant == "tenant"
    assert prepared.request.configuration.HasField("history_length")


def test_private_wire_shapes_and_missing_required_fields_refuse():
    with pytest.raises(TypeError, match="official SendMessageRequest"):
        prepare_native_task_input({"sessionId": "draft"})
    original = request()
    original.message.message_id = ""
    with pytest.raises(InvalidParamsError):
        prepare_native_task_input(original)
