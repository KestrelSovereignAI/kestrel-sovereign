# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.
"""Decode peer response data separately from transport and admission failures."""

import json
import math

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, NoReturn, TypeVar, cast

from google.protobuf import json_format
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from google.protobuf.message import Message
from jsonrpc.jsonrpc2 import JSONRPC20Response

from kestrel_sovereign._vendor.a2a.client.errors import A2AClientError
from kestrel_sovereign._vendor.a2a.utils.proto_utils import _field_is_repeated, parse_lenient


_MessageT = TypeVar('_MessageT', bound=Message)


@contextmanager
def _decoding_errors() -> Iterator[None]:
    """Normalize only failures from a synchronous peer-data decoder."""
    try:
        yield
    except json.JSONDecodeError as error:
        raise A2AClientError(f'JSON Decode Error: {error}') from error
    except (
        json_format.Error,
        ValueError,
        TypeError,
        OverflowError,
        RecursionError,
    ) as error:
        raise A2AClientError('Invalid A2A response payload') from error


def _reject_nonfinite(value: str) -> NoReturn:
    raise ValueError(f'JSON cannot contain the constant {value}')


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('JSON number exceeds the finite floating-point range')
    return number


def _response_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError('A2A response payload must be a JSON object')
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'Duplicate JSON object key: {key}')
        value[key] = item
    return value


def _validate_json_unicode(value: Any) -> None:
    """Reject unpaired surrogate escapes before native protobuf conversion.

    The upb decoder can raise SystemError for these even in an unknown key
    or an enum. Validate the complete peer JSON value, including future fields,
    rather than depending on which protobuf backend or schema handles it.
    """
    if isinstance(value, str):
        value.encode('utf-8')
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_json_unicode(key)
            _validate_json_unicode(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json_unicode(item)


def load_json_value(payload: str | bytes) -> Any:
    """Decode JSON with the same duplicate-key rule as protobuf JSON."""
    with _decoding_errors():
        return json.loads(
            payload,
            parse_constant=_reject_nonfinite,
            parse_float=_finite_float,
            object_pairs_hook=_unique_json_object,
        )


def load_json_response(payload: str | bytes) -> dict[str, Any]:
    """Decode one JSON response body or one SSE data value."""
    value = load_json_value(payload)
    with _decoding_errors():
        return _response_object(value)


def decode_response_text(payload: bytes) -> str:
    """Decode the non-SSE response fallback without hiding transport errors."""
    with _decoding_errors():
        return payload.decode('utf-8')


def _jsonrpc_error_code(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('JSON-RPC error code must be an integer')
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        raise ValueError('JSON-RPC error code must be an integer')
    return int(value)


def parse_jsonrpc_response(
    payload: Any, *, request_id: str | int | None
) -> JSONRPC20Response:
    """Validate the JSON-RPC envelope after its HTTP request has completed."""
    with _decoding_errors():
        body = dict(_response_object(payload))
        if (
            body.get('jsonrpc') != '2.0'
            or ('id' not in body and 'error' not in body)
            or ('result' in body) == ('error' in body)
        ):
            raise ValueError('Invalid JSON-RPC response envelope')
        if 'error' in body:
            error = body['error']
            if not isinstance(error, dict):
                raise TypeError('JSON-RPC error must be an object')
            if not isinstance(error.get('message'), str):
                raise TypeError('JSON-RPC error message must be a string')
            code = _jsonrpc_error_code(error.get('code'))
            body['error'] = {**error, 'code': code}
        response_id = body.get('id')
        # An unidentified error remains a failure on this HTTP response.
        # Native servers can emit it before their streaming handler attaches
        # the request ID. Never relax correlation for a successful result.
        if isinstance(response_id, bool) or (
            response_id != request_id
            and not (response_id is None and 'error' in body)
        ):
            raise ValueError('JSON-RPC response ID does not match the request')
        return JSONRPC20Response(**body)


_SCALAR_JSON_MESSAGE_TYPES = frozenset(
    {
        'google.protobuf.Timestamp',
        'google.protobuf.Duration',
        'google.protobuf.FieldMask',
        'google.protobuf.DoubleValue',
        'google.protobuf.FloatValue',
        'google.protobuf.Int64Value',
        'google.protobuf.UInt64Value',
        'google.protobuf.Int32Value',
        'google.protobuf.UInt32Value',
        'google.protobuf.BoolValue',
        'google.protobuf.StringValue',
        'google.protobuf.BytesValue',
    }
)


def _validate_message_shapes(value: Any, descriptor: Descriptor) -> None:
    """Check ProtoJSON containers while leaving scalar conversion to protobuf.

    Lenient protobuf parsing can iterate a string or an empty array as though
    it were an ordinary message object and return an empty successful message.
    Unknown fields remain ignored, but known fields retain their schema shape.
    Well-known types use their standard JSON representations instead.
    """
    if descriptor.full_name == 'google.protobuf.Value':
        return
    if descriptor.full_name in _SCALAR_JSON_MESSAGE_TYPES:
        return  # Protobuf owns the special scalar representation and validation.
    if descriptor.full_name == 'google.protobuf.ListValue':
        if not isinstance(value, list):
            raise TypeError('ListValue must be a JSON array')
        return
    if not isinstance(value, dict):
        raise TypeError(f'{descriptor.full_name} must be a JSON object')
    if descriptor.full_name == 'google.protobuf.Struct':
        return  # Its keys are arbitrary JSON keys, not protobuf field names.
    json_fields = {field.json_name: field for field in descriptor.fields}
    for key, item in value.items():
        field = descriptor.fields_by_name.get(key) or json_fields.get(key)
        if field is None or item is None:
            continue  # Unknown fields and field-level null retain native semantics.
        _validate_field_shape(item, field)


def _validate_field_shape(value: Any, field: FieldDescriptor) -> None:
    """Validate one known field's map, repeated or ordinary message shape."""
    nested = field.message_type
    if nested is not None and nested.GetOptions().map_entry:
        if not isinstance(value, dict):
            raise TypeError(f'{field.full_name} must be a JSON object')
        mapped_message = nested.fields_by_name['value'].message_type
        if mapped_message is not None:
            for mapped in value.values():
                _validate_message_shapes(mapped, mapped_message)
    elif _field_is_repeated(field):
        if not isinstance(value, list):
            raise TypeError(f'{field.full_name} must be a JSON array')
        if nested is not None:
            for element in value:
                _validate_message_shapes(element, nested)
    elif nested is not None:
        _validate_message_shapes(value, nested)


def parse_protobuf_response(payload: Any, message: _MessageT) -> _MessageT:
    """Parse a native response and require its JSON representation to exist."""
    with _decoding_errors():
        payload = _response_object(payload)
        _validate_json_unicode(payload)
        # Python and upb descriptors implement the same public interface,
        # while protobuf's stubs expose separate concrete descriptor classes.
        _validate_message_shapes(
            payload, cast('Descriptor', message.DESCRIPTOR)
        )
        parse_lenient(payload, message)
        if (
            'payload' in message.DESCRIPTOR.oneofs_by_name
            and message.WhichOneof('payload') is None
        ):
            raise ValueError('A2A response must contain one payload')
        # Value.number_value admits NaN/Infinity in memory, but cannot encode
        # them in the native JSON representation. Check before returning it.
        json_format.MessageToDict(message)
    return message
