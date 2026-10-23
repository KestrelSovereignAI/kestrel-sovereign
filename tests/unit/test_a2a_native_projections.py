"""Native oneof selection is preserved in text-only presentation."""

import pytest
from kestrel_sovereign._vendor.a2a.types import Message, Part
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Value

from kestrel_sovereign.a2a.projections import message_text


def test_text_projection_ignores_other_parts_and_preserves_order_and_explicit_empty():
    message = Message(
        parts=[
            Part(data=ParseDict({"text": "not message text"}, Value())),
            Part(text=""),
            Part(raw=b"payload"),
            Part(text="first"),
            Part(text="second"),
        ]
    )
    before = message.SerializeToString()
    assert message_text(message) == "\nfirst\nsecond"
    assert message.SerializeToString() == before
    assert message_text(Message(parts=[Part(data=ParseDict({}, Value()))])) is None
    assert message_text(Message(parts=[Part(text="")])) == ""


def test_text_projection_refuses_draft_or_duck_typed_messages():
    with pytest.raises(TypeError, match="official A2A Message"):
        message_text({"parts": [{"type": "text", "text": "draft"}]})


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        3,
        1153921230456789012,
        "",
        [],
        {"nested": [None, {}, 1153921230456789012]},
    ],
)
def test_embedded_json_bytes_preserve_exact_values_and_explicit_null(value):
    import json

    from kestrel_sovereign.a2a.projections import embedded_json_value

    part = Part(
        raw=json.dumps(value).encode("utf-8"),
        media_type="application/json; charset=utf-8",
    )
    frozen = part.SerializeToString()
    recognized, actual = embedded_json_value(part)
    assert recognized is True and actual == value and type(actual) is type(value)
    assert part.SerializeToString() == frozen
    assert embedded_json_value(Part(data=ParseDict(value, Value())))[0] is True


@pytest.mark.parametrize(
    "content", [b"NaN", b'{"id":Infinity}', b'{"id":-Infinity}', b"{", b"\xff"]
)
def test_embedded_json_bytes_fail_clearly_on_non_json_or_invalid_utf8(content):
    from kestrel_sovereign.a2a.projections import embedded_json_value

    with pytest.raises(ValueError):
        embedded_json_value(Part(raw=content, media_type="application/json"))


def test_embedded_json_projection_does_not_fetch_urls_or_parse_untyped_bytes():
    from kestrel_sovereign.a2a.projections import embedded_json_value

    assert embedded_json_value(
        Part(url="https://invalid.test/private", media_type="application/json")
    ) == (False, None)
    assert embedded_json_value(Part(raw=b'{"number":3}')) == (False, None)
    assert embedded_json_value(Part(text='{"number":3}')) == (False, None)
    with pytest.raises(TypeError, match="official A2A Part"):
        embedded_json_value({"data": {"number": 3}})


def test_embedded_json_projection_rejects_a_duck_typed_part():
    from kestrel_sovereign.a2a.projections import embedded_json_value

    class ForeignPart:
        def HasField(self, name):
            return False

    with pytest.raises(TypeError, match="official A2A Part"):
        embedded_json_value(ForeignPart())
