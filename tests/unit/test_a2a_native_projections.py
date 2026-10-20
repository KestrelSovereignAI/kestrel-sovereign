"""Native oneof selection is preserved in text-only presentation."""

import pytest
from a2a.types import Message, Part
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
