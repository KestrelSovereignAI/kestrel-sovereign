"""Native part MIME defaults, ranges and refusal without malformed fallback."""

import pytest
from a2a.types import Part
from a2a.utils.errors import ContentTypeNotSupportedError, InvalidParamsError
from google.protobuf.json_format import ParseDict

from kestrel_sovereign.a2a.media import part_media_type, validate_input_parts


def test_inline_json_preserves_exact_integers_and_filesystem_names():
    import json
    import os

    from kestrel_sovereign.a2a.media import encode_inline_json

    value = {"id": 10**30, "name": os.fsdecode(b"report-\xff.txt")}
    encoded = encode_inline_json(value)
    assert encoded.isascii()
    decoded = json.loads(encoded)
    assert decoded == value and type(decoded["id"]) is int
    assert os.fsencode(decoded["name"]) == b"report-\xff.txt"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_inline_json_rejects_nonfinite_numbers(value):
    from kestrel_sovereign.a2a.media import encode_inline_json

    with pytest.raises(ValueError):
        encode_inline_json({"value": value})


@pytest.mark.parametrize(
    "value", [{"nested": [None, {}, []]}, [], 0, False, None, "scalar"]
)
def test_native_data_part_accepts_every_official_json_value(value):
    part = ParseDict({"data": value}, Part())
    assert part_media_type(part) == ("application", "json")
    validate_input_parts([part], ["application/json"])


@pytest.mark.parametrize(
    "wire,mime",
    [
        ({"text": "hello"}, ("text", "plain")),
        ({"raw": "eA=="}, ("application", "octet-stream")),
        ({"url": "https://input.test/file"}, ("application", "octet-stream")),
        (
            {"text": "hello", "mediaType": "Text/Plain; charset=UTF-8"},
            ("text", "plain"),
        ),
    ],
)
def test_native_mime_defaults_are_based_on_content_kind(wire, mime):
    assert part_media_type(ParseDict(wire, Part())) == mime


@pytest.mark.parametrize("mode", ["text/plain", "text/*", "*/*"])
def test_supported_media_ranges_accept_matching_native_part(mode):
    validate_input_parts([Part(text="hello")], [mode])


@pytest.mark.parametrize("mime", ["bad", " ", "text/plain; bad", "*/*", "text/*"])
def test_malformed_or_wildcard_part_type_cannot_degrade_to_plain_text(mime):
    with pytest.raises(InvalidParamsError, match="MIME"):
        validate_input_parts(
            [Part(text="hello", media_type=mime)], ["text/plain", "*/*"]
        )


def test_part_requires_one_content_member():
    with pytest.raises(InvalidParamsError, match="content member"):
        validate_input_parts([Part()], ["*/*"])


def test_declared_unsupported_type_is_refused_instead_of_using_content_default():
    with pytest.raises(ContentTypeNotSupportedError, match="image/png"):
        validate_input_parts(
            [Part(raw=b"image", media_type="image/png")],
            ["text/plain", "application/json"],
        )


def test_malformed_host_mode_is_not_a_valid_negotiation_policy():
    with pytest.raises(InvalidParamsError, match="MIME"):
        validate_input_parts([Part(text="hello")], ["not-a-media-type"])


@pytest.mark.parametrize("mode", ["*/json", "te*xt/plain", "application/*json"])
def test_invalid_host_media_range_cannot_admit_input(mode):
    with pytest.raises(InvalidParamsError, match="MIME"):
        validate_input_parts([Part(data={})], [mode])
