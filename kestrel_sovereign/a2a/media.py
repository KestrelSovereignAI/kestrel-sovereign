"""Native content encoding and one MIME policy for message admission."""

import json
from collections.abc import Sequence
from email import policy
from typing import Any

from a2a.types import Part
from a2a.utils.errors import ContentTypeNotSupportedError, InvalidParamsError

DEFAULT_NATIVE_MEDIA_TYPES = ("text/plain", "application/json")


def encode_inline_json(value: Any) -> bytes:
    """Retain exact local JSON values in standard inline content.

    Escaping makes surrogateescaped filesystem names representable without
    losing bytes, and avoids protobuf Struct's binary64 conversion of ints.
    """
    return json.dumps(value, allow_nan=False, ensure_ascii=True).encode("utf-8")


def _media_type(value: str, *, allow_range: bool) -> tuple[str, str]:
    header = policy.default.header_factory("Content-Type", value)
    wildcard = "*" in header.maintype or "*" in header.subtype
    valid_range = (header.maintype == "*" and header.subtype == "*") or (
        "*" not in header.maintype and header.subtype == "*"
    )
    if header.defects or (wildcard and (not allow_range or not valid_range)):
        raise InvalidParamsError(message="A2A media types must be valid MIME types")
    return header.maintype, header.subtype


def part_media_type(part: Part) -> tuple[str, str]:
    if not isinstance(part, Part):
        raise TypeError("Native message content requires official Part values")
    kind = part.WhichOneof("content")
    if kind is None:
        raise InvalidParamsError(message="A2A parts require exactly one content member")
    if part.media_type:
        return _media_type(part.media_type, allow_range=False)
    # Optional MIME absence is an input-kind default, never recovery from a
    # malformed declared MIME type. Opaque file content is binary by default.
    defaults = {
        "text": "text/plain",
        "data": "application/json",
        "raw": "application/octet-stream",
        "url": "application/octet-stream",
    }
    return _media_type(defaults[kind], allow_range=False)


def validate_input_parts(parts: Sequence[Part], modes: Sequence[str]) -> None:
    supported = [_media_type(mode, allow_range=True) for mode in modes]
    for part in parts:
        main, sub = part_media_type(part)
        if not any(
            accepted_main in {"*", main} and accepted_sub in {"*", sub}
            for accepted_main, accepted_sub in supported
        ):
            raise ContentTypeNotSupportedError(
                message=f"A2A input mode is not supported: {main}/{sub}"
            )
