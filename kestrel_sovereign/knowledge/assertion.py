"""Versioned, dependency-free semantic assertion value contracts.

This module is deliberately a value-model boundary.  It does not know about a
database, RDF codec, reasoner, or validation engine.  In particular, the
identity codec is defined here rather than delegated to a JSON or RDF library:
``kestrel-assertion-id-v1`` is SHA-256 over the RFC 8785 UTF-8 preimage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, TypeAlias
import unicodedata


MAPPING_SCHEMA_VERSION = 1
"""The first public JSON-safe mapping schema version."""

IDENTITY_VERSION = "kestrel-assertion-id-v1"
IRI_PROFILE = "iri-normalization-v1-rfc3986-200501"
LITERAL_PROFILE = "literal-v1-xsd11-20120405"

RDF_LANG_STRING = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
XSD_BOOLEAN = "http://www.w3.org/2001/XMLSchema#boolean"
XSD_DATE = "http://www.w3.org/2001/XMLSchema#date"
XSD_DATETIME = "http://www.w3.org/2001/XMLSchema#dateTime"
XSD_DATETIME_STAMP = "http://www.w3.org/2001/XMLSchema#dateTimeStamp"
XSD_DECIMAL = "http://www.w3.org/2001/XMLSchema#decimal"
XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"
XSD_TIME = "http://www.w3.org/2001/XMLSchema#time"


class AssertionValidationError(ValueError):
    """A public assertion value violates the v1 contract."""


class EpistemicState(str, Enum):
    ASSERTED = "asserted"
    OBSERVED = "observed"
    REPORTED = "reported"
    INFERRED = "inferred"
    HYPOTHESIS = "hypothesis"
    DISPUTED = "disputed"
    RETRACTED = "retracted"


class AssertionStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    QUARANTINED = "quarantined"
    DELETED = "deleted"


class Visibility(str, Enum):
    PRIVATE = "private"
    TENANT = "tenant"
    DELEGATED = "delegated"
    PUBLIC = "public"


_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_PCT_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_REG_NAME_RE = re.compile(r"^(?:[A-Za-z0-9._~!$&'()*+,;=-]|%[0-9A-Fa-f]{2})+$")
_USERINFO_RE = re.compile(r"^(?:[A-Za-z0-9._~!$&'()*+,;=:%-])*$")
_PATH_RE = re.compile(r"^(?:[A-Za-z0-9._~!$&'()*+,;=:@%/-])*$")
_QUERY_FRAGMENT_RE = re.compile(r"^(?:[A-Za-z0-9._~!$&'()*+,;=:@%/?-])*$")
_IPV_FUTURE_RE = re.compile(r"^[Vv][0-9A-Fa-f]+\.[A-Za-z0-9._~!$&'()*+,;=:-]+$")
_INTEGER_RE = re.compile(r"^[+-]?[0-9]+$")
_DECIMAL_RE = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$")
_DATE_RE = re.compile(r"^([0-9]{4})-([0-9]{2})-([0-9]{2})$")
_TIME_RE = re.compile(
    r"^([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,9}))?(Z|[+-][0-9]{2}:[0-9]{2})$"
)
_TIMESTAMP_RE = re.compile(
    r"^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.([0-9]{1,9}))?(Z|[+-][0-9]{2}:[0-9]{2})$"
)
_LANGUAGE_RE = re.compile(
    r"^(?:(?:[A-Za-z]{2,3}(?:-[A-Za-z]{3}){0,3}|[A-Za-z]{4}|[A-Za-z]{5,8})"
    r"(?:-[A-Za-z]{4})?(?:-(?:[A-Za-z]{2}|[0-9]{3}))?"
    r"(?:-(?:[A-Za-z0-9]{5,8}|[0-9][A-Za-z0-9]{3}))*"
    r"(?:-[0-9A-WY-Za-wy-z](?:-[A-Za-z0-9]{2,8})+)*"
    r"(?:-x(?:-[A-Za-z0-9]{1,8})+)?|x(?:-[A-Za-z0-9]{1,8})+)$"
)
_GRANDFATHERED_LANGUAGE_TAGS = frozenset(
    {
        "art-lojban", "cel-gaulish", "en-gb-oed", "i-ami", "i-bnn", "i-default",
        "i-enochian", "i-hak", "i-klingon", "i-lux", "i-mingo", "i-navajo",
        "i-pwn", "i-tao", "i-tay", "i-tsu", "no-bok", "no-nyn", "sgn-be-fr",
        "sgn-be-nl", "sgn-ch-de", "zh-guoyu", "zh-hakka", "zh-min", "zh-min-nan",
        "zh-xiang",
    }
)
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443, "ftp": 21}


def _fail(message: str) -> None:
    raise AssertionValidationError(message)


def _text(value: object, field_name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _fail(f"{field_name} must be a string")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        _fail(f"{field_name} must not contain a Unicode surrogate")
    if nonempty and not value:
        _fail(f"{field_name} must not be empty")
    return value


def _opaque_identifier(value: object, field_name: str, *, max_utf8_bytes: int = 512) -> str:
    value = _text(value, field_name)
    encoded = value.encode("utf-8")
    if len(encoded) > max_utf8_bytes:
        _fail(f"{field_name} must be at most {max_utf8_bytes} UTF-8 bytes")
    if unicodedata.normalize("NFC", value) != value:
        _fail(f"{field_name} must already be Unicode NFC")
    for char in value:
        codepoint = ord(char)
        if codepoint <= 0x1F or 0x7F <= codepoint <= 0x9F or char in "\t\n\v\f\r ":
            _fail(f"{field_name} contains prohibited control or ASCII whitespace")
    return value


def _plain_identifier(value: object, field_name: str) -> str:
    value = _text(value, field_name)
    if any(ord(char) <= 0x1F or 0x7F <= ord(char) <= 0x9F for char in value):
        _fail(f"{field_name} contains a control character")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        _fail(f"{name} mapping keys must be strings")
    return value


def _mapping_fields(
    value: object,
    name: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    data = _mapping(value, name)
    optional = optional or set()
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        _fail(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        _fail(f"{name} has ambiguous unknown fields: {', '.join(sorted(unknown))}")
    return data


def _mapping_version(data: Mapping[str, Any], name: str) -> None:
    """Decode v1 and the pre-versioned v1 shape, but never guess later forms."""
    version = data.get("schema_version", MAPPING_SCHEMA_VERSION)
    if type(version) is not int or version != MAPPING_SCHEMA_VERSION:
        _fail(f"{name}.schema_version must be {MAPPING_SCHEMA_VERSION}")


def _normalize_percent(component: str) -> str:
    result: list[str] = []
    position = 0
    while position < len(component):
        char = component[position]
        if char != "%":
            result.append(char)
            position += 1
            continue
        escape = component[position : position + 3]
        if not _PCT_RE.fullmatch(escape):
            _fail("unsupported_iri_form: invalid percent escape")
        octet = int(escape[1:], 16)
        decoded = chr(octet)
        if decoded in _UNRESERVED:
            result.append(decoded)
        else:
            result.append(f"%{octet:02X}")
        position += 3
    return "".join(result)


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 section 5.2.4, applied exactly once."""
    output = ""
    input_buffer = path
    while input_buffer:
        if input_buffer.startswith("../"):
            input_buffer = input_buffer[3:]
        elif input_buffer.startswith("./"):
            input_buffer = input_buffer[2:]
        elif input_buffer.startswith("/./"):
            input_buffer = "/" + input_buffer[3:]
        elif input_buffer == "/.":
            input_buffer = "/"
        elif input_buffer.startswith("/../"):
            input_buffer = "/" + input_buffer[4:]
            slash = output.rfind("/")
            output = output[:slash] if slash >= 0 else ""
        elif input_buffer == "/..":
            input_buffer = "/"
            slash = output.rfind("/")
            output = output[:slash] if slash >= 0 else ""
        elif input_buffer in {".", ".."}:
            input_buffer = ""
        else:
            if input_buffer.startswith("/"):
                next_slash = input_buffer.find("/", 1)
            else:
                next_slash = input_buffer.find("/")
            if next_slash == -1:
                output += input_buffer
                input_buffer = ""
            else:
                output += input_buffer[:next_slash]
                input_buffer = input_buffer[next_slash:]
    return output


def _split_authority(authority: str) -> tuple[str, str, int | None]:
    if not authority:
        _fail("unsupported_iri_form: authority host is empty")
    if authority.count("@") > 1:
        _fail("unsupported_iri_form: authority has multiple user-info separators")
    if "@" in authority:
        userinfo, host_port = authority.rsplit("@", 1)
        if not _USERINFO_RE.fullmatch(userinfo):
            _fail("unsupported_iri_form: invalid user-info")
    else:
        userinfo, host_port = "", authority

    port: int | None = None
    if host_port.startswith("["):
        closing = host_port.find("]")
        if closing < 0:
            _fail("unsupported_iri_form: unterminated IP-literal")
        host = host_port[: closing + 1]
        remainder = host_port[closing + 1 :]
        if remainder:
            if not remainder.startswith(":"):
                _fail("unsupported_iri_form: invalid authority after IP-literal")
            port_text = remainder[1:]
            if not port_text or not port_text.isascii() or not port_text.isdecimal():
                _fail("unsupported_iri_form: invalid port")
            port = int(port_text)
        literal = host[1:-1]
        if not (_IPV_FUTURE_RE.fullmatch(literal) or _valid_ipv6(literal)):
            _fail("unsupported_iri_form: invalid IP-literal")
    else:
        if host_port.count(":") > 1:
            _fail("unsupported_iri_form: IPv6 must use brackets")
        if ":" in host_port:
            host, port_text = host_port.rsplit(":", 1)
            if not port_text or not port_text.isascii() or not port_text.isdecimal():
                _fail("unsupported_iri_form: invalid port")
            port = int(port_text)
        else:
            host = host_port
        if not host or not _REG_NAME_RE.fullmatch(host):
            _fail("unsupported_iri_form: invalid host")

    if port is not None and not 0 <= port <= 65535:
        _fail("unsupported_iri_form: port is outside 0-65535")
    return userinfo, host, port


def _valid_ipv6(value: str) -> bool:
    try:
        ipaddress.IPv6Address(value)
    except ValueError:
        return False
    return True


def normalize_iri(value: object) -> str:
    """Apply the pinned ``iri-normalization-v1-rfc3986-200501`` profile."""
    value = _text(value, "IRI")
    if not value.isascii() or any(ord(char) <= 0x20 or 0x7F <= ord(char) <= 0x9F for char in value):
        _fail("unsupported_iri_form: IRI must be non-control ASCII")
    if not _SCHEME_RE.match(value):
        _fail("unsupported_iri_form: IRI must be absolute and have an RFC 3986 scheme")
    if "%" in value:
        position = 0
        while position < len(value):
            if value[position] == "%" and not _PCT_RE.fullmatch(value[position : position + 3]):
                _fail("unsupported_iri_form: invalid percent escape")
            position += 1

    scheme_end = value.index(":")
    scheme = value[:scheme_end].lower()
    rest = value[scheme_end + 1 :]
    fragment: str | None = None
    if "#" in rest:
        rest, fragment = rest.split("#", 1)
        if not _QUERY_FRAGMENT_RE.fullmatch(fragment):
            _fail("unsupported_iri_form: invalid fragment")
    query: str | None = None
    if "?" in rest:
        rest, query = rest.split("?", 1)
        if not _QUERY_FRAGMENT_RE.fullmatch(query):
            _fail("unsupported_iri_form: invalid query")

    authority: tuple[str, str, int | None] | None = None
    path = rest
    if rest.startswith("//"):
        authority_text, separator, suffix = rest[2:].partition("/")
        authority = _split_authority(authority_text)
        path = f"/{suffix}" if separator else ""
    if not _PATH_RE.fullmatch(path):
        _fail("unsupported_iri_form: invalid path")

    path = _remove_dot_segments(_normalize_percent(path))
    if authority is not None:
        userinfo, host, port = authority
        userinfo = _normalize_percent(userinfo)
        host = _normalize_percent(host)
        if host.startswith("["):
            host = "[" + host[1:-1].lower() + "]"
        else:
            host = host.lower()
        authority_text = f"{userinfo}@" if userinfo else ""
        authority_text += host
        if port is not None and _DEFAULT_PORTS.get(scheme) != port:
            authority_text += f":{port}"
        normalized = f"{scheme}://{authority_text}{path}"
    else:
        normalized = f"{scheme}:{path}"
    if query is not None:
        normalized += "?" + _normalize_percent(query)
    if fragment is not None:
        normalized += "#" + _normalize_percent(fragment)
    return normalized


def _normalize_language(value: object) -> str:
    value = _text(value, "language tag")
    normalized = value.lower()
    if normalized not in _GRANDFATHERED_LANGUAGE_TAGS and not _LANGUAGE_RE.fullmatch(value):
        _fail("language tag must be well-formed BCP 47")
    return normalized


def _normalize_timezone(offset: str) -> timedelta:
    if offset == "Z":
        return timedelta()
    sign = 1 if offset[0] == "+" else -1
    hours, minutes = int(offset[1:3]), int(offset[4:6])
    if hours > 14 or minutes > 59 or (hours == 14 and minutes != 0):
        _fail("timestamp timezone offset is outside -14:00 through +14:00")
    return sign * timedelta(hours=hours, minutes=minutes)


def _canonical_fraction(fraction: str | None) -> str:
    if not fraction:
        return ""
    fraction = fraction.rstrip("0")
    return f".{fraction}" if fraction else ""


def _normalized_datetime_string(value: object) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            _fail("timestamp must include a timezone offset")
        offset = value.utcoffset()
        if offset is None:  # defensive: a custom tzinfo can be inconsistent across calls
            _fail("timestamp must include a timezone offset")
        offset_seconds = offset.total_seconds()
        if offset_seconds % 60 or abs(offset_seconds) > 14 * 60 * 60:
            _fail("timestamp timezone offset is outside -14:00 through +14:00")
        try:
            utc = value.astimezone(timezone.utc)
        except OverflowError as error:
            _fail(f"timestamp is invalid: {error}")
        fraction = f".{utc.microsecond:06d}" if utc.microsecond else ""
        return utc.strftime("%Y-%m-%dT%H:%M:%S") + _canonical_fraction(fraction[1:] if fraction else None) + "Z"
    value = _text(value, "timestamp")
    match = _TIMESTAMP_RE.fullmatch(value)
    if not match:
        _fail("timestamp must be RFC 3339 date-time with an explicit timezone")
    year, month, day, hour, minute, second, fraction, offset = match.groups()
    try:
        local = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        utc = local - _normalize_timezone(offset)
    except (ValueError, OverflowError) as error:
        _fail(f"timestamp is invalid: {error}")
    return utc.strftime("%Y-%m-%dT%H:%M:%S") + _canonical_fraction(fraction) + "Z"


def _instant_sort_key(value: str) -> tuple[int, int, int, int, int, int, int]:
    """Chronological key for an already-normalized ``Instant`` value."""
    match = _TIMESTAMP_RE.fullmatch(value)
    if match is None or match.group(8) != "Z":
        _fail("timestamp must be a normalized UTC instant")
    year, month, day, hour, minute, second, fraction, _ = match.groups()
    fraction_nanoseconds = int((fraction or "").ljust(9, "0"))
    return (
        int(year), int(month), int(day), int(hour), int(minute), int(second), fraction_nanoseconds,
    )


def _normalize_literal(lexical_form: object, datatype_iri: object, language: object | None) -> tuple[str, str, str | None]:
    lexical_form = unicodedata.normalize("NFC", _text(lexical_form, "literal lexical_form", nonempty=False))
    datatype_iri = normalize_iri(datatype_iri)
    normalized_language = _normalize_language(language) if language is not None else None
    if normalized_language is not None:
        if datatype_iri != RDF_LANG_STRING:
            _fail("a language-tagged literal must use rdf:langString")
        return lexical_form, datatype_iri, normalized_language
    if datatype_iri == RDF_LANG_STRING:
        _fail("rdf:langString requires a language tag")
    if datatype_iri == XSD_STRING:
        return lexical_form, datatype_iri, None
    if datatype_iri == XSD_BOOLEAN:
        if lexical_form not in {"true", "false", "1", "0"}:
            _fail("invalid_literal_lexical_form: xsd:boolean")
        return ("true" if lexical_form in {"true", "1"} else "false"), datatype_iri, None
    if datatype_iri == XSD_INTEGER:
        if not _INTEGER_RE.fullmatch(lexical_form):
            _fail("invalid_literal_lexical_form: xsd:integer")
        integer = int(lexical_form)
        return str(integer), datatype_iri, None
    if datatype_iri == XSD_DECIMAL:
        if not _DECIMAL_RE.fullmatch(lexical_form):
            _fail("invalid_literal_lexical_form: xsd:decimal")
        try:
            number = Decimal(lexical_form)
        except InvalidOperation as error:  # guarded by the grammar; protects future edits
            _fail(f"invalid_literal_lexical_form: xsd:decimal ({error})")
        if number.is_zero():
            return "0.0", datatype_iri, None
        rendered = format(number, "f")
        sign = ""
        if rendered.startswith("-"):
            sign, rendered = "-", rendered[1:]
        whole, _, fractional = rendered.partition(".")
        whole = whole.lstrip("0") or "0"
        fractional = fractional.rstrip("0")
        return f"{sign}{whole}.{fractional or '0'}", datatype_iri, None
    if datatype_iri == XSD_DATE:
        match = _DATE_RE.fullmatch(lexical_form)
        if not match:
            _fail("invalid_literal_lexical_form: xsd:date")
        try:
            date(*(int(part) for part in match.groups()))
        except ValueError as error:
            _fail(f"invalid_literal_lexical_form: xsd:date ({error})")
        return lexical_form, datatype_iri, None
    if datatype_iri == XSD_TIME:
        match = _TIME_RE.fullmatch(lexical_form)
        if not match:
            _fail("invalid_literal_lexical_form: xsd:time")
        hour, minute, second, fraction, offset = match.groups()
        try:
            local = datetime(2000, 1, 1, int(hour), int(minute), int(second))
            utc = local - _normalize_timezone(offset)
        except ValueError as error:
            _fail(f"invalid_literal_lexical_form: xsd:time ({error})")
        return utc.strftime("%H:%M:%S") + _canonical_fraction(fraction) + "Z", datatype_iri, None
    if datatype_iri in {XSD_DATETIME, XSD_DATETIME_STAMP}:
        return _normalized_datetime_string(lexical_form), datatype_iri, None
    _fail("unsupported_literal_datatype")


@dataclass(frozen=True, slots=True)
class Resource:
    """Base marker for a resource term; use one of its three concrete forms."""

    kind: ClassVar[str]

    @classmethod
    def from_mapping(cls, value: object) -> Resource:
        data = _mapping(value, "resource")
        kind = data.get("kind")
        if kind == "iri":
            return IRI.from_mapping(data)
        if kind == "blank":
            return BlankNode.from_mapping(data)
        if kind == "local":
            return LocalIdentifier.from_mapping(data)
        _fail("resource.kind must be one of iri, blank, or local")


@dataclass(frozen=True, slots=True)
class IRI(Resource):
    """An absolute term normalized by the v1 RFC 3986 profile."""

    value: str
    kind: ClassVar[str] = "iri"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", normalize_iri(self.value))

    def identity_mapping(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}

    def to_mapping(self) -> dict[str, object]:
        return {"schema_version": MAPPING_SCHEMA_VERSION, **self.identity_mapping()}

    @classmethod
    def from_mapping(cls, value: object) -> IRI:
        data = _mapping_fields(value, "IRI", required={"kind", "value"}, optional={"schema_version"})
        _mapping_version(data, "IRI")
        if data["kind"] != cls.kind:
            _fail("IRI.kind must be iri")
        return cls(data["value"])


@dataclass(frozen=True, slots=True)
class BlankNode(Resource):
    """A source-side blank identifier.  It is intentionally not identity-safe."""

    value: str
    kind: ClassVar[str] = "blank"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _plain_identifier(self.value, "blank node value"))

    def to_mapping(self) -> dict[str, object]:
        return {"schema_version": MAPPING_SCHEMA_VERSION, "kind": self.kind, "value": self.value}

    @classmethod
    def from_mapping(cls, value: object) -> BlankNode:
        data = _mapping_fields(value, "BlankNode", required={"kind", "value"}, optional={"schema_version"})
        _mapping_version(data, "BlankNode")
        if data["kind"] != cls.kind:
            _fail("BlankNode.kind must be blank")
        return cls(data["value"])


@dataclass(frozen=True, slots=True)
class LocalIdentifier(Resource):
    """A source-local identifier.  It cannot be silently promoted to an IRI."""

    value: str
    kind: ClassVar[str] = "local"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _plain_identifier(self.value, "local identifier value"))

    def to_mapping(self) -> dict[str, object]:
        return {"schema_version": MAPPING_SCHEMA_VERSION, "kind": self.kind, "value": self.value}

    @classmethod
    def from_mapping(cls, value: object) -> LocalIdentifier:
        data = _mapping_fields(value, "LocalIdentifier", required={"kind", "value"}, optional={"schema_version"})
        _mapping_version(data, "LocalIdentifier")
        if data["kind"] != cls.kind:
            _fail("LocalIdentifier.kind must be local")
        return cls(data["value"])


@dataclass(frozen=True, slots=True)
class Literal:
    """A typed/language literal, normalized under the pinned literal-v1 profile.

    ``direction`` is represented so source adapters cannot conflate an RDF 1.2
    directional string with a normal language literal.  It is rejected when a
    caller attempts to use it in a canonical assertion identity.
    """

    lexical_form: str
    datatype_iri: str = XSD_STRING
    language: str | None = None
    direction: str | None = None
    kind: ClassVar[str] = "literal"

    def __post_init__(self) -> None:
        direction = self.direction
        if direction is not None and direction not in {"ltr", "rtl"}:
            _fail("literal direction must be ltr or rtl")
        lexical_form = unicodedata.normalize("NFC", _text(self.lexical_form, "literal lexical_form", nonempty=False))
        datatype_iri = normalize_iri(self.datatype_iri)
        language = _normalize_language(self.language) if self.language is not None else None
        if direction is not None:
            if language is None or datatype_iri != RDF_LANG_STRING:
                _fail("a directional literal must be a language-tagged rdf:langString")
        else:
            lexical_form, datatype_iri, language = _normalize_literal(lexical_form, datatype_iri, language)
        object.__setattr__(self, "lexical_form", lexical_form)
        object.__setattr__(self, "datatype_iri", datatype_iri)
        object.__setattr__(self, "language", language)

    @property
    def value(self) -> str:
        """The canonical lexical form, named ``value`` in identity mappings."""
        return self.lexical_form

    def identity_mapping(self) -> dict[str, str | None]:
        if self.direction is not None:
            _fail("unsupported_rdf12_term: directional literals have no v1 assertion identity")
        return {
            "datatype": self.datatype_iri,
            "kind": self.kind,
            "language": self.language,
            "value": self.lexical_form,
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "kind": self.kind,
            "lexical_form": self.lexical_form,
            "datatype_iri": self.datatype_iri,
            "language": self.language,
            "direction": self.direction,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Literal:
        data = _mapping_fields(
            value,
            "Literal",
            required={"kind", "lexical_form", "datatype_iri", "language", "direction"},
            optional={"schema_version"},
        )
        _mapping_version(data, "Literal")
        if data["kind"] != cls.kind:
            _fail("Literal.kind must be literal")
        return cls(
            lexical_form=data["lexical_form"],
            datatype_iri=data["datatype_iri"],
            language=data["language"],
            direction=data["direction"],
        )


AssertionObject: TypeAlias = IRI | Literal


@dataclass(frozen=True, slots=True)
class Instant:
    """A UTC RFC 3339 instant retaining up to nine fractional-second digits."""

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _normalized_datetime_string(self.value))

    @classmethod
    def coerce(cls, value: Instant | datetime | str) -> Instant:
        if isinstance(value, cls):
            return value
        if isinstance(value, datetime):
            return cls(_normalized_datetime_string(value))
        return cls(value)

    def to_mapping(self) -> dict[str, object]:
        return {"schema_version": MAPPING_SCHEMA_VERSION, "value": self.value}

    @classmethod
    def from_mapping(cls, value: object) -> Instant:
        data = _mapping_fields(value, "Instant", required={"value"}, optional={"schema_version"})
        _mapping_version(data, "Instant")
        return cls(data["value"])


@dataclass(frozen=True, slots=True)
class TemporalInterval:
    """A closed or open interval with normalized UTC endpoints."""

    start: Instant | datetime | str | None = None
    end: Instant | datetime | str | None = None

    def __post_init__(self) -> None:
        start = Instant.coerce(self.start) if self.start is not None else None
        end = Instant.coerce(self.end) if self.end is not None else None
        if start is None and end is None:
            _fail("a temporal interval must have a start or end")
        if start is not None and end is not None and _instant_sort_key(start.value) > _instant_sort_key(end.value):
            _fail("temporal interval start must not be after its end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "start": self.start.to_mapping() if self.start else None,
            "end": self.end.to_mapping() if self.end else None,
        }

    @classmethod
    def from_mapping(cls, value: object) -> TemporalInterval:
        data = _mapping_fields(value, "TemporalInterval", required={"start", "end"}, optional={"schema_version"})
        _mapping_version(data, "TemporalInterval")
        start = Instant.from_mapping(data["start"]) if data["start"] is not None else None
        end = Instant.from_mapping(data["end"]) if data["end"] is not None else None
        return cls(start=start, end=end)


@dataclass(frozen=True, slots=True)
class OntologyRef:
    """The immutable schema/ontology interpretation accepted for a revision."""

    namespace: str
    version: str
    content_digest: str
    compatibility_profile: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _plain_identifier(self.namespace, "ontology namespace"))
        object.__setattr__(self, "version", _plain_identifier(self.version, "ontology version"))
        object.__setattr__(self, "content_digest", _plain_identifier(self.content_digest, "ontology content_digest"))
        object.__setattr__(self, "compatibility_profile", _plain_identifier(self.compatibility_profile, "ontology compatibility_profile"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "namespace": self.namespace,
            "version": self.version,
            "content_digest": self.content_digest,
            "compatibility_profile": self.compatibility_profile,
        }

    @classmethod
    def from_mapping(cls, value: object) -> OntologyRef:
        data = _mapping_fields(
            value,
            "OntologyRef",
            required={"namespace", "version", "content_digest", "compatibility_profile"},
            optional={"schema_version"},
        )
        _mapping_version(data, "OntologyRef")
        return cls(**{key: data[key] for key in ("namespace", "version", "content_digest", "compatibility_profile")})


@dataclass(frozen=True, slots=True)
class SourceOccurrence:
    """Immutable source-side provenance for a particular evidence encounter.

    An assertion records source-occurrence IDs in its direct lineage.  This
    companion value type lets callers carry the source locator and retention-
    safe metadata without leaking a storage row or an RDF object into the
    public assertion contract.
    """

    source_occurrence_id: str
    source_kind: str
    locator: str
    received_at: Instant | datetime | str
    content_digest: str | None = None
    actor: str | None = None
    selector: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_occurrence_id", "source_kind", "locator"):
            object.__setattr__(self, name, _plain_identifier(getattr(self, name), f"source occurrence {name}"))
        object.__setattr__(self, "received_at", Instant.coerce(self.received_at))
        for name in ("content_digest", "actor", "selector"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _plain_identifier(value, f"source occurrence {name}"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "source_occurrence_id": self.source_occurrence_id,
            "source_kind": self.source_kind,
            "locator": self.locator,
            "received_at": self.received_at.to_mapping(),
            "content_digest": self.content_digest,
            "actor": self.actor,
            "selector": self.selector,
        }

    @classmethod
    def from_mapping(cls, value: object) -> SourceOccurrence:
        fields = {
            "source_occurrence_id", "source_kind", "locator", "received_at", "content_digest", "actor", "selector",
        }
        data = _mapping_fields(value, "SourceOccurrence", required=fields, optional={"schema_version"})
        _mapping_version(data, "SourceOccurrence")
        return cls(
            source_occurrence_id=data["source_occurrence_id"],
            source_kind=data["source_kind"],
            locator=data["locator"],
            received_at=Instant.from_mapping(data["received_at"]),
            content_digest=data["content_digest"],
            actor=data["actor"],
            selector=data["selector"],
        )


def _unique_identifiers(values: Sequence[object], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail(f"{field_name} must be an ordered sequence")
    normalized = tuple(_plain_identifier(value, field_name) for value in values)
    if not normalized:
        _fail(f"{field_name} must not be empty")
    if len(set(normalized)) != len(normalized):
        _fail(f"{field_name} must be an ordered set without duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class DirectLineage:
    """Direct provenance, represented by a non-empty ordered source-ID set."""

    source_occurrence_ids: tuple[str, ...]
    kind: ClassVar[str] = "direct"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_occurrence_ids", _unique_identifiers(self.source_occurrence_ids, "source_occurrence_ids"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "kind": self.kind,
            "source_occurrence_ids": list(self.source_occurrence_ids),
        }

    @classmethod
    def from_mapping(cls, value: object) -> DirectLineage:
        data = _mapping_fields(value, "DirectLineage", required={"kind", "source_occurrence_ids"}, optional={"schema_version"})
        _mapping_version(data, "DirectLineage")
        if data["kind"] != cls.kind:
            _fail("DirectLineage.kind must be direct")
        return cls(_mapping_sequence(data["source_occurrence_ids"], "DirectLineage.source_occurrence_ids"))


@dataclass(frozen=True, slots=True)
class DerivedLineage:
    """Complete lineage required for a derived/inferred assertion revision."""

    rule_id: str
    engine_version: str
    profile_version: str
    input_revision_ids: tuple[str, ...]
    input_digest: str
    run_id: str
    generated_at: Instant | datetime | str
    derivation_reference: str | None = None
    kind: ClassVar[str] = "derived"

    def __post_init__(self) -> None:
        for name in ("rule_id", "engine_version", "profile_version", "input_digest", "run_id"):
            object.__setattr__(self, name, _plain_identifier(getattr(self, name), f"derived lineage {name}"))
        object.__setattr__(self, "input_revision_ids", _unique_identifiers(self.input_revision_ids, "input_revision_ids"))
        object.__setattr__(self, "generated_at", Instant.coerce(self.generated_at))
        if self.derivation_reference is not None:
            object.__setattr__(self, "derivation_reference", _plain_identifier(self.derivation_reference, "derivation_reference"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "kind": self.kind,
            "rule_id": self.rule_id,
            "engine_version": self.engine_version,
            "profile_version": self.profile_version,
            "input_revision_ids": list(self.input_revision_ids),
            "input_digest": self.input_digest,
            "run_id": self.run_id,
            "generated_at": self.generated_at.to_mapping(),
            "derivation_reference": self.derivation_reference,
        }

    @classmethod
    def from_mapping(cls, value: object) -> DerivedLineage:
        fields = {
            "kind", "rule_id", "engine_version", "profile_version", "input_revision_ids",
            "input_digest", "run_id", "generated_at", "derivation_reference",
        }
        data = _mapping_fields(value, "DerivedLineage", required=fields, optional={"schema_version"})
        _mapping_version(data, "DerivedLineage")
        if data["kind"] != cls.kind:
            _fail("DerivedLineage.kind must be derived")
        return cls(
            rule_id=data["rule_id"],
            engine_version=data["engine_version"],
            profile_version=data["profile_version"],
            input_revision_ids=_mapping_sequence(data["input_revision_ids"], "DerivedLineage.input_revision_ids"),
            input_digest=data["input_digest"],
            run_id=data["run_id"],
            generated_at=Instant.from_mapping(data["generated_at"]),
            derivation_reference=data["derivation_reference"],
        )


Lineage: TypeAlias = DirectLineage | DerivedLineage


def lineage_from_mapping(value: object) -> Lineage:
    data = _mapping(value, "lineage")
    if data.get("kind") == DirectLineage.kind:
        return DirectLineage.from_mapping(data)
    if data.get("kind") == DerivedLineage.kind:
        return DerivedLineage.from_mapping(data)
    _fail("lineage.kind must be direct or derived")


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        _fail(f"{name} must be a string enum value")
    try:
        return enum_type(value)
    except ValueError:
        _fail(f"{name} has unsupported value {value!r}")


def _confidence(value: object) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        _fail("confidence must be a Decimal, integer, or decimal string; binary floats are not accepted")
    if not isinstance(value, (Decimal, int, str)):
        _fail("confidence must be a Decimal, integer, or decimal string")
    try:
        confidence = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        _fail(f"confidence is invalid: {error}")
    if not confidence.is_finite() or not Decimal("0") <= confidence <= Decimal("1"):
        _fail("confidence must be within the inclusive range [0, 1]")
    return confidence


def _finite_decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        _fail(f"{field_name} must be a Decimal, integer, or decimal string; binary floats are not accepted")
    if not isinstance(value, (Decimal, int, str)):
        _fail(f"{field_name} must be a Decimal, integer, or decimal string")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        _fail(f"{field_name} is invalid: {error}")
    if not result.is_finite():
        _fail(f"{field_name} must be finite")
    return result


def _decimal_mapping(value: Decimal) -> str:
    return format(value, "f")


def _mapping_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{field_name} must be an array")
    return tuple(value)


def _object_from_mapping(value: object) -> AssertionObject:
    data = _mapping(value, "assertion object")
    if data.get("kind") == "iri":
        return IRI.from_mapping(data)
    if data.get("kind") == "literal":
        return Literal.from_mapping(data)
    _fail("assertion object must be an IRI or literal")


def identity_preimage(*, tenant_id: object, subject: IRI, predicate: IRI, object: AssertionObject) -> bytes:
    """Return the exact RFC 8785 v1 identity preimage bytes.

    The preimage contains only string/null values, so Python's sorted compact
    JSON encoding is byte-for-byte the RFC 8785 encoding for this closed
    grammar.  Keeping it here makes the assertion hash independent of storage
    layout and any RDF/JSON implementation.
    """
    tenant_id = _opaque_identifier(tenant_id, "tenant_id")
    if not isinstance(subject, IRI):
        _fail("assertion subject must be an IRI")
    if not isinstance(predicate, IRI):
        _fail("assertion predicate must be an IRI")
    if not isinstance(object, (IRI, Literal)):
        _fail("assertion object must be an IRI or literal")
    object_mapping: dict[str, str | None]
    if isinstance(object, Literal):
        object_mapping = object.identity_mapping()
    else:
        object_mapping = {
            "datatype": None,
            "kind": "iri",
            "language": None,
            "value": object.value,
        }
    preimage = {
        "identity_version": IDENTITY_VERSION,
        "object": object_mapping,
        "predicate": predicate.value,
        "subject": subject.identity_mapping(),
        "tenant_id": tenant_id,
    }
    return json.dumps(preimage, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def derive_assertion_id(*, tenant_id: object, subject: IRI, predicate: IRI, object: AssertionObject) -> str:
    """Derive the sole valid v1 assertion ID for a tenant-scoped claim."""
    digest = hashlib.sha256(identity_preimage(tenant_id=tenant_id, subject=subject, predicate=predicate, object=object)).hexdigest()
    return f"urn:kestrel:assertion:sha256:{digest}"


_ASSERTION_ID_RE = re.compile(r"^urn:kestrel:assertion:sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class Assertion:
    """An immutable canonical assertion revision value.

    This is intentionally not a persistence model: it has no tables, foreign
    keys, RDF values, or backend handles.  ``assertion_id`` can be supplied by
    a caller only when it exactly equals the deterministic v1 identity.
    """

    tenant_id: str
    owning_agent_id: str
    subject: IRI
    predicate: IRI
    object: AssertionObject
    revision_id: str
    confidence: Decimal | int | str
    confidence_method: str
    confidence_basis: str
    epistemic_state: EpistemicState | str
    asserted_at: Instant | datetime | str
    ontology_version: OntologyRef
    lineage: Lineage
    privacy_classification: str
    release_policy_reference: str
    observed_time: TemporalInterval | None = None
    valid_time: TemporalInterval | None = None
    status: AssertionStatus | str = AssertionStatus.ACTIVE
    supersedes_revision_id: str | None = None
    visibility: Visibility | str = Visibility.PRIVATE
    identity_version: str = IDENTITY_VERSION
    iri_profile: str = IRI_PROFILE
    literal_profile: str = LITERAL_PROFILE
    assertion_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _opaque_identifier(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "owning_agent_id", _plain_identifier(self.owning_agent_id, "owning_agent_id"))
        if not isinstance(self.subject, IRI):
            _fail("assertion subject must be a normalized IRI; blank/local identifiers are source-only")
        if not isinstance(self.predicate, IRI):
            _fail("assertion predicate must be a normalized IRI")
        if not isinstance(self.object, (IRI, Literal)):
            _fail("assertion object must be an IRI or Literal")
        if isinstance(self.object, Literal):
            self.object.identity_mapping()  # rejects directional RDF 1.2 strings at the boundary
        object.__setattr__(self, "revision_id", _plain_identifier(self.revision_id, "revision_id"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "confidence_method", _plain_identifier(self.confidence_method, "confidence_method"))
        object.__setattr__(self, "confidence_basis", _plain_identifier(self.confidence_basis, "confidence_basis"))
        object.__setattr__(self, "epistemic_state", _enum(self.epistemic_state, EpistemicState, "epistemic_state"))
        object.__setattr__(self, "asserted_at", Instant.coerce(self.asserted_at))
        if not isinstance(self.ontology_version, OntologyRef):
            _fail("ontology_version must be an OntologyRef")
        if not isinstance(self.lineage, (DirectLineage, DerivedLineage)):
            _fail("lineage must be DirectLineage or DerivedLineage")
        object.__setattr__(self, "privacy_classification", _plain_identifier(self.privacy_classification, "privacy_classification"))
        object.__setattr__(self, "release_policy_reference", _plain_identifier(self.release_policy_reference, "release_policy_reference"))
        if self.observed_time is not None and not isinstance(self.observed_time, TemporalInterval):
            _fail("observed_time must be a TemporalInterval or null")
        if self.valid_time is not None and not isinstance(self.valid_time, TemporalInterval):
            _fail("valid_time must be a TemporalInterval or null")
        object.__setattr__(self, "status", _enum(self.status, AssertionStatus, "status"))
        object.__setattr__(self, "visibility", _enum(self.visibility, Visibility, "visibility"))
        if self.identity_version != IDENTITY_VERSION:
            _fail(f"identity_version must be {IDENTITY_VERSION!r}")
        if self.iri_profile != IRI_PROFILE:
            _fail(f"iri_profile must be {IRI_PROFILE!r}")
        if self.literal_profile != LITERAL_PROFILE:
            _fail(f"literal_profile must be {LITERAL_PROFILE!r}")
        if self.supersedes_revision_id is not None:
            object.__setattr__(self, "supersedes_revision_id", _plain_identifier(self.supersedes_revision_id, "supersedes_revision_id"))
            if self.supersedes_revision_id == self.revision_id:
                _fail("an assertion revision cannot supersede itself")
        if self.status in {AssertionStatus.RETRACTED, AssertionStatus.QUARANTINED, AssertionStatus.DELETED} and self.supersedes_revision_id is not None:
            _fail("retracted, quarantined, or deleted revisions cannot supersede another revision")
        if self.epistemic_state is EpistemicState.RETRACTED and self.status is not AssertionStatus.RETRACTED:
            _fail("epistemic_state retracted requires lifecycle status retracted")
        if self.status is AssertionStatus.RETRACTED and self.epistemic_state is not EpistemicState.RETRACTED:
            _fail("lifecycle status retracted requires epistemic_state retracted")
        if isinstance(self.lineage, DirectLineage) and self.epistemic_state is EpistemicState.INFERRED:
            _fail("an inferred assertion requires DerivedLineage")
        if isinstance(self.lineage, DerivedLineage):
            if self.epistemic_state is not EpistemicState.INFERRED:
                _fail("DerivedLineage requires epistemic_state inferred")
            if self.revision_id in self.lineage.input_revision_ids:
                _fail("derived lineage cannot name its own revision as input")
        derived_id = derive_assertion_id(
            tenant_id=self.tenant_id,
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
        )
        if self.assertion_id is not None:
            supplied = _text(self.assertion_id, "assertion_id")
            if not _ASSERTION_ID_RE.fullmatch(supplied):
                _fail("assertion_id must be a lowercase sha256 Kestrel assertion URN")
            if supplied != derived_id:
                _fail("caller-supplied assertion_id does not match the deterministic v1 identity")
        object.__setattr__(self, "assertion_id", derived_id)

    @property
    def identity_preimage(self) -> bytes:
        return identity_preimage(tenant_id=self.tenant_id, subject=self.subject, predicate=self.predicate, object=self.object)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "kind": "assertion",
            "assertion_id": self.assertion_id,
            "identity_version": self.identity_version,
            "revision_id": self.revision_id,
            "tenant_id": self.tenant_id,
            "owning_agent_id": self.owning_agent_id,
            "subject": self.subject.to_mapping(),
            "predicate": self.predicate.to_mapping(),
            "object": self.object.to_mapping(),
            "confidence": _decimal_mapping(self.confidence),
            "confidence_method": self.confidence_method,
            "confidence_basis": self.confidence_basis,
            "epistemic_state": self.epistemic_state.value,
            "asserted_at": self.asserted_at.to_mapping(),
            "observed_time": self.observed_time.to_mapping() if self.observed_time else None,
            "valid_time": self.valid_time.to_mapping() if self.valid_time else None,
            "status": self.status.value,
            "supersedes_revision_id": self.supersedes_revision_id,
            "visibility": self.visibility.value,
            "privacy_classification": self.privacy_classification,
            "release_policy_reference": self.release_policy_reference,
            "ontology_version": self.ontology_version.to_mapping(),
            "iri_profile": self.iri_profile,
            "literal_profile": self.literal_profile,
            "lineage": self.lineage.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> Assertion:
        required = {
            "kind", "assertion_id", "identity_version", "revision_id", "tenant_id", "owning_agent_id",
            "subject", "predicate", "object", "confidence", "confidence_method", "confidence_basis",
            "epistemic_state", "asserted_at", "observed_time", "valid_time", "status",
            "supersedes_revision_id", "visibility", "privacy_classification", "release_policy_reference",
            "ontology_version", "iri_profile", "literal_profile", "lineage",
        }
        data = _mapping_fields(value, "Assertion", required=required, optional={"schema_version"})
        _mapping_version(data, "Assertion")
        if data["kind"] != "assertion":
            _fail("Assertion.kind must be assertion")
        subject = IRI.from_mapping(data["subject"])
        predicate = IRI.from_mapping(data["predicate"])
        return cls(
            assertion_id=data["assertion_id"],
            identity_version=data["identity_version"],
            revision_id=data["revision_id"],
            tenant_id=data["tenant_id"],
            owning_agent_id=data["owning_agent_id"],
            subject=subject,
            predicate=predicate,
            object=_object_from_mapping(data["object"]),
            confidence=data["confidence"],
            confidence_method=data["confidence_method"],
            confidence_basis=data["confidence_basis"],
            epistemic_state=data["epistemic_state"],
            asserted_at=Instant.from_mapping(data["asserted_at"]),
            observed_time=TemporalInterval.from_mapping(data["observed_time"]) if data["observed_time"] is not None else None,
            valid_time=TemporalInterval.from_mapping(data["valid_time"]) if data["valid_time"] is not None else None,
            status=data["status"],
            supersedes_revision_id=data["supersedes_revision_id"],
            visibility=data["visibility"],
            privacy_classification=data["privacy_classification"],
            release_policy_reference=data["release_policy_reference"],
            ontology_version=OntologyRef.from_mapping(data["ontology_version"]),
            iri_profile=data["iri_profile"],
            literal_profile=data["literal_profile"],
            lineage=lineage_from_mapping(data["lineage"]),
        )


@dataclass(frozen=True, slots=True)
class AssertionQuery:
    """A typed read request that can only narrow a resolver-owned read scope."""

    subject: IRI | None = None
    predicate: IRI | None = None
    object: AssertionObject | None = None
    assertion_ids: tuple[str, ...] = field(default_factory=tuple)
    statuses: tuple[AssertionStatus | str, ...] = field(default_factory=tuple)
    epistemic_states: tuple[EpistemicState | str, ...] = field(default_factory=tuple)
    valid_at: Instant | datetime | str | None = None
    observed_at: Instant | datetime | str | None = None
    limit: int = 100
    cursor: str | None = None

    def __post_init__(self) -> None:
        if self.subject is not None and not isinstance(self.subject, IRI):
            _fail("query subject must be an IRI")
        if self.predicate is not None and not isinstance(self.predicate, IRI):
            _fail("query predicate must be an IRI")
        if self.object is not None and not isinstance(self.object, (IRI, Literal)):
            _fail("query object must be an IRI, Literal, or null")
        if isinstance(self.object, Literal):
            self.object.identity_mapping()
        object.__setattr__(self, "assertion_ids", _unique_identifiers(self.assertion_ids, "assertion_ids") if self.assertion_ids else ())
        statuses = tuple(_enum(value, AssertionStatus, "query status") for value in self.statuses)
        states = tuple(_enum(value, EpistemicState, "query epistemic_state") for value in self.epistemic_states)
        if len(set(statuses)) != len(statuses) or len(set(states)) != len(states):
            _fail("query status filters must not contain duplicates")
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "epistemic_states", states)
        if self.valid_at is not None:
            object.__setattr__(self, "valid_at", Instant.coerce(self.valid_at))
        if self.observed_at is not None:
            object.__setattr__(self, "observed_at", Instant.coerce(self.observed_at))
        if type(self.limit) is not int or not 1 <= self.limit <= 1000:
            _fail("query limit must be an integer in [1, 1000]")
        if self.cursor is not None:
            object.__setattr__(self, "cursor", _plain_identifier(self.cursor, "query cursor"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "kind": "assertion_query",
            "subject": self.subject.to_mapping() if self.subject else None,
            "predicate": self.predicate.to_mapping() if self.predicate else None,
            "object": self.object.to_mapping() if self.object else None,
            "assertion_ids": list(self.assertion_ids),
            "statuses": [item.value for item in self.statuses],
            "epistemic_states": [item.value for item in self.epistemic_states],
            "valid_at": self.valid_at.to_mapping() if self.valid_at else None,
            "observed_at": self.observed_at.to_mapping() if self.observed_at else None,
            "limit": self.limit,
            "cursor": self.cursor,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AssertionQuery:
        fields = {"kind", "subject", "predicate", "object", "assertion_ids", "statuses", "epistemic_states", "valid_at", "observed_at", "limit", "cursor"}
        data = _mapping_fields(value, "AssertionQuery", required=fields, optional={"schema_version"})
        _mapping_version(data, "AssertionQuery")
        if data["kind"] != "assertion_query":
            _fail("AssertionQuery.kind must be assertion_query")
        for field_name in ("assertion_ids", "statuses", "epistemic_states"):
            if isinstance(data[field_name], (str, bytes)) or not isinstance(data[field_name], Sequence):
                _fail(f"AssertionQuery.{field_name} must be an array")
        return cls(
            subject=IRI.from_mapping(data["subject"]) if data["subject"] is not None else None,
            predicate=IRI.from_mapping(data["predicate"]) if data["predicate"] is not None else None,
            object=_object_from_mapping(data["object"]) if data["object"] is not None else None,
            assertion_ids=tuple(data["assertion_ids"]),
            statuses=tuple(data["statuses"]),
            epistemic_states=tuple(data["epistemic_states"]),
            valid_at=Instant.from_mapping(data["valid_at"]) if data["valid_at"] is not None else None,
            observed_at=Instant.from_mapping(data["observed_at"]) if data["observed_at"] is not None else None,
            limit=data["limit"],
            cursor=data["cursor"],
        )


@dataclass(frozen=True, slots=True)
class AssertionResult:
    """A backend-neutral typed result; score is optional presentation metadata."""

    assertion: Assertion
    score: Decimal | int | str | None = None
    matched_revision_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.assertion, Assertion):
            _fail("AssertionResult.assertion must be an Assertion")
        if self.score is not None:
            score = _finite_decimal(self.score, "score")
            object.__setattr__(self, "score", score)
        if self.matched_revision_id is not None:
            object.__setattr__(self, "matched_revision_id", _plain_identifier(self.matched_revision_id, "matched_revision_id"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "kind": "assertion_result",
            "assertion": self.assertion.to_mapping(),
            "score": _decimal_mapping(self.score) if self.score is not None else None,
            "matched_revision_id": self.matched_revision_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> AssertionResult:
        fields = {"kind", "assertion", "score", "matched_revision_id"}
        data = _mapping_fields(value, "AssertionResult", required=fields, optional={"schema_version"})
        _mapping_version(data, "AssertionResult")
        if data["kind"] != "assertion_result":
            _fail("AssertionResult.kind must be assertion_result")
        return cls(
            assertion=Assertion.from_mapping(data["assertion"]),
            score=data["score"],
            matched_revision_id=data["matched_revision_id"],
        )


# Explicitly named aliases keep the first contract discoverable without making
# a second, near-equivalent representation in another package.
AssertionTerm: TypeAlias = IRI | Literal
SourceProvenance: TypeAlias = DirectLineage


__all__ = [
    "Assertion",
    "AssertionObject",
    "AssertionQuery",
    "AssertionResult",
    "AssertionStatus",
    "AssertionTerm",
    "AssertionValidationError",
    "BlankNode",
    "DerivedLineage",
    "DirectLineage",
    "EpistemicState",
    "IDENTITY_VERSION",
    "IRI",
    "IRI_PROFILE",
    "Instant",
    "LITERAL_PROFILE",
    "Lineage",
    "Literal",
    "LocalIdentifier",
    "MAPPING_SCHEMA_VERSION",
    "OntologyRef",
    "RDF_LANG_STRING",
    "Resource",
    "SourceOccurrence",
    "SourceProvenance",
    "TemporalInterval",
    "Visibility",
    "XSD_BOOLEAN",
    "XSD_DATE",
    "XSD_DATETIME",
    "XSD_DATETIME_STAMP",
    "XSD_DECIMAL",
    "XSD_INTEGER",
    "XSD_STRING",
    "XSD_TIME",
    "derive_assertion_id",
    "identity_preimage",
    "lineage_from_mapping",
    "normalize_iri",
]
