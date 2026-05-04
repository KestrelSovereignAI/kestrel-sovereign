"""
Multikey / multibase encoding for Kestrel public keys.

Wave 1 sub-PR 2 of the Quantum Hardening epic (#921, #916). Implements
the W3C Multikey shape needed by identity-package v2 (Wave 1 sub-PR 3)
and ``did:web`` verification methods (Wave 2):

    publicKeyMultibase = "z" || base58btc(multicodec_prefix || raw_pubkey)

The ``z`` prefix is the multibase identifier for base58btc
(base58btc-encoded; no padding; alphabet is the Bitcoin alphabet).

Multicodec values for the algorithms registered in Waves 1-3:

| alg_id                     | multicodec hex | varint bytes |
|---                         |---             |---           |
| ecdsa-secp256k1-sha256     | 0xe7 (secp256k1-pub) | 0xe7 0x01 |
| ed25519                    | 0xed (ed25519-pub)   | 0xed 0x01 |
| ml-dsa-65                  | TBD (W3C work in progress) | placeholder |
| slh-dsa-sha2-128s          | TBD                | placeholder |

Only ``Secp256k1Suite``'s codec is used today. Wave 2 adds the ed25519
and ml-dsa-65 codecs alongside the new suites; Wave 3 adds slh-dsa-sha2-128s.
The ``public_key_multicodec`` class attribute on ``CryptoSuite`` is the
single source of truth: each suite owns its codec and registers via
``crypto_suite.register_suite``.

Why base58btc rather than the codebase's existing base32lower
-----------------------------------------------------------------

The CAR-builder already implements base32lower with multibase prefix
``b`` (per the IPLD/Filecoin convention). Multikey / DID Core mandates
base58btc with prefix ``z``. Both are W3C-blessed multibase encodings;
they coexist in the same document. We follow the spec for each surface.
"""

from __future__ import annotations

from typing import Tuple

# Bitcoin alphabet for base58btc (no 0, O, I, l)
_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_ALPHABET_REVERSE = {c: i for i, c in enumerate(_B58_ALPHABET)}

MULTIBASE_BASE58BTC_PREFIX = "z"


# ---------------------------------------------------------------------------
# Unsigned LEB128 varint (multicodec encoding)
# ---------------------------------------------------------------------------

def encode_varint(value: int) -> bytes:
    """Unsigned LEB128 varint encode. The multicodec convention.

    Values 0-127 → single byte. Larger values use the high bit as a
    continuation flag in 7-bit little-endian groups.
    """
    if value < 0:
        raise ValueError(f"varint encodes unsigned ints; got {value}")
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def decode_varint(data: bytes) -> Tuple[int, int]:
    """Decode an unsigned LEB128 varint from the start of ``data``.

    Returns ``(value, num_bytes_consumed)``. Raises ``ValueError`` on
    truncated or overlong input.
    """
    value = 0
    shift = 0
    for i, byte in enumerate(data):
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, i + 1
        shift += 7
        if shift > 63:
            raise ValueError("varint too long (>63 bits)")
    raise ValueError("varint truncated (no terminator byte)")


# ---------------------------------------------------------------------------
# base58btc
# ---------------------------------------------------------------------------

def base58btc_encode(data: bytes) -> str:
    """Encode bytes to base58btc (Bitcoin alphabet, no padding).

    Leading zero bytes are preserved as leading ``1`` characters per the
    Bitcoin convention.
    """
    if not data:
        return ""

    # Count leading zero bytes
    leading_zeros = 0
    for b in data:
        if b == 0:
            leading_zeros += 1
        else:
            break

    # Convert the rest as a big-endian integer
    n = int.from_bytes(data, "big")
    out = bytearray()
    while n > 0:
        n, rem = divmod(n, 58)
        out.append(_B58_ALPHABET[rem])
    out.reverse()

    # Prepend a '1' for each leading zero byte
    return ("1" * leading_zeros) + out.decode("ascii")


def base58btc_decode(s: str) -> bytes:
    """Decode a base58btc string back to bytes. Raises on non-alphabet chars."""
    if not s:
        return b""

    leading_ones = 0
    for ch in s:
        if ch == "1":
            leading_ones += 1
        else:
            break

    n = 0
    for ch in s:
        try:
            digit = _B58_ALPHABET_REVERSE[ord(ch)]
        except KeyError as e:
            raise ValueError(
                f"non-base58btc character {ch!r} in input"
            ) from e
        n = n * 58 + digit

    # n is the integer value of the non-leading-1 portion
    if n == 0:
        body = b""
    else:
        body = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return (b"\x00" * leading_ones) + body


# ---------------------------------------------------------------------------
# Multikey: multibase(base58btc, multicodec || raw_pubkey)
# ---------------------------------------------------------------------------

def public_key_to_multibase(suite, public_key) -> str:
    """Encode a suite's public key as a W3C Multikey-compatible string.

    Returns ``"z" + base58btc(multicodec_prefix || multikey_pubkey_bytes)``.

    Uses the suite's ``serialize_public_key_for_multikey`` method, NOT
    the legacy ``serialize_public_key`` — the multicodec table mandates
    a specific on-the-wire shape per algorithm (e.g. 33-byte compressed
    point for secp256k1-pub 0xe7, raw 32-byte for ed25519-pub 0xed) and
    other implementations will reject or rederive a different key shape
    if a different format is shipped under the same codec.

    Raises if the suite has no ``public_key_multicodec`` set or no
    multikey-specific serializer (Wave 2 suites must declare both
    before they can produce v2 verification methods).
    """
    codec = getattr(suite, "public_key_multicodec", b"")
    if not codec:
        from .crypto_suite import CryptoSuiteError
        raise CryptoSuiteError(
            f"Suite {type(suite).__name__} (alg_id={suite.alg_id!r}) has no "
            f"public_key_multicodec set; cannot produce a Multikey "
            f"publicKeyMultibase string."
        )
    raw = suite.serialize_public_key_for_multikey(public_key)
    return MULTIBASE_BASE58BTC_PREFIX + base58btc_encode(codec + raw)


def multibase_to_public_key(multibase_str: str):
    """Decode a Multikey ``z...`` string back to ``(suite, public_key)``.

    Looks up the registered suite by its ``public_key_multicodec`` prefix
    and routes the body through the suite's
    ``deserialize_public_key_from_multikey``. Raises if the prefix doesn't
    match any registered suite.
    """
    from .crypto_suite import _REGISTRY, CryptoSuiteError

    if not multibase_str.startswith(MULTIBASE_BASE58BTC_PREFIX):
        raise CryptoSuiteError(
            f"Expected multibase base58btc prefix {MULTIBASE_BASE58BTC_PREFIX!r}; "
            f"got {multibase_str[:1]!r}"
        )
    raw = base58btc_decode(multibase_str[len(MULTIBASE_BASE58BTC_PREFIX):])
    if not raw:
        raise CryptoSuiteError("empty multibase payload")

    # Decode the multicodec varint to find the algorithm
    codec_value, consumed = decode_varint(raw)
    codec_bytes = raw[:consumed]
    pub_bytes = raw[consumed:]

    for suite in _REGISTRY.values():
        if getattr(suite, "public_key_multicodec", b"") == codec_bytes:
            return suite, suite.deserialize_public_key_from_multikey(pub_bytes)

    raise CryptoSuiteError(
        f"No registered suite for multicodec 0x{codec_value:x}. "
        f"Registered suites: {sorted(_REGISTRY)}."
    )
