"""Read the historical SPHINCS+ archival signatures made with pqcrypto 0.4.

That distribution and today's FIPS 205 verifier use the same module namespace
and different algorithms. Only an explicitly selected historical proof uses
the operator's isolated interpreter. The child receives public material only.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kestrel_sovereign.identity.succession import SuccessionStatement


ENV_PYTHON = "KESTREL_HISTORICAL_SPHINCS_PYTHON"
_KEY_BYTES = 32
_SIGNATURE_BYTES = 7856
_MAX_PAYLOAD = 1024 * 1024
_VERIFIED = b"verified sphincs-sha2-128s-simple\n"


class HistoricalVerifierError(ValueError):
    """The operator-selected archival verifier could not run."""


def verify_historical_archival_signature(
    statement: SuccessionStatement, *, trusted_archival_multibase: str,
) -> bool:
    from kestrel_sovereign.identity.succession import signable_payload
    from kestrel_sovereign.security.crypto_suite import ALG_SLH_DSA_SHA2_128S
    from kestrel_sovereign.security.signature_verification import (
        decode_verification_method_key, verification_method_kid,
    )

    method = statement.archival_verification_method
    entry = statement.archival_signature
    if (not isinstance(method, dict) or not isinstance(entry, dict)
            or not trusted_archival_multibase
            or method.get("publicKeyMultibase") != trusted_archival_multibase):
        return False
    method_id = method.get("id")
    encoded = entry.get("sig")
    if (not isinstance(method_id, str) or not method_id
            or not entry.get("kid") or entry["kid"] != verification_method_kid(method_id)
            or entry.get("alg") != ALG_SLH_DSA_SHA2_128S
            or not isinstance(encoded, str) or len(encoded) != 2 * _SIGNATURE_BYTES):
        return False
    decoded, _ = decode_verification_method_key(method)
    if decoded is None or decoded[0].alg_id != ALG_SLH_DSA_SHA2_128S:
        return False
    try:
        signature = bytes.fromhex(encoded)
    except ValueError:
        return False
    payload = signable_payload(statement)
    public = decoded[1]
    if (len(signature) != _SIGNATURE_BYTES or len(public) != _KEY_BYTES
            or not 0 < len(payload) <= _MAX_PAYLOAD):
        return False
    interpreter = os.environ.get(ENV_PYTHON)
    if not interpreter or not Path(interpreter).is_absolute():
        raise HistoricalVerifierError(f"{ENV_PYTHON} must name the absolute path to the isolated pqcrypto 0.4.0 Python")
    try:
        result = subprocess.run(
            [interpreter, "-I", str(Path(__file__).resolve())],
            input=public + signature + payload,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env={},
            timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HistoricalVerifierError("Historical SPHINCS+ verification process unavailable") from exc
    if result.returncode == 1 and result.stdout == b"invalid\n":
        return False
    if result.returncode != 0 or result.stdout != _VERIFIED:
        raise HistoricalVerifierError("Historical SPHINCS+ verifier requires a working pqcrypto 0.4.0 environment")
    return True


def _verify_public_input() -> int:
    """The isolated interpreter never imports the main Kestrel environment."""
    try:
        if version("pqcrypto") != "0.4.0":
            return 2
        from pqcrypto.sign import sphincs_sha2_128s_simple
    except (PackageNotFoundError, ImportError):
        return 2
    boundary = _KEY_BYTES + _SIGNATURE_BYTES
    data = sys.stdin.buffer.read(boundary + _MAX_PAYLOAD + 1)
    if not boundary < len(data) <= boundary + _MAX_PAYLOAD:
        return 2
    valid = sphincs_sha2_128s_simple.verify(data[:_KEY_BYTES], data[boundary:], data[_KEY_BYTES:boundary])
    sys.stdout.buffer.write(_VERIFIED if valid is True else b"invalid\n")
    return 0 if valid is True else 1


if __name__ == "__main__":
    raise SystemExit(_verify_public_input())
