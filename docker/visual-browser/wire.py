"""Bounded private IPC; credentials arrive on stdin, never in argv or env."""
import json
import re
from pathlib import Path

DIRECTORY = Path("/tmp/kestrel-browser")
MAX_REQUEST = 512 * 1024  # Includes worst-case JSON escaping of bounded Unicode text.
MAX_RESPONSE = 23 * 1024 * 1024
SECRET = re.compile(r"^[a-f0-9]{64}$")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def decode(data):
    def invalid(_):
        raise ValueError("non-finite JSON value")
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=invalid)


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"
