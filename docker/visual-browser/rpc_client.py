"""One bounded stdin/stdout exchange. Executed in the exact admitted container."""
import os
import hmac
import socket
import sys

from wire import DIRECTORY, MAX_REQUEST, MAX_RESPONSE, decode, encode
from worker import bootstrap


def exchange(raw, *, priority=False):
    if not raw.endswith(b"\n") or len(raw) > MAX_REQUEST:
        raise ValueError("invalid_request")
    decode(raw)
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(22)
        connection.connect(str(DIRECTORY / ("control.sock" if priority else "worker.sock")))
        connection.sendall(raw)
        with connection.makefile("rb") as stream:
            response = stream.readline(MAX_RESPONSE + 1)
        if not response.endswith(b"\n") or len(response) > MAX_RESPONSE:
            raise ValueError("invalid_response")
        decode(response)
        return response


def initialize(raw):
    # Trusted host bootstrap, not an agent RPC. Atomic rename prevents a reader
    # seeing partial JSON; O_EXCL and one-shot startup prohibit a second session.
    if len(raw) > MAX_REQUEST:
        raise ValueError("invalid_bootstrap")
    value = bootstrap(decode(raw))
    if not hmac.compare_digest(value["incarnation"], (DIRECTORY / "incarnation").read_text()):
        raise ValueError("worker_incarnation_mismatch")
    (DIRECTORY / "initialized").touch(mode=0o600, exist_ok=False)
    path = DIRECTORY / "bootstrap.pending"
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(raw)
    os.rename(path, DIRECTORY / "bootstrap.json")
    return encode({"schema_version": 1, "submitted": True})


if __name__ == "__main__":
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if sys.argv[1:] == ["initialize"]:
            result = initialize(raw)
        elif sys.argv[1:] == ["priority"]:
            result = exchange(raw, priority=True)
        elif not sys.argv[1:]:
            result = exchange(raw)
        else:
            raise ValueError("invalid_arguments")
        sys.stdout.buffer.write(result)
    except Exception:  # noqa: BLE001 — boundary keeps private browser details out of IPC/logs
        # The caller must reconcile a lost response, never infer no effect.
        sys.stdout.buffer.write(encode({"schema_version": 1, "error": "transport_outcome_unknown"}))
        raise SystemExit(1) from None
