"""Fixed supervisor and two role children; byte bridge, never an RPC engine."""
import sys
from kestrel_sovereign.visual_computer.worker_entry import ProcessLifetime

_process_lifetime = None
if __name__ == "__main__" and sys.argv[1:] != ["--initialize"]:
    _process_lifetime = ProcessLifetime()
    if _process_lifetime.requested:
        raise SystemExit(0)  # No resource has been admitted.

import asyncio
import hmac
import os
from pathlib import Path
import secrets

from wire import MAX_REQUEST, decode, encode
from kestrel_sovereign.visual_computer.worker_broker import (
    LaneRole, WorkerLane, require_ipc, server_context, supervise, validate_configuration,
)

PRIVATE = Path("/tmp/kestrel-broker")


def initialize(raw):
    if len(raw) > MAX_REQUEST:
        raise ValueError("bounded_original_broker_bootstrap_required")
    value = decode(raw)
    validate_configuration(value)
    if not hmac.compare_digest(value["broker_incarnation"], (PRIVATE/"incarnation").read_text()):
        raise ValueError("original_broker_incarnation_differs")
    (PRIVATE/"initialized").touch(mode=0o600, exist_ok=False)
    path = PRIVATE/"bootstrap.pending"
    with open(path, "xb", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        stream.write(raw)
    os.rename(path, PRIVATE/"bootstrap.json")
    return encode({"schema_version": 1, "submitted": True})


async def main(lifecycle):
    return await ProcessLifetime.run(lifecycle, _main(lifecycle))


async def _main(lifecycle):
    if lifecycle.requested:
        return
    if os.getuid() != 10001 or os.getgid() != 10003:
        raise ValueError("original_broker_role_identity_required")
    os.umask(0o077)
    loop, retirement, lane = asyncio.get_running_loop(), asyncio.Event(), None
    def withdraw():
        retirement.set()
        if lane is not None:
            lane.revoke()
    lifecycle.bind(loop, withdraw)
    if lifecycle.requested:
        return
    if not sys.argv[1:]:
        PRIVATE.mkdir(mode=0o700, exist_ok=False)
        (PRIVATE/"incarnation").write_text(secrets.token_hex(32))
        path = PRIVATE/"bootstrap.json"
        if not await ProcessLifetime.wait_for_bootstrap(lifecycle, path, retirement):
            return
        with path.open("rb") as stream:
            raw = stream.read(MAX_REQUEST+1)
        path.unlink()
        if len(raw) > MAX_REQUEST:
            raise ValueError("bounded_original_broker_bootstrap_required")
        value = decode(raw)
        validate_configuration(value)
        if not hmac.compare_digest(value["broker_incarnation"], (PRIVATE/"incarnation").read_text()):
            raise ValueError("original_broker_incarnation_differs")
        await supervise(value, __file__, _retirement=retirement)
    elif len(sys.argv) == 3 and sys.argv[1] == "--lane":
        role = LaneRole(sys.argv[2])
        raw = sys.stdin.buffer.read(MAX_REQUEST+1)
        if lifecycle.requested:
            return
        if len(raw) > MAX_REQUEST:
            raise ValueError("bounded_original_lane_bootstrap_required")
        value = decode(raw)
        validate_configuration(value, role=role)
        original_ipc = require_ipc(value, role)
        original_ipc.close()
        # Normal role de-prioritization is not a real-time CPU guarantee.
        if role is LaneRole.NORMAL:
            os.nice(5)
        lane = WorkerLane(value, role, server_context(value, role))
        if lifecycle.requested:
            lane.revoke()
            await lane._close_originals()
            return
        await lane.serve()
    else:
        raise ValueError("fixed_original_broker_role_required")


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--initialize"]:
            sys.stdout.buffer.write(initialize(sys.stdin.buffer.read(MAX_REQUEST+1)))
        else:
            asyncio.run(main(_process_lifetime))
    except Exception:  # noqa: BLE001 — never log private TLS/bootstrap payloads
        raise SystemExit(1) from None
