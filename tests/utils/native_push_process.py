"""Owned child running the real native push worker through an actual TLS close."""

import asyncio
import json
import sys

import pytest

from kestrel_sovereign.a2a.push_outbox import NativePushOutbox
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.push_config_store import (
    PushConfigStore,
    PushDeliveryAttempt,
)
from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
from kestrel_sovereign.multi_agent.liveness import process_identity
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_a2a_native_push_worker import worker
from tests.utils.outbound_tls import (
    hold_outbound_response_close,
    route_outbound_tls_peer,
)


async def main():
    request = json.loads(sys.stdin.readline())
    with pytest.MonkeyPatch.context() as patch:
        route_outbound_tls_peer(
            patch, hostname=request["hostname"], target_port=request["port"], dns_validations=4
        )

        def entered():
            # This pipe is the test controller protocol, not a diagnostic log.
            sys.stdout.write(json.dumps({"event": "native-response-close", "owner": process_identity(require_kernel_session=True)}) + "\n")
            sys.stdout.flush()

        barrier = hold_outbound_response_close(patch, entered_callback=entered)
        if request["backend_type"] == "sqlite":
            backend = SQLiteBackend(request["connection"])
        elif request["backend_type"] == "postgres":
            backend = PostgresBackend(request["connection"], min_pool_size=1, max_pool_size=2)
        else:
            raise ValueError("Native push child requires the selected real test backend")
        await backend.connect()
        try:
            scope = NativeRecipientScope(request["recipient_id"], request["tenant"])
            outbox = NativePushOutbox(
                PushConfigStore(TaskStore(backend, scope=scope, owns_backend=False)),
                DurableSignalStore(backend), retention_days=30, lease_seconds=60,
            )
            await outbox.initialize()
            if request.get("operation") == "inspect-original":
                fields = dict(request["attempt"])
                fields["_owner_identity_bytes"] = fields["_owner_identity_bytes"].encode("utf-8")
                attempt = PushDeliveryAttempt(**fields)
                result = {
                    "event": "foreign-process-observation",
                    "owner": process_identity(require_kernel_session=True),
                    "current": await outbox.configurations.delivery_is_current(attempt),
                    "retired": await outbox.configurations.retire_delivery(attempt),
                }
                sys.stdout.write(json.dumps(result) + "\n")
                sys.stdout.flush()
                return
            await worker(outbox).run_once()
        finally:
            barrier.release.set()
            await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
