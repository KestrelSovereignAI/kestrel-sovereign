"""Controlled external SDK process for host-dispatch lifecycle regressions.

No provider network or Core state is emulated. The journal records actual
JSON-RPC callbacks in the child process, including its physical identity.
"""

import asyncio
import json
import os
from pathlib import Path

from kestrel_sdk.isolated_feature import IsolatedFeatureService, ToolMetadata


class HostedDispatchService(IsolatedFeatureService):
    def __init__(self):
        super().__init__(name="hosted-dispatch-process", version="1")
        self.binding = None
        self.advertise_inbound_producer(False)
        self.register_host_ingress("telegram-hosted-activate", self.activate)
        self.register_tool(ToolMetadata(
            name="telegram_send", description="Controlled offline send",
            input_schema={"type": "object", "properties": {
                "to": {"type": "string"}, "message": {"type": "string"}},
                "required": ["to", "message"]},
        ), self.send)
        self.record("started")

    def record(self, kind, **fields):
        with Path(os.environ["HOSTED_DISPATCH_JOURNAL"]).open("a") as journal:
            journal.write(json.dumps({"kind": kind, "pid": os.getpid(), **fields}) + "\n")

    async def activate(self, payload):
        self.binding = payload
        self.record("activate", binding=payload)
        return {"status": "ok", "http_status": 200, "transport": "hosted"}

    async def send(self, arguments):
        if self.binding is None or self.binding["chat_id"] != arguments["to"]:
            raise RuntimeError("Child has no matching hosted binding")
        self.record("send", arguments=arguments)
        return {"status": "ok", "confirmation": "Controlled send completed",
                "to": arguments["to"], "message_id": str(os.getpid())}


if __name__ == "__main__":
    asyncio.run(HostedDispatchService().run_stdio())
