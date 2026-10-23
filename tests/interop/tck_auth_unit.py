"""Independent TCK-environment unit checks with the real Go proof signer.

The fixture launches this file directly, so no Core imports or pytest plugins
can replace the TCK's protobuf schema. No protocol server is simulated here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

import httpx
from kestrel_tck_auth import KestrelTckAuth


class TckAuthRoutingTests(unittest.TestCase):
    """Authentication signs exactly the native fields carried by the request."""

    def test_http_path_fields_are_decoded_once(self):
        with tempfile.TemporaryDirectory(prefix="tck-auth-unit-") as work:
            root = Path(work)
            config_path = root / "peer.json"
            binary = Path(os.environ["KESTREL_TCK_GO_PEER"])
            config = {
                "keys": str(root / "keys.json"),
                "coreCardUrl": "http://127.0.0.1:1/card",
                "coreStable": "did:kestrel:recipient",
                "foreignDocument": {
                    "id": "did:web:tck-auth.test", "verificationMethod": [
                        {"id": "did:web:tck-auth.test#ed25519"},
                        {"id": "did:web:tck-auth.test#ml-dsa-65"},
                    ],
                },
                "extensions": ["urn:kestrel:test#signed-sender-v1"],
            }
            config_path.write_text(json.dumps(config))
            subprocess.run([str(binary), "keys", str(config_path)], capture_output=True, check=True, timeout=10)
            previous = os.environ.get("KESTREL_TCK_PEER_CONFIG")
            os.environ["KESTREL_TCK_PEER_CONFIG"] = str(config_path)
            try:
                endpoint = "http://127.0.0.1:1/native/interface"
                card = {"supportedInterfaces": [{"protocolBinding": "HTTP+JSON", "url": endpoint}]}
                auth = KestrelTckAuth(transport="http_json", agent_card=card)
                cases = [
                    ("", "task%2Fid"), ("", "task%252Fid"), ("", "task雪"),
                    ("team%2Fname", "task"), ("team/name", "task"),
                    ("team雪", "task%2Fid"),
                ]
                for tenant, task_id in cases:
                    with self.subTest(tenant=tenant, task_id=task_id):
                        prefix = "/" + quote(tenant, safe="") if tenant else ""
                        request = httpx.Request("GET", endpoint + prefix + "/tasks/" + quote(task_id, safe=""))
                        original = request.method, request.url, request.content
                        outgoing, = auth.auth_flow(request)
                        self.assertEqual((outgoing.method, outgoing.url, outgoing.content), original)
                        proof = outgoing.headers["Kestrel-A2A-Sender-Proof"]
                        claim = json.loads(base64.urlsafe_b64decode(proof.split(".")[0] + "=="))
                        params = {"id": task_id, **({"tenant": tenant} if tenant else {})}
                        # JCS for this string-only object has no numeric or
                        # UTF-16 property-order edge cases.
                        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                        self.assertEqual(claim["operation"], "GetTask")
                        self.assertEqual(claim["request_digest"], hashlib.sha256(canonical).hexdigest())
            finally:
                if previous is None:
                    os.environ.pop("KESTREL_TCK_PEER_CONFIG", None)
                else:
                    os.environ["KESTREL_TCK_PEER_CONFIG"] = previous


if __name__ == "__main__":
    unittest.main()
