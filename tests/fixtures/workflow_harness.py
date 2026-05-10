"""Deterministic WorkflowHarness for Phase 1 tests.

The harness gives workflow/gate tests a real SQLite-backed WorkflowStore,
SignalLogStore, SourceRegistry, and SignalDispatcher without reaching for
networked actors. Non-deterministic actor cassette replay lands in a later
chunk; this foundation pins the local deterministic path first.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from kestrel_sdk.signals import RedactionPolicy, SignalMode, SourceRegistration

from kestrel_sovereign.features.workflows.runner import WorkflowRunner
from kestrel_sovereign.features.workflows.signing import sign_workflow_spec
from kestrel_sovereign.features.workflows.store import WorkflowStore
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.storage.db import SQLiteBackend

CASSETTE_FORMAT = "kestrel-workflow-cassette-v1"
CASSETTE_KEY_ENV = "KESTREL_WORKFLOW_CASSETTE_KEY"
CASSETTE_SECRET_STORE_ENV = "KESTREL_WORKFLOW_CASSETTE_SECRET_STORE"


class WorkflowCassetteError(RuntimeError):
    """Raised when encrypted workflow cassette access fails closed."""


class EncryptedWorkflowCassetteStore:
    """Encrypted-at-rest cassette store for non-deterministic workflow actors.

    The envelope intentionally exposes only routing metadata. The cassette
    body, including red-team PR diffs and blocker findings, is AES-GCM
    encrypted with a per-owner key derived from the operator secret and
    owner DID, and AAD-bound to the owner DID and cassette id.
    """

    def __init__(
        self,
        root: Path,
        *,
        key: bytes | str,
        retention_seconds: int = 7 * 24 * 60 * 60,
    ) -> None:
        if isinstance(key, str):
            key = key.encode("utf-8")
        if not key:
            raise WorkflowCassetteError("workflow cassette key is required")
        if retention_seconds <= 0:
            raise WorkflowCassetteError(
                "workflow cassette retention_seconds must be positive"
            )
        self.root = Path(root)
        self._root_key = bytes(key)
        self.retention_seconds = retention_seconds

    @classmethod
    def from_environment(
        cls,
        root: Path,
        *,
        retention_seconds: int = 7 * 24 * 60 * 60,
    ) -> "EncryptedWorkflowCassetteStore":
        secret_store = os.environ.get(CASSETTE_SECRET_STORE_ENV)
        if secret_store != "approved":
            raise WorkflowCassetteError(
                "workflow cassette replay requires manual approval via "
                f"{CASSETTE_SECRET_STORE_ENV}=approved"
            )
        key = os.environ.get(CASSETTE_KEY_ENV)
        if not key:
            raise WorkflowCassetteError(
                f"workflow cassette replay requires {CASSETTE_KEY_ENV}"
            )
        return cls(root, key=key, retention_seconds=retention_seconds)

    def record(
        self,
        *,
        owner_did: str,
        cassette_id: str,
        payload: dict[str, Any],
        now: Optional[float] = None,
        retention_seconds: Optional[int] = None,
    ) -> Path:
        if not isinstance(payload, dict):
            raise WorkflowCassetteError("workflow cassette payload must be a dict")
        created_at = int(time.time() if now is None else now)
        ttl = self.retention_seconds if retention_seconds is None else retention_seconds
        if ttl <= 0:
            raise WorkflowCassetteError(
                "workflow cassette retention_seconds must be positive"
            )
        expires_at = created_at + ttl
        plaintext = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._owner_key(owner_did)).encrypt(
            nonce,
            plaintext,
            self._aad(owner_did, cassette_id),
        )
        envelope = {
            "format": CASSETTE_FORMAT,
            "version": 1,
            "owner_did": owner_did,
            "cassette_id_sha256": self._cassette_digest(owner_did, cassette_id),
            "created_at": created_at,
            "expires_at": expires_at,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(owner_did=owner_did, cassette_id=cassette_id)
        path.write_text(json.dumps(envelope, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def replay(
        self,
        *,
        owner_did: str,
        cassette_id: str,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        path = self.path_for(owner_did=owner_did, cassette_id=cassette_id)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WorkflowCassetteError("workflow cassette not found") from exc
        except json.JSONDecodeError as exc:
            raise WorkflowCassetteError("workflow cassette envelope is invalid") from exc
        self._validate_envelope(envelope, owner_did, cassette_id)
        observed = int(time.time() if now is None else now)
        if envelope["expires_at"] < observed:
            raise WorkflowCassetteError("workflow cassette is expired")
        try:
            nonce = base64.b64decode(envelope["nonce"])
            ciphertext = base64.b64decode(envelope["ciphertext"])
            plaintext = AESGCM(self._owner_key(owner_did)).decrypt(
                nonce,
                ciphertext,
                self._aad(owner_did, cassette_id),
            )
        except (ValueError, InvalidTag) as exc:
            raise WorkflowCassetteError(
                "workflow cassette authentication failed"
            ) from exc
        try:
            payload = json.loads(plaintext.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkflowCassetteError("workflow cassette payload is invalid") from exc
        if not isinstance(payload, dict):
            raise WorkflowCassetteError("workflow cassette payload must be a dict")
        return payload

    def purge_expired(self, *, now: Optional[float] = None) -> list[Path]:
        observed = int(time.time() if now is None else now)
        purged: list[Path] = []
        if not self.root.exists():
            return purged
        for path in self.root.glob("*.workflow-cassette.enc"):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            expires_at = envelope.get("expires_at")
            if isinstance(expires_at, int) and expires_at < observed:
                path.unlink()
                purged.append(path)
        return purged

    def path_for(self, *, owner_did: str, cassette_id: str) -> Path:
        digest = self._cassette_digest(owner_did, cassette_id)
        return self.root / f"{digest}.workflow-cassette.enc"

    def _owner_key(self, owner_did: str) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=owner_did.encode("utf-8"),
            info=b"kestrel-workflow-cassette-v1",
        ).derive(self._root_key)

    @staticmethod
    def _aad(owner_did: str, cassette_id: str) -> bytes:
        return (
            f"{CASSETTE_FORMAT}\nowner={owner_did}\n"
            f"cassette={cassette_id}"
        ).encode("utf-8")

    @staticmethod
    def _cassette_digest(owner_did: str, cassette_id: str) -> str:
        return hashlib.sha256(
            f"{owner_did}\0{cassette_id}".encode("utf-8")
        ).hexdigest()

    def _validate_envelope(
        self, envelope: Any, owner_did: str, cassette_id: str
    ) -> None:
        if not isinstance(envelope, dict):
            raise WorkflowCassetteError("workflow cassette envelope must be a dict")
        expected = {
            "format": CASSETTE_FORMAT,
            "version": 1,
            "owner_did": owner_did,
            "cassette_id_sha256": self._cassette_digest(owner_did, cassette_id),
        }
        for key, value in expected.items():
            if envelope.get(key) != value:
                raise WorkflowCassetteError(
                    f"workflow cassette envelope {key} mismatch"
                )
        for key in ("created_at", "expires_at"):
            if not isinstance(envelope.get(key), int):
                raise WorkflowCassetteError(
                    f"workflow cassette envelope {key} is invalid"
                )
        for key in ("nonce", "ciphertext"):
            if not isinstance(envelope.get(key), str):
                raise WorkflowCassetteError(
                    f"workflow cassette envelope {key} is invalid"
                )


def assert_no_plaintext_workflow_cassettes(root: Path) -> None:
    offenders = [
        path
        for path in Path(root).rglob("*.workflow-cassette.json")
        if ".git" not in path.parts
    ]
    if offenders:
        joined = ", ".join(str(path) for path in offenders[:5])
        raise WorkflowCassetteError(
            "plaintext workflow cassettes are forbidden; use encrypted "
            f"*.workflow-cassette.enc envelopes instead: {joined}"
        )


class HarnessAgent:
    def __init__(self, identity: AgentIdentity) -> None:
        self.identity = identity
        self.did = identity.legacy_did
        self.background_tasks: list[asyncio.Task] = []
        self.process_input_calls: list[str] = []

    async def process_input(self, prompt: str):
        self.process_input_calls.append(prompt)
        return "ok"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


class WorkflowHarness:
    """Async context manager for deterministic workflow runner tests."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        did: str = "did:web:k.example",
        cassette_key: bytes | str | None = None,
        cassette_dir: Path | None = None,
        cassette_retention_seconds: int = 7 * 24 * 60 * 60,
        require_cassette_secret_store: bool = False,
    ) -> None:
        self.tmp_path = tmp_path
        self.identity = _identity(did)
        self.agent = HarnessAgent(self.identity)
        self.backend: Optional[SQLiteBackend] = None
        self.registry = SourceRegistry()
        self.signal_store: Optional[SignalLogStore] = None
        self.store: Optional[WorkflowStore] = None
        self.dispatcher: Optional[SignalDispatcher] = None
        self.runner: Optional[WorkflowRunner] = None
        if cassette_key is not None and require_cassette_secret_store:
            raise WorkflowCassetteError(
                "pass either cassette_key or require_cassette_secret_store"
            )
        cassette_root = cassette_dir or tmp_path / ".workflow-cassettes"
        if require_cassette_secret_store:
            self.cassette_store: Optional[EncryptedWorkflowCassetteStore] = (
                EncryptedWorkflowCassetteStore.from_environment(
                    cassette_root,
                    retention_seconds=cassette_retention_seconds,
                )
            )
        elif cassette_key is not None:
            self.cassette_store = EncryptedWorkflowCassetteStore(
                cassette_root,
                key=cassette_key,
                retention_seconds=cassette_retention_seconds,
            )
        else:
            self.cassette_store = None

    async def __aenter__(self) -> "WorkflowHarness":
        backend = SQLiteBackend(str(self.tmp_path / "workflow-harness.db"))
        await backend.connect()
        self.backend = backend
        self.signal_store = SignalLogStore(backend)
        await self.signal_store.initialize()
        self.store = WorkflowStore(backend)
        await self.store.initialize()
        self.dispatcher = SignalDispatcher(
            agent=self.agent,
            registry=self.registry,
            lock_manager=OrderedLockManager(),
            store=self.signal_store,
        )
        self.runner = WorkflowRunner(
            store=self.store,
            dispatcher=self.dispatcher,
            registry=self.registry,
            agent_identity=self.identity,
            public_key_resolver=self.resolve_public_key,
            verification_methods_resolver=self.resolve_verification_methods,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pending = [task for task in self.agent.background_tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self.backend is not None:
            await self.backend.close()

    def resolve_public_key(self, did: str) -> bytes:
        if did != self.identity.legacy_did:
            raise KeyError(did)
        suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
        return suite.serialize_public_key(self.identity.legacy_keypair.public_key)

    def resolve_verification_methods(self, did: str) -> list:
        if did != self.identity.signing_did or not self.identity.is_hybrid:
            raise KeyError(did)
        return list(self.identity.new_verification_methods or [])

    def register_action(
        self,
        name: str,
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> None:
        self.registry.register(
            SourceRegistration(
                name=name,
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=handler,
                log_redaction=RedactionPolicy(summarize=lambda p: "<redacted>"),
            )
        )

    async def put_signed(self, spec):
        if self.store is None:
            raise RuntimeError("WorkflowHarness is not entered")
        signed = sign_workflow_spec(spec, self.identity)
        await self.store.put_definition(signed)
        return signed

    @property
    def parts(self):
        return SimpleNamespace(
            agent=self.agent,
            backend=self.backend,
            dispatcher=self.dispatcher,
            identity=self.identity,
            registry=self.registry,
            runner=self.runner,
            store=self.store,
        )


def _identity(did: str) -> AgentIdentity:
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    return AgentIdentity(
        legacy_did=did,
        legacy_keypair=suite.generate_keypair(),
        legacy_did_document={},
    )


__all__ = [
    "EncryptedWorkflowCassetteStore",
    "HarnessAgent",
    "WorkflowCassetteError",
    "WorkflowHarness",
    "assert_no_plaintext_workflow_cassettes",
]
