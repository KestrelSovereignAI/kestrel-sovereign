"""Isolated, content-free Kite HTTP evidence harness.

This module intentionally owns process lifecycle only for a Kite process it
started itself.  It never discovers or kills an arbitrary listener, never
binds anything except ``127.0.0.1``, and keeps the HTTP transcript in the
ephemeral evidence home.  Its public observations are aggregate counts and
digests only.

The harness is not registered as a catalog workload until an isolated live
run has produced reviewable evidence.  In particular, a response string is
not evidence of provenance, import quarantine, or a canonical migration
count.  Those requirements are read only from the fixed, content-free HTTP
diagnostics commands; otherwise the harness fails closed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .capabilities import SemanticRuntimeCapabilities, semantic_capabilities_from_config
from .registry import get_knowledge_registry
from .release_evidence_models import DrillBinding, ErasureStage, ReleaseEvidenceError, _canonical_json, _sha256
from .release_evidence_postgres import DisposablePostgresDatabase


class KiteEvidenceError(ReleaseEvidenceError):
    """The isolated Kite harness cannot safely make an evidence claim."""


class KiteEvidenceUnsupported(KiteEvidenceError):
    """A required live HTTP observation is not exposed by the product."""


class KiteGate(StrEnum):
    """The three catalog-bound live-agent gates owned by this harness."""

    STABLE_ONLY = "kite_http_stable_only_release_drill"
    EXPERIMENTAL_ENABLED = "kite_http_experimental_enabled_release_drill"
    PERSISTED_STABLE = "stable_persisted_data_no_canonical_migration_drill"


_GATE_PROFILE = {
    KiteGate.STABLE_ONLY: "stable_only",
    KiteGate.EXPERIMENTAL_ENABLED: "experimental_enabled",
    KiteGate.PERSISTED_STABLE: "stable_only",
}
_MARKER_PREFIX = "kite-evidence-"
_ISOLATION_MARKER = ".kite-evidence-isolation.json"
_ALLOWED_HTTP_HOST = "127.0.0.1"
_KITE_EVIDENCE_CONTRACT = "kite-http-evidence-v1"
_KITE_EXPERIMENTAL_CAPABILITIES: dict[str, object] = {
    "mode": "experimental",
    "rdf12": {
        "capability": "rdf-profile:rdf12-cr-20260407-experimental",
        "version": "0.1.0",
    },
    "sparql12": {
        "capability": "query-profile:sparql12-20260605-experimental",
        "version": "0.1.0",
    },
    "shacl12": {
        "capability": "validation-profile:shacl12-core-20260602-experimental",
        "version": "0.1.0",
    },
    "shape_set": {
        "identifier": "kestrel-assertion-shapes-shacl12-experimental",
        "version": "0.1.0",
    },
}
_CHILD_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
    }
)


def _kite_inference_config() -> dict[str, object]:
    """Build Kite's exact inference approval from locally verified registry pins.

    The isolated agent still carries a normal operator-style inference config,
    but the harness never copies a static ontology digest or compatibility
    contract into its generated TOML.  Agent startup parses and validates this
    mapping before the typed HTTP probe can use it.
    """
    registry = get_knowledge_registry()
    ontology = registry.select_capability("vocabulary:rdfs11").resource
    rules = registry.select_capability("reasoning-profile:rdfs-v1").resource
    return {
        "enabled": True,
        "rdfs_version": str(rules.version),
        "ontology": {
            "namespace": ontology.namespace,
            "version": str(ontology.version),
            "content_digest": ontology.sha256,
            "compatibility_profile": registry.contract_version,
        },
    }


@dataclass(frozen=True, slots=True)
class KiteStorageConfig:
    """The one storage backend an isolated Kite process may receive.

    PostgreSQL is accepted only as a live :class:`DisposablePostgresDatabase`
    created by the core release-evidence isolation authority.  A string DSN is
    intentionally not accepted here, preventing ambient or production database
    configuration from reaching the HTTP child process.
    """

    backend: str = "sqlite"
    disposable_postgres: DisposablePostgresDatabase | None = None

    def __post_init__(self) -> None:
        if self.backend == "sqlite" and self.disposable_postgres is None:
            return
        if (
            self.backend == "postgres"
            and isinstance(self.disposable_postgres, DisposablePostgresDatabase)
            and self.disposable_postgres.database_name.startswith("kestrel_semantic_release_")
        ):
            return
        raise KiteEvidenceError(
            "Kite storage must be sqlite or an operator-created disposable PostgreSQL database"
        )


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class KiteIsolationConfig:
    """Explicit local-only host identity for exactly one Kite evidence run."""

    worktree: Path
    home: Path
    port: int
    gate: KiteGate
    agent_name: str = "kite"
    host: str = _ALLOWED_HTTP_HOST
    storage: KiteStorageConfig = KiteStorageConfig()

    def __post_init__(self) -> None:
        worktree = self.worktree.expanduser().resolve()
        home = self.home.expanduser().resolve()
        if not worktree.is_dir() or not (worktree / ".git").exists() and not (worktree / ".git").is_file():
            raise KiteEvidenceError("Kite worktree must be an existing git worktree")
        if home == worktree or _inside(worktree, home) or _inside(home, worktree):
            raise KiteEvidenceError("Kite evidence home must be disjoint from the source worktree")
        if self.host != _ALLOWED_HTTP_HOST:
            raise KiteEvidenceError("Kite evidence may bind only to 127.0.0.1")
        if self.agent_name != "kite":
            raise KiteEvidenceError("Kite evidence endpoint is fixed to the isolated kite agent")
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise KiteEvidenceError("Kite evidence port must be an unprivileged TCP port")
        if not isinstance(self.storage, KiteStorageConfig):
            raise KiteEvidenceError("Kite evidence requires an explicit storage configuration")
        object.__setattr__(self, "worktree", worktree)
        object.__setattr__(self, "home", home)

    @property
    def config_path(self) -> Path:
        return self.home / "multi_agent.toml"

    @property
    def marker_path(self) -> Path:
        return self.home / _ISOLATION_MARKER

    @property
    def log_path(self) -> Path:
        return self.home / "kite-http.log"

    @property
    def python_path(self) -> Path:
        return self.worktree / ".venv" / "bin" / "python"

    @property
    def kestrel_path(self) -> Path:
        return self.worktree / ".venv" / "bin" / "kestrel"

    @property
    def profile(self) -> str:
        return _GATE_PROFILE[self.gate]

    def environment(self) -> dict[str, str]:
        # Deliberately do not inherit a live KESTREL_HOME, active venv,
        # multi-agent manifest, provider, or credential configuration.
        # A denylist will miss a new credential, trust anchor, provider pin,
        # or database selector.  Preserve only OS execution/locale settings;
        # every Kestrel-specific setting below is constructed for this run.
        env = {
            key: value
            for key, value in os.environ.items()
            if key in _CHILD_ENV_ALLOWLIST
        }
        isolated_db = self.home / "kite-evidence.sqlite3"
        env["KESTREL_HOME"] = str(self.home)
        # The signer treats this fresh, owner-only home as its explicit trust
        # anchor.  It may safely be located beneath a system temporary root.
        env["KESTREL_KITE_RELEASE_EVIDENCE_ROOT"] = str(self.home)
        env["KESTREL_MULTI_AGENT_CONFIG"] = str(self.config_path)
        env["KESTREL_DEMO_SERVER"] = "1"
        env["KESTREL_KITE_RELEASE_EVIDENCE"] = "1"
        # The isolated test identity must not borrow a production DID domain.
        env["KESTREL_DID_WEB_DOMAIN"] = "kite.invalid"
        # Explicit local-only persistence wins over every inherited backend
        # selection or DSN.  The secondary URL is intentionally a SQLite URL,
        # never a production service address, for legacy consumers that read it.
        env["KESTREL_DB_BACKEND"] = self.storage.backend
        if self.storage.backend == "sqlite":
            env["KESTREL_DB_PATH"] = str(isolated_db)
            env["KESTREL_DATABASE_URL"] = f"sqlite:///{isolated_db}"
            env["DATABASE_URL"] = f"sqlite:///{isolated_db}"
        else:
            assert self.storage.disposable_postgres is not None
            # This is a generated database from the core disposable authority,
            # never an inherited TEST_POSTGRES_URL/DATABASE_URL value.
            env["KESTREL_DATABASE_URL"] = self.storage.disposable_postgres.dsn
            env["DATABASE_URL"] = self.storage.disposable_postgres.dsn
        return env


@dataclass(frozen=True, slots=True)
class KiteAggregateObservation:
    """Content-free aggregate observation accepted by a Kite gate schema."""

    gate_id: str
    invoke_count: int
    scenario_count: int
    provenance_check_count: int = 0
    experimental_selection_count: int = 0
    persisted_assertion_count: int = 0
    canonical_migration_count: int = 0

    def to_mapping(self) -> dict[str, int]:
        if self.gate_id == KiteGate.PERSISTED_STABLE:
            return {
                "persisted_assertion_count": self.persisted_assertion_count,
                "canonical_migration_count": self.canonical_migration_count,
            }
        fields: dict[str, int] = {
            "invoke_count": self.invoke_count,
            "scenario_count": self.scenario_count,
        }
        if self.gate_id == KiteGate.STABLE_ONLY:
            fields["provenance_check_count"] = self.provenance_check_count
        elif self.gate_id == KiteGate.EXPERIMENTAL_ENABLED:
            fields["experimental_selection_count"] = self.experimental_selection_count
        return fields


@dataclass(frozen=True, slots=True)
class SurfaceErasureObservation:
    """One content-free erasure surface result from an independently owned probe."""

    stage: ErasureStage
    erased_count: int
    remaining_count: int
    drill: DrillBinding

    def __post_init__(self) -> None:
        if self.erased_count <= 0 or self.remaining_count != 0:
            raise KiteEvidenceError("erasure surface must report positive erasure and zero remaining")

    def to_mapping(self) -> dict[str, int]:
        return {"erased_count": self.erased_count, "remaining_count": self.remaining_count}


class ErasureSurfaceProbe(Protocol):
    """Content-free observer for core and external erasure surfaces.

    The probe is deliberately separate from the HTTP driver: core and optional
    external consumers own their own lifecycle proof, while this harness only
    enforces one shared drill binding and emits no tenant or assertion data.
    """

    def observe(self, drill: DrillBinding) -> tuple[SurfaceErasureObservation, ...]: ...


class KiteHttpHarness:
    """Prepare, launch, drive, restart, and stop one owned Kite process safely."""

    def __init__(
        self,
        config: KiteIsolationConfig,
        *,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        opener: Callable[..., Any] = urlopen,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._popen = popen
        self._run = run
        self._opener = opener
        self._clock = clock
        self._sleep = sleeper
        self._process: subprocess.Popen[bytes] | None = None
        self._log: Any | None = None
        self._api_key: str | None = None
        self._evidence_public_key: str | None = None
        self._invoke_count = 0
        self._scenario_count = 0

    def prepare(self) -> None:
        """Create a fresh test-only Kestrel home without starting a server."""
        if self.config.home.exists():
            raise KiteEvidenceError("Kite evidence home must be fresh; refusing to reuse any agent state")
        if not self.config.python_path.is_file() or not self.config.kestrel_path.is_file():
            raise KiteEvidenceError("Kite worktree must provide its own .venv python and kestrel executable")
        self._assert_port_unused()
        self.config.home.mkdir(mode=0o700, parents=True)
        self.config.marker_path.write_text(
            _canonical_json({
                "contract": "kite-http-release-evidence-v1",
                "gate": self.config.gate.value,
                "port": self.config.port,
                "profile": self.config.profile,
            }),
            encoding="utf-8",
        )
        try:
            self._run(
                [str(self.config.kestrel_path), "create", "kite", "--test", "--port", str(self.config.port)],
                cwd=self.config.worktree,
                env=self.config.environment(),
                check=True,
                text=True,
                capture_output=True,
            )
            self._write_profile_config()
        except BaseException:
            # A partial test home is intentionally retained for manual forensic
            # inspection, but can never be used by a subsequent run because
            # prepare() requires freshness.
            raise

    async def seed_disposable_postgres_test_identity(self) -> None:
        """Copy only the freshly incepted test identity into the empty PG run.

        ``kestrel create --test`` deliberately keeps its inception identity in
        the fresh local agent directory.  A disposable PostgreSQL runtime is
        a separate, initially empty storage plane, so it needs that one agent
        node before the server can prove that its typed evidence endpoint is
        serving a test instance.  This is not a semantic import: no facts,
        assertions, vectors, corpus artifacts, or caller data cross here.
        """
        storage = self.config.storage
        if storage.backend != "postgres" or storage.disposable_postgres is None:
            raise KiteEvidenceError("PostgreSQL identity seed requires a disposable authority")
        local_path = self.config.home / "agent_data" / self.config.agent_name / "kestrel_prime.db"
        if not local_path.is_file():
            raise KiteEvidenceError("Kite test inception did not create its local identity store")

        from kestrel_sovereign.storage.async_storage import AsyncStorage

        local_storage = AsyncStorage(db_path=str(local_path), backend="sqlite")
        postgres_storage: AsyncStorage | None = None
        try:
            await local_storage.initialize()
            nodes = await local_storage.get_nodes_by_type("agent")
            if len(nodes) != 1 or not bool(
                nodes[0].properties.get("is_test_instance", False)
            ):
                raise KiteEvidenceError("Kite PostgreSQL seed requires exactly one incepted test identity")
            identity = nodes[0]
            postgres_storage = AsyncStorage(
                backend="postgres",
                dsn=storage.disposable_postgres.dsn,
                agent_id=identity.node_id,
            )
            await postgres_storage.initialize()
            if await postgres_storage.get_node(identity.node_id) is not None:
                raise KiteEvidenceError("Kite disposable PostgreSQL identity was unexpectedly pre-populated")
            await postgres_storage.add_node(identity)
        finally:
            if postgres_storage is not None:
                await postgres_storage.close()
            await local_storage.close()

    def _write_profile_config(self) -> None:
        try:
            import toml
        except ImportError as exc:  # pragma: no cover - package dependency
            raise KiteEvidenceError("toml is required to write the isolated Kite config") from exc
        if not self.config.config_path.is_file():
            raise KiteEvidenceError("Kite inception did not produce an isolated multi_agent.toml")
        parsed = toml.load(self.config.config_path)
        agent = parsed.get("agents", {}).get("kite")
        if not isinstance(agent, dict):
            raise KiteEvidenceError("Kite inception did not register exactly the kite agent")
        agent["port"] = self.config.port
        agent["autostart"] = True
        # Multi-agent config resolves relative data paths from the launched
        # worktree, not from ``KESTREL_HOME``.  Pin the freshly created
        # evidence home explicitly so the server never falls back to a path
        # beneath the checkout (or an existing agent's state).
        agent["data_dir"] = str(self.config.home / "agent_data" / "kite")
        agent["semantic_inference"] = _kite_inference_config()
        if self.config.profile == "stable_only":
            agent.pop("semantic_capabilities", None)
        else:
            agent["semantic_capabilities"] = _KITE_EXPERIMENTAL_CAPABILITIES
        # Uvicorn owns the explicitly selected test port.  The hosted agent's
        # LocalAgentConfig port must remain distinct from HostConfig's port,
        # otherwise MultiAgentConfig correctly rejects it as a conflict.
        parsed.setdefault("host", {})["bind"] = _ALLOWED_HTTP_HOST
        with self.config.config_path.open("w", encoding="utf-8") as destination:
            toml.dump(parsed, destination)

    def start(self, *, ready_timeout_seconds: float = 60.0) -> None:
        """Start exactly this worktree's server and fetch the bootstrap key once."""
        if self._process is not None:
            raise KiteEvidenceError("Kite process is already owned and running")
        self._assert_owned_home()
        self._assert_port_unused()
        log = self.config.log_path.open("ab")
        try:
            process = self._popen(
                [
                    str(self.config.python_path), "-m", "uvicorn",
                    "kestrel_sovereign.server:app", "--host", _ALLOWED_HTTP_HOST,
                    "--port", str(self.config.port),
                ],
                cwd=self.config.worktree,
                env=self.config.environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log.close()
            raise
        self._process = process
        self._log = log
        try:
            self._api_key, self._evidence_public_key = self._wait_for_bootstrap_key(
                ready_timeout_seconds
            )
        except BaseException:
            self.stop()
            raise

    def stop(self, *, timeout_seconds: float = 15.0) -> None:
        """Stop only the exact ``Popen`` process this object launched."""
        process, self._process = self._process, None
        log, self._log = self._log, None
        self._api_key = None
        self._evidence_public_key = None
        if process is None or process.poll() is not None:
            if log is not None:
                log.close()
            return
        try:
            process.terminate()
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout_seconds)
        finally:
            if log is not None:
                log.close()

    def restart(self, *, ready_timeout_seconds: float = 60.0) -> None:
        """Full module reload through an owned PID stop/start cycle."""
        if self._process is None:
            raise KiteEvidenceError("cannot restart a Kite process this harness did not start")
        pinned_public_key = self._evidence_public_key
        self.stop()
        self.start(ready_timeout_seconds=ready_timeout_seconds)
        if (
            pinned_public_key is not None
            and self._evidence_public_key != pinned_public_key
        ):
            self.stop()
            raise KiteEvidenceError("Kite restart changed its pinned evidence public key")

    def invoke(
        self,
        prompt: str,
        *,
        kite_evidence: Mapping[str, object] | None = None,
        timeout_seconds: float = 120.0,
    ) -> Mapping[str, Any]:
        """Use only the documented multi-agent HTTP invoke route."""
        if self._process is None or self._process.poll() is not None or not self._api_key:
            raise KiteEvidenceError("Kite HTTP invoke requires a live owned process and one bootstrap key")
        payload: dict[str, object] = {"input": prompt}
        if kite_evidence is not None:
            payload["kite_evidence"] = dict(kite_evidence)
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"http://{_ALLOWED_HTTP_HOST}:{self.config.port}/api/agents/kite/api/agent/invoke",
            data=body,
            headers={"X-API-Key": self._api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, ValueError) as exc:
            raise KiteEvidenceError("isolated Kite HTTP invoke failed") from exc
        if not isinstance(decoded, Mapping):
            raise KiteEvidenceError("isolated Kite HTTP invoke returned a non-object response")
        self._invoke_count += 1
        return decoded

    def run_release_gate(self) -> KiteAggregateObservation:
        """Drive the non-destructive/live scenarios required for one gate.

        The raw values are fresh in-memory markers and are never written to an
        artifact.  Any unavailable semantic observability contract blocks the
        gate rather than yielding a synthetic aggregate pass.
        """
        if self._process is None:
            raise KiteEvidenceError("start the isolated Kite process before driving a release gate")
        initial = self._marker()
        replacement = self._marker()
        self._evidence("save", value=initial)
        baseline = self._evidence("diagnostics")
        self._assert_profile(baseline)
        if baseline["explicit_fact_provenance_present_count"] < 1:
            raise KiteEvidenceError("Kite diagnostics did not observe explicit-fact provenance")
        self._evidence("save", value=replacement)
        recall = self._evidence("paraphrase_recall")
        if (
            type(recall.get("retrieval_count")) is not int
            or recall["retrieval_count"] < 1
            or type(recall.get("provenance_check_count")) is not int
            or recall["provenance_check_count"] < 1
        ):
            raise KiteEvidenceError("Kite paraphrase recall did not return provenance-bound candidates")
        quarantine = self._evidence("quarantine")
        if quarantine != {"invalid_import_quarantine_count": 1}:
            raise KiteEvidenceError("Kite invalid-import probe did not report one quarantine")
        sleep = self._evidence("sleep")
        if sleep != {"sleep_success_count": 1}:
            raise KiteEvidenceError("Kite sleep operation did not complete")

        if self.config.gate is KiteGate.PERSISTED_STABLE:
            previous_pid = self._owned_pid()
            self.restart()
            if self._owned_pid() == previous_pid:
                raise KiteEvidenceError("Kite restart did not replace the owned server PID")
            restarted = self._evidence("diagnostics")
            self._assert_profile(restarted)
            migration_count = (
                restarted["canonical_migration_count"]
                - baseline["canonical_migration_count"]
            )
            if migration_count != 0:
                raise KiteEvidenceError("stable Kite restart ran a canonical migration")
            self._evidence("delete")
            deleted = self._evidence("diagnostics")
            if deleted["explicit_fact_provenance_present_count"] != 0:
                raise KiteEvidenceError("Kite diagnostics retained deleted fact provenance")
            return KiteAggregateObservation(
                gate_id=self.config.gate.value,
                invoke_count=self._invoke_count,
                scenario_count=self._scenario_count,
                persisted_assertion_count=1,
                canonical_migration_count=migration_count,
            )
        if self.config.gate is KiteGate.EXPERIMENTAL_ENABLED:
            return KiteAggregateObservation(
                gate_id=self.config.gate.value,
                invoke_count=self._invoke_count,
                scenario_count=self._scenario_count,
                experimental_selection_count=1,
            )
        return KiteAggregateObservation(
            gate_id=self.config.gate.value,
            invoke_count=self._invoke_count,
            scenario_count=self._scenario_count,
            provenance_check_count=1,
        )

    def measure_sleep(self, *, changed: bool, iterations: int = 3) -> tuple[float, ...]:
        """Measure exactly the typed HTTP sleep operation in milliseconds.

        Changed samples save a fresh governed fact before the timer; unchanged
        samples perform no mutation.  Setup and response validation are not
        relabeled as an in-process maintenance metric: the timed call is the
        live loopback invoke route that reaches ``agent.sleep``.
        """
        if type(iterations) is not int or iterations < 3:
            raise KiteEvidenceError("Kite sleep benchmark requires at least three samples")
        operation = "sleep_changed" if changed else "sleep_unchanged"
        samples: list[float] = []
        for _ in range(iterations):
            if changed:
                self._evidence("save", value=self._marker())
            started = time.perf_counter()
            observation = self._evidence(operation)
            elapsed = (time.perf_counter() - started) * 1_000
            if observation != {"sleep_success_count": 1}:
                raise KiteEvidenceError("Kite sleep operation did not return its typed success observation")
            samples.append(max(elapsed, float.fromhex("0x1.0p-52")))
        return tuple(samples)

    def correlated_erasure(self, probe: ErasureSurfaceProbe, drill: DrillBinding) -> tuple[SurfaceErasureObservation, ...]:
        """Validate the shared release drill across every observed surface."""
        self._evidence("delete")
        observed = probe.observe(drill)
        expected = set(ErasureStage)
        actual = {item.stage for item in observed}
        if actual != expected:
            raise KiteEvidenceError("correlated erasure probe must observe every catalog erasure stage")
        if any(item.drill != drill for item in observed):
            raise KiteEvidenceError("correlated erasure probe returned a mismatched drill binding")
        return tuple(sorted(observed, key=lambda item: item.stage.value))

    def core_erasure_stage(self, stage: ErasureStage) -> SurfaceErasureObservation:
        """Read one server-owned, correlated core erasure drill result.

        The HTTP client cannot name an assertion, revision, tenant, artifact,
        vector profile, policy, or consumer.  Those bindings are created and
        checked by the production storage owner before it emits this aggregate.
        The optional serving-adapter stage is deliberately absent: it belongs
        to the separately installed parametric-self feature.
        """
        if stage is ErasureStage.SERVED_ADAPTER_ELIGIBILITY:
            raise KiteEvidenceError("served-adapter eligibility is external evidence")
        snapshot = self._evidence("erasure_core_snapshot")
        expected = {item.value for item in ErasureStage if item is not ErasureStage.SERVED_ADAPTER_ELIGIBILITY}
        if set(snapshot) != expected:
            raise KiteEvidenceError("core erasure drill has an unexpected stage set")
        result = snapshot.get(stage.value)
        if not isinstance(result, dict) or set(result) != {"erased_count", "remaining_count"}:
            raise KiteEvidenceError("core erasure drill stage is malformed")
        erased_count, remaining_count = result["erased_count"], result["remaining_count"]
        if type(erased_count) is not int or type(remaining_count) is not int:
            raise KiteEvidenceError("core erasure drill must return integer aggregates")
        from .release_evidence import erasure_drill_binding

        return SurfaceErasureObservation(stage, erased_count, remaining_count, erasure_drill_binding())

    def write_content_free_artifact(self, observation: KiteAggregateObservation, destination: Path) -> str:
        """Write only aggregate counts and an integrity digest, never transcript data."""
        if observation.gate_id != self.config.gate.value:
            raise KiteEvidenceError("Kite aggregate observation is bound to another gate")
        destination = destination.expanduser().resolve()
        if not _inside(destination, self.config.home):
            raise KiteEvidenceError("Kite evidence artifacts must remain inside the isolated evidence home")
        if destination.exists():
            raise KiteEvidenceError("refusing to overwrite an existing Kite evidence artifact")
        payload = {
            "contract": "kite-http-release-evidence-v1",
            "gate_id": observation.gate_id,
            "observation": observation.to_mapping(),
        }
        payload["artifact_digest"] = _sha256(_canonical_json(payload))
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_text(_canonical_json(payload), encoding="utf-8")
        return payload["artifact_digest"]

    def _evidence(self, operation: str, *, value: str | None = None) -> dict[str, Any]:
        """Execute a test-only typed runtime operation, never assistant prose."""
        nonce = secrets.token_hex(32)
        request: dict[str, object] = {"operation": operation, "nonce": nonce}
        if value is not None:
            request["value"] = value
        response = self.invoke("", kite_evidence=request)
        envelope = response.get("kite_evidence")
        if not isinstance(envelope, Mapping):
            raise KiteEvidenceError("Kite invoke did not return typed runtime evidence")
        expected_fields = {"contract", "nonce", "operation", "observation", "signature"}
        if set(envelope) != expected_fields:
            raise KiteEvidenceError("Kite runtime evidence had an unexpected envelope")
        observation = envelope["observation"]
        signature = envelope["signature"]
        if (
            envelope["contract"] != _KITE_EVIDENCE_CONTRACT
            or envelope["nonce"] != nonce
            or envelope["operation"] != operation
            or not isinstance(observation, dict)
            or not isinstance(signature, str)
        ):
            raise KiteEvidenceError("Kite runtime evidence was malformed or not request-bound")
        if not self._evidence_public_key:
            raise KiteEvidenceError("Kite runtime evidence requires a pinned server public key")
        signed = {
            "contract": envelope["contract"],
            "nonce": envelope["nonce"],
            "observation": observation,
            "operation": envelope["operation"],
        }
        try:
            Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(self._evidence_public_key)
            ).verify(
                bytes.fromhex(signature),
                _canonical_json(signed).encode("utf-8"),
            )
        except (InvalidSignature, ValueError) as error:
            raise KiteEvidenceError("Kite runtime evidence signature did not verify") from error
        self._scenario_count += 1
        return dict(observation)

    def _assert_profile(self, diagnostics: Mapping[str, Any]) -> None:
        allowed = {
            "capability_mode",
            "active_capability_pins",
            "canonical_migration_count",
            "explicit_fact_provenance_present_count",
        }
        if set(diagnostics) != allowed:
            raise KiteEvidenceError("Kite semantic evidence diagnostics had an unexpected field")
        mode = diagnostics["capability_mode"]
        pins = diagnostics["active_capability_pins"]
        migrations = diagnostics["canonical_migration_count"]
        provenance = diagnostics["explicit_fact_provenance_present_count"]
        if (
            mode not in {"stable", "experimental"}
            or not isinstance(pins, list)
            or not all(isinstance(pin, str) for pin in pins)
            or type(migrations) is not int
            or migrations < 0
            or type(provenance) is not int
            or provenance < 0
        ):
            raise KiteEvidenceError("Kite semantic evidence diagnostics had invalid values")
        expected = "experimental" if self.config.gate is KiteGate.EXPERIMENTAL_ENABLED else "stable"
        if mode != expected:
            raise KiteEvidenceError("Kite semantic capability mode differs from its evidence gate")
        expected_capabilities = (
            semantic_capabilities_from_config(_KITE_EXPERIMENTAL_CAPABILITIES)
            if expected == "experimental"
            else SemanticRuntimeCapabilities.stable()
        )
        expected_pins = [
            f"{key}={value}"
            for key, value in sorted(
                expected_capabilities.capability_versions().items()
            )
        ]
        if pins != expected_pins:
            raise KiteEvidenceError("Kite diagnostics do not match exact registry capability pins")

    def _owned_pid(self) -> int:
        if self._process is None or self._process.poll() is not None:
            raise KiteEvidenceError("Kite owned process is not live")
        return self._process.pid

    def _wait_for_bootstrap_key(self, timeout_seconds: float) -> tuple[str, str]:
        deadline = self._clock() + timeout_seconds
        endpoint = f"http://{_ALLOWED_HTTP_HOST}:{self.config.port}/api/auth/key"
        while self._clock() < deadline:
            if self._process is None or self._process.poll() is not None:
                raise KiteEvidenceError("Kite process exited before becoming ready")
            try:
                with self._opener(endpoint, timeout=2) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                key = payload.get("key") if isinstance(payload, Mapping) else None
                public_key = (
                    payload.get("kite_evidence_public_key")
                    if isinstance(payload, Mapping)
                    else None
                )
                if (
                    isinstance(key, str)
                    and key
                    and isinstance(public_key, str)
                    and len(public_key) == 64
                ):
                    try:
                        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key))
                    except ValueError:
                        self._sleep(0.2)
                        continue
                    return key, public_key
            except (OSError, URLError, ValueError):
                self._sleep(0.2)
        raise KiteEvidenceError("isolated Kite did not become ready before timeout")

    def _assert_owned_home(self) -> None:
        if not self.config.marker_path.is_file() or not self.config.config_path.is_file():
            raise KiteEvidenceError("Kite home was not prepared by this evidence harness")

    def _assert_port_unused(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex((_ALLOWED_HTTP_HOST, self.config.port)) == 0:
                raise KiteEvidenceError("refusing to use an occupied Kite evidence port")

    @staticmethod
    def _marker() -> str:
        # Marker data is intentionally process-local and excluded from every
        # public artifact; it only lets the HTTP response prove continuity.
        return _MARKER_PREFIX + secrets.token_urlsafe(18)
