"""
Global resource registry for test cleanup.

Historical Runpod IDs are retained for canonical original-lease reconciliation;
they never authorize automatic cloud cleanup. Other resource types retain the
existing best-effort legacy behavior, not a verified physical cleanup guarantee:
- Docker containers
- Server subprocesses
- Temporary files/directories

Resources are cleaned up via multiple mechanisms:
1. Normal pytest teardown
2. atexit handlers
3. Signal handlers (SIGINT, SIGTERM)
4. pytest_sessionfinish hook
"""

import atexit
import signal
import os
import sys
import subprocess
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import logging

from kestrel_sovereign.legacy_runpod import LegacyRunpodPathError

logger = logging.getLogger(__name__)

# Registry file for crash recovery
REGISTRY_FILE = Path("/tmp/kestrel_test_resources.json")


@dataclass
class TrackedResource:
    """A resource that needs cleanup."""
    resource_type: str  # "runpod", "docker", "subprocess", "tempdir"
    resource_id: str    # pod_id, container_id, pid, path
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleanup_fn: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackedResource':
        """Deserialize from JSON storage."""
        return cls(
            resource_type=data['resource_type'],
            resource_id=data['resource_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {})
        )


class ResourceRegistry:
    """
    Singleton registry for tracked resources.

    Records legacy test resources and installs best-effort cleanup hooks:
    - atexit handler (normal exit)
    - signal handlers (Ctrl+C, kill)
    - Persistent file for crash recovery
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _initialize(self):
        """Initialize the registry (called once)."""
        if self._initialized:
            return

        self._resources: Dict[str, TrackedResource] = {}
        self._cleanup_in_progress = False
        self._handlers_installed = False
        self._setup_handlers()
        self._recover_from_crash()
        self._initialized = True

    def _setup_handlers(self):
        """Set up cleanup handlers for various exit scenarios."""
        if self._handlers_installed:
            return

        atexit.register(self.cleanup_all)

        # Only install signal handlers if we're in the main thread
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            # Signal handlers can only be set in main thread
            pass

        self._handlers_installed = True

    def _signal_handler(self, signum, frame):
        """Attempt cleanup, then restore and re-send the original signal.

        An unresolved Runpod record or failed diagnostic must not consume
        shutdown or imply successful cleanup.
        """
        print(f"\n[CLEANUP] Received signal {signum}, cleaning up resources...")
        try:
            self.cleanup_all()
        except LegacyRunpodPathError as error:
            logger.error("Unresolved cloud resource during signal cleanup: %s", error)
        finally:
            # A refusal/logging failure must never convert SIGINT/SIGTERM into
            # an ordinary caught error. Original unresolved records remain.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _recover_from_crash(self):
        """Check for resources from a crashed previous run."""
        if not REGISTRY_FILE.exists():
            return

        data = json.loads(REGISTRY_FILE.read_text())
        if not isinstance(data, dict):
            raise ValueError("Resource registry is invalid; original records retained")
        # Historical IDs are reconciliation evidence, not provider authority.
        # Refuse BEFORE the old generic recovery handler can erase this file
        # or perform any unrelated cleanup under an unverified registry.
        for resource_id, record in data.items():
            if not isinstance(record, dict):
                raise ValueError("Resource registry is invalid; original records retained")
            self._cleanup_resource_by_type(
                record.get("resource_type"), resource_id,
                record.get("metadata", {}), validate_only=True,
            )
            if record.get("resource_type") == "runpod":
                self._cleanup_runpod(resource_id)

        try:
            if data:
                print(f"\n[RECOVERY] Found {len(data)} orphaned resources from previous crash")
                for resource_id, resource_data in data.items():
                    print(f"  - Cleaning {resource_data['resource_type']}:{resource_id}")
                    self._cleanup_resource_by_type(
                        resource_data['resource_type'],
                        resource_id,
                        resource_data.get('metadata', {})
                    )
            REGISTRY_FILE.unlink()
        except Exception as e:
            print(f"[RECOVERY] Error recovering resources: {e}")
            # Remove corrupt file
            try:
                REGISTRY_FILE.unlink()
            except Exception:
                pass

    def register(self, resource: TrackedResource) -> str:
        """
        Register a resource for tracking.

        Returns:
            str: The resource key (resource_type:resource_id)
        """
        self._initialize()
        key = f"{resource.resource_type}:{resource.resource_id}"
        self._resources[key] = resource
        self._persist_registry()
        logger.debug(f"Registered resource: {key}")
        return key

    def unregister(self, key: str):
        """Unregister a resource after successful cleanup."""
        self._initialize()
        if key in self._resources:
            del self._resources[key]
            self._persist_registry()
            logger.debug(f"Unregistered resource: {key}")

    def get_resource(self, key: str) -> Optional[TrackedResource]:
        """Get a tracked resource by key."""
        self._initialize()
        return self._resources.get(key)

    def list_resources(self, resource_type: Optional[str] = None) -> Dict[str, TrackedResource]:
        """List all tracked resources, optionally filtered by type."""
        self._initialize()
        if resource_type is None:
            return dict(self._resources)
        return {k: v for k, v in self._resources.items() if v.resource_type == resource_type}

    def _persist_registry(self):
        """Write registry to disk for crash recovery."""
        data = {}
        for key, resource in self._resources.items():
            data[resource.resource_id] = resource.to_dict()

        try:
            REGISTRY_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to persist resource registry: {e}")

    def cleanup_all(self):
        """Clean up all tracked resources."""
        if not hasattr(self, '_initialized') or not self._initialized:
            return

        if self._cleanup_in_progress:
            return
        # No provider operation, memory deletion or file unlink is permitted
        # for legacy cloud records without the original canonical lease.
        for resource in self._resources.values():
            self._cleanup_resource_by_type(
                resource.resource_type, resource.resource_id,
                resource.metadata, validate_only=True,
            )
            if resource.resource_type == "runpod":
                self._cleanup_runpod(resource.resource_id)
        self._cleanup_in_progress = True

        if self._resources:
            print(f"\n[CLEANUP] Cleaning up {len(self._resources)} tracked resources...")

        for key, resource in list(self._resources.items()):
            try:
                if resource.cleanup_fn:
                    resource.cleanup_fn()
                else:
                    self._cleanup_resource_by_type(
                        resource.resource_type,
                        resource.resource_id,
                        resource.metadata
                    )
                del self._resources[key]
                print(f"[CLEANUP] ✓ Cleaned {key}")
            except Exception as e:
                print(f"[CLEANUP] ✗ Error cleaning {key}: {e}")

        # Remove registry file
        try:
            if REGISTRY_FILE.exists():
                REGISTRY_FILE.unlink()
        except Exception:
            pass

        self._cleanup_in_progress = False

    def _cleanup_resource_by_type(
        self, resource_type: object, resource_id: str, metadata: Dict[str, Any],
        *, validate_only: bool = False,
    ):
        """Validate against the same finite dispatch before any cleanup."""
        cleaners = {
            "runpod": self._cleanup_runpod,
            "docker": self._cleanup_docker,
            "subprocess": self._cleanup_subprocess,
            "tempdir": self._cleanup_tempdir,
        }
        if not isinstance(resource_type, str) or resource_type not in cleaners:
            raise ValueError("Resource registry type is invalid; original records retained")
        if validate_only:
            return
        cleaner = cleaners[resource_type]
        if resource_type == "subprocess":
            cleaner(resource_id, metadata)
        else:
            cleaner(resource_id)

    def _cleanup_runpod(self, pod_id: str):
        """Retain an unresolved historical ID; never infer cleanup authority."""
        raise LegacyRunpodPathError("test resource cleanup", resource_id=pod_id)

    def _cleanup_docker(self, container_id: str):
        """Stop and remove a Docker container."""
        try:
            # Stop with timeout
            subprocess.run(
                ["docker", "stop", "-t", "5", container_id],
                capture_output=True, timeout=15
            )
            # Force remove
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True, timeout=10
            )
            print(f"[CLEANUP] Removed Docker container {container_id}")
        except subprocess.TimeoutExpired:
            # Force kill
            subprocess.run(
                ["docker", "kill", container_id],
                capture_output=True
            )
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True
            )
        except Exception as e:
            print(f"[CLEANUP] Error removing Docker container {container_id}: {e}")

    def _cleanup_subprocess(self, pid_str: str, metadata: Dict[str, Any]):
        """Kill a subprocess and its children."""
        try:
            pid = int(pid_str)
            pgid = metadata.get('pgid', pid)

            # Try graceful termination first
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return  # Already dead
            except PermissionError:
                # Try killing just the process
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    return

            # Wait briefly then force kill
            time.sleep(1)

            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            print(f"[CLEANUP] Killed process group {pgid}")
        except Exception as e:
            print(f"[CLEANUP] Error killing subprocess {pid_str}: {e}")

    def _cleanup_tempdir(self, path: str):
        """Remove a temporary directory."""
        import shutil
        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                print(f"[CLEANUP] Removed temp directory {path}")
        except Exception as e:
            print(f"[CLEANUP] Error removing temp directory {path}: {e}")


# Global singleton instance
registry = ResourceRegistry()


# Convenience functions for common resource types
def track_runpod(pod_id: str, name: str = "", cost_per_hr: float = 0.0) -> str:
    """Retain a historical ID for reconciliation, never cleanup authority."""
    resource = TrackedResource(
        resource_type="runpod",
        resource_id=pod_id,
        created_at=datetime.now(timezone.utc),
        metadata={'name': name, 'cost_per_hr': cost_per_hr}
    )
    return registry.register(resource)


def track_docker(container_id: str, image: str = "") -> str:
    """Track a Docker container for cleanup."""
    resource = TrackedResource(
        resource_type="docker",
        resource_id=container_id,
        created_at=datetime.now(timezone.utc),
        metadata={'image': image}
    )
    return registry.register(resource)


def track_subprocess(process: subprocess.Popen, name: str = "") -> str:
    """Track a subprocess for cleanup."""
    try:
        pgid = os.getpgid(process.pid)
    except (ProcessLookupError, OSError):
        pgid = process.pid

    resource = TrackedResource(
        resource_type="subprocess",
        resource_id=str(process.pid),
        created_at=datetime.now(timezone.utc),
        metadata={'pgid': pgid, 'name': name}
    )
    return registry.register(resource)


def track_tempdir(path: str) -> str:
    """Track a temporary directory for cleanup."""
    resource = TrackedResource(
        resource_type="tempdir",
        resource_id=path,
        created_at=datetime.now(timezone.utc),
        metadata={}
    )
    return registry.register(resource)


def untrack(key: str):
    """Untrack a resource (call after successful cleanup)."""
    registry.unregister(key)
