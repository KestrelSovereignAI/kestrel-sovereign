"""
Global resource registry for test cleanup.

This module provides crash-safe tracking of:
- RunPod GPU instances
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

    Ensures resources are cleaned up even on crashes via:
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
        """Handle signals by cleaning up then re-raising.

        Note: We use raise SystemExit instead of sys.exit() to allow pytest
        to handle the exit properly and complete its hooks.
        """
        print(f"\n[CLEANUP] Received signal {signum}, cleaning up resources...")
        self.cleanup_all()
        # Re-raise the original signal to let the default handler run
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _recover_from_crash(self):
        """Check for resources from a crashed previous run."""
        if not REGISTRY_FILE.exists():
            return

        try:
            data = json.loads(REGISTRY_FILE.read_text())
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

    def _cleanup_resource_by_type(self, resource_type: str, resource_id: str, metadata: Dict[str, Any]):
        """Clean up a resource based on its type."""
        if resource_type == "runpod":
            self._cleanup_runpod(resource_id)
        elif resource_type == "docker":
            self._cleanup_docker(resource_id)
        elif resource_type == "subprocess":
            self._cleanup_subprocess(resource_id, metadata)
        elif resource_type == "tempdir":
            self._cleanup_tempdir(resource_id)
        else:
            logger.warning(f"Unknown resource type: {resource_type}")

    def _cleanup_runpod(self, pod_id: str):
        """Terminate a RunPod instance."""
        try:
            import runpod
            api_key = os.environ.get("RUNPOD_API_KEY")
            if api_key:
                runpod.api_key = api_key
                try:
                    runpod.stop_pod(pod_id)
                except Exception:
                    pass
                runpod.terminate_pod(pod_id)
                print(f"[CLEANUP] Terminated RunPod {pod_id}")
            else:
                print(f"[CLEANUP] Cannot cleanup RunPod {pod_id}: No API key")
        except ImportError:
            print(f"[CLEANUP] Cannot cleanup RunPod {pod_id}: runpod module not installed")
        except Exception as e:
            print(f"[CLEANUP] Error terminating RunPod {pod_id}: {e}")

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
    """Track a RunPod instance for cleanup."""
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
