"""Real subprocess refusal controls: no cloud fixture or successful mock."""

from __future__ import annotations

import ast
import errno
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY_PATHS = (
    "scripts/poll_runpod_start.py",
    "tests/infrastructure/check_pods.py",
    "tests/infrastructure/manual_debug_pod.py",
    "tests/infrastructure/stop_all_pods.py",
    "tests/infrastructure/test_runpod_gpu.py",
    "tests/infrastructure/test_basic_container.py",
    "tests/infrastructure/test_simple_pod.py",
    "tests/infrastructure/test_ssh_pod.py",
    "tests/infrastructure/test_gpu_basic.py",
)
AUDIT_RUNNER = """
import runpy, sys
def audit(event, arguments):
    if event == "import" and arguments[0].split(".")[0] in {"runpod", "dotenv"}:
        raise AssertionError("retired path imported provider/credential code")
    if event == "open" and isinstance(arguments[0], (str, bytes)) and str(arguments[0]).endswith(".env"):
        raise AssertionError("retired path opened dotenv")
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("retired path attempted network access")
sys.addaudithook(audit)
runpy.run_path(sys.argv[1], run_name="__main__")
"""
REGISTRY_RUNNER = """
import atexit, hashlib, importlib.util, json, pathlib, sys
from datetime import datetime, timezone
repo, registry_path, operation = map(str, sys.argv[1:4])
sys.path.insert(0, repo)
def audit(event, arguments):
    if event == "import" and arguments[0].split(".")[0] in {"runpod", "dotenv"}:
        raise AssertionError("legacy registry imported provider/credential code")
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("legacy registry attempted network access")
sys.addaudithook(audit)
spec = importlib.util.spec_from_file_location("isolated_legacy_registry", pathlib.Path(repo) / "tests/shared/resource_registry.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.REGISTRY_FILE = pathlib.Path(registry_path)
record = module.TrackedResource("runpod", "synthetic-original-id", datetime.now(timezone.utc), {"unresolved": True})
registry = module.registry
if operation == "corrupt_recover":
    fixture = json.loads(sys.argv[4])
    corrupt = record.to_dict()
    if fixture["missing"]:
        del corrupt["resource_type"]
    else:
        corrupt["resource_type"] = fixture["resource_type"]
    unrelated = module.TrackedResource("tempdir", fixture["unrelated_path"], datetime.now(timezone.utc))
    module.REGISTRY_FILE.write_text(json.dumps({unrelated.resource_id: unrelated.to_dict(), record.resource_id: corrupt}, indent=2))
elif operation == "all":
    registry.register(record)
else:
    module.REGISTRY_FILE.write_text(json.dumps({record.resource_id: record.to_dict()}, indent=2))
before = module.REGISTRY_FILE.read_bytes()
error = None
try:
    if operation in {"recover", "corrupt_recover"}:
        registry._recover_from_crash()
    elif operation == "all":
        registry.cleanup_all()
    else:
        registry._cleanup_runpod(record.resource_id)
except (module.LegacyRunpodPathError, ValueError) as failure:
    error = str(failure)
finally:
    atexit.unregister(registry.cleanup_all)
after = module.REGISTRY_FILE.read_bytes() if module.REGISTRY_FILE.exists() else None
print(json.dumps({"error": error, "same_original_bytes": before == after,
    "original_file_retained": after is not None,
    "memory_record_retained": "runpod:" + record.resource_id in getattr(registry, "_resources", {}),
    "cleanup_in_progress": getattr(registry, "_cleanup_in_progress", False)}))
"""


SIGNAL_PROBE = """
import atexit, errno, hashlib, importlib.util, json, logging, os, pathlib, signal, sys
from datetime import datetime, timezone
repo, path, number, log_failure = sys.argv[1:]
sys.path.insert(0, repo)
def audit(event, arguments):
    if event == "import" and arguments[0].split(".")[0] in {"runpod", "dotenv"}:
        raise AssertionError("signal cleanup imported provider/credentials")
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("signal cleanup attempted network")
sys.addaudithook(audit)
spec = importlib.util.spec_from_file_location("owned_signal_registry", pathlib.Path(repo) / "tests/shared/resource_registry.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.REGISTRY_FILE = pathlib.Path(path)
if log_failure == "fail-log":
    class FailedOwnedLogSink(logging.Handler):
        def emit(self, record):
            os.write(-1, b"actual failed signal cleanup log write")
    module.logger.addHandler(FailedOwnedLogSink())
registry = module.registry
record = module.TrackedResource("runpod", "synthetic-original-signal-id", datetime.now(timezone.utc))
registry.register(record)
print(json.dumps({"before_sha256": hashlib.sha256(module.REGISTRY_FILE.read_bytes()).hexdigest()}), flush=True)
if log_failure in {"closed-stdout", "broken-stdout"}:
    descriptor = sys.stdout.fileno()
    if log_failure == "closed-stdout":
        os.close(descriptor)
        expected_errno = errno.EBADF
    else:
        read_descriptor, write_descriptor = os.pipe()
        os.close(read_descriptor)
        try:
            os.dup2(write_descriptor, descriptor)
        finally:
            os.close(write_descriptor)
        expected_errno = errno.EPIPE
    try:
        os.write(descriptor, b"actual signal diagnostic failure control")
    except OSError as error:
        if error.errno != expected_errno:
            raise
        print(json.dumps({"failed_stdout_errno": error.errno}), file=sys.stderr, flush=True)
    else:
        raise AssertionError("original stdout failure was not established")
try:
    try:
        os.kill(os.getpid(), int(number))
    except module.LegacyRunpodPathError as error:
        print(json.dumps({"ordinary_cleanup_refusal_caught": str(error)}), flush=True)
finally:
    atexit.unregister(registry.cleanup_all)
"""


class LegacyRunpodPathsTests(unittest.TestCase):
    def test_corrupt_recovery_retains_ledger_before_any_unrelated_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="owned-corrupt-registry-", dir=REPO.parent) as directory:
            cases = (
                ("missing", None), ("null", None), ("empty", ""),
                ("number", 7), ("boolean", False), ("array", []),
                ("object", {}), ("unknown", "unrecognized-resource-kind"),
                ("known-local-control", "tempdir"),
            )
            for index, (label, value) in enumerate(cases):
                with self.subTest(corrupt_type=label):
                    case = Path(directory, str(index))
                    case.mkdir()
                    unrelated = case / "original-unrelated-tempdir"
                    unrelated.mkdir()
                    sentinel = unrelated / "original.txt"
                    sentinel.write_text("original unrelated test evidence")
                    fixture = {"missing": label == "missing", "resource_type": value, "unrelated_path": str(unrelated)}
                    child = subprocess.run(
                        [sys.executable, "-I", "-S", "-B", "-c", REGISTRY_RUNNER,
                         str(REPO), str(case / "original-ledger.json"), "corrupt_recover", json.dumps(fixture)],
                        cwd=case, env={}, stdin=subprocess.DEVNULL,
                        capture_output=True, text=True, timeout=10, check=False,
                    )
                    self.assertEqual(child.returncode, 0, (child.stdout, child.stderr))
                    observed = [json.loads(line) for line in child.stdout.splitlines() if line.startswith("{")]
                    self.assertEqual(len(observed), 1, observed)
                    if label == "known-local-control":
                        self.assertIsNone(observed[0]["error"], observed)
                        self.assertFalse(observed[0]["original_file_retained"], observed)
                        self.assertFalse(sentinel.exists(), "known local dispatch did not run")
                        continue
                    self.assertTrue(observed[0]["error"], observed)
                    self.assertTrue(observed[0]["same_original_bytes"], observed)
                    self.assertTrue(observed[0]["original_file_retained"], observed)
                    self.assertTrue(sentinel.exists(), "malformed ledger reached unrelated cleanup")
                    self.assertEqual(sentinel.read_text(), "original unrelated test evidence")

    @unittest.skipUnless(os.name == "posix", "native POSIX signal exit observation")
    def test_refusal_never_swallows_real_sigint_or_sigterm(self):
        with tempfile.TemporaryDirectory(prefix="owned-registry-signal-", dir=REPO.parent) as directory:
            for number, log_failure in (
                (signal.SIGINT, "normal"), (signal.SIGTERM, "normal"),
                (signal.SIGINT, "fail-log"), (signal.SIGTERM, "fail-log"),
                (signal.SIGINT, "closed-stdout"), (signal.SIGTERM, "closed-stdout"),
                (signal.SIGINT, "broken-stdout"), (signal.SIGTERM, "broken-stdout"),
            ):
                with self.subTest(signal=number.name, log_failure=log_failure):
                    path = Path(directory, number.name + log_failure + ".json")
                    child = subprocess.run(
                        [sys.executable, "-I", "-S", "-B", "-u", "-c", SIGNAL_PROBE, str(REPO), str(path), str(int(number)), log_failure],
                        cwd=directory, env={}, stdin=subprocess.DEVNULL,
                        capture_output=True, text=True, timeout=10, check=False,
                    )
                    observed = [json.loads(line) for line in child.stdout.splitlines() if line.startswith("{")]
                    self.assertEqual(len(observed), 1, observed)
                    self.assertTrue(path.exists())
                    if log_failure == "normal":
                        self.assertIn("remains unresolved", child.stderr)
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), observed[0]["before_sha256"])
                    self.assertIn("synthetic-original-signal-id", json.loads(path.read_text()))
                    if log_failure in {"closed-stdout", "broken-stdout"}:
                        failures = [json.loads(line) for line in child.stderr.splitlines() if line.startswith("{")]
                        self.assertEqual(failures, [{"failed_stdout_errno": errno.EBADF if log_failure == "closed-stdout" else errno.EPIPE}])
                    self.assertEqual(child.returncode, -int(number), (child.stdout, child.stderr))

    def assert_refused(self, result: subprocess.CompletedProcess[str], path: str) -> None:
        self.assertEqual(result.returncode, 1, (path, result.stdout, result.stderr))
        self.assertIn("is retired; no provider operation was performed", result.stderr)
        self.assertIn("kestrel-cloud-runpod", result.stderr)
        self.assertIn("/issues/3828", result.stderr)
        self.assertNotIn("AssertionError", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_all_direct_entrypoints_refuse_even_with_cloud_opt_in(self) -> None:
        with tempfile.TemporaryDirectory(prefix="retired-runpod-cli-") as directory:
            for path in LEGACY_PATHS:
                with self.subTest(path=path):
                    result = subprocess.run(
                        [sys.executable, "-I", "-S", "-B", str(REPO / path)],
                        cwd=directory, env={"RUN_CLOUD_TESTS": "1"},
                        stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        timeout=10, check=False,
                    )
                    self.assert_refused(result, path)

    def test_actual_audited_entrypoints_do_not_import_credentials_or_connect(self) -> None:
        with tempfile.TemporaryDirectory(prefix="retired-runpod-audit-") as directory:
            Path(directory, ".env").write_text("RUNPOD_API_KEY=NOT-A-CREDENTIAL\n")
            for path in LEGACY_PATHS:
                with self.subTest(path=path):
                    result = subprocess.run(
                        [sys.executable, "-I", "-S", "-B", "-c", AUDIT_RUNNER, str(REPO / path)],
                        cwd=directory, env={"RUN_CLOUD_TESTS": "1"},
                        stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        timeout=10, check=False,
                    )
                    self.assert_refused(result, path)

    def test_tombstones_have_no_environment_or_provider_import_path(self) -> None:
        for path in (*LEGACY_PATHS, "kestrel_sovereign/legacy_runpod.py"):
            with self.subTest(path=path):
                tree = ast.parse((REPO / path).read_text())
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        imports.add(node.module)
                self.assertLessEqual(
                    imports, {"__future__", "pathlib", "sys", "kestrel_sovereign.legacy_runpod"}
                )

    def test_registry_refusal_preserves_original_file_and_memory_records(self) -> None:
        with tempfile.TemporaryDirectory(prefix="retired-runpod-registry-") as directory:
            for operation in ("direct", "recover", "all"):
                with self.subTest(operation=operation):
                    result = subprocess.run(
                        [sys.executable, "-I", "-S", "-B", "-c", REGISTRY_RUNNER,
                         str(REPO), str(Path(directory, operation + ".json")), operation],
                        cwd=directory, env={}, stdin=subprocess.DEVNULL,
                        capture_output=True, text=True, timeout=10, check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    observed = json.loads(result.stdout.splitlines()[-1])
                    self.assertIsInstance(observed["error"], str, observed)
                    self.assertIn("synthetic-original-id", observed["error"])
                    self.assertIn("remains unresolved", observed["error"])
                    self.assertTrue(observed["same_original_bytes"])
                    self.assertTrue(observed["original_file_retained"])
                    self.assertFalse(observed["cleanup_in_progress"])
                    self.assertEqual(observed["memory_record_retained"], operation == "all")


if __name__ == "__main__":
    unittest.main()
