#!/usr/bin/env python3
"""
Kestrel Smart Test Runner - Unified Test System

A comprehensive test runner that combines all features:
- Fast import validation (<5s)
- Service health checks (PostgreSQL, Redis, API)
- Parallel execution with pytest-xdist
- Smart test selection (affected files, last failed)
- Coverage reporting
- Cloud/Docker target testing

Usage:
    # Basic usage
    ./run_tests.py                          # Run all tests
    ./run_tests.py --kestrel                # Kestrel tests only

    # Speed optimizations
    ./run_tests.py --parallel auto          # Parallel execution
    ./run_tests.py --affected               # Only changed files
    ./run_tests.py --failed                 # Re-run last failed

    # Test filtering
    ./run_tests.py --unit                   # Unit tests only
    ./run_tests.py --integration            # Integration tests only
    ./run_tests.py --fast                   # Skip slow tests
    ./run_tests.py -k "auth"                # Pattern match

    # Cloud/Docker testing
    ./run_tests.py --api-only --target https://dev.YOUR_DOMAIN.com

    # CI/CD mode
    ./run_tests.py --ci                     # Full suite + parallel + coverage
"""

import argparse
import asyncio
import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent

# This is a regression budget for the duration pytest reports in its collection
# summary. It deliberately excludes uv resolution and subprocess startup.
IMPORT_VALIDATION_COLLECTION_BUDGET_SECONDS = 60.0
# Provisional early-warning threshold on the same pytest-reported metric. The
# import-validation output now exposes that metric so ubuntu-latest can provide
# a direct baseline; do not calibrate it from the old uv-inclusive wall time.
IMPORT_VALIDATION_COLLECTION_WARNING_SECONDS = 45.0
# This separate wall-clock timeout is only a guard against a hung pytest process.
IMPORT_VALIDATION_PROCESS_TIMEOUT_SECONDS = 120

_RUN_TESTS_REEXEC_ENV = "_KESTREL_RUN_TESTS_REEXEC"
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_COLLECTION_SUMMARY_RE = re.compile(
    r"^(?:=+\s*)?"
    r"(?:\d+/)?(?P<count>\d+|no) tests? collected"
    r"(?: \([^\r\n)]*\))? in "
    r"(?P<seconds>\d+(?:\.\d+)?)s"
    r"(?: \([^\r\n)]*\))?"
    r"(?:\s*=+)?$",
    re.MULTILINE,
)

# Test-owned seams avoid replacing process-wide stdlib functions.
_execvpe = os.execvpe
_find_spec = importlib.util.find_spec
_now = time.perf_counter
_run = subprocess.run

# Service configuration - loaded lazily when needed
# Environment variables are validated only when service checks are performed
API_BASE_URL = os.getenv("BASE_URL", "http://localhost:7777")
KESTREL_API_URL = os.getenv("KESTREL_URL", "http://localhost:8888")


def _project_environment_dir() -> Path:
    """Return the uv environment directory for this checkout."""
    configured = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if configured:
        environment_dir = Path(configured).expanduser()
        if not environment_dir.is_absolute():
            environment_dir = PROJECT_ROOT / environment_dir
    else:
        environment_dir = PROJECT_ROOT / ".venv"
    return environment_dir.resolve()


def interpreter_uses_project_environment(interpreter: str | Path) -> bool:
    """Return whether an interpreter path belongs to this project's uv env."""
    executable_dir = Path(interpreter).absolute().parent.resolve()
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    return executable_dir == (_project_environment_dir() / scripts_dir).resolve()


def is_project_environment() -> bool:
    """Return whether this process can run tests in the project environment."""
    return (
        interpreter_uses_project_environment(sys.executable)
        and _find_spec("pytest") is not None
    )


def ensure_project_environment() -> None:
    """Re-exec a bare runner invocation once inside the uv project env."""
    if is_project_environment():
        return

    if os.environ.get(_RUN_TESTS_REEXEC_ENV) == "1":
        print(
            "run_tests.py could not enter the project test environment at "
            f"{_project_environment_dir()} with pytest installed. "
            "Run `uv sync --group test`, then retry.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    command = [
        "uv",
        "run",
        "--project",
        str(PROJECT_ROOT),
        "python",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    environment = os.environ.copy()
    environment[_RUN_TESTS_REEXEC_ENV] = "1"
    try:
        _execvpe(command[0], command, environment)
    except FileNotFoundError:
        print(
            "run_tests.py requires uv to enter the project test environment. "
            "Install uv and retry.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


def parse_collection_summary(output: str) -> tuple[int, float] | None:
    """Parse the last complete pytest collection summary line."""
    normalized_output = _ANSI_ESCAPE_RE.sub("", output)
    matches = list(_COLLECTION_SUMMARY_RE.finditer(normalized_output))
    if not matches:
        return None
    match = matches[-1]
    count_text = match.group("count")
    count = 0 if count_text == "no" else int(count_text)
    return count, float(match.group("seconds"))


def parse_collection_duration(output: str) -> float | None:
    """Return pytest's reported collection duration, when present."""
    summary = parse_collection_summary(output)
    return summary[1] if summary is not None else None


def get_database_url() -> str:
    """Get DATABASE_URL, raising an error if not set."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL environment variable is required.\n"
            "Example: export DATABASE_URL='postgresql://user:pass@localhost:5433/kestrel'\n"
            "See .env.test.example for reference."
        )
    return url


def get_redis_url() -> str:
    """Get REDIS_URL, raising an error if not set."""
    url = os.getenv("REDIS_URL")
    if not url:
        raise EnvironmentError(
            "REDIS_URL environment variable is required.\n"
            "Example: export REDIS_URL='redis://:password@localhost:6380'\n"
            "See .env.test.example for reference."
        )
    return url


class ServiceChecker:
    """Check health of required services."""

    def __init__(self):
        self.results = {}

    async def check_postgresql(self) -> bool:
        """Check PostgreSQL connectivity."""
        try:
            import asyncpg
            conn = await asyncpg.connect(get_database_url())
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            await conn.close()
            self.results["PostgreSQL"] = {"healthy": True, "tables": len(tables)}
            return True
        except ImportError:
            self.results["PostgreSQL"] = {"healthy": False, "error": "asyncpg not installed"}
            return False
        except Exception as e:
            self.results["PostgreSQL"] = {"healthy": False, "error": str(e)}
            return False

    async def check_redis(self) -> bool:
        """Check Redis connectivity."""
        try:
            import redis.asyncio as redis_lib
            client = await redis_lib.from_url(get_redis_url())
            await client.ping()
            await client.close()
            self.results["Redis"] = {"healthy": True}
            return True
        except ImportError:
            self.results["Redis"] = {"healthy": False, "error": "redis not installed"}
            return False
        except Exception as e:
            self.results["Redis"] = {"healthy": False, "error": str(e)}
            return False

    async def check_api(self, url: str, name: str) -> bool:
        """Check API server health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    self.results[name] = {"healthy": True, "data": response.json()}
                    return True
                else:
                    self.results[name] = {"healthy": False, "status": response.status_code}
                    return False
        except ImportError:
            self.results[name] = {"healthy": False, "error": "httpx not installed"}
            return False
        except Exception as e:
            self.results[name] = {"healthy": False, "error": str(e)}
            return False

    async def check_all(self, include_kestrel: bool = False) -> bool:
        """Check all required services."""
        checks = [
            self.check_postgresql(),
            self.check_redis(),
        ]
        if include_kestrel:
            checks.append(self.check_api(KESTREL_API_URL, "Kestrel API"))

        results = await asyncio.gather(*checks, return_exceptions=True)
        return all(r is True for r in results if not isinstance(r, Exception))

    def print_status(self):
        """Print service status summary."""
        print("\n📊 Service Status:")
        for service, info in self.results.items():
            if info.get("healthy"):
                print(f"   ✅ {service}")
            else:
                error = info.get("error", "Unknown error")
                print(f"   ❌ {service}: {error}")


class SmartTestRunner:
    """Unified smart test runner with all features."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.root_dir = PROJECT_ROOT
        self.kestrel_tests = self.root_dir / "tests"
        self.pytest_interpreter = sys.executable

    def log(self, msg: str, level: str = "info"):
        """Log with appropriate formatting."""
        icons = {
            "info": "ℹ️ ",
            "success": "✅",
            "error": "❌",
            "warning": "⚠️ ",
            "running": "🔄",
            "time": "⏱️ ",
        }
        icon = icons.get(level, "")
        print(f"{icon} {msg}")

    def validate_imports(self, test_dirs: list[Path]) -> bool:
        """
        FAST validation: Check that all test files can be imported.
        Catches missing modules, syntax errors, import cycles.
        """
        self.log(
            "Validating test imports "
            f"({IMPORT_VALIDATION_COLLECTION_BUDGET_SECONDS:g}s collection budget; "
            f"{IMPORT_VALIDATION_PROCESS_TIMEOUT_SECONDS:g}s process hang guard)...",
            "running",
        )
        start = _now()

        cmd = [
            self.pytest_interpreter,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--color=no",
        ]
        for test_dir in test_dirs:
            cmd.append(str(test_dir))

        # Ignore heavy test directories
        cmd.extend([
            "--ignore=tests/load/",
            "--ignore=tests/infrastructure/",
        ])

        # Parent-shell pytest options and color settings can change the summary
        # format that carries the collection duration. Pin them for this fixed
        # validation command; pyproject.toml's repository config still applies.
        validation_environment = os.environ.copy()
        validation_environment["PYTEST_ADDOPTS"] = ""
        validation_environment["PY_COLORS"] = "0"
        validation_environment.pop("FORCE_COLOR", None)

        result = _run(
            cmd,
            capture_output=True,
            text=True,
            timeout=IMPORT_VALIDATION_PROCESS_TIMEOUT_SECONDS,
            cwd=self.root_dir,
            env=validation_environment,
        )

        elapsed = _now() - start

        if result.returncode != 0:
            self.log(f"Import validation FAILED in {elapsed:.1f}s", "error")
            print("\n--- STDERR ---")
            print(result.stderr)
            print("\n--- STDOUT ---")
            print(result.stdout)

            if "ModuleNotFoundError" in result.stderr or "ModuleNotFoundError" in result.stdout:
                self.log("Missing module detected! Check for orphaned imports.", "warning")

            return False

        output = result.stdout + result.stderr
        summary = parse_collection_summary(output)
        if summary is None:
            self.log(
                "Import validation FAILED: pytest exited successfully but its "
                f"collection summary could not be parsed after {elapsed:.1f}s",
                "error",
            )
            print("\n--- STDERR ---")
            print(result.stderr)
            print("\n--- STDOUT ---")
            print(result.stdout)
            return False

        test_count, collection_duration = summary
        if collection_duration > IMPORT_VALIDATION_COLLECTION_BUDGET_SECONDS:
            self.log(
                "Import validation FAILED: pytest reported "
                f"{collection_duration:.2f}s collection time "
                f"(>{IMPORT_VALIDATION_COLLECTION_BUDGET_SECONDS:g}s budget; "
                f"{elapsed:.1f}s process wall time)",
                "error",
            )
            return False

        if collection_duration > IMPORT_VALIDATION_COLLECTION_WARNING_SECONDS:
            self.log(
                f"Pytest reported {collection_duration:.2f}s collection time "
                f"(>{IMPORT_VALIDATION_COLLECTION_WARNING_SECONDS:g}s warning threshold; "
                f"{elapsed:.1f}s process wall time)",
                "warning",
            )

        self.log(
            f"Validated {test_count} tests in {collection_duration:.2f}s collection "
            f"({elapsed:.1f}s process wall time)",
            "success",
        )
        return True

    def get_changed_files(self, base: str = "origin/main") -> list[str]:
        """Get Python files changed since base branch."""
        files = []

        # Uncommitted changes
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, cwd=self.root_dir
        )
        if result.stdout.strip():
            files.extend(result.stdout.strip().split("\n"))

        # Committed changes since base
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base}...HEAD"],
            capture_output=True, text=True, cwd=self.root_dir
        )
        if result.stdout.strip():
            files.extend(result.stdout.strip().split("\n"))

        # Filter to Python files and deduplicate
        return list(set(f for f in files if f.endswith(".py")))

    def get_affected_tests(self, changed_files: list[str]) -> list[str]:
        """Map changed source files to affected test files."""
        tests = set()

        for f in changed_files:
            # Direct test file changes
            if "/tests/" in f and f.endswith(".py"):
                tests.add(f)
                continue

            # Map source files to tests
            if f.startswith("storage/"):
                tests.add("tests/unit/test_storage.py")
                tests.add("tests/integration/test_sovereignty_e2e.py")
            elif f.startswith("llm/"):
                tests.add("tests/llm/")
                tests.add("tests/unit/test_adapter_list_models.py")
            elif f.startswith("features/"):
                tests.add("tests/unit/")
            elif f.startswith("agent/"):
                tests.add("tests/integration/test_api_e2e.py")
            elif f == "kestrel_agent.py":
                tests.add("tests/integration/")
            elif f == "server.py" or f == "kestrel_sovereign/server.py":
                tests.add("tests/integration/test_api_e2e.py")

        return [t for t in tests if t]

    def build_pytest_command(
        self,
        test_dirs: list[Path],
        test_type: Optional[str] = None,
        pattern: Optional[str] = None,
        marker: Optional[str] = None,
        fast: bool = False,
        fail_fast: bool = True,
        parallel: Optional[int] = None,
        coverage: bool = False,
        failed_only: bool = False,
        specific_tests: Optional[list[str]] = None,
        feedback: bool = False,
    ) -> list[str]:
        """Build the pytest command with all options."""
        cmd = [self.pytest_interpreter, "-m", "pytest"]

        # Test paths
        if specific_tests:
            cmd.extend(specific_tests)
        else:
            for test_dir in test_dirs:
                if test_type == "unit":
                    unit_dir = test_dir / "unit"
                    if unit_dir.exists():
                        cmd.append(str(unit_dir))
                elif test_type == "integration":
                    int_dir = test_dir / "integration"
                    if int_dir.exists():
                        cmd.append(str(int_dir))
                elif test_type == "llm":
                    llm_dir = test_dir / "llm"
                    if llm_dir.exists():
                        cmd.append(str(llm_dir))
                else:
                    cmd.append(str(test_dir))

        # Common options
        cmd.extend(["-v", "--tb=short", "--color=yes"])

        # Ignore heavy directories for 'all' runs
        if not test_type and not specific_tests:
            cmd.extend([
                "--ignore=tests/load/",
                "--ignore=tests/infrastructure/",
            ])

        # Fail fast
        if fail_fast:
            cmd.append("-x")
        else:
            # pyproject.toml supplies ``-x`` through addopts.  A later
            # ``--maxfail=0`` is required to make --no-fail-fast real and to
            # let coverage CI finish combining/reporting after test failures.
            cmd.append("--maxfail=0")

        # Skip slow tests
        if fast:
            cmd.extend(["-m", "not slow"])

        # Marker filter
        if marker:
            cmd.extend(["-m", marker])

        # Pattern matching
        if pattern:
            cmd.extend(["-k", pattern])

        # Parallel execution
        if parallel:
            workers = parallel if parallel > 0 else "auto"
            cmd.extend(["-n", str(workers)])

        # Coverage
        if coverage:
            cmd.extend([
                "--cov=kestrel_sovereign",
                "--cov-report=term-missing",
                "--cov-report=html:coverage_html",
                "--cov-report=json:coverage.json",
                "--cov-report=xml:coverage.xml",
            ])

        # Failed only (last failed)
        if failed_only:
            cmd.extend(["--lf", "--lfnf=none"])

        # Feedback bridge (submit failures to feedback system)
        if feedback:
            cmd.append("--feedback-bridge")

        # Verbose
        if self.verbose:
            cmd.append("-vv")
            cmd.append("--capture=no")

        return cmd

    def run_tests(self, cmd: list[str]) -> int:
        """Execute pytest command and return exit code."""
        self.log(f"Running: {' '.join(cmd)}", "running")
        print("=" * 60)

        start = time.time()
        result = subprocess.run(cmd, cwd=self.root_dir)
        elapsed = time.time() - start

        print("=" * 60)
        if result.returncode == 0:
            self.log(f"All tests passed in {elapsed:.1f}s", "success")
        else:
            self.log(f"Tests failed (exit code {result.returncode}) in {elapsed:.1f}s", "error")

        return result.returncode


def configure_ci_defaults(args: argparse.Namespace) -> None:
    """Apply the deterministic, report-complete defaults for ``--ci``."""
    if not args.ci:
        return
    args.parallel = args.parallel or "auto"
    args.coverage = True
    # A weekly failure must still leave complete, diagnosable coverage
    # artifacts.  Failures remain failures; pytest merely runs the rest of the
    # suite before returning its non-zero status.
    args.no_fail_fast = True


def main():
    ensure_project_environment()

    parser = argparse.ArgumentParser(
        description="Kestrel Smart Test Runner - Unified Test System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic
    ./run_tests.py                          # Run all tests
    ./run_tests.py --kestrel                # Kestrel tests only

    # Speed
    ./run_tests.py --parallel auto          # Auto-detect workers
    ./run_tests.py --parallel 4             # 4 workers
    ./run_tests.py --affected               # Changed files only
    ./run_tests.py --failed                 # Re-run last failed

    # Filtering
    ./run_tests.py --unit                   # Unit tests only
    ./run_tests.py --integration            # Integration tests
    ./run_tests.py --fast                   # Skip slow tests
    ./run_tests.py -k "auth"                # Pattern match
    ./run_tests.py --auth                   # Auth marker

    # Cloud testing
    ./run_tests.py --api-only --target https://dev.YOUR_DOMAIN.com

    # CI/CD
    ./run_tests.py --ci                     # Full + parallel + coverage
        """
    )

    # Project selection
    project_group = parser.add_mutually_exclusive_group()
    project_group.add_argument(
        "--kestrel", action="store_true",
        help="Run Kestrel tests only"
    )

    # Test type selection
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument(
        "--unit", action="store_true",
        help="Run unit tests only"
    )
    type_group.add_argument(
        "--integration", action="store_true",
        help="Run integration tests only"
    )
    type_group.add_argument(
        "--llm", action="store_true",
        help="Run LLM tests only"
    )

    # Speed optimizations
    parser.add_argument(
        "--parallel", "-n", nargs="?", const="auto", metavar="N",
        help="Run tests in parallel (N workers, or 'auto')"
    )
    parser.add_argument(
        "--affected", action="store_true",
        help="Only run tests affected by changed files"
    )
    parser.add_argument(
        "--failed", action="store_true",
        help="Re-run only tests that failed last time"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip slow-running tests"
    )

    # Platform-specific markers
    parser.add_argument("--auth", action="store_true", help="Run auth tests")
    parser.add_argument("--chat", action="store_true", help="Run chat tests (kestrel)")
    parser.add_argument("--isolation", action="store_true", help="Run isolation tests (kestrel)")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--privacy", action="store_true", help="Run privacy tests")

    # Pattern and filtering
    parser.add_argument(
        "--test", "-k", type=str, metavar="PATTERN",
        help="Run tests matching pattern"
    )

    # Service/target options
    parser.add_argument(
        "--skip-check", action="store_true",
        help="Skip service health checks"
    )
    parser.add_argument(
        "--api-only", action="store_true",
        help="Run only API-based tests (for cloud targets)"
    )
    parser.add_argument(
        "--target", type=str, metavar="URL",
        help="Target URL for testing (e.g., https://dev.YOUR_DOMAIN.com)"
    )

    # Output options
    parser.add_argument(
        "--coverage", action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Extra verbose output"
    )

    # Validation options
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate imports, don't run tests"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip import validation"
    )
    parser.add_argument(
        "--no-fail-fast", action="store_true",
        help="Don't stop on first failure"
    )

    # CI mode
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: parallel + coverage + JSON output"
    )

    # Feedback integration
    parser.add_argument(
        "--feedback", action="store_true",
        help="Submit test failures to feedback system for agent reflection"
    )
    parser.add_argument(
        "--feedback-db", type=str, metavar="PATH",
        help="Path to feedback database (default: uses KESTREL_FEEDBACK_DB env)"
    )

    args = parser.parse_args()

    # Handle --target
    if args.target:
        os.environ["BASE_URL"] = args.target
        print(f"🎯 Target URL: {args.target}")

    # --api-only implies --skip-check
    if args.api_only:
        args.skip_check = True

    # --ci mode sets defaults
    configure_ci_defaults(args)

    # --feedback setup
    if args.feedback:
        feedback_db = args.feedback_db or os.environ.get("KESTREL_FEEDBACK_DB")
        if feedback_db:
            os.environ["KESTREL_FEEDBACK_DB"] = feedback_db
            print(f"📝 Feedback will be submitted to: {feedback_db}")
        else:
            print("⚠️  --feedback requires --feedback-db or KESTREL_FEEDBACK_DB env var")
            args.feedback = False

    runner = SmartTestRunner(verbose=args.verbose)

    # Determine test directories
    test_dirs = []
    if args.kestrel:
        test_dirs = [runner.kestrel_tests]
    
    else:
        test_dirs = [runner.kestrel_tests]

    # Phase 1: Service health checks (for kestrel tests)
    if not args.skip_check and not args.kestrel:
        print("\n" + "=" * 60)
        print("PHASE 0: Service Health Check")
        print("=" * 60)

        checker = ServiceChecker()
        try:
            healthy = asyncio.run(checker.check_all(
                include_kestrel=args.kestrel
            ))
            checker.print_status()

            if not healthy:
                print("\n💡 Quick setup:")
                if not args.api_only:
                    sys.exit(1)
        except KeyboardInterrupt:
            print("\n⏹️ Health check interrupted")
            sys.exit(1)

    # Phase 2: Import validation
    if not args.skip_validation and not args.failed:
        print("\n" + "=" * 60)
        print("PHASE 1: Fast Import Validation")
        print("=" * 60)

        try:
            if not runner.validate_imports(test_dirs):
                print("\n" + "=" * 60)
                print("VALIDATION FAILED - See diagnostics above")
                print("=" * 60)
                sys.exit(1)
        except subprocess.TimeoutExpired:
            runner.log(
                "Import validation process TIMED OUT after "
                f"{IMPORT_VALIDATION_PROCESS_TIMEOUT_SECONDS:g}s "
                "(process hang guard; collection budget is "
                f"{IMPORT_VALIDATION_COLLECTION_BUDGET_SECONDS:g}s of "
                "pytest-reported time)",
                "error",
            )
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n⏹️ Validation interrupted")
            sys.exit(1)

    if args.validate_only:
        print("\n✅ Validation passed!")
        sys.exit(0)

    # Phase 3: Run tests
    print("\n" + "=" * 60)
    print("PHASE 2: Running Tests")
    print("=" * 60)

    # Determine test type
    test_type = None
    if args.unit:
        test_type = "unit"
    elif args.integration:
        test_type = "integration"
    elif args.llm:
        test_type = "llm"

    # Determine marker
    marker = None
    if args.auth:
        marker = "auth"
    elif args.chat:
        marker = "chat"
    elif args.isolation:
        marker = "isolation"
    elif args.performance:
        marker = "performance"
    elif args.privacy:
        marker = "privacy"
    elif args.api_only:
        marker = "api_only"

    # Handle --affected
    specific_tests = None
    if args.affected:
        changed = runner.get_changed_files()
        if not changed:
            runner.log("No Python files changed", "info")
            sys.exit(0)

        specific_tests = runner.get_affected_tests(changed)
        if not specific_tests:
            runner.log("No tests affected by changes", "info")
            sys.exit(0)

        runner.log(f"Running {len(specific_tests)} affected test paths", "info")

    # Parse parallel argument
    parallel = None
    if args.parallel:
        if args.parallel == "auto":
            parallel = -1  # pytest-xdist auto
        else:
            parallel = int(args.parallel)

    # Build and run
    try:
        cmd = runner.build_pytest_command(
            test_dirs=test_dirs,
            test_type=test_type,
            pattern=args.test,
            marker=marker,
            fast=args.fast,
            fail_fast=not args.no_fail_fast,
            parallel=parallel,
            coverage=args.coverage,
            failed_only=args.failed,
            specific_tests=specific_tests,
            feedback=args.feedback,
        )
        exit_code = runner.run_tests(cmd)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Test run interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
