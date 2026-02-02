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

import asyncio
import os
import subprocess
import sys
import argparse
import time
import re
from pathlib import Path
from typing import Optional

# Use uv for running pytest to ensure correct environment
UV_PREFIX = ["uv", "run"]

# Service configuration - loaded lazily when needed
# Environment variables are validated only when service checks are performed
API_BASE_URL = os.getenv("BASE_URL", "http://localhost:7777")
KESTREL_API_URL = os.getenv("KESTREL_URL", "http://localhost:8888")


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
        self.root_dir = Path(__file__).parent
        self.kestrel_tests = self.root_dir / "tests"

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
        self.log("Validating test imports (should be <5s)...", "running")
        start = time.time()

        cmd = UV_PREFIX + ["pytest", "--collect-only", "-q"]
        for test_dir in test_dirs:
            cmd.append(str(test_dir))

        # Ignore heavy test directories
        cmd.extend([
            "--ignore=tests/load/",
            "--ignore=tests/infrastructure/",
        ])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.root_dir,
        )

        elapsed = time.time() - start

        if result.returncode != 0:
            self.log(f"Import validation FAILED in {elapsed:.1f}s", "error")
            print("\n--- STDERR ---")
            print(result.stderr)
            print("\n--- STDOUT ---")
            print(result.stdout)

            if "ModuleNotFoundError" in result.stderr or "ModuleNotFoundError" in result.stdout:
                self.log("Missing module detected! Check for orphaned imports.", "warning")

            return False

        # Warn about slow collection but don't fail - CI environments are slower
        if elapsed > 30:
            self.log(f"Collection took {elapsed:.1f}s - check for heavy imports!", "warning")
            return False
        elif elapsed > 10:
            self.log(f"Collection took {elapsed:.1f}s (slow but acceptable)", "warning")

        # Parse test count
        output = result.stdout + result.stderr
        match = re.search(r'(\d+)\s+tests?\s+collected', output)
        test_count = int(match.group(1)) if match else 0

        self.log(f"Validated {test_count} tests in {elapsed:.1f}s", "success")
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
            elif f == "server.py":
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
        cmd = UV_PREFIX + ["pytest"]

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
                "--cov=.",
                "--cov-report=term-missing",
                "--cov-report=html:coverage_html",
                "--cov-report=json:coverage.json",
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


def main():
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
    if args.ci:
        args.parallel = args.parallel or "auto"
        args.coverage = True

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
                print("VALIDATION FAILED - Fix import errors first")
                print("=" * 60)
                sys.exit(1)
        except subprocess.TimeoutExpired:
            runner.log("Import validation TIMED OUT (>30s)", "error")
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
