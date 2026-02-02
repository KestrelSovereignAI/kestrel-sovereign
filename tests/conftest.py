"""
Pytest configuration and shared fixtures for Kestrel tests.

This module provides reusable fixtures for:
- Temporary directories and files with automatic cleanup
- Test database instances
- Mock agent environments
- Common test utilities
"""
import pytest
import tempfile
import shutil
import os
import threading
import asyncio
from pathlib import Path
from typing import Generator

# Import shared test infrastructure for resource cleanup
from tests.shared.pytest_cleanup_plugin import (
    pytest_configure as _cleanup_configure,
    pytest_sessionstart as _cleanup_sessionstart,
    pytest_keyboard_interrupt,
    pytest_collection_modifyitems as _cleanup_collection_modifyitems,
    resource_tracker,
    cost_tracking,
)
from tests.shared.resource_registry import registry

# Import feedback bridge for test-to-reflection integration
from tests.utils.feedback_bridge import (
    pytest_addoption as _feedback_addoption,
    pytest_configure as _feedback_configure,
)


def pytest_addoption(parser):
    """Add command-line options from all plugins."""
    _feedback_addoption(parser)
    # Add --run-load and --run-cloud options
    parser.addoption(
        "--run-load",
        action="store_true",
        default=False,
        help="Run load/stress tests (skipped by default)"
    )
    parser.addoption(
        "--run-cloud",
        action="store_true",
        default=False,
        help="Run tests that require cloud resources (RunPod, etc.)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip cloud_resource tests unless --run-cloud is provided."""
    _cleanup_collection_modifyitems(config, items)


def pytest_configure(config):
    """Configure pytest with all plugins."""
    _cleanup_configure(config)
    _feedback_configure(config)


def pytest_sessionfinish(session, exitstatus):
    """
    Combined hook for resource cleanup and debug output.

    Cleans up any tracked resources (subprocesses, docker containers, etc)
    and reports on any remaining threads/tasks that might cause hangs.

    After all tests have completed, if orphaned aiosqlite threads remain,
    we force exit to prevent CI from hanging.
    """
    import sys
    import os as _os
    from tests.shared.cost_tracker import cost_tracker

    # 1. Clean up any tracked resources first
    registry.cleanup_all()

    # 2. Print cost report if any cloud resources were used
    if cost_tracker.total_cost > 0:
        print(cost_tracker.report())
        cost_tracker.save_report()

    # 3. Shutdown default ThreadPoolExecutor used by asyncio.to_thread
    # This is critical - asyncio.to_thread uses a global executor that creates
    # non-daemon threads, which prevent Python from exiting
    try:
        # Get the default executor from any existing loop
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            pass

        if loop and hasattr(loop, '_default_executor') and loop._default_executor:
            print("\n[CLEANUP] Shutting down asyncio default executor...", file=sys.stderr)
            loop._default_executor.shutdown(wait=False, cancel_futures=True)
            loop._default_executor = None
    except Exception as e:
        print(f"\n[CLEANUP] Error shutting down executor: {e}", file=sys.stderr)

    # 4. Stop aiosqlite threads by sending them the stop sentinel
    # aiosqlite.Connection objects ARE threads (they extend Thread)
    # We can stop them by putting the _STOP_RUNNING_SENTINEL on their queue
    active_threads = threading.enumerate()
    aiosqlite_threads = []
    for t in active_threads:
        if t.name == "MainThread" or t.daemon:
            continue
        # Check if this is an aiosqlite Connection thread
        if type(t).__module__ == 'aiosqlite.core' and type(t).__name__ == 'Connection':
            aiosqlite_threads.append(t)

    if aiosqlite_threads:
        print(f"\n[CLEANUP] Found {len(aiosqlite_threads)} orphaned aiosqlite threads, stopping them...", file=sys.stderr)
        from aiosqlite.core import _STOP_RUNNING_SENTINEL
        for conn_thread in aiosqlite_threads:
            try:
                # Send stop sentinel to the thread's queue
                if hasattr(conn_thread, '_tx'):
                    conn_thread._tx.put_nowait(_STOP_RUNNING_SENTINEL)
            except Exception as e:
                print(f"  [CLEANUP] Error stopping thread {conn_thread.name}: {e}", file=sys.stderr)

        # Give threads time to process the stop sentinel
        for t in aiosqlite_threads:
            t.join(timeout=1.0)

    # 5. Final check for any remaining non-daemon threads
    active_threads = threading.enumerate()
    non_daemon_threads = [t for t in active_threads
                         if t.name != "MainThread" and not t.daemon and t.is_alive()]

    if non_daemon_threads:
        print(f"\n[CLEANUP] {len(non_daemon_threads)} non-daemon threads still alive:", file=sys.stderr)
        for t in non_daemon_threads[:5]:
            print(f"  - {t.name} ({type(t).__module__}.{type(t).__name__})", file=sys.stderr)

        # Force exit to prevent CI timeout
        # This is safe because ALL tests have completed at this point
        print(f"\n[CLEANUP] Force exiting to prevent CI hang (exit code: {exitstatus})", file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        _os._exit(exitstatus)


# =============================================================================
# Session-scoped fixtures (shared across all tests)
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_config():
    """Setup test configuration before running tests."""
    # Create a temporary config for tests
    config_dir = Path(__file__).parent.parent
    config_path = config_dir / "llm_config.toml"
    
    # Only create if it doesn't exist
    if not config_path.exists():
        example_path = config_dir / "llm_config.toml.example"
        if example_path.exists():
            config_path.write_text(example_path.read_text())


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


# =============================================================================
# Temporary directory fixtures with automatic cleanup
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test files.
    
    Automatically cleans up after the test completes.
    
    Usage:
        def test_something(temp_dir):
            test_file = temp_dir / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
        # temp_dir is automatically deleted after test
    """
    with tempfile.TemporaryDirectory(prefix="kestrel_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """
    Create a temporary file for testing.
    
    Returns a Path to a temporary file that is automatically deleted.
    
    Usage:
        def test_file_ops(temp_file):
            temp_file.write_text("test data")
            assert temp_file.read_text() == "test data"
    """
    with tempfile.NamedTemporaryFile(
        prefix="kestrel_test_",
        suffix=".tmp",
        delete=False
    ) as f:
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_db(temp_dir: Path) -> Generator[Path, None, None]:
    """
    Create a temporary SQLite database file.
    
    Usage:
        def test_storage(temp_db):
            storage = Storage(db_path=str(temp_db))
            # ... test with storage
        # Database file is automatically cleaned up
    """
    db_path = temp_dir / "test.db"
    yield db_path
    # Cleanup handled by temp_dir fixture


# =============================================================================
# Agent environment fixtures
# =============================================================================

@pytest.fixture
def agent_data_dir(temp_dir: Path) -> Generator[Path, None, None]:
    """
    Create a temporary agent data directory with proper structure.
    
    Creates the directory structure expected by Kestrel agents.
    
    Usage:
        def test_agent(agent_data_dir):
            # agent_data_dir is a clean directory for agent files
            agent = create_agent(data_dir=agent_data_dir)
    """
    # Create subdirectories that agents expect
    (temp_dir / "keys").mkdir(exist_ok=True)
    (temp_dir / "files").mkdir(exist_ok=True)
    
    yield temp_dir


@pytest.fixture
def mock_agent_env(agent_data_dir: Path, monkeypatch) -> Generator[Path, None, None]:
    """
    Set up a complete mock agent environment.
    
    Patches storage.get_default_agent_data_dir to return the temp directory
    and optionally sets up other environment variables.
    
    Usage:
        def test_with_mock_env(mock_agent_env):
            # Storage and other components will use mock_agent_env
            from kestrel_sovereign.storage import get_default_agent_data_dir
            assert get_default_agent_data_dir() == str(mock_agent_env)
    """
    import kestrel_sovereign.storage
    monkeypatch.setattr(storage, "get_default_agent_data_dir", lambda: str(agent_data_dir))
    
    yield agent_data_dir


# =============================================================================
# Test data fixtures
# =============================================================================

@pytest.fixture
def sample_text_file(temp_dir: Path) -> Generator[Path, None, None]:
    """
    Create a sample text file for testing.
    
    Usage:
        def test_file_processing(sample_text_file):
            content = sample_text_file.read_text()
            assert "sample" in content
    """
    file_path = temp_dir / "sample.txt"
    file_path.write_text("This is sample text for testing.\nLine 2.\nLine 3.")
    yield file_path


@pytest.fixture
def sample_json_file(temp_dir: Path) -> Generator[Path, None, None]:
    """
    Create a sample JSON file for testing.
    """
    import json
    file_path = temp_dir / "sample.json"
    file_path.write_text(json.dumps({
        "name": "test",
        "value": 42,
        "items": ["a", "b", "c"]
    }, indent=2))
    yield file_path


# =============================================================================
# Configuration fixtures
# =============================================================================

@pytest.fixture
def default_config_path(project_root: Path) -> Path:
    """Get the default config path."""
    return project_root / "llm_config.toml"


@pytest.fixture
def test_master_key() -> str:
    """
    Provide a test master key for encryption tests.
    
    This is a known key for testing only - never use in production.
    """
    return "test-master-key-for-encryption-32chars!"


@pytest.fixture
def env_with_master_key(test_master_key: str, monkeypatch) -> Generator[str, None, None]:
    """
    Set KESTREL_DATA_KEY environment variable for tests.
    
    Usage:
        def test_encryption(env_with_master_key):
            # KESTREL_DATA_KEY is now set
            storage = SecureKeyStorage()
            # ... test encryption
    """
    monkeypatch.setenv("KESTREL_DATA_KEY", test_master_key)
    yield test_master_key


# =============================================================================
# Async fixtures
# =============================================================================

@pytest.fixture
def anyio_backend():
    """Configure anyio to use asyncio backend."""
    return "asyncio"


@pytest.fixture
async def async_llm_service():
    """
    Create an LLMService instance that is properly closed after the test.

    This prevents "Event loop is closed" errors during test teardown.

    Usage:
        @pytest.mark.asyncio
        async def test_llm(async_llm_service):
            response = await async_llm_service.get_response(...)
        # Service is automatically closed after test
    """
    from kestrel_sovereign.llm.service import LLMService
    service = LLMService()
    yield service
    await service.close()


@pytest.fixture
async def async_kestrel_agent(temp_dir: Path, monkeypatch):
    """
    Create a KestrelAgent instance that is properly shutdown after the test.

    This prevents "Event loop is closed" errors from unclosed HTTP clients.

    IMPORTANT: This fixture sets KESTREL_DATA_KEY to enable encryption at rest,
    which is required for Data Sanctity (Article I) compliance.

    Usage:
        @pytest.mark.asyncio
        async def test_agent(async_kestrel_agent):
            response = await async_kestrel_agent.process_input("hello")
        # Agent is automatically shutdown after test
    """
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.inception_service import create_kestrel_identity_async

    # Enable encryption at rest - required for Data Sanctity (Article I) compliance
    # This key is for testing only - never use in production
    test_encryption_key = "test-master-key-for-encryption-32chars!"
    monkeypatch.setenv("KESTREL_DATA_KEY", test_encryption_key)

    # Create agent identity (use async version since we're in async context)
    await create_kestrel_identity_async(str(temp_dir), "docs/principles/KESTREL_CONSTITUTION.md")

    # Initialize storage and agent
    db_files = list(temp_dir.glob("*.db"))
    db_path = str(db_files[0]) if db_files else str(temp_dir / "test.db")

    llm_service = LLMService()

    # Use new KestrelAgent API: storage_path instead of storage object
    agent = KestrelAgent(
        did="did:test:agent",
        storage_path=db_path,
        llm_service=llm_service
    )
    await agent.initialize()

    yield agent

    # Cleanup both agent and llm_service
    await agent.shutdown()
    await llm_service.close()


# =============================================================================
# Cleanup utilities
# =============================================================================

def safe_cleanup(path: Path) -> None:
    """
    Safely clean up a path (file or directory).

    Ignores errors if path doesn't exist.
    """
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
    except (OSError, FileNotFoundError):
        pass


# =============================================================================
# Dual-Backend Testing Fixtures (SQLite + PostgreSQL)
# =============================================================================

@pytest.fixture(params=["sqlite", "postgres"])
async def db_backend(request, tmp_path):
    """
    Parametrized fixture for testing against both database backends.

    Tests using this fixture run twice: once with SQLite, once with PostgreSQL.
    PostgreSQL tests are skipped if the database is not available.

    Usage:
        @pytest.mark.asyncio
        @pytest.mark.dual_backend
        async def test_storage_operation(db_backend):
            await db_backend.execute("CREATE TABLE test (id INTEGER)")
            # This test runs against both SQLite and PostgreSQL
    """
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    if request.param == "sqlite":
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.connect()
        yield backend
        await backend.close()
    else:
        # PostgreSQL requires the database to be running
        try:
            from kestrel_sovereign.storage.db.postgres import PostgresBackend
        except ImportError:
            pytest.skip("PostgresBackend not available")
            return

        postgres_url = os.environ.get("TEST_POSTGRES_URL")
        if not postgres_url:
            pytest.skip(
                "TEST_POSTGRES_URL environment variable required for PostgreSQL tests.\n"
                "Example: export TEST_POSTGRES_URL='postgresql://user:pass@localhost:5433/kestrel_test'"
            )
            return

        try:
            backend = PostgresBackend(postgres_url)
            await backend.connect()
        except Exception as e:
            pytest.skip(f"PostgreSQL not available: {e}")
            return

        try:
            yield backend
        finally:
            await backend.close()


@pytest.fixture
async def sqlite_backend(tmp_path):
    """
    SQLite-only fixture for tests that don't need dual-backend testing.

    Use this when a test specifically targets SQLite behavior or when
    PostgreSQL testing is not needed.

    Usage:
        @pytest.mark.asyncio
        async def test_sqlite_specific(sqlite_backend):
            # This test only runs against SQLite
            await sqlite_backend.execute("...")
    """
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "test.db"))
    await backend.connect()
    yield backend
    await backend.close()



