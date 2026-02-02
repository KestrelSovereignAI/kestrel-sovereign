"""
Parallel test execution support utilities.

Provides worker-isolated unique identifiers and test data factories
for safe parallel execution with pytest-xdist.

When running tests in parallel with pytest-xdist (-n N), each worker
gets a unique ID (gw0, gw1, etc.). These utilities ensure test data
created by different workers doesn't collide.

Usage:
    from tests.utils.parallel_support import unique_email, TestDataFactory

    # Simple usage
    email = unique_email("alice")  # -> alice_gw0_abc123@test.YOUR_DOMAIN.com

    # Factory pattern (recommended for cleanup tracking)
    factory = TestDataFactory()
    email = factory.email("bob")
    name = factory.companion_name("MyBot")
    # Later: await factory.cleanup(db_connection)
"""

import os
import uuid
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def get_worker_id() -> str:
    """
    Get the pytest-xdist worker ID.

    Returns:
        Worker ID string (e.g., 'gw0', 'gw1') or 'master' for sequential runs
    """
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def unique_email(prefix: str, domain: str = "test.YOUR_DOMAIN.com") -> str:
    """
    Generate worker-isolated unique email address.

    Args:
        prefix: Email prefix (e.g., 'alice', 'auth.test')
        domain: Email domain (default: test.YOUR_DOMAIN.com)

    Returns:
        Unique email like 'alice_gw0_abc123@test.YOUR_DOMAIN.com'

    Example:
        >>> unique_email("alice")
        'alice_gw0_a1b2c3d4@test.YOUR_DOMAIN.com'
    """
    worker = get_worker_id()
    unique = uuid.uuid4().hex[:8]
    return f"{prefix}_{worker}_{unique}@{domain}"


def unique_name(prefix: str) -> str:
    """
    Generate worker-isolated unique name.

    Args:
        prefix: Name prefix (e.g., 'AliceBot', 'TestCompanion')

    Returns:
        Unique name like 'AliceBot_gw0_abc123'

    Example:
        >>> unique_name("TestBot")
        'TestBot_gw0_a1b2c3d4'
    """
    worker = get_worker_id()
    unique = uuid.uuid4().hex[:8]
    return f"{prefix}_{worker}_{unique}"


def worker_cleanup_pattern() -> str:
    """
    Get SQL LIKE pattern for current worker's test data only.

    Returns:
        SQL LIKE pattern matching this worker's email addresses

    Example:
        >>> worker_cleanup_pattern()
        '%_gw0_%@test.kestrel%'
    """
    worker = get_worker_id()
    return f"%_{worker}_%@test.kestrel%"


def redis_key(key: str) -> str:
    """
    Prefix Redis key with worker ID for parallel isolation.

    Args:
        key: The original Redis key

    Returns:
        Worker-prefixed key like 'test:gw0:session:user123'

    Example:
        >>> redis_key("session:user123")
        'test:gw0:session:user123'
    """
    worker = get_worker_id()
    return f"test:{worker}:{key}"


class TestDataFactory:
    """
    Factory for creating isolated test data with automatic cleanup tracking.

    Each factory instance generates unique identifiers scoped to:
    - The pytest-xdist worker (gw0, gw1, etc.)
    - A unique run ID per factory instance

    This ensures complete isolation even when the same test runs
    multiple times in the same worker.

    Usage:
        factory = TestDataFactory()
        email = factory.email("alice")
        name = factory.companion_name("BotHelper")

        # Later, cleanup only this factory's data
        await factory.cleanup(db_connection)

    Example:
        @pytest.fixture
        def test_data(real_database):
            factory = TestDataFactory()
            yield factory
            # Cleanup happens automatically or manually
    """

    def __init__(self, worker_id: Optional[str] = None):
        """
        Initialize factory with optional worker ID override.

        Args:
            worker_id: Override worker ID (mainly for testing)
        """
        self.worker_id = worker_id or get_worker_id()
        self.run_id = uuid.uuid4().hex[:8]
        self._created_emails: list[str] = []
        self._created_names: list[str] = []
        self._created_redis_keys: list[str] = []

    def email(self, prefix: str, domain: str = "test.YOUR_DOMAIN.com") -> str:
        """
        Generate and track a unique email.

        Args:
            prefix: Email prefix
            domain: Email domain

        Returns:
            Unique email address
        """
        email = f"{prefix}_{self.worker_id}_{self.run_id}_{uuid.uuid4().hex[:6]}@{domain}"
        self._created_emails.append(email)
        return email

    def companion_name(self, prefix: str) -> str:
        """
        Generate and track a unique companion name.

        Args:
            prefix: Name prefix

        Returns:
            Unique companion name
        """
        name = f"{prefix}_{self.worker_id}_{self.run_id}_{uuid.uuid4().hex[:6]}"
        self._created_names.append(name)
        return name

    def redis_key(self, key: str) -> str:
        """
        Generate and track a Redis key.

        Args:
            key: Base key name

        Returns:
            Worker-isolated Redis key
        """
        full_key = f"test:{self.worker_id}:{self.run_id}:{key}"
        self._created_redis_keys.append(full_key)
        return full_key

    def cleanup_email_pattern(self) -> str:
        """
        Get SQL LIKE pattern for this factory's emails only.

        Returns:
            Pattern like '%_gw0_abc123_%@test.kestrel%'
        """
        return f"%_{self.worker_id}_{self.run_id}_%"

    def cleanup_redis_pattern(self) -> str:
        """
        Get Redis SCAN pattern for this factory's keys only.

        Returns:
            Pattern like 'test:gw0:abc123:*'
        """
        return f"test:{self.worker_id}:{self.run_id}:*"

    async def cleanup_database(self, db) -> int:
        """
        Clean up all database data created by this factory.

        Args:
            db: asyncpg connection

        Returns:
            Number of records deleted
        """
        pattern = self.cleanup_email_pattern()
        deleted = 0

        # Delete in correct order (respecting foreign keys)
        try:
            result = await db.execute(
                "DELETE FROM messages WHERE user_id IN "
                "(SELECT id FROM users WHERE email LIKE $1)",
                pattern,
            )
            deleted += int(result.split()[-1]) if result else 0
        except Exception:
            pass

        try:
            result = await db.execute(
                "DELETE FROM companions WHERE user_id IN "
                "(SELECT id FROM users WHERE email LIKE $1)",
                pattern,
            )
            deleted += int(result.split()[-1]) if result else 0
        except Exception:
            pass

        try:
            result = await db.execute(
                "DELETE FROM users WHERE email LIKE $1",
                pattern,
            )
            deleted += int(result.split()[-1]) if result else 0
        except Exception:
            pass

        return deleted

    async def cleanup_redis(self, redis_client) -> int:
        """
        Clean up all Redis keys created by this factory.

        Args:
            redis_client: Redis async client

        Returns:
            Number of keys deleted
        """
        pattern = self.cleanup_redis_pattern()
        deleted = 0

        try:
            keys = []
            async for key in redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await redis_client.delete(*keys)
        except Exception:
            pass

        return deleted

    @property
    def created_count(self) -> dict:
        """Get count of created test data."""
        return {
            "emails": len(self._created_emails),
            "names": len(self._created_names),
            "redis_keys": len(self._created_redis_keys),
        }
