"""SQLite cache for GitHub content."""
import json
import logging
import os
import aiosqlite
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .models import FileContent

logger = logging.getLogger(__name__)


class GitHubCache:
    """Async SQLite-backed cache for GitHub file content."""
    
    DEFAULT_TTL = timedelta(hours=1)
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize cache.
        
        Args:
            db_path: Path to SQLite database. Defaults to KESTREL_DB_PATH/github_cache.db
        """
        if db_path is None:
            base_path = os.getenv("KESTREL_DB_PATH", "./agent_data")
            db_path = os.path.join(base_path, "github_cache.db")
        
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Initialize database schema if not already done."""
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS file_cache (
                    cache_key TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    path TEXT NOT NULL,
                    ref TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sha TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    cached_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_cache_repo 
                ON file_cache(repo, path)
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tree_cache (
                    cache_key TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    ref TEXT NOT NULL,
                    tree_json TEXT NOT NULL,
                    cached_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)
            await conn.commit()
        self._initialized = True
    
    def _make_key(self, repo: str, path: str, ref: str) -> str:
        """Create cache key."""
        return f"{repo}:{ref}:{path}"
    
    async def get(self, repo: str, path: str, ref: str = "main") -> Optional[FileContent]:
        """Get file from cache if not expired.
        
        Args:
            repo: Repository in 'owner/repo' format
            path: Path to file
            ref: Branch, tag, or SHA
            
        Returns:
            FileContent if cached and not expired, else None
        """
        await self._ensure_initialized()
        key = self._make_key(repo, path, ref)
        now = datetime.now(timezone.utc).isoformat()
        
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                """
                SELECT content, sha, size, cached_at 
                FROM file_cache 
                WHERE cache_key = ? AND expires_at > ?
                """,
                (key, now),
            )
            row = await cursor.fetchone()
            
            if row:
                return FileContent(
                    path=path,
                    content=row[0],
                    sha=row[1],
                    size=row[2],
                    repo=repo,
                    ref=ref,
                    cached_at=datetime.fromisoformat(row[3]),
                )
        
        return None
    
    async def set(
        self,
        file_content: FileContent,
        ttl: Optional[timedelta] = None,
    ):
        """Cache file content.
        
        Args:
            file_content: Content to cache
            ttl: Time to live, defaults to 1 hour
        """
        await self._ensure_initialized()
        if ttl is None:
            ttl = self.DEFAULT_TTL
        
        key = self._make_key(file_content.repo, file_content.path, file_content.ref)
        now = datetime.now(timezone.utc)
        expires = now + ttl
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO file_cache 
                (cache_key, repo, path, ref, content, sha, size, cached_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    file_content.repo,
                    file_content.path,
                    file_content.ref,
                    file_content.content,
                    file_content.sha,
                    file_content.size,
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )
            await conn.commit()
    
    async def get_tree(self, repo: str, ref: str = "main") -> Optional[list[dict]]:
        """Get cached repository tree.
        
        Args:
            repo: Repository in 'owner/repo' format
            ref: Branch, tag, or SHA
            
        Returns:
            List of tree entries if cached, else None
        """
        await self._ensure_initialized()
        key = f"{repo}:{ref}:tree"
        now = datetime.now(timezone.utc).isoformat()
        
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                """
                SELECT tree_json FROM tree_cache 
                WHERE cache_key = ? AND expires_at > ?
                """,
                (key, now),
            )
            row = await cursor.fetchone()
            
            if row:
                return json.loads(row[0])
        
        return None
    
    async def set_tree(
        self,
        repo: str,
        ref: str,
        tree: list[dict],
        ttl: Optional[timedelta] = None,
    ):
        """Cache repository tree.
        
        Args:
            repo: Repository in 'owner/repo' format
            ref: Branch, tag, or SHA
            tree: Tree entries to cache
            ttl: Time to live
        """
        await self._ensure_initialized()
        if ttl is None:
            ttl = self.DEFAULT_TTL
        
        key = f"{repo}:{ref}:tree"
        now = datetime.now(timezone.utc)
        expires = now + ttl
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO tree_cache 
                (cache_key, repo, ref, tree_json, cached_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, repo, ref, json.dumps(tree), now.isoformat(), expires.isoformat()),
            )
            await conn.commit()
    
    async def invalidate(self, repo: str, path: Optional[str] = None, ref: Optional[str] = None):
        """Invalidate cache entries.
        
        Args:
            repo: Repository to invalidate
            path: Specific path to invalidate (optional)
            ref: Specific ref to invalidate (optional)
        """
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            if path and ref:
                key = self._make_key(repo, path, ref)
                await conn.execute("DELETE FROM file_cache WHERE cache_key = ?", (key,))
            elif ref:
                await conn.execute(
                    "DELETE FROM file_cache WHERE repo = ? AND ref = ?",
                    (repo, ref),
                )
                await conn.execute(
                    "DELETE FROM tree_cache WHERE repo = ? AND ref = ?",
                    (repo, ref),
                )
            else:
                await conn.execute("DELETE FROM file_cache WHERE repo = ?", (repo,))
                await conn.execute("DELETE FROM tree_cache WHERE repo = ?", (repo,))
            await conn.commit()
    
    async def clear_expired(self):
        """Remove all expired cache entries."""
        await self._ensure_initialized()
        now = datetime.now(timezone.utc).isoformat()
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("DELETE FROM file_cache WHERE expires_at < ?", (now,))
            await conn.execute("DELETE FROM tree_cache WHERE expires_at < ?", (now,))
            await conn.commit()
    
    async def stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        await self._ensure_initialized()
        now = datetime.now(timezone.utc).isoformat()
        
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM file_cache WHERE expires_at > ?", (now,)
            )
            file_count = (await cursor.fetchone())[0]
            
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM tree_cache WHERE expires_at > ?", (now,)
            )
            tree_count = (await cursor.fetchone())[0]
            
            cursor = await conn.execute(
                "SELECT COALESCE(SUM(size), 0) FROM file_cache WHERE expires_at > ?", (now,)
            )
            total_size = (await cursor.fetchone())[0]
            
            cursor = await conn.execute(
                "SELECT DISTINCT repo FROM file_cache WHERE expires_at > ?", (now,)
            )
            repos = await cursor.fetchall()
        
        return {
            "cached_files": file_count,
            "cached_trees": tree_count,
            "total_size_bytes": total_size,
            "repos": [r[0] for r in repos],
        }
