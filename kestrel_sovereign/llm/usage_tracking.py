"""Usage tracking and model management for LLM Service.

Uses the abstract data layer for both SQLite (local) and PostgreSQL (cloud) backends.
"""
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


class UsageTrackingMixin:
    """Mixin class providing usage tracking methods for LLMService.
    
    Supports both SQLite and PostgreSQL backends via the abstract data layer.
    Backend is determined by:
    1. Explicit database_url parameter (PostgreSQL)
    2. KESTREL_DATABASE_URL env var (PostgreSQL)
    3. DATABASE_URL env var (PostgreSQL) 
    4. Default to SQLite at KESTREL_DB_PATH/agent_data/llm_usage.db
    """

    # Instance attributes to be initialized
    _usage_db: Optional['AsyncDatabase']
    _db_backend: str
    _db_initialized: bool

    def _init_usage_tracking(self, database_url: Optional[str] = None):
        """Initialize usage tracking state (async initialization done in _ensure_db_initialized).
        
        Args:
            database_url: Optional PostgreSQL connection URL. If provided, uses PostgreSQL.
                         If None, checks env vars, then falls back to SQLite.
        """
        self._usage_db = None
        self._db_initialized = False
        
        # Determine backend and connection info
        self._usage_database_url = (
            database_url or 
            os.environ.get("KESTREL_DATABASE_URL") or 
            os.environ.get("DATABASE_URL")
        )
        
        if self._usage_database_url:
            self._db_backend = "postgres"
            logger.info(f"Model usage tracking configured: PostgreSQL")
        else:
            self._db_backend = "sqlite"
            db_path = os.environ.get("KESTREL_DB_PATH", "./agent_data")
            os.makedirs(db_path, exist_ok=True)
            self._usage_db_path = os.path.join(db_path, "llm_usage.db")
            logger.info(f"Model usage tracking configured: SQLite at {self._usage_db_path}")

    async def _ensure_db_initialized(self):
        """Ensure database connection is initialized (lazy initialization).
        
        Uses the abstract data layer to support both SQLite and PostgreSQL.
        The model_usage table is part of the core schema in AsyncDatabase.
        """
        if self._db_initialized:
            return

        try:
            from kestrel_sovereign.storage.async_database import AsyncDatabase
            
            if self._db_backend == "postgres":
                # PostgreSQL backend via abstract data layer
                self._usage_db = await AsyncDatabase.postgres(self._usage_database_url)
                logger.info("Model usage database initialized: PostgreSQL")
            else:
                # SQLite backend via abstract data layer
                self._usage_db = await AsyncDatabase.sqlite(self._usage_db_path)
                logger.info(f"Model usage database initialized: SQLite at {self._usage_db_path}")
            
            self._db_initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize usage tracking: {e}")
            self._usage_db = None

    async def _track_model_usage(self, model_id: str, provider: str, tokens: int = 0):
        """Track model usage for cleanup decisions."""
        await self._ensure_db_initialized()
        if not self._usage_db:
            return

        try:
            # Use UTC datetime but strip timezone for PostgreSQL TIMESTAMP compatibility
            # PostgreSQL TIMESTAMP (without timezone) requires naive datetimes
            # SQLite doesn't care about timezone info
            now = datetime.now(timezone.utc).replace(tzinfo=None)

            # Use abstract data layer - queries are backend-agnostic
            # Note: Column references in ON CONFLICT must be qualified for PostgreSQL
            await self._usage_db.execute("""
                INSERT INTO model_usage (model_id, provider, last_used, use_count, total_tokens)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    last_used = ?,
                    use_count = model_usage.use_count + 1,
                    total_tokens = model_usage.total_tokens + ?
            """, (model_id, provider, now, tokens, now, tokens))
        except Exception as e:
            logger.warning(f"Failed to track usage for {model_id}: {e}")

    async def get_storage_info(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get storage information for Ollama models."""
        # Check cache
        if use_cache and self._storage_cache is not None and self._storage_cache_timestamp is not None:
            age = time.time() - self._storage_cache_timestamp
            if age < self._storage_cache_ttl:
                logger.info(f"Using cached storage info (age: {age:.0f}s)")
                return self._storage_cache

        ollama_dir = Path.home() / ".ollama"
        if not ollama_dir.exists():
            logger.warning(f"Ollama directory not found: {ollama_dir}")
            return {"total_gb": 0, "used_gb": 0, "available_gb": 0, "models": []}

        total, used, free = shutil.disk_usage(ollama_dir)

        models_info = []
        try:
            ollama_models = await self._discover_ollama_models()
            await self._ensure_db_initialized()

            for model in ollama_models:
                model_entry = {
                    "id": model["id"],
                    "size_gb": model.get("size_gb", 0),
                    "last_used": None
                }

                if self._usage_db:
                    rows = await self._usage_db.fetchall(
                        "SELECT last_used FROM model_usage WHERE model_id = ?",
                        (model["id"],)
                    )
                    if rows:
                        model_entry["last_used"] = rows[0][0]

                models_info.append(model_entry)

        except Exception as e:
            logger.warning(f"Failed to get model details: {e}")

        storage_info = {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "available_gb": free / (1024**3),
            "models": models_info
        }

        self._storage_cache = storage_info
        self._storage_cache_timestamp = time.time()

        return storage_info

    async def pull_model(
        self,
        model_name: str,
        auto_confirm: bool = True,
        progress_callback=None
    ) -> bool:
        """Pull (download) an Ollama model."""
        ollama_provider = None
        for provider in self.providers:
            if provider.get("vendor") == "ollama":
                ollama_provider = provider
                break

        if not ollama_provider:
            raise RuntimeError("Ollama provider not configured. Cannot pull models.")

        estimated_size_gb = 2.0
        if ":0.5b" in model_name or ":1b" in model_name:
            estimated_size_gb = 0.5
        elif ":3b" in model_name:
            estimated_size_gb = 2.0
        elif ":7b" in model_name:
            estimated_size_gb = 5.0
        elif ":14b" in model_name:
            estimated_size_gb = 10.0
        elif ":70b" in model_name:
            estimated_size_gb = 50.0

        try:
            storage = await self.get_storage_info(use_cache=False)
            if storage["available_gb"] < estimated_size_gb * 1.1:
                raise RuntimeError(
                    f"Insufficient disk space. Need ~{estimated_size_gb:.1f}GB, "
                    f"available: {storage['available_gb']:.1f}GB"
                )
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            if not auto_confirm:
                raise RuntimeError(f"Cannot verify disk space: {e}")

        client = ollama_provider["client"]

        try:
            logger.info(f"Pulling model: {model_name}")

            if progress_callback:
                current_status = None
                stream = await client.pull(model=model_name, stream=True)
                async for progress in stream:
                    status = progress.status if hasattr(progress, 'status') else str(progress)
                    completed = progress.completed if hasattr(progress, 'completed') else 0
                    total = progress.total if hasattr(progress, 'total') else 0

                    if status != current_status:
                        current_status = status
                        logger.info(f"Pull status: {status}")

                    progress_callback(status, completed or 0, total or 0)

                progress_callback("success", 1, 1)
            else:
                await client.pull(model=model_name)

            logger.info(f"Successfully pulled model: {model_name}")

            from .model_cache import get_shared_model_cache
            get_shared_model_cache().clear()
            self._storage_cache = None

            return True

        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            raise RuntimeError(f"Failed to pull model: {e}")

    async def cleanup_unused_models(
        self,
        threshold_days: int = 30,
        min_free_space_pct: int = 10,
        dry_run: bool = False
    ) -> List[str]:
        """Clean up unused Ollama models to free space."""
        from datetime import timedelta

        storage = await self.get_storage_info(use_cache=False)
        free_space_pct = (storage["available_gb"] / storage["total_gb"]) * 100

        if free_space_pct >= min_free_space_pct:
            logger.info(f"Cleanup not needed. Free space: {free_space_pct:.1f}%")
            return []

        logger.info(f"Starting cleanup. Free space: {free_space_pct:.1f}%")

        protected_models = set()
        for provider in self.providers:
            if provider.get("vendor") == "ollama":
                protected_models.add(provider["model"])

        if self.mandate_config:
            defaults = self.mandate_config.get("defaults", {})
            if "preferred" in defaults:
                protected_models.add(defaults["preferred"])
            mandates = self.mandate_config.get("mandates", {})
            for model in mandates.values():
                if "ollama" in model or ":" in model:
                    protected_models.add(model)

        logger.info(f"Protected models: {protected_models}")

        models_to_delete = []
        threshold_date = datetime.now(timezone.utc) - timedelta(days=threshold_days)
        recent_date = datetime.now(timezone.utc) - timedelta(days=7)

        for model in storage["models"]:
            model_id = model["id"]

            if model_id in protected_models:
                logger.info(f"Skipping protected model: {model_id}")
                continue

            if model["last_used"]:
                last_used = datetime.fromisoformat(model["last_used"])
                # Ensure timezone-aware comparison
                if last_used.tzinfo is None:
                    last_used = last_used.replace(tzinfo=timezone.utc)

                if last_used > recent_date:
                    logger.info(f"Skipping recent model: {model_id}")
                    continue

                if last_used < threshold_date:
                    models_to_delete.append(model_id)
            else:
                logger.info(f"Skipping model with no usage record: {model_id}")

        if not models_to_delete:
            logger.info("No models eligible for cleanup")
            return []

        logger.info(f"Models to delete: {models_to_delete}")

        if dry_run:
            logger.info(f"DRY RUN: Would delete {len(models_to_delete)} models")
            return models_to_delete

        ollama_provider = None
        for provider in self.providers:
            if provider.get("vendor") == "ollama":
                ollama_provider = provider
                break

        if not ollama_provider:
            logger.warning("Ollama provider not found. Cannot delete models.")
            return []

        client = ollama_provider["client"]
        deleted_models = []

        await self._ensure_db_initialized()

        for model_id in models_to_delete:
            try:
                logger.info(f"Deleting model: {model_id}")
                await client.delete(model=model_id)
                deleted_models.append(model_id)
                logger.info(f"Deleted model: {model_id}")

                if self._usage_db:
                    await self._usage_db.execute(
                        "DELETE FROM model_usage WHERE model_id = ?",
                        (model_id,)
                    )

            except Exception as e:
                logger.error(f"Failed to delete model {model_id}: {e}")

        from .model_cache import get_shared_model_cache
        get_shared_model_cache().clear()
        self._storage_cache = None

        logger.info(f"Cleanup complete. Deleted {len(deleted_models)} models")
        return deleted_models

    async def _check_and_cleanup_if_needed(self):
        """Check storage and automatically cleanup if space is low."""
        try:
            storage = await self.get_storage_info(use_cache=False)
            free_space_pct = (storage["available_gb"] / storage["total_gb"]) * 100

            if free_space_pct < 10:
                logger.warning(f"Low disk space detected: {free_space_pct:.1f}%")
                deleted = await self.cleanup_unused_models(
                    threshold_days=30,
                    min_free_space_pct=10,
                    dry_run=False
                )
                if deleted:
                    logger.info(f"Auto-cleanup freed space by deleting: {deleted}")
        except Exception as e:
            logger.error(f"Auto-cleanup failed: {e}")

    async def close_usage_db(self):
        """Close the usage tracking database connection."""
        if self._usage_db:
            try:
                await self._usage_db.close()
                logger.info("Usage tracking database closed.")
            except Exception as e:
                logger.debug(f"Error closing usage database: {e}")
            finally:
                self._usage_db = None
                self._db_initialized = False
