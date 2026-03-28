#!/usr/bin/env python3
"""
Embedding Worker - Background embedding pre-computation.

Continuously scans Frinz PostgreSQL for content without embeddings,
then computes and stores them using Ollama nomic-embed-text (~0.3GB).

Designed to run alongside Kimi with negligible memory footprint.

Usage:
    python scripts/embedding_worker.py
    python scripts/embedding_worker.py --batch-size 50
    python scripts/embedding_worker.py --once  # Single pass, then exit

Environment Variables:
    DATABASE_URL - Frinz PostgreSQL URL
    OLLAMA_HOST - Ollama server URL (default: http://localhost:11434)
"""

import argparse
import asyncio
import json
import logging
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")
frinz_env = Path("/Volumes/data2/projects/frinz/.env")
if frinz_env.exists():
    load_dotenv(frinz_env, override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("embedding_worker")


def check_memory_pressure() -> str:
    """Check macOS memory pressure."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            level = int(result.stdout.strip())
            if level >= 4:
                return "red"
            elif level >= 2:
                return "yellow"
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return "green"


def ensure_ollama_model():
    """Ensure nomic-embed-text is pulled in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=30,
        )
        if "nomic-embed-text" in result.stdout:
            return True

        logger.info("Pulling nomic-embed-text model...")
        subprocess.run(
            ["ollama", "pull", "nomic-embed-text"],
            timeout=300,
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Ollama not available: {e}")
        return False


def embedding_to_bytes(embedding: list[float]) -> bytes:
    """Convert float list to compact binary for pgvector."""
    return struct.pack(f"{len(embedding)}f", *embedding)


async def get_db_pool():
    """Get asyncpg connection pool to Frinz database."""
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg not installed")
        sys.exit(1)

    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)

    return await asyncpg.create_pool(database_url, min_size=1, max_size=3)


async def ensure_embedding_column(pool):
    """Ensure embedding column exists on relevant tables."""
    async with pool.acquire() as conn:
        # Add embedding column to messages if not exists
        await conn.execute("""
            ALTER TABLE messages
            ADD COLUMN IF NOT EXISTS embedding vector(768)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_embedding
            ON messages USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)

        # Add embedding column to document_chunks if not exists
        await conn.execute("""
            ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS embedding_vec vector(768)
        """)


async def get_unembedded_messages(pool, batch_size: int) -> list[dict]:
    """Get messages that don't have embeddings yet."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, role
            FROM messages
            WHERE embedding IS NULL
            AND role IN ('user', 'assistant')
            AND length(content) > 10
            ORDER BY created_at DESC
            LIMIT $1
        """, batch_size)
        return [dict(r) for r in rows]


async def get_unembedded_chunks(pool, batch_size: int) -> list[dict]:
    """Get document chunks without embeddings."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT chunk_id, content
            FROM document_chunks
            WHERE embedding_vec IS NULL
            AND length(content) > 10
            LIMIT $1
        """, batch_size)
        return [dict(r) for r in rows]


async def store_message_embeddings(pool, updates: list[tuple]):
    """Store embeddings for messages. updates = [(id, embedding_str), ...]"""
    async with pool.acquire() as conn:
        await conn.executemany("""
            UPDATE messages SET embedding = $2::vector WHERE id = $1
        """, updates)


async def store_chunk_embeddings(pool, updates: list[tuple]):
    """Store embeddings for document chunks."""
    async with pool.acquire() as conn:
        await conn.executemany("""
            UPDATE document_chunks SET embedding_vec = $2::vector WHERE chunk_id = $1
        """, updates)


async def embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """Embed a batch of texts using Ollama."""
    from kestrel_sovereign.llm.embedding_service import EmbeddingService

    service = EmbeddingService()
    try:
        return await service.aembed_batch(texts)
    except Exception as e:
        logger.error(f"Embedding batch failed: {e}")
        return [None] * len(texts)


async def process_messages(pool, batch_size: int) -> int:
    """Process unembedded messages. Returns count processed."""
    messages = await get_unembedded_messages(pool, batch_size)
    if not messages:
        return 0

    texts = [m["content"] for m in messages]
    embeddings = await embed_batch(texts)

    updates = []
    for msg, emb in zip(messages, embeddings):
        if emb is not None:
            # Convert to pgvector format string: [0.1, 0.2, ...]
            emb_str = "[" + ",".join(str(v) for v in emb) + "]"
            updates.append((msg["id"], emb_str))

    if updates:
        await store_message_embeddings(pool, updates)

    return len(updates)


async def process_chunks(pool, batch_size: int) -> int:
    """Process unembedded document chunks. Returns count processed."""
    chunks = await get_unembedded_chunks(pool, batch_size)
    if not chunks:
        return 0

    texts = [c["content"] for c in chunks]
    embeddings = await embed_batch(texts)

    updates = []
    for chunk, emb in zip(chunks, embeddings):
        if emb is not None:
            emb_str = "[" + ",".join(str(v) for v in emb) + "]"
            updates.append((chunk["chunk_id"], emb_str))

    if updates:
        await store_chunk_embeddings(pool, updates)

    return len(updates)


async def run_worker(batch_size: int = 50, once: bool = False):
    """Main worker loop."""
    if not ensure_ollama_model():
        logger.error("Cannot start without Ollama and nomic-embed-text")
        return

    pool = await get_db_pool()

    try:
        await ensure_embedding_column(pool)
        logger.info(f"Embedding worker started (batch_size={batch_size})")

        total_embedded = 0
        pass_count = 0

        while True:
            # Check memory pressure
            if check_memory_pressure() == "red":
                logger.warning("Memory pressure RED — pausing embeddings")
                await asyncio.sleep(60)
                continue

            pass_count += 1
            embedded_this_pass = 0

            # Process messages
            msg_count = await process_messages(pool, batch_size)
            embedded_this_pass += msg_count

            # Process document chunks
            chunk_count = await process_chunks(pool, batch_size)
            embedded_this_pass += chunk_count

            total_embedded += embedded_this_pass

            if embedded_this_pass > 0:
                logger.info(
                    f"Pass {pass_count}: embedded {msg_count} messages + {chunk_count} chunks "
                    f"(total: {total_embedded})"
                )
            else:
                logger.debug(f"Pass {pass_count}: nothing to embed")

            if once:
                logger.info(f"Single pass complete: {total_embedded} embedded total")
                break

            # Sleep between passes — longer if nothing to do
            if embedded_this_pass == 0:
                await asyncio.sleep(300)  # 5 minutes if idle
            else:
                await asyncio.sleep(10)  # Quick turnaround if there's work

    except asyncio.CancelledError:
        logger.info("Embedding worker cancelled")
    except Exception as e:
        logger.error(f"Embedding worker error: {e}", exc_info=True)
    finally:
        await pool.close()
        logger.info(f"Embedding worker stopped (total: {total_embedded} embedded)")


def main():
    parser = argparse.ArgumentParser(description="Background Embedding Worker")
    parser.add_argument("--batch-size", type=int, default=50, help="Texts per batch")
    parser.add_argument("--once", action="store_true", help="Single pass then exit")
    args = parser.parse_args()

    asyncio.run(run_worker(batch_size=args.batch_size, once=args.once))


if __name__ == "__main__":
    main()
