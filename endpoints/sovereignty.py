"""Sovereignty export/import and file browser endpoints."""
import asyncio
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pathlib import Path
import json
import re
import time
import os
import logging

from kestrel_sovereign.kestrel_config.constants import MAX_SOVEREIGNTY_PREVIEW_SIZE
from endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["sovereignty"])

STORAGE_CACHE_DIR = Path(__file__).parent.parent / "storage_cache"

# Allowlist of valid storage tiers for sovereignty export.
# Legacy names (ipfs, filecoin) map to CLOUD_HOT / CLOUD_COLD via SovereigntyFeature.
# "storacha" and "cloud_hot" are the modern preferred values.
ALLOWED_TIERS = {"local", "ipfs", "filecoin", "storacha", "cloud_hot", "cloud_cold"}

# CID format: alphanumeric characters only (covers CIDv0 Qm... and CIDv1 bafy...)
CID_PATTERN = re.compile(r'^[a-zA-Z0-9]+$')


def _read_metadata_file(meta_path: Path):
    with open(meta_path, 'r') as f:
        content = f.read().strip()
    if content.startswith('{'):
        return json.loads(content)
    return {"raw": content}


def _list_storage_cache_files(cache_dir: Path):
    files = []
    total_size = 0

    for filepath in cache_dir.iterdir():
        if filepath.is_file():
            stat = filepath.stat()
            size = stat.st_size
            total_size += size

            ext = filepath.suffix.lower()
            file_type = "cache" if ext == ".cache" else "meta" if ext == ".meta" else "other"

            metadata = None
            if file_type == "cache":
                meta_path = filepath.with_suffix(".meta")
                if meta_path.exists():
                    try:
                        metadata = _read_metadata_file(meta_path)
                    except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
                        logger.warning(f"Could not read metadata file {meta_path}: {e}")
                        metadata = {"error": "Could not read metadata"}

            files.append({
                "name": filepath.name,
                "size": size,
                "type": file_type,
                "modified": stat.st_mtime,
                "modified_iso": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(stat.st_mtime)),
                "hash": filepath.stem,
                "has_meta": (file_type == "cache" and filepath.with_suffix(".meta").exists()),
                "metadata": metadata,
            })

    files.sort(key=lambda x: x["modified"], reverse=True)
    return files, total_size


def _read_preview_bytes(real_path: Path, max_size: int):
    stat = real_path.stat()
    size = stat.st_size
    with open(real_path, 'rb') as f:
        content_bytes = f.read(max_size)
    return size, content_bytes


@router.get("/storage/stats")
async def get_storage_stats(request: Request):
    """Get storage statistics and breakdown."""
    try:
        agent = get_agent(request)
        storage = agent.storage
        db_path = storage.db_path

        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0

        # Use async database query with agent_id filter
        agent_id = getattr(storage, 'agent_id', '') or getattr(storage._storage, 'agent_id', '')
        conv_row = await storage.db.fetchone(
            "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?",
            (agent_id,)
        )
        conversation_count = conv_row[0] if conv_row else 0

        # Use async database query for graph nodes
        node_rows = await storage.db.fetchall(
            "SELECT node_type, COUNT(*) FROM graph_nodes GROUP BY node_type"
        )
        node_counts = dict(node_rows) if node_rows else {}

        try:
            file_row = await storage.db.fetchone(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0) FROM files"
            )
            file_count = file_row[0] if file_row else 0
            file_size = file_row[1] if file_row else 0
        except Exception:
            file_count = 0
            file_size = 0

        # Use async storage methods
        exports = await storage.get_nodes_by_type("sovereignty_receipt")
        backups = await storage.get_nodes_by_type("backup_artifact")

        return {
            "database": {"path": db_path, "size_bytes": db_size},
            "conversations": {"count": conversation_count},
            "graph_nodes": node_counts,
            "files": {"count": file_count, "size_bytes": file_size},
            "sovereignty_exports": len(exports),
            "backups": len(backups),
        }
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving storage stats.")


@router.get("/sovereignty/exports")
async def list_sovereignty_exports(request: Request):
    """List all sovereignty export receipts."""
    try:
        agent = get_agent(request)
        storage = agent.storage
        receipts = await storage.get_nodes_by_type("sovereignty_receipt")
        backups = await storage.get_nodes_by_type("backup_artifact")

        return {
            "exports": [
                {
                    "node_id": r.node_id,
                    "cid": r.properties.get("ipfs_cid") or r.properties.get("cid"),
                    "storage_tier": r.properties.get("storage_tier"),
                    "created_at": r.properties.get("created_at") or r.properties.get("timestamp"),
                    "encrypted": r.properties.get("encrypted", True),
                    "properties": r.properties,
                }
                for r in receipts
            ],
            "backups": [
                {
                    "node_id": b.node_id,
                    "cid": b.properties.get("ipfs_cid"),
                    "storage_tier": b.properties.get("storage_tier"),
                    "created_at": b.properties.get("created_at"),
                    "encrypted": b.properties.get("encrypted", False),
                }
                for b in backups
            ],
        }
    except Exception as e:
        logger.error(f"Error listing sovereignty exports: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving sovereignty exports.")


@router.post("/sovereignty/export")
async def trigger_sovereignty_export(request: Request):
    """Trigger a sovereignty export with the specified tier and encryption settings."""
    try:
        data = await request.json()
        tier = data.get("tier", "ipfs")
        encrypt = data.get("encrypt", True)

        if tier not in ALLOWED_TIERS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier '{tier}'. Must be one of: {', '.join(sorted(ALLOWED_TIERS))}",
            )

        agent = get_agent(request)
        sovereignty = getattr(agent, 'features', {}).get("SovereigntyFeature")
        if not sovereignty:
            raise HTTPException(status_code=500, detail="Sovereignty feature not available.")

        result = await sovereignty.export_sovereignty(storage_tier=tier, encrypt=encrypt)
        return {"success": True, "message": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering sovereignty export: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error triggering export.")


@router.post("/sovereignty/import")
async def trigger_sovereignty_import(request: Request):
    """Trigger a sovereignty import via the agent's !import-sovereignty command."""
    try:
        data = await request.json()
        cid = data.get("cid")
        if not cid:
            raise HTTPException(status_code=400, detail="CID is required.")
        if not CID_PATTERN.match(cid):
            raise HTTPException(
                status_code=400,
                detail="Invalid CID format. CID must contain only alphanumeric characters.",
            )

        agent = get_agent(request)
        cmd = f"!import-sovereignty {cid}"
        result = await agent.process_input(cmd)

        return {"success": True, "message": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering sovereignty import: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error triggering import.")


@router.get("/sovereignty/files")
async def list_sovereignty_files(request: Request):
    """List files in storage_cache/ directory for the local file browser."""
    if not STORAGE_CACHE_DIR.exists():
        return {"files": [], "total_size": 0, "file_count": 0}

    try:
        files, total_size = await asyncio.to_thread(_list_storage_cache_files, STORAGE_CACHE_DIR)

        return {
            "files": files,
            "total_size": total_size,
            "file_count": len(files),
            "cache_dir": str(STORAGE_CACHE_DIR),
        }
    except Exception as e:
        logger.error(f"Error listing sovereignty files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing files.")


@router.get("/sovereignty/files/{filename}")
async def download_sovereignty_file(request: Request, filename: str):
    """Download a specific file from storage_cache/."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    filepath = STORAGE_CACHE_DIR / filename

    try:
        # Resolve once with strict=True (raises OSError if path doesn't exist)
        real_path = filepath.resolve(strict=True)
        cache_dir = STORAGE_CACHE_DIR.resolve()

        if not str(real_path).startswith(str(cache_dir) + os.sep):
            raise HTTPException(status_code=400, detail="Invalid file path.")

        if not real_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file.")

        # Use the resolved real path for FileResponse to prevent symlink swap
        return FileResponse(path=str(real_path), filename=filename, media_type="application/octet-stream")
    except HTTPException:
        raise
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="File not found.")


@router.get("/sovereignty/files/{filename}/preview")
async def preview_sovereignty_file(
    request: Request,
    filename: str,
    max_size: int = Query(default=MAX_SOVEREIGNTY_PREVIEW_SIZE, gt=0, le=MAX_SOVEREIGNTY_PREVIEW_SIZE),
):
    """Get a preview of a file's content."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    filepath = STORAGE_CACHE_DIR / filename

    try:
        # Resolve once with strict=True (raises OSError if path doesn't exist)
        real_path = filepath.resolve(strict=True)
        cache_dir = STORAGE_CACHE_DIR.resolve()

        if not str(real_path).startswith(str(cache_dir) + os.sep):
            raise HTTPException(status_code=400, detail="Invalid file path.")

        if not real_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file.")
    except HTTPException:
        raise
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="File not found.")

    # Use the resolved real path for all subsequent operations
    try:
        size, content_bytes = await asyncio.to_thread(_read_preview_bytes, real_path, max_size)

        try:
            content = content_bytes.decode('utf-8')
            is_text = True
            try:
                json.loads(content)
                content_type = "json"
            except json.JSONDecodeError:
                content_type = "text"
        except UnicodeDecodeError:
            content = content_bytes.hex()[:1000]
            is_text = False
            content_type = "binary"

        return {
            "filename": filename,
            "size": size,
            "truncated": size > max_size,
            "is_text": is_text,
            "content_type": content_type,
            "content": content,
            "preview_size": len(content_bytes),
        }
    except Exception as e:
        logger.error(f"Error previewing file {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error reading file.")
