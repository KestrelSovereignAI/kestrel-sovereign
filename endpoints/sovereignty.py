"""Sovereignty export/import and file browser endpoints."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path
import json
import time
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["sovereignty"])

STORAGE_CACHE_DIR = Path(__file__).parent.parent / "storage_cache"


@router.get("/storage/stats")
async def get_storage_stats(request: Request):
    """Get storage statistics and breakdown."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
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
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
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
    """Trigger a sovereignty export via the agent's !export-sovereignty command."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        data = await request.json()
        tier = data.get("tier", "ipfs")
        encrypt = data.get("encrypt", True)

        agent = request.app.state.agent
        cmd = f"!export-sovereignty --tier={tier}"
        if not encrypt:
            cmd += " --no-encrypt"

        result = await agent.process_input(cmd)
        return {"success": True, "message": result}
    except Exception as e:
        logger.error(f"Error triggering sovereignty export: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error triggering export.")


@router.post("/sovereignty/import")
async def trigger_sovereignty_import(request: Request):
    """Trigger a sovereignty import via the agent's !import-sovereignty command."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        data = await request.json()
        cid = data.get("cid")
        if not cid:
            raise HTTPException(status_code=400, detail="CID is required.")

        agent = request.app.state.agent
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
        files = []
        total_size = 0

        for filepath in STORAGE_CACHE_DIR.iterdir():
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
                            with open(meta_path, 'r') as f:
                                content = f.read().strip()
                                if content.startswith('{'):
                                    metadata = json.loads(content)
                                else:
                                    metadata = {"raw": content}
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

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    if not filepath.is_file():
        raise HTTPException(status_code=400, detail="Not a file.")

    try:
        filepath.resolve().relative_to(STORAGE_CACHE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path.")

    return FileResponse(path=filepath, filename=filename, media_type="application/octet-stream")


@router.get("/sovereignty/files/{filename}/preview")
async def preview_sovereignty_file(request: Request, filename: str, max_size: int = 10000):
    """Get a preview of a file's content."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    filepath = STORAGE_CACHE_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    if not filepath.is_file():
        raise HTTPException(status_code=400, detail="Not a file.")

    try:
        filepath.resolve().relative_to(STORAGE_CACHE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path.")

    stat = filepath.stat()
    size = stat.st_size

    try:
        with open(filepath, 'rb') as f:
            content_bytes = f.read(max_size)

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
