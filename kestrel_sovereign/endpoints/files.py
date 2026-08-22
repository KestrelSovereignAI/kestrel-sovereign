"""File serving endpoint for Kestrel storage.

Serves stored files (avatars, documents, etc.) via content-addressable hashes.
"""

import asyncio
import logging
import re
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response
from kestrel_sovereign.endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["files"])

# Channel types are short lowercase identifiers; the regex doubles as a
# path-traversal guard for the artifact filename built from it.
_CHANNEL_TYPE_RE = re.compile(r"^[a-z0-9_]{1,32}$")


def _require_bound_file_scope(agent, storage) -> str:
    """Return the request agent id only when storage is bound to that tenant."""
    agent_id = getattr(agent, "agent_id", None)
    storage_agent_id = getattr(storage, "agent_id", None)
    if not agent_id or storage_agent_id != agent_id:
        raise HTTPException(
            status_code=503,
            detail="Agent-scoped file storage is not available.",
        )
    return agent_id


def channel_artifact_path(agent, channel_type: str, name: str) -> Path | None:
    """Resolve a channel linking artifact path under the agent's data dir.

    Isolated channel features (e.g. WhatsApp) push their pairing QR PNG to the
    host, which persists it here so the chat UI can render it over http (the
    sanitizer blocks ``data:`` image URIs). Mirrors the canonical isolated
    feature runtime scope: storage-backed standalone agents use the DB parent;
    hosted agents use their explicitly validated runtime namespace.
    """
    from kestrel_sovereign.features.isolated_runtime import (
        IsolatedRuntimeNamespaceError,
        IsolatedRuntimePreparationError,
        resolve_agent_runtime_dir,
    )

    # The proxy's legacy no-storage fallback exists solely to preserve its
    # standalone provisioning behavior. This HTTP endpoint historically had no
    # artifact location without a filesystem agent, so expose one only for an
    # explicitly scoped hosted runtime.
    storage_path = getattr(agent, "storage_path", None)
    runtime_root = getattr(agent, "isolated_runtime_root", None)
    runtime_namespace = getattr(agent, "isolated_runtime_namespace", None)
    has_storage_root = isinstance(storage_path, (str, Path)) and bool(storage_path)
    has_explicit_runtime_scope = isinstance(runtime_root, (str, Path)) or isinstance(
        runtime_namespace, (str, Path)
    )
    if not has_storage_root and not has_explicit_runtime_scope:
        return None

    try:
        base = resolve_agent_runtime_dir(agent)
    except (IsolatedRuntimeNamespaceError, IsolatedRuntimePreparationError, OSError):
        return None
    return base / "channel_link_artifacts" / f"{channel_type}_{name}"


@router.get("/api/agent/channels/{channel_type}/link-qr.png")
async def serve_channel_link_qr(channel_type: str, request: Request):
    """Serve the current pairing QR PNG for an isolated channel feature.

    The image is fetched by the persisted ``channel_link`` card
    (``channelLinkPartRenderer`` in chat.js, #2081), which resolves the current
    QR state live on render/refresh; an http(s) ``<img>`` survives the DOMPurify
    sanitizer where an inline ``data:`` URI would be stripped. Served
    ``no-store`` because the QR rotates (~20s) and is single-use.
    """
    if not _CHANNEL_TYPE_RE.match(channel_type):
        raise HTTPException(status_code=400, detail="Invalid channel type")
    agent = get_agent(request)
    path = channel_artifact_path(agent, channel_type, "link_qr.png")
    if path is None:
        raise HTTPException(status_code=404, detail="No pairing QR available")
    from kestrel_sovereign.features.isolated_runtime import (
        IsolatedRuntimeNamespaceError,
        IsolatedRuntimePreparationError,
        read_private_artifact,
    )

    try:
        # Keep both the no-follow open and bounded descriptor read off the
        # event loop. Returning bytes is deliberate: FileResponse would reopen
        # the tenant-controlled pathname after validation and restore the race.
        content = await asyncio.to_thread(read_private_artifact, path)
    except (IsolatedRuntimeNamespaceError, IsolatedRuntimePreparationError, OSError):
        logger.warning(
            "Refusing unsafe or unavailable %s channel QR artifact",
            channel_type,
            exc_info=True,
        )
        content = None
    if content is None:
        raise HTTPException(status_code=404, detail="No pairing QR available")
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.get("/api/files/{content_hash}")
async def serve_file(content_hash: str, request: Request):
    """
    Serve stored files by content hash.

    Returns the file content with appropriate MIME type and caching headers.
    Supports avatars, documents, and any other stored files.

    Args:
        content_hash: SHA256 hash of the file content

    Returns:
        File content with appropriate headers
    """
    try:
        agent = get_agent(request)
        storage = agent.storage

        if not storage or not hasattr(storage, 'retrieve_file'):
            raise HTTPException(status_code=503, detail="Storage not available.")
        _require_bound_file_scope(agent, storage)

        # Retrieve through the privacy wrapper so ISOLATED-mode files buffered
        # in the session store (#1662 attachments) are served too — going
        # straight to storage.files would 404 for exactly those.
        content = await storage.retrieve_file(content_hash)
        if not content:
            raise HTTPException(status_code=404, detail="File not found")

        # Metadata (MIME) lives on the persistent store; session-buffered files
        # have none, so fall back to a safe default.
        file_store = getattr(storage, 'files', None)
        metadata = await file_store.get_file_metadata(content_hash) if file_store else None
        mime_type = "application/octet-stream"
        if metadata:
            mime_type = metadata.get("mime_type", "image/jpeg")

        # Return with long cache headers (content-addressable = immutable)
        return Response(
            content=content,
            media_type=mime_type,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Content-Hash": content_hash,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {content_hash}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving file.")


@router.head("/api/files/{content_hash}")
async def check_file(content_hash: str, request: Request):
    """
    Check if a file exists by content hash.

    Returns 200 with headers if file exists, 404 if not.
    Useful for cache validation and existence checks.
    """
    try:
        agent = get_agent(request)
        storage = agent.storage

        if not storage or not hasattr(storage, 'files'):
            raise HTTPException(status_code=503, detail="Storage not available.")
        _require_bound_file_scope(agent, storage)

        file_store = storage.files

        # Check file existence first. Metadata is optional, so a file can exist
        # even when no metadata row payload is present.
        exists = await file_store.file_exists(content_hash)
        if not exists:
            raise HTTPException(status_code=404, detail="File not found")

        metadata = await file_store.get_file_metadata(content_hash)
        mime_type = metadata.get("mime_type", "application/octet-stream") if metadata else "application/octet-stream"

        return Response(
            content=b"",
            media_type=mime_type,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Content-Hash": content_hash,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking file {content_hash}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error checking file.")
