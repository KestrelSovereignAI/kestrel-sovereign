"""File serving endpoint for Kestrel storage.

Serves stored files (avatars, documents, etc.) via content-addressable hashes.
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response, FileResponse
import logging
import re
from pathlib import Path
from kestrel_sovereign.endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["files"])

# Channel types are short lowercase identifiers; the regex doubles as a
# path-traversal guard for the artifact filename built from it.
_CHANNEL_TYPE_RE = re.compile(r"^[a-z0-9_]{1,32}$")


def channel_artifact_path(agent, channel_type: str, name: str) -> Path | None:
    """Resolve a channel linking artifact path under the agent's data dir.

    Isolated channel features (e.g. WhatsApp) push their pairing QR PNG to the
    host, which persists it here so the chat UI can render it over http (the
    sanitizer blocks ``data:`` image URIs). Mirrors ``_agent_data_dir`` in
    ``features/isolated_runtime`` (agent data dir = storage DB's parent).
    """
    storage_path = getattr(agent, "storage_path", None)
    if not storage_path:
        return None
    base = Path(storage_path).expanduser().resolve().parent
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
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="No pairing QR available")
    return FileResponse(
        str(path),
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
