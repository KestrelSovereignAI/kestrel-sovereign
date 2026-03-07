"""File serving endpoint for Kestrel storage.

Serves stored files (avatars, documents, etc.) via content-addressable hashes.
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response
import logging
from endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["files"])


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

        if not storage or not hasattr(storage, 'files'):
            raise HTTPException(status_code=503, detail="Storage not available.")

        file_store = storage.files

        # Retrieve the file content
        content = await file_store.retrieve_file(content_hash)
        if not content:
            raise HTTPException(status_code=404, detail="File not found")

        # Get metadata for MIME type
        metadata = await file_store.get_file_metadata(content_hash)
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

        # Just check metadata existence (faster than retrieving content)
        metadata = await file_store.get_file_metadata(content_hash)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        mime_type = metadata.get("mime_type", "application/octet-stream")

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
