"""Conversation and session endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from datetime import datetime
import json
import logging

from kestrel_sovereign.rate_limit import limiter

from kestrel_sovereign.storage.session_grouping import (
    coalesce_sessions_by_session_id,
    group_messages_into_sessions,
)
from kestrel_sovereign.security.encryption import get_fernet, get_agent_fernet, decrypt_string_fernet as decrypt_string
from kestrel_sovereign.security.demo_isolation import enforce_destructive_op
from kestrel_sovereign.agent.context_builder import extract_raw_user_content
from kestrel_sovereign.endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["conversations"])

SESSION_GAP_OVERSAMPLE = 20
MAX_CONVERSATION_LIST_ROWS = 1000


def _public_metadata(meta):
    """Return metadata safe for user-facing conversation history payloads."""
    if not isinstance(meta, dict):
        return {}
    public = dict(meta)
    public.pop("pre_tool_reasoning", None)
    public.pop("key_version", None)
    return public


@router.get("/sessions")
async def list_sessions(request: Request, limit: int = Query(50, ge=1, le=500)):
    """List conversation sessions with summary info."""
    try:
        agent = get_agent(request)
        storage = agent.storage
        history = await storage.get_conversation_history(limit)
        for message in history:
            if "metadata" in message:
                public_meta = _public_metadata(message.get("metadata"))
                if public_meta:
                    message["metadata"] = public_meta
                else:
                    message.pop("metadata", None)
        total_messages = len(history)
        user_messages = sum(1 for m in history if m.get("role") == "user")
        agent_messages = sum(1 for m in history if m.get("role") == "assistant")

        return {
            "messages": history,
            "total": total_messages,
            "user_messages": user_messages,
            "agent_messages": agent_messages,
        }
    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving sessions.")


@router.get("/conversations")
async def list_conversations(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    decrypt: bool = True,
    view: str = Query("active"),
    q: str = Query(None, min_length=1, max_length=256),
):
    """List conversation sessions grouped by date/time.

    ``view=active`` (default) lists live, non-archived sessions;
    ``view=archived`` lists archived sessions (#2149). Any other value
    falls back to ``active``.

    ``q`` switches the endpoint into full-text search: only sessions whose
    message content (decrypted server-side, sent-form unwrapped) or
    user-assigned title matches is returned, each decorated with
    ``match_count`` / ``match_role`` / ``match_snippet``. Search goes through
    the privacy-wrapped storage, so EPHEMERAL mode returns nothing and
    ISOLATED mode searches only the in-memory session buffer.
    """
    try:
        if view not in ("active", "archived"):
            view = "active"
        agent = get_agent(request)
        storage = agent.storage

        # Use privacy-aware agent_id accessor
        agent_id = getattr(storage, 'agent_id', '')

        # Check if encryption is enabled via the wrapper's safe accessor
        encrypted_at_rest = getattr(storage, 'encryption_enabled', False)

        def _decorate_preview(session):
            # The shared grouper hands back the raw first-user-message content
            # plus its metadata dict; the UI layer decrypts + unwraps it here.
            preview_content = session.pop("preview_content", None)
            meta = session.pop("preview_metadata", None) or {}
            session.setdefault("messages", [])
            if preview_content is None:
                session["preview"] = ""
                return

            is_encrypted = False
            decryption_failed = False
            preview_is_sent_form = bool(meta.get('sent_form'))
            if meta.get('enc'):
                is_encrypted = True
                if decrypt:
                    fernet = get_agent_fernet(agent_id) if agent_id else get_fernet()
                    if fernet:
                        try:
                            preview_content = decrypt_string(preview_content, meta, fernet)
                        except Exception as decrypt_err:
                            logger.warning(f"Failed to decrypt preview: {decrypt_err}")
                            decryption_failed = True

            # Unwrap sent-form so the UI shows raw user text, not the
            # <retrieved_context>.../<user_input>... wrappers that were stored
            # for byte-stable history replay.
            if preview_is_sent_form and not decryption_failed:
                preview_content = extract_raw_user_content(preview_content)
            session["preview"] = preview_content[:100] + ("..." if len(preview_content) > 100 else "")
            session["preview_encrypted"] = is_encrypted and (not decrypt or decryption_failed)

        # Full-text search path: the privacy-wrapped storage decrypts,
        # unwraps and groups server-side; sessions come back already decorated
        # with match_count/match_role/match_snippet and any user-assigned name.
        search_query = q.strip() if q else ""
        if search_query:
            search = getattr(storage, "search_conversations", None)
            sessions = []
            if search is not None:
                sessions = await search(agent_id, search_query, limit=limit, view=view)
            # Matching necessarily decrypts server-side (SQL cannot see
            # encrypted content), but decrypt=false callers asked for no
            # plaintext in the RESPONSE — redact the decrypted snippet and
            # preview rather than leak readable excerpts (codex P2).
            redact = (not decrypt) and encrypted_at_rest
            for session in sessions:
                if redact:
                    session.pop("preview_content", None)
                    session.pop("preview_metadata", None)
                    session.setdefault("messages", [])
                    session["preview"] = ""
                    session["preview_encrypted"] = True
                    session["match_snippet"] = None
                else:
                    _decorate_preview(session)
            return {
                "conversations": sessions,
                "total": len(sessions),
                "encrypted_at_rest": encrypted_at_rest,
                "query": search_query,
            }

        # Python still groups raw messages into sessions by 30-minute gaps, so
        # fetch more rows than the requested session count while keeping a hard
        # SQL-side budget for large histories.
        row_limit = min(limit * SESSION_GAP_OVERSAMPLE, MAX_CONVERSATION_LIST_ROWS)

        # Use privacy-aware query method instead of direct storage.db access.
        rows = await storage.query_conversations(agent_id, limit=row_limit, view=view)

        if not rows:
            return {"conversations": [], "total": 0, "encrypted_at_rest": encrypted_at_rest}

        # Normalize raw rows (newest-first from SQL) into oldest-first dicts and
        # run them through the shared session-boundary algorithm (#2019) — the
        # same one the agent's list_conversations tool uses, so the UI and the
        # agent never disagree on where a session begins. Metadata is parsed to
        # a dict here but its enc/sent_form flags are preserved for preview
        # decoration above.
        normalized = []
        for row in reversed(rows):
            msg_id, role, content, metadata_json, created_at = row[0], row[1], row[2], row[3], row[4]
            meta = {}
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for message {msg_id}: {e}")
                    meta = {}
            normalized.append({
                "id": msg_id,
                "role": role,
                "content": content,
                "metadata": meta,
                "created_at": created_at,
            })

        # Coalesce same-UUID clusters (resumed-past-the-gap conversations) so a
        # listed session_id is a unique delete target, matching the lifecycle
        # tools and avoiding a delete-one-destroys-both collision (#2019).
        # keep_empty_markers=True so a just-started conversation (a session
        # marker with no messages yet) is list-visible immediately (#2222) —
        # the UI optimistically prepends a tile for it on New, and this
        # reconciling list must include it or the tile vanishes on refresh.
        grouped = coalesce_sessions_by_session_id(
            group_messages_into_sessions(normalized, keep_empty_markers=True)
        )
        # Newest-first by last activity so a resumed conversation ranks by its
        # latest message rather than its first cluster's position (#2019).
        grouped.sort(key=lambda s: s["last_message_at"], reverse=True)
        sessions = grouped[:limit]
        for session in sessions:
            _decorate_preview(session)

        # Decorate with user-assigned display names (#716).  Single bulk
        # read rather than per-row so long conversation lists stay fast.
        names = {}
        get_names = getattr(storage, "get_conversation_names", None)
        if get_names is not None:
            try:
                names = await get_names() or {}
            except Exception as e:
                logger.warning(f"Failed to load conversation names: {e}")
        for session in sessions:
            sid = session.get("session_id")
            if sid and sid in names:
                session["name"] = names[sid]

        return {
            "conversations": sessions,
            "total": len(sessions),
            "encrypted_at_rest": encrypted_at_rest
        }
    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving conversations.")


@router.get("/conversations/{session_id}")
async def get_conversation(request: Request, session_id: str, limit: int = Query(100, ge=1, le=500), decrypt: bool = True):
    """Get messages for a specific conversation session."""
    try:
        agent = get_agent(request)
        storage = agent.storage

        # Use privacy-aware agent_id accessor
        agent_id = getattr(storage, 'agent_id', '')

        # Check if encryption is enabled via the wrapper's safe accessor
        encrypted_at_rest = getattr(storage, 'encryption_enabled', False)

        # Resolve the session's messages via the store's canonical dual-scheme
        # resolver (time-gap clustering AND explicit metadata.session_id / UUID
        # membership). The previous time-gap-only walk here would stop at the
        # first new_session marker and miss continued turns that were filed
        # under a different key — the empty-pane-after-refresh bug (#2012).
        rows = await storage.query_session_rows(session_id, limit=limit)

        # The resolver strips session_marker rows, so a freshly started
        # (marker-only) session legitimately yields zero messages — only 404
        # when the session genuinely doesn't exist (#2012).
        if not rows and not await storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")

        messages = []

        for row in rows:
            msg_id, role, content, metadata_json, created_at = row[0], row[1], row[2], row[3], row[4]
            model = row[5] if len(row) > 5 else None
            provider = row[6] if len(row) > 6 else None

            try:
                if isinstance(created_at, str):
                    timestamp = None
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                        try:
                            timestamp = datetime.strptime(created_at, fmt)
                            break
                        except ValueError:
                            continue
                    if timestamp is None:
                        timestamp = datetime.now()
                elif created_at is not None:
                    timestamp = created_at
                else:
                    timestamp = datetime.now()
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to parse timestamp for message {msg_id} in get_conversation: {e}")
                timestamp = datetime.now()

            meta = None
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for message {msg_id}: {e}")

            display_content = content
            is_encrypted = False
            decryption_failed = False
            if meta and meta.get('enc'):
                is_encrypted = True
                # Decrypt content if requested - use per-agent key
                if decrypt:
                    fernet = get_agent_fernet(agent_id) if agent_id else get_fernet()
                    if fernet:
                        try:
                            display_content = decrypt_string(content, meta, fernet)
                            is_encrypted = False  # Successfully decrypted
                        except Exception as decrypt_err:
                            logger.warning(f"Failed to decrypt message {msg_id}: {decrypt_err}")
                            decryption_failed = True
                            # Keep encrypted content, mark as still encrypted

            # Skip session markers from message list (they're just metadata).
            # The resolver already strips these; this is a defensive backstop.
            if meta and meta.get('type') == 'session_marker':
                continue

            # Unwrap user-turn sent-form so the chat UI shows raw user text,
            # not the <retrieved_context>.../<user_input>... wrappers that
            # history replay requires. Only applies to rows stored with the
            # sent_form contract; legacy raw rows pass through unchanged.
            if (
                role == "user"
                and meta
                and meta.get("sent_form")
                and not decryption_failed
            ):
                display_content = extract_raw_user_content(display_content)

            message = {
                "id": msg_id,
                "role": role,
                "content": display_content,
                "encrypted": is_encrypted or decryption_failed,
                "metadata": _public_metadata(meta),
                "created_at": timestamp.isoformat()
            }
            if role == "assistant":
                message["model"] = model
                message["provider"] = provider
            messages.append(message)

            if len(messages) >= limit:
                break

        return {
            "session_id": session_id,
            "messages": messages,
            "message_count": len(messages),
            "encrypted_at_rest": encrypted_at_rest,
            "showing_decrypted": decrypt
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving conversation.")


@router.post("/conversations/new")
@limiter.limit("30/minute")
async def start_new_conversation(request: Request):
    """Start a new conversation by adding a session marker."""
    try:
        agent = get_agent(request)
        storage = agent.storage

        # Mint the canonical UUID up front and stamp it on the marker, so the
        # client round-trips the SAME id the list endpoint advertises and any
        # rename lands under it (#2012). Without this the response returned the
        # marker row-id, diverging from the list/rename key.
        import uuid as _uuid
        new_session_id = str(_uuid.uuid4())
        await storage.add_conversation(
            role="system",
            content="[New conversation started]",
            metadata={"type": "session_marker", "new_session": True},
            session_id=new_session_id,
        )

        # Surface the created-at for the new marker (privacy-aware accessor).
        agent_id = getattr(storage, 'agent_id', '')
        row = await storage.query_last_conversation_row(agent_id)

        return {
            "success": True,
            "session_id": new_session_id,
            "started_at": row[1] if row else None,
            "message": "New conversation started"
        }
    except Exception as e:
        logger.error(f"Error starting new conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error starting new conversation.")


@router.patch("/conversations/{session_id}")
@limiter.limit("30/minute")
async def rename_conversation(request: Request, session_id: str):
    """Set (or clear) a user-assigned display name for a conversation.

    Request body: ``{"name": "..."}``.  Empty string / null / whitespace-
    only clears the override and reverts the UI to the computed preview.
    Non-empty values are trimmed and capped at 120 chars server-side to
    prevent unbounded growth.

    Returns:
        200 with ``{"success": true, "session_id": ..., "name": final}``
             where ``final`` is the stored value (trimmed) or ``null``
             when cleared.
        400 when the request body is missing / malformed.

    Agent-scoped.  Rejects in ephemeral privacy mode.

    See issue #716.
    """
    try:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Request body must be JSON with a 'name' field.",
            )
        if not isinstance(body, dict) or "name" not in body:
            raise HTTPException(
                status_code=400,
                detail="Request body must include a 'name' field.",
            )
        raw = body["name"]
        if raw is not None and not isinstance(raw, str):
            raise HTTPException(
                status_code=400,
                detail="'name' must be a string or null.",
            )

        agent = get_agent(request)
        storage = agent.storage
        final_name = await storage.set_conversation_name(session_id, raw)

        return {
            "success": True,
            "session_id": session_id,
            "name": final_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error renaming conversation {session_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error renaming conversation."
        )


@router.delete(
    "/conversations/messages/{message_id}",
    dependencies=[Depends(enforce_destructive_op)],
)
@limiter.limit("30/minute")
async def delete_message(request: Request, message_id: int):
    """Delete a single message by ID with agent_id isolation."""
    try:
        agent = get_agent(request)
        storage = agent.storage
        agent_id = getattr(storage, 'agent_id', '')

        # Use privacy-aware delete method instead of direct storage.db access
        deleted = await storage.delete_conversation_message(message_id, agent_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Message not found.")

        return {"success": True, "message_id": message_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message {message_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting message.")


@router.delete(
    "/conversations/{session_id}",
    dependencies=[Depends(enforce_destructive_op)],
)
@limiter.limit("30/minute")
async def delete_conversation(request: Request, session_id: str):
    """Soft-delete a conversation session (#763).

    The session moves to Trash — every message is stamped with
    ``deleted_at`` so the user can restore it from the Trash UI (#765).
    Use ``POST /conversations/{session_id}/purge`` for permanent removal.

    Resolution supports both explicit UUID-based sessions (session_id in
    message metadata) and legacy time-gap-based sessions (session_id is
    the first message's row id — see
    AsyncConversationStore._get_session_messages).  Agent-scoped; cannot
    touch another agent's data.  Rejects ephemeral mode up front — there
    is nothing persistent to delete in that mode.

    Returns:
        200 with {"success": true, "session_id": ..., "deleted_count": N}
             when one or more messages were soft-deleted.
        404 when the session doesn't exist or every row is already in
            trash.

    See issues #715 (original delete behavior), #763 (soft-delete migration).
    """
    try:
        agent = get_agent(request)
        storage = agent.storage
        agent_id = getattr(storage, 'agent_id', '')

        deleted_count = await storage.delete_conversation_session(
            session_id, agent_id
        )
        if deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found or already empty.",
            )

        return {
            "success": True,
            "session_id": session_id,
            "deleted_count": deleted_count,
            "soft_deleted": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error deleting conversation {session_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error deleting conversation."
        )


@router.post("/conversations/{session_id}/restore")
@limiter.limit("30/minute")
async def restore_conversation(request: Request, session_id: str):
    """Restore a soft-deleted conversation from Trash (#763 / #765).

    Clears ``deleted_at`` on every soft-deleted message that belongs to
    the session, making the session visible to normal reads again.

    Returns:
        200 with {"success": true, "session_id": ..., "restored_count": N}
        404 when the session has no soft-deleted rows to restore.
    """
    try:
        agent = get_agent(request)
        storage = agent.storage
        agent_id = getattr(storage, 'agent_id', '')

        restored = await storage.restore_conversation_session(
            session_id, agent_id
        )
        if restored == 0:
            raise HTTPException(
                status_code=404,
                detail="No soft-deleted messages found for this session.",
            )

        return {
            "success": True,
            "session_id": session_id,
            "restored_count": restored,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error restoring conversation {session_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error restoring conversation."
        )


@router.post("/conversations/{session_id}/archive")
@limiter.limit("30/minute")
async def archive_conversation(request: Request, session_id: str):
    """Archive a conversation (#2149).

    Stamps ``archived_at`` on every live message that belongs to the
    session, moving it out of the active list into the archived view.
    Non-destructive — rows stay intact and reversible via ``/unarchive``.

    Returns:
        200 with {"success": true, "session_id": ..., "archived_count": N}
        404 when the session has no live rows to archive.
    """
    try:
        agent = get_agent(request)
        storage = agent.storage
        agent_id = getattr(storage, 'agent_id', '')

        archived = await storage.archive_conversation_session(
            session_id, agent_id
        )
        if archived == 0:
            raise HTTPException(
                status_code=404,
                detail="No messages found to archive for this session.",
            )

        return {
            "success": True,
            "session_id": session_id,
            "archived_count": archived,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error archiving conversation {session_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error archiving conversation."
        )


@router.post("/conversations/{session_id}/unarchive")
@limiter.limit("30/minute")
async def unarchive_conversation(request: Request, session_id: str):
    """Unarchive a conversation (#2149).

    Clears ``archived_at`` on every archived message belonging to the
    session, making it visible in the active list again.

    Returns:
        200 with {"success": true, "session_id": ..., "unarchived_count": N}
        404 when the session has no archived rows to unarchive.
    """
    try:
        agent = get_agent(request)
        storage = agent.storage
        agent_id = getattr(storage, 'agent_id', '')

        unarchived = await storage.unarchive_conversation_session(
            session_id, agent_id
        )
        if unarchived == 0:
            raise HTTPException(
                status_code=404,
                detail="No archived messages found for this session.",
            )

        return {
            "success": True,
            "session_id": session_id,
            "unarchived_count": unarchived,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error unarchiving conversation {session_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error unarchiving conversation."
        )


@router.post("/conversations/{session_id}/purge")
@limiter.limit("10/minute")
async def purge_conversation(request: Request, session_id: str):
    """Permanently delete a conversation (#763).

    Hard SQL DELETE — bypasses Trash. Wipes both currently-live rows and
    rows already in trash. Body: ``{"reason": "..."}`` (optional, default
    ``"user-initiated"``); the reason lands in the audit log (#750).

    Returns:
        200 with {"success": true, "session_id": ..., "purged_count": N}
        404 when nothing matched.
    """
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        reason = (
            body.get("reason") if isinstance(body, dict) else None
        ) or "user-initiated"

        agent = get_agent(request)
        storage = agent.storage
        agent_id = getattr(storage, 'agent_id', '')

        purged = await storage.purge_conversation_session(
            session_id, agent_id, reason=str(reason)
        )
        if purged == 0:
            raise HTTPException(
                status_code=404, detail="Conversation not found."
            )

        return {
            "success": True,
            "session_id": session_id,
            "purged_count": purged,
            "reason": reason,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error purging conversation {session_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error purging conversation."
        )


@router.post("/conversations/messages/{message_id}/restore")
@limiter.limit("30/minute")
async def restore_message(request: Request, message_id: int):
    """Restore a single soft-deleted message (#763 / #765)."""
    try:
        agent = get_agent(request)
        storage = agent.storage
        agent_id = getattr(storage, 'agent_id', '')

        restored = await storage.restore_conversation_message(
            message_id, agent_id
        )
        if not restored:
            raise HTTPException(
                status_code=404,
                detail="Message not found or not in trash.",
            )

        return {"success": True, "message_id": message_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error restoring message {message_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error restoring message."
        )


@router.post("/conversations/messages/{message_id}/purge")
@limiter.limit("10/minute")
async def purge_message(request: Request, message_id: int):
    """Permanently delete a single message (#763).

    Body: ``{"reason": "..."}`` (optional). The reason is recorded in
    the audit log.
    """
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        reason = (
            body.get("reason") if isinstance(body, dict) else None
        ) or "user-initiated"

        agent = get_agent(request)
        storage = agent.storage
        agent_id = getattr(storage, 'agent_id', '')

        purged = await storage.purge_conversation_message(
            message_id, agent_id, reason=str(reason)
        )
        if not purged:
            raise HTTPException(
                status_code=404, detail="Message not found."
            )

        return {
            "success": True,
            "message_id": message_id,
            "reason": reason,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error purging message {message_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error purging message."
        )


@router.get("/trash")
async def list_trash(request: Request, limit: int = Query(200, ge=1, le=1000)):
    """List soft-deleted messages for the Trash UI (#763 / #765).

    Returns rows where ``deleted_at IS NOT NULL``, sorted by
    ``deleted_at`` descending (most recently trashed first). Each row
    includes ``deleted_at`` so the UI can group by Today / Yesterday /
    Last 7 days / Older.

    The response is at the message level — the UI groups by session_id
    in metadata to present "deleted conversation X (N messages)".
    """
    try:
        agent = get_agent(request)
        storage = agent.storage

        history = await storage.list_trashed_conversations(limit=limit)
        return {
            "messages": history,
            "total": len(history),
        }
    except Exception as e:
        logger.error(f"Error listing trash: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error retrieving trash."
        )


@router.get("/conversations/{session_id}/transcript")
async def get_conversation_transcript(request: Request, session_id: str, decrypt: bool = True):
    """Get a human-readable markdown transcript for a conversation session."""
    try:
        agent = get_agent(request)
        storage = agent.storage

        # Use privacy-aware agent_id accessor
        agent_id = getattr(storage, 'agent_id', '')

        # Check if encryption is enabled via the wrapper's safe accessor
        encrypted_at_rest = getattr(storage, 'encryption_enabled', False)

        # Resolve the session's messages via the canonical dual-scheme
        # resolver so transcripts work for the UUID identifiers the list API
        # now advertises (not just numeric row-ids) — #2012.
        rows = await storage.query_session_rows(session_id, limit=1000)

        # A freshly started (marker-only) session resolves to zero displayable
        # rows; only 404 when the session genuinely doesn't exist (#2012).
        if not rows and not await storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")

        start_time = rows[0][4] if rows else ""

        # Build markdown transcript
        transcript_lines = [
            f"# Conversation Transcript - Session {session_id}",
            f"## {start_time}",
            ""
        ]

        if encrypted_at_rest:
            transcript_lines.append("_Note: This conversation was encrypted at rest._")
            transcript_lines.append("")

        message_count = 0

        for row in rows:
            msg_id, role, content, metadata_json, created_at = row[0], row[1], row[2], row[3], row[4]

            # Parse timestamp
            try:
                if isinstance(created_at, str):
                    timestamp = None
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                        try:
                            timestamp = datetime.strptime(created_at, fmt)
                            break
                        except ValueError:
                            continue
                    if timestamp is None:
                        timestamp = datetime.now()
                elif created_at is not None:
                    timestamp = created_at
                else:
                    timestamp = datetime.now()
            except (TypeError, ValueError):
                timestamp = datetime.now()

            # Parse metadata
            meta = None
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                except json.JSONDecodeError:
                    pass

            # Skip session markers (the resolver already strips these; this is
            # a defensive backstop). Session boundaries are handled upstream by
            # query_session_rows, so no gap/new_session break is needed here.
            if meta and meta.get('type') == 'session_marker':
                continue

            # Decrypt content if needed
            display_content = content
            is_encrypted = False
            if meta and meta.get('enc') and decrypt:
                fernet = get_agent_fernet(agent_id) if agent_id else None
                if fernet:
                    try:
                        display_content = decrypt_string(content, meta, fernet)
                    except Exception:
                        is_encrypted = True

            # Format role for display
            role_display = {
                "user": "**User**",
                "assistant": "**Assistant**",
                "system": "_System_"
            }.get(role, f"**{role.capitalize()}**")

            # Add message to transcript
            time_str = timestamp.strftime("%H:%M:%S")
            transcript_lines.append(f"### [{time_str}] {role_display}")

            if is_encrypted:
                transcript_lines.append("_[Encrypted content - decryption failed]_")
            else:
                # Add metadata annotations if present
                annotations = []
                if meta:
                    if meta.get('type') in ('compaction', 'context_summary'):
                        annotations.append(f"Type: {meta.get('type')}")
                    if meta.get('original_message_ids'):
                        orig_ids = meta.get('original_message_ids')
                        first = orig_ids[0] if orig_ids else None
                        last = orig_ids[-1] if orig_ids else None
                        if first and last:
                            annotations.append(f"Original messages: {first}-{last}")
                    if meta.get('excluded_from_context'):
                        annotations.append("Excluded from context")
                    if meta.get('summarized_into'):
                        annotations.append(f"Summarized into message #{meta.get('summarized_into')}")

                if annotations:
                    transcript_lines.append(f"_({', '.join(annotations)})_")

                transcript_lines.append(display_content)

            transcript_lines.append("")

            message_count += 1

        # Add footer
        transcript_lines.append("---")
        transcript_lines.append(f"_Generated: {datetime.now().isoformat()}_")
        transcript_lines.append(f"_Total messages: {message_count}_")

        transcript = "\n".join(transcript_lines)

        # Return as markdown
        return PlainTextResponse(
            content=transcript,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'inline; filename="transcript_{session_id}.md"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating transcript for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating transcript.")
