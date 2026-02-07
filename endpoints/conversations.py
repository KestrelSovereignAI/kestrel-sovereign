"""Conversation and session endpoints."""
from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
import json
import logging

from kestrel_sovereign.kestrel_config.constants import SESSION_GAP_MINUTES
from kestrel_sovereign.storage.encryption import get_fernet, get_agent_fernet, decrypt_string

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["conversations"])


@router.get("/sessions")
async def list_sessions(request: Request, limit: int = 50):
    """List conversation sessions with summary info."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        storage = agent.storage
        history = await storage.get_conversation_history(limit)
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
async def list_conversations(request: Request, limit: int = 50, decrypt: bool = True):
    """List conversation sessions grouped by date/time."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        storage = agent.storage

        # Use async database query with agent_id filter for multi-tenant isolation
        agent_id = getattr(storage, 'agent_id', '') or getattr(storage._storage, 'agent_id', '')

        # Check if encryption is enabled - traverse through wrappers to find conversation store
        encrypted_at_rest = False
        conv_store = getattr(storage, 'conversation', None) or getattr(getattr(storage, '_storage', None), 'conversation', None)
        if conv_store and hasattr(conv_store, 'encryption_enabled'):
            encrypted_at_rest = conv_store.encryption_enabled

        rows = await storage.db.fetchall("""
            SELECT id, role, content, metadata, created_at
            FROM conversation_history
            WHERE agent_id = ?
            ORDER BY created_at DESC
        """, (agent_id,))

        if not rows:
            return {"conversations": [], "total": 0, "encrypted_at_rest": encrypted_at_rest}

        sessions = []
        current_session = None

        for row in reversed(rows):
            msg_id, role, content, metadata_json, created_at = row[0], row[1], row[2], row[3], row[4]

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
                logger.warning(f"Failed to parse timestamp for message {msg_id}: {e}")
                timestamp = datetime.now()

            # Check for explicit new_session marker in metadata
            is_new_session_marker = False
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                    if meta.get('new_session'):
                        is_new_session_marker = True
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for message {msg_id}: {e}")

            if current_session is None:
                current_session = {
                    "session_id": str(msg_id),
                    "started_at": timestamp.isoformat(),
                    "last_message_at": timestamp.isoformat(),
                    "message_count": 0,
                    "user_message_count": 0,
                    "preview": "",
                    "messages": []
                }

            last_ts = datetime.fromisoformat(current_session["last_message_at"])
            gap_minutes = (timestamp - last_ts).total_seconds() / 60

            # Start new session if time gap OR explicit new_session marker
            if gap_minutes > SESSION_GAP_MINUTES or is_new_session_marker:
                if current_session["message_count"] > 0:
                    sessions.append(current_session)
                current_session = {
                    "session_id": str(msg_id),
                    "started_at": timestamp.isoformat(),
                    "last_message_at": timestamp.isoformat(),
                    "message_count": 0,
                    "user_message_count": 0,
                    "preview": "",
                    "messages": []
                }
                # Skip counting the session marker itself
                if is_new_session_marker:
                    continue

            current_session["message_count"] += 1
            current_session["last_message_at"] = timestamp.isoformat()
            if role == "user":
                current_session["user_message_count"] += 1
                if not current_session["preview"]:
                    preview_content = content
                    is_encrypted = False
                    decryption_failed = False
                    if metadata_json:
                        try:
                            meta = json.loads(metadata_json)
                            if meta.get('enc'):
                                is_encrypted = True
                                # Decrypt preview if requested - use per-agent key
                                if decrypt:
                                    fernet = get_agent_fernet(agent_id) if agent_id else get_fernet()
                                    if fernet:
                                        try:
                                            preview_content = decrypt_string(content, meta, fernet)
                                        except Exception as decrypt_err:
                                            logger.warning(f"Failed to decrypt preview: {decrypt_err}")
                                            decryption_failed = True
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse metadata for preview in message {msg_id}: {e}")
                    current_session["preview"] = preview_content[:100] + ("..." if len(preview_content) > 100 else "")
                    current_session["preview_encrypted"] = is_encrypted and (not decrypt or decryption_failed)

        if current_session and current_session["message_count"] > 0:
            sessions.append(current_session)

        sessions = list(reversed(sessions))[:limit]

        return {
            "conversations": sessions,
            "total": len(sessions),
            "encrypted_at_rest": encrypted_at_rest
        }
    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving conversations.")


@router.get("/conversations/{session_id}")
async def get_conversation(request: Request, session_id: str, limit: int = 100, decrypt: bool = True):
    """Get messages for a specific conversation session."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        storage = agent.storage

        # Get agent_id for multi-tenant isolation
        agent_id = getattr(storage, 'agent_id', '') or getattr(storage._storage, 'agent_id', '')

        # Check if encryption is enabled
        encrypted_at_rest = False
        conv_store = getattr(storage, 'conversation', None) or getattr(getattr(storage, '_storage', None), 'conversation', None)
        if conv_store and hasattr(conv_store, 'encryption_enabled'):
            encrypted_at_rest = conv_store.encryption_enabled

        start_row = await storage.db.fetchone(
            "SELECT created_at FROM conversation_history WHERE id = ? AND agent_id = ?",
            (session_id, agent_id)
        )

        if not start_row:
            raise HTTPException(status_code=404, detail="Session not found.")

        start_time = start_row[0]

        rows = await storage.db.fetchall("""
            SELECT id, role, content, metadata, created_at
            FROM conversation_history
            WHERE agent_id = ? AND created_at >= ?
            ORDER BY created_at ASC
            LIMIT ?
        """, (agent_id, start_time, limit * 2))

        messages = []
        last_timestamp = None
        is_first_message = True

        for row in rows:
            msg_id, role, content, metadata_json, created_at = row[0], row[1], row[2], row[3], row[4]

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

            # Check for new_session marker (but include first message even if it's a marker)
            meta = None
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for message {msg_id}: {e}")

            # Stop at new_session marker (unless it's the first message which starts the session)
            if not is_first_message and meta and meta.get('new_session'):
                break

            if last_timestamp:
                gap_minutes = (timestamp - last_timestamp).total_seconds() / 60
                if gap_minutes > SESSION_GAP_MINUTES:
                    break

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

            # Skip session markers from message list (they're just metadata)
            if meta and meta.get('type') == 'session_marker':
                last_timestamp = timestamp
                is_first_message = False
                continue

            messages.append({
                "id": msg_id,
                "role": role,
                "content": display_content,
                "encrypted": is_encrypted or decryption_failed,
                "metadata": meta or {},
                "created_at": timestamp.isoformat()
            })

            last_timestamp = timestamp
            is_first_message = False

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
async def start_new_conversation(request: Request):
    """Start a new conversation by adding a session marker."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        storage = agent.storage

        await storage.add_conversation(
            role="system",
            content="[New conversation started]",
            metadata={"type": "session_marker", "new_session": True}
        )

        # Get the newly created session with agent_id filter
        agent_id = getattr(storage, 'agent_id', '') or getattr(storage._storage, 'agent_id', '')
        row = await storage.db.fetchone("""
            SELECT id, created_at FROM conversation_history
            WHERE agent_id = ?
            ORDER BY id DESC LIMIT 1
        """, (agent_id,))

        return {
            "success": True,
            "session_id": str(row[0]) if row else None,
            "started_at": row[1] if row else None,
            "message": "New conversation started"
        }
    except Exception as e:
        logger.error(f"Error starting new conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error starting new conversation.")


@router.get("/conversations/{session_id}/transcript")
async def get_conversation_transcript(request: Request, session_id: str, decrypt: bool = True):
    """Get a human-readable markdown transcript for a conversation session."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        storage = agent.storage

        # Get agent_id for multi-tenant isolation
        agent_id = getattr(storage, 'agent_id', '') or getattr(storage._storage, 'agent_id', '')

        # Check if encryption is enabled
        encrypted_at_rest = False
        conv_store = getattr(storage, 'conversation', None) or getattr(getattr(storage, '_storage', None), 'conversation', None)
        if conv_store and hasattr(conv_store, 'encryption_enabled'):
            encrypted_at_rest = conv_store.encryption_enabled

        # Get the session start time
        start_row = await storage.db.fetchone(
            "SELECT created_at FROM conversation_history WHERE id = ? AND agent_id = ?",
            (session_id, agent_id)
        )

        if not start_row:
            raise HTTPException(status_code=404, detail="Session not found.")

        start_time = start_row[0]

        # Get all messages in this session
        rows = await storage.db.fetchall("""
            SELECT id, role, content, metadata, created_at
            FROM conversation_history
            WHERE agent_id = ? AND created_at >= ?
            ORDER BY created_at ASC
        """, (agent_id, start_time))

        if not rows:
            raise HTTPException(status_code=404, detail="No messages found in session.")

        # Build markdown transcript
        transcript_lines = [
            f"# Conversation Transcript - Session {session_id}",
            f"## {start_time}",
            ""
        ]

        if encrypted_at_rest:
            transcript_lines.append("_Note: This conversation was encrypted at rest._")
            transcript_lines.append("")

        last_timestamp = None
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

            # Check for session boundary
            if meta and meta.get('new_session') and message_count > 0:
                break

            # Check for time gap
            if last_timestamp:
                from kestrel_sovereign.kestrel_config.constants import SESSION_GAP_MINUTES
                gap_minutes = (timestamp - last_timestamp).total_seconds() / 60
                if gap_minutes > SESSION_GAP_MINUTES:
                    break

            # Skip session markers
            if meta and meta.get('type') == 'session_marker':
                last_timestamp = timestamp
                continue

            # Decrypt content if needed
            display_content = content
            is_encrypted = False
            if meta and meta.get('enc') and decrypt:
                from kestrel_sovereign.storage.encryption import get_agent_fernet, decrypt_string
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
                    if meta.get('type') in ('compression', 'context_summary'):
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

            last_timestamp = timestamp
            message_count += 1

        # Add footer
        transcript_lines.append("---")
        transcript_lines.append(f"_Generated: {datetime.now().isoformat()}_")
        transcript_lines.append(f"_Total messages: {message_count}_")

        transcript = "\n".join(transcript_lines)

        # Return as markdown
        from fastapi.responses import PlainTextResponse
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
