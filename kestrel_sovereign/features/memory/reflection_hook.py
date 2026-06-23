"""Pre-sleep memory application attestation.

The retriever increments ``access_count`` when a memory is surfaced into
context. This hook adds the stronger signal: before consolidation, ask the
agent's LLM which recently retrieved memories materially changed a response,
then route positive attestations through ``MemorySystem.mark_applied``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_MAX_CANDIDATES = 20
_MAX_MEMORY_CHARS = 1200
_MAX_CONTEXT_MESSAGES = 40
_MAX_CONTEXT_CHARS = 6000
_MAX_REASON_CHARS = 240


@dataclass
class RetrievedMemoryCandidate:
    message_id: int
    content: str
    retrieved_at: str
    created_at: Any
    role: str


class ReflectionSleepHook:
    """Sleep hook that marks load-bearing retrieved memories as applied."""

    async def on_pre_sleep(self, agent) -> Dict[str, Any]:
        memory = getattr(agent, "memory", None) or getattr(agent, "memory_system", None)
        if memory is None or not hasattr(memory, "mark_applied"):
            return {
                "success": True,
                "skipped": True,
                "reason": "memory_system_unavailable",
                "insights_generated": 0,
                "candidates": 0,
                "applied_count": 0,
            }

        db = self._resolve_db(agent)
        if db is None:
            return {
                "success": True,
                "skipped": True,
                "reason": "database_unavailable",
                "insights_generated": 0,
                "candidates": 0,
                "applied_count": 0,
            }

        cutoff = self._session_cutoff(agent)
        candidates = await self._recently_retrieved_memories(
            db,
            conversation=self._resolve_conversation(agent),
            agent_id=getattr(agent, "agent_id", None) or getattr(agent, "did", ""),
            cutoff=cutoff,
        )
        if not candidates:
            return {
                "success": True,
                "skipped": False,
                "insights_generated": 0,
                "candidates": 0,
                "applied_count": 0,
            }

        session_context = await self._session_context(agent, cutoff=cutoff)
        applied = 0
        attested_ids: List[int] = []
        for candidate in candidates:
            try:
                attestation = await self._attest_application(
                    agent,
                    candidate=candidate,
                    session_context=session_context,
                )
            except Exception as exc:  # noqa: BLE001 - hook must not block sleep
                logger.warning(
                    "memory reflection attestation failed for message %s: %s",
                    candidate.message_id,
                    exc,
                )
                continue

            if not attestation.get("applied"):
                continue
            reason = self._clean_reason(attestation.get("reason"))
            await memory.mark_applied(candidate.message_id, reason=reason)
            applied += 1
            attested_ids.append(candidate.message_id)

        return {
            "success": True,
            "skipped": False,
            "insights_generated": applied,
            "candidates": len(candidates),
            "applied_count": applied,
            "attested_message_ids": attested_ids,
        }

    def _resolve_db(self, agent):
        raw_storage = getattr(agent, "_raw_storage", None)
        if raw_storage is not None and getattr(raw_storage, "db", None) is not None:
            return raw_storage.db
        storage = getattr(agent, "storage", None)
        if storage is not None:
            wrapped = getattr(storage, "_storage", None)
            if wrapped is not None and getattr(wrapped, "db", None) is not None:
                return wrapped.db
            if getattr(storage, "db", None) is not None:
                return storage.db
        memory = getattr(agent, "memory_system", None)
        memory_storage = getattr(memory, "storage", None)
        return getattr(memory_storage, "db", None)

    def _resolve_conversation(self, agent):
        raw_storage = getattr(agent, "_raw_storage", None)
        if (
            raw_storage is not None
            and getattr(raw_storage, "conversation", None) is not None
        ):
            return raw_storage.conversation
        storage = getattr(agent, "storage", None)
        if storage is not None:
            if getattr(storage, "conversation", None) is not None:
                return storage.conversation
            wrapped = getattr(storage, "_storage", None)
            if (
                wrapped is not None
                and getattr(wrapped, "conversation", None) is not None
            ):
                return wrapped.conversation
        memory = getattr(agent, "memory_system", None)
        memory_storage = getattr(memory, "storage", None)
        return getattr(memory_storage, "conversation", None)

    def _session_cutoff(self, agent) -> datetime:
        gap_minutes = 30
        consolidator = getattr(
            getattr(agent, "memory_system", None),
            "consolidator",
            None,
        ) or getattr(agent, "memory_consolidator", None)
        if consolidator is not None:
            gap_minutes = getattr(consolidator, "SESSION_GAP_MINUTES", gap_minutes)
        else:
            try:
                from kestrel_sdk.config.constants import SESSION_GAP_MINUTES

                gap_minutes = SESSION_GAP_MINUTES
            except Exception:  # pragma: no cover - defensive fallback
                pass
        return datetime.now(timezone.utc) - timedelta(minutes=int(gap_minutes))

    async def _recently_retrieved_memories(
        self,
        db,
        *,
        conversation,
        agent_id: str,
        cutoff: datetime,
    ) -> List[RetrievedMemoryCandidate]:
        rows = await db.fetchall(
            """SELECT id, role, content, metadata, created_at
               FROM conversation_history
               WHERE agent_id = ?
                 AND deleted_at IS NULL
                 AND metadata LIKE ?
               ORDER BY id DESC
               LIMIT ?""",
            (agent_id, "%last_accessed%", _MAX_CANDIDATES * 4),
        )

        candidates: List[RetrievedMemoryCandidate] = []
        for row in rows:
            msg_id, role, content, metadata_raw, created_at = row
            metadata = self._parse_metadata(metadata_raw)
            retrieved_at = metadata.get("last_accessed")
            if not retrieved_at:
                continue
            retrieved_dt = self._parse_datetime(retrieved_at)
            if retrieved_dt is None or retrieved_dt < cutoff:
                continue
            content = await self._decode_content(conversation, content, metadata)
            candidates.append(
                RetrievedMemoryCandidate(
                    message_id=int(msg_id),
                    role=role,
                    content=content or "",
                    retrieved_at=retrieved_dt.isoformat(),
                    created_at=created_at,
                )
            )
            if len(candidates) >= _MAX_CANDIDATES:
                break

        candidates.sort(key=lambda item: item.retrieved_at)
        return candidates

    async def _decode_content(
        self,
        conversation,
        content: str,
        metadata: Dict[str, Any],
    ) -> str:
        if conversation is None or not hasattr(conversation, "_decrypt_with_fallback"):
            return content or ""
        try:
            decoded, _needs_migration = conversation._decrypt_with_fallback(
                content or "",
                metadata,
            )
            return decoded
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not decrypt retrieved memory candidate: %s", exc)
            return content or ""

    async def _session_context(self, agent, *, cutoff: datetime) -> str:
        conversation = self._resolve_conversation(agent)

        if conversation is None or not hasattr(conversation, "get_conversation_history"):
            return ""

        try:
            history = await conversation.get_conversation_history(
                limit=_MAX_CONTEXT_MESSAGES
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not load session context for memory attestation: %s", exc)
            return ""

        lines: List[str] = []
        for msg in history:
            created_at = self._parse_datetime(msg.get("created_at"))
            if created_at is not None and created_at < cutoff:
                continue
            role = msg.get("role", "unknown")
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        context = "\n".join(lines)
        return context[-_MAX_CONTEXT_CHARS:]

    async def _attest_application(
        self,
        agent,
        *,
        candidate: RetrievedMemoryCandidate,
        session_context: str,
    ) -> Dict[str, Any]:
        llm_service = getattr(agent, "llm_service", None)
        if llm_service is None or not hasattr(llm_service, "generate"):
            return {"applied": False, "reason": "LLM unavailable"}

        prompt = (
            "A memory was retrieved during the just-ended session. Decide whether "
            "it materially influenced one of the assistant's responses or actions. "
            "A memory is applied only if it changed what the assistant said or did "
            "next, not merely because it appeared in context.\n\n"
            f"Retrieved memory id: {candidate.message_id}\n"
            f"Retrieved at: {candidate.retrieved_at}\n"
            f"Memory role: {candidate.role}\n"
            f"Memory content:\n{candidate.content[:_MAX_MEMORY_CHARS]}\n\n"
            f"Session context:\n{session_context or '(no recent session context available)'}\n\n"
            "Answer as JSON only, with this shape: "
            '{"applied": true|false, "reason": "one sentence"}.'
        )
        response = await llm_service.generate(
            system_prompt=(
                "You are auditing memory application. Be conservative. "
                "Return JSON only."
            ),
            user_prompt=prompt,
        )
        return self._parse_attestation(self._response_text(response))

    def _parse_attestation(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"applied": False, "reason": ""}

        json_text = text
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            json_text = fenced.group(1).strip()
        try:
            data = json.loads(json_text)
            if isinstance(data, dict):
                applied = self._coerce_bool(data.get("applied", data.get("yes")))
                return {
                    "applied": applied,
                    "reason": str(data.get("reason") or "").strip(),
                }
        except (json.JSONDecodeError, TypeError):
            pass

        first = text.splitlines()[0].strip()
        lowered = first.lower()
        applied = lowered.startswith("yes") or lowered.startswith("applied: yes")
        reason = re.sub(r"^(applied:\s*)?yes\b\s*[-:,.]?\s*", "", first, flags=re.I)
        if not applied:
            reason = re.sub(r"^(applied:\s*)?no\b\s*[-:,.]?\s*", "", first, flags=re.I)
        return {"applied": applied, "reason": reason.strip()}

    def _response_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        content = getattr(response, "content", None)
        if content is not None:
            return str(content)
        return str(response)

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"yes", "true", "applied"}
        return False

    def _parse_metadata(self, metadata: Any) -> Dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str) and metadata:
            try:
                parsed = json.loads(metadata)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str) and value:
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _clean_reason(self, reason: Any) -> str:
        text = str(reason or "").strip()
        if not text:
            return "LLM attested this retrieved memory materially influenced the session."
        first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        return first_sentence[:_MAX_REASON_CHARS]
