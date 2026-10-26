"""Canonical producer tools: original caller, one service, immutable receipts.

No tool resolves a filesystem path or creates another content store. Workflow
and computer producers can consume the returned revision-bound artifact part.
"""

from __future__ import annotations

from hashlib import sha256

from pydantic import ValidationError
from kestrel_sdk.artifacts import ArtifactCommit, ArtifactFile, ArtifactManifest, ArtifactProvenance, ArtifactRef
from kestrel_sdk.features.base import tool
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.artifacts.repository import ArtifactIntegrityError, ArtifactQuotaExceeded
from kestrel_sovereign.artifacts.service import ArtifactAccessDenied, ArtifactAction
from kestrel_sovereign.security.host_authority import HostAuthorityError

_TEXT_TYPES = frozenset({"text/plain", "text/markdown", "text/x-python", "text/javascript",
                         "text/css", "application/json", "application/javascript", "application/xml", "text/xml"})
_MAX_TEXT_BYTES = 1024 * 1024


def artifact_part(revision):
    """Presentation carries the canonical pinned ref, never an alternative URL."""
    return {"type": "artifact", "data": {"ref": revision.ref.model_dump(mode="json"), "title": revision.title}}


def _failure(error, publication_key=None):
    data = {"publication_key": publication_key} if publication_key is not None else {}
    if isinstance(error, ResourceOutcomeUnknown):
        return ToolResult.partial(
            "Publication outcome is unknown.",
            "Check artifact_publication_status with the original publication_key before retrying the identical request.",
            data={**data, "state": "unknown"},
        )
    if isinstance(error, (ArtifactAccessDenied, HostAuthorityError)):
        message, state = "Artifact authority is unavailable for the original caller.", "denied"
    elif isinstance(error, ResourceConflict):
        message, state = "Artifact changed or the publication key was reused. Compare the current head before a new save.", "conflict"
    elif isinstance(error, ResourceNotFound):
        message, state = "Artifact or staging is unavailable. Check the original publication key.", "unavailable"
    elif isinstance(error, ArtifactQuotaExceeded):
        message, state = "Artifact quota exceeded; this publication was rejected.", "rejected"
    elif isinstance(error, ArtifactIntegrityError):
        message, state = "Artifact storage cannot be verified.", "unavailable"
    else:
        message, state = "Invalid artifact request; text must be UTF-8, at most 1 MiB, with a supported media type and relative file path.", "rejected"
    return ToolResult.failed(message, data={**data, "state": state})


_ERRORS = (ArtifactAccessDenied, HostAuthorityError, ResourceConflict, ResourceNotFound,
           ResourceOutcomeUnknown, ArtifactIntegrityError, ArtifactQuotaExceeded, ValueError, ValidationError)


class ArtifactProducerTools:
    """Tools hosted by ArtifactsFeature; binding and storage remain its owners."""

    @tool("artifact_publish_text",
          "Save text, Markdown or code as a durable artifact, or publish a CAS revision. Choose a publication_key before the call; unknown outcomes require that same key and identical content.",
          category=ToolCategory.UTILITY)
    async def artifact_publish_text(self, title: str, text: str, path: str, publication_key: str,
                                    media_type: str = "text/markdown", artifact_id: str | None = None,
                                    expected_revision_id: str | None = None) -> ToolResult:
        """Publish bounded text through the canonical staging/receipt service.

        Args:
            title: Saved artifact title.
            text: Complete source text to retain.
            path: Relative single-file name, such as report.md.
            publication_key: Original stable idempotency key, reused only for the identical request.
            media_type: Supported text media type.
            artifact_id: Existing artifact ID for an edit; omit for creation.
            expected_revision_id: Exact current revision for an edit; omit for creation.
        """
        try:
            async with self._resolve_tool_context() as context:
                content = text.encode("utf-8")
                if len(content) > _MAX_TEXT_BYTES or media_type not in _TEXT_TYPES:
                    raise ValueError("unsupported text source")
                file = ArtifactFile(path=path, media_type=media_type, size_bytes=len(content), sha256=sha256(content).hexdigest())
                request = ArtifactCommit(idempotency_key=publication_key, title=title, artifact_id=artifact_id,
                    expected_revision_id=expected_revision_id,
                    manifest=ArtifactManifest(files=(file,), entrypoint=path),
                    provenance=ArtifactProvenance(tool_name="artifact_publish_text"))
                service = self._service
                async with service.authority.operation(context, ArtifactAction.WRITE, service.repository.storage_id):
                    receipt = await service.publication_status(context, publication_key)
                    if receipt is not None:
                        if receipt.request_sha256 != request.digest():
                            raise ResourceConflict("publication key already accepted a different request")
                    else:
                        try:
                            staged = await service.stage(context, content, publication_key=publication_key, path=path, media_type=media_type)
                            if staged != file:
                                raise ArtifactIntegrityError("staging mismatch")
                        except ResourceConflict:
                            # A concurrent identical publication may have won
                            # since status. Only canonical commit can decide.
                            pass
                        receipt = await service.commit(context, request)
                    try:
                        revision = await service.get(context, receipt.ref)
                    except _ERRORS:
                        raise ResourceOutcomeUnknown("publication accepted without metadata delivery") from None
                    return ToolResult.ok("Saved durable artifact revision.", data={"state": "saved", "receipt": receipt.model_dump(mode="json")},
                                         parts=[artifact_part(revision)])
        except _ERRORS as error:
            return _failure(error, publication_key)

    @tool("artifact_list", "Discover saved artifact heads by title, file name or media type. Continue with next_cursor even when a search page has no matches.", category=ToolCategory.UTILITY)
    async def artifact_list(self, query: str = "", cursor: str | None = None, limit: int = 50) -> ToolResult:
        try:
            async with self._resolve_tool_context() as context:
                page = await self._service.list(context, query=query, cursor=cursor, limit=limit)
                return ToolResult.ok("Saved artifact discovery page.", data=page.model_dump(mode="json"),
                                     parts=[artifact_part(item) for item in page.items])
        except _ERRORS as error:
            return _failure(error)

    @tool("artifact_get", "Open an exact saved revision, or its authoritative head. Set include_text to read bounded text/Markdown/code source for editing. Returns a shared artifact card.", category=ToolCategory.UTILITY)
    async def artifact_get(self, artifact_id: str, revision_id: str | None = None, include_text: bool = False) -> ToolResult:
        try:
            async with self._resolve_tool_context() as context:
                revision = await (self._service.get(context, ArtifactRef(artifact_id=artifact_id, revision_id=revision_id))
                                  if revision_id is not None else self._service.head(context, artifact_id))
                data = revision.model_dump(mode="json")
                if include_text:
                    file = next(file for file in revision.manifest.files if file.path == revision.manifest.entrypoint)
                    if file.media_type not in _TEXT_TYPES or file.size_bytes > _MAX_TEXT_BYTES:
                        return ToolResult.failed("Source is not bounded editable text; open or download it in the shared panel.",
                                                 data={"ref": data["ref"]}, parts=[artifact_part(revision)])
                    data["source_text"] = (await self._service.read(context, revision.ref, file.path)).decode("utf-8")
                return ToolResult.ok("Saved artifact revision.", data=data, parts=[artifact_part(revision)])
        except _ERRORS as error:
            return _failure(error)

    @tool("artifact_publication_status", "Reconcile publication with its original key. A missing receipt is unknown, never confirmation that a prior attempt failed.", category=ToolCategory.UTILITY)
    async def artifact_publication_status(self, publication_key: str) -> ToolResult:
        try:
            async with self._resolve_tool_context() as context:
                receipt = await self._service.publication_status(context, publication_key)
                if receipt is None:
                    return ToolResult.partial("No durable receipt was found.", "Outcome remains unknown; retain the original key and identical request.",
                                              data={"state": "unknown", "publication_key": publication_key})
                revision = await self._service.get(context, receipt.ref)
                return ToolResult.ok("Saved durable artifact revision.", data={"state": "saved", "receipt": receipt.model_dump(mode="json")},
                                     parts=[artifact_part(revision)])
        except _ERRORS as error:
            return _failure(error, publication_key)
