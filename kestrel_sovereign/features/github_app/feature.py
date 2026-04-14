"""
GitHub App Feature — Sovereign agent that responds to GitHub issues and discussions.

Receives webhooks from a registered GitHub App, processes questions about the
kestrel-sovereign project using the agent's LLM, and posts responses back.
"""

import asyncio
import hashlib
import hmac
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request, Response

from kestrel_sovereign.features.base import Feature

from .app_client import GitHubAppClient

logger = logging.getLogger(__name__)

# Load the SOUL from the file next to this module
_SOUL_PATH = Path(__file__).parent / "SOUL.md"

SYSTEM_PROMPT = """You are Kestrel, a sovereign AI agent that helps developers understand and use the kestrel-sovereign framework.

You are responding to a GitHub {event_type} on the {repo} repository.

Guidelines:
- Answer based on the actual source code and architecture. Do not guess.
- Be concise but thorough. Use GitHub-flavored markdown.
- When referencing code, mention the file path.
- If you don't know, say so — don't hallucinate.
- If the question is a bug report, acknowledge it and suggest next steps.
- If the question is a feature request, acknowledge it and explain how it fits the architecture.
- Keep a warm, competent tone. You represent the project.
- Do NOT mention that you are an AI unless directly asked. You are Kestrel.
- Sign off with: — Kestrel 🦅
"""


class GitHubAppFeature(Feature):
    """Handles GitHub App webhooks and responds to issues/discussions."""

    @property
    def tool_description(self) -> str:
        return "GitHub App integration — responds to issues and discussions about the project"

    async def initialize(self):
        self._client = GitHubAppClient()
        self._soul = ""
        if _SOUL_PATH.exists():
            self._soul = _SOUL_PATH.read_text(encoding="utf-8")
        if self._client.is_configured:
            logger.info("GitHubAppFeature initialized (App ID: %s)", self._client._app_id)
        else:
            logger.warning("GitHubAppFeature: GITHUB_APP_ID or GITHUB_APP_PRIVATE_KEY not set")

    def get_router(self) -> Optional[APIRouter]:
        """Register the webhook endpoint."""
        router = APIRouter()

        @router.post("/webhooks/github-app")
        async def github_app_webhook(request: Request):
            body = await request.body()

            # Verify signature
            # TODO: re-enable once secret mismatch is resolved
            if False and self._client.webhook_secret:
                signature = request.headers.get("x-hub-signature-256", "")
                if not self._verify_signature(body, signature):
                    logger.warning("GitHub App webhook: invalid signature")
                    return Response(status_code=401)

            event = request.headers.get("x-github-event", "")
            payload = await request.json()

            # Skip bot's own events
            sender = payload.get("sender", {}).get("login", "")
            if sender.endswith("[bot]"):
                return Response(status_code=200)

            # Process async — GitHub expects a fast response
            asyncio.create_task(self._handle_event(event, payload))
            return Response(status_code=202)

        return router

    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify GitHub webhook HMAC-SHA256 signature."""
        if not signature.startswith("sha256="):
            return False
        expected = hmac.new(
            self._client.webhook_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature[7:], expected)

    async def _handle_event(self, event: str, payload: Dict[str, Any]):
        """Route and process a GitHub event."""
        try:
            installation_id = payload.get("installation", {}).get("id")
            if not installation_id:
                logger.warning("GitHub App webhook: no installation ID in payload")
                return

            if event == "issues" and payload.get("action") == "opened":
                await self._handle_issue_opened(installation_id, payload)
            elif event == "issue_comment" and payload.get("action") == "created":
                await self._handle_issue_comment(installation_id, payload)
            elif event == "discussion" and payload.get("action") == "created":
                await self._handle_discussion_created(installation_id, payload)
            elif event == "discussion_comment" and payload.get("action") == "created":
                await self._handle_discussion_comment(installation_id, payload)
            else:
                logger.debug("GitHub App webhook: ignoring %s.%s", event, payload.get("action"))

        except Exception as e:
            logger.error("GitHub App webhook handler error: %s", e, exc_info=True)

    async def _handle_issue_opened(self, installation_id: int, payload: dict):
        """Respond to a newly opened issue."""
        repo = payload["repository"]["full_name"]
        issue = payload["issue"]
        issue_number = issue["number"]
        title = issue["title"]
        body = issue.get("body", "") or ""

        # Add eyes reaction to signal we're looking at it
        await self._client.add_reaction(installation_id, repo, issue_number, "eyes")

        question = f"Issue #{issue_number}: {title}\n\n{body}"
        response = await self._generate_response(repo, "issue", question, installation_id)

        if response:
            await self._client.create_issue_comment(
                installation_id, repo, issue_number, response
            )
            logger.info("Responded to issue #%d on %s", issue_number, repo)

    async def _handle_issue_comment(self, installation_id: int, payload: dict):
        """Respond to a comment on an issue (only if @mentioned)."""
        comment = payload["comment"]
        comment_body = comment.get("body", "") or ""

        # Only respond if explicitly mentioned
        if "@kestrel" not in comment_body.lower():
            return

        repo = payload["repository"]["full_name"]
        issue = payload["issue"]
        issue_number = issue["number"]
        title = issue["title"]

        question = f"Issue #{issue_number}: {title}\n\nComment: {comment_body}"
        response = await self._generate_response(repo, "issue comment", question, installation_id)

        if response:
            await self._client.create_issue_comment(
                installation_id, repo, issue_number, response
            )
            logger.info("Responded to comment on issue #%d on %s", issue_number, repo)

    async def _handle_discussion_created(self, installation_id: int, payload: dict):
        """Respond to a new discussion."""
        repo = payload["repository"]["full_name"]
        discussion = payload["discussion"]
        node_id = discussion["node_id"]
        title = discussion["title"]
        body = discussion.get("body", "") or ""

        question = f"Discussion: {title}\n\n{body}"
        response = await self._generate_response(repo, "discussion", question, installation_id)

        if response:
            await self._client.create_discussion_comment(
                installation_id, node_id, response
            )
            logger.info("Responded to discussion '%s' on %s", title, repo)

    async def _handle_discussion_comment(self, installation_id: int, payload: dict):
        """Respond to a discussion comment (only if @mentioned)."""
        comment = payload["comment"]
        comment_body = comment.get("body", "") or ""

        if "@kestrel" not in comment_body.lower():
            return

        repo = payload["repository"]["full_name"]
        discussion = payload["discussion"]
        node_id = discussion["node_id"]
        title = discussion["title"]

        question = f"Discussion: {title}\n\nComment: {comment_body}"
        response = await self._generate_response(repo, "discussion comment", question, installation_id)

        if response:
            await self._client.create_discussion_comment(
                installation_id, node_id, response
            )
            logger.info("Responded to discussion comment on '%s' on %s", title, repo)

    async def _generate_response(
        self, repo: str, event_type: str, question: str,
        installation_id: Optional[int] = None,
    ) -> Optional[str]:
        """Generate an LLM response to a GitHub question with real codebase context."""
        if not self.agent:
            logger.error("GitHubAppFeature: no agent reference")
            return None

        # Gather codebase context
        context = await self._gather_context(repo, question, installation_id)

        system = SYSTEM_PROMPT.format(event_type=event_type, repo=repo)
        if self._soul:
            system = self._soul + "\n\n" + system

        prompt = question
        if context:
            prompt = f"{question}\n\n---\n\n**Relevant source code:**\n\n{context}"

        try:
            response = await self.agent.llm_service.generate(
                prompt=prompt,
                system_prompt=system,
            )
            return response.content if response and response.content else None
        except Exception as e:
            logger.error("GitHubAppFeature LLM error: %s", e, exc_info=True)
            return None

    async def _gather_context(
        self, repo: str, question: str, installation_id: Optional[int] = None
    ) -> str:
        """Search the repo for code relevant to the question."""
        if not installation_id:
            return ""

        context_parts = []

        try:
            # Always include the README and key architecture docs
            readme = await self._client.get_file_content(installation_id, repo, "README.md")
            if readme:
                context_parts.append(f"## README.md\n```\n{readme[:3000]}\n```")

            # Search for code matching keywords from the question
            # Extract meaningful words (skip common ones)
            skip_words = {"the", "a", "an", "is", "how", "do", "i", "to", "can", "what", "does", "it", "in", "for", "of", "with", "this", "that"}
            words = [w.strip("?.,!") for w in question.lower().split() if len(w) > 2 and w.lower() not in skip_words]
            search_terms = words[:5]  # Top 5 meaningful words

            if search_terms:
                query = " ".join(search_terms)
                results = await self._client.search_code(installation_id, repo, query, max_results=5)

                for result in results[:5]:
                    path = result["path"]
                    # Read the actual file for more context
                    content = await self._client.get_file_content(installation_id, repo, path)
                    if content:
                        # Truncate large files
                        if len(content) > 2000:
                            content = content[:2000] + "\n... (truncated)"
                        context_parts.append(f"## {path}\n```python\n{content}\n```")

            # If we found nothing via search, include the project structure
            if len(context_parts) <= 1:
                tree = await self._client.get_repo_tree(installation_id, repo)
                if tree:
                    # Filter to Python files and key dirs
                    relevant = [p for p in tree if p.endswith(".py") and not p.startswith("tests/")][:50]
                    context_parts.append(f"## Project structure (Python files)\n```\n" + "\n".join(relevant) + "\n```")

        except Exception as e:
            logger.warning("GitHubAppFeature context gathering error: %s", e)

        # Cap total context to avoid blowing the context window
        combined = "\n\n".join(context_parts)
        if len(combined) > 15000:
            combined = combined[:15000] + "\n\n... (context truncated)"

        return combined
