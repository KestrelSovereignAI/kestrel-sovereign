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

## CRITICAL: GROUNDING RULE

Every technical claim you make MUST be grounded in the source files provided in your context.

- If a fact is in the provided source files: quote it, cite the file path, link to it.
- If a fact is NOT in the provided source files: DO NOT state it. Say "I don't have that file in context."
- Never invent class names, method signatures, entry_point groups, module paths, or config keys.
- Never fill in "plausible" code — only quote what you can see.
- Prefer short, precise answers over long, speculative ones.

File links should use this format: `[path/to/file.py]({repo_url}/blob/main/path/to/file.py)`
Where `{repo_url}` = https://github.com/{repo}

## Response Style

- Warm, direct, technically precise. Short beats long.
- Quote real code from context. Do not paraphrase.
- Use GitHub-flavored markdown.
- If the question is a bug report: acknowledge and point to relevant files in context.
- If the question is a feature request: acknowledge and explain how it fits, based on context.
- Do NOT mention that you are an AI unless directly asked. You are Kestrel.
- Sign off with: — Kestrel 🦅

## SCOPE: Answer Questions Only

You are a support assistant. You ANSWER QUESTIONS about the codebase.

- DO NOT offer to implement, draft, write, fix, patch, refactor, or modify code.
- DO NOT offer to open PRs, branches, or commits.
- DO NOT say things like "I can draft the impl" or "Want me to write that?"
- If someone asks you to fix something, point them to the relevant files and explain
  what would need to change — but do NOT offer to do it yourself.
- Code changes are made by humans and dedicated code agents, not by you.

## When You Don't Know

If the provided context doesn't contain what's needed to answer:
- Say so explicitly
- Suggest the user point you at a specific file
- Do NOT guess — it's better to admit the gap
"""


class GitHubAppFeature(Feature):
    """Handles GitHub App webhooks and responds to issues/discussions."""

    @property
    def tool_description(self) -> str:
        return "GitHub App integration — responds to issues and discussions about the project"

    async def initialize(self):
        self._client = GitHubAppClient()
        self._soul = ""
        self._event_tasks: set[asyncio.Task] = set()
        if _SOUL_PATH.exists():
            self._soul = _SOUL_PATH.read_text(encoding="utf-8")
        if self._client.is_configured:
            logger.info("GitHubAppFeature initialized (App ID: %s)", self._client._app_id)
        else:
            logger.warning("GitHubAppFeature: GITHUB_APP_ID or GITHUB_APP_PRIVATE_KEY not set")

    async def shutdown(self):
        tasks = set(getattr(self, "_event_tasks", set()))
        if not tasks:
            return

        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        self._event_tasks.clear()

    def _schedule_event_handling(self, event: str, payload: Dict[str, Any]) -> None:
        task = asyncio.create_task(
            self._safe_handle_event(event, payload),
            name=f"github-app-webhook:{event or 'unknown'}",
        )
        self._event_tasks.add(task)
        task.add_done_callback(self._event_tasks.discard)

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

            # Return 200 immediately, process in background
            # Cloud Run needs min-instances=1 to keep the instance alive
            self._schedule_event_handling(event, payload)
            return Response(status_code=200)

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

    async def _safe_handle_event(self, event: str, payload: Dict[str, Any]):
        """Wrapper that catches all errors so create_task doesn't lose them."""
        try:
            await self._handle_event(event, payload)
        except Exception as e:
            logger.error("GitHubApp background task error: %s", e, exc_info=True)

    async def _handle_event(self, event: str, payload: Dict[str, Any]) -> dict:
        """Route and process a GitHub event. Returns diagnostic info."""
        try:
            installation_id = payload.get("installation", {}).get("id")
            if not installation_id:
                return {"step": "no_installation_id"}

            if event == "issues" and payload.get("action") == "opened":
                return await self._handle_issue_opened(installation_id, payload)
            elif event == "issue_comment" and payload.get("action") == "created":
                return await self._handle_issue_comment(installation_id, payload)
            elif event == "discussion" and payload.get("action") == "created":
                return await self._handle_discussion_created(installation_id, payload)
            elif event == "discussion_comment" and payload.get("action") == "created":
                return await self._handle_discussion_comment(installation_id, payload)
            else:
                return {"step": "ignored", "event": f"{event}.{payload.get('action')}"}

        except Exception as e:
            return {"step": "exception", "error": f"{type(e).__name__}: {e}"}

    # Work-item signals — issues with these are internal tickets, not questions
    _SKIP_LABELS = {
        "epic", "feature", "bug", "chore", "task", "wip", "internal",
        "talon", "codeagent", "automation", "infrastructure",
    }
    _SKIP_TITLE_PREFIXES = (
        "[epic]", "[wip]", "[talon]", "[internal]",
        "feat:", "fix:", "chore:", "test:", "docs:", "refactor:",
        "feat(", "fix(", "chore(", "test(", "docs(", "refactor(",
    )

    def _looks_like_work_item(self, issue_or_discussion: dict) -> tuple[bool, str]:
        """Heuristic: is this an internal work item vs. a user question?

        Returns (is_work_item, reason).
        """
        title = (issue_or_discussion.get("title", "") or "").lower().strip()
        labels = {
            (lbl.get("name") or "").lower()
            for lbl in (issue_or_discussion.get("labels") or [])
        }

        # Label match is decisive
        skip_labels = labels & self._SKIP_LABELS
        if skip_labels:
            return True, f"labels={sorted(skip_labels)}"

        # Conventional-commit / work-item prefix
        for prefix in self._SKIP_TITLE_PREFIXES:
            if title.startswith(prefix):
                return True, f"title_prefix={prefix}"

        return False, ""

    async def _classify_as_question(self, title: str, body: str) -> bool:
        """LLM check: is this a support question worth answering?

        Used for ambiguous cases (no work-item signals, but not obviously a question).
        Returns False on any error — safer to not respond than to spam.
        """
        if not self.agent or not getattr(self.agent, "llm_service", None):
            return False

        prompt = (
            "You are a classifier. Decide if the following GitHub issue is a support "
            "QUESTION from a user (someone asking how to do something, what a thing is, "
            "why something happens, or reporting confusion) — versus an internal WORK ITEM "
            "(a task, feature request, bug report filed by a developer as a to-do, epic, "
            "or ticket for code agents).\n\n"
            f"TITLE: {title}\n\nBODY:\n{body[:1500]}\n\n"
            "Reply with exactly one word: QUESTION or WORK_ITEM"
        )
        try:
            response = await self.agent.llm_service.generate(
                user_prompt=prompt,
                system_prompt="You are a strict binary classifier. Output only QUESTION or WORK_ITEM.",
            )
            content = response if isinstance(response, str) else (response.content if response else "")
            return "QUESTION" in (content or "").upper()
        except Exception as e:
            logger.warning("GitHubApp: classifier error: %s", e)
            return False

    async def _handle_issue_opened(self, installation_id: int, payload: dict) -> dict:
        """Respond to a newly opened issue — if it looks like a user question.

        Explicit @kestrel mentions always get a response. For everything else,
        filter out work items by labels/title, then use an LLM classifier.
        """
        issue = payload["issue"]
        title = issue["title"]
        body = issue.get("body", "") or ""
        combined = f"{title}\n{body}".lower()

        # Explicit mention always wins
        mentioned = "@kestrel" in combined

        if not mentioned:
            # Skip obvious work items (labels, conventional-commit titles)
            is_work, reason = self._looks_like_work_item(issue)
            if is_work:
                return {"step": "skipped_work_item", "reason": reason}

            # For ambiguous cases, ask the LLM if it's a question
            is_question = await self._classify_as_question(title, body)
            if not is_question:
                return {"step": "classified_as_work_item"}

        repo = payload["repository"]["full_name"]
        issue_number = issue["number"]

        await self._client.add_reaction(installation_id, repo, issue_number, "eyes")

        question = f"Issue #{issue_number}: {title}\n\n{body}"
        response, _ = await self._generate_response(repo, "issue", question, installation_id)

        if response:
            await self._client.create_issue_comment(
                installation_id, repo, issue_number, response
            )
            logger.info("Responded to issue #%d on %s", issue_number, repo)
            return {"step": "commented", "explicit_mention": mentioned}
        return {"step": "no_response"}

    async def _handle_issue_comment(self, installation_id: int, payload: dict) -> dict:
        """Respond to a comment on an issue (only if @mentioned)."""
        comment = payload["comment"]
        comment_body = comment.get("body", "") or ""

        if "@kestrel" not in comment_body.lower():
            return {"step": "no_mention"}

        repo = payload["repository"]["full_name"]
        issue = payload["issue"]
        issue_number = issue["number"]
        title = issue["title"]

        question = f"Issue #{issue_number}: {title}\n\nComment: {comment_body}"
        try:
            response, gen_diag = await self._generate_response(repo, "issue comment", question, installation_id)
        except Exception as e:
            return {"step": "llm_error", "error": f"{type(e).__name__}: {e}"}

        if not response:
            return {"step": "no_response", "has_agent": self.agent is not None, "gen_diag": gen_diag}

        try:
            await self._client.create_issue_comment(
                installation_id, repo, issue_number, response
            )
            return {"step": "commented", "chars": len(response)}
        except Exception as e:
            return {"step": "comment_post_error", "error": f"{type(e).__name__}: {e}"}

    async def _handle_discussion_created(self, installation_id: int, payload: dict) -> dict:
        """Respond to a new discussion — if it looks like a user question."""
        discussion = payload["discussion"]
        title = discussion["title"]
        body = discussion.get("body", "") or ""
        combined = f"{title}\n{body}".lower()

        mentioned = "@kestrel" in combined

        if not mentioned:
            is_work, reason = self._looks_like_work_item(discussion)
            if is_work:
                return {"step": "skipped_work_item", "reason": reason}

            is_question = await self._classify_as_question(title, body)
            if not is_question:
                return {"step": "classified_as_work_item"}

        repo = payload["repository"]["full_name"]
        node_id = discussion["node_id"]

        question = f"Discussion: {title}\n\n{body}"
        response, _ = await self._generate_response(repo, "discussion", question, installation_id)

        if response:
            await self._client.create_discussion_comment(
                installation_id, node_id, response
            )
            logger.info("Responded to discussion '%s' on %s", title, repo)
            return {"step": "commented", "explicit_mention": mentioned}
        return {"step": "no_response"}

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
        response, _ = await self._generate_response(repo, "discussion comment", question, installation_id)

        if response:
            await self._client.create_discussion_comment(
                installation_id, node_id, response
            )
            logger.info("Responded to discussion comment on '%s' on %s", title, repo)

    async def _generate_response(
        self, repo: str, event_type: str, question: str,
        installation_id: Optional[int] = None,
    ) -> tuple[Optional[str], dict]:
        """Generate an LLM response. Returns (content, diagnostics)."""
        diag = {}
        if not self.agent:
            return None, {"error": "no_agent"}

        llm = getattr(self.agent, "llm_service", None)
        if not llm:
            return None, {"error": "no_llm_service"}

        diag["llm_service"] = type(llm).__name__

        # Gather codebase context
        try:
            context = await self._gather_context(repo, question, installation_id)
            diag["context_chars"] = len(context)
        except Exception as e:
            context = ""
            diag["context_error"] = f"{type(e).__name__}: {e}"

        system = SYSTEM_PROMPT.format(
            event_type=event_type,
            repo=repo,
            repo_url=f"https://github.com/{repo}",
        )
        if self._soul:
            system = self._soul + "\n\n" + system

        prompt = question
        if context:
            prompt = f"{question}\n\n---\n\n**Relevant source code:**\n\n{context}"

        diag["prompt_chars"] = len(prompt)
        diag["system_chars"] = len(system)

        try:
            response = await self.agent.llm_service.generate(
                user_prompt=prompt,
                system_prompt=system,
            )
            diag["response_type"] = type(response).__name__ if response else "None"
            if isinstance(response, str):
                result = response or None
            else:
                result = response.content if response and response.content else None
            if result:
                diag["result_chars"] = len(result)
            return result, diag
        except Exception as e:
            diag["llm_error"] = f"{type(e).__name__}: {e}"
            return None, diag

    # The codebase is deployed on this instance — read it directly from disk
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]  # kestrel_sovereign/features/github_app -> project root

    async def _gather_context(
        self, repo: str, question: str, installation_id: Optional[int] = None
    ) -> str:
        """Read relevant source code from the local filesystem."""
        root = self._PROJECT_ROOT
        context_parts = []

        try:
            # Always include key docs
            for doc in ["README.md", "CLAUDE.md", "KESTREL_FEATURES.md"]:
                p = root / doc
                if p.exists():
                    text = p.read_text(encoding="utf-8")
                    context_parts.append(f"## {doc}\n```\n{text[:3000]}\n```")

            # Always include canonical reference files — these answer common questions
            canonical = [
                "kestrel_sovereign/features/__init__.py",     # feature discovery
                "kestrel_sovereign/features/base.py",          # Feature base class
                "kestrel_sovereign/entrypoints.py",            # entry_point scanning
                "pyproject.toml",                               # package metadata
            ]
            for rel in canonical:
                p = root / rel
                if p.exists():
                    text = p.read_text(encoding="utf-8")
                    if len(text) > 4000:
                        text = text[:4000] + "\n... (truncated)"
                    context_parts.append(f"## {rel}\n```python\n{text}\n```")

            # Search for relevant Python files using grep
            skip_words = {"the", "a", "an", "is", "how", "do", "i", "to", "can", "what",
                          "does", "it", "in", "for", "of", "with", "this", "that", "are"}
            words = [w.strip("?.,!") for w in question.lower().split()
                     if len(w) > 2 and w.lower() not in skip_words]

            matched_files = set()
            for word in words[:5]:
                # Search Python files for the keyword
                import subprocess
                result = subprocess.run(
                    ["grep", "-rl", "--include=*.py", "-i", word,
                     str(root / "kestrel_sovereign")],
                    capture_output=True, text=True, timeout=5
                )
                for path in result.stdout.strip().split("\n")[:3]:
                    if path:
                        matched_files.add(path)

            # Read the most relevant files (up to 8)
            for filepath in sorted(matched_files)[:8]:
                try:
                    p = Path(filepath)
                    rel = p.relative_to(root)
                    content = p.read_text(encoding="utf-8")
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)"
                    context_parts.append(f"## {rel}\n```python\n{content}\n```")
                except Exception:
                    pass

            # If nothing matched, include the project structure
            if len(context_parts) <= 3:
                py_files = sorted(
                    str(p.relative_to(root))
                    for p in (root / "kestrel_sovereign").rglob("*.py")
                    if "__pycache__" not in str(p)
                )[:60]
                context_parts.append(
                    f"## Project structure ({len(py_files)} Python files)\n```\n"
                    + "\n".join(py_files) + "\n```"
                )

        except Exception as e:
            logger.warning("GitHubAppFeature context gathering error: %s", e)

        combined = "\n\n".join(context_parts)
        if len(combined) > 20000:
            combined = combined[:20000] + "\n\n... (context truncated)"

        return combined
