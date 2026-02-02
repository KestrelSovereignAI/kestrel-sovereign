"""Claude Agent SDK integration for autonomous code execution."""

import asyncio
import re
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from claude_agent_sdk import ClaudeAgentOptions, query

from .config import GitHubProcessorConfig
from .models import ClarificationOption, ClarificationQuestion, ClarificationRequest, IssueContext


@dataclass
class AgentResult:
    """Result from agent execution."""

    success: bool
    blocked: bool = False
    blocking_question: Optional[str] = None
    error: Optional[str] = None
    session_id: Optional[str] = None
    output: str = ""


@dataclass
class AnalysisResult:
    """Result from issue analysis phase."""

    ready_to_implement: bool
    clarification_request: Optional[ClarificationRequest] = None
    implementation_plan: Optional[str] = None
    error: Optional[str] = None


class AgentRunner:
    """Runs Claude Agent SDK for autonomous code execution."""

    def __init__(self, config: GitHubProcessorConfig):
        self.config = config

    def _build_prompt(self, context: IssueContext) -> str:
        """Build the prompt for Claude from issue context."""
        return f"""You are an autonomous coding agent processing a GitHub issue. Your task is to fully implement the requested changes.

{context.format_for_prompt()}

## Instructions

1. **Understand the requirements**: Read the issue carefully. If file paths are mentioned, read those files first.

2. **Implement the changes**: Make all necessary code changes to fulfill the request. This may involve:
   - Reading existing code to understand the codebase
   - Creating new files
   - Modifying existing files
   - Running commands to test your changes

3. **Commit your work**: After making changes, commit with a clear message that references issue #{context.number}.
   Use: `git add -A && git commit -m "Your message (#{context.number})"`

4. **Handle blocking issues**: If you genuinely cannot proceed without human input:
   - Output exactly: `BLOCKED: <your specific question>`
   - Only do this if you truly cannot make progress
   - Be specific about what information you need

5. **Quality standards**:
   - Follow existing code patterns and style
   - Don't introduce security vulnerabilities
   - Keep changes focused - don't refactor unrelated code
   - Don't add unnecessary comments or documentation

## Important

- Work autonomously. Make decisions and implement changes without asking for permission.
- You have full access to read files, write files, edit files, and run bash commands.
- If tests fail, try to fix them. Only report as blocked if you've tried and can't fix.
- Commit frequently with meaningful messages.
- Your commits will be pushed and CI will run automatically.

Begin by reading any referenced files to understand the current state, then implement the requested changes.
"""

    async def run(self, context: IssueContext) -> AgentResult:
        """Run the agent to process an issue."""
        prompt = self._build_prompt(context)
        output_lines: list[str] = []
        session_id: Optional[str] = None
        blocked = False
        blocking_question: Optional[str] = None
        error: Optional[str] = None

        try:
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                allowed_tools=[
                    "Read",
                    "Write",
                    "Edit",
                    "Bash",
                    "Glob",
                    "Grep",
                    "WebSearch",
                    "WebFetch",
                ],
                model=self.config.model,
                max_turns=self.config.max_turns,
                cwd=self.config.worktree_path,  # Run in worktree if set
            )

            async for message in query(prompt=prompt, options=options):
                # Capture session ID from init message
                if hasattr(message, "subtype") and message.subtype == "init":
                    session_id = getattr(message, "session_id", None)

                # Capture result/output
                if hasattr(message, "result"):
                    output_lines.append(str(message.result))
                elif hasattr(message, "content"):
                    output_lines.append(str(message.content))

            # Check output for BLOCKED marker
            full_output = "\n".join(output_lines)
            blocked_match = re.search(r"BLOCKED:\s*(.+?)(?:\n|$)", full_output, re.IGNORECASE)
            if blocked_match:
                blocked = True
                blocking_question = blocked_match.group(1).strip()

            return AgentResult(
                success=not blocked,
                blocked=blocked,
                blocking_question=blocking_question,
                session_id=session_id,
                output=full_output,
            )

        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e),
                output="\n".join(output_lines),
            )

    async def analyze_issue(self, context: IssueContext) -> AnalysisResult:
        """Analyze an issue to determine if clarification is needed before implementation.

        Returns an AnalysisResult with either:
        - ready_to_implement=True and an implementation_plan
        - ready_to_implement=False and a clarification_request with structured questions
        """
        prompt = f"""You are analyzing a GitHub issue to determine if it's ready for implementation or needs clarification first.

{context.format_for_prompt()}

## Your Task

Analyze this issue and determine:

1. **Is the issue clear enough to implement?** Consider:
   - Are the requirements specific and unambiguous?
   - Is the scope well-defined?
   - Are there multiple valid approaches that need a decision?
   - Are there missing details that would block implementation?

2. **If clarification is needed**, identify 1-4 specific questions. For each question:
   - Give it a short ID (e.g., SCOPE, APPROACH, DEPENDENCY)
   - Provide 2-4 concrete options when possible
   - Be specific, not vague

## Output Format

If READY to implement, output:
```
READY_TO_IMPLEMENT

## Implementation Plan
[Brief plan of what you would do]
```

If NEEDS CLARIFICATION, output:
```
NEEDS_CLARIFICATION

QUESTION[ID]: <question text>
- OPTION: <option 1>
- OPTION: <option 2>
[repeat for each question, max 4 questions]
```

Example clarification output:
```
NEEDS_CLARIFICATION

QUESTION[SCOPE]: Should rate limiting apply to all endpoints or just auth endpoints?
- OPTION: All API endpoints
- OPTION: Auth endpoints only (login, register, password reset)
- OPTION: Auth + payment endpoints

QUESTION[BACKEND]: Which rate limiting backend should we use?
- OPTION: Redis (recommended for production)
- OPTION: In-memory (simpler, good for dev)
```

Analyze the issue now and provide your assessment.
"""

        try:
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                allowed_tools=["Read", "Glob", "Grep"],  # Read-only for analysis
                model=self.config.model,
                max_turns=10,  # Quick analysis, not full implementation
                cwd=self.config.worktree_path,  # Run in worktree if set
            )

            output_lines: list[str] = []
            async for message in query(prompt=prompt, options=options):
                if hasattr(message, "result"):
                    output_lines.append(str(message.result))
                elif hasattr(message, "content"):
                    output_lines.append(str(message.content))

            full_output = "\n".join(output_lines)

            # Normalize escaped newlines to actual newlines
            full_output = full_output.replace("\\n", "\n")

            # Parse the output
            if "READY_TO_IMPLEMENT" in full_output:
                # Extract implementation plan
                plan_match = re.search(r"## Implementation Plan\s*\n(.+?)(?:\n```|$)", full_output, re.DOTALL)
                plan = plan_match.group(1).strip() if plan_match else "Implementation plan not provided"
                return AnalysisResult(ready_to_implement=True, implementation_plan=plan)

            elif "NEEDS_CLARIFICATION" in full_output:
                # Parse questions - use dict to dedupe by ID
                questions_dict: dict[str, ClarificationQuestion] = {}
                question_pattern = r"QUESTION\[(\w+)\]:\s*(.+?)(?=QUESTION\[|$)"
                for match in re.finditer(question_pattern, full_output, re.DOTALL):
                    q_id = match.group(1)

                    # Skip if we already have this question ID
                    if q_id in questions_dict:
                        continue

                    q_block = match.group(2).strip()

                    # Extract question text (first line)
                    q_lines = q_block.split("\n")
                    q_text = q_lines[0].strip()

                    # Extract options
                    options_list = []
                    for line in q_lines[1:]:
                        opt_match = re.match(r"- OPTION:\s*(.+)", line.strip())
                        if opt_match:
                            opt_text = opt_match.group(1).strip()
                            # Clean artifacts from agent output serialization
                            # Remove trailing quotes and brackets (but preserve balanced parens)
                            opt_text = re.sub(r'[\"\'\]]+$', '', opt_text).strip()
                            # Remove trailing ) only if unbalanced
                            open_count = opt_text.count('(')
                            close_count = opt_text.count(')')
                            while close_count > open_count and opt_text.endswith(')'):
                                opt_text = opt_text[:-1].strip()
                                close_count -= 1
                            if opt_text:
                                options_list.append(ClarificationOption(label=opt_text))

                    questions_dict[q_id] = ClarificationQuestion(
                        id=q_id,
                        question=q_text,
                        options=options_list,
                    )

                questions = list(questions_dict.values())
                if questions:
                    return AnalysisResult(
                        ready_to_implement=False,
                        clarification_request=ClarificationRequest(questions=questions),
                    )

            # Fallback: assume ready if we can't parse
            return AnalysisResult(ready_to_implement=True, implementation_plan="Analysis inconclusive, proceeding with implementation")

        except Exception as e:
            return AnalysisResult(ready_to_implement=False, error=str(e))

    async def run_fix_ci(self, context: IssueContext, failure_summary: str) -> AgentResult:
        """Run the agent to fix CI failures."""
        prompt = f"""The CI pipeline failed for issue #{context.number}. Fix the failures.

## CI Failure Summary
{failure_summary}

## Original Issue
{context.format_for_prompt()}

## Instructions

1. Analyze the CI failure output
2. Identify what's broken (tests, linting, type errors, etc.)
3. Fix the issues in the code
4. Commit your fixes with message: "Fix CI failures (#{context.number})"

If you cannot fix the issue after reasonable attempts, output:
`BLOCKED: <description of what's failing and what you tried>`

Begin by understanding the failure, then implement fixes.
"""

        return await self._run_with_prompt(prompt)

    async def _run_with_prompt(self, prompt: str) -> AgentResult:
        """Run agent with a specific prompt."""
        output_lines: list[str] = []
        session_id: Optional[str] = None

        try:
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                allowed_tools=[
                    "Read",
                    "Write",
                    "Edit",
                    "Bash",
                    "Glob",
                    "Grep",
                ],
                model=self.config.model,
                max_turns=self.config.max_turns,
                cwd=self.config.worktree_path,  # Run in worktree if set
            )

            async for message in query(prompt=prompt, options=options):
                if hasattr(message, "subtype") and message.subtype == "init":
                    session_id = getattr(message, "session_id", None)

                if hasattr(message, "result"):
                    output_lines.append(str(message.result))
                elif hasattr(message, "content"):
                    output_lines.append(str(message.content))

            full_output = "\n".join(output_lines)
            blocked_match = re.search(r"BLOCKED:\s*(.+?)(?:\n|$)", full_output, re.IGNORECASE)

            if blocked_match:
                return AgentResult(
                    success=False,
                    blocked=True,
                    blocking_question=blocked_match.group(1).strip(),
                    session_id=session_id,
                    output=full_output,
                )

            return AgentResult(
                success=True,
                session_id=session_id,
                output=full_output,
            )

        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e),
                output="\n".join(output_lines),
            )
