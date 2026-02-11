"""
Constitutional Council Deliberation Protocol

Multi-round deliberation where foundation models analyze evidence,
discuss with each other, and reach consensus.

Key features:
- Model-agnostic: any provider/model combination works
- Parallel invocation: models respond simultaneously
- Multi-round discussion: models see each other's reasoning
- Structured verdicts: typed responses for consensus checking
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sovereign.kestrel_config.defaults import (
    get_ollama_url,
    get_xai_api_url,
    get_groq_api_url,
)

from .models import (
    CouncilMember,
    CouncilSession,
    ConsensusRule,
    Decision,
    DeliberationRound,
    Evidence,
    SessionOutcome,
    Verdict,
)

logger = logging.getLogger(__name__)

# System prompt for council members
COUNCIL_SYSTEM_PROMPT = """You are a member of the Constitutional Council for Kestrel, a sovereign AI agent framework.

Your role: {role}

You are deliberating on an important decision that requires careful analysis. You will be presented with:
1. A question to answer
2. Evidence to review
3. (In later rounds) Other council members' perspectives

Your responsibilities:
- Analyze the evidence thoroughly
- Consider risks, benefits, and alternatives
- Be honest about concerns, even if approving
- Provide clear reasoning for your position
- Engage constructively with other perspectives

When voting, you MUST respond with a structured verdict in this exact JSON format:
```json
{{
    "decision": "APPROVE" | "REJECT" | "ABSTAIN",
    "confidence": 0.0-1.0,
    "reasoning": "Your detailed reasoning",
    "concerns": ["List of concerns even if approving"],
    "conditions": ["List of conditions for approval, if any"]
}}
```

Remember: This is about making the right decision, not the easy one. Dissent is valuable."""


DELIBERATION_PROMPT = """## Council Deliberation: {question}

### Your Role
{role}

### Evidence Package
{evidence}

{previous_round}

### Instructions
{instructions}

Provide your analysis and, if this is the final round, your structured verdict."""


async def convene_council(
    question: str,
    evidence: Evidence,
    members: List[CouncilMember],
    max_rounds: int = 3,
    consensus_rule: ConsensusRule = ConsensusRule.UNANIMOUS,
) -> CouncilSession:
    """
    Convene the Constitutional Council for deliberation.

    Args:
        question: The question to deliberate on
        evidence: Evidence package for review
        members: List of council members (models) to participate
        max_rounds: Maximum deliberation rounds before forcing verdict
        consensus_rule: Rule for determining consensus

    Returns:
        Complete CouncilSession with all rounds and verdicts
    """
    if len(members) < 2:
        raise ValueError("Council requires at least 2 members")

    session = CouncilSession(
        question=question,
        evidence=evidence,
        members=members,
        consensus_rule=consensus_rule,
    )

    logger.info(
        f"Convening council session {session.id} with {len(members)} members"
    )

    # Initialize adapters for each member
    adapters = await _initialize_adapters(members)

    # Round 1: Independent analysis
    round1 = session.add_round()
    await _run_deliberation_round(
        round=round1,
        members=members,
        adapters=adapters,
        question=question,
        evidence=evidence,
        session=session,
        is_first_round=True,
        is_final_round=(max_rounds == 1),
    )

    # Subsequent rounds: See others' reasoning
    for round_num in range(2, max_rounds + 1):
        is_final = (round_num == max_rounds)
        new_round = session.add_round()
        await _run_deliberation_round(
            round=new_round,
            members=members,
            adapters=adapters,
            question=question,
            evidence=evidence,
            session=session,
            previous_rounds=session.rounds[:-1],  # All but current
            is_first_round=False,
            is_final_round=is_final,
        )

        # Check if we have consensus from the messages
        if is_final:
            # Extract verdicts from final round
            for msg in new_round.messages:
                verdict = _extract_verdict(
                    msg.member_name,
                    msg.model,
                    msg.content
                )
                if verdict:
                    session.add_verdict(verdict)

    # If we didn't get verdicts from discussion, request them explicitly
    if not session.verdicts:
        verdicts_round = session.add_round()
        await _request_verdicts(
            round=verdicts_round,
            members=members,
            adapters=adapters,
            question=question,
            evidence=evidence,
            previous_rounds=session.rounds[:-1],
            session=session,
        )
        for msg in verdicts_round.messages:
            verdict = _extract_verdict(
                msg.member_name,
                msg.model,
                msg.content
            )
            if verdict:
                session.add_verdict(verdict)

    # Determine outcome
    session._update_outcome()

    logger.info(
        f"Council session {session.id} completed with outcome: {session.outcome}"
    )

    return session


async def _initialize_adapters(
    members: List[CouncilMember]
) -> Dict[str, Tuple[Any, Any]]:
    """
    Initialize LLM adapters for each council member.

    Returns dict mapping member name to (client, adapter) tuple.
    """
    adapters = {}

    for member in members:
        try:
            client, adapter = await _get_adapter_for_provider(
                member.provider,
                member.model
            )
            adapters[member.name] = (client, adapter, member.model)
            logger.info(f"Initialized adapter for {member.name} ({member.provider})")
        except Exception as e:
            logger.error(f"Failed to initialize {member.name}: {e}")
            raise RuntimeError(
                f"Could not initialize council member {member.name}: {e}"
            )

    return adapters


async def _get_adapter_for_provider(
    provider: str,
    model: str
) -> Tuple[Any, Any]:
    """Get the appropriate client and adapter for a provider."""

    if provider == "openai":
        import openai
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = openai.AsyncOpenAI(api_key=api_key)
        adapter = OpenAIAdapter()
        return client, adapter

    elif provider == "anthropic":
        import anthropic
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        adapter = AnthropicAdapter()
        return client, adapter

    elif provider == "google":
        import google.generativeai as genai
        from kestrel_sovereign.llm.google_adapter import GoogleAdapter

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model)
        adapter = GoogleAdapter()
        return client, adapter

    elif provider == "vertex_ai":
        from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter

        adapter = VertexAIAdapter()
        # Vertex uses internal client
        return None, adapter

    elif provider == "ollama":
        import ollama
        from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter

        # Use get_ollama_url() for canonical URL resolution
        # Support legacy OLLAMA_HOST env var for backwards compatibility
        host = os.environ.get("OLLAMA_HOST") or get_ollama_url()
        client = ollama.AsyncClient(host=host)
        adapter = OllamaAdapter()
        return client, adapter

    elif provider == "xai":
        import openai
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not set")

        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=get_xai_api_url()
        )
        adapter = OpenAIAdapter()
        return client, adapter

    elif provider == "groq":
        import openai
        from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=get_groq_api_url()
        )
        adapter = OpenAIAdapter()
        return client, adapter

    else:
        raise ValueError(f"Unknown provider: {provider}")


async def _run_deliberation_round(
    round: DeliberationRound,
    members: List[CouncilMember],
    adapters: Dict[str, Tuple[Any, Any, str]],
    question: str,
    evidence: Evidence,
    session: CouncilSession,
    previous_rounds: Optional[List[DeliberationRound]] = None,
    is_first_round: bool = True,
    is_final_round: bool = False,
) -> None:
    """Run a single round of deliberation with all members in parallel."""

    # Build previous round transcript
    previous_text = ""
    if previous_rounds:
        transcripts = [r.to_transcript() for r in previous_rounds]
        previous_text = "\n\n".join(transcripts)

    # Build instructions based on round
    if is_first_round:
        instructions = (
            "This is the first round. Analyze the evidence independently. "
            "Share your initial thoughts and key observations."
        )
    elif is_final_round:
        instructions = (
            "This is the FINAL round. You must now provide your verdict. "
            "Include your structured JSON verdict at the end of your response."
        )
    else:
        instructions = (
            "Review the other council members' perspectives. "
            "Respond to their points and refine your position."
        )

    # Create tasks for parallel invocation
    tasks = []
    for member in members:
        client, adapter, model = adapters[member.name]
        task = _invoke_member(
            member=member,
            client=client,
            adapter=adapter,
            model=model,
            question=question,
            evidence=evidence,
            previous_round=previous_text,
            instructions=instructions,
        )
        tasks.append(task)

    # Run all invocations in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Add messages to round and track token usage
    for member, result in zip(members, results):
        if isinstance(result, Exception):
            logger.error(f"Member {member.name} failed: {result}")
            content = f"[Error: {str(result)}]"
            input_tokens = 0
            output_tokens = 0
        else:
            content, input_tokens, output_tokens = result
            # Record token usage
            session.add_token_usage(
                member_name=member.name,
                provider=member.provider,
                model=member.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                round_number=round.round_number,
            )

        round.add_message(
            member_name=member.name,
            model=member.model,
            content=content
        )


async def _invoke_member(
    member: CouncilMember,
    client: Any,
    adapter: Any,
    model: str,
    question: str,
    evidence: Evidence,
    previous_round: str,
    instructions: str,
) -> Tuple[str, int, int]:
    """
    Invoke a single council member for their response.

    Returns:
        Tuple of (content, input_tokens, output_tokens)
    """

    system_prompt = COUNCIL_SYSTEM_PROMPT.format(role=member.role)

    user_prompt = DELIBERATION_PROMPT.format(
        question=question,
        role=member.role,
        evidence=evidence.to_prompt(),
        previous_round=(
            f"### Previous Discussion\n{previous_round}"
            if previous_round
            else ""
        ),
        instructions=instructions,
    )

    messages = adapter.create_messages(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )

    try:
        response = await adapter.get_response(
            client=client,
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
        )

        # Extract token counts
        input_tokens = getattr(response, 'input_tokens', None) or 0
        output_tokens = getattr(response, 'output_tokens', None) or 0

        # Extract content
        if hasattr(response, 'content') and response.content:
            content = response.content
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)

        return (content, input_tokens, output_tokens)

    except Exception as e:
        logger.error(f"Failed to invoke {member.name}: {e}")
        raise


async def _request_verdicts(
    round: DeliberationRound,
    members: List[CouncilMember],
    adapters: Dict[str, Tuple[Any, Any, str]],
    question: str,
    evidence: Evidence,
    previous_rounds: List[DeliberationRound],
    session: CouncilSession,
) -> None:
    """Request explicit verdicts from all members."""

    previous_text = "\n\n".join(r.to_transcript() for r in previous_rounds)

    verdict_prompt = f"""## Final Verdict Required

Based on the deliberation above, you must now provide your final verdict.

Question: {question}

Respond ONLY with a JSON verdict in this exact format:
```json
{{
    "decision": "APPROVE" | "REJECT" | "ABSTAIN",
    "confidence": 0.0-1.0,
    "reasoning": "Your detailed reasoning",
    "concerns": ["List of concerns"],
    "conditions": ["List of conditions, if any"]
}}
```"""

    tasks = []
    for member in members:
        client, adapter, model = adapters[member.name]
        task = _invoke_member(
            member=member,
            client=client,
            adapter=adapter,
            model=model,
            question=question,
            evidence=evidence,
            previous_round=previous_text,
            instructions=verdict_prompt,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for member, result in zip(members, results):
        if isinstance(result, Exception):
            content = f"[Error: {str(result)}]"
        else:
            content, input_tokens, output_tokens = result
            # Record token usage
            session.add_token_usage(
                member_name=member.name,
                provider=member.provider,
                model=member.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                round_number=round.round_number,
            )

        round.add_message(
            member_name=member.name,
            model=member.model,
            content=content
        )


def _extract_verdict(
    member_name: str,
    model: str,
    content: str
) -> Optional[Verdict]:
    """Extract structured verdict from model response."""

    # Try to find JSON block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            logger.warning(f"No verdict JSON found in {member_name}'s response")
            return None

    try:
        data = json.loads(json_str)

        decision_str = data.get("decision", "").upper()
        try:
            decision = Decision(decision_str)
        except ValueError:
            logger.warning(f"Invalid decision '{decision_str}' from {member_name}")
            return None

        return Verdict(
            member_name=member_name,
            model=model,
            decision=decision,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            concerns=data.get("concerns", []),
            conditions=data.get("conditions", []),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse verdict JSON from {member_name}: {e}")
        return None


def apply_human_override(
    session: CouncilSession,
    override_decision: str,
    reason: str
) -> None:
    """Apply a human override to a council session."""
    session.human_override = f"{override_decision}: {reason}"
    if override_decision.upper() == "APPROVE":
        session.outcome = SessionOutcome.APPROVED
    elif override_decision.upper() == "REJECT":
        session.outcome = SessionOutcome.REJECTED
    session.completed_at = datetime.utcnow()
    logger.info(f"Human override applied to session {session.id}: {override_decision}")
