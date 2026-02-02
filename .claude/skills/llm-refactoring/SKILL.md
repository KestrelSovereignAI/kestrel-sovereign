---
name: llm-refactoring
description: Use when working on LLM service refactoring, adding streaming support, structured output, vision capabilities, or provider routing. Automatically delegates to llm-service-specialist subagent for Kestrel LLM improvements.
---

# LLM Service Refactoring Skill

This skill automatically activates when detecting work on Kestrel's LLM service layer.

## Trigger Keywords
- "LLM service", "llm service", "streaming", "structured output"
- "vision support", "image handling", "provider routing"
- "OpenAI adapter", "Ollama adapter", "Anthropic adapter"
- "llm/service.py", "llm/adapter.py", "LLMServiceRouter"
- "Pydantic models", "response_format"

## What This Skill Does

When activated, this skill:
1. Detects you're working on LLM service improvements
2. Suggests using the **llm-service-specialist** subagent
3. Provides guidance on Kestrel LLM architecture

## Quick Guidance

### Key Capabilities to Implement
1. **Streaming** - Both structured (Pydantic) and unstructured (text)
2. **Vision** - Multiple image formats (path, base64, bytes)
3. **Provider Routing** - Intelligent model selection
4. **Structured Output** - Pydantic model support
5. **Privacy Integration** - Respect Kestrel privacy modes

### Kestrel-Specific Considerations
- Maintain sovereignty principles
- Respect privacy modes (EPHEMERAL = local only)
- Integrate with constitutional governance
- Support model mandate system

## Recommendation

Use the llm-service-specialist subagent for focused LLM work:
```
"Use the llm-service-specialist subagent to refactor the LLM service"
```

Or use the parallel workflow:
```
/parallel-work
```
