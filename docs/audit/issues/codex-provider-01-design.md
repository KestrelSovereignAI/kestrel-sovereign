# Add a first-slice design for an `openai_plan` provider in `kestrel-sovereign`

## Problem

The ecosystem now has a working OpenAI plan execution lane in `kestrel-talon`, but `kestrel-sovereign` still treats OpenAI, Anthropic, Claude plan, Ollama, and others as first-class runtime providers while the OpenAI plan lane remains external.

If the OpenAI plan lane is going to become a first-class sovereign runtime option, the codebase needs a small design truth first rather than an ad hoc subprocess hack.

## Goal

Write the first-slice design for an `openai_plan` provider path in the sovereign LLM runtime.

## Scope

- define what an `openai_plan` provider means in sovereign terms
- define how it differs from:
  - `openai`
  - `anthropic`
  - `claude_plan`
- define the auth/session model assumptions
- define what the first implementation should and should not do
- identify where it plugs into provider registry, adapter layer, and model preference flow

## Suggested output

- `docs/architecture/OPENAI_PLAN_PROVIDER.md`

## Acceptance criteria

- a canonical design doc exists
- the doc names concrete files and integration points
- the first slice is explicitly scoped to text/runtime use, not all possible plan-client features

## References

- `kestrel_sovereign/llm/provider_registry.py`
- `kestrel_sovereign/llm/openai_adapter.py`
- `kestrel_sovereign/llm/claude_max_adapter.py`
- `kestrel-talon` OpenAI plan backend work in sibling repo

## Talon note

Do not implement the provider in this issue. This is the architecture-setting first move.
