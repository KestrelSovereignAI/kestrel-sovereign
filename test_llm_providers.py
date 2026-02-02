#!/usr/bin/env python3
"""Test LLM providers: OpenAI, Anthropic, and Vertex AI (Gemini)."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()


async def test_providers():
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService()

    system_prompt = "You are a helpful assistant. Be concise."
    test_prompt = "What is 2+2? Answer with just the number."

    # Test using model_override which matches against provider["model"] or provider["name"]
    providers_to_test = [
        ("openai", "gpt-5.1"),
        ("anthropic", "claude-opus-4-5-20251101"),
        ("vertex_ai", "gemini-3-pro-preview"),  # Provider name matching
    ]

    print("=" * 60)
    print("LLM Provider Test")
    print("=" * 60)
    print(f"Initialized providers: {[p['name'] for p in service.providers]}")

    results = []
    for provider_name, model in providers_to_test:
        print(f"\nTesting {provider_name} ({model})...")
        try:
            # Use model_override to target specific provider
            # The service will match against provider["model"] or provider["name"]
            response = await service.get_response(
                system_prompt=system_prompt,
                user_prompt=test_prompt,
                model_override=provider_name,  # Match by provider name
            )
            content = str(response)[:100] if response else "No content"
            print(f"  {provider_name}: {content}")
            results.append((provider_name, True, content))
        except Exception as e:
            print(f"  {provider_name}: {type(e).__name__}: {e}")
            results.append((provider_name, False, str(e)))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for provider_name, success, msg in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {provider_name}")

    passed = sum(1 for _, s, _ in results if s)
    print(f"\n{passed}/{len(results)} providers working")


if __name__ == "__main__":
    asyncio.run(test_providers())
