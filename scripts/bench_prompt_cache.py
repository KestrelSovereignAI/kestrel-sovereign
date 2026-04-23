#!/usr/bin/env python3
"""Benchmark prompt-cache behavior against a running llama-server.

Issues a multi-turn conversation against http://localhost:8001/v1 and reports
wall-clock time-to-first-token for each turn.  A turn 2+ that takes roughly
the same time as turn 1 means the cache is MISSING and the full prompt is
being reprocessed every turn.  A turn 2+ substantially faster than turn 1
means the cache is HITTING and only the new tokens are being prefilled.

Usage:
    uv run python scripts/bench_prompt_cache.py [--base-url URL] [--turns N]

Exits non-zero if turn-2 TTFT is not meaningfully below turn-1 TTFT (the
ticket #703 acceptance threshold is ≥80% reduction, but this script reports
the raw numbers so CI can decide the exact threshold).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import List

import httpx


def build_stable_system_prompt() -> str:
    """Simulate Kestrel's stable system prefix.  The content is immaterial;
    what matters is that the same string is sent every turn.
    """
    constitution = "You are an agent governed by the Kestrel Constitution.\n"
    constitution += "Principles:\n"
    constitution += "- Sovereignty of the user\n"
    constitution += "- Honesty and calibrated uncertainty\n"
    constitution += "- Fidelity to the user's instructions\n"
    constitution += "- Freedom of thought\n"
    constitution += "- Data sanctity\n"
    # Pad so the prefix is non-trivially sized (closer to real agent runtime
    # where system prompt + features + anti-injection can be thousands of
    # tokens).
    padding = "".join(
        f"- Directive {i}: Consider the long-term impact of every action.\n"
        for i in range(200)
    )
    return constitution + padding


async def one_turn(
    client: httpx.AsyncClient,
    base_url: str,
    messages: List[dict],
    *,
    max_tokens: int = 32,
) -> tuple[float, int, dict]:
    """Send one chat completion; return (wall_clock_seconds, generated_tokens,
    full_response_json).
    """
    start = time.perf_counter()
    resp = await client.post(
        f"{base_url}/chat/completions",
        json={
            "model": "auto",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=300.0,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return elapsed, tokens, data


async def run_benchmark(base_url: str, turns: int) -> int:
    """Run `turns` sequential chat completions, share the same system prefix
    and grow the history append-only.  Print per-turn timings.
    """
    system_prompt = build_stable_system_prompt()
    print(f"System prompt length: {len(system_prompt)} chars "
          f"(~{len(system_prompt) // 4} tokens)")
    print(f"Running {turns} turns against {base_url}\n")

    history: List[dict] = []
    timings: List[float] = []

    async with httpx.AsyncClient() as client:
        for i in range(1, turns + 1):
            user_content = (
                f"Turn {i}: give me a one-sentence reply, nothing else. "
                f"What is {i} plus {i}?"
            )
            messages = (
                [{"role": "system", "content": system_prompt}]
                + history
                + [{"role": "user", "content": user_content}]
            )
            elapsed, completion_tokens, data = await one_turn(
                client, base_url, messages
            )
            timings.append(elapsed)
            usage = data.get("usage", {})
            print(
                f"Turn {i}: {elapsed:.2f}s wall-clock | "
                f"prompt_tokens={usage.get('prompt_tokens', '?')} | "
                f"cached_tokens={usage.get('cached_tokens', usage.get('prompt_tokens_details', {}).get('cached_tokens', '?'))} | "
                f"completion_tokens={completion_tokens}"
            )

            # Append to history for next turn (raw, the way Kestrel stores).
            assistant_content = (
                data.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            history.append({"role": "user", "content": user_content})
            history.append({"role": "assistant", "content": assistant_content})

    print()
    if len(timings) >= 2:
        t1 = timings[0]
        t2_plus = timings[1:]
        mean_t2plus = statistics.mean(t2_plus)
        reduction = (t1 - mean_t2plus) / t1 if t1 > 0 else 0.0
        print(
            f"Turn 1: {t1:.2f}s | "
            f"Turns 2+: mean={mean_t2plus:.2f}s | "
            f"reduction={reduction:.1%}"
        )
        if reduction < 0.0:
            print(
                "WARN: turns 2+ are SLOWER than turn 1. Cache appears to miss; "
                "investigate system-prompt stability and slot assignment."
            )
            return 2
        if reduction < 0.5:
            print(
                "WARN: reduction below 50%; cache hit is weak. Check whether "
                "the system prompt is byte-stable across turns."
            )
            return 1
        print("OK: turn 2+ substantially faster than turn 1; cache is hitting.")
        return 0
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-url",
        default="http://localhost:8001/v1",
        help="OpenAI-compatible base URL (default: local llama-server)",
    )
    p.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Number of conversation turns (default: 3)",
    )
    args = p.parse_args()
    return asyncio.run(run_benchmark(args.base_url, args.turns))


if __name__ == "__main__":
    sys.exit(main())
