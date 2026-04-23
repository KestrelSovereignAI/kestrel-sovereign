#!/usr/bin/env python3
"""Multi-provider prompt-cache benchmark for issue #703.

Runs a controlled multi-turn conversation against every provider the user
has configured (API key present / endpoint reachable) and reports both
wall-clock TTFT and whatever cache metric the provider exposes in its
response.  The messages sent mimic what Kestrel emits after #703 —
stable system, wrapped historical user turns, per-turn retrieved-context
block on the current turn — so the numbers reflect what a real Kestrel
conversation would see against each backend.

Providers probed:
    * llama.cpp         — local OpenAI-compatible, default :8001
    * ollama            — local OpenAI-compatible, default :11434/v1
    * openai            — cloud (needs OPENAI_API_KEY)
    * anthropic         — cloud (needs ANTHROPIC_API_KEY)
    * openrouter        — cloud (needs OPENROUTER_API_KEY)

Cache signals surfaced (per provider, as they report):
    * llama.cpp   — no cache field in response; relies on TTFT drop
    * ollama      — prompt_eval_count drops on cache hits (if reported)
    * openai      — usage.prompt_tokens_details.cached_tokens
    * anthropic   — usage.cache_read_input_tokens /
                    cache_creation_input_tokens  (will be zero until #705
                    adds cache_control markers)
    * openrouter  — varies by underlying model

Usage:
    uv run python scripts/bench_prompt_cache_providers.py            # all
    uv run python scripts/bench_prompt_cache_providers.py --only openai,anthropic
    uv run python scripts/bench_prompt_cache_providers.py --turns 5

Exit code: non-zero if ANY probed provider shows no meaningful cache signal
            on turns 2+ (treating "no TTFT reduction and no cache tokens"
            as a failure).  Skipped providers (no key / unreachable) do
            not count.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import httpx


STABLE_SYSTEM = (
    "You are a benchmark assistant.  Reply with a single short sentence.\n"
    "Constitution padding follows so the system prefix is large enough to\n"
    "exceed the 1024-token floor that OpenAI's prefix cache requires:\n"
    + "".join(
        f"- Principle {i}: behave consistently and deterministically.\n"
        for i in range(220)
    )
)


@dataclass
class TurnResult:
    turn: int
    wall_seconds: float
    prompt_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    notes: str = ""


@dataclass
class ProviderResult:
    name: str
    status: str                         # "ok" | "skipped" | "error"
    reason: str = ""
    turns: List[TurnResult] = field(default_factory=list)

    def speedup(self) -> Optional[float]:
        if len(self.turns) < 2:
            return None
        t1 = self.turns[0].wall_seconds
        rest_mean = statistics.mean(t.wall_seconds for t in self.turns[1:])
        if t1 <= 0:
            return None
        return (t1 - rest_mean) / t1

    def max_cached_tokens(self) -> int:
        return max((t.cached_tokens or 0) for t in self.turns)


def _build_messages(
    history: list, user_content_template: str, query: str, include_ctx: bool
) -> list:
    """Kestrel-shaped messages for one turn after issue #703.  Current-user
    content includes a <retrieved_context> block only when include_ctx is
    True; history user turns are already-wrapped in <user_input> tags.
    """
    if include_ctx:
        current = (
            "<retrieved_context>\n"
            f"<memories>\n[Memory] benchmark turn memory for '{query[:40]}'\n</memories>\n"
            "</retrieved_context>\n"
            + user_content_template.format(query=query)
        )
    else:
        current = user_content_template.format(query=query)
    return (
        [{"role": "system", "content": STABLE_SYSTEM}]
        + history
        + [{"role": "user", "content": current}]
    )


def _wrap(q: str) -> str:
    return f"<user_input>\n{q}\n</user_input>"


async def _probe_openai_compatible(
    *, name: str, base_url: str, api_key: Optional[str], model: str, turns: int
) -> ProviderResult:
    """Run the turn-sweep against any OpenAI-compatible endpoint and parse
    usage fields the OpenAI schema provides.
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(headers=headers, timeout=300.0) as client:
        # Quick reachability check
        try:
            r = await client.get(f"{base_url}/models", timeout=10.0)
            if r.status_code >= 500:
                return ProviderResult(
                    name=name, status="skipped",
                    reason=f"/models returned {r.status_code}",
                )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            return ProviderResult(
                name=name, status="skipped", reason=f"unreachable: {e!s}"
            )

        history: list = []
        results: List[TurnResult] = []

        for i in range(1, turns + 1):
            q = f"Turn {i}: what is {i}+{i}?"
            messages = _build_messages(
                history=history,
                user_content_template="{query}",
                query=_wrap(q),
                include_ctx=True,
            )
            start = time.perf_counter()
            try:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": 32,
                        "temperature": 0.0,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
            except (httpx.HTTPError, httpx.RequestError) as e:
                return ProviderResult(
                    name=name, status="error", reason=f"turn {i}: {e!s}",
                    turns=results,
                )
            elapsed = time.perf_counter() - start
            data = resp.json()
            usage = data.get("usage") or {}
            cached = None
            details = usage.get("prompt_tokens_details") or {}
            if isinstance(details, dict) and "cached_tokens" in details:
                cached = details["cached_tokens"]
            elif "cached_tokens" in usage:
                cached = usage["cached_tokens"]

            results.append(
                TurnResult(
                    turn=i,
                    wall_seconds=elapsed,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    cached_tokens=cached,
                )
            )

            content = (
                data.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            history.append({"role": "user", "content": _wrap(q)})
            history.append({"role": "assistant", "content": content})

        return ProviderResult(name=name, status="ok", turns=results)


async def _probe_anthropic(
    *, api_key: Optional[str], model: str, turns: int
) -> ProviderResult:
    if not api_key:
        return ProviderResult(
            name="anthropic", status="skipped", reason="no ANTHROPIC_API_KEY"
        )
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    base_url = "https://api.anthropic.com/v1"
    async with httpx.AsyncClient(headers=headers, timeout=300.0) as client:
        history: list = []
        results: List[TurnResult] = []

        for i in range(1, turns + 1):
            q = f"Turn {i}: what is {i}+{i}?"
            # Anthropic wants system top-level, not in messages; caller's
            # Kestrel emits an OpenAI shape and the adapter would split it.
            # For this direct-HTTP probe we split it here.
            current_content = (
                "<retrieved_context>\n"
                f"<memories>\n[Memory] benchmark turn memory for '{q[:40]}'\n</memories>\n"
                "</retrieved_context>\n" + _wrap(q)
            )
            messages = history + [{"role": "user", "content": current_content}]
            payload = {
                "model": model,
                "system": STABLE_SYSTEM,
                "messages": messages,
                "max_tokens": 32,
                "temperature": 0.0,
            }
            start = time.perf_counter()
            try:
                resp = await client.post(
                    f"{base_url}/messages", json=payload
                )
                resp.raise_for_status()
            except (httpx.HTTPError, httpx.RequestError) as e:
                return ProviderResult(
                    name="anthropic", status="error",
                    reason=f"turn {i}: {e!s}", turns=results,
                )
            elapsed = time.perf_counter() - start
            data = resp.json()
            usage = data.get("usage") or {}
            results.append(
                TurnResult(
                    turn=i,
                    wall_seconds=elapsed,
                    prompt_tokens=usage.get("input_tokens"),
                    completion_tokens=usage.get("output_tokens"),
                    cached_tokens=usage.get("cache_read_input_tokens"),
                    cache_creation_tokens=usage.get(
                        "cache_creation_input_tokens"
                    ),
                    notes=(
                        "no cache_control markers in request — expect zero "
                        "cache_read until #705 lands"
                    ),
                )
            )

            # Extract response text
            content_blocks = data.get("content", [])
            text = "".join(
                b.get("text", "")
                for b in content_blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )
            history.append({"role": "user", "content": current_content})
            history.append({"role": "assistant", "content": text})

        return ProviderResult(name="anthropic", status="ok", turns=results)


async def _probe_all(turns: int, only: Optional[List[str]]) -> List[ProviderResult]:
    probes: List[Callable] = []

    def want(name: str) -> bool:
        return only is None or name in only

    if want("llama_cpp"):
        probes.append(
            _probe_openai_compatible(
                name="llama_cpp",
                base_url="http://localhost:8001/v1",
                api_key=None,
                model="auto",
                turns=turns,
            )
        )
    if want("ollama"):
        # Default model; override via OLLAMA_BENCH_MODEL if the local
        # install doesn't have llama3.2:latest.
        ollama_model = os.environ.get("OLLAMA_BENCH_MODEL", "llama3.2:latest")
        probes.append(
            _probe_openai_compatible(
                name="ollama",
                base_url="http://localhost:11434/v1",
                api_key=None,
                model=ollama_model,
                turns=turns,
            )
        )
    if want("openai"):
        probes.append(
            _probe_openai_compatible(
                name="openai",
                base_url="https://api.openai.com/v1",
                api_key=os.environ.get("OPENAI_API_KEY"),
                model="gpt-4o-mini",
                turns=turns,
            )
        )
    if want("openrouter"):
        probes.append(
            _probe_openai_compatible(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                model="openai/gpt-4o-mini",
                turns=turns,
            )
        )
    if want("anthropic"):
        probes.append(
            _probe_anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                model="claude-sonnet-4-5-20250929",
                turns=turns,
            )
        )

    return await asyncio.gather(*probes)


def _print_report(results: List[ProviderResult]) -> int:
    print(f"\n{'Provider':<12}  {'Status':<10}  {'T1 (s)':>7}  {'T2+ mean':>9}  "
          f"{'Δ%':>6}  {'Max cached':>11}  Notes")
    print("-" * 110)
    any_failure = False
    for r in results:
        if r.status != "ok":
            print(f"{r.name:<12}  {r.status:<10}  {'—':>7}  {'—':>9}  "
                  f"{'—':>6}  {'—':>11}  {r.reason}")
            continue
        if len(r.turns) < 2:
            print(f"{r.name:<12}  {r.status:<10}  {'—':>7}  {'—':>9}  "
                  f"{'—':>6}  {'—':>11}  insufficient turns")
            continue
        t1 = r.turns[0].wall_seconds
        t2plus_mean = statistics.mean(t.wall_seconds for t in r.turns[1:])
        speedup = r.speedup() or 0
        cached = r.max_cached_tokens()
        provider_notes = r.turns[-1].notes
        print(
            f"{r.name:<12}  {r.status:<10}  {t1:>7.2f}  {t2plus_mean:>9.2f}  "
            f"{speedup*100:>5.1f}%  {cached:>11}  {provider_notes}"
        )

        # Per-provider cache-hit expectation:
        # - llama.cpp / ollama / openai: expect speedup OR cached > 0
        # - anthropic (pre-#705): expect NO cache (any is a bug)
        # - openrouter: varies, be lenient
        if r.name == "anthropic":
            # Without cache_control markers, we expect zero reads. Zero
            # creation is also fine.  Non-zero reads here would be a
            # surprise worth flagging but not a failure.
            continue
        if speedup < 0.05 and cached == 0:
            print(f"  ⚠  {r.name}: no meaningful cache signal on turns 2+")
            any_failure = True

    return 0 if not any_failure else 1


async def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--turns", type=int, default=3, help="turns per provider (default 3)"
    )
    p.add_argument(
        "--only",
        default=None,
        help="comma-separated subset (e.g. llama_cpp,ollama,openai)",
    )
    args = p.parse_args()
    only = args.only.split(",") if args.only else None

    print(f"Running {args.turns}-turn sweep" + (f" (subset: {only})" if only else ""))
    results = await _probe_all(turns=args.turns, only=only)
    return _print_report(results)


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
