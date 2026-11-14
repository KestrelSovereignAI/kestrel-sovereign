"""Record provider token counts for the six #4068 fixture classes (#4085).

``TokenCounter.upper_bound`` starts from 2.0 bytes per token for Claude, a
floor taken from one mixed payload. This script asks the provider to count
six synthetic texts (``tests/fixtures/token_bound``) and writes the record the
row-14 regression in ``tests/unit/test_token_upper_bound_4068.py`` compares
the bound against.

Operator-run, once, through the metered API-key route, where Anthropic's
``count_tokens`` is reachable. It is a paid call: get the Sovereign's go
first. The texts are synthetic; no agent data is sent. Unit tests never run
it.

    ANTHROPIC_API_KEY=... uv run python scripts/record_token_bound_counts.py \\
        --model claude-fable-5-2
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "token_bound"
RECORD_PATH = FIXTURE_DIR / "recorded_counts.json"

#: Class name -> fixture file. ``tool_schemas`` is a tool set; the rest are text.
FIXTURE_CLASSES: Dict[str, str] = {
    "hex_and_ids": "hex_and_ids.txt",
    "base64": "base64.txt",
    "cjk": "cjk.txt",
    "emoji": "emoji.txt",
    "json_escapes": "json_escapes.txt",
    "tool_schemas": "tool_schemas.json",
}
#: The user message that carries a tool-schema class, which needs one.
TOOL_CLASS_PROMPT = "List the tools."


def fixture_request(name: str, fixture_dir: Path = FIXTURE_DIR) -> Dict[str, Any]:
    """The request one class is counted and bounded as: system, messages, tools."""
    text = (fixture_dir / FIXTURE_CLASSES[name]).read_text(encoding="utf-8")
    if name == "tool_schemas":
        return {
            "system": None,
            "messages": [{"role": "user", "content": TOOL_CLASS_PROMPT}],
            "tools": json.loads(text),
        }
    return {"system": None, "messages": [{"role": "user", "content": text}], "tools": None}


async def record_counts(
    model: str,
    count: Callable[[Dict[str, Any]], Awaitable[Optional[int]]],
    fixture_dir: Path = FIXTURE_DIR,
) -> Dict[str, Any]:
    """Count every class with ``count`` and return the record to commit.

    Each entry holds the request's wire bytes as ``upper_bound`` measures them
    and the provider's count. Nothing else, and no content.
    """
    from kestrel_sovereign.agent.token_counter import MeasurementRefusal, get_token_counter

    counter = get_token_counter(model)
    classes: Dict[str, Dict[str, int]] = {}
    for name in FIXTURE_CLASSES:
        request = fixture_request(name, fixture_dir)
        bound = counter.upper_bound(request["system"], request["messages"], request["tools"])
        if isinstance(bound, MeasurementRefusal):
            raise RuntimeError(f"{name}: upper_bound refused the fixture: {bound.reason}")
        provider_count = await count(request)
        if not isinstance(provider_count, int) or provider_count <= 0:
            raise RuntimeError(f"{name}: the provider returned no count ({provider_count!r})")
        classes[name] = {"wire_bytes": bound.wire_bytes, "provider_count": provider_count}
    return {
        "model": model,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "classes": classes,
    }


async def _anthropic_count(model: str) -> Callable[[Dict[str, Any]], Awaitable[Optional[int]]]:
    """A ``count`` that calls Anthropic's ``count_tokens`` on the API-key route."""
    import anthropic

    from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

    adapter = AnthropicAdapter()
    client = anthropic.AsyncAnthropic()

    async def count(request: Dict[str, Any]) -> Optional[int]:
        result = await adapter.count_tokens(
            client, model, request["messages"], tools=request["tools"],
            system_prompt=request["system"],
        )
        return None if result is None else result.input_tokens

    return count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--model", required=True, help="Anthropic model id to count with")
    parser.add_argument("--output", type=Path, default=RECORD_PATH)
    args = parser.parse_args(argv)

    async def run() -> Dict[str, Any]:
        return await record_counts(args.model, await _anthropic_count(args.model))

    record = asyncio.run(run())
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
