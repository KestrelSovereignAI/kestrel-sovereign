"""System-prompt assembler — priority-ordered clause composition.

Produces the full in-agent system prompt by combining:

1. Anchored constitution (priority 1; never dropped)
2. Anchored doctrine files — TORTOISE_DOCTRINE.md, AGENTS.md, and any
   operator-declared additions (priorities 2-3)
3. SOUL.md identity (priority 4)
4. Prompt-adaptation preamble + state-of-mind block (priority 5)
5. Other BootstrapLoader files (priority 6)
6. Style reminder (priority 7)
7. Caller-supplied additional context (priority 7)

When the assembled prompt exceeds a configured byte budget, clauses
are dropped highest-priority-number first (within a priority, in
reverse emission order) until the prompt fits. The constitution is
never droppable; if it alone exceeds budget the budget is honored
in spirit (return whatever fits) but the constitution stays.

Both the kept and dropped clause lists are returned for forensic
recording on `signal_log` (`injected_clauses_json`,
`dropped_clauses_json`).

This module is the pure-function core. `ContextBuilder.build_system_prompt`
delegates to it; `ContextBuilder.build_system_prompt_with_tracking`
returns the structured `SystemPromptResult` for dispatcher integration
(kestrel-sovereign#1137 chunk 1G).

See `docs/architecture/CONSTITUTION_INJECTION.md` §7 for the priority
table this implements.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Priority ladder. Lower number = higher priority = kept first.
# Drop algorithm walks priorities in DESCENDING order, removing the
# most-recently-emitted clause at the highest priority number until
# the total fits the budget.
PRIORITY_CONSTITUTION = 1
PRIORITY_TORTOISE_DOCTRINE = 2
PRIORITY_ANCHORED_DOCTRINE = 3  # AGENTS.md + operator-declared additions
PRIORITY_SOUL = 4
PRIORITY_STATE_OF_MIND = 5
PRIORITY_OTHER_BOOTSTRAP = 6
PRIORITY_STYLE_REMINDER = 7

# Canonical clause names. Used in `signal_log.injected_clauses_json`
# and `dropped_clauses_json`. Filenames stay literal (e.g. "AGENTS.md")
# so an auditor querying signal_log knows exactly which file
# contributed; synthetic blocks (constitution, state of mind, style
# reminder) get UPPER_SNAKE_CASE names.
CLAUSE_KESTREL_CONSTITUTION = "KESTREL_CONSTITUTION"
CLAUSE_STATE_OF_MIND = "STATE_OF_MIND"
CLAUSE_PROMPT_ADAPTATION = "PROMPT_ADAPTATION"
CLAUSE_STYLE_REMINDER = "STYLE_REMINDER"
CLAUSE_ADDITIONAL_CONTEXT = "ADDITIONAL_CONTEXT"
CLAUSE_SESSION_BRIEFING = "SESSION_BRIEFING"

# The Tortoise Doctrine file is a special-cased anchored doctrine
# entry that gets priority 2 (between constitution and AGENTS.md).
# Other anchored files default to priority 3.
TORTOISE_DOCTRINE_FILENAME = "TORTOISE_DOCTRINE.md"
AGENTS_FILENAME = "AGENTS.md"


# Bytes added per joiner ("\n\n" between clauses).
_JOINER = "\n\n"
_JOINER_BYTES = len(_JOINER.encode("utf-8"))


@dataclass(frozen=True)
class SystemPromptResult:
    """Structured output of priority-aware system-prompt assembly.

    `prompt` is the joined string passed to the LLM. `injected_clauses`
    and `dropped_clauses` are the canonical names (see CLAUSE_* and
    file-basename conventions above) used by the dispatcher to write
    to `signal_log.injected_clauses_json` / `dropped_clauses_json`.

    `injected_clauses` is in legacy emission order (the order the
    clauses appear in `prompt`). `dropped_clauses` is in the order
    the budget-fit loop removed them — useful for an auditor
    reconstructing what would have been included next if the budget
    were larger.
    """

    prompt: str
    injected_clauses: List[str]
    dropped_clauses: List[str]


@dataclass
class _Clause:
    """Internal — one section of the assembled system prompt.

    `body` is the FULL fenced block including the `--- BEGIN X ---`
    and `--- END X ---` lines. `priority` decides drop order;
    `emit_index` preserves the legacy emission order so kept clauses
    re-render in the same sequence they were appended.
    """

    name: str
    priority: int
    body: str
    emit_index: int = 0
    bytes_size: int = field(init=False)

    def __post_init__(self) -> None:
        # `__setattr__` because the dataclass is mutable but we want
        # bytes_size derived from body at construction. (Frozen would
        # need object.__setattr__.)
        self.bytes_size = len(self.body.encode("utf-8"))


def section_name_for_anchored_file(filename: str) -> str:
    """Derive the fence label for an anchored doctrine file.

    `TORTOISE_DOCTRINE.md` → `TORTOISE DOCTRINE`
    `AGENTS.md` → `AGENTS`

    Underscores become spaces so the fence line reads naturally;
    the .md extension is stripped. Matches the existing
    `--- TORTOISE DOCTRINE ---` convention.
    """
    stem = Path(filename).stem  # strips ".md"
    return stem.replace("_", " ").upper()


def _wrap(label: str, content: str) -> str:
    """Render a `--- LABEL ---\\n<content>\\n--- END LABEL ---` block."""
    return f"--- {label} ---\n{content}\n--- END {label} ---"


def assemble_system_prompt(
    *,
    constitution: str,
    bootstrap_files: "OrderedDict[str, str]",
    anchored_doctrine: Optional["OrderedDict[str, str]"] = None,
    session_briefing: Optional[str] = None,
    prompt_adaptation_preamble: Optional[str] = None,
    state_of_mind_block: Optional[str] = None,
    style_reminder: Optional[str] = None,
    additional_context: Optional[str] = None,
    budget_bytes: Optional[int] = None,
) -> SystemPromptResult:
    """Assemble the system prompt with priority-ordered truncation.

    Inputs are pre-rendered strings — the caller (ContextBuilder) is
    responsible for building each block's contents (briefing text,
    state-of-mind body, etc.). This function only handles fence-
    wrapping, priority assignment, and budget enforcement.

    `bootstrap_files` is the full `BootstrapLoader.load()` cache —
    SOUL.md is special-cased; HEARTBEAT.md is skipped (loaded
    separately by the heartbeat runner); AGENTS.md is skipped IF
    `anchored_doctrine` already supplies it (avoids duplication).

    `anchored_doctrine` is an OrderedDict of `filename → content` for
    constitutional-injection bundle members. Order is the caller's
    responsibility (typically: TORTOISE_DOCTRINE.md, AGENTS.md, then
    operator-declared additions). All entries get priority
    `PRIORITY_ANCHORED_DOCTRINE` except `TORTOISE_DOCTRINE.md` which
    gets `PRIORITY_TORTOISE_DOCTRINE` per the design's two-tier rule.

    If `budget_bytes` is None, no truncation is applied. If supplied,
    clauses are dropped highest-priority-number first until the
    assembled UTF-8 byte length ≤ `budget_bytes`. The constitution is
    never droppable.
    """
    clauses: List[_Clause] = []
    emit_counter = 0

    def add(name: str, priority: int, body: str) -> None:
        nonlocal emit_counter
        clauses.append(
            _Clause(name=name, priority=priority, body=body, emit_index=emit_counter)
        )
        emit_counter += 1

    # --------------------------------------------------------------
    # Emit clauses in legacy textual order. Priority is independent
    # of emission order; only used for drop decisions.
    # --------------------------------------------------------------

    # Bootstrap files (priorities 4 / 6) — SOUL.md is priority 4 with
    # its identity wrapper; HEARTBEAT.md is loaded by the heartbeat
    # runner separately and must NOT appear here. AGENTS.md is
    # excluded if anchored_doctrine already supplies it (avoids
    # duplication when the dispatcher passes AGENTS.md as anchored).
    anchored = anchored_doctrine or OrderedDict()
    skip_agents_from_bootstrap = AGENTS_FILENAME in anchored

    for filename, content in bootstrap_files.items():
        if filename == "HEARTBEAT.md":
            continue
        if filename == AGENTS_FILENAME and skip_agents_from_bootstrap:
            continue
        if filename == "SOUL.md":
            add(
                name="SOUL.md",
                priority=PRIORITY_SOUL,
                body=_wrap("YOUR IDENTITY", content),
            )
        else:
            label = filename.replace(".md", "").upper()
            add(
                name=filename,
                priority=PRIORITY_OTHER_BOOTSTRAP,
                body=_wrap(label, content),
            )

    # Session briefing (priority 6 — non-critical, droppable before
    # bootstrap files only because it has higher emit_index in the
    # same priority bucket; design doesn't mandate a slot for it).
    if session_briefing:
        add(
            name=CLAUSE_SESSION_BRIEFING,
            priority=PRIORITY_OTHER_BOOTSTRAP,
            body=session_briefing.strip(),
        )

    # Prompt-adaptation preamble (priority 5).
    if prompt_adaptation_preamble:
        add(
            name=CLAUSE_PROMPT_ADAPTATION,
            priority=PRIORITY_STATE_OF_MIND,
            body=prompt_adaptation_preamble.strip(),
        )

    # Anchored doctrine (priorities 2 / 3) — emitted BEFORE the
    # constitution because conventionally the constitution is the
    # final authoritative voice; doctrine prefaces it. (Priority
    # affects drop order only, not text order.)
    for filename, content in anchored.items():
        priority = (
            PRIORITY_TORTOISE_DOCTRINE
            if filename == TORTOISE_DOCTRINE_FILENAME
            else PRIORITY_ANCHORED_DOCTRINE
        )
        label = section_name_for_anchored_file(filename)
        add(name=filename, priority=priority, body=_wrap(label, content))

    # Constitution (priority 1 — never droppable).
    add(
        name=CLAUSE_KESTREL_CONSTITUTION,
        priority=PRIORITY_CONSTITUTION,
        body=_wrap("GOVERNING CONSTITUTION", constitution),
    )

    # State of mind (priority 5) — pre-rendered by the caller.
    if state_of_mind_block:
        add(
            name=CLAUSE_STATE_OF_MIND,
            priority=PRIORITY_STATE_OF_MIND,
            body=state_of_mind_block.strip(),
        )

    # Style reminder (priority 7).
    if style_reminder:
        add(
            name=CLAUSE_STYLE_REMINDER,
            priority=PRIORITY_STYLE_REMINDER,
            body=style_reminder.strip(),
        )

    # Additional context (priority 7) — caller-supplied free-form.
    if additional_context:
        add(
            name=CLAUSE_ADDITIONAL_CONTEXT,
            priority=PRIORITY_STYLE_REMINDER,
            body=_wrap("ADDITIONAL CONTEXT", additional_context),
        )

    # --------------------------------------------------------------
    # Truncation.
    # --------------------------------------------------------------
    kept, dropped_in_drop_order = _drop_to_fit(clauses, budget_bytes)

    # Re-emit kept clauses in original emission order.
    kept_sorted = sorted(kept, key=lambda c: c.emit_index)
    prompt = _JOINER.join(c.body for c in kept_sorted)

    return SystemPromptResult(
        prompt=prompt,
        injected_clauses=[c.name for c in kept_sorted],
        dropped_clauses=[c.name for c in dropped_in_drop_order],
    )


def _drop_to_fit(
    clauses: List[_Clause], budget_bytes: Optional[int]
) -> tuple[List[_Clause], List[_Clause]]:
    """Walk clauses by descending priority, dropping until size <= budget.

    Returns `(kept, dropped)`. `kept` retains original list order;
    `dropped` is in the order they were removed (drop-time order, not
    emission order — useful for forensics).

    Budget None means no truncation. The constitution
    (`PRIORITY_CONSTITUTION`) is never dropped even when its size
    alone exceeds budget — the design treats constitutional integrity
    as the load-bearing invariant; an oversized constitution is an
    operator-config problem, not something the assembler should hide.
    """
    kept = list(clauses)
    dropped: List[_Clause] = []

    if budget_bytes is None:
        return kept, dropped

    def total_bytes(items: List[_Clause]) -> int:
        if not items:
            return 0
        body_total = sum(c.bytes_size for c in items)
        joiner_total = _JOINER_BYTES * (len(items) - 1)
        return body_total + joiner_total

    while total_bytes(kept) > budget_bytes:
        # Candidates: clauses with priority > CONSTITUTION, sorted
        # so the highest-priority-number-and-latest-emit_index goes
        # first. Same priority → drop later-emitted first.
        droppable = [c for c in kept if c.priority != PRIORITY_CONSTITUTION]
        if not droppable:
            # Only the constitution remains and it's still over budget.
            # Honor the constitution; let the prompt exceed budget.
            break
        droppable.sort(key=lambda c: (-c.priority, -c.emit_index))
        victim = droppable[0]
        kept.remove(victim)
        dropped.append(victim)

    return kept, dropped
