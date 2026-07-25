"""Structural contract tests for canonical context-management documentation."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from scripts import check_docs_links, generate_feature_docs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_DIR = PROJECT_ROOT / "docs" / "architecture"
CANONICAL_PATH = ARCHITECTURE_DIR / "CONTEXT_SYSTEM_DESIGN.md"
SALVAGE_PATH = ARCHITECTURE_DIR / "CONTEXT_C_DURABLE_SALVAGE.md"
ARCHITECTURE_INDEX = ARCHITECTURE_DIR / "README.md"
INVENTORY_PATH = PROJECT_ROOT / "KESTREL_FEATURES.md"

REQUIRED_OWNERS = {
    "Production section ordering, retrieval gates, elastic allocation, pruning, and degraded mode": (
        "[`kestrel_sovereign/agent/context_manager.py`]"
        "(../../kestrel_sovereign/agent/context_manager.py)"
    ),
    "System construction, history rendering, token measurement, and diagnostic breakdown": (
        "[`kestrel_sovereign/agent/context_builder.py`]"
        "(../../kestrel_sovereign/agent/context_builder.py)"
    ),
    "Shared section vocabulary, rendered-message emission, wrappers, and lumpy-anchor primitives": (
        "[`kestrel_sovereign/agent/context_stages.py`]"
        "(../../kestrel_sovereign/agent/context_stages.py)"
    ),
    "Fixed, adaptive, and production elastic budgets; response reserve": (
        "[`kestrel_sovereign/agent/token_budget.py`]"
        "(../../kestrel_sovereign/agent/token_budget.py)"
    ),
    "Canonical and rendered conversation persistence": (
        "[`kestrel_sovereign/storage/async_conversation_store.py`]"
        "(../../kestrel_sovereign/storage/async_conversation_store.py)"
    ),
    "`/api/agent/context-status` and shared status computation": (
        "[`kestrel_sovereign/endpoints/agent.py`]"
        "(../../kestrel_sovereign/endpoints/agent.py)"
    ),
    "Context tools and manual context-management operations": (
        "[`kestrel_sovereign/features/context/feature.py`]"
        "(../../kestrel_sovereign/features/context/feature.py)"
    ),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _frontmatter(path: Path) -> dict[str, object]:
    match = re.match(r"\A---\n(?P<body>.*?)\n---\n", _read(path), re.DOTALL)
    assert match, f"{path.relative_to(PROJECT_ROOT)} is missing frontmatter"
    metadata = yaml.safe_load(match["body"])
    assert isinstance(metadata, dict)
    return metadata


def _section(text: str, heading: str) -> str:
    """Return one exact Markdown section, stopping at a peer/parent heading."""
    match = re.search(
        rf"^(?P<marks>#+) {re.escape(heading)}\s*$",
        text,
        re.MULTILINE,
    )
    assert match, f"missing section heading: {heading}"
    level = len(match["marks"])
    next_heading = re.search(
        rf"^#{{1,{level}}} ",
        text[match.end() :],
        re.MULTILINE,
    )
    end = (
        match.end() + next_heading.start()
        if next_heading is not None
        else len(text)
    )
    return text[match.end() : end].strip()


def _table_rows(section: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    assert len(rows) >= 2, "section is missing a Markdown table"
    return rows


def _two_column_table(section: str) -> dict[str, str]:
    rows = _table_rows(section)
    assert len(rows[0]) == 2
    return {row[0]: row[1] for row in rows[1:]}


def _assert_links_resolve_with_anchors(path: Path) -> None:
    broken = check_docs_links.check_file(path)
    assert broken == [], "\n".join(link.format() for link in broken)


def test_context_system_design_is_the_only_canonical_context_document():
    context_docs = sorted(ARCHITECTURE_DIR.glob("CONTEXT*.md"))
    canonical_docs = [
        path for path in context_docs if _frontmatter(path).get("canonical") is True
    ]

    assert canonical_docs == [CANONICAL_PATH]
    assert _frontmatter(CANONICAL_PATH) == {
        "type": "Architecture Spec",
        "title": "Kestrel Context Management Contract",
        "description": (
            "Canonical current-state contract for prompt assembly, persistence, "
            "budgeting, pruning, retrieval, provider rendering, and context "
            "diagnostics."
        ),
        "resource": "/docs/architecture/CONTEXT_SYSTEM_DESIGN.md",
        "tags": ["docs", "architecture", "architecture-spec", "context"],
        "timestamp": "2026-07-25T00:00:00Z",
        "status": "active",
        "owner": "context-runtime",
        "canonical": True,
        "generated": False,
        "privacy": "public",
    }


def test_status_vocabulary_and_runtime_ownership_are_structural_contracts():
    text = _read(CANONICAL_PATH)
    statuses = _two_column_table(_section(text, "Status vocabulary"))
    assert statuses == {
        "**Shipped**": "Runs on the normal production turn path.",
        "**Conditional**": (
            "Implemented, but only runs for a named route, privacy mode, "
            "configuration, feature flag, or data shape."
        ),
        "**Diagnostic**": (
            "Observes or estimates runtime state; it is not proof of the exact "
            "provider payload or production decision path."
        ),
        "**Aspirational**": (
            "Design intent that is not a current runtime guarantee."
        ),
    }

    owners = _two_column_table(_section(text, "Ownership and source anchors"))
    for responsibility, link in REQUIRED_OWNERS.items():
        assert owners[responsibility] == link


def test_production_contract_is_scoped_to_its_numbered_owners():
    text = _read(CANONICAL_PATH)
    expected_headings = [
        "1. Resolve route identity and the usable window",
        "2. Establish the mandatory system floor",
        "3. Read canonical, session-scoped history",
        "4. Build stable and dynamic sections",
        "5. Apply retrieval insertion and exclusion gates",
        "6. Select cache-stable history with lumpy pruning",
        "7. Gate the loaded-feature prompt",
        "8. Persist the sent form and invoke the adapter",
    ]
    actual_headings = re.findall(r"^### ([1-8]\. .+)$", text, re.MULTILINE)
    assert actual_headings == expected_headings

    limits = _section(text, expected_headings[0])
    assert "route-qualified (`vendor:route/model`)" in limits
    assert "1,024 tokens" in limits

    floor = _section(text, expected_headings[1])
    allocation_rows = _table_rows(floor)
    assert allocation_rows[0] == [
        "Conversation size",
        "System",
        "History",
        "Episodes",
        "Memories",
        "RAG",
    ]
    assert allocation_rows[1:] == [
        ["Short, fewer than 10 messages", "15%", "60%", "5%", "5%", "15%"],
        ["Medium, 10–29 messages", "15%", "40%", "20%", "10%", "15%"],
        ["Long, 30 or more messages", "15%", "25%", "35%", "10%", "15%"],
    ]

    history = _section(text, expected_headings[2])
    assert "latest **50** eligible entries from the active session" in history
    assert "`ISOLATED` history" in history
    assert "has no persistent row ids" in history
    assert "Legacy **user** rows" in history
    assert "`content`" in history and "`rendered_content`" in history

    retrieval = _section(text, expected_headings[4])
    assert "`0.3` and `0.2` by default" in retrieval
    assert "`0.5` by default" in retrieval

    pruning = _section(text, expected_headings[5])
    assert "at-most-50-entry production preload" in pruning
    assert "75% of the available budget by default" in _squash(pruning)
    assert "`KESTREL_PRUNE_TARGET_FRAC`" in pruning

    feature_gate = _section(text, expected_headings[6])
    assert "This does **not** unregister tools" in feature_gate
    assert "provider-exact tool-schema/framing count" in feature_gate

    persistence = _section(text, expected_headings[7])
    assert "`sent_form=true`" in persistence
    assert "streaming and non-streaming paths" in persistence


def test_provider_and_diagnostic_sections_preserve_current_limitations():
    text = _read(CANONICAL_PATH)
    provider = _section(text, "Provider transport constraints and route caps")
    provider_rows = _two_column_table(provider)
    assert set(provider_rows) == {
        "Anthropic / Claude",
        "Native OpenAI",
        "Gemini / Vertex",
        "Ollama",
        "`openai:plan` Codex",
    }
    assert "selected route **and model** advertise support" in provider_rows[
        "Anthropic / Claude"
    ]
    assert "resets the Codex thread only after that compaction reports success" in (
        _squash(provider)
    )
    assert "allow the turn to proceed" in _squash(provider)

    diagnostics = _section(text, "Context diagnostics")
    honesty = _section(text, "Honesty boundary: issue #2534")
    normalized_diagnostics = _squash(diagnostics)
    normalized_honesty = _squash(honesty)
    assert "up to **10,000** stored rows" in normalized_diagnostics
    assert "production turn's latest-50 preload" in normalized_diagnostics
    assert "`silently_pruned_path_active` is derived" in normalized_diagnostics
    assert "not observation that a particular turn" in normalized_diagnostics
    assert "Issue #2534" in normalized_honesty
    assert "remains open" in normalized_honesty
    assert "neither mode is an exact dry-run" in normalized_honesty
    assert "up to 10,000 stored rows" in normalized_honesty
    assert "latest 50 eligible entries" in normalized_honesty
    assert "planning signal, not a receipt" in normalized_honesty
    assert "does not change or close" in normalized_honesty
    assert not re.search(
        r"(?im)^\s*(?:closes|fixes|resolves)\s+#2534\b",
        text,
    )


def test_durable_salvage_document_keeps_partial_paths_conditional():
    canonical = _read(CANONICAL_PATH)
    salvage = _read(SALVAGE_PATH)
    assert _frontmatter(SALVAGE_PATH)["status"] == "aspirational"
    assert _frontmatter(SALVAGE_PATH)["canonical"] is False

    canonical_statuses = _two_column_table(
        _section(canonical, "Durable salvage status")
    )
    assert canonical_statuses[
        "Synchronous salvage marker/write plus background `SalvageWorker` processing during automatic pruning"
    ] == (
        "**Conditional**, behind `KESTREL_CONTEXT_C_DURABLE_SALVAGE` and "
        "limited to pruned spans that map to id-bearing persistent history"
    )
    assert canonical_statuses[
        "Complete automatic Context C lifecycle and all guarantees in the original design"
    ] == "**Aspirational**"

    salvage_statuses = _two_column_table(_section(salvage, "Status matrix"))
    assert salvage_statuses["Automatic salvage is the default for all routes"] == (
        "**Not shipped**"
    )
    assert salvage_statuses[
        "Original SignalDispatcher-based orchestration"
    ].startswith("**Not shipped**")

    evidence = _section(salvage, "What the partial implementation proves")
    normalized_evidence = _squash(evidence)
    assert "When no id-bearing span can be computed" in normalized_evidence
    assert "`ISOLATED` in-memory history" in normalized_evidence
    assert "reflects flag configuration rather than per-turn salvage evidence" in (
        normalized_evidence
    )


def test_context_honesty_contract_cascades_verbatim_to_every_audience_doc():
    source = _read(INVENTORY_PATH)
    contract = generate_feature_docs.extract_context_contract(source)
    assert source.count(contract) == 1

    for audience in generate_feature_docs.AUDIENCES:
        generated = (
            generate_feature_docs.OUTPUT_DIR / f"FEATURES_{audience}.md"
        )
        assert _read(generated).count(contract) == 1


def test_context_documents_are_indexed_and_links_include_valid_anchors():
    index = _read(ARCHITECTURE_INDEX)
    assert re.search(
        r"^\| \[CONTEXT_SYSTEM_DESIGN\.md\]\(CONTEXT_SYSTEM_DESIGN\.md\) "
        r"\| \*\*Active\*\* \|",
        index,
        re.MULTILINE,
    )
    assert re.search(
        r"^\| \[CONTEXT_C_DURABLE_SALVAGE\.md\]"
        r"\(CONTEXT_C_DURABLE_SALVAGE\.md\) \| \*\*Planning\*\* \|",
        index,
        re.MULTILINE,
    )

    _assert_links_resolve_with_anchors(CANONICAL_PATH)
    _assert_links_resolve_with_anchors(SALVAGE_PATH)
