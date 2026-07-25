"""Contract tests for the canonical context-management documentation."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_DIR = PROJECT_ROOT / "docs" / "architecture"
CANONICAL_PATH = ARCHITECTURE_DIR / "CONTEXT_SYSTEM_DESIGN.md"
SALVAGE_PATH = ARCHITECTURE_DIR / "CONTEXT_C_DURABLE_SALVAGE.md"
ARCHITECTURE_INDEX = ARCHITECTURE_DIR / "README.md"
LOCAL_LINK_RE = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:)([^)]+)\)")

REQUIRED_SOURCE_LINKS = {
    "../../kestrel_sovereign/agent/context_manager.py",
    "../../kestrel_sovereign/agent/context_builder.py",
    "../../kestrel_sovereign/agent/context_stages.py",
    "../../kestrel_sovereign/agent/token_budget.py",
    "../../kestrel_sovereign/agent/token_counter.py",
    "../../kestrel_sovereign/storage/async_conversation_store.py",
    "../../kestrel_sovereign/kestrel_agent.py",
    "../../kestrel_sovereign/agent/streaming.py",
    "../../kestrel_sovereign/llm/",
    "../../kestrel_sovereign/endpoints/agent.py",
    "../../kestrel_sovereign/features/context/feature.py",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _frontmatter(path: Path) -> dict[str, object]:
    text = _read(path)
    match = re.match(r"\A---\n(?P<body>.*?)\n---\n", text, re.DOTALL)
    assert match, f"{path.relative_to(PROJECT_ROOT)} is missing frontmatter"
    metadata = yaml.safe_load(match["body"])
    assert isinstance(metadata, dict)
    return metadata


def _assert_local_links_resolve(path: Path) -> None:
    for target in LOCAL_LINK_RE.findall(_read(path)):
        local_target = target.split("#", 1)[0]
        if not local_target:
            continue
        resolved = (path.parent / local_target).resolve()
        assert resolved.exists(), (
            f"{path.relative_to(PROJECT_ROOT)} links to missing target {target}"
        )


def test_context_system_design_is_the_only_canonical_context_document():
    context_docs = sorted(ARCHITECTURE_DIR.glob("CONTEXT*.md"))
    canonical_docs = [
        path for path in context_docs if _frontmatter(path).get("canonical") is True
    ]

    assert canonical_docs == [CANONICAL_PATH]
    metadata = _frontmatter(CANONICAL_PATH)
    assert metadata["status"] == "active"
    assert metadata["owner"] == "context-runtime"

    text = _read(CANONICAL_PATH)
    assert "needs-revalidation" not in text
    assert "No code in this branch" not in text
    assert "Status: active and canonical" in text


def test_canonical_context_contract_names_every_runtime_owner_and_surface():
    text = _read(CANONICAL_PATH)
    link_targets = set(LOCAL_LINK_RE.findall(text))

    assert REQUIRED_SOURCE_LINKS <= link_targets
    assert "GET /api/agent/context-status" in text
    assert "`full=true`" in text
    assert "`context_status`" in text
    assert "`rendered_content`" in text
    assert "`metadata.sent_form=true`" in text
    assert "1,024" in text
    assert "KESTREL_PRUNE_TARGET_FRAC" in text
    assert "KESTREL_CONTEXT_C_DURABLE_SALVAGE" in text


def test_context_contract_preserves_status_and_diagnostic_honesty():
    canonical = _read(CANONICAL_PATH)
    salvage = _read(SALVAGE_PATH)
    issue_url = (
        "https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2534"
    )

    for status in ("Shipped", "Conditional", "Diagnostic", "Aspirational"):
        assert status in canonical
    assert issue_url in canonical
    assert "neither mode is an exact dry-run" in canonical
    assert "planning signal, not a receipt" in canonical

    salvage_metadata = _frontmatter(SALVAGE_PATH)
    assert salvage_metadata["status"] == "aspirational"
    assert salvage_metadata["canonical"] is False
    normalized_salvage = salvage.replace("\n> ", " ")
    assert "complete state machine described here has not shipped" in (
        normalized_salvage.casefold()
    )
    assert "[Kestrel Context Management Contract](CONTEXT_SYSTEM_DESIGN.md)" in salvage
    assert issue_url in salvage


def test_context_documents_are_cross_linked_indexed_and_resolvable():
    canonical = _read(CANONICAL_PATH)
    salvage = _read(SALVAGE_PATH)
    index = _read(ARCHITECTURE_INDEX)

    assert "[Context C Durable Salvage](CONTEXT_C_DURABLE_SALVAGE.md)" in canonical
    assert "[CONTEXT_SYSTEM_DESIGN.md](CONTEXT_SYSTEM_DESIGN.md)" in index
    assert "[CONTEXT_C_DURABLE_SALVAGE.md](CONTEXT_C_DURABLE_SALVAGE.md)" in index
    assert "| **Active** |" in index
    assert "| **Planning** |" in index

    _assert_local_links_resolve(CANONICAL_PATH)
    _assert_local_links_resolve(SALVAGE_PATH)
