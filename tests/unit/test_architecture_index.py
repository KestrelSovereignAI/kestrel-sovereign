"""Contract tests for the curated architecture route map."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "docs" / "architecture" / "README.md"
SECTION_RE = re.compile(
    r"<!-- architecture-index:(?P<section>primary|reference):start -->"
    r"(?P<body>.*?)"
    r"<!-- architecture-index:(?P=section):end -->",
    re.DOTALL,
)
STATUS_RE = re.compile(r"^\| \*\*(?P<status>[^*]+)\*\* \|", re.MULTILINE)
ROW_RE = re.compile(
    r"^\| \[(?P<label>[^\]]+)\]\((?P<target>[^)]+)\) "
    r"\| \*\*(?P<status>[^*]+)\*\* "
    r"\| (?P<owner>[^|]+) "
    r"\| (?P<scope>[^|]+) \|$",
    re.MULTILINE,
)
RAW_ROW_RE = re.compile(r"^\| \[[^\]]+\]\([^)]+\) \|.*$", re.MULTILINE)


def _text() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")


def _sections(text: str) -> dict[str, str]:
    return {match["section"]: match["body"] for match in SECTION_RE.finditer(text)}


def _taxonomy(text: str) -> set[str]:
    legend = text.split("## Primary implementation map", 1)[0]
    return set(STATUS_RE.findall(legend))


def _rows(section: str) -> list[dict[str, str]]:
    raw_rows = RAW_ROW_RE.findall(section)
    parsed_rows = [match.groupdict() for match in ROW_RE.finditer(section)]
    assert len(parsed_rows) == len(raw_rows), (
        "every indexed document row must use Document, documented Status, "
        "Owner, and Scope cells"
    )
    return parsed_rows


def test_architecture_index_uses_the_documented_status_taxonomy():
    text = _text()
    sections = _sections(text)
    taxonomy = _taxonomy(text)

    assert taxonomy == {
        "Active",
        "Design of record",
        "Experimental",
        "Planning",
        "Historical",
        "Strategy",
    }
    assert sections.keys() == {"primary", "reference"}

    rows = _rows(sections["primary"]) + _rows(sections["reference"])
    assert rows
    assert {row["status"] for row in rows} <= taxonomy


def test_every_indexed_document_has_an_owner_and_resolves():
    sections = _sections(_text())
    rows = _rows(sections["primary"]) + _rows(sections["reference"])

    targets = [row["target"] for row in rows]
    assert len(targets) == len(
        set(targets)
    ), "architecture documents must be indexed once"

    for row in rows:
        assert row["owner"].strip(), f"{row['target']} is missing an owner"
        assert row["scope"].strip(), f"{row['target']} is missing a scope/status note"
        assert (INDEX_PATH.parent / row["target"]).is_file(), (
            f"{row['target']} does not resolve from the architecture index"
        )


def test_primary_map_only_contains_current_contracts():
    sections = _sections(_text())
    primary_rows = _rows(sections["primary"])
    reference_rows = _rows(sections["reference"])

    assert primary_rows
    assert reference_rows
    assert {row["status"] for row in primary_rows} <= {
        "Active",
        "Design of record",
    }
    assert {row["status"] for row in reference_rows}.isdisjoint(
        {"Active", "Design of record"}
    )


def test_active_documents_name_a_precise_implementation_owner():
    sections = _sections(_text())
    active_rows = [
        row
        for section in sections.values()
        for row in _rows(section)
        if row["status"] == "Active"
    ]

    assert active_rows
    for row in active_rows:
        owner = row["owner"].strip()
        assert owner
        assert owner not in {"architecture", "documentation", "Kestrel Team"}
        assert "`" in owner, f"{row['target']} must name a module or package owner"


def test_index_frontmatter_records_completed_revalidation():
    frontmatter = _text().split("---", 2)[1]

    assert "\nstatus: active\n" in frontmatter
    assert "needs-revalidation" not in frontmatter
