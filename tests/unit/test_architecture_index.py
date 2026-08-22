"""Contract tests for the curated architecture route map."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "docs" / "architecture" / "README.md"
ECOSYSTEM_PATH = PROJECT_ROOT / "docs" / "ECOSYSTEM.md"
FEATURE_REGISTRY_PATH = (
    PROJECT_ROOT / "kestrel_sovereign" / "data" / "feature_registry.toml"
)
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
EVIDENCE_RE = re.compile(r"`([^`]+)`")
STATUS_TO_OKF = {
    "Active": "active",
    "Design of record": "design-of-record",
    "Experimental": "experimental",
    "Planning": "aspirational",
    "Historical": "historical",
    "Strategy": "aspirational",
}
EXTERNAL_MODULE_OWNERS = {
    "kestrel_feature_github": "kestrel-feature-github",
    "kestrel_feature_wallet": "kestrel-feature-wallet",
    "kestrel_sdk.llm": "kestrel-sovereign-sdk",
    "kestrel_sdk.payer_policy": "kestrel-sovereign-sdk",
}
LOCAL_ROOT_FILES = {"KESTREL_FEATURES.md", "run_tests.py"}
LOCAL_PATH_PREFIXES = (
    ".github/",
    ".kestreltalon/",
    "docs/",
    "kestrel_sovereign/",
    "tests/",
)


def _text() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")


def _sections(text: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(text))
    names = [match["section"] for match in matches]
    assert names == ["primary", "reference"], (
        "the architecture index must contain exactly one ordered primary section "
        "and one ordered reference section"
    )
    return {match["section"]: match["body"] for match in matches}


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


def _frontmatter(target: str) -> dict[str, str]:
    text = (INDEX_PATH.parent / target).read_text(encoding="utf-8")
    match = re.match(r"\A---\n(?P<body>.*?)\n---\n", text, re.DOTALL)
    assert match, f"{target} is missing OKF frontmatter"
    fields: dict[str, str] = {}
    for line in match["body"].splitlines():
        field = re.match(r"^(?P<key>[a-z_]+):\s*(?P<value>.*)$", line)
        if field:
            fields[field["key"]] = field["value"].strip().strip("'\"")
    return fields


def _assert_owner_evidence(target: str, owner: str) -> None:
    ecosystem = ECOSYSTEM_PATH.read_text(encoding="utf-8")
    with FEATURE_REGISTRY_PATH.open("rb") as registry_file:
        registry = tomllib.load(registry_file)
    catalogued_features = {
        entry["package"]
        for entry in registry.values()
        if isinstance(entry, dict)
        and entry.get("boundary") == "feature-package"
        and isinstance(entry.get("package"), str)
    }
    evidence = EVIDENCE_RE.findall(owner)
    if not evidence:
        assert owner.strip().endswith("no runtime owner"), (
            f"{target} must name checkable owner evidence or explicitly state that "
            "it has no runtime owner"
        )
        return

    for token in evidence:
        if token == "kestrel-sovereign":
            with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
                project_name = tomllib.load(pyproject_file)["project"]["name"]
            assert project_name.replace("_", "-") == token
            continue

        external_package = EXTERNAL_MODULE_OWNERS.get(token)
        if external_package is not None:
            assert f"[{external_package}]" in ecosystem, (
                f"{target} names external module {token} without package evidence"
            )
            if external_package.startswith("kestrel-feature-"):
                assert external_package in catalogued_features, (
                    f"{target} names catalogued feature module {token} without a "
                    "matching feature-registry package"
                )
            continue

        if token.startswith("kestrel-"):
            assert f"[{token}]" in ecosystem, (
                f"{target} names external package {token} absent from docs/ECOSYSTEM.md"
            )
            if token.startswith("kestrel-feature-"):
                assert token in catalogued_features, (
                    f"{target} names feature package {token} absent from the "
                    "feature registry"
                )
            continue

        if token in LOCAL_ROOT_FILES or token.startswith(LOCAL_PATH_PREFIXES):
            local_path = PROJECT_ROOT / token.rstrip("/")
            assert local_path.exists(), (
                f"{target} names missing repository owner evidence: {token}"
            )
            continue

        raise AssertionError(
            f"{target} owner reference {token!r} is not checkable as a repository "
            "path, distribution, or external module"
        )


def test_architecture_index_uses_the_documented_status_taxonomy():
    text = _text()
    sections = _sections(text)
    taxonomy = _taxonomy(text)

    assert taxonomy == set(STATUS_TO_OKF)
    assert sections.keys() == {"primary", "reference"}

    rows = _rows(sections["primary"]) + _rows(sections["reference"])
    assert rows
    assert {row["status"] for row in rows} <= taxonomy


def test_every_indexed_document_has_an_owner_and_resolves():
    sections = _sections(_text())
    rows = _rows(sections["primary"]) + _rows(sections["reference"])

    targets = [row["target"] for row in rows]
    assert len(targets) >= 52, (
        "the curated baseline must not shrink; new architecture documents may be added"
    )
    assert len(targets) == len(
        set(targets)
    ), "architecture documents must be indexed once"

    for row in rows:
        assert row["owner"].strip(), f"{row['target']} is missing an owner"
        assert row["scope"].strip(), f"{row['target']} is missing a scope/status note"
        assert (INDEX_PATH.parent / row["target"]).is_file(), (
            f"{row['target']} does not resolve from the architecture index"
        )
        _assert_owner_evidence(row["target"], row["owner"])


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


def test_index_status_matches_linked_document_metadata():
    sections = _sections(_text())
    primary_targets = {
        row["target"] for row in _rows(sections["primary"])
    }
    rows = _rows(sections["primary"]) + _rows(sections["reference"])

    for row in rows:
        metadata = _frontmatter(row["target"])
        assert metadata.get("status") == STATUS_TO_OKF[row["status"]], (
            f"{row['target']} table status and OKF status disagree"
        )
        expected_canonical = "true" if row["target"] in primary_targets else "false"
        assert metadata.get("canonical") == expected_canonical, (
            f"{row['target']} canonical metadata disagrees with its index section"
        )


def test_index_frontmatter_records_completed_revalidation():
    frontmatter = _text().split("---", 2)[1]

    assert "\nstatus: active\n" in frontmatter
    assert "needs-revalidation" not in frontmatter
