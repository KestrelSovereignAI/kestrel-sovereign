"""Validate raw TCK evidence without turning coverage gaps into conformance."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from xml.etree import ElementTree


def _junit_address(node_id: str) -> tuple[str, str]:
    address, bracket, parameters = node_id.partition("[")
    parts = address.split("::")
    module = parts[0].removesuffix(".py").replace("/", ".")
    return ".".join([module, *parts[1:-1]]), parts[-1] + bracket + parameters


def validate_tck_report(
    reports: Path, *, level: str, returncode: int,
    provenance_dir: Path | None = None,
) -> dict:
    """Reject failed checks and runner errors; preserve explicit coverage gaps.

    COVERAGE.json is an exact reviewed observation matrix, including missing
    probes, conditional skips and manual checks. It is not a waiver or a claim
    that these requirements have passed. A changed matrix needs review.
    """
    if level not in {"must", "full"}:
        raise ValueError(f"Unknown TCK level: {level}")
    if returncode not in {0, 1}:
        raise ValueError(f"TCK runner did not finish its checks: exit {returncode}")
    provenance = provenance_dir or Path(__file__).resolve().parents[1] / "interop/a2a_tck"
    source = tomllib.loads((provenance / "SOURCE.toml").read_text())
    profile = json.loads((provenance / "COVERAGE.json").read_text())
    if profile["schema_version"] != 2 or (
        profile["patched_tree"], profile["protocol_version"]
    ) != (source["patched_tree"], source["protocol_version"]):
        raise ValueError("TCK coverage profile differs from the reviewed source")
    expected = profile["must_requirements"]
    if not expected or not any(row["status"] == "PASS" for row in expected.values()):
        raise ValueError("TCK coverage profile contains no positive MUST evidence")
    gaps = {key for key, row in expected.items() if row["status"] == "NOT TESTED"}
    if gaps != set(profile["unverified_requirements"]) or any(
        not reason.strip() for reason in profile["unverified_requirements"].values()
    ):
        raise ValueError("TCK coverage gaps need an explicit reviewed description")

    advisories = profile.get("required_advisory_requirements")
    if not isinstance(advisories, dict) or not advisories or any(
        not isinstance(row, dict)
        or row.get("level") not in {"SHOULD", "MAY"}
        or row.get("status") != "PASS"
        or not row.get("transports")
        or set(row["transports"].values()) != {"PASS"}
        or not row.get("test_ids")
        for row in advisories.values()
    ):
        raise ValueError("TCK advisory coverage profile requires passing probes")

    report = json.loads((reports / "compatibility.json").read_text())
    requirements = report["per_requirement"]
    actual = {}
    for key, row in requirements.items():
        if row["level"] not in {"MUST", "SHOULD", "MAY"}:
            raise ValueError(f"Unknown TCK requirement level: {key}")
        if row["level"] == "MUST":
            if row["status"] == "FAIL" or "FAIL" in row["transports"].values():
                raise ValueError(f"TCK MUST failure: {key}: {row['errors']}")
            actual[key] = {"status": row["status"], "transports": row["transports"]}
    if actual != expected:
        changed = sorted(key for key in actual.keys() | expected.keys()
                         if actual.get(key) != expected.get(key))
        raise ValueError(f"TCK MUST coverage changed; review the raw evidence: {changed}")
    if level == "full":
        for key, expected_row in advisories.items():
            row = requirements.get(key)
            if not isinstance(row, dict) or any(
                row.get(field) != expected_row[field]
                for field in ("level", "status", "transports")
            ) or set(row.get("test_ids", ())) != set(expected_row["test_ids"]):
                raise ValueError(f"TCK advisory coverage changed; review raw evidence: {key}")

    summary = report["summary"]
    if summary["score_scope"] != "automated" or (
        summary["must_compatibility"] != profile["must_compatibility"]
    ):
        raise ValueError("TCK raw automated score differs from the reviewed coverage")
    bindings = {row["protocolBinding"] for row in report["agent_card"]["supportedInterfaces"]}
    if bindings != set(profile["protocol_bindings"]):
        raise ValueError("TCK did not exercise the reviewed native publication")

    junit = ElementTree.parse(reports / "junitreport.xml").getroot()
    cases = list(junit.iter("testcase"))
    if not cases or list(junit.iter("error")):
        raise ValueError("TCK JUnit contains an error or no collected checks")
    outcomes = {}
    for case in cases:
        identity = case.attrib["classname"], case.attrib["name"]
        if identity in outcomes:
            raise ValueError(f"Duplicate TCK JUnit identity: {identity}")
        outcomes[identity] = (
            "FAIL" if case.find("failure") is not None else
            "SKIPPED" if case.find("skipped") is not None else "PASS"
        )
    failures = {identity for identity, outcome in outcomes.items() if outcome == "FAIL"}
    if bool(failures) != (returncode == 1):
        raise ValueError("TCK exit status and JUnit failures disagree")
    if level == "must" and failures:
        raise ValueError("TCK MUST-only run has failed checks")
    if level == "full" and failures:
        raise ValueError("TCK full profile has failed checks")
    if level == "full":
        for key, row in requirements.items():
            if row["status"] == "FAIL" or "FAIL" in row["transports"].values():
                raise ValueError(f"TCK full profile reports failed requirement: {key}")
        for key, row in advisories.items():
            if any(outcomes.get(_junit_address(node)) != "PASS" for node in row["test_ids"]):
                raise ValueError(f"TCK advisory pass lacks passing JUnit probes: {key}")
    for key, row in requirements.items():
        if row["level"] == "MUST" and row["status"] == "PASS" and not any(
            outcomes.get(_junit_address(node)) == "PASS" for node in row["test_ids"]
        ):
            raise ValueError(f"TCK MUST pass has no passing JUnit check: {key}")
    return report
