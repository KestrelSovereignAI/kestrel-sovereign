"""The report gate preserves coverage gaps and refuses false green evidence."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from tests.utils.native_tck_report import validate_tck_report


@pytest.fixture
def evidence(tmp_path):
    """File-format samples for the parser, not an A2A peer or conformance run."""
    reports = tmp_path / "reports"
    provenance = tmp_path / "provenance"
    reports.mkdir()
    provenance.mkdir()
    tree = "a" * 40
    (provenance / "SOURCE.toml").write_text(
        f'patched_tree = "{tree}"\nprotocol_version = "1.0.1"\n'
    )
    requirements = {
        "DM-MSG-001": {
            "level": "MUST", "status": "PASS", "errors": [],
            "transports": {"jsonrpc": "PASS", "http_json": "PASS"},
            "test_ids": ["tests/wire/test_message.py::TestMessage::test_reply[part::text]"],
        },
        "VER-CLIENT-002": {
            "level": "MUST", "status": "NOT TESTED", "errors": [],
            "transports": {}, "test_ids": [],
        },
        "CARD-CACHE-001": {
            "level": "SHOULD", "status": "PASS", "errors": [],
            "transports": {"agent_card": "PASS"},
            "test_ids": ["tests/wire/test_card.py::test_cache"],
        },
    }
    report = {
        "summary": {"score_scope": "automated", "must_compatibility": "50.0%"},
        "per_requirement": requirements,
        "agent_card": {"supportedInterfaces": [
            {"protocolBinding": "JSONRPC"}, {"protocolBinding": "HTTP+JSON"},
        ]},
    }
    profile = {
        "schema_version": 2, "patched_tree": tree, "protocol_version": "1.0.1",
        "protocol_bindings": ["JSONRPC", "HTTP+JSON"], "must_compatibility": "50.0%",
        "unverified_requirements": {"VER-CLIENT-002": "Patch negotiation has no probe."},
        "required_advisory_requirements": {
            "CARD-CACHE-001": {key: requirements["CARD-CACHE-001"][key]
                               for key in ("level", "status", "transports", "test_ids")},
        },
        "must_requirements": {
            key: {"status": row["status"], "transports": row["transports"]}
            for key, row in requirements.items() if row["level"] == "MUST"
        },
    }
    (reports / "compatibility.json").write_text(json.dumps(report))
    (provenance / "COVERAGE.json").write_text(json.dumps(profile))
    (reports / "junitreport.xml").write_text(
        '<testsuites><testsuite>'
        '<testcase classname="tests.wire.test_message.TestMessage" name="test_reply[part::text]"/>'
        '<testcase classname="tests.wire.test_card" name="test_cache"/>'
        '</testsuite></testsuites>'
    )
    return reports, provenance


def _validate(evidence, *, level="full", returncode=0):
    reports, provenance = evidence
    return validate_tck_report(reports, level=level, returncode=returncode, provenance_dir=provenance)


def _rewrite(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))



def _fail_advisory(evidence):
    reports, _ = evidence
    _rewrite(reports / "compatibility.json", lambda data:
             data["per_requirement"]["CARD-CACHE-001"].update(
                 status="FAIL", transports={"agent_card": "FAIL"}, errors=["no max-age"]))
    path = reports / "junitreport.xml"
    tree = ElementTree.parse(path)
    ElementTree.SubElement(tree.getroot().find("testsuite")[1], "failure").text = "no max-age"
    tree.write(path)

def test_full_pass_retains_unmodified_raw_gaps(evidence):
    reports, _ = evidence
    before = (reports / "compatibility.json").read_bytes()
    report = _validate(evidence)
    assert report["per_requirement"]["VER-CLIENT-002"]["status"] == "NOT TESTED"
    assert report["per_requirement"]["CARD-CACHE-001"]["status"] == "PASS"
    assert report["summary"]["must_compatibility"] == "50.0%"
    assert (reports / "compatibility.json").read_bytes() == before


@pytest.mark.parametrize("status", ["FAIL", "SKIPPED", "MANUAL", "NOT TESTED"])
def test_must_positive_cannot_disappear(evidence, status):
    reports, _ = evidence
    _rewrite(reports / "compatibility.json",
             lambda data: data["per_requirement"]["DM-MSG-001"].update(status=status))
    with pytest.raises(ValueError, match="MUST failure|MUST coverage changed"):
        _validate(evidence)


@pytest.mark.parametrize("binding", ["jsonrpc", "http_json"])
@pytest.mark.parametrize("status", ["FAIL", "SKIPPED"])
def test_each_previously_passing_binding_is_required(evidence, binding, status):
    reports, _ = evidence
    _rewrite(reports / "compatibility.json", lambda data:
             data["per_requirement"]["DM-MSG-001"]["transports"].update({binding: status}))
    with pytest.raises(ValueError, match="MUST failure|MUST coverage changed"):
        _validate(evidence)


@pytest.mark.parametrize("mutation", ["remove", "add", "change_gap", "change_level"])
def test_requirement_inventory_requires_review(evidence, mutation):
    reports, _ = evidence
    def change(data):
        rows = data["per_requirement"]
        if mutation == "remove":
            del rows["VER-CLIENT-002"]
        elif mutation == "add":
            rows["NEW-001"] = dict(rows["VER-CLIENT-002"])
        elif mutation == "change_gap":
            rows["VER-CLIENT-002"]["status"] = "MANUAL"
        else:
            rows["DM-MSG-001"]["level"] = "SHOULD"
    _rewrite(reports / "compatibility.json", change)
    with pytest.raises(ValueError, match="MUST coverage changed"):
        _validate(evidence)


@pytest.mark.parametrize("mutation", ["tree", "protocol", "schema", "empty", "no_positive", "gap", "blank_gap"])
def test_coverage_profile_must_be_complete_and_bound_to_source(evidence, mutation):
    _, provenance = evidence
    def change(data):
        if mutation == "tree":
            data["patched_tree"] = "b" * 40
        elif mutation == "protocol":
            data["protocol_version"] = "1.0.0"
        elif mutation == "schema":
            data["schema_version"] = 3
        elif mutation == "empty":
            data["must_requirements"] = {}
        elif mutation == "no_positive":
            data["must_requirements"]["DM-MSG-001"]["status"] = "SKIPPED"
        elif mutation == "gap":
            data["unverified_requirements"] = {}
        else:
            data["unverified_requirements"]["VER-CLIENT-002"] = " "
    _rewrite(provenance / "COVERAGE.json", change)
    with pytest.raises(ValueError, match="profile|coverage gaps"):
        _validate(evidence)


@pytest.mark.parametrize("field,value", [("must_compatibility", "100.0%"), ("score_scope", "manual")])
def test_fabricated_score_is_not_a_pass(evidence, field, value):
    reports, _ = evidence
    _rewrite(reports / "compatibility.json", lambda data: data["summary"].update({field: value}))
    with pytest.raises(ValueError, match="raw automated score"):
        _validate(evidence)


@pytest.mark.parametrize("returncode", [-9, 2, 3, 4, 5])
def test_runner_abort_is_never_advisory(evidence, returncode):
    with pytest.raises(ValueError, match="did not finish"):
        _validate(evidence, returncode=returncode)


@pytest.mark.parametrize("mutation", ["error", "empty", "unknown_failure", "mixed_level", "duplicate", "no_pass"])
def test_junit_errors_and_unaccounted_failures_are_fatal(evidence, mutation):
    reports, _ = evidence
    if mutation in {"unknown_failure", "mixed_level"}:
        _fail_advisory(evidence)
    path = reports / "junitreport.xml"
    tree = ElementTree.parse(path)
    suite = tree.getroot().find("testsuite")
    if mutation == "error":
        ElementTree.SubElement(suite[0], "error").text = "fixture failed"
    elif mutation == "empty":
        suite.clear()
    elif mutation == "unknown_failure":
        suite[1].set("name", "test_unmapped")
    elif mutation == "mixed_level":
        _rewrite(reports / "compatibility.json", lambda data:
                 data["per_requirement"]["DM-MSG-001"]["test_ids"].append("tests/wire/test_card.py::test_cache"))
    elif mutation == "duplicate":
        suite.append(ElementTree.fromstring(ElementTree.tostring(suite[0])))
    else:
        ElementTree.SubElement(suite[0], "skipped")
    tree.write(path)
    with pytest.raises(ValueError, match="JUnit|advisory coverage|failed checks"):
        _validate(evidence, returncode=1 if mutation in {"unknown_failure", "mixed_level"} else 0)


def test_must_phase_never_accepts_failed_advisory_checks(evidence):
    _fail_advisory(evidence)
    with pytest.raises(ValueError, match="MUST-only run"):
        _validate(evidence, level="must", returncode=1)


def test_successful_must_phase(evidence):
    _validate(evidence, level="must", returncode=0)


def test_exit_zero_cannot_hide_assertion_failures(evidence):
    reports, _ = evidence
    path = reports / "junitreport.xml"
    tree = ElementTree.parse(path)
    ElementTree.SubElement(tree.getroot().find("testsuite")[1], "failure").text = "hidden assertion"
    tree.write(path)
    with pytest.raises(ValueError, match="exit status and JUnit"):
        _validate(evidence, returncode=0)


@pytest.mark.parametrize("filename", ["compatibility.json", "junitreport.xml"])
def test_missing_report_is_fatal(evidence, filename):
    reports, _ = evidence
    (reports / filename).unlink()
    with pytest.raises(FileNotFoundError):
        _validate(evidence)


def test_wrong_publication_is_fatal(evidence):
    reports, _ = evidence
    _rewrite(reports / "compatibility.json", lambda data:
             data["agent_card"].update(supportedInterfaces=[{"protocolBinding": "JSONRPC"}]))
    with pytest.raises(ValueError, match="native publication"):
        _validate(evidence)


def test_checked_in_profile_preserves_all_known_obligations():
    provenance = Path(__file__).resolve().parents[1] / "interop/a2a_tck"
    profile = json.loads((provenance / "COVERAGE.json").read_text())
    assert len(profile["must_requirements"]) == 115
    assert len(profile["unverified_requirements"]) == 16
    assert profile["must_compatibility"] == "81.8%"
    assert set(profile["required_advisory_requirements"]) == {"CARD-CACHE-001", "HTTP_JSON-SVC-001"}
    for row in profile["required_advisory_requirements"].values():
        assert row["level"] == "SHOULD" and row["status"] == "PASS"
        assert set(row["transports"].values()) == {"PASS"}
        assert len(row["test_ids"]) == 2
    for key in ("DM-MSG-001", "STREAM-SUB-002"):
        assert profile["must_requirements"][key]["status"] == "PASS"
        for binding in ("jsonrpc", "http_json"):
            assert profile["must_requirements"][key]["transports"][binding] == "PASS"


@pytest.mark.parametrize("mutation", ["remove", "skip", "manual", "change_binding", "weaken_level", "lose_probe"])
def test_resolved_advisory_coverage_cannot_disappear(evidence, mutation):
    reports, _ = evidence
    def change(data):
        rows = data["per_requirement"]
        if mutation == "remove":
            del rows["CARD-CACHE-001"]
        elif mutation == "skip":
            rows["CARD-CACHE-001"]["status"] = "SKIPPED"
        elif mutation == "manual":
            rows["CARD-CACHE-001"]["status"] = "MANUAL"
        elif mutation == "change_binding":
            rows["CARD-CACHE-001"]["transports"] = {"http_json": "PASS"}
        elif mutation == "weaken_level":
            rows["CARD-CACHE-001"]["level"] = "MAY"
        else:
            rows["CARD-CACHE-001"]["test_ids"] = []
    _rewrite(reports / "compatibility.json", change)
    with pytest.raises(ValueError, match="advisory coverage"):
        _validate(evidence)


@pytest.mark.parametrize("mutation", ["missing", "empty", "failed", "no_probes"])
def test_resolved_advisory_profile_cannot_be_waived(evidence, mutation):
    _, provenance = evidence
    def change(data):
        if mutation == "missing":
            del data["required_advisory_requirements"]
        elif mutation == "empty":
            data["required_advisory_requirements"] = {}
        elif mutation == "failed":
            data["required_advisory_requirements"]["CARD-CACHE-001"]["status"] = "FAIL"
        else:
            data["required_advisory_requirements"]["CARD-CACHE-001"]["test_ids"] = []
    _rewrite(provenance / "COVERAGE.json", change)
    with pytest.raises(ValueError, match="advisory coverage profile"):
        _validate(evidence)


@pytest.mark.parametrize("mutation", ["remove", "skip"])
def test_every_resolved_advisory_probe_needs_a_passing_junit_case(evidence, mutation):
    reports, _ = evidence
    path = reports / "junitreport.xml"
    tree = ElementTree.parse(path)
    suite = tree.getroot().find("testsuite")
    if mutation == "remove":
        suite.remove(suite[1])
    else:
        ElementTree.SubElement(suite[1], "skipped")
    tree.write(path)
    with pytest.raises(ValueError, match="advisory.*JUnit"):
        _validate(evidence)


@pytest.mark.parametrize("level", ["SHOULD", "MAY"])
def test_full_profile_rejects_other_failed_advisories(evidence, level):
    reports, _ = evidence
    _rewrite(reports / "compatibility.json", lambda data:
             data["per_requirement"].update({"OPTIONAL-001": {
                 "level": level, "status": "FAIL", "errors": ["another failure"],
                 "transports": {"jsonrpc": "FAIL"},
                 "test_ids": ["tests/wire/test_optional.py::test_optional"],
             }}))
    path = reports / "junitreport.xml"
    tree = ElementTree.parse(path)
    case = ElementTree.SubElement(tree.getroot().find("testsuite"), "testcase",
                                  classname="tests.wire.test_optional", name="test_optional")
    ElementTree.SubElement(case, "failure").text = "another failure"
    tree.write(path)
    with pytest.raises(ValueError, match="full profile has failed checks"):
        _validate(evidence, returncode=1)
