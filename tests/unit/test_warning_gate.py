"""Unit tests for the #2391 warning-count aggregation + gate.

Covers ``scripts/warning_gate.py``: per-worker aggregation, artifact writing,
baseline seeding (informational when a category is unset), and the fail path
when runtime/resource warnings rise above the committed baseline.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_GATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "warning_gate.py"
_spec = importlib.util.spec_from_file_location("warning_gate", _GATE_PATH)
warning_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(warning_gate)


def _write_worker(report_dir: Path, worker: str, by_category: dict) -> None:
    payload = {
        "worker": worker,
        "total": sum(by_category.values()),
        "by_category": by_category,
    }
    (report_dir / f"warnings-{worker}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_aggregate_sums_across_workers(tmp_path):
    _write_worker(tmp_path, "gw0", {"RuntimeWarning": 2, "DeprecationWarning": 5})
    _write_worker(tmp_path, "gw1", {"RuntimeWarning": 3, "ResourceWarning": 1})

    report = warning_gate.aggregate(tmp_path)

    assert report["by_category"] == {
        "DeprecationWarning": 5,
        "ResourceWarning": 1,
        "RuntimeWarning": 5,
    }
    assert report["total"] == 11
    assert report["workers"] == ["gw0", "gw1"]
    assert report["source_files"] == 2


def test_aggregate_skips_unreadable_file(tmp_path):
    _write_worker(tmp_path, "gw0", {"RuntimeWarning": 1})
    (tmp_path / "warnings-bad.json").write_text("not json", encoding="utf-8")

    report = warning_gate.aggregate(tmp_path)

    assert report["by_category"] == {"RuntimeWarning": 1}


def test_regression_fails_above_baseline_plus_tolerance(tmp_path):
    # baseline 2 + tolerance max(5, ceil(2*0.25)=1) = 7; observed 20 clears it.
    report = {"by_category": {"RuntimeWarning": 20}, "total": 20}
    baseline = {"RuntimeWarning": 2}

    failures = warning_gate.check_regressions(report, baseline)

    assert len(failures) == 1
    assert "RuntimeWarning" in failures[0]
    assert "+18 vs baseline" in failures[0]


def test_jitter_within_tolerance_does_not_fail(tmp_path):
    # baseline 4 + tolerance 5 = 9; observed 7 is jitter, not a regression.
    report = {"by_category": {"RuntimeWarning": 7}, "total": 7}
    baseline = {"RuntimeWarning": 4}

    assert warning_gate.check_regressions(report, baseline) == []


def test_zero_baseline_gates_strictly_even_with_default_tolerance():
    # A ``ResourceWarning: 0`` baseline must reject the very first new leak,
    # even under the default jitter tolerance (abs 5). The tolerance only
    # applies above an established nonzero high-water mark (#2391).
    report = {"by_category": {"ResourceWarning": 1}, "total": 1}
    baseline = {"ResourceWarning": 0}

    failures = warning_gate.check_regressions(report, baseline)

    assert len(failures) == 1
    assert "ResourceWarning" in failures[0]


def test_threshold_zero_baseline_is_strict():
    assert warning_gate._threshold(0, 5, 0.25) == 0
    # A nonzero baseline still gets the jitter margin.
    assert warning_gate._threshold(4, 5, 0.25) == 9


def test_zero_tolerance_gates_strictly(tmp_path):
    report = {"by_category": {"ResourceWarning": 1}, "total": 1}
    baseline = {"ResourceWarning": 0}

    failures = warning_gate.check_regressions(report, baseline, tol_abs=0, tol_rel=0.0)

    assert len(failures) == 1
    assert "ResourceWarning" in failures[0]


def test_regression_ok_at_or_below_baseline(tmp_path):
    report = {"by_category": {"RuntimeWarning": 2, "ResourceWarning": 0}, "total": 2}
    baseline = {"RuntimeWarning": 2, "ResourceWarning": 1}

    assert warning_gate.check_regressions(report, baseline) == []


def test_unset_baseline_is_informational_not_failing():
    report = {"by_category": {"RuntimeWarning": 7}, "total": 7}
    baseline = {}  # no entry for RuntimeWarning → informational only

    assert warning_gate.check_regressions(report, baseline) == []


def test_main_writes_artifact_and_gates(tmp_path):
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _write_worker(report_dir, "gw0", {"ResourceWarning": 20})

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"by_category": {"ResourceWarning": 2}}), encoding="utf-8"
    )
    out_path = tmp_path / "warning-report.json"

    rc = warning_gate.main(
        [
            "--report-dir",
            str(report_dir),
            "--out",
            str(out_path),
            "--baseline",
            str(baseline_path),
        ]
    )

    assert rc == 1  # 20 observed > baseline 2 + tolerance 5 = 7
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["by_category"] == {"ResourceWarning": 20}


def test_update_baseline_writes_and_passes(tmp_path):
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _write_worker(report_dir, "gw0", {"RuntimeWarning": 3})

    baseline_path = tmp_path / "baseline.json"
    out_path = tmp_path / "warning-report.json"

    rc = warning_gate.main(
        [
            "--report-dir",
            str(report_dir),
            "--out",
            str(out_path),
            "--baseline",
            str(baseline_path),
            "--update-baseline",
        ]
    )

    assert rc == 0
    written = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert written == {"by_category": {"RuntimeWarning": 3}}


def test_missing_report_dir_returns_error(tmp_path):
    rc = warning_gate.main(
        ["--report-dir", str(tmp_path / "nope"), "--baseline", str(tmp_path / "b.json")]
    )
    assert rc == 1
