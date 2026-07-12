#!/usr/bin/env python3
"""Aggregate the test-suite warning tally and gate on regressions (#2391).

The unit suite historically passed while emitting thousands of warnings, which
buried real lifecycle failures (unawaited coroutines, unclosed resources). This
script turns that noise into a first-class, enforceable signal:

1. It reads every per-worker ``warnings-*.json`` file written by the pytest
   hook in ``tests/conftest.py`` (active when ``KESTREL_WARNING_REPORT_DIR`` is
   set) and merges them into a single ``warning-report.json`` artifact that CI
   uploads.
2. It compares the *actionable* warning categories — ``RuntimeWarning`` and
   ``ResourceWarning`` — against a committed baseline
   (``tests/warning_baseline.json``) and exits non-zero when either rises above
   its baseline. Existing warnings are not blanket-suppressed: the baseline is
   the agreed high-water mark, and only *new* runtime/resource warnings fail
   the build.

Usage::

    python scripts/warning_gate.py --report-dir .warning-report \
        --out warning-report.json --baseline tests/warning_baseline.json

    # Re-baseline after intentionally accepting the current counts:
    python scripts/warning_gate.py --report-dir .warning-report \
        --baseline tests/warning_baseline.json --update-baseline
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Categories whose regressions fail the build. These are the actionable
# lifecycle/leak signals called out in #2391. DeprecationWarning and friends
# are reported in the artifact but not gated (yet).
GATED_CATEGORIES = ("RuntimeWarning", "ResourceWarning")

# "coroutine never awaited" RuntimeWarnings and unclosed-resource
# ResourceWarnings fire on GC timing, so the exact per-run count jitters a
# little across platforms and xdist worker counts. The gate therefore allows
# a small margin above baseline — enough to absorb that jitter, but far below
# the size of a real regression wave (a PR that leaks a resource in a shared
# fixture adds dozens). Threshold = baseline + max(abs, ceil(baseline*rel)).
DEFAULT_TOLERANCE_ABS = 5
DEFAULT_TOLERANCE_REL = 0.25


def aggregate(report_dir: Path) -> dict:
    """Merge every per-worker ``warnings-*.json`` file into one tally."""
    by_category: dict[str, int] = {}
    workers: list[str] = []
    files = sorted(report_dir.glob("warnings-*.json"))
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            print(f"warning_gate: skipping unreadable {path}: {exc}", file=sys.stderr)
            continue
        workers.append(payload.get("worker", path.stem))
        for name, count in (payload.get("by_category") or {}).items():
            by_category[name] = by_category.get(name, 0) + int(count)
    return {
        "total": sum(by_category.values()),
        "by_category": dict(sorted(by_category.items())),
        "workers": sorted(workers),
        "source_files": len(files),
    }


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("by_category", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _threshold(baseline: int, tol_abs: int, tol_rel: float) -> int:
    """Allowed ceiling for a gated category.

    A **zero baseline is enforced strictly**: a category with no accepted
    warnings has nothing to jitter around, so the very first new warning is a
    regression (ceiling 0). The jitter tolerance exists only to absorb GC-timing
    noise *around an established nonzero high-water mark* — applying it to a zero
    baseline would silently let the first N leaks through, which is exactly the
    ``ResourceWarning: 0`` hole #2391 is meant to close.

    For a nonzero baseline:
        ceiling = baseline + max(absolute, ceil(baseline * relative)).
    """
    if baseline <= 0:
        return 0
    return baseline + max(tol_abs, math.ceil(baseline * tol_rel))


def check_regressions(
    report: dict,
    baseline: dict,
    tol_abs: int = DEFAULT_TOLERANCE_ABS,
    tol_rel: float = DEFAULT_TOLERANCE_REL,
) -> list[str]:
    """Return human-readable failure messages for gated-category regressions."""
    failures: list[str] = []
    observed = report["by_category"]
    for category in GATED_CATEGORIES:
        seen = observed.get(category, 0)
        allowed = baseline.get(category)
        if allowed is None:
            # No baseline recorded for this category yet — informational only.
            if seen:
                print(
                    f"warning_gate: no baseline for {category}; observed {seen} "
                    f"(commit tests/warning_baseline.json to enforce)",
                    file=sys.stderr,
                )
            continue
        ceiling = _threshold(allowed, tol_abs, tol_rel)
        if seen > ceiling:
            failures.append(
                f"{category}: {seen} observed > {ceiling} allowed "
                f"(baseline {allowed} + tolerance; +{seen - allowed} vs baseline)"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        default=".warning-report",
        help="Directory holding per-worker warnings-*.json files.",
    )
    parser.add_argument(
        "--out",
        default="warning-report.json",
        help="Path for the aggregated warning-count artifact.",
    )
    parser.add_argument(
        "--baseline",
        default="tests/warning_baseline.json",
        help="Committed baseline of allowed warning counts.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite the baseline with the observed counts and exit 0.",
    )
    parser.add_argument(
        "--tolerance-abs",
        type=int,
        default=DEFAULT_TOLERANCE_ABS,
        help="Absolute jitter margin allowed above baseline per category.",
    )
    parser.add_argument(
        "--tolerance-rel",
        type=float,
        default=DEFAULT_TOLERANCE_REL,
        help="Relative jitter margin (fraction of baseline) allowed above baseline.",
    )
    args = parser.parse_args(argv)

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        print(
            f"warning_gate: report dir {report_dir} missing — did the suite run "
            f"with KESTREL_WARNING_REPORT_DIR set?",
            file=sys.stderr,
        )
        return 1

    report = aggregate(report_dir)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"warning_gate: wrote {out_path} (total={report['total']})")
    for name, count in report["by_category"].items():
        print(f"  {name}: {count}")

    baseline_path = Path(args.baseline)
    if args.update_baseline:
        baseline_path.write_text(
            json.dumps({"by_category": report["by_category"]}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"warning_gate: baseline updated at {baseline_path}")
        return 0

    baseline = load_baseline(baseline_path)
    failures = check_regressions(
        report, baseline, tol_abs=args.tolerance_abs, tol_rel=args.tolerance_rel
    )
    if failures:
        print("\nwarning_gate: FAILED — new actionable warnings:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print("warning_gate: OK — no runtime/resource warning regressions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
