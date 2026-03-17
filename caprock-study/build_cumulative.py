"""
Caprock Clinical Validation Study — Cumulative Tracker
=======================================================
Reads all daily snapshots and builds a rolling cumulative.json
plus a simple ASCII trend chart for quick review.

Usage:
  python build_cumulative.py
  python build_cumulative.py --since 2026-03-17
"""

import os
import json
import glob
import argparse
from datetime import date, datetime


SNAPSHOTS_DIR = "caprock-study/snapshots"
CUMULATIVE_PATH = "caprock-study/cumulative.json"


def load_snapshots(since: date = None) -> list:
    paths = sorted(glob.glob(f"{SNAPSHOTS_DIR}/*.json"))
    snaps = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        snap_date = date.fromisoformat(data["study_date"])
        if since and snap_date < since:
            continue
        snaps.append(data)
    return snaps


def build_cumulative(snaps: list) -> dict:
    if not snaps:
        return {}

    total_days = len(snaps)
    all_response_rates = [s["daily_sms"]["response_rate_pct"] for s in snaps]
    avg_response_rate  = round(sum(all_response_rates) / total_days, 1) if total_days else 0

    # Per-patient rolling adherence
    patient_days  = {}   # patient_name → days they replied
    patient_total = {}   # patient_name → days they were active

    for snap in snaps:
        for p in snap.get("patient_details", []):
            name = p["name"]
            patient_total[name] = patient_total.get(name, 0) + 1
            if p.get("replied_today"):
                patient_days[name] = patient_days.get(name, 0) + 1

    per_patient_adherence = {}
    for name, total in patient_total.items():
        replied = patient_days.get(name, 0)
        per_patient_adherence[name] = {
            "days_active":   total,
            "days_replied":  replied,
            "adherence_pct": round(replied / total * 100, 1) if total else 0,
        }

    # Study-level trend (last 7 days vs prior 7)
    if total_days >= 7:
        last7  = snaps[-7:]
        prior7 = snaps[-14:-7] if total_days >= 14 else []
        last7_avg  = round(sum(s["daily_sms"]["response_rate_pct"] for s in last7) / len(last7), 1)
        prior7_avg = round(sum(s["daily_sms"]["response_rate_pct"] for s in prior7) / len(prior7), 1) if prior7 else None
        study_trend = "improving" if (prior7_avg and last7_avg > prior7_avg) else \
                      "declining" if (prior7_avg and last7_avg < prior7_avg) else "stable"
    else:
        last7_avg  = avg_response_rate
        prior7_avg = None
        study_trend = "insufficient_data"

    # Escalations
    total_escalations = sum(
        s.get("daily_kestrel_ai", {}).get("escalations", 0) for s in snaps
    )

    return {
        "generated_at":          datetime.utcnow().isoformat() + "Z",
        "study_start":           snaps[0]["study_date"],
        "study_latest":          snaps[-1]["study_date"],
        "total_days_collected":  total_days,
        "cohort_size":           snaps[-1]["cohort_size"],
        "avg_daily_response_rate_pct": avg_response_rate,
        "last_7d_response_rate_pct":   last7_avg,
        "prior_7d_response_rate_pct":  prior7_avg,
        "study_trend":           study_trend,
        "total_escalations":     total_escalations,
        "per_patient_adherence": per_patient_adherence,
        "daily_series": [
            {
                "date":          s["study_date"],
                "response_rate": s["daily_sms"]["response_rate_pct"],
                "patients_replied": s["daily_sms"]["patients_replied"],
                "patients_messaged": s["daily_sms"]["patients_messaged"],
                "at_risk": s.get("at_risk_patients", []),
            }
            for s in snaps
        ],
    }


def ascii_trend_chart(cumulative: dict, width: int = 40) -> str:
    series = cumulative.get("daily_series", [])
    if not series:
        return "(no data yet)"
    rates = [s["response_rate"] for s in series]
    max_r = max(rates) if rates else 100
    lines = []
    lines.append(f"Daily Response Rate — {series[0]['date']} to {series[-1]['date']}")
    lines.append(f"{'100%':>5} |")
    for pct in [75, 50, 25, 0]:
        bar_chars = ""
        for r in rates[-width:]:
            bar_chars += "█" if r >= pct else " "
        lines.append(f"{pct:>4}% |{bar_chars}")
    lines.append(f"       +{'-' * min(len(rates), width)}")
    lines.append(f"  avg={cumulative['avg_daily_response_rate_pct']}%  trend={cumulative['study_trend']}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default=None, help="Only include snapshots from this date onward")
    args = parser.parse_args()

    since = date.fromisoformat(args.since) if args.since else None
    snaps = load_snapshots(since)
    print(f"Loaded {len(snaps)} snapshots")

    cumulative = build_cumulative(snaps)
    with open(CUMULATIVE_PATH, "w") as f:
        json.dump(cumulative, f, indent=2, default=str)
    print(f"Saved {CUMULATIVE_PATH}")
    print()
    print(ascii_trend_chart(cumulative))
