"""
Generate realistic mock data for Caprock dashboard testing.
Creates baseline.json + 14 days of daily snapshots + cumulative.json.

Usage:
  python caprock-study/generate_mock_data.py
"""

import json
import os
import random
from datetime import date, timedelta

random.seed(42)

SNAPSHOTS_DIR = "caprock-study/snapshots"
BASELINE_PATH = "caprock-study/baseline.json"

PATIENTS = [
    {"id": "1001", "name": "Martinez, Rosa", "rpm_active": True, "rpm_medicaid": True},
    {"id": "1002", "name": "Johnson, Earl", "rpm_active": True, "rpm_medicaid": True},
    {"id": "1003", "name": "Thompson, Betty", "rpm_active": True, "rpm_medicaid": True},
    {"id": "1004", "name": "Garcia, Manuel", "rpm_active": True, "rpm_medicaid": True},
    {"id": "1005", "name": "Williams, Dorothy", "rpm_active": True, "rpm_medicaid": True},
    {"id": "1006", "name": "Davis, James", "rpm_active": True, "rpm_medicaid": True},
    {"id": "1007", "name": "Wilson, Mary", "rpm_active": True, "rpm_medicaid": True},
    {"id": "1008", "name": "Brown, Robert", "rpm_active": True, "rpm_medicaid": True},
    {"id": "1009", "name": "Jones, Linda", "rpm_active": True, "rpm_medicaid": True},
    {"id": "1010", "name": "Miller, Charles", "rpm_active": True, "rpm_medicaid": True},
]

# Baseline reply probabilities (pre-Kestrel, lower engagement)
BASELINE_REPLY_PROB = {
    "1001": 0.45, "1002": 0.30, "1003": 0.55, "1004": 0.20,
    "1005": 0.60, "1006": 0.10, "1007": 0.40, "1008": 0.35,
    "1009": 0.50, "1010": 0.15,
}

# Post-Kestrel reply probabilities (improved engagement)
KESTREL_REPLY_PROB = {
    "1001": 0.70, "1002": 0.50, "1003": 0.75, "1004": 0.40,
    "1005": 0.80, "1006": 0.25, "1007": 0.65, "1008": 0.55,
    "1009": 0.72, "1010": 0.35,
}

STUDY_START = date(2026, 3, 17)
BASELINE_DAYS = 30


def generate_baseline():
    """Generate pre-Kestrel baseline.json."""
    start_date = STUDY_START - timedelta(days=BASELINE_DAYS)

    per_patient_sms = {}
    for p in PATIENTS:
        pid = p["id"]
        prob = BASELINE_REPLY_PROB[pid]
        days_replied = sum(1 for _ in range(BASELINE_DAYS) if random.random() < prob)
        outbound = BASELINE_DAYS  # one message per day
        inbound = days_replied
        per_patient_sms[pid] = {
            "patient_id": pid,
            "outbound": outbound,
            "inbound": inbound,
            "days_replied": days_replied,
            "baseline_days": BASELINE_DAYS,
            "response_adherence_pct": round(days_replied / BASELINE_DAYS * 100, 1),
        }

    total_outbound = sum(p["outbound"] for p in per_patient_sms.values())
    total_inbound = sum(p["inbound"] for p in per_patient_sms.values())
    silent = [p["name"] for p in PATIENTS if per_patient_sms[p["id"]]["inbound"] == 0]

    all_adherence = [p["response_adherence_pct"] for p in per_patient_sms.values()]

    baseline = {
        "generated_at": "2026-03-17T00:00:00Z",
        "golive_date": STUDY_START.isoformat(),
        "baseline_window": {
            "start": start_date.isoformat(),
            "end": STUDY_START.isoformat(),
            "days": BASELINE_DAYS,
        },
        "cohort": {
            "total_patients": len(PATIENTS),
            "patient_list": [
                {"id": p["id"], "name": p["name"],
                 "rpm_active": p["rpm_active"], "rpm_medicaid": p["rpm_medicaid"]}
                for p in PATIENTS
            ],
        },
        "sms_baseline": {
            "summary": {
                "total_messages": total_outbound + total_inbound,
                "outbound_total": total_outbound,
                "inbound_total": total_inbound,
                "patients_messaged": 10,
                "patients_replied": len([p for p in per_patient_sms.values() if p["inbound"] > 0]),
                "days_with_activity": BASELINE_DAYS,
            },
            "response_rate_pct": round(total_inbound / total_outbound * 100, 1),
            "avg_patient_adherence_pct": round(sum(all_adherence) / len(all_adherence), 1),
            "silent_patients": silent,
            "per_patient": per_patient_sms,
        },
        "vitals_baseline": {
            "status": "pending_schema_confirmation",
            "ref": "jaslogic1/RemoteCares#68",
        },
        "study_hypothesis": {
            "metric": "reading_compliance_pct",
            "target_delta": "+20% over baseline",
        },
    }
    return baseline


def generate_daily_snapshot(target_date, day_number):
    """Generate a single day's metrics snapshot."""
    patients_replied = 0
    patient_details = []
    at_risk = []
    total_inbound = 0
    total_outbound = 0

    for p in PATIENTS:
        pid = p["id"]
        prob = KESTREL_REPLY_PROB[pid]
        # Slightly increase probability over time (Kestrel learning)
        adjusted_prob = min(prob + day_number * 0.005, 0.95)
        replied = random.random() < adjusted_prob
        out = random.randint(1, 3)
        inb = random.randint(1, 2) if replied else 0
        total_outbound += out
        total_inbound += inb

        if replied:
            patients_replied += 1

        days_silent = 0 if replied else random.randint(0, 5)
        streak = random.randint(0, day_number) if replied else 0

        rec = {
            "patient_id": pid,
            "name": p["name"],
            "sms_out_today": out,
            "sms_in_today": inb,
            "replied_today": replied,
            "reply_streak_days": streak,
            "days_since_last_reply": days_silent,
            "weekly_trend": {"this_week_reply_days": random.randint(3, 7),
                             "prev_week_reply_days": random.randint(2, 6),
                             "trend": random.choice(["improving", "stable", "improving"])},
            "flag_at_risk": days_silent >= 3,
        }
        patient_details.append(rec)
        if rec["flag_at_risk"]:
            at_risk.append(p["name"])

    response_rate = round(patients_replied / 10 * 100, 1)

    week_number = (day_number // 7) + 1
    if week_number <= 4:
        phase = {"phase": 1, "week": week_number, "name": "Safe Zone Only",
                 "target_improvement_pct": 10, "triage_zone_active": False, "after_hours_active": False}
    elif week_number <= 8:
        phase = {"phase": 2, "week": week_number, "name": "Add Triage Zone",
                 "target_improvement_pct": 20, "triage_zone_active": True, "after_hours_active": False}
    else:
        phase = {"phase": 3, "week": week_number, "name": "After-Hours Expansion",
                 "target_improvement_pct": 25, "triage_zone_active": True, "after_hours_active": True}

    return {
        "study_date": target_date.isoformat(),
        "collected_at": f"{target_date.isoformat()}T06:00:00Z",
        "study_phase": phase,
        "cohort_size": 10,
        "daily_sms": {
            "outbound_total": total_outbound,
            "inbound_total": total_inbound,
            "patients_messaged": 10,
            "patients_replied": patients_replied,
            "response_rate_pct": response_rate,
        },
        "daily_vitals": {"status": "pending_schema_confirmation"},
        "daily_kestrel_ai": {
            "ai_messages_sent": total_outbound,
            "safezone_triggered": random.randint(0, 3),
            "escalations": random.randint(0, 2),
        },
        "after_hours": {
            "after_hours_outbound": random.randint(0, 5),
            "after_hours_questions": 0,
            "phase1_violation": False,
        },
        "at_risk_patients": at_risk,
        "patient_details": patient_details,
    }


if __name__ == "__main__":
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    # Generate baseline
    baseline = generate_baseline()
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Generated {BASELINE_PATH}")

    # Generate 14 days of snapshots
    for day in range(14):
        target = STUDY_START + timedelta(days=day)
        snap = generate_daily_snapshot(target, day + 1)
        path = f"{SNAPSHOTS_DIR}/{target.isoformat()}.json"
        with open(path, "w") as f:
            json.dump(snap, f, indent=2)
        print(f"Generated {path}")

    print(f"\nGenerated baseline + 14 daily snapshots")
    print("Now run: python caprock-study/build_cumulative.py")
