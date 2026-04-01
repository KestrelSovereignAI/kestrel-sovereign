"""
Caprock Clinical Validation Study — Daily Metrics Collector
============================================================
Queries the RemoteCares SQL Server database and emits a structured JSON
snapshot of patient adherence and engagement metrics.

Run daily (via GitHub Action or cron). Output goes to:
  caprock-study/snapshots/YYYY-MM-DD.json

Environment variables:
  DB_SERVER          — SQL Server host (e.g. myserver.database.windows.net)
  DB_NAME            — (default: remotecares)
  DB_USER
  DB_PASSWORD
  STUDY_START        — ISO date when Kestrel went live (e.g. 2026-03-17)
  BUSINESS_HRS_START — Hour (local, 24h) when care managers start (default: 8)
  BUSINESS_HRS_END   — Hour (local, 24h) when care managers finish (default: 18)
  ODBC_DRIVER        — ODBC driver name (default: ODBC Driver 18 for SQL Server)

Usage:
  python collect_metrics.py --date 2026-03-17 --out snapshots/2026-03-17.json
  python collect_metrics.py            # uses today's date, auto output path
"""

import os
import json
import argparse
import pyodbc
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Study Phase Configuration (Safe Zones Framework — 3-phase rollout)
# ---------------------------------------------------------------------------

STUDY_PHASES = {
    1: {
        "name":                   "Safe Zone Only",
        "weeks":                  (1, 4),
        "description":            "AI sends personalized encouragement + contextual reminders. No open-ended wellness questions.",
        "target_improvement_pct": 10,
        "triage_zone_active":     False,
        "after_hours_active":     False,
    },
    2: {
        "name":                   "Add Triage Zone",
        "weeks":                  (5, 8),
        "description":            "Business-hours wellness check-ins + instant keyword detection + routing.",
        "target_improvement_pct": 20,
        "triage_zone_active":     True,
        "after_hours_active":     False,
    },
    3: {
        "name":                   "After-Hours Expansion",
        "weeks":                  (9, 12),
        "description":            "24/7 triage and routing with on-call care manager escalation.",
        "target_improvement_pct": 25,
        "triage_zone_active":     True,
        "after_hours_active":     True,
    },
}

BUSINESS_HRS_START = int(os.environ.get("BUSINESS_HRS_START", 8))
BUSINESS_HRS_END   = int(os.environ.get("BUSINESS_HRS_END",  18))
ODBC_DRIVER = os.environ.get("ODBC_DRIVER", "ODBC Driver 18 for SQL Server")


def get_study_phase(study_start: date, target_date: date) -> dict:
    """Return the current study phase config based on days elapsed."""
    if target_date < study_start:
        return {"phase": 0, "week": 0, "name": "Pre-study", "target_improvement_pct": None,
                "triage_zone_active": False, "after_hours_active": False}
    week_number = ((target_date - study_start).days // 7) + 1
    for phase_num in sorted(STUDY_PHASES.keys(), reverse=True):
        cfg = STUDY_PHASES[phase_num]
        if week_number >= cfg["weeks"][0]:
            return {"phase": phase_num, "week": week_number, **cfg}
    return {"phase": 1, "week": week_number, **STUDY_PHASES[1]}


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection():
    server = os.environ["DB_SERVER"]
    database = os.environ.get("DB_NAME", "remotecares")
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    conn_str = (
        f"DRIVER={{{ODBC_DRIVER}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        "Encrypt=yes;TrustServerCertificate=no;"
    )
    return pyodbc.connect(conn_str)


def rows_to_dicts(cursor):
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def row_to_dict(cursor):
    columns = [col[0] for col in cursor.description]
    row = cursor.fetchone()
    return dict(zip(columns, row)) if row else {}


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def fetch_active_patients(cur):
    """Return list of active Caprock RPM patients."""
    cur.execute("""
        SELECT
            p.Oid               AS patient_id,
            p.FullName          AS name,
            p.MobilePhone       AS mobile,
            p.IsRpmActive       AS rpm_active,
            p.IsRpmMedicaidActive AS rpm_medicaid
        FROM Patients p
        WHERE (p.IsRpmActive = 1 OR p.IsRpmMedicaidActive = 1)
    """)
    return rows_to_dicts(cur)


def fetch_sms_stats(cur, target_date: date):
    """Daily SMS engagement metrics from SmsLogs."""
    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())

    cur.execute("""
        SELECT
            SUM(CASE WHEN Direction = 0 THEN 1 ELSE 0 END)                    AS outbound_total,
            SUM(CASE WHEN Direction = 1 THEN 1 ELSE 0 END)                    AS inbound_total,
            COUNT(DISTINCT CASE WHEN Direction = 0 THEN PatientId END)         AS patients_messaged,
            COUNT(DISTINCT CASE WHEN Direction = 1 THEN PatientId END)         AS patients_replied
        FROM SmsLogs
        WHERE CreatedOn >= ? AND CreatedOn < ?
    """, day_start, day_end)
    result = row_to_dict(cur)
    return {k: (v or 0) for k, v in result.items()}


def fetch_after_hours_compliance(cur, target_date: date) -> dict:
    """
    Phase 1 rule: AI must NOT send open-ended questions after business hours.
    Flags any outbound AI messages sent outside business hours that contain a '?'.
    """
    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())

    try:
        cur.execute("""
            SELECT
                COUNT(*) AS after_hours_total,
                SUM(CASE WHEN Content LIKE '%?%' THEN 1 ELSE 0 END) AS after_hours_questions
            FROM SmsLogs
            WHERE Direction = 0
              AND CreatedOn >= ? AND CreatedOn < ?
              AND (DATEPART(HOUR, CreatedOn) < ? OR DATEPART(HOUR, CreatedOn) >= ?)
        """, day_start, day_end, BUSINESS_HRS_START, BUSINESS_HRS_END)
        row = row_to_dict(cur)
        after_hrs_total = row.get("after_hours_total", 0) or 0
        after_hrs_questions = row.get("after_hours_questions", 0) or 0
        return {
            "after_hours_outbound":  after_hrs_total,
            "after_hours_questions": after_hrs_questions,
            "phase1_violation":      after_hrs_questions > 0,
        }
    except Exception:
        return {"after_hours_outbound": None, "after_hours_questions": None, "phase1_violation": None}


def fetch_vitals_compliance(cur, target_date: date, patient_ids: list) -> dict:
    """
    PRIMARY OUTCOME METRIC: % of patients who submitted a device reading today.
    Source = 0 means reading came from a physical device (not manual/missed).
    """
    if not patient_ids:
        return {"status": "no_patients"}

    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
    placeholders = ",".join("?" for _ in patient_ids)

    try:
        cur.execute(f"""
            SELECT
                COUNT(DISTINCT vs.PatientId)  AS patients_with_readings,
                COUNT(*)                      AS total_readings
            FROM VitalSigns vs
            JOIN Encounters e ON e.Oid = vs.EncounterId
            WHERE vs.PatientId IN ({placeholders})
              AND e.StartOn >= ? AND e.StartOn < ?
              AND vs.Source = 0
        """, patient_ids + [day_start, day_end])
        row = row_to_dict(cur)
        total = len(patient_ids)
        with_readings = row.get("patients_with_readings", 0) or 0
        return {
            "patients_with_readings": with_readings,
            "total_patients":         total,
            "compliance_pct":         round(with_readings / total * 100, 1) if total else 0.0,
            "total_readings_today":   row.get("total_readings", 0) or 0,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def fetch_vitals_by_type(cur, target_date: date, patient_ids: list) -> dict:
    """Breakdown of today's readings by vital sign type (BP, glucose, SpO2, etc.)."""
    if not patient_ids:
        return {}

    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
    placeholders = ",".join("?" for _ in patient_ids)

    try:
        cur.execute(f"""
            SELECT
                CASE
                    WHEN vs.Systolic IS NOT NULL THEN 'BloodPressure'
                    WHEN vs.BloodGlucose IS NOT NULL THEN 'BloodGlucose'
                    WHEN vs.SpO2 IS NOT NULL THEN 'SpO2'
                    WHEN vs.WeightInPounds IS NOT NULL THEN 'Weight'
                    ELSE 'Other'
                END AS reading_type,
                COUNT(*) AS count,
                COUNT(DISTINCT vs.PatientId) AS patients
            FROM VitalSigns vs
            JOIN Encounters e ON e.Oid = vs.EncounterId
            WHERE vs.PatientId IN ({placeholders})
              AND e.StartOn >= ? AND e.StartOn < ?
              AND vs.Source = 0
            GROUP BY
                CASE
                    WHEN vs.Systolic IS NOT NULL THEN 'BloodPressure'
                    WHEN vs.BloodGlucose IS NOT NULL THEN 'BloodGlucose'
                    WHEN vs.SpO2 IS NOT NULL THEN 'SpO2'
                    WHEN vs.WeightInPounds IS NOT NULL THEN 'Weight'
                    ELSE 'Other'
                END
        """, patient_ids + [day_start, day_end])
        return {row["reading_type"]: {"count": row["count"], "patients": row["patients"]}
                for row in rows_to_dicts(cur)}
    except Exception:
        return {}


def fetch_result_outcome_breakdown(cur, target_date: date, patient_ids: list) -> dict:
    """Breakdown of reading results: Normal, Abnormal, Critical."""
    if not patient_ids:
        return {}

    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
    placeholders = ",".join("?" for _ in patient_ids)

    try:
        cur.execute(f"""
            SELECT
                vs.ResultOutcome,
                COUNT(*) AS count
            FROM VitalSigns vs
            JOIN Encounters e ON e.Oid = vs.EncounterId
            WHERE vs.PatientId IN ({placeholders})
              AND e.StartOn >= ? AND e.StartOn < ?
              AND vs.Source = 0
              AND vs.ResultOutcome IS NOT NULL
            GROUP BY vs.ResultOutcome
        """, patient_ids + [day_start, day_end])
        # ResultOutcome enum: 0=Normal, 1=Abnormal, 2=Critical
        outcome_labels = {0: "Normal", 1: "Abnormal", 2: "Critical"}
        return {outcome_labels.get(row["ResultOutcome"], str(row["ResultOutcome"])): row["count"]
                for row in rows_to_dicts(cur)}
    except Exception:
        return {}


def fetch_per_patient_sms(cur, target_date: date, patient_ids: list):
    """Per-patient SMS activity for the target date."""
    if not patient_ids:
        return {}

    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
    placeholders = ",".join("?" for _ in patient_ids)

    cur.execute(f"""
        SELECT
            PatientId                                                       AS patient_id,
            SUM(CASE WHEN Direction = 0 THEN 1 ELSE 0 END)                AS outbound,
            SUM(CASE WHEN Direction = 1 THEN 1 ELSE 0 END)                AS inbound,
            MAX(CASE WHEN Direction = 1 THEN CreatedOn END)                AS last_reply_at
        FROM SmsLogs
        WHERE PatientId IN ({placeholders})
          AND CreatedOn >= ? AND CreatedOn < ?
        GROUP BY PatientId
    """, patient_ids + [day_start, day_end])
    return {str(r["patient_id"]): r for r in rows_to_dicts(cur)}


def fetch_response_streak(cur, patient_id, as_of: date, lookback_days: int = 30):
    """Count consecutive days ending on as_of date that patient replied."""
    cur.execute("""
        SELECT DISTINCT CAST(CreatedOn AS DATE) AS reply_date
        FROM SmsLogs
        WHERE PatientId = ?
          AND Direction = 1
          AND CreatedOn >= ? AND CreatedOn <= ?
        ORDER BY reply_date DESC
    """, patient_id, as_of - timedelta(days=lookback_days), as_of)
    rows = [row[0] for row in cur.fetchall()]
    streak = 0
    check = as_of
    while check in rows:
        streak += 1
        check -= timedelta(days=1)
    return streak


def fetch_days_since_last_reply(cur, patient_id, as_of: date):
    """How many days since patient last sent an inbound SMS."""
    cur.execute("""
        SELECT MAX(CAST(CreatedOn AS DATE)) AS last_day
        FROM SmsLogs
        WHERE PatientId = ? AND Direction = 1
    """, patient_id)
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    return (as_of - row[0]).days


def fetch_weekly_trend(cur, patient_id, week_end: date):
    """Response rate for prior 2 weeks to detect trend direction."""
    def week_rate(start, end):
        cur.execute("""
            SELECT COUNT(DISTINCT CAST(CreatedOn AS DATE)) AS reply_days
            FROM SmsLogs
            WHERE PatientId = ? AND Direction = 1
              AND CreatedOn >= ? AND CreatedOn < ?
        """, patient_id, start, end)
        row = cur.fetchone()
        return row[0] if row else 0

    this_week_start = week_end - timedelta(days=6)
    prev_week_start = this_week_start - timedelta(days=7)
    this_week = week_rate(this_week_start, week_end + timedelta(days=1))
    prev_week = week_rate(prev_week_start, this_week_start)

    if prev_week == 0:
        trend = "new"
    elif this_week > prev_week:
        trend = "improving"
    elif this_week < prev_week:
        trend = "declining"
    else:
        trend = "stable"

    return {"this_week_reply_days": this_week, "prev_week_reply_days": prev_week, "trend": trend}


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------

def collect(target_date: date) -> dict:
    study_start_env = os.environ.get("STUDY_START")
    study_start = date.fromisoformat(study_start_env) if study_start_env else target_date
    study_phase = get_study_phase(study_start, target_date)

    conn = get_connection()
    cur = conn.cursor()

    try:
        patients    = fetch_active_patients(cur)
        patient_ids = [p["patient_id"] for p in patients]

        sms_daily       = fetch_sms_stats(cur, target_date)
        vitals_daily    = fetch_vitals_compliance(cur, target_date, patient_ids)
        vitals_by_type  = fetch_vitals_by_type(cur, target_date, patient_ids)
        outcome_breakdown = fetch_result_outcome_breakdown(cur, target_date, patient_ids)
        after_hrs       = fetch_after_hours_compliance(cur, target_date)
        per_pat         = fetch_per_patient_sms(cur, target_date, patient_ids)

        patients_messaged = sms_daily.get("patients_messaged", 0)
        patients_replied = sms_daily.get("patients_replied", 0)
        response_rate = (
            round(patients_replied / patients_messaged * 100, 1)
            if patients_messaged > 0 else 0.0
        )

        patient_details = []
        at_risk = []
        for p in patients:
            pid       = str(p["patient_id"])
            sms       = per_pat.get(pid, {"outbound": 0, "inbound": 0, "last_reply_at": None})
            days_silent = fetch_days_since_last_reply(cur, p["patient_id"], target_date)
            streak      = fetch_response_streak(cur, p["patient_id"], target_date)
            trend       = fetch_weekly_trend(cur, p["patient_id"], target_date)

            rec = {
                "patient_id":           pid,
                "name":                 p["name"],
                "sms_out_today":        sms.get("outbound", 0) or 0,
                "sms_in_today":         sms.get("inbound", 0) or 0,
                "replied_today":        (sms.get("inbound", 0) or 0) > 0,
                "reply_streak_days":    streak,
                "days_since_last_reply": days_silent,
                "weekly_trend":         trend,
                "flag_at_risk":         days_silent is not None and days_silent >= 3,
            }
            patient_details.append(rec)
            if rec["flag_at_risk"]:
                at_risk.append(p["name"])

        snapshot = {
            "study_date":     target_date.isoformat(),
            "collected_at":   datetime.utcnow().isoformat() + "Z",
            "study_phase":    study_phase,
            "cohort_size":    len(patients),
            "daily_sms": {
                **sms_daily,
                "response_rate_pct": response_rate,
            },
            "daily_vitals":           vitals_daily,
            "vitals_by_type":         vitals_by_type,
            "outcome_breakdown":      outcome_breakdown,
            "after_hours":            after_hrs,
            "at_risk_patients":       at_risk,
            "patient_details":        patient_details,
        }

        return snapshot

    finally:
        cur.close()
        conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Caprock daily metrics")
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date YYYY-MM-DD")
    parser.add_argument("--out",  default=None, help="Output JSON path (default: snapshots/YYYY-MM-DD.json)")
    args = parser.parse_args()

    target   = date.fromisoformat(args.date)
    out_path = args.out or f"caprock-study/snapshots/{target.isoformat()}.json"

    print(f"Collecting metrics for {target}...")
    snapshot = collect(target)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    phase = snapshot["study_phase"]
    print(f"Saved to {out_path}")
    print(f"  Study Phase {phase['phase']}: {phase['name']} (Week {phase['week']})")
    print(f"  Cohort: {snapshot['cohort_size']} active patients")
    print(f"  SMS response rate: {snapshot['daily_sms']['response_rate_pct']}%")
    vitals = snapshot["daily_vitals"]
    if isinstance(vitals, dict) and "compliance_pct" in vitals:
        print(f"  Reading compliance: {vitals['compliance_pct']}% "
              f"({vitals['patients_with_readings']}/{vitals['total_patients']} patients)")
    else:
        print(f"  Reading compliance: {vitals.get('status', 'n/a')}")
    if snapshot["vitals_by_type"]:
        print(f"  Reading types: {snapshot['vitals_by_type']}")
    if snapshot["outcome_breakdown"]:
        print(f"  Outcomes: {snapshot['outcome_breakdown']}")
    after = snapshot["after_hours"]
    if after.get("phase1_violation"):
        print(f"  PHASE 1 VIOLATION: {after['after_hours_questions']} after-hours questions detected!")
    print(f"  At-risk (3+ days silent): {snapshot['at_risk_patients']}")
