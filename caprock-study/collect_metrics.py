"""
Caprock Clinical Validation Study — Daily Metrics Collector
============================================================
Queries the RemoteCares staging PostgreSQL database and emits a
structured JSON snapshot of patient adherence and engagement metrics.

Run daily (via GitHub Action or cron). Output goes to:
  caprock-study/snapshots/YYYY-MM-DD.json

Activation requirements:
  - DB_HOST         environment variable (or CAPROCK_DB_HOST secret)
  - DB_PORT         (default: 5432)
  - DB_NAME         (default: remotecares)
  - DB_USER
  - DB_PASSWORD
  - STUDY_START     ISO date when Kestrel went live (e.g. 2026-03-17)
  - VITALS_TABLE    Table name for readings/vitals (set once Jason confirms) — optional
  - BUSINESS_HRS_START   Hour (local, 24h) when care managers start (default: 8)
  - BUSINESS_HRS_END     Hour (local, 24h) when care managers finish (default: 18)

Usage:
  python collect_metrics.py --date 2026-03-17 --out snapshots/2026-03-17.json
  python collect_metrics.py            # uses today's date, auto output path
"""

import os
import json
import argparse
import psycopg2
import psycopg2.extras
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

BUSINESS_HRS_START = int(os.environ.get("BUSINESS_HRS_START", 8))   # 08:00 local
BUSINESS_HRS_END   = int(os.environ.get("BUSINESS_HRS_END",  18))   # 18:00 local
VITALS_TABLE       = os.environ.get("VITALS_TABLE")                  # e.g. "VitalSigns" — set once confirmed


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
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ.get("DB_NAME", "remotecares"),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        sslmode="require",
    )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def fetch_active_patients(cur):
    """Return list of active Caprock RPM patients."""
    cur.execute("""
        SELECT
            p."Id"                  AS patient_id,
            p."FullNameLF"          AS name,
            p."MobilePhone"         AS mobile,
            pa."IsAgentActive"      AS agent_active,
            pa."IsRpmActive"        AS rpm_active,
            pa."IsRpmMedicaidActive" AS rpm_medicaid
        FROM "PatientAssistant" pa
        JOIN "Person" p ON p."Id" = pa."PersonId"
        WHERE (pa."IsRpmActive" = true OR pa."IsRpmMedicaidActive" = true)
          AND pa."IsAgentActive" = true
    """)
    return [dict(r) for r in cur.fetchall()]


def fetch_sms_stats(cur, target_date: date):
    """
    Daily SMS engagement metrics from SmsLogs.
    Direction: 0 = outbound (system → patient), 1 = inbound (patient → system)
    """
    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())

    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE "Direction" = 0)                      AS outbound_total,
            COUNT(*) FILTER (WHERE "Direction" = 1)                      AS inbound_total,
            COUNT(DISTINCT "PatientId") FILTER (WHERE "Direction" = 0)   AS patients_messaged,
            COUNT(DISTINCT "PatientId") FILTER (WHERE "Direction" = 1)   AS patients_replied
        FROM "SmsLogs"
        WHERE "CreatedAt" >= %s AND "CreatedAt" < %s
    """, (day_start, day_end))
    return dict(cur.fetchone())


def fetch_after_hours_compliance(cur, target_date: date) -> dict:
    """
    Phase 1 rule: AI must NOT send open-ended questions after business hours.
    Flags any outbound AI messages sent outside business hours that contain a '?'.
    A violation in Phase 1 means AI asked a wellness question when no care manager
    was available to respond to a potential urgent reply.
    """
    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())

    try:
        cur.execute("""
            SELECT
                COUNT(*) AS after_hours_total,
                COUNT(*) FILTER (WHERE "Message" LIKE '%%?%%') AS after_hours_questions
            FROM "SmsLogs"
            WHERE "Direction" = 0
              AND "CreatedAt" >= %s AND "CreatedAt" < %s
              AND (EXTRACT(HOUR FROM "CreatedAt") < %s
                   OR EXTRACT(HOUR FROM "CreatedAt") >= %s)
        """, (day_start, day_end, BUSINESS_HRS_START, BUSINESS_HRS_END))
        row = dict(cur.fetchone())
        return {
            "after_hours_outbound":  row["after_hours_total"],
            "after_hours_questions": row["after_hours_questions"],
            "phase1_violation":      row["after_hours_questions"] > 0,
        }
    except Exception:
        return {"after_hours_outbound": None, "after_hours_questions": None, "phase1_violation": None}


def fetch_kestrel_ai_stats(cur, target_date: date) -> dict:
    """
    Kestrel AI message counts from AgentMessages.
    Attempts to read extended columns (UrgencyLevel, ConfidenceScore, Zone)
    — falls back gracefully if columns don't yet exist.
    Tracked by: jaslogic1/RemoteCares#67
    """
    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())

    # Base query — always works once table has any rows
    base_result = {"ai_messages_sent": 0, "safezone_triggered": 0, "escalations": 0,
                   "triage_breakdown": None, "avg_confidence_score": None, "zone_breakdown": None}
    try:
        cur.execute("""
            SELECT
                COUNT(*)                                            AS ai_messages_sent,
                COUNT(*) FILTER (WHERE "SafeZoneTriggered" = true) AS safezone_triggered,
                COUNT(*) FILTER (WHERE "EscalationLevel" > 0)      AS escalations
            FROM "AgentMessages"
            WHERE "CreatedAt" >= %s AND "CreatedAt" < %s
        """, (day_start, day_end))
        row = cur.fetchone()
        if row:
            base_result.update({"ai_messages_sent": row["ai_messages_sent"],
                                 "safezone_triggered": row["safezone_triggered"],
                                 "escalations": row["escalations"]})
    except Exception:
        return base_result

    # Extended query — needs jaslogic1/RemoteCares#67 (UrgencyLevel, ConfidenceScore, Zone)
    try:
        cur.execute("""
            SELECT
                "UrgencyLevel",
                COUNT(*) AS cnt,
                AVG("ConfidenceScore") AS avg_confidence
            FROM "AgentMessages"
            WHERE "CreatedAt" >= %s AND "CreatedAt" < %s
              AND "UrgencyLevel" IS NOT NULL
            GROUP BY "UrgencyLevel"
        """, (day_start, day_end))
        rows = cur.fetchall()
        if rows:
            breakdown = {r["UrgencyLevel"]: {"count": r["cnt"],
                                              "avg_confidence": round(float(r["avg_confidence"] or 0), 3)}
                         for r in rows}
            base_result["triage_breakdown"] = breakdown
            all_conf = [r["avg_confidence"] for r in rows if r["avg_confidence"]]
            if all_conf:
                base_result["avg_confidence_score"] = round(sum(all_conf) / len(all_conf), 3)
    except Exception:
        pass  # columns not yet added — tracked in #67

    # Zone breakdown
    try:
        cur.execute("""
            SELECT "Zone", COUNT(*) AS cnt
            FROM "AgentMessages"
            WHERE "CreatedAt" >= %s AND "CreatedAt" < %s
            GROUP BY "Zone"
        """, (day_start, day_end))
        rows = cur.fetchall()
        if rows:
            base_result["zone_breakdown"] = {r["Zone"]: r["cnt"] for r in rows if r["Zone"]}
    except Exception:
        pass  # Zone column not yet added

    return base_result


def fetch_vitals_compliance(cur, target_date: date, patient_ids: list) -> dict:
    """
    PRIMARY OUTCOME METRIC: % of patients who submitted a reading today.
    Requires VITALS_TABLE env var to be set.
    Schema confirmation tracked in: jaslogic1/RemoteCares#68
    """
    if not VITALS_TABLE:
        return {"status": "pending_schema_confirmation", "ref": "jaslogic1/RemoteCares#68"}

    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())

    try:
        cur.execute(f"""
            SELECT
                COUNT(DISTINCT "PatientId")  AS patients_with_readings,
                COUNT(*)                     AS total_readings
            FROM "{VITALS_TABLE}"
            WHERE "PatientId" = ANY(%s)
              AND "CreatedAt" >= %s AND "CreatedAt" < %s
        """, (patient_ids, day_start, day_end))
        row = dict(cur.fetchone())
        total = len(patient_ids)
        with_readings = row["patients_with_readings"]
        return {
            "patients_with_readings": with_readings,
            "total_patients":         total,
            "compliance_pct":         round(with_readings / total * 100, 1) if total else 0.0,
            "total_readings_today":   row["total_readings"],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def fetch_per_patient_sms(cur, target_date: date, patient_ids: list):
    """Per-patient SMS activity for the target date."""
    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())

    cur.execute("""
        SELECT
            "PatientId"                                          AS patient_id,
            COUNT(*) FILTER (WHERE "Direction" = 0)             AS outbound,
            COUNT(*) FILTER (WHERE "Direction" = 1)             AS inbound,
            MAX("CreatedAt") FILTER (WHERE "Direction" = 1)     AS last_reply_at
        FROM "SmsLogs"
        WHERE "PatientId" = ANY(%s)
          AND "CreatedAt" >= %s AND "CreatedAt" < %s
        GROUP BY "PatientId"
    """, (patient_ids, day_start, day_end))
    return {str(r["patient_id"]): dict(r) for r in cur.fetchall()}


def fetch_response_streak(cur, patient_id, as_of: date, lookback_days: int = 30):
    """Count consecutive days ending on as_of date that patient replied."""
    cur.execute("""
        SELECT DISTINCT DATE("CreatedAt") AS reply_date
        FROM "SmsLogs"
        WHERE "PatientId" = %s
          AND "Direction" = 1
          AND "CreatedAt" >= %s AND "CreatedAt" <= %s
        ORDER BY reply_date DESC
    """, (patient_id, as_of - timedelta(days=lookback_days), as_of))
    rows = [r["reply_date"] for r in cur.fetchall()]
    streak = 0
    check = as_of
    while check in rows:
        streak += 1
        check -= timedelta(days=1)
    return streak


def fetch_days_since_last_reply(cur, patient_id, as_of: date):
    """How many days since patient last sent an inbound SMS."""
    cur.execute("""
        SELECT MAX(DATE("CreatedAt")) AS last_day
        FROM "SmsLogs"
        WHERE "PatientId" = %s AND "Direction" = 1
    """, (patient_id,))
    row = cur.fetchone()
    if not row or not row["last_day"]:
        return None
    return (as_of - row["last_day"]).days


def fetch_weekly_trend(cur, patient_id, week_end: date):
    """Response rate for prior 2 weeks to detect trend direction."""
    def week_rate(start, end):
        cur.execute("""
            SELECT COUNT(DISTINCT DATE("CreatedAt")) AS reply_days
            FROM "SmsLogs"
            WHERE "PatientId" = %s AND "Direction" = 1
              AND "CreatedAt" >= %s AND "CreatedAt" < %s
        """, (patient_id, start, end))
        return cur.fetchone()["reply_days"]

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
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        patients    = fetch_active_patients(cur)
        patient_ids = [p["patient_id"] for p in patients]

        sms_daily    = fetch_sms_stats(cur, target_date)
        ai_daily     = fetch_kestrel_ai_stats(cur, target_date)
        vitals_daily = fetch_vitals_compliance(cur, target_date, patient_ids)
        after_hrs    = fetch_after_hours_compliance(cur, target_date)
        per_pat      = fetch_per_patient_sms(cur, target_date, patient_ids)

        response_rate = (
            round(sms_daily["patients_replied"] / sms_daily["patients_messaged"] * 100, 1)
            if sms_daily["patients_messaged"] > 0 else 0.0
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
                "sms_out_today":        sms["outbound"],
                "sms_in_today":         sms["inbound"],
                "replied_today":        sms["inbound"] > 0,
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
            "daily_vitals":     vitals_daily,      # PRIMARY OUTCOME METRIC
            "daily_kestrel_ai": ai_daily,
            "after_hours":      after_hrs,          # Phase 1 compliance check
            "at_risk_patients": at_risk,
            "patient_details":  patient_details,
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
        print(f"  Reading compliance: {vitals['compliance_pct']}% ({vitals['patients_with_readings']}/{vitals['total_patients']} patients)")
    else:
        print(f"  Reading compliance: {vitals.get('status', 'n/a')} (set VITALS_TABLE env var to activate)")
    after = snapshot["after_hours"]
    if after.get("phase1_violation"):
        print(f"  ⚠️  PHASE 1 VIOLATION: {after['after_hours_questions']} after-hours questions detected!")
    print(f"  At-risk (3+ days silent): {snapshot['at_risk_patients']}")
