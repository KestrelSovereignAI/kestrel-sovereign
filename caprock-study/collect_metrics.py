"""
Caprock Clinical Validation Study — Daily Metrics Collector
============================================================
Queries the RemoteCares staging PostgreSQL database and emits a
structured JSON snapshot of patient adherence and engagement metrics.

Run daily (via GitHub Action or cron). Output goes to:
  caprock-study/snapshots/YYYY-MM-DD.json

Activation requirements:
  - DB_HOST       environment variable (or CAPROCK_DB_HOST secret)
  - DB_PORT       (default: 5432)
  - DB_NAME       (default: remotecares)
  - DB_USER
  - DB_PASSWORD
  - STUDY_START   ISO date when Kestrel went live (e.g. 2026-03-17)

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
    return psycopg2.extras.RealDictCursor and [dict(r) for r in cur.fetchall()]


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


def fetch_kestrel_ai_stats(cur, target_date: date):
    """Kestrel AI message counts from AgentMessages (empty until live)."""
    day_start = datetime.combine(target_date, datetime.min.time())
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())

    cur.execute("""
        SELECT
            COUNT(*)                                            AS ai_messages_sent,
            COUNT(*) FILTER (WHERE "SafeZoneTriggered" = true) AS safezone_triggered,
            COUNT(*) FILTER (WHERE "EscalationLevel" > 0)      AS escalations
        FROM "AgentMessages"
        WHERE "CreatedAt" >= %s AND "CreatedAt" < %s
    """, (day_start, day_end))
    # Table may be empty or columns may differ — return zeros on failure
    try:
        return dict(cur.fetchone())
    except Exception:
        return {"ai_messages_sent": 0, "safezone_triggered": 0, "escalations": 0}


def fetch_response_streak(cur, patient_id, as_of: date, lookback_days: int = 30):
    """
    Count consecutive days ending on as_of date that patient replied.
    A 'reply day' = at least 1 inbound SmsLog on that calendar date.
    """
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
        return None  # never replied
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
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        patients = fetch_active_patients(cur)
        patient_ids = [p["patient_id"] for p in patients]

        sms_daily = fetch_sms_stats(cur, target_date)
        ai_daily  = fetch_kestrel_ai_stats(cur, target_date)
        per_pat   = fetch_per_patient_sms(cur, target_date, patient_ids)

        response_rate = (
            round(sms_daily["patients_replied"] / sms_daily["patients_messaged"] * 100, 1)
            if sms_daily["patients_messaged"] > 0 else 0.0
        )

        patient_details = []
        at_risk = []
        for p in patients:
            pid = str(p["patient_id"])
            sms = per_pat.get(pid, {"outbound": 0, "inbound": 0, "last_reply_at": None})
            days_silent = fetch_days_since_last_reply(cur, p["patient_id"], target_date)
            streak      = fetch_response_streak(cur, p["patient_id"], target_date)
            trend       = fetch_weekly_trend(cur, p["patient_id"], target_date)

            rec = {
                "patient_id":    pid,
                "name":          p["name"],
                "sms_out_today": sms["outbound"],
                "sms_in_today":  sms["inbound"],
                "replied_today": sms["inbound"] > 0,
                "reply_streak_days": streak,
                "days_since_last_reply": days_silent,
                "weekly_trend":  trend,
                "flag_at_risk":  days_silent is not None and days_silent >= 3,
            }
            patient_details.append(rec)
            if rec["flag_at_risk"]:
                at_risk.append(p["name"])

        snapshot = {
            "study_date":        target_date.isoformat(),
            "collected_at":      datetime.utcnow().isoformat() + "Z",
            "cohort_size":       len(patients),
            "daily_sms": {
                **sms_daily,
                "response_rate_pct": response_rate,
            },
            "daily_kestrel_ai":  ai_daily,
            "at_risk_patients":  at_risk,
            "patient_details":   patient_details,
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

    target = date.fromisoformat(args.date)
    out_path = args.out or f"caprock-study/snapshots/{target.isoformat()}.json"

    print(f"Collecting metrics for {target}...")
    snapshot = collect(target)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"Saved to {out_path}")
    print(f"  Cohort: {snapshot['cohort_size']} active patients")
    print(f"  Response rate: {snapshot['daily_sms']['response_rate_pct']}%")
    print(f"  At-risk (3+ days silent): {snapshot['at_risk_patients']}")
