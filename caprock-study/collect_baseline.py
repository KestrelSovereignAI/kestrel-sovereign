"""
Caprock Clinical Validation Study — Pre-Kestrel Baseline Collector
===================================================================
Queries the PRODUCTION RemoteCares database for the 30 days BEFORE
Kestrel go-live to establish the "before" comparison for the study.

This script must run ONCE, against PRODUCTION (not staging), ideally
on or just before Kestrel go-live day for these patients.

Output:  caprock-study/baseline.json

Environment variables:
  PROD_DB_HOST       — production PostgreSQL host
  PROD_DB_PORT       — (default: 5432)
  PROD_DB_NAME       — (default: remotecares)
  PROD_DB_USER       — read-only user with SELECT on relevant tables
  PROD_DB_PASSWORD
  KESTREL_GOLIVE     — ISO date when Kestrel goes live (e.g. 2026-03-24)
  BASELINE_DAYS      — number of days to look back (default: 30)
  VITALS_TABLE       — table name for vitals/readings (optional — set when confirmed)

Usage:
  python collect_baseline.py
  python collect_baseline.py --golive 2026-03-24 --days 30 --out caprock-study/baseline.json
"""

import os
import json
import argparse
import psycopg2
import psycopg2.extras
from datetime import date, datetime, timedelta


VITALS_TABLE = os.environ.get("VITALS_TABLE")  # e.g. "VitalSigns" — set once confirmed


def get_connection():
    return psycopg2.connect(
        host=os.environ["PROD_DB_HOST"],
        port=int(os.environ.get("PROD_DB_PORT", 5432)),
        dbname=os.environ.get("PROD_DB_NAME", "remotecares"),
        user=os.environ["PROD_DB_USER"],
        password=os.environ["PROD_DB_PASSWORD"],
        sslmode="require",
    )


def fetch_caprock_patients(cur):
    """Identify the 10 Caprock RPM Medicaid patients."""
    cur.execute("""
        SELECT
            p."Id"                      AS patient_id,
            p."FullNameLF"              AS name,
            p."MobilePhone"             AS mobile,
            pa."IsAgentActive"          AS agent_active,
            pa."IsRpmActive"            AS rpm_active,
            pa."IsRpmMedicaidActive"    AS rpm_medicaid
        FROM "PatientAssistant" pa
        JOIN "Person" p ON p."Id" = pa."PersonId"
        WHERE (pa."IsRpmActive" = true OR pa."IsRpmMedicaidActive" = true)
    """)
    return [dict(r) for r in cur.fetchall()]


def fetch_sms_baseline(cur, patient_ids, start_date, end_date):
    """
    SMS engagement baseline: how often did patients reply to the OLD
    auto-bot system before Kestrel was introduced?
    """
    cur.execute("""
        SELECT
            COUNT(*)                                                    AS total_messages,
            COUNT(*) FILTER (WHERE "Direction" = 0)                    AS outbound_total,
            COUNT(*) FILTER (WHERE "Direction" = 1)                    AS inbound_total,
            COUNT(DISTINCT "PatientId") FILTER (WHERE "Direction" = 0) AS patients_messaged,
            COUNT(DISTINCT "PatientId") FILTER (WHERE "Direction" = 1) AS patients_replied,
            COUNT(DISTINCT DATE("CreatedAt"))                          AS days_with_activity
        FROM "SmsLogs"
        WHERE "PatientId" = ANY(%s)
          AND "CreatedAt" >= %s AND "CreatedAt" < %s
    """, (patient_ids, start_date, end_date))
    return dict(cur.fetchone())


def fetch_per_patient_sms_baseline(cur, patient_ids, start_date, end_date, baseline_days):
    """Per-patient SMS engagement during the baseline window."""
    cur.execute("""
        SELECT
            "PatientId"                                             AS patient_id,
            COUNT(*) FILTER (WHERE "Direction" = 0)                AS outbound,
            COUNT(*) FILTER (WHERE "Direction" = 1)                AS inbound,
            COUNT(DISTINCT DATE("CreatedAt"))
                FILTER (WHERE "Direction" = 1)                     AS days_replied,
            MIN("CreatedAt")                                       AS first_message,
            MAX("CreatedAt")                                       AS last_message
        FROM "SmsLogs"
        WHERE "PatientId" = ANY(%s)
          AND "CreatedAt" >= %s AND "CreatedAt" < %s
        GROUP BY "PatientId"
    """, (patient_ids, start_date, end_date))

    results = {}
    for row in cur.fetchall():
        r = dict(row)
        pid = str(r["patient_id"])
        days_replied = r["days_replied"]
        results[pid] = {
            "patient_id":    pid,
            "outbound":      r["outbound"],
            "inbound":       r["inbound"],
            "days_replied":  days_replied,
            "baseline_days": baseline_days,
            "response_adherence_pct": round(days_replied / baseline_days * 100, 1),
            "first_message": r["first_message"],
            "last_message":  r["last_message"],
        }
    return results


def fetch_vitals_baseline(cur, patient_ids, start_date, end_date, baseline_days):
    """
    PRIMARY OUTCOME BASELINE: reading compliance before Kestrel.
    Requires VITALS_TABLE to be set. Returns None if not configured.
    Schema confirmation tracked in: jaslogic1/RemoteCares#68
    """
    if not VITALS_TABLE:
        return {
            "status": "pending_schema_confirmation",
            "ref": "jaslogic1/RemoteCares#68",
            "note": "Set VITALS_TABLE env var once Jason confirms the table name",
        }

    try:
        # Overall compliance
        cur.execute(f"""
            SELECT
                COUNT(DISTINCT "PatientId")                AS patients_with_readings,
                COUNT(*)                                   AS total_readings,
                COUNT(DISTINCT DATE("CreatedAt"))          AS days_with_any_reading
            FROM "{VITALS_TABLE}"
            WHERE "PatientId" = ANY(%s)
              AND "CreatedAt" >= %s AND "CreatedAt" < %s
        """, (patient_ids, start_date, end_date))
        overall = dict(cur.fetchone())

        # Per-patient reading compliance
        cur.execute(f"""
            SELECT
                "PatientId"                                AS patient_id,
                COUNT(DISTINCT DATE("CreatedAt"))          AS days_with_reading,
                COUNT(*)                                   AS total_readings
            FROM "{VITALS_TABLE}"
            WHERE "PatientId" = ANY(%s)
              AND "CreatedAt" >= %s AND "CreatedAt" < %s
            GROUP BY "PatientId"
        """, (patient_ids, start_date, end_date))

        per_patient = {}
        for row in cur.fetchall():
            r = dict(row)
            pid = str(r["patient_id"])
            days = r["days_with_reading"]
            per_patient[pid] = {
                "patient_id":       pid,
                "days_with_reading": days,
                "total_readings":   r["total_readings"],
                "baseline_days":    baseline_days,
                "compliance_pct":   round(days / baseline_days * 100, 1),
            }

        total_patients = len(patient_ids)
        patients_with = overall["patients_with_readings"]
        return {
            "status":                   "collected",
            "patients_with_readings":   patients_with,
            "patients_without_readings": total_patients - patients_with,
            "total_readings":           overall["total_readings"],
            "days_with_any_reading":    overall["days_with_any_reading"],
            "avg_compliance_pct":       round(
                sum(p["compliance_pct"] for p in per_patient.values()) / len(per_patient), 1
            ) if per_patient else 0,
            "per_patient":              per_patient,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def collect_baseline(golive_date: date, baseline_days: int) -> dict:
    """Collect the full pre-Kestrel baseline snapshot."""
    end_date = golive_date
    start_date = golive_date - timedelta(days=baseline_days)

    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        patients = fetch_caprock_patients(cur)
        patient_ids = [p["patient_id"] for p in patients]

        sms_overall = fetch_sms_baseline(cur, patient_ids, start_date, end_date)
        sms_per_patient = fetch_per_patient_sms_baseline(
            cur, patient_ids, start_date, end_date, baseline_days
        )
        vitals = fetch_vitals_baseline(
            cur, patient_ids, start_date, end_date, baseline_days
        )

        # Calculate cohort-level SMS response adherence
        all_adherence = [p["response_adherence_pct"] for p in sms_per_patient.values()]
        avg_sms_adherence = round(sum(all_adherence) / len(all_adherence), 1) if all_adherence else 0

        # Patients with zero replies in the baseline window
        silent_patients = [
            p["name"] for p in patients
            if str(p["patient_id"]) not in sms_per_patient
            or sms_per_patient[str(p["patient_id"])]["inbound"] == 0
        ]

        baseline = {
            "generated_at":    datetime.utcnow().isoformat() + "Z",
            "golive_date":     golive_date.isoformat(),
            "baseline_window": {
                "start":  start_date.isoformat(),
                "end":    end_date.isoformat(),
                "days":   baseline_days,
            },
            "cohort": {
                "total_patients":    len(patients),
                "patient_list":      [
                    {"id": str(p["patient_id"]), "name": p["name"],
                     "rpm_active": p["rpm_active"], "rpm_medicaid": p["rpm_medicaid"]}
                    for p in patients
                ],
            },
            "sms_baseline": {
                "summary":  sms_overall,
                "response_rate_pct": round(
                    sms_overall["inbound_total"] / sms_overall["outbound_total"] * 100, 1
                ) if sms_overall["outbound_total"] > 0 else 0,
                "avg_patient_adherence_pct": avg_sms_adherence,
                "silent_patients":  silent_patients,
                "per_patient":      sms_per_patient,
            },
            "vitals_baseline": vitals,
            "study_hypothesis": {
                "metric":       "reading_compliance_pct",
                "target_delta": "+20% over baseline",
                "note":         "Primary outcome = % days patients submit vitals. "
                                "SMS adherence is secondary engagement metric.",
            },
        }

        return baseline

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect pre-Kestrel baseline for Caprock study")
    parser.add_argument("--golive", default=os.environ.get("KESTREL_GOLIVE", date.today().isoformat()),
                        help="Kestrel go-live date (baseline ends here)")
    parser.add_argument("--days", type=int, default=int(os.environ.get("BASELINE_DAYS", 30)),
                        help="Number of days to look back (default: 30)")
    parser.add_argument("--out", default="caprock-study/baseline.json",
                        help="Output path (default: caprock-study/baseline.json)")
    args = parser.parse_args()

    golive = date.fromisoformat(args.golive)
    print(f"Collecting {args.days}-day baseline ending {golive}...")
    print(f"  Window: {golive - timedelta(days=args.days)} to {golive}")

    baseline = collect_baseline(golive, args.days)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(baseline, f, indent=2, default=str)

    cohort = baseline["cohort"]
    sms = baseline["sms_baseline"]
    print(f"\nBaseline saved to {args.out}")
    print(f"  Cohort: {cohort['total_patients']} patients")
    print(f"  SMS: {sms['summary']['outbound_total']} outbound, "
          f"{sms['summary']['inbound_total']} inbound")
    print(f"  SMS response rate: {sms['response_rate_pct']}%")
    print(f"  Avg patient SMS adherence: {sms['avg_patient_adherence_pct']}%")
    print(f"  Silent patients (0 replies): {sms['silent_patients']}")

    vitals = baseline["vitals_baseline"]
    if vitals.get("status") == "collected":
        print(f"  Vitals: avg compliance {vitals['avg_compliance_pct']}% "
              f"({vitals['patients_with_readings']}/{cohort['total_patients']} patients with readings)")
    else:
        print(f"  Vitals: {vitals.get('status', 'n/a')} — set VITALS_TABLE to activate")
