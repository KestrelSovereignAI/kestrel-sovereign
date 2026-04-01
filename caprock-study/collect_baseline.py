"""
Caprock Clinical Validation Study — Pre-Kestrel Baseline Collector
===================================================================
Queries the PRODUCTION RemoteCares SQL Server database for the 30 days
BEFORE Kestrel go-live to establish the "before" comparison for the study.

This script must run ONCE, against PRODUCTION (not staging), ideally
on or just before Kestrel go-live day for these patients.

Output:  caprock-study/baseline.json

Environment variables:
  PROD_DB_SERVER     — production SQL Server host (e.g. myserver.database.windows.net)
  PROD_DB_NAME       — (default: remotecares)
  PROD_DB_USER       — read-only user with SELECT on relevant tables
  PROD_DB_PASSWORD
  KESTREL_GOLIVE     — ISO date when Kestrel goes live (e.g. 2026-03-24)
  BASELINE_DAYS      — number of days to look back (default: 30)
  ODBC_DRIVER        — ODBC driver name (default: ODBC Driver 18 for SQL Server)

Usage:
  python collect_baseline.py
  python collect_baseline.py --golive 2026-03-24 --days 30 --out caprock-study/baseline.json
"""

import os
import json
import argparse
import pyodbc
from datetime import date, datetime, timedelta


ODBC_DRIVER = os.environ.get("ODBC_DRIVER", "ODBC Driver 18 for SQL Server")


def get_connection():
    server = os.environ["PROD_DB_SERVER"]
    database = os.environ.get("PROD_DB_NAME", "remotecares")
    user = os.environ["PROD_DB_USER"]
    password = os.environ["PROD_DB_PASSWORD"]
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
    """Convert pyodbc cursor rows to list of dicts."""
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def row_to_dict(cursor):
    """Convert a single pyodbc cursor row to a dict."""
    columns = [col[0] for col in cursor.description]
    row = cursor.fetchone()
    return dict(zip(columns, row)) if row else {}


def fetch_caprock_patients(cur):
    """Identify Caprock RPM Medicaid patients."""
    cur.execute("""
        SELECT
            p.Oid              AS patient_id,
            p.FullName         AS name,
            p.MobilePhone      AS mobile,
            p.IsRpmActive      AS rpm_active,
            p.IsRpmMedicaidActive AS rpm_medicaid
        FROM Patients p
        WHERE (p.IsRpmActive = 1 OR p.IsRpmMedicaidActive = 1)
    """)
    return rows_to_dicts(cur)


def fetch_sms_baseline(cur, patient_ids, start_date, end_date):
    """
    SMS engagement baseline: how often did patients reply to the OLD
    auto-bot system before Kestrel was introduced?
    Direction: 0 = outbound, 1 = inbound
    """
    placeholders = ",".join("?" for _ in patient_ids)
    cur.execute(f"""
        SELECT
            COUNT(*)                                                            AS total_messages,
            SUM(CASE WHEN Direction = 0 THEN 1 ELSE 0 END)                    AS outbound_total,
            SUM(CASE WHEN Direction = 1 THEN 1 ELSE 0 END)                    AS inbound_total,
            COUNT(DISTINCT CASE WHEN Direction = 0 THEN PatientId END)         AS patients_messaged,
            COUNT(DISTINCT CASE WHEN Direction = 1 THEN PatientId END)         AS patients_replied,
            COUNT(DISTINCT CAST(CreatedOn AS DATE))                            AS days_with_activity
        FROM SmsLogs
        WHERE PatientId IN ({placeholders})
          AND CreatedOn >= ? AND CreatedOn < ?
    """, patient_ids + [start_date, end_date])
    return row_to_dict(cur)


def fetch_per_patient_sms_baseline(cur, patient_ids, start_date, end_date, baseline_days):
    """Per-patient SMS engagement during the baseline window."""
    placeholders = ",".join("?" for _ in patient_ids)
    cur.execute(f"""
        SELECT
            PatientId                                                       AS patient_id,
            SUM(CASE WHEN Direction = 0 THEN 1 ELSE 0 END)                AS outbound,
            SUM(CASE WHEN Direction = 1 THEN 1 ELSE 0 END)                AS inbound,
            COUNT(DISTINCT CASE WHEN Direction = 1
                           THEN CAST(CreatedOn AS DATE) END)               AS days_replied,
            MIN(CreatedOn)                                                  AS first_message,
            MAX(CreatedOn)                                                  AS last_message
        FROM SmsLogs
        WHERE PatientId IN ({placeholders})
          AND CreatedOn >= ? AND CreatedOn < ?
        GROUP BY PatientId
    """, patient_ids + [start_date, end_date])

    results = {}
    for row in rows_to_dicts(cur):
        pid = str(row["patient_id"])
        days_replied = row["days_replied"] or 0
        results[pid] = {
            "patient_id":    pid,
            "outbound":      row["outbound"] or 0,
            "inbound":       row["inbound"] or 0,
            "days_replied":  days_replied,
            "baseline_days": baseline_days,
            "response_adherence_pct": round(days_replied / baseline_days * 100, 1) if baseline_days else 0,
            "first_message": row["first_message"],
            "last_message":  row["last_message"],
        }
    return results


def fetch_vitals_baseline(cur, patient_ids, start_date, end_date, baseline_days):
    """
    PRIMARY OUTCOME BASELINE: reading compliance before Kestrel.
    Uses VitalSigns table joined to Encounters for date filtering.
    Only counts Source = 0 (Device readings), excluding missed readings.
    """
    placeholders = ",".join("?" for _ in patient_ids)
    try:
        # Overall compliance
        cur.execute(f"""
            SELECT
                COUNT(DISTINCT vs.PatientId)                      AS patients_with_readings,
                COUNT(*)                                          AS total_readings,
                COUNT(DISTINCT CAST(e.StartOn AS DATE))           AS days_with_any_reading
            FROM VitalSigns vs
            JOIN Encounters e ON e.Oid = vs.EncounterId
            WHERE vs.PatientId IN ({placeholders})
              AND e.StartOn >= ? AND e.StartOn < ?
              AND vs.Source = 0
        """, patient_ids + [start_date, end_date])
        overall = row_to_dict(cur)

        # Per-patient reading compliance
        cur.execute(f"""
            SELECT
                vs.PatientId                                      AS patient_id,
                COUNT(DISTINCT CAST(e.StartOn AS DATE))           AS days_with_reading,
                COUNT(*)                                          AS total_readings
            FROM VitalSigns vs
            JOIN Encounters e ON e.Oid = vs.EncounterId
            WHERE vs.PatientId IN ({placeholders})
              AND e.StartOn >= ? AND e.StartOn < ?
              AND vs.Source = 0
            GROUP BY vs.PatientId
        """, patient_ids + [start_date, end_date])

        per_patient = {}
        for row in rows_to_dicts(cur):
            pid = str(row["patient_id"])
            days = row["days_with_reading"] or 0
            per_patient[pid] = {
                "patient_id":       pid,
                "days_with_reading": days,
                "total_readings":   row["total_readings"],
                "baseline_days":    baseline_days,
                "compliance_pct":   round(days / baseline_days * 100, 1) if baseline_days else 0,
            }

        total_patients = len(patient_ids)
        patients_with = overall.get("patients_with_readings", 0) or 0
        return {
            "status":                   "collected",
            "patients_with_readings":   patients_with,
            "patients_without_readings": total_patients - patients_with,
            "total_readings":           overall.get("total_readings", 0) or 0,
            "days_with_any_reading":    overall.get("days_with_any_reading", 0) or 0,
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
    cur = conn.cursor()

    try:
        patients = fetch_caprock_patients(cur)
        patient_ids = [p["patient_id"] for p in patients]

        if not patient_ids:
            return {"error": "No active RPM patients found", "patients": []}

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

        outbound_total = sms_overall.get("outbound_total", 0) or 0
        inbound_total = sms_overall.get("inbound_total", 0) or 0

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
                    inbound_total / outbound_total * 100, 1
                ) if outbound_total > 0 else 0,
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

    cohort = baseline.get("cohort", {})
    sms = baseline.get("sms_baseline", {})
    summary = sms.get("summary", {})
    print(f"\nBaseline saved to {args.out}")
    print(f"  Cohort: {cohort.get('total_patients', 0)} patients")
    print(f"  SMS: {summary.get('outbound_total', 0)} outbound, "
          f"{summary.get('inbound_total', 0)} inbound")
    print(f"  SMS response rate: {sms.get('response_rate_pct', 0)}%")
    print(f"  Avg patient SMS adherence: {sms.get('avg_patient_adherence_pct', 0)}%")
    print(f"  Silent patients (0 replies): {sms.get('silent_patients', [])}")

    vitals = baseline.get("vitals_baseline", {})
    if vitals.get("status") == "collected":
        print(f"  Vitals: avg compliance {vitals['avg_compliance_pct']}% "
              f"({vitals['patients_with_readings']}/{cohort.get('total_patients', 0)} patients with readings)")
    else:
        print(f"  Vitals: {vitals.get('status', 'n/a')} — {vitals.get('message', '')}")
