"""
Caprock Clinical Validation Study — Weekly Report Generator
============================================================
Generates a formatted weekly summary from cumulative.json and
posts it as a comment to:
  - jaslogic1/RemoteCares#58  (CAPROCK MASTER — daily tracking)
  - jaslogic1/RemoteCares#44  (Weekly reports to Carlos)

Run every Friday (via GitHub Action schedule: "0 9 * * 5")

Requires:
  GH_TOKEN   — GitHub token with repo scope
  WEEK_END   — ISO date of the Friday being reported (default: today)
"""

import os
import json
import argparse
import urllib.request
import urllib.error
from datetime import date, timedelta


CUMULATIVE_PATH = "caprock-study/cumulative.json"
REPORT_ISSUES = [
    ("jaslogic1/RemoteCares", 58),
    ("jaslogic1/RemoteCares", 44),
]


def load_cumulative() -> dict:
    with open(CUMULATIVE_PATH) as f:
        return json.load(f)


def build_weekly_markdown(cumulative: dict, week_end: date) -> str:
    week_start = week_end - timedelta(days=6)

    # Filter series to this week
    series = [
        s for s in cumulative.get("daily_series", [])
        if week_start.isoformat() <= s["date"] <= week_end.isoformat()
    ]

    if not series:
        days_with_data = 0
        avg_rate = 0
        replied_count = 0
        patients_messaged = 0
    else:
        rates = [s["response_rate"] for s in series]
        avg_rate = round(sum(rates) / len(rates), 1)
        days_with_data = len(series)
        replied_count = max(s["patients_replied"] for s in series)
        patients_messaged = max(s["patients_messaged"] for s in series)

    # Per-patient adherence table
    adherence = cumulative.get("per_patient_adherence", {})
    patient_rows = []
    for name, stats in sorted(adherence.items()):
        pct = stats["adherence_pct"]
        trend_icon = "📈" if cumulative.get("study_trend") == "improving" else \
                     "📉" if cumulative.get("study_trend") == "declining" else "➡️"
        status = "🟢" if pct >= 70 else "🟡" if pct >= 40 else "🔴"
        patient_rows.append(
            f"| {name} | {stats['days_replied']}/{stats['days_active']} days | {pct}% | {status} |"
        )

    # At-risk patients this week
    at_risk_this_week = set()
    for s in series:
        at_risk_this_week.update(s.get("at_risk", []))

    risk_section = ""
    if at_risk_this_week:
        risk_section = "\n### ⚠️ At-Risk Patients (3+ days without reply)\n"
        for name in sorted(at_risk_this_week):
            risk_section += f"- {name} — please follow up\n"

    study_days_total = cumulative.get("total_days_collected", 0)
    cohort = cumulative.get("cohort_size", 10)

    return f"""## Kestrel AI Companion — Weekly Report
**Practice:** Caprock Home Health Services
**Week:** {week_start.strftime('%B %d')} – {week_end.strftime('%B %d, %Y')}
**Study Day:** {study_days_total} of 168 (24-week pilot)

---

### 📊 This Week Summary

| Metric | Value |
|--------|-------|
| Active patients | {cohort} |
| Days with data | {days_with_data}/7 |
| Avg daily response rate | **{avg_rate}%** |
| 7-day trend | {cumulative.get('study_trend', 'n/a')} |
| Total escalations to date | {cumulative.get('total_escalations', 0)} |

---

### 👤 Per-Patient Adherence (study-to-date)

| Patient | Reply Days | Adherence | Status |
|---------|-----------|-----------|--------|
{chr(10).join(patient_rows) if patient_rows else "| No data yet | — | — | — |"}

🟢 ≥70% · 🟡 40–69% · 🔴 <40%
{risk_section}
---

*Generated automatically by Kestrel study pipeline · {date.today().isoformat()}*
*Source data: `caprock-study/cumulative.json` · Full history: `caprock-study/snapshots/`*
"""


def post_github_comment(repo: str, issue_num: int, body: str):
    token = os.environ["GH_TOKEN"]
    url = f"https://api.github.com/repos/{repo}/issues/{issue_num}/comments"
    payload = json.dumps({"body": body}).encode()
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "KestrelBot/1.0",
            "Accept": "application/vnd.github.v3+json",
        }
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            print(f"  Posted to {repo}#{issue_num}: {result.get('html_url')}")
    except urllib.error.HTTPError as e:
        print(f"  FAILED {repo}#{issue_num}: {e.read().decode()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week-end", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true", help="Print report, don't post")
    args = parser.parse_args()

    week_end = date.fromisoformat(args.week_end)
    cumulative = load_cumulative()
    report_md = build_weekly_markdown(cumulative, week_end)

    print(report_md)
    print()

    if not args.dry_run:
        for repo, num in REPORT_ISSUES:
            post_github_comment(repo, num, report_md)
