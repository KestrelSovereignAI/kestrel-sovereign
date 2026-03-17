# Caprock Clinical Validation Study — Metrics Pipeline

Automated daily data collection for the Caprock RPM pilot.

**Study hypothesis:** Replacing static auto-bots with Kestrel AI Companion increases patient
medication adherence and engagement by ≥20% over baseline.

**Study period:** Jan 7 – Jun 20, 2026 (24 weeks) · Cohort: 10 Caprock patients

---

## Files

| File | Purpose |
|------|---------|
| `collect_metrics.py` | Daily SQL collection against staging PostgreSQL |
| `build_cumulative.py` | Rolls up all snapshots into `cumulative.json` |
| `weekly_report.py` | Posts formatted weekly summary to GitHub issues #58 and #44 |
| `caprock-daily-metrics.yml.staged` | GitHub Action (inactive — rename to activate) |
| `requirements.txt` | Python deps (`psycopg2-binary`) |
| `snapshots/` | One JSON per day once pipeline is live |
| `cumulative.json` | Rolling cumulative (rebuilt daily) |

---

## Metrics collected

### Daily
- Outbound / inbound SMS counts per day
- Patient response rate (% who replied)
- Per-patient: replied today, streak days, days-since-last-reply
- At-risk flag: patients silent for 3+ consecutive days

### Per-patient (rolling)
- Days active in study
- Days with at least one reply
- Adherence % (reply days / active days)
- 7-day trend: improving / stable / declining

### Weekly (Carlos report)
- Per-patient adherence table with status indicators (🟢🟡🔴)
- At-risk patient list for follow-up
- Study-to-date summary posted to #58 and #44

---

## Activation (when A2P 10DLC clears and SMS pipeline is live)

### Step 1 — Create a read-only DB user on staging

```sql
CREATE USER caprock_metrics WITH PASSWORD '<generate-strong-password>';
GRANT CONNECT ON DATABASE remotecares TO caprock_metrics;
GRANT USAGE ON SCHEMA public TO caprock_metrics;
GRANT SELECT ON "SmsLogs", "AgentMessages", "PatientAssistant", "Person" TO caprock_metrics;
-- Add VitalSigns / Adherence tables when confirmed:
-- GRANT SELECT ON "VitalSigns" TO caprock_metrics;
```

### Step 2 — Add GitHub Secrets to jaslogic1/RemoteCares

Go to: `Settings → Secrets and variables → Actions`

| Secret name | Value |
|-------------|-------|
| `CAPROCK_DB_HOST` | staging.onehs.net (or Container App DB endpoint) |
| `CAPROCK_DB_PORT` | 5432 |
| `CAPROCK_DB_NAME` | remotecares |
| `CAPROCK_DB_USER` | caprock_metrics |
| `CAPROCK_DB_PASSWORD` | (from Step 1) |
| `GH_TOKEN` | Token with `repo` scope (already exists) |

### Step 3 — Activate the workflow

Copy the staged workflow to the GitHub Actions folder:

```bash
cp caprock-study/caprock-daily-metrics.yml.staged \
   .github/workflows/caprock-daily-metrics.yml
git add .github/workflows/caprock-daily-metrics.yml
git commit -m "feat(study): activate Caprock daily metrics pipeline"
git push
```

### Step 4 — Trigger first run manually

Go to: `Actions → Caprock Daily Metrics → Run workflow`

Verify the first snapshot appears at `caprock-study/snapshots/YYYY-MM-DD.json`.

---

## Before-Kestrel baseline

Pre-Kestrel SMS data confirmed to exist in production:
- 18,838 SmsLogs records
- 587 inbound patient replies
- Documented in jaslogic1/RemoteCares#61

The case study document (jaslogic1/RemoteCares#48) will compare this baseline
against live study data collected by this pipeline.
