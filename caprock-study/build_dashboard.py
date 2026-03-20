"""
Caprock Clinical Validation Study — Dashboard Generator
=========================================================
Reads baseline.json, cumulative.json, and snapshots to produce a
self-contained dashboard.html with embedded charts and data.

Usage:
  python caprock-study/build_dashboard.py
  python caprock-study/build_dashboard.py --out caprock-study/dashboard.html
"""

import json
import os
import glob
import argparse
from datetime import date, timedelta


SNAPSHOTS_DIR = "caprock-study/snapshots"
BASELINE_PATH = "caprock-study/baseline.json"
CUMULATIVE_PATH = "caprock-study/cumulative.json"

STUDY_PHASES = {
    1: {"name": "Safe Zone Only", "weeks": "1-4", "color": "#22c55e"},
    2: {"name": "Add Triage Zone", "weeks": "5-8", "color": "#f59e0b"},
    3: {"name": "After-Hours 24/7", "weeks": "9-12", "color": "#8b5cf6"},
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_snapshots():
    paths = sorted(glob.glob(f"{SNAPSHOTS_DIR}/*.json"))
    snaps = []
    for p in paths:
        with open(p) as f:
            snaps.append(json.load(f))
    return snaps


def build_html(baseline, cumulative, snapshots):
    # Prepare data for JS
    baseline_json = json.dumps(baseline or {})
    cumulative_json = json.dumps(cumulative or {})
    snapshots_json = json.dumps(snapshots or [])

    # Latest snapshot for current state
    latest = snapshots[-1] if snapshots else {}
    phase = latest.get("study_phase", {})
    phase_num = phase.get("phase", 0)
    week_num = phase.get("week", 0)

    # Baseline comparison
    baseline_rate = 0
    if baseline and "sms_baseline" in baseline:
        baseline_rate = baseline["sms_baseline"].get("avg_patient_adherence_pct", 0)

    current_rate = 0
    if cumulative:
        pa = cumulative.get("per_patient_adherence", {})
        if pa:
            rates = [v["adherence_pct"] for v in pa.values()]
            current_rate = round(sum(rates) / len(rates), 1) if rates else 0

    delta = round(current_rate - baseline_rate, 1)
    delta_sign = "+" if delta > 0 else ""
    delta_color = "#22c55e" if delta > 0 else "#ef4444" if delta < 0 else "#6b7280"

    cohort_size = latest.get("cohort_size", 0)
    total_days = cumulative.get("total_days_collected", 0) if cumulative else 0
    trend = cumulative.get("study_trend", "n/a") if cumulative else "n/a"
    total_escalations = cumulative.get("total_escalations", 0) if cumulative else 0

    # At-risk patients from latest snapshot
    at_risk = latest.get("at_risk_patients", [])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Caprock Clinical Validation Study - Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-bottom: 1px solid #334155; padding: 24px 32px; }}
  .header h1 {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
  .header .subtitle {{ font-size: 14px; color: #94a3b8; margin-top: 4px; }}
  .header .meta {{ display: flex; gap: 24px; margin-top: 12px; font-size: 13px; color: #64748b; }}
  .header .meta span {{ background: #1e293b; border: 1px solid #334155; padding: 4px 12px; border-radius: 6px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; padding: 24px 32px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }}
  .card .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 4px; }}
  .card .value {{ font-size: 32px; font-weight: 700; color: #f8fafc; }}
  .card .sub {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
  .card.highlight {{ border-color: #3b82f6; }}
  .card.delta-positive .value {{ color: #22c55e; }}
  .card.delta-negative .value {{ color: #ef4444; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 0 32px 24px; }}
  .chart-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }}
  .chart-card h3 {{ font-size: 15px; color: #cbd5e1; margin-bottom: 16px; }}
  .chart-card canvas {{ max-height: 300px; }}
  .table-section {{ padding: 0 32px 24px; }}
  .table-section h3 {{ font-size: 15px; color: #cbd5e1; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b; border: 1px solid #334155; border-radius: 12px; overflow: hidden; }}
  th {{ background: #0f172a; text-align: left; padding: 12px 16px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; border-bottom: 1px solid #334155; }}
  td {{ padding: 10px 16px; font-size: 14px; border-bottom: 1px solid #1e293b; }}
  tr:hover td {{ background: #334155; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .badge-green {{ background: #064e3b; color: #34d399; }}
  .badge-yellow {{ background: #713f12; color: #fbbf24; }}
  .badge-red {{ background: #7f1d1d; color: #f87171; }}
  .phase-bar {{ display: flex; gap: 4px; padding: 0 32px 24px; }}
  .phase-segment {{ flex: 1; height: 8px; border-radius: 4px; background: #334155; position: relative; }}
  .phase-segment.active {{ background: #3b82f6; }}
  .phase-segment.completed {{ background: #22c55e; }}
  .phase-label {{ display: flex; justify-content: space-between; padding: 0 32px; margin-bottom: 8px; }}
  .phase-label span {{ font-size: 11px; color: #64748b; }}
  .alert-bar {{ margin: 0 32px 24px; padding: 16px 20px; background: #7f1d1d33; border: 1px solid #7f1d1d; border-radius: 8px; }}
  .alert-bar h4 {{ color: #f87171; font-size: 14px; margin-bottom: 8px; }}
  .alert-bar ul {{ list-style: none; padding: 0; }}
  .alert-bar li {{ font-size: 13px; color: #fca5a5; padding: 2px 0; }}
  .section-title {{ padding: 0 32px; margin-bottom: 8px; font-size: 18px; color: #cbd5e1; font-weight: 600; }}
  .footer {{ text-align: center; padding: 24px; font-size: 12px; color: #475569; border-top: 1px solid #1e293b; }}
  @media (max-width: 768px) {{
    .charts {{ grid-template-columns: 1fr; }}
    .grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Caprock Clinical Validation Study</h1>
  <div class="subtitle">Kestrel AI Companion — Patient Engagement & Adherence Dashboard</div>
  <div class="meta">
    <span>Cohort: {cohort_size} patients</span>
    <span>Study Day: {total_days}</span>
    <span>Phase {phase_num}: {phase.get('name', 'N/A')}</span>
    <span>Week {week_num}</span>
  </div>
</div>

<!-- Phase Progress -->
<div style="padding: 24px 32px 0;">
  <div class="section-title" style="padding: 0; margin-bottom: 12px;">Study Phase Progress</div>
</div>
<div class="phase-label">
  <span>Phase 1: Safe Zone (Wk 1-4)</span>
  <span>Phase 2: Triage (Wk 5-8)</span>
  <span>Phase 3: 24/7 (Wk 9-12)</span>
</div>
<div class="phase-bar">
  {"".join(f'<div class="phase-segment {"completed" if w < week_num else "active" if w == week_num else ""}"></div>' for w in range(1, 13))}
</div>

<!-- KPI Cards -->
<div class="grid">
  <div class="card highlight">
    <div class="label">Current Avg Adherence</div>
    <div class="value">{current_rate}%</div>
    <div class="sub">vs {baseline_rate}% baseline</div>
  </div>
  <div class="card {"delta-positive" if delta > 0 else "delta-negative" if delta < 0 else ""}">
    <div class="label">Change from Baseline</div>
    <div class="value">{delta_sign}{delta}%</div>
    <div class="sub">Target: +20%</div>
  </div>
  <div class="card">
    <div class="label">7-Day Trend</div>
    <div class="value">{cumulative.get('last_7d_response_rate_pct', 0) if cumulative else 0}%</div>
    <div class="sub">{"Improving" if trend == "improving" else "Stable" if trend == "stable" else "Declining" if trend == "declining" else "Collecting..."}</div>
  </div>
  <div class="card">
    <div class="label">Total Escalations</div>
    <div class="value">{total_escalations}</div>
    <div class="sub">AI-triggered care alerts</div>
  </div>
</div>

<!-- At-Risk Alert -->
{"" if not at_risk else f'''<div class="alert-bar">
  <h4>At-Risk Patients (3+ days without reply)</h4>
  <ul>{"".join(f"<li>{name} - please follow up</li>" for name in at_risk)}</ul>
</div>'''}

<!-- Charts -->
<div class="charts">
  <div class="chart-card">
    <h3>Daily Response Rate (%)</h3>
    <canvas id="responseChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Daily SMS Volume</h3>
    <canvas id="smsChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Per-Patient Adherence (Study-to-Date)</h3>
    <canvas id="adherenceChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Baseline vs Current Adherence</h3>
    <canvas id="comparisonChart"></canvas>
  </div>
</div>

<!-- Per-Patient Table -->
<div class="table-section">
  <h3>Per-Patient Detail</h3>
  <table>
    <thead>
      <tr>
        <th>Patient</th>
        <th>Days Replied</th>
        <th>Adherence %</th>
        <th>Status</th>
        <th>Baseline</th>
        <th>Change</th>
        <th>Trend</th>
      </tr>
    </thead>
    <tbody id="patientTable"></tbody>
  </table>
</div>

<div class="footer">
  Kestrel AI Companion &middot; Caprock Home Health Services &middot; Generated {date.today().isoformat()}
  <br>Data source: caprock-study/snapshots/ &middot; Refresh by running: python caprock-study/build_dashboard.py
</div>

<script>
const baseline = {baseline_json};
const cumulative = {cumulative_json};
const snapshots = {snapshots_json};

// Chart defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

// 1. Response Rate Chart
const dates = (cumulative.daily_series || []).map(s => s.date.slice(5));
const rates = (cumulative.daily_series || []).map(s => s.response_rate);
const baselineRate = baseline?.sms_baseline?.avg_patient_adherence_pct || 0;

new Chart(document.getElementById('responseChart'), {{
  type: 'line',
  data: {{
    labels: dates,
    datasets: [
      {{
        label: 'Daily Response Rate',
        data: rates,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 4,
        pointBackgroundColor: '#3b82f6',
      }},
      {{
        label: 'Baseline',
        data: dates.map(() => baselineRate),
        borderColor: '#ef4444',
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
      }},
      {{
        label: 'Target (+20%)',
        data: dates.map(() => baselineRate + 20),
        borderColor: '#22c55e',
        borderDash: [3, 3],
        pointRadius: 0,
        fill: false,
      }}
    ]
  }},
  options: {{
    responsive: true,
    scales: {{
      y: {{ min: 0, max: 100, ticks: {{ callback: v => v + '%' }} }}
    }},
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }}
    }}
  }}
}});

// 2. SMS Volume Chart
const outbound = snapshots.map(s => s.daily_sms?.outbound_total || 0);
const inbound = snapshots.map(s => s.daily_sms?.inbound_total || 0);

new Chart(document.getElementById('smsChart'), {{
  type: 'bar',
  data: {{
    labels: dates,
    datasets: [
      {{
        label: 'Outbound (AI to Patient)',
        data: outbound,
        backgroundColor: '#3b82f6',
        borderRadius: 4,
      }},
      {{
        label: 'Inbound (Patient Reply)',
        data: inbound,
        backgroundColor: '#22c55e',
        borderRadius: 4,
      }}
    ]
  }},
  options: {{
    responsive: true,
    scales: {{ y: {{ beginAtZero: true }} }},
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }}
    }}
  }}
}});

// 3. Per-Patient Adherence Bar Chart
const adherence = cumulative.per_patient_adherence || {{}};
const patientNames = Object.keys(adherence).sort((a, b) =>
  adherence[b].adherence_pct - adherence[a].adherence_pct
);
const adherencePcts = patientNames.map(n => adherence[n].adherence_pct);
const adherenceColors = adherencePcts.map(p =>
  p >= 70 ? '#22c55e' : p >= 40 ? '#f59e0b' : '#ef4444'
);

new Chart(document.getElementById('adherenceChart'), {{
  type: 'bar',
  data: {{
    labels: patientNames.map(n => n.split(',')[0]),
    datasets: [{{
      label: 'Adherence %',
      data: adherencePcts,
      backgroundColor: adherenceColors,
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    scales: {{ x: {{ min: 0, max: 100, ticks: {{ callback: v => v + '%' }} }} }},
    plugins: {{
      legend: {{ display: false }},
    }}
  }}
}});

// 4. Baseline vs Current Comparison Chart
const baselinePerPatient = baseline?.sms_baseline?.per_patient || {{}};
const compNames = patientNames;
const compBaseline = compNames.map(n => {{
  const patients = baseline?.cohort?.patient_list || [];
  const p = patients.find(p => p.name === n);
  if (p && baselinePerPatient[p.id]) return baselinePerPatient[p.id].response_adherence_pct;
  return 0;
}});
const compCurrent = compNames.map(n => adherence[n]?.adherence_pct || 0);

new Chart(document.getElementById('comparisonChart'), {{
  type: 'bar',
  data: {{
    labels: compNames.map(n => n.split(',')[0]),
    datasets: [
      {{
        label: 'Baseline (Pre-Kestrel)',
        data: compBaseline,
        backgroundColor: '#64748b',
        borderRadius: 4,
      }},
      {{
        label: 'Current (With Kestrel)',
        data: compCurrent,
        backgroundColor: '#3b82f6',
        borderRadius: 4,
      }}
    ]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    scales: {{ x: {{ min: 0, max: 100, ticks: {{ callback: v => v + '%' }} }} }},
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }}
    }}
  }}
}});

// 5. Per-Patient Table
const tableBody = document.getElementById('patientTable');
const latestSnap = snapshots[snapshots.length - 1] || {{}};
const latestDetails = latestSnap.patient_details || [];

patientNames.forEach(name => {{
  const ad = adherence[name];
  const patients = baseline?.cohort?.patient_list || [];
  const p = patients.find(p => p.name === name);
  const bl = p && baselinePerPatient[p.id]
    ? baselinePerPatient[p.id].response_adherence_pct : 0;
  const cur = ad?.adherence_pct || 0;
  const change = (cur - bl).toFixed(1);
  const detail = latestDetails.find(d => d.name === name) || {{}};
  const trend = detail.weekly_trend?.trend || 'n/a';

  let statusBadge;
  if (cur >= 70) statusBadge = '<span class="badge badge-green">On Track</span>';
  else if (cur >= 40) statusBadge = '<span class="badge badge-yellow">Monitor</span>';
  else statusBadge = '<span class="badge badge-red">At Risk</span>';

  const changeColor = parseFloat(change) > 0 ? '#22c55e' : parseFloat(change) < 0 ? '#ef4444' : '#94a3b8';
  const changeSign = parseFloat(change) > 0 ? '+' : '';

  const trendIcon = trend === 'improving' ? 'Improving' : trend === 'declining' ? 'Declining' : 'Stable';

  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${{name}}</td>
    <td>${{ad?.days_replied || 0}}/${{ad?.days_active || 0}}</td>
    <td>${{cur}}%</td>
    <td>${{statusBadge}}</td>
    <td>${{bl}}%</td>
    <td style="color:${{changeColor}}">${{changeSign}}${{change}}%</td>
    <td>${{trendIcon}}</td>
  `;
  tableBody.appendChild(row);
}});
</script>
</body>
</html>"""
    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Caprock study dashboard")
    parser.add_argument("--out", default="caprock-study/dashboard.html")
    args = parser.parse_args()

    baseline = load_json(BASELINE_PATH)
    cumulative = load_json(CUMULATIVE_PATH)
    snapshots = load_snapshots()

    print(f"Loaded: baseline={'yes' if baseline else 'no'}, "
          f"cumulative={'yes' if cumulative else 'no'}, "
          f"snapshots={len(snapshots)}")

    html = build_html(baseline, cumulative, snapshots)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard saved to {args.out}")
    print(f"Open in browser: file:///{os.path.abspath(args.out).replace(os.sep, '/')}")
