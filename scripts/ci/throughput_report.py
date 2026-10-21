#!/usr/bin/env python3
"""Measure where a repository's development time goes, from GitHub alone.

Reports, for PRs merged in a window:

* stage times: issue ``agent-ready`` -> ``agent-claimed`` (queue),
  ``agent-claimed`` -> PR opened (implement), PR opened -> merged;
* work in flight: how many PRs were between their claim (or first commit)
  and their merge, sampled every 10 minutes;
* merge-from-base commits and the gap between consecutive merges;
* per-job CI durations for recent successful ``pull_request`` runs.

It reads through the ``gh`` CLI, so it uses whatever GitHub credentials
``gh`` already has. The research note that uses it is
``docs/research/DEVELOPMENT_THROUGHPUT.md``; re-run it after a change to
compare against that note's baseline.

    python3 scripts/ci/throughput_report.py --repo OWNER/NAME --since 2026-09-16
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone

PR_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: MERGED, first: 40, after: $cursor,
                 orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number createdAt mergedAt
        commits(first: 100) {
          nodes { commit { authoredDate parents { totalCount } } }
        }
        closingIssuesReferences(first: 1) {
          nodes {
            timelineItems(itemTypes: [LABELED_EVENT], first: 50) {
              nodes { ... on LabeledEvent { createdAt label { name } } }
            }
          }
        }
      }
    }
  }
}
"""


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def gh_json(*args: str) -> object:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def merged_prs(repo: str, since: datetime) -> list[dict]:
    owner, name = repo.split("/", 1)
    cursor: str | None = None
    prs: list[dict] = []
    while True:
        args = ["api", "graphql", "-f", f"query={PR_QUERY}", "-f", f"owner={owner}", "-f", f"name={name}"]
        if cursor:
            args += ["-f", f"cursor={cursor}"]
        page = gh_json(*args)["data"]["repository"]["pullRequests"]
        prs += page["nodes"]
        # Creation order supplies no cutoff for a merge-time window: even a
        # years-old PR on a later page may have merged inside that window.
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return [pr for pr in prs if parse_time(pr["mergedAt"]) >= since]


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def describe(label: str, values: list[float], unit: str) -> str:
    if not values:
        return f"  {label:34} (no data)"
    return (
        f"  {label:34} n={len(values):4}  median {statistics.median(values):7.1f}{unit}"
        f"  p75 {quantile(values, 0.75):7.1f}{unit}  p90 {quantile(values, 0.9):7.1f}{unit}"
    )


def label_times(pr: dict, label: str) -> list[datetime]:
    issues = pr["closingIssuesReferences"]["nodes"]
    if not issues:
        return []
    return [
        parse_time(event["createdAt"])
        for event in issues[0]["timelineItems"]["nodes"]
        if event and event["label"]["name"] == label
    ]


def report_prs(prs: list[dict], since: datetime) -> None:
    queue, implement, review, merges_from_base = [], [], [], 0
    spans: list[tuple[datetime, datetime]] = []
    for pr in prs:
        opened, merged = parse_time(pr["createdAt"]), parse_time(pr["mergedAt"])
        review.append((merged - opened).total_seconds() / 60)
        commits = [node["commit"] for node in pr["commits"]["nodes"]]
        merges_from_base += any(c["parents"]["totalCount"] > 1 for c in commits)
        # Talon amends and force-pushes, so a commit's author date is
        # usually minutes before the PR opens. The claim is the better start.
        start = min([parse_time(c["authoredDate"]) for c in commits] + [opened])
        claims = label_times(pr, "agent-claimed")
        if claims:
            # The claim that led to this PR: the last one before it opened.
            claim = max([t for t in claims if t <= opened] or [min(claims)])
            implement.append((opened - claim).total_seconds() / 60)
            start = min(start, claim)
            ready = label_times(pr, "agent-ready")
            if ready and min(ready) <= claim:
                queue.append((claim - min(ready)).total_seconds() / 3600)
        spans.append((start, merged))

    days = max((datetime.now(timezone.utc) - since).days, 1)
    print(f"Merged PRs: {len(prs)} in {days} days ({len(prs) / days:.1f}/day)")
    print(describe("ready -> claimed (queue)", queue, "h"))
    print(describe("claimed -> PR opened (implement)", implement, "m"))
    print(describe("PR opened -> merged", review, "m"))
    merged_at = sorted(end for _, end in spans)
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(merged_at, merged_at[1:])]
    print(describe("gap between merges", gaps, "m"))
    print(f"  PRs with a merge-from-base commit: {merges_from_base} ({100 * merges_from_base / max(len(prs), 1):.0f}%)")

    events = sorted([(start, 1) for start, _ in spans] + [(end, -1) for _, end in spans])
    if not events or events[0][0] == events[-1][0]:
        print("  Work in flight (claim or first commit..merge): (no data)")
        return
    samples, in_flight, index = [], 0, 0
    moment, last = events[0][0], events[-1][0]
    while moment < last:
        while index < len(events) and events[index][0] <= moment:
            in_flight += events[index][1]
            index += 1
        samples.append(in_flight)
        moment += timedelta(minutes=10)
    print(
        f"  Work in flight (claim or first commit..merge): mean {statistics.mean(samples):.1f}, "
        f"p90 {quantile(samples, 0.9)}, max {max(samples)}, "
        f"none in flight {100 * samples.count(0) / len(samples):.0f}% of the time"
    )


def report_ci(repo: str, workflow: str, runs: int) -> None:
    listed = gh_json(
        "run", "list", "-R", repo, "-w", workflow, "--event", "pull_request",
        "--status", "success", "--limit", str(runs), "--json", "databaseId",
    )
    durations: dict[str, list[float]] = defaultdict(list)
    waits: list[float] = []
    for run in listed:
        jobs = gh_json("api", f"repos/{repo}/actions/runs/{run['databaseId']}/jobs", "--jq", ".jobs")
        for job in jobs:
            if job["conclusion"] != "success" or not job["started_at"] or not job["completed_at"]:
                continue
            started = parse_time(job["started_at"])
            durations[job["name"]].append((parse_time(job["completed_at"]) - started).total_seconds() / 60)
            waits.append((started - parse_time(job["created_at"])).total_seconds() / 60)
    print(f"CI jobs over the last {len(listed)} successful pull_request runs of {workflow}:")
    for name, values in sorted(durations.items(), key=lambda item: -statistics.median(item[1])):
        print(describe(name, values, "m"))
    print(describe("wait for a runner", waits, "m"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--since", required=True, help="window start, YYYY-MM-DD (UTC)")
    parser.add_argument("--workflow", default="ci.yml", help="workflow file for CI job timings")
    parser.add_argument("--ci-runs", type=int, default=25, help="successful PR runs to sample")
    args = parser.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    report_prs(merged_prs(args.repo, since), since)
    if args.ci_runs:
        print()
        report_ci(args.repo, args.workflow, args.ci_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
