"""Offline deterministic throughput reporting; no GitHub credentials/network."""

from datetime import datetime, timezone

import pytest

from scripts.ci import throughput_report as report


def test_empty_merge_window_reports_no_work_in_flight(capsys):
    report.report_prs([], datetime(2026, 10, 1, tzinfo=timezone.utc))
    output = capsys.readouterr().out
    assert "Merged PRs: 0" in output
    assert "Work in flight" in output and "(no data)" in output


def test_zero_duration_merge_window_reports_no_work_in_flight(capsys):
    report.report_prs(
        [
            {
                "number": 1,
                "createdAt": "2026-10-01T00:00:00Z",
                "mergedAt": "2026-10-01T00:00:00Z",
                "commits": {"nodes": []},
                "closingIssuesReferences": {"nodes": []},
            }
        ],
        datetime(2026, 10, 1, tzinfo=timezone.utc),
    )
    output = capsys.readouterr().out
    assert "Merged PRs: 1" in output
    assert "Work in flight" in output and "(no data)" in output


def test_creation_order_does_not_truncate_recent_merge_window(monkeypatch):
    # Declared GitHub response pages test the report's selection logic only.
    # They are not live GitHub/API or independent vendor-metric evidence.
    pages = [
        {
            "pageInfo": {"hasNextPage": True, "endCursor": "older-creation-page"},
            "nodes": [
                {
                    "number": 1,
                    "createdAt": "2026-08-01T00:00:00Z",
                    "mergedAt": "2026-09-01T00:00:00Z",
                }
            ],
        },
        {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": [
                {
                    "number": 2,
                    "createdAt": "2026-01-01T00:00:00Z",
                    "mergedAt": "2026-10-10T00:00:00Z",
                }
            ],
        },
    ]
    requests = []

    def read_page(*args):
        requests.append(args)
        return {"data": {"repository": {"pullRequests": pages[len(requests) - 1]}}}

    monkeypatch.setattr(report, "gh_json", read_page)
    selected = report.merged_prs(
        "local/declared-response", datetime(2026, 10, 1, tzinfo=timezone.utc)
    )
    assert [pr["number"] for pr in selected] == [2]
    assert len(requests) == 2
    assert "cursor=older-creation-page" in requests[1]


@pytest.mark.parametrize("prs", [[], [{"mergedAt": "2026-09-30T00:00:00Z"}]])
def test_completed_pagination_filters_only_by_merge_time(monkeypatch, prs):
    monkeypatch.setattr(
        report,
        "gh_json",
        lambda *_: {
            "data": {
                "repository": {
                    "pullRequests": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": prs,
                    }
                }
            }
        },
    )
    assert (
        report.merged_prs(
            "local/declared-response", datetime(2026, 10, 1, tzinfo=timezone.utc)
        )
        == []
    )
