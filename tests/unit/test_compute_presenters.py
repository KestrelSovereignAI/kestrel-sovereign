"""Golden contracts for compute read-tool presentation."""

import hashlib
import inspect
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.features.compute.feature import ComputeFeature
from kestrel_sovereign.features.compute.models import (
    ComputeScript,
    ExecutionRecord,
    ScriptState,
    SecurityFinding,
)
from kestrel_sovereign.features.compute.presenters import (
    present_execution_history,
    present_script_detail,
    present_script_list,
)


def _script(
    *,
    script_id: str,
    name: str,
    state: ScriptState,
    risk_score: int = 0,
) -> ComputeScript:
    return ComputeScript(
        id=script_id,
        name=name,
        language="python",
        content="pass",
        purpose="presenter contract",
        state=state,
        risk_score=risk_score,
        created_at=datetime(2026, 1, 2, 3, 4, 5),
        updated_at=datetime(2026, 1, 2, 3, 4, 5),
    )


def test_present_script_list_empty_golden() -> None:
    result = present_script_list([])

    assert result.to_dict() == {
        "status": "ok",
        "confirmation": "No scripts found.",
        "data": {"scripts": [], "count": 0},
    }


def test_present_script_list_all_states_and_long_name_golden() -> None:
    long_name = "12345678901234567890-this-part-is-not-rendered"
    scripts = [
        _script(
            script_id=script_id,
            name=long_name,
            state=state,
            risk_score=risk_score,
        )
        for script_id, state, risk_score in (
            ("draft000-full", ScriptState.DRAFT, 0),
            ("signed00-full", ScriptState.SIGNED, 1),
            ("pending0-full", ScriptState.PENDING_REVIEW, 2),
            ("approved-full", ScriptState.APPROVED, 3),
            ("rejected-full", ScriptState.REJECTED, 4),
            ("queued00-full", ScriptState.QUEUED, 5),
            ("running0-full", ScriptState.RUNNING, 6),
            ("complete-full", ScriptState.COMPLETED, 7),
            ("failed00-full", ScriptState.FAILED, 8),
        )
    ]

    result = present_script_list(scripts)

    assert result.confirmation == (
        "📜 Scripts:\n\n"
        "  📝 draft000 | 12345678901234567890 | python | "
        "draft          | risk:  0\n"
        "  ✍️ signed00 | 12345678901234567890 | python | "
        "signed         | risk:  1\n"
        "  ⏳ pending0 | 12345678901234567890 | python | "
        "pending_review | risk:  2\n"
        "  ✅ approved | 12345678901234567890 | python | "
        "approved       | risk:  3\n"
        "  ⛔ rejected | 12345678901234567890 | python | "
        "rejected       | risk:  4\n"
        "  📋 queued00 | 12345678901234567890 | python | "
        "queued         | risk:  5\n"
        "  ⚡ running0 | 12345678901234567890 | python | "
        "running        | risk:  6\n"
        "  ✅ complete | 12345678901234567890 | python | "
        "completed      | risk:  7\n"
        "  ❌ failed00 | 12345678901234567890 | python | "
        "failed         | risk:  8"
    )
    assert result.data == {
        "count": 9,
        "scripts": [
            {
                "id": script.id,
                "name": long_name,
                "language": "python",
                "state": script.state.value,
                "risk_score": index,
            }
            for index, script in enumerate(scripts)
        ],
    }


def test_present_script_detail_long_content_and_finding_limit_golden() -> None:
    severities = ("critical", "high", "medium", "low", "info", "critical")
    findings = [
        SecurityFinding(
            severity=severity,
            category="test",
            description=f"finding-{index}",
            pattern_matched=str(index) * 60,
            recommendation="fix it",
            line_number=index,
        )
        for index, severity in enumerate(severities, 1)
    ]
    script = ComputeScript(
        id="script-detail-identifier",
        name="a script name that remains deliberately very long",
        language="bash",
        content="\n".join(f"line-{line_number:02}" for line_number in range(1, 23)),
        purpose="show every bounded detail",
        state=ScriptState.REJECTED,
        signed_by="s" * 40,
        security_findings=findings,
        risk_score=99,
        review_notes="manual review required",
        requirements=["alpha", "beta>=2"],
        created_at=datetime(2026, 1, 2, 3, 4, 5),
        updated_at=datetime(2026, 1, 2, 3, 4, 5),
    )

    result = present_script_detail(script)

    assert result.confirmation == (
        "📜 Script: a script name that remains deliberately very long\n"
        "   ID: script-detail-identifier\n"
        "   Language: bash\n"
        "   State: rejected\n"
        "   Purpose: show every bounded detail\n"
        "   Risk Score: 99/100\n"
        "   Created: 2026-01-02 03:04:05\n"
        "   Signed by: ssssssssssssssssssssssssssssssss...\n"
        "   Requirements: alpha, beta>=2\n"
        "   Review Notes: manual review required\n"
        "\n🔒 Security Findings:\n"
        "   🔴 [CRITICAL] finding-1\n"
        f"      Line 1: {'1' * 50}\n"
        "   🟠 [HIGH] finding-2\n"
        f"      Line 2: {'2' * 50}\n"
        "   🟡 [MEDIUM] finding-3\n"
        f"      Line 3: {'3' * 50}\n"
        "   🟢 [LOW] finding-4\n"
        f"      Line 4: {'4' * 50}\n"
        "   ℹ️ [INFO] finding-5\n"
        f"      Line 5: {'5' * 50}\n"
        "\n📝 Content:\n"
        "     1| line-01\n"
        "     2| line-02\n"
        "     3| line-03\n"
        "     4| line-04\n"
        "     5| line-05\n"
        "     6| line-06\n"
        "     7| line-07\n"
        "     8| line-08\n"
        "     9| line-09\n"
        "    10| line-10\n"
        "    11| line-11\n"
        "    12| line-12\n"
        "    13| line-13\n"
        "    14| line-14\n"
        "    15| line-15\n"
        "    16| line-16\n"
        "    17| line-17\n"
        "    18| line-18\n"
        "    19| line-19\n"
        "    20| line-20\n"
        "   ... (2 more lines)"
    )
    assert result.data == {
        "script_id": "script-detail-identifier",
        "name": "a script name that remains deliberately very long",
        "language": "bash",
        "state": "rejected",
        "risk_score": 99,
        "purpose": "show every bounded detail",
        "requirements": ["alpha", "beta>=2"],
        "review_notes": "manual review required",
        "findings_count": 6,
    }
    assert "finding-6" not in result.confirmation
    assert "line-21" not in result.confirmation


def test_present_execution_history_empty_golden() -> None:
    result = present_execution_history([])

    assert result.to_dict() == {
        "status": "ok",
        "confirmation": "No executions found.",
        "data": {"executions": [], "count": 0},
    }


def test_present_execution_history_success_failure_and_zero_duration_golden() -> None:
    started_at = datetime(2026, 1, 2, 3, 4, 5)
    executions = [
        ExecutionRecord(
            id="success0-full",
            script_id="script01-full",
            started_at=started_at,
            completed_at=started_at + timedelta(seconds=1.25),
            exit_code=0,
            executor="uv",
        ),
        ExecutionRecord(
            id="failure0-full",
            script_id="script02-full",
            started_at=started_at,
            completed_at=started_at + timedelta(seconds=2),
            exit_code=7,
            executor="docker",
        ),
        ExecutionRecord(
            id="zero0000-full",
            script_id="script03-full",
            started_at=started_at,
            completed_at=started_at,
            exit_code=0,
            executor="local",
        ),
    ]

    result = present_execution_history(executions)

    assert result.confirmation == (
        "📊 Execution History:\n\n"
        "  ✅ success0 | script:script01 | exit:0 | 1.25s | uv\n"
        "  ❌ failure0 | script:script02 | exit:7 | 2.00s | docker\n"
        "  ✅ zero0000 | script:script03 | exit:0 | N/A | local"
    )
    assert result.data == {
        "count": 3,
        "executions": [
            {
                "id": "success0-full",
                "script_id": "script01-full",
                "exit_code": 0,
                "duration_seconds": 1.25,
                "executor": "uv",
                "succeeded": True,
            },
            {
                "id": "failure0-full",
                "script_id": "script02-full",
                "exit_code": 7,
                "duration_seconds": 2.0,
                "executor": "docker",
                "succeeded": False,
            },
            {
                "id": "zero0000-full",
                "script_id": "script03-full",
                "exit_code": 0,
                "duration_seconds": 0.0,
                "executor": "local",
                "succeeded": True,
            },
        ],
    }


@pytest.mark.asyncio
async def test_list_scripts_preserves_filtered_and_recent_queries() -> None:
    script = _script(
        script_id="script-list",
        name="listed",
        state=ScriptState.APPROVED,
    )
    store = SimpleNamespace(
        list_by_state=AsyncMock(return_value=[script]),
        list_recent=AsyncMock(return_value=[]),
    )
    feature = ComputeFeature(SimpleNamespace())
    feature.script_store = store

    filtered = await feature.list_scripts(state=" Approved ", limit=7)
    recent = await feature.list_scripts(limit=3)

    store.list_by_state.assert_awaited_once_with(ScriptState.APPROVED, 7)
    store.list_recent.assert_awaited_once_with(3)
    assert filtered == present_script_list([script])
    assert recent == present_script_list([])


@pytest.mark.asyncio
async def test_show_script_preserves_lookup_and_missing_error() -> None:
    script = _script(
        script_id="script-detail-full",
        name="details",
        state=ScriptState.SIGNED,
    )
    store = SimpleNamespace(
        find_by_id_prefix=AsyncMock(side_effect=[script, None]),
    )
    feature = ComputeFeature(SimpleNamespace())
    feature.script_store = store

    found = await feature.show_script("script-d")
    missing = await feature.show_script("missing")

    assert store.find_by_id_prefix.await_args_list[0].args == ("script-d",)
    assert store.find_by_id_prefix.await_args_list[1].args == ("missing",)
    assert found == present_script_detail(script)
    assert missing.to_dict() == {
        "status": "error",
        "error": "Error: Script not found with ID starting with 'missing'",
        "data": {"script_id": "missing"},
    }


@pytest.mark.asyncio
async def test_execution_history_preserves_query_paths() -> None:
    script = _script(
        script_id="script-history-full",
        name="history",
        state=ScriptState.COMPLETED,
    )
    execution = ExecutionRecord(
        id="execution-history-full",
        script_id=script.id,
        exit_code=0,
    )
    store = SimpleNamespace(
        find_by_id_prefix=AsyncMock(side_effect=[script, None]),
        get_executions_for_script=AsyncMock(return_value=[execution]),
        list_recent_executions=AsyncMock(return_value=[]),
    )
    feature = ComputeFeature(SimpleNamespace())
    feature.script_store = store

    filtered = await feature.execution_history("script-h", limit=4)
    recent = await feature.execution_history(limit=2)
    missing = await feature.execution_history("missing")

    store.get_executions_for_script.assert_awaited_once_with(script.id, 4)
    store.list_recent_executions.assert_awaited_once_with(2)
    assert filtered == present_execution_history([execution])
    assert recent == present_execution_history([])
    assert missing.to_dict() == {
        "status": "error",
        "error": "Error: Script not found with ID starting with 'missing'",
        "data": {"script_id": "missing"},
    }


def test_compute_tool_inventory_and_schemas_are_byte_stable() -> None:
    """Freeze all ten public schemas while their implementations move."""
    schemas = []
    for _, method in inspect.getmembers(ComputeFeature, predicate=inspect.isfunction):
        if not hasattr(method, "_tool_schema"):
            continue
        schema = method._tool_schema
        schemas.append(
            {
                "name": schema["name"],
                "description": schema["description"],
                "category": schema["category"].value,
                "parameters": [asdict(parameter) for parameter in schema["parameters"]],
                "command_prefix": schema.get("command_prefix"),
            }
        )

    assert [schema["name"] for schema in schemas] == [
        "empty_trash",
        "execution_history",
        "get_compute_capabilities",
        "get_compute_policy",
        "list_scripts",
        "list_trash",
        "restore_from_trash",
        "run_script",
        "show_script",
        "write_script",
    ]
    payload = json.dumps(
        schemas,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(payload).hexdigest() == (
        "5f6e3ef51d7f792c48ed9cd99c5c812a61debc5c709c060bfc998e7bb42b8920"
    )
