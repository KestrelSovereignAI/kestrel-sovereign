"""A test ID too long for a CI log reader refuses the run, serially and under xdist (#3669)."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from tests.shared.node_id_length import (
    MAX_NODE_ID_LENGTH,
    pytest_xdist_node_collection_finished,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: A liveness bound on the child, not a measurement: it collects one file.
SUBPROCESS_TIMEOUT_SECONDS = 120

CHILD_CONFTEST = textwrap.dedent(
    """
    from tests.shared.node_id_length import (  # noqa: F401 - exposed as a pytest hook
        pytest_xdist_node_collection_finished,
        refuse_overlong_node_ids,
    )


    def pytest_collection_modifyitems(config, items):
        refuse_overlong_node_ids(items)
    """
)


def _run_child(tmp_path: Path, payload_length: int, *, workers: int) -> subprocess.CompletedProcess:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(CHILD_CONFTEST, encoding="utf-8")
    (tmp_path / "test_probe.py").write_text(
        textwrap.dedent(
            f"""
            import pytest


            @pytest.mark.parametrize("payload", ["[" * {payload_length}])
            def test_payload(payload):
                pass
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    # The package root only: forwarding sys.path lets a package directory
    # shadow a stdlib module in the child.
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_PLUGINS", None)
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [
            sys.executable, "-m", "pytest", "test_probe.py",
            "-c", str(tmp_path / "pytest.ini"), "--rootdir", str(tmp_path),
            "-p", "xdist", "-p", "no:cacheprovider", "-n", str(workers), "-v",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


# The probe ID is "test_probe.py::test_payload[" + payload + "]".
_ID_OVERHEAD = len("test_probe.py::test_payload[]")


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_an_overlong_id_refuses_the_run_and_says_why(tmp_path, workers):
    completed = _run_child(
        tmp_path, MAX_NODE_ID_LENGTH - _ID_OVERHEAD + 1, workers=workers,
    )
    output = completed.stdout

    assert completed.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert f"longer than {MAX_NODE_ID_LENGTH} characters" in output, output
    assert f"({MAX_NODE_ID_LENGTH + 1} characters)" in output, output
    # The refusal quotes a bounded prefix, never the ID itself.
    assert "[" * (MAX_NODE_ID_LENGTH // 2) not in output, output
    assert "INTERNALERROR" not in output, output


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_SECONDS + 15)
@pytest.mark.parametrize("workers", [0, 2], ids=["serial", "xdist"])
def test_an_id_at_the_limit_runs(tmp_path, workers):
    completed = _run_child(tmp_path, MAX_NODE_ID_LENGTH - _ID_OVERHEAD, workers=workers)

    assert completed.returncode == pytest.ExitCode.OK, completed.stdout
    assert "1 passed" in completed.stdout, completed.stdout


def test_the_controller_refuses_a_worker_collection_over_the_limit():
    pytest_xdist_node_collection_finished(node=None, ids=["t.py::ok"])
    with pytest.raises(pytest.UsageError, match="explicit `id`"):
        pytest_xdist_node_collection_finished(
            node=None, ids=["t.py::ok", "t.py::t[" + "x" * MAX_NODE_ID_LENGTH + "]"],
        )


def test_the_refusal_itself_has_no_line_over_the_limit():
    """Many refused IDs must not join into one line a log reader stops at."""
    refused = [f"t.py::t[{i}-" + "x" * MAX_NODE_ID_LENGTH + "]" for i in range(600)]
    with pytest.raises(pytest.UsageError) as excinfo:
        pytest_xdist_node_collection_finished(node=None, ids=refused)

    lines = str(excinfo.value).splitlines()
    assert max(len(line) for line in lines) < MAX_NODE_ID_LENGTH, lines
    assert "and 580 more" in lines[-1], lines
