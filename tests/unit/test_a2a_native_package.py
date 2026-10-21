"""The native package must import without loading retired private wire models."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Import liveness only; correctness is checked by exit status and loaded modules.
NATIVE_IMPORT_TIMEOUT_SECONDS = 90


@pytest.mark.parametrize(
    "first_import",
    [
        "import kestrel_sovereign.a2a.task_manager",
        "import kestrel_sovereign.a2a.native_host",
        "import kestrel_sovereign.a2a.native_sender; import kestrel_sovereign.a2a.stores.unified.task_store",
        "import kestrel_sovereign.a2a.stores.unified.task_store; import kestrel_sovereign.a2a.native_sender",
    ],
)
@pytest.mark.timeout(NATIVE_IMPORT_TIMEOUT_SECONDS + 15)
def test_package_import_does_not_load_private_sdk_protocol(first_import):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            first_import + "; import sys; import kestrel_sovereign.a2a; "
            "assert not any(name == 'kestrel_sdk.a2a' or name.startswith('kestrel_sdk.a2a.') "
            "for name in sys.modules), 'native package loaded retired private A2A models'",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=NATIVE_IMPORT_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stdout + result.stderr
