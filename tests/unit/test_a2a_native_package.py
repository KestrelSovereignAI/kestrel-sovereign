"""The native package must import without loading retired private wire models."""

import os
import subprocess
import sys
from pathlib import Path


def test_package_import_does_not_load_private_sdk_protocol():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kestrel_sovereign.a2a; import kestrel_sovereign.a2a.task_manager; "
            "assert not any(name == 'kestrel_sdk.a2a' or name.startswith('kestrel_sdk.a2a.') "
            "for name in sys.modules), 'native package loaded retired private A2A models'",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
