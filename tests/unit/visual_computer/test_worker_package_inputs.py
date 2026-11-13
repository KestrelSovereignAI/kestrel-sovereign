"""Worker build pins must follow the actual Core version on this branch."""
from pathlib import Path
import re
import tomllib

import pytest


@pytest.mark.parametrize("suffix", [".in", ".txt"])
def test_worker_build_pin_matches_core_project_version(suffix):
    root = Path(__file__).parents[3]
    version = tomllib.loads((root/"pyproject.toml").read_text())["project"]["version"]
    requirements = root/"docker/visual-browser"/("requirements-two-domain"+suffix)
    pins = re.findall(r"^kestrel-sovereign==([^\s;\\]+)", requirements.read_text(), re.MULTILINE)
    assert pins == [version], "worker build input and generated lock must use this Core project version"
