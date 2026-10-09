"""Wheels and editable checkouts a real installer can use with no index (#3502).

``write_wheel`` builds a pure-Python wheel by hand: a zip with the
``METADATA``, ``WHEEL`` and ``RECORD`` an installer reads, and nothing to
build. ``write_checkout`` makes a project directory whose build backend is in
the tree and requires nothing, so ``uv pip install -e <checkout>`` builds it
offline. The backend imports this module, which is copied into the checkout
for that reason.
"""

from __future__ import annotations

import base64
import hashlib
import zipfile
from pathlib import Path

_BACKEND = '''\
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fixture_wheels import write_wheel  # noqa: E402

NAME, VERSION = {name!r}, {version!r}


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    files = {{f"{{NAME}}/__init__.py": ""}}
    return write_wheel(wheel_directory, NAME, VERSION, files=files).name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    here = os.path.dirname(os.path.abspath(__file__))
    files = {{f"_{{NAME}}_editable.pth": here + "\\n"}}
    return write_wheel(wheel_directory, NAME, VERSION, files=files).name
'''


def _record_line(path: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{path},sha256={digest.decode()},{len(data)}"


def write_wheel(directory, name, version, requires=(), files=None) -> Path:
    """``<name>-<version>-py3-none-any.whl`` in *directory*, requiring *requires*."""
    files = dict(files or {})
    stem = f"{name.replace('-', '_')}-{version}"
    dist_info = f"{stem}.dist-info"
    files[f"{dist_info}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)
    )
    files[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: kestrel-tests\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    path = Path(directory) / f"{stem}-py3-none-any.whl"
    record = []
    with zipfile.ZipFile(path, "w") as archive:
        for name_in_archive, text in files.items():
            data = text.encode("utf-8")
            archive.writestr(name_in_archive, data)
            record.append(_record_line(name_in_archive, data))
        record.append(f"{dist_info}/RECORD,,")
        archive.writestr(f"{dist_info}/RECORD", "\n".join(record) + "\n")
    return path


def write_checkout(directory, name, version) -> Path:
    """A project at *directory* building *name* *version*, with no build requirements."""
    checkout = Path(directory)
    checkout.mkdir(parents=True, exist_ok=True)
    (checkout / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "_backend"\n'
        'backend-path = ["."]\n\n'
        f'[project]\nname = "{name}"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (checkout / "_backend.py").write_text(
        _BACKEND.format(name=name, version=version), encoding="utf-8",
    )
    (checkout / "_fixture_wheels.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8",
    )
    return checkout
