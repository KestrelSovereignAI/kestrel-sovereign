"""The ``Typing :: Typed`` classifier must be backed by a PEP 561 marker (#2683).

``pyproject.toml`` advertises ``Typing :: Typed``, but until #2683 the wheel
shipped no ``py.typed``. Type checkers ignore inline annotations of an
installed package that lacks the marker, so the classifier was a claim
consumers could not rely on.

These tests pin the claim at every layer it can drift:

1. the source tree carries a full-package (not ``partial``) marker;
2. the built wheel contains it and its metadata carries the classifier;
3. a clean install of that wheel exposes the marker to the import system,
   with this checkout kept off ``sys.path``.

If the project ever stops supporting inline typing, remove the classifier
and the marker together; these tests fail on any mismatch between them.
"""
from __future__ import annotations

import email
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

import kestrel_sovereign

REPOSITORY = Path(__file__).resolve().parents[2]
TYPED_CLASSIFIER = "Typing :: Typed"
MARKER_ARCNAME = "kestrel_sovereign/py.typed"


def _declared_classifiers() -> list[str]:
    with (REPOSITORY / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["classifiers"]


def _isolated_environment() -> dict[str, str]:
    """Environment in which only the installed wheel can satisfy imports."""
    environment = os.environ.copy()
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    return environment


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    dist_dir = tmp_path_factory.mktemp("typed-marker-dist")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=REPOSITORY,
        env=_isolated_environment(),
        check=True,
    )
    wheels = list(dist_dir.glob("kestrel_sovereign-*.whl"))
    assert len(wheels) == 1, f"expected one Kestrel wheel, found {wheels}"
    return wheels[0]


def test_pyproject_declares_typed_classifier():
    assert TYPED_CLASSIFIER in _declared_classifiers()


def test_source_tree_carries_full_package_marker():
    marker = Path(kestrel_sovereign.__file__).resolve().parent / "py.typed"
    assert marker.is_file(), f"{marker} is missing; the Typed classifier is unbacked"
    # PEP 561: a marker whose content is ``partial`` declares an incomplete
    # stub package. Kestrel ships inline annotations for the whole package.
    assert "partial" not in marker.read_text(encoding="utf-8")


def test_wheel_contents_agree_with_its_metadata(built_wheel):
    with zipfile.ZipFile(built_wheel) as archive:
        names = set(archive.namelist())
        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1, metadata_names
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        classifiers = metadata.get_all("Classifier") or []

        assert TYPED_CLASSIFIER in classifiers
        assert MARKER_ARCNAME in names, (
            f"wheel advertises {TYPED_CLASSIFIER!r} but does not ship {MARKER_ARCNAME}"
        )
        record_names = [n for n in names if n.endswith(".dist-info/RECORD")]
        assert len(record_names) == 1, record_names
        record = archive.read(record_names[0]).decode("utf-8")
        assert any(
            line.split(",", 1)[0] == MARKER_ARCNAME for line in record.splitlines()
        ), f"{MARKER_ARCNAME} is not listed in the wheel RECORD"


def test_clean_wheel_install_exposes_marker(built_wheel, tmp_path):
    environment = _isolated_environment()
    venv_dir = tmp_path / "typed-marker-venv"
    subprocess.run(
        ["uv", "venv", "--no-project", str(venv_dir)],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    python_name = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    python = venv_dir / python_name
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", str(built_wheel)],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    # ``--no-deps`` leaves the package unimportable, so discover the marker the
    # way a type checker does: through the installed distribution and the
    # package's location on the interpreter's search path, not by importing it.
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as md, importlib.util as iu, pathlib, sys; "
                "dist = md.distribution('kestrel_sovereign'); "
                "files = {str(f).replace('\\\\', '/') for f in dist.files}; "
                f"assert {MARKER_ARCNAME!r} in files, sorted(files)[:5]; "
                "assert 'Typing :: Typed' in dist.metadata.get_all('Classifier'); "
                "spec = iu.find_spec('kestrel_sovereign'); "
                "root = pathlib.Path(spec.submodule_search_locations[0]); "
                "marker = root / 'py.typed'; "
                "assert marker.is_file(), marker; "
                "assert str(root).startswith(sys.prefix), (str(root), sys.prefix); "
                "print(marker)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    installed_marker = Path(probe.stdout.strip())
    assert installed_marker.is_file()
    assert REPOSITORY not in installed_marker.resolve().parents
