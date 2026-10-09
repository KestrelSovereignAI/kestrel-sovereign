"""Core's ``uv.lock`` read as the declaration of what the venv holds (#3502).

A host ran ``anthropic`` 1.11.0 while ``uv.lock`` pinned 0.117.0: CI tested one
version and prod ran another. These pin the two halves of the fix that live in
``core_lock``: reading the lock for THIS environment (so the pins a feature
install carries are the versions CI resolved here), and comparing a venv
against it (so a mismatch is reported). The install-path half is in
``test_core_lock_install.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kestrel_sovereign import core_lock as cl
from kestrel_sovereign.feature_reconcile import Provenance

REPO_LOCK = Path(__file__).resolve().parents[2] / "uv.lock"

LOCK = """\
version = 1
revision = 3
requires-python = ">=3.11"

[[package]]
name = "anthropic"
version = "0.117.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "Kestrel_Sovereign"
version = "0.53.24"
source = { editable = "." }

[[package]]
name = "workspace-root"
version = "0.0.0"
source = { virtual = "." }

[[package]]
name = "sqlean-py"
version = "3.49.1"
source = { registry = "https://pypi.org/simple" }
resolution-markers = [
    "sys_platform == 'win32'",
]

[[package]]
name = "sqlean-py"
version = "3.50.4.5"
source = { registry = "https://pypi.org/simple" }
resolution-markers = [
    "sys_platform != 'win32'",
]
"""

POSIX = {"sys_platform": "linux"}
WINDOWS = {"sys_platform": "win32"}


def _lock(tmp_path, text=LOCK) -> Path:
    path = tmp_path / "uv.lock"
    path.write_text(text, encoding="utf-8")
    return path


def _site_packages(tmp_path, installed: dict) -> Path:
    """A fixture venv: one ``<name>-<version>.dist-info`` per installed package."""
    site = tmp_path / "site-packages"
    site.mkdir()
    for name, version in installed.items():
        dist_info = site / f"{name.replace('-', '_')}-{version}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
    return site


def _check(lock_path, site, environment=POSIX):
    return cl.check_venv_against_lock(
        lambda: cl.read_core_lock(lock_path, environment=environment),
        installed=lambda names: cl.installed_versions(names, [str(site)]),
    )


# --- reading the lock -------------------------------------------------------


def test_a_lock_pins_each_package_at_the_version_this_environment_resolves(tmp_path):
    lock = cl.read_core_lock(_lock(tmp_path), environment=POSIX)

    assert lock.versions() == {"anthropic": "0.117.0", "sqlean-py": "3.50.4.5"}
    assert lock.undetermined == ()


def test_a_forked_package_resolves_to_the_fork_whose_markers_apply(tmp_path):
    """uv locks one version per fork; the pin is the one THIS host installs."""
    lock = cl.read_core_lock(_lock(tmp_path), environment=WINDOWS)

    assert lock.versions()["sqlean-py"] == "3.49.1"


def test_core_and_virtual_packages_are_not_pinned(tmp_path):
    """Core has its own source policy, and a virtual package is never installed.

    Core is matched by its canonical name, so the lock's underscore spelling
    is still core.
    """
    names = set(cl.read_core_lock(_lock(tmp_path), environment=POSIX).versions())

    assert "kestrel-sovereign" not in names
    assert "workspace-root" not in names


def test_two_applicable_versions_are_undetermined_rather_than_guessed(tmp_path):
    text = LOCK + (
        '\n[[package]]\nname = "anthropic"\nversion = "1.11.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    lock = cl.read_core_lock(_lock(tmp_path, text), environment=POSIX)

    assert "anthropic" not in lock.versions()
    assert lock.undetermined == ("anthropic",)


def test_constraint_lines_pin_every_locked_package_but_the_excluded(tmp_path):
    lock = cl.read_core_lock(_lock(tmp_path), environment=POSIX)

    assert lock.constraint_lines() == ["anthropic==0.117.0", "sqlean-py==3.50.4.5"]
    assert lock.constraint_lines({"Anthropic"}) == ["sqlean-py==3.50.4.5"]


def test_constraint_lines_refuse_while_a_locked_package_is_undetermined(tmp_path):
    """Lines holding every package but one would let an install move that one.

    The lock names no single version of anthropic here, so there is no line to
    carry for it. Returning the others anyway is how an install ran with
    anthropic free while looking held (#3502). Excluded for a deliberate reason
    (a link a pin would replace), it is no hole in the hold.
    """
    text = LOCK + (
        '\n[[package]]\nname = "anthropic"\nversion = "1.11.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    lock = cl.read_core_lock(_lock(tmp_path, text), environment=POSIX)

    with pytest.raises(cl.CoreLockError, match="no single version .*: anthropic"):
        lock.constraint_lines()
    with pytest.raises(cl.CoreLockError):
        lock.constraint_lines({"sqlean-py"})
    assert lock.constraint_lines({"Anthropic"}) == ["sqlean-py==3.50.4.5"]
    assert lock.names() == ("sqlean-py", "anthropic")


@pytest.mark.parametrize(
    "text",
    ["this is = not [toml", 'version = 1\nrevision = 3\n'],
    ids=["unparseable", "no-packages"],
)
def test_a_lock_that_is_not_a_lock_raises(tmp_path, text):
    with pytest.raises(cl.CoreLockError):
        cl.read_core_lock(_lock(tmp_path, text))


def test_a_lock_that_is_not_utf8_raises_core_lock_error(tmp_path):
    """tomllib decodes the bytes itself; its UnicodeDecodeError must not escape."""
    path = tmp_path / "uv.lock"
    path.write_bytes(b'version = 1\n\xff\xfe = "x"\n')

    with pytest.raises(cl.CoreLockError):
        cl.read_core_lock(path)


def test_a_package_none_of_whose_forks_applies_is_undetermined(tmp_path):
    """uv's forks cover every supported environment, so no applicable fork
    means markers this reader cannot evaluate (a conflict fork on an extra)."""
    text = LOCK + (
        '\n[[package]]\nname = "torch"\nversion = "2.5.1"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        "resolution-markers = [\n    \"extra == 'extra-5-demo-cpu'\",\n]\n"
    )
    lock = cl.read_core_lock(_lock(tmp_path, text), environment=POSIX)

    assert "torch" not in lock.versions()
    assert lock.undetermined == ("torch",)


def test_the_repository_lock_reads_with_one_version_per_package():
    """The reader against the real thing, not only a hand-written fixture.

    anthropic is the package this issue is about: it must be pinned, at a
    single version, on the interpreter running the suite.
    """
    lock = cl.read_core_lock(REPO_LOCK)

    assert lock.undetermined == ()
    assert "anthropic" in lock.versions()
    assert len(lock.packages) > 100


# --- which lock belongs to this core ----------------------------------------


def test_only_a_core_installed_from_a_local_directory_has_a_lock(tmp_path):
    checkout = tmp_path / "kestrel-sovereign"
    checkout.mkdir()

    assert cl.core_lock_checkout(Provenance.direct(str(checkout), editable=True)) == checkout
    assert cl.core_lock_checkout(Provenance.direct(str(checkout))) == checkout
    assert cl.core_lock_checkout(Provenance.from_index_install()) is None
    assert cl.core_lock_checkout(
        Provenance.direct(str(checkout), vcs="git", revision="abc")
    ) is None
    assert cl.core_lock_checkout(
        Provenance.direct(str(checkout), archive_hash="sha256=00")
    ) is None
    assert cl.core_lock_checkout(
        Provenance.direct("https://example.invalid/kestrel-sovereign")
    ) is None


def test_unknown_core_provenance_is_not_read_as_no_lock():
    """Damaged metadata may hide a checkout with a lock beside it (#3502).

    "No lock" would let every install run unheld on exactly the host whose
    install is already broken, so it is an error the guard and the drift check
    both surface, and it names the remedy.
    """
    with pytest.raises(cl.CoreLockError, match="--reinstall-package kestrel-sovereign"):
        cl.core_lock_checkout(Provenance.unknown())


def test_a_checkout_without_a_lock_has_nothing_to_load(tmp_path):
    assert cl.load_core_lock(tmp_path) is None
    assert cl.load_core_lock(tmp_path / "gone") is None  # no directory at all
    not_a_dir = tmp_path / "core.whl"
    not_a_dir.write_bytes(b"")
    assert cl.load_core_lock(not_a_dir) is None  # a file holds no lock
    _lock(tmp_path)
    assert cl.load_core_lock(tmp_path).versions()["anthropic"] == "0.117.0"


def _lock_that_exists_but_is_not_a_file(tmp_path, kind):
    path = tmp_path / "uv.lock"
    if kind == "directory":
        path.mkdir()
    elif kind == "dangling-link":
        path.symlink_to(tmp_path / "nowhere.lock")
    elif kind == "fifo":
        os.mkfifo(path)
    return tmp_path


@pytest.mark.parametrize(
    "kind",
    [
        "directory",
        pytest.param(
            "dangling-link",
            marks=pytest.mark.skipif(os.name == "nt", reason="symlinks need privilege"),
        ),
        pytest.param(
            "fifo",
            marks=pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX only"),
        ),
    ],
)
def test_a_lock_path_that_exists_but_is_not_a_lock_file_raises(tmp_path, kind):
    """Only an absent path is "no lock" (#3502).

    ``is_file()`` read a directory named ``uv.lock`` as no lock at all, so
    installs ran unheld and the drift check said there was nothing to compare.
    A FIFO is checked on the open descriptor, without blocking.
    """
    checkout = _lock_that_exists_but_is_not_a_file(tmp_path, kind)

    with pytest.raises(cl.CoreLockError):
        cl.load_core_lock(checkout)


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions, non-root",
)
def test_a_checkout_whose_lock_status_is_refused_raises(tmp_path):
    checkout = tmp_path / "core"
    checkout.mkdir()
    _lock(checkout)
    checkout.chmod(0o000)
    try:
        with pytest.raises(cl.CoreLockError):
            cl.load_core_lock(checkout)
    finally:
        checkout.chmod(0o700)


def test_a_directory_lock_is_reported_not_called_nothing_to_compare(tmp_path):
    site = _site_packages(tmp_path, {"anthropic": "1.11.0"})
    checkout = tmp_path / "core"
    checkout.mkdir()
    _lock_that_exists_but_is_not_a_file(checkout, "directory")

    check = cl.check_venv_against_lock(
        lambda: cl.load_core_lock(checkout),
        installed=lambda names: cl.installed_versions(names, [str(site)]),
    )

    assert check.needs_attention
    assert check.headline == "venv was not compared against core's uv.lock"


# --- the venv against the lock (the drift check) -----------------------------


def test_an_installed_version_other_than_the_locked_one_is_reported(tmp_path):
    """The gate: a fixture venv holding anthropic 1.11.0 against a 0.117.0 lock."""
    site = _site_packages(
        tmp_path, {"anthropic": "1.11.0", "sqlean-py": "3.50.4.5"},
    )

    check = _check(_lock(tmp_path), site)

    assert check.compared
    assert not check.matches
    assert check.needs_attention
    assert check.drift == (
        cl.LockDrift(name="anthropic", locked="0.117.0", installed="1.11.0"),
    )
    lines = check.report_lines()
    assert "anthropic 1.11.0 installed, uv.lock pins 0.117.0" in lines
    assert any("uv lock" in line for line in lines)
    assert check.headline == "venv differs from core's uv.lock"


def test_a_venv_at_the_locked_versions_matches(tmp_path):
    site = _site_packages(
        tmp_path, {"anthropic": "0.117.0", "sqlean-py": "3.50.4.5"},
    )

    check = _check(_lock(tmp_path), site)

    assert check.matches
    assert not check.needs_attention
    assert check.report_lines() == []
    assert check.headline == "venv matches core's uv.lock (2 locked packages)"


def test_a_locked_package_that_is_not_installed_is_not_drift(tmp_path):
    """The lock covers every extra and group; a host installs the ones it uses."""
    site = _site_packages(tmp_path, {"anthropic": "0.117.0"})

    assert _check(_lock(tmp_path), site).matches


def test_equal_versions_spelled_differently_are_not_drift(tmp_path):
    text = LOCK.replace('version = "0.117.0"', 'version = "0.117"')
    site = _site_packages(tmp_path, {"anthropic": "0.117.0"})

    assert _check(_lock(tmp_path, text), site).matches


def test_an_undetermined_package_is_named_rather_than_called_a_match(tmp_path):
    text = LOCK + (
        '\n[[package]]\nname = "anthropic"\nversion = "1.11.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    site = _site_packages(tmp_path, {"anthropic": "1.11.0"})

    check = _check(_lock(tmp_path, text), site)

    assert not check.matches
    assert check.needs_attention
    assert any("anthropic" in line and "no single version" in line
               for line in check.report_lines())


def test_an_unreadable_lock_is_reported_not_raised(tmp_path):
    site = _site_packages(tmp_path, {"anthropic": "1.11.0"})

    check = _check(_lock(tmp_path, "not = [a lock"), site)

    assert not check.compared
    assert check.needs_attention
    assert check.headline == "venv was not compared against core's uv.lock"
    assert check.report_lines()[0].startswith("core lock could not be read:")


def test_no_lock_is_neither_a_match_nor_a_finding():
    check = cl.check_venv_against_lock(lambda: None, installed=dict)

    assert not check.matches
    assert not check.needs_attention
    assert check.report_lines() == []
    assert "nothing to compare" in check.headline


def test_installed_versions_reads_the_first_copy_on_the_path(tmp_path):
    first = _site_packages(tmp_path, {"Anthropic": "0.117.0"})
    later = tmp_path / "later"
    later.mkdir()
    shadowed = later / "anthropic-1.11.0.dist-info"
    shadowed.mkdir()
    (shadowed / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: anthropic\nVersion: 1.11.0\n",
        encoding="utf-8",
    )
    broken = first / "broken-1.0.dist-info"
    broken.mkdir()  # no METADATA: found, with no version to compare

    found = cl.installed_versions(
        ["anthropic", "broken", "absent"], [str(first), str(later)],
    )

    assert found == {"anthropic": "0.117.0", "broken": None}


def test_a_locked_package_whose_metadata_names_no_version_is_not_a_match(tmp_path):
    """A damaged dist-info still imports whatever is beside it (#3502).

    Skipping it called the venv a match without comparing that package.
    """
    site = _site_packages(tmp_path, {"sqlean-py": "3.50.4.5"})
    (site / "anthropic-1.11.0.dist-info").mkdir()  # METADATA missing

    check = _check(_lock(tmp_path), site)

    assert check.unreadable == ("anthropic",)
    assert check.drift == ()
    assert not check.matches
    assert check.needs_attention
    assert check.headline == "venv was only partly compared against core's uv.lock"
    assert "installed metadata names no version for: anthropic" in check.report_lines()
