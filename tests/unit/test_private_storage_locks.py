"""Real descriptor custody; no sentinel expiry or background load processes."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from kestrel_sovereign.private_storage import PrivateStorageError, exclusive_private_file_lock
from kestrel_sovereign.inception_service import _cleanup_owned_inception_database


def test_nonblocking_lock_refuses_same_process_contention_then_rearms(tmp_path):
    path = tmp_path / "custody" / "owner.lock"
    with exclusive_private_file_lock(path, blocking=False):
        with pytest.raises(PrivateStorageError, match="cannot lock"):
            with exclusive_private_file_lock(path, blocking=False):
                pytest.fail("a live exclusive owner was ignored")
    inode = path.stat().st_ino
    with exclusive_private_file_lock(path, blocking=False):
        assert path.stat().st_ino == inode


def test_nonblocking_lock_refuses_another_process(tmp_path):
    path = tmp_path / "custody" / "owner.lock"
    code = """
import sys
from pathlib import Path
from kestrel_sovereign.private_storage import PrivateStorageError, exclusive_private_file_lock
try:
    with exclusive_private_file_lock(Path(sys.argv[1]), blocking=False):
        raise AssertionError('competing process acquired custody')
except PrivateStorageError:
    print('REFUSED')
"""
    with exclusive_private_file_lock(path, blocking=False):
        result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "REFUSED"


@pytest.mark.parametrize("replacement", ["regular", "symlink", "hardlink"])
def test_database_cleanup_refuses_a_replacement_or_shared_inode(tmp_path, replacement):
    path = tmp_path / "kestrel_prime.db"
    path.write_bytes(b"owned attempt")
    owned = path.stat()
    owner = (owned.st_dev, owned.st_ino)
    witness = tmp_path / "winner"
    if replacement == "hardlink":
        os.link(path, witness)
    else:
        path.rename(tmp_path / "old-owned")
        witness.write_bytes(b"committed winner")
        if replacement == "regular":
            path.write_bytes(b"committed winner")
        else:
            path.symlink_to(witness)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="exclusive creation ownership"):
        _cleanup_owned_inception_database(str(path), owner)
    assert path.read_bytes() == before
    assert witness.read_bytes() == before


def test_database_cleanup_removes_only_the_exclusively_created_inode(tmp_path):
    path = tmp_path / "kestrel_prime.db"
    path.write_bytes(b"failed public birth record")
    st = path.stat()
    _cleanup_owned_inception_database(str(path), (st.st_dev, st.st_ino))
    assert not path.exists()
