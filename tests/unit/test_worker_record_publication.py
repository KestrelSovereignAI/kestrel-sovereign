"""Real temporary-file publication only; no worker role or native startup."""
import os
from pathlib import Path
import stat

import pytest

from kestrel_sovereign.visual_computer import worker_publication as publication


@pytest.mark.parametrize("mode", [0o600, 0o640])
def test_record_is_complete_and_has_exact_mode_at_first_visibility(tmp_path, monkeypatch, mode):
    destination = tmp_path/"record"
    payload = b"complete original startup record"
    original_link = os.link
    observed = []

    def publish(source, target, **kwargs):
        assert not destination.exists()
        assert Path(source).read_bytes() == payload
        assert stat.S_IMODE(Path(source).stat().st_mode) == mode
        original_link(source, target, **kwargs)
        observed.append(destination.read_bytes())

    monkeypatch.setattr(publication.os, "link", publish)
    previous = os.umask(0o777)
    try:
        publication.publish_record(destination, payload, mode=mode)
    finally:
        os.umask(previous)
    assert observed == [payload]
    assert stat.S_IMODE(destination.stat().st_mode) == mode
    assert list(tmp_path.iterdir()) == [destination]


def test_partial_original_write_never_exposes_final_name(tmp_path, monkeypatch):
    destination = tmp_path/"record"
    original_open = open

    class StalledWrite:
        def __init__(self, *args, **kwargs):
            self.stream = original_open(*args, **kwargs)
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def fileno(self):
            return self.stream.fileno()
        def write(self, payload):
            self.stream.write(payload[:3])
            self.stream.flush()
            assert not destination.exists()
            return 3+self.stream.write(payload[3:])

    monkeypatch.setattr(publication, "open", StalledWrite, raising=False)
    publication.publish_record(destination, b"complete original bytes")
    assert destination.read_bytes() == b"complete original bytes"


@pytest.mark.parametrize("collision", ["file", "symlink", "stage"])
def test_existing_record_or_stage_is_never_replaced_or_removed(tmp_path, collision):
    destination = tmp_path/"record"
    stage = tmp_path/"record.pending"
    if collision == "file":
        destination.write_bytes(b"original")
    elif collision == "symlink":
        destination.symlink_to(tmp_path/"absent-original")
    else:
        stage.write_bytes(b"other publisher")
    with pytest.raises(FileExistsError):
        publication.publish_record(destination, b"replacement")
    if collision == "stage":
        assert stage.read_bytes() == b"other publisher"
        assert not destination.exists()
    else:
        assert not stage.exists()
        if collision == "file":
            assert destination.read_bytes() == b"original"
        else:
            assert destination.is_symlink() and not destination.exists()


@pytest.mark.parametrize("operation", ["fchmod", "link"])
def test_failed_stage_or_publication_exposes_no_record_and_cleans_own_stage(tmp_path, monkeypatch, operation):
    destination = tmp_path/"record"
    original = OSError("original scripted publication failure")
    def fail(*_args, **_kwargs):
        raise original
    monkeypatch.setattr(publication.os, operation, fail)
    with pytest.raises(OSError) as caught:
        publication.publish_record(destination, b"complete bytes")
    assert caught.value is original
    assert list(tmp_path.iterdir()) == []


def test_invalid_mode_refuses_before_creating_any_record(tmp_path):
    with pytest.raises(ValueError, match="fixed_worker_record_mode_required"):
        publication.publish_record(tmp_path/"record", b"bytes", mode=0o666)
    assert list(tmp_path.iterdir()) == []


def test_final_name_cannot_also_be_its_private_stage(tmp_path):
    with pytest.raises(ValueError, match="distinct_worker_record_stage_required"):
        publication.publish_record(tmp_path/"record.pending", b"bytes")
    assert list(tmp_path.iterdir()) == []
