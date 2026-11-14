"""Availability contract only; no vendor, browser, worker or native ACK doubles."""
import importlib
import sys

import pytest

from tests.shared import installed_worker_requirement as requirement


def outcome(operation):
    # Capture genuine pytest outcomes so a mistaken skip fails the required test.
    with pytest.raises(BaseException) as captured:
        operation()
    return captured.value


@pytest.mark.parametrize("required", [False, True])
def test_missing_optional_module_skips_source_but_fails_required(monkeypatch, required):
    monkeypatch.setenv(requirement.REQUIRE_ENV, "1" if required else "0")
    missing = ModuleNotFoundError("original missing package", name="absent_worker_component")
    original = importlib.import_module
    def import_component(name):
        if name == "absent_worker_component.transport":
            raise missing
        return original(name)
    monkeypatch.setattr(importlib, "import_module", import_component)
    error = outcome(lambda: requirement.worker_module("absent_worker_component.transport"))
    assert type(error) is (pytest.fail.Exception if required else pytest.skip.Exception)
    assert "absent_worker_component.transport is not installed" in str(error)


@pytest.mark.parametrize("required", [False, True])
def test_missing_real_resource_skips_source_but_fails_required(monkeypatch, tmp_path, required):
    monkeypatch.setenv(requirement.REQUIRE_ENV, "1" if required else "0")
    monkeypatch.setattr(requirement.resources, "files", lambda package: tmp_path)
    error = outcome(lambda: requirement.worker_program("controller.py"))
    assert type(error) is (pytest.fail.Exception if required else pytest.skip.Exception)
    assert "installed Core worker resource _worker_programs/controller.py is absent" in str(error)


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("error", [ModuleNotFoundError("broken transitive", name="transitive_dependency"), ImportError("broken installed module")])
def test_broken_import_is_the_same_failure_not_an_optional_skip(monkeypatch, required, error):
    monkeypatch.setenv(requirement.REQUIRE_ENV, "1" if required else "0")
    def broken(name):
        raise error
    monkeypatch.setattr(importlib, "import_module", broken)
    assert outcome(lambda: requirement.worker_module("original_worker.transport")) is error


@pytest.mark.parametrize("required", [False, True])
def test_missing_transport_callable_obeys_required_contract(monkeypatch, required):
    monkeypatch.setenv(requirement.REQUIRE_ENV, "1" if required else "0")
    error = outcome(lambda: requirement.worker_module("sys", required_callables=("absent_worker_transport",)))
    assert type(error) is (pytest.fail.Exception if required else pytest.skip.Exception)
    assert "sys.absent_worker_transport is unavailable" in str(error)


def test_present_module_and_real_file_are_used_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv(requirement.REQUIRE_ENV, "1")
    assert requirement.worker_module("sys", required_callables=("exit",)) is sys
    path = tmp_path/"_worker_programs/controller.py"
    path.parent.mkdir()
    path.write_text("# inert availability DATA, never imported\n")
    monkeypatch.setattr(requirement.resources, "files", lambda package: tmp_path)
    assert requirement.worker_program("controller.py") == path


@pytest.mark.parametrize("value", ["true", "yes", "2"])
def test_invalid_required_flag_refuses_even_when_module_exists(monkeypatch, value):
    monkeypatch.setenv(requirement.REQUIRE_ENV, value)
    error = outcome(lambda: requirement.worker_module("sys"))
    assert type(error) is pytest.UsageError
    assert requirement.REQUIRE_ENV in str(error)
