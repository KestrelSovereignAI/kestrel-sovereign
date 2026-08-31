"""Route and key boundaries for automatic A2A transport admission."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from kestrel_sovereign.a2a import transport_auth


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/agents"),
        ("POST", "/api/agent/invoke"),
        ("POST", "/api/agent/tasks/send"),
        ("POST", "/api/agent/tasks/task-1/read"),
        ("POST", "/api/agent/tasks/task-1/cancel"),
        ("POST", "/api/agent/tasks/task-1/subscribe"),
        ("POST", "/api/agent/tasks/team/queue/task-1/read"),
        ("POST", "/api/agent/tasks/team/queue/task-1/cancel"),
        ("POST", "/api/agent/tasks/team/queue/task-1/subscribe"),
        ("POST", "/api/agents/recipient/api/agent/invoke"),
        ("POST", "/api/agents/recipient/api/agent/tasks/send"),
        ("POST", "/api/agents/recipient/api/agent/tasks/task-1/read"),
        ("POST", "/api/agents/recipient/api/agent/tasks/task-1/cancel"),
        ("POST", "/api/agents/recipient/api/agent/tasks/task-1/subscribe"),
        (
            "POST",
            "/api/agents/recipient/api/agent/tasks/team/queue/task-1/read",
        ),
    ],
)
def test_transport_allowlist_contains_only_peer_operations(method, path):
    assert transport_auth.is_a2a_transport_path(method, path)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/agent/tasks"),
        ("GET", "/api/agent/tasks/task-1"),
        ("GET", "/api/agent/tasks/task-1/subscribe"),
        ("GET", "/api/agents/recipient/api/agent/tasks"),
        ("GET", "/api/agents/recipient/api/agent/tasks/task-1"),
        ("GET", "/api/agents/recipient/api/agent/tasks/task-1/subscribe"),
        ("POST", "/api/agent/tasks/task-1"),
        ("POST", "/api/agent/tasks/task-1/read/extra"),
        ("DELETE", "/api/agent/tasks/task-1/cancel"),
        ("GET", "/api/agents/recipient"),
    ],
)
def test_transport_allowlist_excludes_operator_and_near_miss_routes(method, path):
    assert not transport_auth.is_a2a_transport_path(method, path)


def test_transport_key_is_generated_once_and_is_not_the_sovereign_key(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(transport_auth.A2A_TRANSPORT_KEY_ENV, raising=False)
    monkeypatch.setenv("KESTREL_API_KEY", "sovereign-key")
    monkeypatch.setattr(
        transport_auth.secrets,
        "token_urlsafe",
        lambda _size: "generated-peer-key",
    )

    first = transport_auth.ensure_a2a_transport_key(project_root=tmp_path)
    second = transport_auth.ensure_a2a_transport_key(project_root=tmp_path)

    assert first == second == "generated-peer-key"
    assert first != os.environ["KESTREL_API_KEY"]


def test_blank_export_is_replaced_and_stable(monkeypatch, tmp_path):
    monkeypatch.setenv(transport_auth.A2A_TRANSPORT_KEY_ENV, "")
    generated = iter(("first-generated-key", "unexpected-second-key"))
    monkeypatch.setattr(
        transport_auth.secrets,
        "token_urlsafe",
        lambda _size: next(generated),
    )

    first = transport_auth.ensure_a2a_transport_key(project_root=tmp_path)
    second = transport_auth.ensure_a2a_transport_key(project_root=tmp_path)

    assert first == second == "first-generated-key"
    assert os.environ[transport_auth.A2A_TRANSPORT_KEY_ENV] == first


def test_non_ascii_transport_key_fails_before_http_serialization(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        transport_auth.A2A_TRANSPORT_KEY_ENV,
        "peer-☃-key",
    )

    with pytest.raises(
        transport_auth.A2ATransportKeyError,
        match="ASCII",
    ):
        transport_auth.ensure_a2a_transport_key(project_root=tmp_path)


def test_non_ascii_transport_key_file_fails_before_http_serialization(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(transport_auth.A2A_TRANSPORT_KEY_ENV, raising=False)
    key_path = tmp_path / transport_auth.A2A_TRANSPORT_KEY_FILE
    key_path.write_text("peer-☃-key\n", encoding="utf-8")
    key_path.chmod(0o600)

    with pytest.raises(
        transport_auth.A2ATransportKeyError,
        match="ASCII",
    ):
        transport_auth.ensure_a2a_transport_key(project_root=tmp_path)


def test_generated_key_survives_independent_launcher_processes(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(transport_auth.A2A_TRANSPORT_KEY_ENV, raising=False)
    generated = iter(("first-launch-key", "second-launch-key"))
    monkeypatch.setattr(
        transport_auth.secrets,
        "token_urlsafe",
        lambda _size: next(generated),
    )

    first_launcher = transport_auth.ensure_a2a_transport_key(
        {},
        project_root=tmp_path,
    )
    monkeypatch.delenv(transport_auth.A2A_TRANSPORT_KEY_ENV)
    second_environment: dict[str, str] = {}
    second_launcher = transport_auth.ensure_a2a_transport_key(
        second_environment,
        project_root=tmp_path,
    )

    assert first_launcher == second_launcher == "first-launch-key"
    assert second_environment[transport_auth.A2A_TRANSPORT_KEY_ENV] == first_launcher
    key_path = tmp_path / transport_auth.A2A_TRANSPORT_KEY_FILE
    assert key_path.read_text(encoding="utf-8").strip() == first_launcher
    if os.name == "posix":
        assert key_path.stat().st_mode & 0o777 == 0o600


def test_concurrent_launchers_publish_one_transport_key(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)
    sequence = iter(("concurrent-key-one", "concurrent-key-two"))

    def generate(_size):
        value = next(sequence)
        barrier.wait(timeout=5)
        return value

    monkeypatch.setattr(transport_auth.secrets, "token_urlsafe", generate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                transport_auth._load_or_create_transport_key,
                (tmp_path, tmp_path),
            )
        )

    assert len(set(results)) == 1
    key_path = tmp_path / transport_auth.A2A_TRANSPORT_KEY_FILE
    assert key_path.read_text(encoding="utf-8").strip() == results[0]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
def test_existing_transport_key_with_group_access_fails_closed(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(transport_auth.A2A_TRANSPORT_KEY_ENV, raising=False)
    key_path = tmp_path / transport_auth.A2A_TRANSPORT_KEY_FILE
    key_path.write_text("overexposed-key\n", encoding="utf-8")
    key_path.chmod(0o640)

    with pytest.raises(
        transport_auth.A2ATransportKeyError,
        match="owned by the current user with mode 0600",
    ):
        transport_auth.ensure_a2a_transport_key(project_root=tmp_path)


def test_project_child_environment_rebinds_launcher_to_same_key(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        transport_auth.A2A_TRANSPORT_KEY_ENV,
        "exported-host-key",
    )
    child_environment = {
        transport_auth.A2A_TRANSPORT_KEY_ENV: "project-child-key",
    }

    selected = transport_auth.ensure_a2a_transport_key(
        child_environment,
        project_root=tmp_path,
    )

    assert selected == "project-child-key"
    assert child_environment[transport_auth.A2A_TRANSPORT_KEY_ENV] == selected
    assert os.environ[transport_auth.A2A_TRANSPORT_KEY_ENV] == selected


def test_project_key_is_used_when_export_is_blank(monkeypatch, tmp_path):
    monkeypatch.setenv(transport_auth.A2A_TRANSPORT_KEY_ENV, "")
    child_environment = {
        transport_auth.A2A_TRANSPORT_KEY_ENV: "project-child-key",
    }

    selected = transport_auth.ensure_a2a_transport_key(
        child_environment,
        project_root=tmp_path,
    )

    assert selected == "project-child-key"
    assert child_environment[transport_auth.A2A_TRANSPORT_KEY_ENV] == selected
    assert os.environ[transport_auth.A2A_TRANSPORT_KEY_ENV] == selected
