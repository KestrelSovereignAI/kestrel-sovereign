"""PR docs checks must remain separate from trusted Pages publication."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def workflow():
    return yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github/workflows/docs-site.yml").read_text()
    )


def test_docs_pr_checks_are_read_only_and_deploy_is_main_only(workflow):
    events = workflow[True]  # YAML 1.1 parses `on` as a boolean.
    paths = events["pull_request"]["paths"]
    assert events["pull_request"]["branches"] == ["main"]
    assert set(paths) == set(events["push"]["paths"])
    for changed in (
        "docs/**", "kestrel_sovereign/data/constitutions/v2/**",
        "scripts/build_docs_site.py", "scripts/docs_okf.py", "scripts/docs_verify.py",
        "tests/docs_site/**", "tests/unit/test_build_docs_site.py",
        "tests/unit/test_docs_site_workflow.py", ".github/workflows/docs-site.yml",
    ):
        assert changed in paths
    assert workflow["permissions"] == {"contents": "read"}
    build = workflow["jobs"]["build"]
    assert build.get("permissions", workflow["permissions"]) == {"contents": "read"}
    assert "concurrency" not in build
    commands = "\n".join(step.get("run", "") for step in build["steps"])
    assert "scripts/build_docs_site.py build" in commands
    assert "npm run build" in commands
    assert "tests/docs_site/test_constitution_handbook.py" in commands
    assert "tests/unit/test_build_docs_site.py" in commands
    assert "tests/unit/test_docs_site_workflow.py" in commands
    for step in build["steps"]:
        if step.get("uses", "").startswith("actions/upload-pages-artifact@"):
            assert step["if"] == "github.event_name != 'pull_request' && github.ref == 'refs/heads/main'"
    deploy = workflow["jobs"]["deploy"]
    assert deploy["if"] == "github.event_name != 'pull_request' && github.ref == 'refs/heads/main'"
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}


def test_docs_publication_lock_covers_build_and_deploy_with_separate_pr_groups(workflow):
    assert "concurrency" in workflow, (
        "The Pages lock must cover the whole workflow so an older, slower build "
        "cannot deploy after a newer publication"
    )
    assert workflow["concurrency"] == {
        "group": (
            "${{ github.event_name != 'pull_request' && github.ref == 'refs/heads/main' "
            "&& 'pages' || format('docs-checks-{0}-{1}', github.event_name, github.ref) }}"
        ),
        "cancel-in-progress": False,
    }, "Only trusted main publications may share the Pages group; PR checks use their own ref"
    for job in workflow["jobs"].values():
        assert "concurrency" not in job, "Publication is serialized across build and deployment"
