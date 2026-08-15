"""Cross-file contracts for the canonical-package coverage gate."""

import argparse
from configparser import ConfigParser
from pathlib import Path
import sys
import tomllib

from run_tests import SmartTestRunner, configure_ci_defaults


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dependency_names(requirements: list[str]) -> set[str]:
    return {
        requirement.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0]
        for requirement in requirements
    }


def test_test_dependency_declarations_match_and_lock_pytest_cov():
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    optional = project["project"]["optional-dependencies"]["test"]
    dependency_group = project["dependency-groups"]["test"]
    assert optional == dependency_group
    assert "pytest-cov>=7.1.0" in optional

    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in lock["package"]}
    assert {"pytest-cov", "coverage"} <= packages.keys()
    assert {"coverage", "pytest"} <= _dependency_names(
        [
            dependency["name"]
            for dependency in packages["pytest-cov"]["dependencies"]
        ]
    )


def test_coverage_config_targets_package_branches_and_measured_ratchet():
    config = ConfigParser()
    config.read(PROJECT_ROOT / ".coveragerc")

    assert config.get("run", "source").split() == ["kestrel_sovereign"]
    assert config.getboolean("run", "branch") is True
    assert config.getboolean("run", "parallel") is True
    assert config.getfloat("report", "fail_under") == 73.0


def test_ci_mode_keeps_xdist_but_disables_fail_fast_for_complete_reports():
    args = argparse.Namespace(
        ci=True,
        parallel=None,
        coverage=False,
        no_fail_fast=False,
    )
    configure_ci_defaults(args)

    assert args.parallel == "auto"
    assert args.coverage is True
    assert args.no_fail_fast is True


def test_runner_emits_canonical_diagnostic_coverage_command():
    runner = SmartTestRunner(verbose=False)
    command = runner.build_pytest_command(
        test_dirs=[PROJECT_ROOT / "tests"],
        parallel=-1,
        coverage=True,
        fail_fast=False,
    )

    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "pytest"]
    assert ["-n", "auto"] == command[command.index("-n"):command.index("-n") + 2]
    assert "--maxfail=0" in command
    assert "-x" not in command
    assert "--cov=kestrel_sovereign" in command
    assert "--cov-report=term-missing" in command
    assert "--cov-report=html:coverage_html" in command
    assert "--cov-report=json:coverage.json" in command
    assert "--cov-report=xml:coverage.xml" in command
