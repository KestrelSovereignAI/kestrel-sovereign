"""
Tests for the Feature CLI commands (kestrel feature list/install/enable/disable/info/scaffold/skills).
"""

import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kestrel_sovereign.feature_registry import (
    FeaturePackageInfo,
    FeatureStatus,
    SkillInfo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry():
    """Create a minimal test registry."""
    return {
        "identity": FeaturePackageInfo(
            name="identity",
            package="kestrel-sovereign",
            git="https://github.com/example/ks.git",
            features=["IdentityFeature"],
            description="DID-based identity",
            tags=["core", "identity"],
            icon="fingerprint",
            core=True,
            skills=[
                SkillInfo(name="get_identity", description="View identity", category="system", tags=["identity"]),
            ],
        ),
        "cloud": FeaturePackageInfo(
            name="cloud",
            package="kestrel-feature-cloud",
            git="https://github.com/example/cloud.git",
            features=["RunPodFeature", "VastAIFeature"],
            description="GPU cloud orchestration",
            tags=["infrastructure", "gpu"],
            icon="cloud",
            core=False,
            skills=[
                SkillInfo(name="list_instances", description="List GPU instances", category="compute", tags=["gpu"]),
                SkillInfo(name="launch_instance", description="Launch GPU instance", category="compute", tags=["gpu"]),
            ],
        ),
        "voice": FeaturePackageInfo(
            name="voice",
            package="kestrel-feature-voice",
            git="https://github.com/example/voice.git",
            features=["VoiceFeature"],
            description="TTS and STT",
            tags=["voice", "tts"],
            icon="microphone",
            core=True,
            skills=[
                SkillInfo(name="speak", description="Text to speech", category="communication", tags=["voice", "tts"]),
            ],
        ),
    }


def _make_args(**kwargs):
    """Create a mock args namespace."""
    args = MagicMock()
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Tests: cmd_feature_list
# ---------------------------------------------------------------------------

class TestFeatureList:

    @patch("kestrel_sovereign.feature_registry.get_registry")
    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=[])
    def test_list_shows_installed_features(self, mock_disabled, mock_get_reg, capsys):
        from kestrel_sovereign.cli import cmd_feature_list

        reg = _make_registry()
        reg["identity"].status = FeatureStatus.INSTALLED
        reg["voice"].status = FeatureStatus.INSTALLED
        reg["cloud"].status = FeatureStatus.AVAILABLE
        mock_get_reg.return_value = reg

        args = _make_args()
        result = cmd_feature_list(args)
        assert result == 0

        output = capsys.readouterr().out
        assert "identity" in output
        assert "voice" in output
        assert "cloud" in output
        assert "INSTALLED" in output or "installed" in output
        assert "AVAILABLE" in output or "available" in output

    @patch("kestrel_sovereign.feature_registry.get_registry")
    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=["VoiceFeature"])
    def test_list_shows_disabled_from_toml(self, mock_disabled, mock_get_reg, capsys):
        from kestrel_sovereign.cli import cmd_feature_list

        reg = _make_registry()
        reg["identity"].status = FeatureStatus.INSTALLED
        reg["voice"].status = FeatureStatus.INSTALLED
        reg["cloud"].status = FeatureStatus.AVAILABLE
        mock_get_reg.return_value = reg

        args = _make_args()
        result = cmd_feature_list(args)
        assert result == 0

        output = capsys.readouterr().out
        assert "disabled" in output


# ---------------------------------------------------------------------------
# Tests: cmd_feature_install
# ---------------------------------------------------------------------------

class TestFeatureInstall:

    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_install_core_feature_noop(self, mock_load, capsys):
        from kestrel_sovereign.cli import cmd_feature_install

        mock_load.return_value = _make_registry()
        args = _make_args(name="identity")
        result = cmd_feature_install(args)
        assert result == 0
        assert "core feature" in capsys.readouterr().out

    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_install_unknown_feature(self, mock_load, capsys):
        from kestrel_sovereign.cli import cmd_feature_install

        mock_load.return_value = _make_registry()
        args = _make_args(name="nonexistent")
        result = cmd_feature_install(args)
        assert result == 1
        assert "Unknown" in capsys.readouterr().out

    @patch("subprocess.run")
    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_install_runs_pip(self, mock_load, mock_run, capsys):
        from kestrel_sovereign.cli import cmd_feature_install

        reg = _make_registry()
        reg["cloud"].core = False
        mock_load.return_value = reg
        mock_run.return_value = MagicMock(returncode=0)

        args = _make_args(name="cloud")
        result = cmd_feature_install(args)
        assert result == 0

        # Verify pip was called with the right package
        call_args = mock_run.call_args[0][0]
        assert "pip" in call_args
        assert "kestrel-feature-cloud" in call_args


# ---------------------------------------------------------------------------
# Tests: cmd_feature_enable / cmd_feature_disable
# ---------------------------------------------------------------------------

class TestFeatureEnableDisable:

    @patch("kestrel_sovereign.cli._set_toml_disabled_features")
    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=["VoiceFeature"])
    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_enable_removes_from_disabled(self, mock_load, mock_get, mock_set, capsys):
        from kestrel_sovereign.cli import cmd_feature_enable

        mock_load.return_value = _make_registry()
        args = _make_args(name="voice")
        result = cmd_feature_enable(args)
        assert result == 0

        # Should save without VoiceFeature
        saved = mock_set.call_args[0][1]
        assert "VoiceFeature" not in saved
        assert "Enabled" in capsys.readouterr().out

    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=[])
    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_enable_already_enabled(self, mock_load, mock_get, capsys):
        from kestrel_sovereign.cli import cmd_feature_enable

        mock_load.return_value = _make_registry()
        args = _make_args(name="voice")
        result = cmd_feature_enable(args)
        assert result == 0
        assert "not disabled" in capsys.readouterr().out

    @patch("kestrel_sovereign.cli._set_toml_disabled_features")
    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=[])
    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_disable_adds_to_disabled(self, mock_load, mock_get, mock_set, capsys):
        from kestrel_sovereign.cli import cmd_feature_disable

        mock_load.return_value = _make_registry()
        args = _make_args(name="cloud")
        result = cmd_feature_disable(args)
        assert result == 0

        saved = mock_set.call_args[0][1]
        assert "RunPodFeature" in saved
        assert "VastAIFeature" in saved
        assert "Disabled" in capsys.readouterr().out

    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=["RunPodFeature", "VastAIFeature"])
    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_disable_already_disabled(self, mock_load, mock_get, capsys):
        from kestrel_sovereign.cli import cmd_feature_disable

        mock_load.return_value = _make_registry()
        args = _make_args(name="cloud")
        result = cmd_feature_disable(args)
        assert result == 0
        assert "already disabled" in capsys.readouterr().out

    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_disable_unknown(self, mock_load, capsys):
        from kestrel_sovereign.cli import cmd_feature_disable

        mock_load.return_value = _make_registry()
        args = _make_args(name="nonexistent")
        result = cmd_feature_disable(args)
        assert result == 1


# ---------------------------------------------------------------------------
# Tests: cmd_feature_info
# ---------------------------------------------------------------------------

class TestFeatureInfo:

    @patch("kestrel_sovereign.feature_registry.get_registry")
    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=[])
    def test_info_shows_metadata(self, mock_disabled, mock_get_reg, capsys):
        from kestrel_sovereign.cli import cmd_feature_info

        reg = _make_registry()
        reg["cloud"].status = FeatureStatus.AVAILABLE
        mock_get_reg.return_value = reg

        args = _make_args(name="cloud")
        result = cmd_feature_info(args)
        assert result == 0

        output = capsys.readouterr().out
        assert "cloud" in output
        assert "GPU cloud" in output
        assert "kestrel-feature-cloud" in output
        assert "RunPodFeature" in output
        assert "VastAIFeature" in output
        assert "list_instances" in output

    @patch("kestrel_sovereign.feature_registry.get_registry")
    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=[])
    def test_info_unknown(self, mock_disabled, mock_get_reg, capsys):
        from kestrel_sovereign.cli import cmd_feature_info

        mock_get_reg.return_value = _make_registry()
        args = _make_args(name="nonexistent")
        result = cmd_feature_info(args)
        assert result == 1

    @patch("kestrel_sovereign.feature_registry.get_registry")
    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=[])
    def test_info_by_class_name(self, mock_disabled, mock_get_reg, capsys):
        """Can look up a feature by its class name."""
        from kestrel_sovereign.cli import cmd_feature_info

        reg = _make_registry()
        reg["cloud"].status = FeatureStatus.INSTALLED
        mock_get_reg.return_value = reg

        args = _make_args(name="RunPodFeature")
        result = cmd_feature_info(args)
        assert result == 0
        assert "cloud" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tests: cmd_feature_scaffold
# ---------------------------------------------------------------------------

class TestFeatureScaffold:

    def test_scaffold_creates_structure(self, tmp_path, capsys, monkeypatch):
        from kestrel_sovereign.cli import cmd_feature_scaffold

        monkeypatch.chdir(tmp_path)
        args = _make_args(name="myfeature")
        result = cmd_feature_scaffold(args)
        assert result == 0

        scaffold_dir = tmp_path / "kestrel-feature-myfeature"
        src_dir = scaffold_dir / "src" / "kestrel_feature_myfeature"
        assert scaffold_dir.exists()
        assert (src_dir / "__init__.py").exists()
        assert (src_dir / "feature.py").exists()
        assert (scaffold_dir / "pyproject.toml").exists()
        assert (scaffold_dir / "tests" / "test_myfeature.py").exists()
        assert (scaffold_dir / "tests" / "conftest.py").exists()
        assert (scaffold_dir / "SKILL.md").exists()
        assert (scaffold_dir / "README.md").exists()

        # Verify feature class name and @tool decorator
        feature_py = (src_dir / "feature.py").read_text()
        assert "class MyfeatureFeature(Feature):" in feature_py
        assert "@tool(" in feature_py
        assert "tool_description" in feature_py

        # Verify entry_point in pyproject.toml
        pyproject = (scaffold_dir / "pyproject.toml").read_text()
        assert "kestrel_sovereign.features" in pyproject
        assert "MyfeatureFeature" in pyproject
        assert 'where = ["src"]' in pyproject

    def test_scaffold_tests_use_mock_agent(self, tmp_path, capsys, monkeypatch):
        from kestrel_sovereign.cli import cmd_feature_scaffold

        monkeypatch.chdir(tmp_path)
        args = _make_args(name="myfeature")
        cmd_feature_scaffold(args)

        test_py = (tmp_path / "kestrel-feature-myfeature" / "tests" / "test_myfeature.py").read_text()
        assert "MockAgent" in test_py
        assert "mock_agent" in test_py

    def test_scaffold_skill_md(self, tmp_path, capsys, monkeypatch):
        from kestrel_sovereign.cli import cmd_feature_scaffold

        monkeypatch.chdir(tmp_path)
        args = _make_args(name="myfeature")
        cmd_feature_scaffold(args)

        skill_md = (tmp_path / "kestrel-feature-myfeature" / "SKILL.md").read_text()
        assert "MyfeatureFeature" in skill_md
        assert "## Skills" in skill_md

    def test_scaffold_multi_word_name(self, tmp_path, capsys, monkeypatch):
        from kestrel_sovereign.cli import cmd_feature_scaffold

        monkeypatch.chdir(tmp_path)
        args = _make_args(name="my_cool_thing")
        result = cmd_feature_scaffold(args)
        assert result == 0

        scaffold_dir = tmp_path / "kestrel-feature-my-cool-thing"
        src_dir = scaffold_dir / "src" / "kestrel_feature_my_cool_thing"
        assert src_dir.exists()

        feature_py = (src_dir / "feature.py").read_text()
        assert "class MyCoolThingFeature(Feature):" in feature_py

        pyproject = (scaffold_dir / "pyproject.toml").read_text()
        assert 'name = "kestrel-feature-my-cool-thing"' in pyproject

    def test_scaffold_existing_dir_fails(self, tmp_path, capsys, monkeypatch):
        from kestrel_sovereign.cli import cmd_feature_scaffold

        monkeypatch.chdir(tmp_path)
        (tmp_path / "kestrel-feature-test").mkdir()
        args = _make_args(name="test")
        result = cmd_feature_scaffold(args)
        assert result == 1


# ---------------------------------------------------------------------------
# Tests: cmd_feature_skills
# ---------------------------------------------------------------------------

class TestFeatureSkills:

    @patch("kestrel_sovereign.feature_registry.load_registry")
    @patch("kestrel_sovereign.feature_registry.get_skills_for_package")
    def test_skills_lists_package_skills(self, mock_skills, mock_load, capsys):
        from kestrel_sovereign.cli import cmd_feature_skills

        mock_load.return_value = _make_registry()
        mock_skills.return_value = [
            SkillInfo(name="list_instances", description="List GPU instances", category="compute", tags=["gpu"]),
        ]

        args = _make_args(name="cloud")
        result = cmd_feature_skills(args)
        assert result == 0
        assert "list_instances" in capsys.readouterr().out

    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_skills_unknown_package(self, mock_load, capsys):
        from kestrel_sovereign.cli import cmd_feature_skills

        mock_load.return_value = _make_registry()
        args = _make_args(name="nonexistent")
        result = cmd_feature_skills(args)
        assert result == 1


# ---------------------------------------------------------------------------
# Tests: cmd_skills_search
# ---------------------------------------------------------------------------

class TestSkillsSearch:

    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_search_by_name(self, mock_load, capsys):
        from kestrel_sovereign.cli import cmd_skills_search

        mock_load.return_value = _make_registry()
        args = _make_args(query="speak")
        result = cmd_skills_search(args)
        assert result == 0

        output = capsys.readouterr().out
        assert "speak" in output
        assert "voice" in output

    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_search_by_tag(self, mock_load, capsys):
        from kestrel_sovereign.cli import cmd_skills_search

        mock_load.return_value = _make_registry()
        args = _make_args(query="gpu")
        result = cmd_skills_search(args)
        assert result == 0

        output = capsys.readouterr().out
        assert "list_instances" in output
        assert "launch_instance" in output

    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_search_no_results(self, mock_load, capsys):
        from kestrel_sovereign.cli import cmd_skills_search

        mock_load.return_value = _make_registry()
        args = _make_args(query="zzzznotfound")
        result = cmd_skills_search(args)
        assert result == 0
        assert "No skills matching" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tests: _resolve_feature_name
# ---------------------------------------------------------------------------

class TestResolveFeatureName:

    def test_exact_package_name(self):
        from kestrel_sovereign.cli import _resolve_feature_name
        reg = _make_registry()
        assert _resolve_feature_name("cloud", reg) == "cloud"

    def test_case_insensitive(self):
        from kestrel_sovereign.cli import _resolve_feature_name
        reg = _make_registry()
        assert _resolve_feature_name("Cloud", reg) == "cloud"
        assert _resolve_feature_name("VOICE", reg) == "voice"

    def test_feature_class_name(self):
        from kestrel_sovereign.cli import _resolve_feature_name
        reg = _make_registry()
        assert _resolve_feature_name("RunPodFeature", reg) == "cloud"
        assert _resolve_feature_name("VoiceFeature", reg) == "voice"

    def test_unknown_returns_none(self):
        from kestrel_sovereign.cli import _resolve_feature_name
        reg = _make_registry()
        assert _resolve_feature_name("nonexistent", reg) is None


# ---------------------------------------------------------------------------
# Tests: build_parser includes feature and skills subcommands
# ---------------------------------------------------------------------------

class TestBuildParser:

    def test_feature_subparser_exists(self):
        from kestrel_sovereign.cli import build_parser
        parser = build_parser()
        # Should parse without error
        args = parser.parse_args(["feature", "list"])
        assert args.command == "feature"
        assert args.feature_command == "list"

    def test_feature_install_args(self):
        from kestrel_sovereign.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["feature", "install", "cloud"])
        assert args.command == "feature"
        assert args.feature_command == "install"
        assert args.name == "cloud"

    def test_feature_enable_args(self):
        from kestrel_sovereign.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["feature", "enable", "RunPodFeature"])
        assert args.feature_command == "enable"
        assert args.name == "RunPodFeature"

    def test_feature_disable_args(self):
        from kestrel_sovereign.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["feature", "disable", "RunPodFeature"])
        assert args.feature_command == "disable"
        assert args.name == "RunPodFeature"

    def test_feature_info_args(self):
        from kestrel_sovereign.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["feature", "info", "cloud"])
        assert args.feature_command == "info"
        assert args.name == "cloud"

    def test_feature_scaffold_args(self):
        from kestrel_sovereign.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["feature", "scaffold", "myfeature"])
        assert args.feature_command == "scaffold"
        assert args.name == "myfeature"

    def test_feature_skills_args(self):
        from kestrel_sovereign.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["feature", "skills", "cloud"])
        assert args.feature_command == "skills"
        assert args.name == "cloud"

    def test_skills_search_args(self):
        from kestrel_sovereign.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["skills", "search", "gpu"])
        assert args.command == "skills"
        assert args.skills_command == "search"
        assert args.query == "gpu"
