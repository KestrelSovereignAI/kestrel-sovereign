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
    PackageBoundary,
    SkillInfo,
)
from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES


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
            package="kestrel-sovereign",
            git="https://github.com/example/ks.git",
            features=["DeployFeature", "ComputeFeature"],
            description="Core deployment and guarded compute surfaces",
            tags=["infrastructure", "gpu"],
            icon="cloud",
            core=True,
            skills=[
                SkillInfo(name="deploy_agent", description="Deploy agent", category="compute", tags=["gpu"]),
                SkillInfo(name="run_script", description="Run guarded script", category="system", tags=["gpu"]),
            ],
        ),
        "wallet": FeaturePackageInfo(
            name="wallet",
            package="kestrel-feature-wallet",
            git="https://github.com/example/wallet.git",
            features=["WalletFeature"],
            description="Wallet tools",
            tags=["wallet"],
            icon="wallet",
            core=False,
        ),
        "voice": FeaturePackageInfo(
            name="voice",
            package="kestrel-feature-voice",
            git="https://github.com/example/voice.git",
            features=["VoiceFeature"],
            description="TTS and STT",
            tags=["voice", "tts"],
            icon="microphone",
            core=False,
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
        reg["wallet"].status = FeatureStatus.AVAILABLE
        mock_get_reg.return_value = reg

        args = _make_args()
        result = cmd_feature_list(args)
        assert result == 0

        output = capsys.readouterr().out
        assert "identity" in output
        assert "voice" in output
        assert "wallet" in output
        assert "INSTALLED" in output or "installed" in output
        assert "AVAILABLE" in output or "available" in output

    @patch("kestrel_sovereign.feature_registry.get_registry")
    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=["VoiceFeature"])
    def test_list_shows_disabled_from_toml(self, mock_disabled, mock_get_reg, capsys):
        from kestrel_sovereign.cli import cmd_feature_list

        reg = _make_registry()
        reg["identity"].status = FeatureStatus.INSTALLED
        reg["voice"].status = FeatureStatus.INSTALLED
        reg["wallet"].status = FeatureStatus.AVAILABLE
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

    @patch("kestrel_sovereign.cli_features._extension_install_run")
    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_install_routes_through_uv_aware_helper(self, mock_load, mock_install, capsys):
        """install goes through _extension_install_run (uv-aware), not bare python -m pip."""
        from kestrel_sovereign.cli import cmd_feature_install

        mock_load.return_value = _make_registry()
        mock_install.return_value = MagicMock(returncode=0)

        args = _make_args(name="wallet")
        result = cmd_feature_install(args)
        assert result == 0

        # Verify the uv-aware helper was called with the package spec
        mock_install.assert_called_once_with(["kestrel-feature-wallet"])

    @patch("kestrel_sovereign.cli_features._extension_install_run")
    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_install_accepts_registered_xai_distribution_name(
        self, mock_load, mock_install, capsys,
    ):
        """The PyPI distribution name is a first-class install identifier."""
        from kestrel_sovereign.cli import cmd_feature_install

        registry = _make_registry()
        registry["voice_xai"] = FeaturePackageInfo(
            name="voice_xai",
            package="kestrel-voice-xai",
            git="https://github.com/example/voice-xai.git",
            features=[],
            provider_classes=[
                "XAITTSProvider",
                "XAISTTProvider",
                "XAIRealtimeConversationProvider",
            ],
            entry_point_groups=[
                "kestrel_feature_voice_providers",
                "kestrel_sovereign.conversation_providers",
            ],
            boundary=PackageBoundary.PROVIDER_PACKAGE,
            description="xAI voice providers",
            tags=["voice", "xai"],
            icon="microphone",
            core=False,
        )
        mock_load.return_value = registry
        mock_install.return_value = MagicMock(returncode=0)

        result = cmd_feature_install(_make_args(name="kestrel-voice-xai"))

        assert result == 0
        mock_install.assert_called_once_with(["kestrel-voice-xai"])
        assert "Installed kestrel-voice-xai" in capsys.readouterr().out

    @patch("kestrel_sovereign.cli_features._extension_install_run")
    @patch("kestrel_sovereign.feature_registry.load_registry")
    def test_install_git_fallback_uses_uv_aware_helper(self, mock_load, mock_install, capsys):
        """A failed PyPI install falls back to git+ through the same uv-aware helper."""
        from kestrel_sovereign.cli import cmd_feature_install

        mock_load.return_value = _make_registry()
        mock_install.side_effect = [
            MagicMock(returncode=1, stderr="boom"),  # PyPI attempt fails
            MagicMock(returncode=0),                 # git fallback succeeds
        ]

        args = _make_args(name="wallet")
        result = cmd_feature_install(args)
        assert result == 0

        assert mock_install.call_args_list[0].args[0] == ["kestrel-feature-wallet"]
        assert mock_install.call_args_list[1].args[0] == [
            "git+https://github.com/example/wallet.git"
        ]


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
        assert "DeployFeature" in saved
        assert "ComputeFeature" in saved
        assert "Disabled" in capsys.readouterr().out

    @patch("kestrel_sovereign.cli._set_toml_disabled_features")
    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=[])
    @patch("kestrel_sovereign.feature_registry.load_registry")
    @pytest.mark.parametrize("feature_name", sorted(MANDATORY_FEATURES))
    def test_disable_rejects_mandatory_feature_without_writing(
        self, mock_load, mock_get, mock_set, feature_name, capsys
    ):
        from kestrel_sovereign.cli import cmd_feature_disable

        registry = _make_registry()
        registry["mandatory-under-test"] = FeaturePackageInfo(
            name="mandatory-under-test",
            package="kestrel-sovereign",
            git="https://github.com/example/ks.git",
            features=[feature_name],
            description="Mandatory feature under test",
            tags=["core"],
            icon="shield",
            core=True,
        )
        mock_load.return_value = registry
        result = cmd_feature_disable(_make_args(name="mandatory-under-test"))

        assert result == 1
        assert feature_name in capsys.readouterr().out
        mock_set.assert_not_called()

    @patch("kestrel_sovereign.cli._get_toml_disabled_features", return_value=["DeployFeature", "ComputeFeature"])
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
        assert "Core deployment" in output
        assert "kestrel-sovereign" in output
        assert "DeployFeature" in output
        assert "ComputeFeature" in output
        assert "deploy_agent" in output

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

        args = _make_args(name="DeployFeature")
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
            SkillInfo(name="deploy_agent", description="Deploy agent", category="compute", tags=["gpu"]),
        ]

        args = _make_args(name="cloud")
        result = cmd_feature_skills(args)
        assert result == 0
        assert "deploy_agent" in capsys.readouterr().out

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
        assert "deploy_agent" in output
        assert "run_script" in output

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
        assert _resolve_feature_name("DeployFeature", reg) == "cloud"
        assert _resolve_feature_name("VoiceFeature", reg) == "voice"

    def test_distribution_name(self):
        from kestrel_sovereign.cli import _resolve_feature_name
        reg = _make_registry()
        assert _resolve_feature_name("kestrel-feature-wallet", reg) == "wallet"

    def test_distribution_name_uses_python_normalization(self):
        from kestrel_sovereign.cli import _resolve_feature_name
        reg = _make_registry()
        assert _resolve_feature_name("KESTREL_FEATURE.WALLET", reg) == "wallet"

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
        args = parser.parse_args(["feature", "enable", "DeployFeature"])
        assert args.feature_command == "enable"
        assert args.name == "DeployFeature"

    def test_feature_disable_args(self):
        from kestrel_sovereign.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["feature", "disable", "DeployFeature"])
        assert args.feature_command == "disable"
        assert args.name == "DeployFeature"

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
