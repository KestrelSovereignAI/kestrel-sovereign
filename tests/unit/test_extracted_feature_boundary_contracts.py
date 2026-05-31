"""Contracts for optional/extracted feature boundaries in core."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_feature_proof_matrix_marks_mcp_as_external_package_boundary():
    matrix = (PROJECT_ROOT / "docs/audit/FEATURE_PROOF_MATRIX.md").read_text()

    assert "`mcp` | `kestrel-feature-mcp`" in matrix
    assert "kestrel_sovereign/features/mcp.py" not in matrix


def test_observability_feature_is_external_to_core():
    registry = (PROJECT_ROOT / "kestrel_sovereign/data/feature_registry.toml").read_text()
    feature_dir = PROJECT_ROOT / "kestrel_sovereign/features/observability"

    assert 'package = "kestrel-feature-observability"' in registry
    assert "[observability]" in registry
    assert "`observability` | `kestrel-feature-observability`" in (
        PROJECT_ROOT / "docs/audit/FEATURE_PROOF_MATRIX.md"
    ).read_text()
    observability_block = registry.split("[observability]", 1)[1].split("\n[", 1)[0]
    assert "core = false" in observability_block
    source_files = [
        path for path in feature_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ] if feature_dir.exists() else []
    assert source_files == []


def test_wallet_feature_is_external_to_core():
    registry = (PROJECT_ROOT / "kestrel_sovereign/data/feature_registry.toml").read_text()
    feature_dir = PROJECT_ROOT / "kestrel_sovereign/features/wallet"

    assert 'package = "kestrel-feature-wallet"' in registry
    assert "[wallet]" in registry
    assert "`wallet` | `kestrel-feature-wallet`" in (
        PROJECT_ROOT / "docs/audit/FEATURE_PROOF_MATRIX.md"
    ).read_text()
    wallet_block = registry.split("[wallet]", 1)[1].split("\n[", 1)[0]
    assert "core = false" in wallet_block
    source_files = [
        path for path in feature_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ] if feature_dir.exists() else []
    assert source_files == []


def test_github_feature_is_external_to_core():
    registry = (PROJECT_ROOT / "kestrel_sovereign/data/feature_registry.toml").read_text()
    feature_dir = PROJECT_ROOT / "kestrel_sovereign/features/github"

    assert 'package = "kestrel-feature-github"' in registry
    assert "[github]" in registry
    assert "`github` | `kestrel-feature-github`" in (
        PROJECT_ROOT / "docs/audit/FEATURE_PROOF_MATRIX.md"
    ).read_text()
    github_block = registry.split("[github]", 1)[1].split("\n[", 1)[0]
    assert "core = false" in github_block
    source_files = [
        path for path in feature_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ] if feature_dir.exists() else []
    assert source_files == []


def test_voice_feature_is_external_to_core():
    registry = (PROJECT_ROOT / "kestrel_sovereign/data/feature_registry.toml").read_text()
    feature_dir = PROJECT_ROOT / "kestrel_sovereign/features/voice"
    support_dir = PROJECT_ROOT / "kestrel_sovereign/voice"

    assert 'package = "kestrel-feature-voice"' in registry
    assert "[voice]" in registry
    assert "`voice` | `kestrel-feature-voice`" in (
        PROJECT_ROOT / "docs/audit/FEATURE_PROOF_MATRIX.md"
    ).read_text()
    voice_block = registry.split("[voice]", 1)[1].split("\n[", 1)[0]
    assert "core = false" in voice_block
    source_files = [
        path for path in feature_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ] if feature_dir.exists() else []
    assert source_files == []
    support_files = [
        path for path in support_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ] if support_dir.exists() else []
    assert support_files == []


def test_reflection_feature_is_external_to_core():
    registry = (PROJECT_ROOT / "kestrel_sovereign/data/feature_registry.toml").read_text()
    feature_dir = PROJECT_ROOT / "kestrel_sovereign/features/reflection"

    assert 'package = "kestrel-feature-reflection"' in registry
    assert "[reflection]" in registry
    assert "`reflection` | `kestrel-feature-reflection`" in (
        PROJECT_ROOT / "docs/audit/FEATURE_PROOF_MATRIX.md"
    ).read_text()
    reflection_block = registry.split("[reflection]", 1)[1].split("\n[", 1)[0]
    assert "core = false" in reflection_block
    source_files = [
        path for path in feature_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ] if feature_dir.exists() else []
    assert source_files == []


def test_council_feature_is_external_to_core():
    registry = (PROJECT_ROOT / "kestrel_sovereign/data/feature_registry.toml").read_text()
    feature_dir = PROJECT_ROOT / "kestrel_sovereign/features/council"

    assert 'package = "kestrel-feature-council"' in registry
    assert "[council]" in registry
    assert "`council` | `kestrel-feature-council`" in (
        PROJECT_ROOT / "docs/audit/FEATURE_PROOF_MATRIX.md"
    ).read_text()
    council_block = registry.split("[council]", 1)[1].split("\n[", 1)[0]
    assert "core = false" in council_block
    source_files = [
        path for path in feature_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ] if feature_dir.exists() else []
    assert source_files == []
