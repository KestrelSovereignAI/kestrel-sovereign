"""Contracts for optional/extracted feature boundaries in core."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_feature_proof_matrix_marks_mcp_as_external_package_boundary():
    matrix = (PROJECT_ROOT / "docs/audit/FEATURE_PROOF_MATRIX.md").read_text()

    assert "`mcp` | `kestrel-feature-mcp`" in matrix
    assert "kestrel_sovereign/features/mcp.py" not in matrix
