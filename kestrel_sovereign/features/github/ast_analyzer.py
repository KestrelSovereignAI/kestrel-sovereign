"""AST-based code analysis - re-exported from kestrel-feature-github package.

This module exists for backward compatibility. The canonical implementation
lives in kestrel_feature_github.ast_analyzer.
"""
from kestrel_feature_github.ast_analyzer import (
    ASTAnalyzer,
    extract_definition,
    list_definitions,
)

__all__ = ["ASTAnalyzer", "extract_definition", "list_definitions"]
