"""AST-based code analysis for Python files."""
import ast
import logging
from typing import Optional

from .models import CodeDefinition

logger = logging.getLogger(__name__)


class ASTAnalyzer:
    """Extract code definitions from Python source using AST."""

    def __init__(self, source: str, path: str = ""):
        """Initialize with source code.

        Args:
            source: Python source code
            path: File path for error messages
        """
        self.source = source
        self.path = path
        self.lines = source.splitlines()
        self._tree: Optional[ast.AST] = None

    def parse(self) -> bool:
        """Parse source into AST.

        Returns:
            True if parsing succeeded
        """
        try:
            self._tree = ast.parse(self.source, filename=self.path)
            return True
        except SyntaxError as e:
            logger.warning(f"Failed to parse {self.path}: {e}")
            return False

    def get_definitions(self) -> list[CodeDefinition]:
        """Extract all function and class definitions.

        Returns:
            List of code definitions
        """
        if self._tree is None:
            if not self.parse():
                return []

        definitions = []

        for node in ast.walk(self._tree):
            if isinstance(node, ast.FunctionDef):
                definitions.append(self._extract_function(node))
            elif isinstance(node, ast.AsyncFunctionDef):
                definitions.append(self._extract_function(node, is_async=True))
            elif isinstance(node, ast.ClassDef):
                definitions.append(self._extract_class(node))

        return definitions

    def get_definition(self, name: str) -> Optional[CodeDefinition]:
        """Get a specific definition by name.

        Args:
            name: Function or class name

        Returns:
            CodeDefinition if found
        """
        for defn in self.get_definitions():
            if defn.name == name:
                return defn
        return None

    def get_imports(self) -> list[dict]:
        """Extract import statements.

        Returns:
            List of import info dicts
        """
        if self._tree is None:
            if not self.parse():
                return []

        imports = []

        for node in ast.walk(self._tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "from",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "level": node.level,
                    })

        return imports

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool = False,
    ) -> CodeDefinition:
        """Extract function definition."""
        # Get signature
        args = self._format_arguments(node.args)
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"

        prefix = "async def" if is_async else "def"
        signature = f"{prefix} {node.name}({args}){returns}"

        # Get docstring
        docstring = ast.get_docstring(node) or ""

        # Get source lines
        end_line = self._find_end_line(node)
        source = "\n".join(self.lines[node.lineno - 1:end_line])

        # Determine type
        def_type = "method" if self._is_method(node) else "function"

        return CodeDefinition(
            name=node.name,
            type=def_type,
            path=self.path,
            start_line=node.lineno,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            source=source,
        )

    def _extract_class(self, node: ast.ClassDef) -> CodeDefinition:
        """Extract class definition."""
        # Get bases
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        signature = f"class {node.name}({bases})" if bases else f"class {node.name}"

        # Get docstring
        docstring = ast.get_docstring(node) or ""

        # Get source lines
        end_line = self._find_end_line(node)
        source = "\n".join(self.lines[node.lineno - 1:end_line])

        return CodeDefinition(
            name=node.name,
            type="class",
            path=self.path,
            start_line=node.lineno,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            source=source,
        )

    def _format_arguments(self, args: ast.arguments) -> str:
        """Format function arguments."""
        parts = []

        # Regular args
        defaults_start = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"

            default_idx = i - defaults_start
            if default_idx >= 0:
                arg_str += f" = {ast.unparse(args.defaults[default_idx])}"

            parts.append(arg_str)

        # *args
        if args.vararg:
            arg_str = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                arg_str += f": {ast.unparse(args.vararg.annotation)}"
            parts.append(arg_str)
        elif args.kwonlyargs:
            parts.append("*")

        # Keyword-only args
        for i, arg in enumerate(args.kwonlyargs):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            if args.kw_defaults[i]:
                arg_str += f" = {ast.unparse(args.kw_defaults[i])}"
            parts.append(arg_str)

        # **kwargs
        if args.kwarg:
            arg_str = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                arg_str += f": {ast.unparse(args.kwarg.annotation)}"
            parts.append(arg_str)

        return ", ".join(parts)

    def _find_end_line(self, node: ast.AST) -> int:
        """Find the last line of a node."""
        end_line = node.lineno

        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                end_line = max(end_line, child.lineno)
            if hasattr(child, "end_lineno") and child.end_lineno:
                end_line = max(end_line, child.end_lineno)

        return end_line

    def _is_method(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if function is a method (inside a class)."""
        # Check if first argument is self or cls
        if node.args.args:
            first_arg = node.args.args[0].arg
            return first_arg in ("self", "cls")
        return False


def extract_definition(source: str, name: str, path: str = "") -> Optional[CodeDefinition]:
    """Convenience function to extract a single definition.

    Args:
        source: Python source code
        name: Name to find
        path: File path for context

    Returns:
        CodeDefinition if found
    """
    analyzer = ASTAnalyzer(source, path)
    return analyzer.get_definition(name)


def list_definitions(source: str, path: str = "") -> list[CodeDefinition]:
    """Convenience function to list all definitions.

    Args:
        source: Python source code
        path: File path for context

    Returns:
        List of all definitions
    """
    analyzer = ASTAnalyzer(source, path)
    return analyzer.get_definitions()
