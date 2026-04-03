"""
SSE-native MCP Test Server with useful tools.

This server demonstrates:
1. Native SSE transport (no wrapper needed)
2. Multiple tool categories
3. Async operations
4. Error handling

Tools provided:
- echo: Echo back text (basic test)
- add: Add two numbers (math)
- multiply: Multiply numbers (math)
- get_time: Get current UTC time
- random_number: Generate random number in range
- calculate: Evaluate a math expression safely
- word_count: Count words in text
- reverse_text: Reverse a string
"""

from mcp.server.fastmcp import FastMCP
import uvicorn
import random
import datetime
import ast
import operator

# Create FastMCP server
mcp = FastMCP("kestrel-tools-server")

# Safe math operators for calculate
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def safe_eval(node):
    """Safely evaluate a math expression AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op(operand)
    else:
        raise ValueError(f"Unsupported expression type: {type(node)}")


# =============================================================================
# Basic Tools
# =============================================================================

@mcp.tool()
def echo(text: str) -> str:
    """Echo back the input text. Useful for testing connectivity."""
    return f"Echo: {text}"


@mcp.tool()
def reverse_text(text: str) -> str:
    """Reverse a string. Returns the text backwards."""
    return text[::-1]


@mcp.tool()
def word_count(text: str) -> str:
    """Count words, characters, and lines in text."""
    words = len(text.split())
    chars = len(text)
    lines = len(text.splitlines()) or 1
    return f"Words: {words}, Characters: {chars}, Lines: {lines}"


# =============================================================================
# Math Tools
# =============================================================================

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Safely evaluate a math expression.
    Supports: +, -, *, /, **, % with integers and floats.
    Example: "2 + 3 * 4" returns "14"
    """
    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# Utility Tools
# =============================================================================

@mcp.tool()
def get_time() -> str:
    """Get the current UTC time in ISO format."""
    return datetime.datetime.utcnow().isoformat() + "Z"


@mcp.tool()
def random_number(min_val: int, max_val: int) -> int:
    """Generate a random integer between min_val and max_val (inclusive)."""
    return random.randint(min_val, max_val)


@mcp.tool()
def uuid() -> str:
    """Generate a random UUID v4."""
    import uuid as uuid_module
    return str(uuid_module.uuid4())


if __name__ == "__main__":
    print("Starting Kestrel Tools MCP Server on 0.0.0.0:8000")
    print("Tools: echo, reverse_text, word_count, add, multiply, calculate, get_time, random_number, uuid")
    uvicorn.run(mcp.sse_app, host="0.0.0.0", port=8000)
