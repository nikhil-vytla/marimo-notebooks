"""Fail if a marimo @app.cell has an effectively empty body.

An empty cell is one whose statements are all one of: `pass`, `return`, or a
bare string literal (docstring). Such cells do nothing and are usually noise.
"""

from __future__ import annotations

import ast
import sys


def _is_cell_decorator(node: ast.expr) -> bool:
    """True for `@app.cell` and `@app.cell(...)`."""
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "cell"
        and isinstance(node.value, ast.Name)
        and node.value.id == "app"
    )


def _is_empty_statement(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Pass, ast.Return)):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return isinstance(node.value.value, str)
    return False


def _empty_cells(source: str) -> list[int]:
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_is_cell_decorator(d) for d in node.decorator_list):
            continue
        if node.body and all(_is_empty_statement(s) for s in node.body):
            lines.append(node.lineno)
    return lines


def main(paths: list[str]) -> int:
    found = False
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        for lineno in _empty_cells(source):
            print(f"{path}:{lineno}")
            found = True
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
