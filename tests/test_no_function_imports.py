"""Every import sits at the top of its file: none inside a function (CLAUDE.md, code rules).

Parses every tracked .py file and fails, naming file:line, on any import statement nested in a
function. Run: python tests/test_no_function_imports.py
"""
import ast
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def function_level_imports(path):
    """Import statements nested inside a function of one source file.

    Args:
        path: str, path to a .py file.
    Returns:
        list of (lineno, function name) for each import found inside a function body.
    """
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    found = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    found.append((node.lineno, fn.name))
    return sorted(set(found))


def main():
    """Scan every tracked .py file; exit 1 and list the offenders if any import is inside a function."""
    files = subprocess.check_output(["git", "ls-files", "*.py"], cwd=REPO).decode().split()
    bad = [(f, ln, fn) for f in files for ln, fn in function_level_imports(os.path.join(REPO, f))]
    for f, ln, fn in bad:
        print(f"{f}:{ln} import inside {fn}()")
    if bad:
        sys.exit(1)
    print(f"ok: {len(files)} files, no imports inside functions")


if __name__ == "__main__":
    main()
