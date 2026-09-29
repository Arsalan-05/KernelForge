"""Static code policy for public deployments.

On an always-on public server, both the PyTorch reference (user-supplied) and
the candidate kernel (model output, steerable through the prompt) end up
executing in the harness subprocess. This AST check is defence in depth on
top of the subprocess, rlimits and scrubbed environment: it allows the
modules and constructs real kernels use and rejects the usual escape routes
(imports of os/subprocess/socket, exec/eval/open, dunder traversal like
`().__class__.__subclasses__()`, torch/numpy file and extension loaders).
It is not a proof of safety; Python has no airtight in-process sandbox.
"""

from __future__ import annotations

import ast
from typing import Optional

ALLOWED_MODULES = {
    "__future__", "torch", "triton", "math", "numpy", "typing", "functools",
    "itertools", "dataclasses", "collections", "operator",
}
FORBIDDEN_CALLS = {
    "exec", "eval", "compile", "open", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint",
}
# Attribute chains that load code or touch the filesystem/network.
FORBIDDEN_ATTRIBUTE_PATHS = {
    ("torch", "load"), ("torch", "save"), ("torch", "hub"), ("torch", "ops"),
    ("torch", "library"), ("torch", "classes"), ("torch", "utils"), ("torch", "package"),
    ("torch", "distributed"), ("torch", "multiprocessing"), ("torch", "cuda", "memory"),
    ("np", "load"), ("np", "save"), ("np", "fromfile"), ("np", "memmap"),
    ("numpy", "load"), ("numpy", "save"), ("numpy", "fromfile"), ("numpy", "memmap"),
}
ALLOWED_DUNDERS = {"__init__", "__name__"}
MAX_SOURCE_CHARS = 20_000


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def check_code(source: str, label: str = "code") -> Optional[str]:
    """Return None if `source` passes the policy, else a one-line reason."""
    if len(source) > MAX_SOURCE_CHARS:
        return f"{label} is longer than {MAX_SOURCE_CHARS} characters"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"{label} has a syntax error on line {exc.lineno}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    return f"{label} imports `{alias.name}`, which isn't allowed on the public server"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                return f"{label} imports from `{node.module}`, which isn't allowed on the public server"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return f"{label} calls `{node.func.id}()`, which isn't allowed on the public server"
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_CALLS or (node.id.startswith("__") and node.id not in ALLOWED_DUNDERS):
                return f"{label} uses `{node.id}`, which isn't allowed on the public server"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
                return f"{label} accesses `.{node.attr}`, which isn't allowed on the public server"
            path = _attribute_path(node)
            for forbidden in FORBIDDEN_ATTRIBUTE_PATHS:
                if path[: len(forbidden)] == forbidden:
                    return f"{label} uses `{'.'.join(forbidden)}`, which isn't allowed on the public server"
    return None
