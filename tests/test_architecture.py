"""The package's shape: imports flow one way down the layers, modules expose classes (no free functions except the
command line's main), and every file stays small enough to read whole."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "fah"
LAYER = {"core": 1, "storage": 2, "sources": 2, "channels": 3, "hub": 3, "commands": 4, "__main__": 5}
# Imports allowed inside the same package: the package's base modules only (so its files stay independent).
SAME_PACKAGE = {
    "core": None,  # any core module
    "storage": {"fah.storage.attrs"},
    "sources": {"fah.sources.http", "fah.sources.arcgis"},
    "channels": {"fah.channels.base"},
    "hub": None,
    "commands": None,
}
MAX_LINES = 400
MODULES = sorted(p for p in ROOT.rglob("*.py"))


def package_of(path: Path) -> str:
    rel = path.relative_to(ROOT)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def fah_imports(tree: ast.Module) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names if a.name.startswith("fah.")}
        elif isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("fah."):
            out.add(n.module)
    return out


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(ROOT)))
def test_imports_point_down_the_layers(path):
    pkg = package_of(path)
    if pkg == "__init__":
        return
    for mod in fah_imports(ast.parse(path.read_text())):
        target = mod.split(".")[1]
        if target == pkg:
            allowed = SAME_PACKAGE[pkg]
            is_registry = path.name == "registry.py" and pkg == "channels"  # the registry lists every channel
            assert allowed is None or mod in allowed or is_registry, f"{path.name} imports {mod} (same package)"
        else:
            assert LAYER[target] < LAYER[pkg], f"{path.name} ({pkg}) imports {mod}: up or sideways"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(ROOT)))
def test_modules_expose_classes_not_free_functions(path):
    tree = ast.parse(path.read_text())
    functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert functions == (["main"] if path.name == "__main__.py" and path.parent == ROOT else [])


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(ROOT)))
def test_modules_stay_short(path):
    assert len(path.read_text().splitlines()) <= MAX_LINES
