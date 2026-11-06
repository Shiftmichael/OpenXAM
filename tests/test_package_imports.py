"""Static check that the package layout holds together.

Needs only the standard library, so it runs without PyTorch, a GPU or any simulator:

    python tests/test_package_imports.py

It reads source files and never imports them. It checks that every internal import and
every Hydra target points at something that exists, and that nothing still uses the
old top-level names (`data`, `model`, `utils`, `runtime`, `trainer`).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src" / "openxam"
CODE_DIRS = ("src", "scripts", "experiments", "tests")
OLD_ROOTS = {"data", "model", "utils", "runtime", "trainer"}


def code_files() -> list[Path]:
    return [path for name in CODE_DIRS for path in sorted((ROOT / name).rglob("*.py"))]


def locate(module: str) -> Path | None:
    """Return the file or directory behind a dotted openxam module, or None."""
    path = ROOT / "src" / Path(*module.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    return path if path.is_dir() else None


def top_level_names(path: Path) -> tuple[set[str], bool]:
    """Names a module defines at top level, and whether it has a star import."""
    names: set[str] = set()
    has_star = False
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    has_star = True
                else:
                    names.add((alias.asname or alias.name).split(".")[0])
    return names, has_star


def has_attribute(module: str, name: str) -> bool:
    target = locate(module)
    if target is None:
        return False
    if locate(f"{module}.{name}") is not None:
        return True
    if target.is_dir():
        return False
    names, has_star = top_level_names(target)
    return name in names or has_star


def import_problems(path: Path) -> list[str]:
    problems = []
    where = path.relative_to(ROOT)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            first = node.module.split(".")[0]
            if first in OLD_ROOTS:
                problems.append(f"{where}:{node.lineno} old root: {node.module}")
            elif first == "openxam":
                if locate(node.module) is None:
                    problems.append(f"{where}:{node.lineno} no module: {node.module}")
                    continue
                for alias in node.names:
                    if alias.name != "*" and not has_attribute(node.module, alias.name):
                        problems.append(f"{where}:{node.lineno} no name: {node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                first = alias.name.split(".")[0]
                if first in OLD_ROOTS:
                    problems.append(f"{where}:{node.lineno} old root: {alias.name}")
                elif first == "openxam" and locate(alias.name) is None:
                    problems.append(f"{where}:{node.lineno} no module: {alias.name}")
    return problems


def relative_import_problems(path: Path) -> list[str]:
    """Check relative imports inside the package against the files on disk."""
    problems = []
    where = path.relative_to(ROOT)
    parts = path.relative_to(ROOT / "src").with_suffix("").parts
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        keep = len(parts) - node.level
        if keep < 1:
            problems.append(f"{where}:{node.lineno} relative import leaves the package")
            continue
        base = ".".join(parts[:keep] + ((node.module,) if node.module else ()))
        if locate(base) is None:
            problems.append(f"{where}:{node.lineno} no module: {base}")
            continue
        for alias in node.names:
            if alias.name != "*" and not has_attribute(base, alias.name):
                problems.append(f"{where}:{node.lineno} no name: {base}.{alias.name}")
    return problems


def target_problems() -> list[str]:
    problems = []
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            match = re.search(r"_target_:\s*([\w.]+)", line)
            if not match:
                continue
            target = match.group(1)
            first = target.split(".")[0]
            where = f"{path.relative_to(ROOT)}:{number}"
            if first in OLD_ROOTS:
                problems.append(f"{where} old root: {target}")
            elif first == "openxam":
                module, _, name = target.rpartition(".")
                if not has_attribute(module, name):
                    problems.append(f"{where} no target: {target}")
    return problems


def test_internal_imports_resolve() -> None:
    problems = [problem for path in code_files() for problem in import_problems(path)]
    assert not problems, "\n".join(problems)


def test_relative_imports_resolve() -> None:
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    problems = [problem for path in files for problem in relative_import_problems(path)]
    assert not problems, "\n".join(problems)


def test_hydra_targets_resolve() -> None:
    problems = target_problems()
    assert not problems, "\n".join(problems)


if __name__ == "__main__":
    for check in (test_internal_imports_resolve, test_relative_imports_resolve, test_hydra_targets_resolve):
        check()
        print(f"ok  {check.__name__}")
