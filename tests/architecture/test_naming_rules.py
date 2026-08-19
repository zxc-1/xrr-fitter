"""Enforce one naming policy over the exact Radon-managed Python file set.

The scanner checks module paths, symbols, package initializers, responsibility
prefixes, permanent task-stage names, and Qt camelCase overrides. Qt methods
are exempt only when an imported Qt base really exposes that method.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAKE_CASE = re.compile(r"_*[a-z][a-z0-9_]*")
CAP_WORDS = re.compile(r"_?[A-Z][A-Za-z0-9]*")
UPPER_SNAKE_CASE = re.compile(r"[A-Z][A-Z0-9_]*")
TASK_STAGE = re.compile(r"(?:^|_)task\d+(?:_|$)")
QT_BASE_PREFIXES = ("PySide6.", "matplotlib.backends.backend_qtagg.")


@dataclass(frozen=True)
class NamingViolation:
    kind: str
    name: str
    line: int


def _is_snake_case(name: str) -> bool:
    return name == "_" or (name.startswith("__") and name.endswith("__")) or SNAKE_CASE.fullmatch(name) is not None


def _valid_module_path(relative: Path) -> bool:
    name = relative.name
    if name in {"__init__.py", "__main__.py"}:
        return True
    if name.startswith("_") or relative.suffix != ".py":
        return False
    stem = relative.stem
    if SNAKE_CASE.fullmatch(stem) is None or TASK_STAGE.search(stem):
        return False
    responsibility = stem.removeprefix("test_")
    parent = relative.parent.name
    return not responsibility.startswith(f"{parent}_")


def _python_files() -> list[Path]:
    return [
        path
        for managed in ("src", "tests", "tools", "examples")
        for path in (ROOT / managed).rglob("*.py")
        if (ROOT / managed).is_dir()
    ]


def _assert_module_name(path: Path) -> None:
    assert _valid_module_path(path.relative_to(ROOT)), path.relative_to(ROOT)
    if path.name == "__init__.py":
        assert path.stat().st_size == 0


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bindings


def _qualified_name(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value, bindings)
        return f"{base}.{node.attr}" if base else None
    return None


def _import_object(qualified: str) -> object | None:
    parts = qualified.split(".")
    for boundary in range(len(parts), 0, -1):
        try:
            value: object = importlib.import_module(".".join(parts[:boundary]))
        except ImportError:
            continue
        for attribute in parts[boundary:]:
            value = getattr(value, attribute, None)
            if value is None:
                break
        return value
    return None


def _class_qt_bases(
    node: ast.ClassDef,
    bindings: dict[str, str],
    local_bases: dict[str, tuple[type[object], ...]],
) -> tuple[type[object], ...]:
    bases: list[type[object]] = []
    for expression in node.bases:
        qualified = _qualified_name(expression, bindings)
        if qualified in local_bases:
            bases.extend(local_bases[qualified])
            continue
        value = _import_object(qualified) if qualified and qualified.startswith(QT_BASE_PREFIXES) else None
        if isinstance(value, type):
            bases.append(value)
    return tuple(bases)


def _qt_overrides(tree: ast.Module) -> set[tuple[str, str]]:
    bindings = _import_bindings(tree)
    local_bases: dict[str, tuple[type[object], ...]] = {}
    overrides: set[tuple[str, str]] = set()
    for node in (item for item in tree.body if isinstance(item, ast.ClassDef)):
        bases = _class_qt_bases(node, bindings, local_bases)
        local_bases[node.name] = bases
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                hasattr(base, item.name) for base in bases
            ):
                overrides.add((node.name, item.name))
    return overrides


def _definition_violations(tree: ast.Module, qt_overrides: set[tuple[str, str]]) -> list[NamingViolation]:
    violations: list[NamingViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and CAP_WORDS.fullmatch(node.name) is None:
            violations.append(NamingViolation("class", node.name, node.lineno))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = next(
                (owner.name for owner in ast.walk(tree) if isinstance(owner, ast.ClassDef) and node in owner.body),
                None,
            )
            if not _is_snake_case(node.name) and (parent, node.name) not in qt_overrides:
                violations.append(NamingViolation("function", node.name, node.lineno))
    return violations


def _assignment_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for target in ast.walk(node):
        if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
            names.add(target.id)
    return names


def _module_constants(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            names.update(_assignment_names(node))
    for owner in ast.walk(tree):
        if isinstance(owner, ast.ClassDef):
            for node in owner.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    names.update(name for name in _assignment_names(node) if UPPER_SNAKE_CASE.fullmatch(name))
    return names


def _type_aliases(tree: ast.Module) -> set[str]:
    aliases: set[str] = set()
    type_expressions = (ast.Name, ast.Attribute, ast.Subscript, ast.BinOp)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, type_expressions):
            aliases.update(name for name in _assignment_names(node) if CAP_WORDS.fullmatch(name))
        elif isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
            aliases.add(node.name.id)
    return aliases


def _variable_violations(tree: ast.Module) -> list[NamingViolation]:
    constants = _module_constants(tree)
    type_aliases = _type_aliases(tree)
    return [
        *_constant_violations(constants, type_aliases),
        *_stored_name_violations(tree, constants | type_aliases),
    ]


def _constant_violations(constants: set[str], type_aliases: set[str]) -> list[NamingViolation]:
    return [
        NamingViolation("constant", name, 0)
        for name in sorted(constants - type_aliases)
        if not (name.startswith("__") and name.endswith("__")) and UPPER_SNAKE_CASE.fullmatch(name) is None
    ]


def _stored_name_violations(tree: ast.Module, exempt: set[str]) -> list[NamingViolation]:
    violations: list[NamingViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and not _is_snake_case(node.arg):
            violations.append(NamingViolation("variable", node.arg, node.lineno))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id not in exempt and not _is_snake_case(node.id):
                violations.append(NamingViolation("variable", node.id, node.lineno))
    return violations


def _symbol_violations(tree: ast.Module, qt_overrides: set[tuple[str, str]]) -> tuple[NamingViolation, ...]:
    violations = [
        *_definition_violations(tree, qt_overrides),
        *_variable_violations(tree),
    ]
    return tuple(sorted(set(violations), key=lambda item: (item.line, item.kind, item.name)))


def _radon_python_files() -> list[Path]:
    path = ROOT / "tools" / "check_radon.py"
    spec = importlib.util.spec_from_file_location("r23_check_radon_for_naming", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    relative, _issues = module.discover_python_files(ROOT)
    return [ROOT / item for item in relative]


def test_python_module_names_follow_r23_rules() -> None:
    python_files = _python_files()
    assert python_files
    for path in python_files:
        _assert_module_name(path)


def test_python_symbols_follow_r23_rules() -> None:
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert _symbol_violations(tree, _qt_overrides(tree)) == (), path.relative_to(ROOT)


def test_no_forwarding_or_compatibility_module_name_exists() -> None:
    names = {path.name for path in _radon_python_files()}
    assert names.isdisjoint({"compat.py", "xrr_core.py", "xrr_app.py"})


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("src/xrr_fitter/fit/_private.py", False),
        ("src/xrr_fitter/fit/fit_resume.py", False),
        ("tests/gui/test_gui_results.py", False),
        ("tests/gui/test_task17_results.py", False),
        ("tests/gui/test_results_task9.py", False),
        ("src/xrr_fitter/fit/resume.py", True),
        ("tests/gui/test_results.py", True),
        ("src/xrr_fitter/__main__.py", True),
    ],
)
def test_module_path_fixture_enforces_private_parent_prefix_and_any_task_stage(relative: str, expected: bool) -> None:
    assert _valid_module_path(Path(relative)) is expected


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("def camelCase():\n    pass\n", "function"),
        ("class lower_name:\n    pass\n", "class"),
        ("module_value = 1\n", "constant"),
        ("def ok(badName):\n    localName = badName\n", "variable"),
    ],
)
def test_symbol_fixture_rejects_noncanonical_names(source: str, kind: str) -> None:
    violations = _symbol_violations(ast.parse(source), {})
    assert kind in {violation.kind for violation in violations}


def test_symbol_fixture_accepts_snake_capwords_constants_and_real_qt_override() -> None:
    source = """
from PySide6.QtWidgets import QWidget

MODULE_VALUE = 1

class MainPanel(QWidget):
    def closeEvent(self, event):
        local_value = event
        return local_value

def public_function(argument_name):
    return argument_name
"""
    tree = ast.parse(source)
    qt_overrides = _qt_overrides(tree)
    assert ("MainPanel", "closeEvent") in qt_overrides
    assert _symbol_violations(tree, qt_overrides) == ()


def test_symbol_fixture_accepts_qt_override_from_matplotlib_qt_canvas() -> None:
    source = """
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

class DiagnosticCanvas(FigureCanvasQTAgg):
    def closeEvent(self, event):
        return event
"""
    tree = ast.parse(source)
    qt_overrides = _qt_overrides(tree)

    assert ("DiagnosticCanvas", "closeEvent") in qt_overrides
    assert _symbol_violations(tree, qt_overrides) == ()


def test_naming_scans_exactly_the_same_python_files_as_radon() -> None:
    assert set(_python_files()) == set(_radon_python_files())


def test_root_conftest_is_the_only_conftest() -> None:
    conftests = {path.relative_to(ROOT).as_posix() for path in _radon_python_files() if path.name == "conftest.py"}
    assert conftests == {"tests/conftest.py"}
