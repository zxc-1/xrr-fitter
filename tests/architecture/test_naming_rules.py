"""Enforce one naming policy over the exact Radon-managed Python file set.

The scanner checks module paths, symbols, package initializers, responsibility
prefixes, permanent task-stage names, and Qt camelCase overrides. Qt methods
are exempt only when a Qt base really exposes that method; the base may be
imported from Qt directly, reached through a project widget that descends from
Qt, or reached through a pyqtgraph graphics item, and a base without Qt
ancestry exempts nothing.
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
# 项目自己的 widget 和 pyqtgraph 的绘图项也可以当基类，但放行仍然只跟 Qt 血缘：解析出来的
# 类得真有 Qt 祖先。所以 pg.AxisItem（一路继承到 QGraphicsWidget）算，pg.ColorMap 这类
# 同库里的纯 Python 类不算——放宽跟的是「是不是 Qt 后代」，不是「出自哪个库」。
QT_DESCENDANT_BASE_PREFIXES = ("xrr_fitter.", "pyqtgraph.")


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


def _has_qt_ancestor(value: type[object]) -> bool:
    """Say whether a class really descends from Qt, by module of its ancestors."""
    return any(getattr(ancestor, "__module__", "").startswith(QT_BASE_PREFIXES) for ancestor in value.__mro__)


def _resolved_base(qualified: str | None) -> type[object] | None:
    """Resolve a base class expression to a Qt class, or to nothing.

    Qt 基类直接认；项目基类与 pyqtgraph 的绘图项要多走一步：导进来看它的 ``__mro__`` 里有
    没有 Qt 祖先。解析得到但不是 Qt 后代的类照样返回 ``None``，所以它下面的 camelCase 仍然违规。
    """
    if not qualified:
        return None
    if qualified.startswith(QT_BASE_PREFIXES):
        value = _import_object(qualified)
        return value if isinstance(value, type) else None
    if not qualified.startswith(QT_DESCENDANT_BASE_PREFIXES):
        return None
    value = _import_object(qualified)
    return value if isinstance(value, type) and _has_qt_ancestor(value) else None


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
        value = _resolved_base(qualified)
        if value is not None:
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


def test_symbol_fixture_accepts_qt_override_from_a_pyqtgraph_item() -> None:
    """pyqtgraph 的绘图项也是 Qt 控件，覆写它的虚函数同样只能照 Qt 的 camelCase 写。

    ``AxisItem`` 一路继承到 ``QGraphicsWidget``。要改对数刻度的写法，pyqtgraph 留的钩子
    叫 ``logTickStrings``，名字由它定：改成 snake_case 就不再是覆写，根本不会被调用。规则
    原先按发行方认基类（PySide6 与 matplotlib 的 Qt 画布），于是同样是 Qt 后代的 pyqtgraph
    被判成命名违规——那等于让门禁替库作者改方法名。
    """
    source = """
import pyqtgraph as pg

class DecadeAxis(pg.AxisItem):
    def logTickStrings(self, values, scale, spacing):
        return [str(value) for value in values]
"""
    tree = ast.parse(source)
    qt_overrides = _qt_overrides(tree)

    assert ("DecadeAxis", "logTickStrings") in qt_overrides
    assert _symbol_violations(tree, qt_overrides) == ()


def test_symbol_fixture_still_rejects_camelcase_on_a_pyqtgraph_base_without_qt_ancestry() -> None:
    """放行跟着 Qt 血缘走，不跟着「基类出自哪个库」走。

    ``pg.ColorMap`` 同在 pyqtgraph 里，却是个纯 Python 类，没有 Qt 祖先，它下面的
    camelCase 方法不覆写任何东西。少了这条对照，「也认 pyqtgraph 基类」就成了按库名整批放宽。
    """
    source = """
import pyqtgraph as pg

class Ramp(pg.ColorMap):
    def notAQtMethod(self):
        return None
"""
    tree = ast.parse(source)
    qt_overrides = _qt_overrides(tree)

    assert qt_overrides == set()
    assert [violation.name for violation in _symbol_violations(tree, qt_overrides)] == ["notAQtMethod"]


def test_symbol_fixture_accepts_qt_override_through_a_project_widget_base() -> None:
    """项目自己的 Qt 子类也算 Qt 基类，跨模块那一跳不能把血缘弄断。

    ``ReorderableTree`` 继承 ``ContentSizedTree``，覆写的仍然是 ``QAbstractItemView``
    的 ``dropEvent``。规则原先只跟两种基类：同文件里定义的，和直接从 Qt 导入的；中间隔
    一个项目模块，真覆写就会被判成命名违规。
    """
    source = """
from xrr_fitter.gui.sizing import ContentSizedTree

class ReorderableTree(ContentSizedTree):
    def dropEvent(self, event):
        return event
"""
    tree = ast.parse(source)
    qt_overrides = _qt_overrides(tree)

    assert ("ReorderableTree", "dropEvent") in qt_overrides
    assert _symbol_violations(tree, qt_overrides) == ()


def test_symbol_fixture_still_rejects_camelcase_on_a_project_base_without_qt_ancestry() -> None:
    """放行跟着 Qt 血缘走，不跟着「是项目里的类」走。

    这条是上一条的对照：项目里的普通类没有 Qt 祖先，它下面的 camelCase 方法不是覆写任何
    东西，仍然是违规。少了这条，「也解析项目基类」就等于把门禁悄悄放宽成谁都能 camelCase。
    """
    source = """
from xrr_fitter.api import MaterialSpec

class Decorated(MaterialSpec):
    def notAQtMethod(self):
        return None
"""
    tree = ast.parse(source)
    qt_overrides = _qt_overrides(tree)

    assert qt_overrides == set()
    assert [violation.name for violation in _symbol_violations(tree, qt_overrides)] == ["notAQtMethod"]


def test_naming_scans_exactly_the_same_python_files_as_radon() -> None:
    assert set(_python_files()) == set(_radon_python_files())


def test_root_conftest_is_the_only_conftest() -> None:
    conftests = {path.relative_to(ROOT).as_posix() for path in _radon_python_files() if path.name == "conftest.py"}
    assert conftests == {"tests/conftest.py"}
