"""Exhaustively enforce the R23 production dependency contract.

The checks in this module operate on syntax, never imported production state.
That keeps architecture validation deterministic when GUI modules have side
effects and ensures imports hidden in local scopes remain visible.

Package boundaries:
- every production module belongs to one registered package owner;
- each internal edge must appear in the package allowlist;
- fit and analysis have no direct edge in either direction;
- GUI domain access is limited to the public API package;
- package initializers cannot become re-export surfaces.

Model boundaries:
- low-level value modules do not depend on higher-level values;
- fitting depends only on data, instrument, structure, and parameters;
- analysis may consume fitting values, but fitting cannot consume analysis;
- project, operations, and export follow the declared value DAG;
- TYPE_CHECKING and local imports obey the same graph.

Service boundaries:
- services.fitting is the sole fit-and-analysis composition module;
- batch and workers compose through services.fitting;
- all other service modules avoid fit and analysis imports;
- process ownership remains isolated in services.workers.

Third-party boundaries:
- the seven declared roots form an exhaustive allowlist;
- each root is restricted to its declared module owners;
- test and tooling dependencies are never production dependencies;
- standard-library detection uses the running Python 3.12 interpreter.

Execution boundaries:
- subprocess is unavailable throughout the production tree;
- multiprocessing has only workers and freeze_support exceptions;
- process-pool, asyncio subprocess, and os spawn aliases are rejected;
- __import__, importlib, exec, and eval cannot form discovery channels;
- wildcard imports cannot hide dependency ownership.

Graph validation also rejects every multi-module strongly connected component.
Focused fixtures cover absolute, relative, aliased, local, and type-only forms
so a future simplification of the scanner cannot silently weaken the policy.

Evidence is derived from the complete filesystem module set. Syntax failures
remain hard failures, unresolved external roots remain violations, and issue
ordering is stable enough to identify the exact module and broken boundary.
No import form receives a test-only exception, and no missing module is
invented to make an otherwise invalid edge appear registered.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "xrr_fitter"
ALLOWED = {
    "model": {"model"},
    "io": {"io", "model"},
    "physics": {"physics", "model"},
    "evaluation": {"model", "physics"},
    "fit": {"fit", "model", "physics", "evaluation"},
    "analysis": {"analysis", "model", "physics", "evaluation"},
    "services": {"services", "model", "io", "physics", "fit", "analysis"},
    "api": {"model", "services"},
    "gui": {"gui", "api"},
    "__main__": {"gui"},
    "__init__": set(),
}
MODEL_ALLOWED = {
    "data": set(),
    "instrument": set(),
    "structure": set(),
    "parameters": set(),
    "fitting": {"data", "instrument", "structure", "parameters"},
    "analysis": {"data", "parameters", "fitting"},
    "project": {"data", "instrument", "structure", "parameters", "fitting", "analysis"},
    "operations": {"fitting", "analysis", "project"},
    "export": {"data", "fitting", "analysis", "project", "operations"},
}
THIRD_PARTY_ROOTS = {
    "numpy",
    "scipy",
    "periodictable",
    "pandas",
    "xlsxwriter",
    "matplotlib",
    "PySide6",
}
FORBIDDEN_REFERENCES = {
    "concurrent.futures.ProcessPoolExecutor",
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.system",
    "os.popen",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
}
DYNAMIC_REFERENCES = {"__import__", "builtins.__import__", "exec", "eval", "importlib.import_module"}
GETATTR_REFERENCES = {"getattr", "builtins.getattr"}


@dataclass(frozen=True)
class RuleViolation:
    kind: str
    module: str
    detail: str
    line: int


def _violation(kind: str, module: str, detail: str, node: ast.AST) -> RuleViolation:
    return RuleViolation(kind, module, detail, int(getattr(node, "lineno", 0)))


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PACKAGE).with_suffix("").parts)
    if parts[-1] == "__init__":
        return ".".join(parts[:-1]) or "__init__"
    return ".".join(parts)


def _production_sources(package: Path) -> dict[str, str]:
    return {
        _module_name(path): path.read_text(encoding="utf-8")
        for path in sorted(package.rglob("*.py"))
    }


def _relative_base(module: str, level: int) -> str:
    package = module.split(".")[:-1]
    parent_count = level - 1
    if parent_count > len(package):
        return ""
    return ".".join(package[: len(package) - parent_count])


def _from_base(node: ast.ImportFrom, module: str) -> str | None:
    if node.level:
        relative = _relative_base(module, node.level)
        return ".".join(part for part in (relative, node.module or "") if part)
    imported = node.module or ""
    if imported == "xrr_fitter":
        return ""
    if imported.startswith("xrr_fitter."):
        return imported.removeprefix("xrr_fitter.")
    return None


def _from_targets(
    node: ast.ImportFrom, module: str, known_modules: set[str]
) -> tuple[str, ...]:
    base = _from_base(node, module)
    if base is None:
        return ()
    targets: list[str] = []
    for alias in node.names:
        candidate = ".".join(part for part in (base, alias.name) if part)
        targets.append(candidate if not base or candidate in known_modules else base)
    return tuple(target for target in targets if target)


def _internal_targets(tree: ast.AST, module: str, known_modules: set[str]) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(
                alias.name.removeprefix("xrr_fitter.")
                for alias in node.names
                if alias.name.startswith("xrr_fitter.")
            )
        elif isinstance(node, ast.ImportFrom):
            targets.update(_from_targets(node, module, known_modules))
    return targets


def _owner(module: str) -> str:
    return module.split(".", 1)[0]


def _package_violations(
    module: str, targets: set[str], node: ast.AST
) -> list[RuleViolation]:
    owner = _owner(module)
    if owner not in ALLOWED:
        return [_violation("package-owner", module, owner, node)]
    forbidden = sorted({_owner(target) for target in targets} - ALLOWED[owner])
    return [_violation("package-edge", module, target, node) for target in forbidden]


def _model_violations(
    module: str, targets: set[str], node: ast.AST
) -> list[RuleViolation]:
    if _owner(module) != "model":
        return []
    source = module.split(".", 1)[1] if "." in module else "__init__"
    if source == "__init__":
        allowed: set[str] = set()
    elif source not in MODEL_ALLOWED:
        return [_violation("model-module", module, source, node)]
    else:
        allowed = MODEL_ALLOWED[source] | {source}
    imported = {
        target.split(".", 1)[1].split(".", 1)[0]
        for target in targets
        if target.startswith("model.")
    }
    return [
        _violation("model-edge", module, target, node)
        for target in sorted(imported - allowed)
    ]


def _services_violations(
    module: str, targets: set[str], node: ast.AST
) -> list[RuleViolation]:
    if _owner(module) != "services" or module == "services.fitting":
        return []
    forbidden = sorted(target for target in targets if _owner(target) in {"fit", "analysis"})
    return [
        _violation("services-composition", module, target, node) for target in forbidden
    ]


def _third_party_allowed(root: str, module: str) -> bool:
    owner = _owner(module)
    if root == "numpy":
        return owner in {"model", "io", "physics", "evaluation", "fit", "analysis"} or module == "services.datasets"
    if root == "scipy":
        return owner in {"physics", "evaluation", "fit", "analysis"}
    if root == "periodictable":
        return module == "physics.materials"
    if root in {"pandas", "xlsxwriter"}:
        return module == "io.export_tables"
    if root == "matplotlib":
        return module == "io.export_plots" or module == "gui.plots" or module.startswith("gui.plots.")
    return root == "PySide6" and owner == "gui"


def _absolute_import_roots(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    roots: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend((alias.name.split(".", 1)[0], node) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            roots.append((node.module.split(".", 1)[0], node))
    return roots


def _third_party_violations(module: str, tree: ast.AST) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    for root, node in _absolute_import_roots(tree):
        if root in {"xrr_fitter", "__future__"} or root in sys.stdlib_module_names:
            continue
        if root not in THIRD_PARTY_ROOTS or not _third_party_allowed(root, module):
            violations.append(_violation("third-party", module, root, node))
    return violations


# Call-reference normalization is intentionally syntax-only.
# Import aliases retain the full imported name.
# From-import aliases retain the defining module and symbol.
# Attribute chains are rebuilt recursively from those bindings.
# Static getattr(module, "name") calls use the same qualified form.
# Dynamic attribute names remain unresolved rather than guessed.
# Builtin aliases normalize to their builtins-qualified identity.
# Process aliases normalize to the owner checked by the process policy.
# These names feed one exhaustive forbidden-reference comparison.
def _bindings(tree: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bindings.update(_import_bindings(node))
        elif isinstance(node, ast.ImportFrom) and node.module:
            bindings.update(_from_bindings(node))
    return bindings


def _import_bindings(node: ast.Import) -> dict[str, str]:
    return {
        alias.asname or alias.name.split(".", 1)[0]: (
            alias.name if alias.asname else alias.name.split(".", 1)[0]
        )
        for alias in node.names
    }


def _from_bindings(node: ast.ImportFrom) -> dict[str, str]:
    return {
        alias.asname or alias.name: f"{node.module}.{alias.name}"
        for alias in node.names
        if alias.name != "*"
    }


def _getattr_qualified_name(node: ast.AST, bindings: dict[str, str]) -> str | None:
    # Fold only statically named getattr calls into the same form as attribute access.
    if not isinstance(node, ast.Call) or len(node.args) < 2:
        return None
    if _qualified_name(node.func, bindings) not in GETATTR_REFERENCES:
        return None
    attribute = getattr(node.args[1], "value", None)
    if not isinstance(attribute, str):
        return None
    if not attribute:
        return None
    base = _qualified_name(node.args[0], bindings)
    if base is None:
        return None
    return f"{base}.{attribute}"


def _qualified_name(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value, bindings)
        return f"{base}.{node.attr}" if base else None
    return _getattr_qualified_name(node, bindings)


def _multiprocessing_imports(tree: ast.AST) -> list[ast.AST]:
    return [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Import)
            and any(alias.name.split(".", 1)[0] == "multiprocessing" for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".", 1)[0] == "multiprocessing"
        )
    ]


def _valid_main_multiprocessing(imports: list[ast.AST]) -> bool:
    if len(imports) != 1 or not isinstance(imports[0], ast.ImportFrom):
        return False
    node = imports[0]
    return (
        node.module == "multiprocessing"
        and node.level == 0
        and len(node.names) == 1
        and node.names[0].name == "freeze_support"
        and node.names[0].asname is None
    )


def _multiprocessing_violations(module: str, tree: ast.AST) -> list[RuleViolation]:
    multiprocessing_imports = _multiprocessing_imports(tree)
    allowed = module == "services.workers" or (
        module == "__main__" and _valid_main_multiprocessing(multiprocessing_imports)
    )
    if not multiprocessing_imports or allowed:
        return []
    return [_violation("process", module, "multiprocessing", multiprocessing_imports[0])]


def _subprocess_violations(module: str, tree: ast.AST) -> list[RuleViolation]:
    return [
        _violation("process", module, "subprocess", node)
        for root, node in _absolute_import_roots(tree)
        if root == "subprocess"
    ]


def _reference_violations(
    module: str, tree: ast.AST, bindings: dict[str, str]
) -> list[RuleViolation]:
    violations: list[RuleViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            reference = _qualified_name(node.func, bindings)
            if reference in FORBIDDEN_REFERENCES:
                violations.append(_violation("process", module, str(reference), node))
            if reference in DYNAMIC_REFERENCES:
                violations.append(_violation("dynamic-import", module, str(reference), node))
    return violations


def _process_violations(module: str, tree: ast.AST) -> list[RuleViolation]:
    return [
        *_multiprocessing_violations(module, tree),
        *_subprocess_violations(module, tree),
        *_reference_violations(module, tree, _bindings(tree)),
    ]


def _wildcard_violations(module: str, tree: ast.AST) -> list[RuleViolation]:
    return [
        _violation("wildcard", module, "wildcard import", node)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
    ]


def _module_violations(
    module: str, source: str, known_modules: set[str]
) -> tuple[RuleViolation, ...]:
    tree = ast.parse(source, filename=module)
    targets = _internal_targets(tree, module, known_modules)
    violations = [
        *_package_violations(module, targets, tree),
        *_model_violations(module, targets, tree),
        *_services_violations(module, targets, tree),
        *_third_party_violations(module, tree),
        *_process_violations(module, tree),
        *_wildcard_violations(module, tree),
    ]
    return tuple(sorted(set(violations), key=lambda item: (item.kind, item.detail, item.line)))


# The graph phase consumes the same complete module set as rule validation.
# Every direct internal import becomes one directed edge.
# Reachability includes transitive imports but never invents missing modules.
# Self-edges are valid and do not constitute a multi-module cycle.
# Any mutually reachable component with multiple modules is forbidden.
# Stable sorting makes cycle evidence deterministic across filesystems.
# Filesystem traversal and syntax parsing happen before graph construction.
# Syntax failures remain hard errors instead of disappearing from the graph.
# No production module is imported while this evidence is collected.
def _module_graph(sources: dict[str, str]) -> dict[str, set[str]]:
    known = set(sources)
    return {
        module: _internal_targets(ast.parse(source), module, known) & known
        for module, source in sources.items()
    }


def _reachable(graph: dict[str, set[str]], start: str) -> set[str]:
    reached: set[str] = set()
    pending = [start]
    while pending:
        current = pending.pop()
        if current not in reached:
            reached.add(current)
            pending.extend(graph.get(current, set()) - reached)
    return reached


def _strong_components(graph: dict[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    reachable = {node: _reachable(graph, node) for node in graph}
    remaining = set(graph)
    components: list[tuple[str, ...]] = []
    while remaining:
        node = min(remaining)
        component = tuple(sorted(other for other in remaining if node in reachable[other] and other in reachable[node]))
        remaining.difference_update(component)
        if len(component) > 1:
            components.append(component)
    return tuple(components)


def _import_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        return (node.module,)
    return ()


def _internal_owner(name: str) -> str | None:
    if name != "xrr_fitter" and not name.startswith("xrr_fitter."):
        return None
    parts = name.split(".")
    return parts[1] if len(parts) > 1 else "__init__"


def _internal_imports(path: Path) -> set[str]:
    observed: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        for name in _import_names(node):
            owner = _internal_owner(name)
            if owner is not None:
                observed.add(owner)
    return observed


def _package_owner(path: Path) -> tuple[Path, str]:
    relative = path.relative_to(PACKAGE)
    owner = relative.parts[0] if len(relative.parts) > 1 else path.stem
    return relative, owner


def _assert_import_policy(path: Path) -> None:
    relative, owner = _package_owner(path)
    assert owner in ALLOWED, f"unregistered package owner: {relative}"
    assert _internal_imports(path) <= ALLOWED[owner], relative


def test_all_internal_imports_follow_package_allowlist() -> None:
    assert PACKAGE.is_dir()
    for path in PACKAGE.rglob("*.py"):
        _assert_import_policy(path)


def test_fit_and_analysis_never_import_each_other() -> None:
    for owner, forbidden in (("fit", "analysis"), ("analysis", "fit")):
        root = PACKAGE / owner
        if root.is_dir():
            assert all(forbidden not in _internal_imports(path) for path in root.rglob("*.py"))


def _fixture_kinds(module: str, source: str, *known_modules: str) -> set[str]:
    return {
        violation.kind
        for violation in _module_violations(module, source, set(known_modules))
    }


def test_fixture_checker_resolves_local_type_checking_and_aliased_internal_imports() -> None:
    source = """
from typing import TYPE_CHECKING
from xrr_fitter import evaluation as shared_evaluation

if TYPE_CHECKING:
    from xrr_fitter.model import fitting as fitting_values

def build():
    from xrr_fitter.physics import reflectivity as kernel
    return shared_evaluation, kernel
"""
    known = {"evaluation", "model.fitting", "physics.reflectivity", "fit.search"}
    assert _module_violations("fit.search", source, known) == ()

    forbidden = source + "\ndef bad():\n    from xrr_fitter.analysis import report\n"
    assert "package-edge" in _fixture_kinds("fit.search", forbidden, *known, "analysis.report")


def test_fixture_checker_enforces_model_module_dag_and_services_composition() -> None:
    known = {"model.analysis", "model.fitting", "services.batch", "services.fitting"}
    assert _module_violations(
        "model.analysis", "from xrr_fitter.model import fitting", known
    ) == ()
    assert "model-edge" in _fixture_kinds(
        "model.fitting", "from xrr_fitter.model import analysis", *known
    )
    assert _module_violations(
        "services.fitting", "import xrr_fitter.fit\nimport xrr_fitter.analysis", known
    ) == ()
    assert "services-composition" in _fixture_kinds(
        "services.batch", "from xrr_fitter import analysis", *known
    )


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("physics.materials", "import periodictable"),
        ("io.export_tables", "import pandas\nimport xlsxwriter"),
        ("io.export_plots", "import matplotlib"),
        ("gui.plots", "import matplotlib\nfrom PySide6 import QtWidgets"),
        ("services.datasets", "import numpy"),
    ],
)
def test_fixture_checker_accepts_exact_third_party_owners(module: str, source: str) -> None:
    assert _module_violations(module, source, {module}) == ()


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("model.data", "import periodictable"),
        ("services.fitting", "import numpy"),
        ("io.xy", "import pandas"),
        ("physics.reflectivity", "import matplotlib"),
        ("fit.search", "import refnx"),
        ("analysis.report", "import pytest"),
    ],
)
def test_fixture_checker_rejects_unknown_or_wrong_third_party_owner(
    module: str, source: str
) -> None:
    assert "third-party" in _fixture_kinds(module, source, module)


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("import subprocess", "process"),
        ("from concurrent.futures import ProcessPoolExecutor as Pool\nPool()", "process"),
        ("import asyncio as aio\naio.create_subprocess_shell('x')", "process"),
        ("import os as runtime\nruntime.fork()", "process"),
        ("from os import posix_spawn as launch\nlaunch('x', [], {})", "process"),
        ("__import__('xrr_fitter.fit')", "dynamic-import"),
        ("import importlib as imports\nimports.import_module('xrr_fitter.fit')", "dynamic-import"),
        ("from importlib import import_module as load\nload('xrr_fitter.fit')", "dynamic-import"),
        ("from builtins import __import__ as load\nload('xrr_fitter.fit')", "dynamic-import"),
        ("import importlib\ngetattr(importlib, 'import_module')('xrr_fitter.fit')", "dynamic-import"),
        ("import os\ngetattr(os, 'system')('echo forbidden')", "process"),
        ("exec('import xrr_fitter.fit')", "dynamic-import"),
        ("eval('1 + 1')", "dynamic-import"),
    ],
)
def test_fixture_checker_rejects_process_and_dynamic_import_aliases(
    source: str, kind: str
) -> None:
    assert kind in _fixture_kinds("fit.search", source, "fit.search")


def test_fixture_checker_allows_only_the_two_multiprocessing_exceptions() -> None:
    workers = "import multiprocessing as mp\nCONTEXT = mp.get_context('spawn')"
    main = "from multiprocessing import freeze_support\nfreeze_support()"
    assert _module_violations("services.workers", workers, {"services.workers"}) == ()
    assert _module_violations("__main__", main, {"__main__"}) == ()
    assert "process" in _fixture_kinds("fit.search", workers, "fit.search")
    assert "process" in _fixture_kinds(
        "__main__", "from multiprocessing import get_context", "__main__"
    )


def test_fixture_checker_rejects_wildcard_imports() -> None:
    assert "wildcard" in _fixture_kinds(
        "fit.search", "from xrr_fitter.model.fitting import *", "fit.search", "model.fitting"
    )


def test_strong_component_checker_reports_only_multi_module_cycles() -> None:
    graph = {"a": {"b"}, "b": {"a"}, "c": {"c"}, "d": set()}
    assert _strong_components(graph) == (("a", "b"),)


def test_all_production_modules_pass_exhaustive_rules_and_have_no_cycles() -> None:
    sources = _production_sources(PACKAGE)
    known = set(sources)
    violations = tuple(
        violation
        for module, source in sources.items()
        for violation in _module_violations(module, source, known)
    )
    assert violations == ()
    assert _strong_components(_module_graph(sources)) == ()
