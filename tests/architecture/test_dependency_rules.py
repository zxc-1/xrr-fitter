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
- multiprocessing has only workers and the two freeze_support entry points;
- process-pool, asyncio subprocess, and os spawn aliases are rejected;
- __import__, importlib, exec, and eval cannot form discovery channels;
- wildcard imports cannot hide dependency ownership.

Fit evaluation is a top-level shared boundary; fit and analysis may consume it
without importing each other or recreating a second numerical chain.

Graph validation also rejects every multi-module strongly connected component.
Focused fixtures cover absolute, relative, aliased, local, and type-only forms
so a future simplification of the scanner cannot silently weaken the policy.

Evidence is derived from the complete filesystem module set. Syntax failures
remain hard failures, unresolved external roots remain violations, and issue
ordering is stable enough to identify the exact module and broken boundary.
No import form receives a test-only exception, and no missing module is
invented to make an otherwise invalid edge appear registered.

The GUI fixture proves that GUI modules may import the public API while direct
model and service edges remain violations. The concrete GUI tree is required
to be nonempty so the filesystem scan can never pass vacuously. The exhaustive
scan applies the same owner rule to every real GUI import, including imports in
local scopes and type-only branches. No fixture grants GUI a second domain
composition path or exempts a production module from graph validation. The
public facade is therefore the only GUI domain vocabulary.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

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
    "cli": {"cli", "api"},
    "__main__": {"gui", "cli"},
    "__init__": set(),
}
PACKAGE_EDGE_EXCEPTIONS = {
    "io.examples": {"physics.reflectivity", "physics.stack"},
}
# Package exceptions are exact directed module edges, never owner-wide grants.
# The mapping key names the only source module that receives an exception.
# Every mapping value names one complete target module rather than its package.
# Ordinary owner rules remain authoritative for every target outside the set.
# Example generation predates services but must execute canonical physics.
# Moving that calculation into model would invert the declared dependency DAG.
# Copying the calculation into IO would create a second numerical authority.
# Opaque committed curve constants would make regeneration unverifiable.
# The two listed targets are the complete calculation surface needed here.
# Importing another physics module from io.examples is therefore a violation.
# Importing either target from another IO module is also a violation.
# Exceptions do not propagate through aliases, packages, or transitive imports.
# The fixture below proves allowed targets, a wrong target, and a wrong source.
# The exhaustive filesystem check applies the same rule to production modules.
# The fast path-level check only rejects owners with no possible valid edge.
# The AST rule is decisive because it retains each resolved module target.
# Package roots cannot appear in this exception table as shorthand targets.
# The architecture document records the same two edges for human review.
# The graph phase still records both real edges for cycle detection.
# Any future exception requires its own exact mapping and three-way fixture.
# Every model module needs a key here even when it imports no sibling at all.
# An absent key is a model-module violation, not an implicit empty allowance.
# So sld_bands is registered with an empty set: it depends on numpy only.
# The analysis entry then gains sld_bands because it re-exports that value.
MODEL_ALLOWED = {
    "data": set(),
    "instrument": set(),
    "automation": {"data", "instrument"},
    "structure": set(),
    "parameters": set(),
    "constraint_expression": {"parameters"},
    "constraint_resolution": {"constraint_expression", "parameters"},
    "project_parameter_graph": {"parameters"},
    "progress": set(),
    "mcmc_samples": set(),
    "sld_bands": set(),
    "fitting": {"data", "instrument", "structure", "parameters", "progress"},
    "provenance": {"fitting"},
    "analysis": {
        "data",
        "parameters",
        "fitting",
        "mcmc_samples",
        "sld_bands",
    },
    "project": {
        "automation",
        "data",
        "instrument",
        "structure",
        "parameters",
        "fitting",
        "analysis",
        "project_parameter_graph",
    },
    "operations": {"automation", "fitting", "analysis", "project"},
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
    "jsonschema",
    "orsopy",
}
THIRD_PARTY_OWNER_ALLOWLIST = {
    "numpy": {"model", "io", "physics", "evaluation", "fit", "analysis"},
    "scipy": {"physics", "evaluation", "fit", "analysis"},
    "PySide6": {"gui"},
}
THIRD_PARTY_MODULE_ALLOWLIST = {
    "numpy": {"services.datasets", "gui.plots"},
    "periodictable": {"physics.materials"},
    "pandas": {"io.export_tables"},
    "xlsxwriter": {"io.export_tables"},
    "matplotlib": {"io.export_plots", "gui.plots"},
    "jsonschema": {"io.orso"},
    "orsopy": {"io.orso"},
}
THIRD_PARTY_PREFIX_ALLOWLIST = {
    "numpy": ("gui.plots.",),
    "matplotlib": ("gui.plots.",),
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
FIT_RUNTIME_MODULE_TOKENS = {"executor", "ipc", "process", "queue", "worker"}
# Frozen Windows builds need freeze_support() in every process entry point, so
# both the GUI launcher and the headless CLI launcher carry the same exception.
FREEZE_SUPPORT_ENTRY_POINTS = {"__main__", "cli.main"}


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
    return {_module_name(path): path.read_text(encoding="utf-8") for path in sorted(package.rglob("*.py"))}


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


def _from_targets(node: ast.ImportFrom, module: str, known_modules: set[str]) -> tuple[str, ...]:
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
                alias.name.removeprefix("xrr_fitter.") for alias in node.names if alias.name.startswith("xrr_fitter.")
            )
        elif isinstance(node, ast.ImportFrom):
            targets.update(_from_targets(node, module, known_modules))
    return targets


def _owner(module: str) -> str:
    return module.split(".", 1)[0]


def _package_violations(module: str, targets: set[str], node: ast.AST) -> list[RuleViolation]:
    owner = _owner(module)
    if owner not in ALLOWED:
        return [_violation("package-owner", module, owner, node)]
    allowed = ALLOWED[owner]
    exceptions = PACKAGE_EDGE_EXCEPTIONS.get(module, set())
    forbidden = sorted(target for target in targets if _owner(target) not in allowed and target not in exceptions)
    return [_violation("package-edge", module, target, node) for target in forbidden]


def _model_violations(module: str, targets: set[str], node: ast.AST) -> list[RuleViolation]:
    if _owner(module) != "model":
        return []
    source = module.split(".", 1)[1] if "." in module else "__init__"
    if source == "__init__":
        allowed: set[str] = set()
    elif source not in MODEL_ALLOWED:
        return [_violation("model-module", module, source, node)]
    else:
        allowed = MODEL_ALLOWED[source] | {source}
    imported = {target.split(".", 1)[1].split(".", 1)[0] for target in targets if target.startswith("model.")}
    return [_violation("model-edge", module, target, node) for target in sorted(imported - allowed)]


def _services_violations(module: str, targets: set[str], node: ast.AST) -> list[RuleViolation]:
    if _owner(module) != "services" or module == "services.fitting":
        return []
    forbidden = sorted(target for target in targets if _owner(target) in {"fit", "analysis"})
    return [_violation("services-composition", module, target, node) for target in forbidden]


def _third_party_allowed(root: str, module: str) -> bool:
    return bool(
        _owner(module) in THIRD_PARTY_OWNER_ALLOWLIST.get(root, set())
        or module in THIRD_PARTY_MODULE_ALLOWLIST.get(root, set())
        or module.startswith(THIRD_PARTY_PREFIX_ALLOWLIST.get(root, ()))
    )


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
        alias.asname or alias.name.split(".", 1)[0]: (alias.name if alias.asname else alias.name.split(".", 1)[0])
        for alias in node.names
    }


def _from_bindings(node: ast.ImportFrom) -> dict[str, str]:
    return {alias.asname or alias.name: f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*"}


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
        or (isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0] == "multiprocessing")
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
        module in FREEZE_SUPPORT_ENTRY_POINTS and _valid_main_multiprocessing(multiprocessing_imports)
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


def _reference_violations(module: str, tree: ast.AST, bindings: dict[str, str]) -> list[RuleViolation]:
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


def _module_violations(module: str, source: str, known_modules: set[str]) -> tuple[RuleViolation, ...]:
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
    return {module: _internal_targets(ast.parse(source), module, known) & known for module, source in sources.items()}


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
    module = ".".join(relative.with_suffix("").parts)
    exception_owners = {_owner(target) for target in PACKAGE_EDGE_EXCEPTIONS.get(module, set())}
    allowed = ALLOWED[owner] | exception_owners
    assert _internal_imports(path) <= allowed, relative


def test_all_internal_imports_follow_package_allowlist() -> None:
    assert PACKAGE.is_dir()
    for path in PACKAGE.rglob("*.py"):
        _assert_import_policy(path)


def test_fit_and_analysis_never_import_each_other() -> None:
    for owner, forbidden in (("fit", "analysis"), ("analysis", "fit")):
        root = PACKAGE / owner
        if root.is_dir():
            assert all(forbidden not in _internal_imports(path) for path in root.rglob("*.py"))


def test_shared_evaluation_and_fit_have_one_declared_numerical_boundary() -> None:
    required = (
        "evaluation.py",
        "fit/__init__.py",
        "fit/objective.py",
        "fit/parameters.py",
        "fit/problem.py",
        "fit/initialization.py",
        "fit/screening.py",
        "fit/candidates.py",
    )
    assert all((PACKAGE / path).is_file() for path in required)
    assert (PACKAGE / "fit" / "__init__.py").read_bytes() == b""
    assert not (PACKAGE / "fit" / "evaluation.py").exists()
    assert not (PACKAGE / "fit" / "jacobian.py").exists()


def test_fit_defines_no_process_queue_worker_executor_or_ipc_modules() -> None:
    # Fit is a pure calculation domain. Filename tokens prevent a later worker
    # abstraction from being hidden behind an otherwise legal stdlib-only
    # module, while the import check closes the direct in-memory queue path.
    # Process creation itself remains covered by the exhaustive call scanner.
    # Tokens are matched as underscore-delimited filename segments, so domain
    # names such as checkpoint or request are not rejected by substring.
    # This guard intentionally covers process, queue, worker, executor, and IPC
    # ownership names together: all five belong to services in the final graph.
    # The queue import is checked separately because a harmlessly named fit
    # module could otherwise construct a private message channel without ever
    # calling a forbidden process primitive.
    fit_root = PACKAGE / "fit"
    for path in sorted(fit_root.rglob("*.py")):
        module_tokens = set(path.stem.split("_"))
        assert module_tokens.isdisjoint(FIT_RUNTIME_MODULE_TOKENS), path.relative_to(PACKAGE)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_roots = {name.split(".", 1)[0] for node in ast.walk(tree) for name in _import_names(node)}
        assert "queue" not in imported_roots, path.relative_to(PACKAGE)


def _fixture_kinds(module: str, source: str, *known_modules: str) -> set[str]:
    return {violation.kind for violation in _module_violations(module, source, set(known_modules))}


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


def test_fixture_checker_allows_only_examples_to_compose_physics() -> None:
    source = """
from xrr_fitter.physics.reflectivity import instrument_reflectivity
from xrr_fitter.physics.stack import expand_structure
"""
    known = {
        "io.examples",
        "io.export_run",
        "physics.parratt",
        "physics.reflectivity",
        "physics.stack",
    }

    assert _module_violations("io.examples", source, known) == ()
    assert "package-edge" in _fixture_kinds("io.export_run", source, *known)
    assert "package-edge" in _fixture_kinds(
        "io.examples",
        "from xrr_fitter.physics.parratt import parratt_reflectivity",
        *known,
    )


def test_fixture_checker_enforces_model_module_dag_and_services_composition() -> None:
    known = {
        "model.analysis",
        "model.fitting",
        "model.provenance",
        "services.batch",
        "services.fitting",
        "services.fitting_phases.automatic_dataset",
    }
    assert _module_violations("model.analysis", "from xrr_fitter.model import fitting", known) == ()
    assert _module_violations("model.provenance", "from xrr_fitter.model import fitting", known) == ()
    assert "model-edge" in _fixture_kinds("model.fitting", "from xrr_fitter.model import analysis", *known)
    assert "model-edge" in _fixture_kinds("model.fitting", "from xrr_fitter.model import provenance", *known)
    assert _module_violations("services.fitting", "import xrr_fitter.fit\nimport xrr_fitter.analysis", known) == ()
    assert "services-composition" in _fixture_kinds(
        "services.fitting_phases.automatic_dataset",
        "import xrr_fitter.fit\nimport xrr_fitter.analysis",
        *known,
    )
    assert "services-composition" in _fixture_kinds("services.batch", "from xrr_fitter import analysis", *known)


def test_fixture_checker_allows_gui_domain_access_only_through_public_api() -> None:
    known = {"api", "gui.document", "model.project", "services.projects"}
    public = "from xrr_fitter.api import XrrProject, set_workspace_state"

    assert _module_violations("gui.document", public, known) == ()
    assert "package-edge" in _fixture_kinds("gui.document", "from xrr_fitter.model.project import XrrProject", *known)
    assert "package-edge" in _fixture_kinds(
        "gui.document", "from xrr_fitter.services.projects import new_project", *known
    )


def test_fixture_checker_allows_cli_domain_access_only_through_public_api() -> None:
    known = {"api", "cli.commands", "services.fitting", "fit.resume"}
    public = "import xrr_fitter.api as api\n\ndef run(path):\n    return api.load_project(path)\n"

    assert _module_violations("cli.commands", public, known) == ()
    assert "package-edge" in _fixture_kinds(
        "cli.commands", "from xrr_fitter.services.fitting import fit_project", *known
    )
    assert "package-edge" in _fixture_kinds(
        "cli.commands",
        "from xrr_fitter.fit.resume import validate_resume_checkpoint",
        *known,
    )


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("physics.materials", "import periodictable"),
        ("io.export_tables", "import pandas\nimport xlsxwriter"),
        ("io.export_plots", "import matplotlib"),
        ("gui.plots", "import matplotlib\nfrom PySide6 import QtWidgets"),
        ("gui.plots.reflectivity", "import numpy"),
        ("services.datasets", "import numpy"),
        ("io.orso", "from jsonschema import ValidationError"),
        ("io.orso", "import orsopy"),
    ],
)
def test_fixture_checker_accepts_exact_third_party_owners(module: str, source: str) -> None:
    assert _module_violations(module, source, {module}) == ()


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("model.data", "import periodictable"),
        ("services.fitting", "import numpy"),
        ("gui.document", "import numpy"),
        ("io.xy", "import pandas"),
        ("physics.reflectivity", "import matplotlib"),
        ("fit.search", "import refnx"),
        ("analysis.report", "import pytest"),
        ("fit.search", "import jsonschema"),
        ("fit.search", "import orsopy"),
    ],
)
def test_fixture_checker_rejects_unknown_or_wrong_third_party_owner(module: str, source: str) -> None:
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
def test_fixture_checker_rejects_process_and_dynamic_import_aliases(source: str, kind: str) -> None:
    assert kind in _fixture_kinds("fit.search", source, "fit.search")


def test_fixture_checker_allows_only_the_declared_multiprocessing_exceptions() -> None:
    workers = "import multiprocessing as mp\nCONTEXT = mp.get_context('spawn')"
    main = "from multiprocessing import freeze_support\nfreeze_support()"
    assert _module_violations("services.workers", workers, {"services.workers"}) == ()
    for entry_point in sorted(FREEZE_SUPPORT_ENTRY_POINTS):
        assert _module_violations(entry_point, main, {entry_point}) == ()
        assert "process" in _fixture_kinds(entry_point, "from multiprocessing import get_context", entry_point)
    assert "process" in _fixture_kinds("fit.search", workers, "fit.search")
    assert "process" in _fixture_kinds("fit.search", main, "fit.search")


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
    # The fixture-only GUI rule above proves the edge policy in isolation, but
    # it cannot prove that the production scan visited a real GUI module. Keep
    # this non-vacuity check adjacent to the exhaustive scan so deleting the GUI
    # tree cannot turn architecture coverage into an empty successful loop.
    # Requiring both the package initializer and one concrete module also keeps
    # an empty placeholder package from satisfying the release boundary.
    assert {"gui"} < {module for module in known if _owner(module) == "gui"}
    violations = tuple(
        violation for module, source in sources.items() for violation in _module_violations(module, source, known)
    )
    assert violations == ()
    assert _strong_components(_module_graph(sources)) == ()
