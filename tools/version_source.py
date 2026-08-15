"""Read a setuptools dynamic version without importing project code."""

from __future__ import annotations

import ast
from pathlib import Path


def _mapping(value: object, message: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(message)
    return value


def _version_attribute(payload: dict[str, object]) -> str:
    project = _mapping(payload.get("project"), "pyproject is missing a project table")
    dynamic = project.get("dynamic", ())
    if not isinstance(dynamic, (list, tuple)) or "version" not in dynamic:
        raise ValueError("project version is not declared as dynamic")
    if "version" in project:
        raise ValueError("project version is declared both statically and dynamically")
    tool = _mapping(payload.get("tool"), "pyproject is missing a tool table")
    setuptools = _mapping(tool.get("setuptools"), "pyproject is missing a setuptools table")
    dynamic_table = _mapping(setuptools.get("dynamic"), "setuptools dynamic metadata is missing")
    version_table = _mapping(dynamic_table.get("version"), "setuptools dynamic version metadata is missing")
    attribute = version_table.get("attr")
    if not isinstance(attribute, str) or not attribute:
        raise ValueError("dynamic project version must use a setuptools attr")
    return attribute


def _attribute_source(root: Path, attribute: str) -> tuple[Path, str]:
    parts = attribute.split(".")
    if len(parts) < 2 or any(not part.isidentifier() for part in parts):
        raise ValueError("dynamic project version attribute is invalid")
    return root / "src" / Path(*parts[:-1]).with_suffix(".py"), parts[-1]


def _assignment(node: ast.stmt) -> tuple[ast.expr | None, ast.expr | None]:
    if isinstance(node, ast.AnnAssign):
        return node.target, node.value
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return node.targets[0], node.value
    return None, None


def _literal_values(source: Path, name: str) -> tuple[object, ...]:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("dynamic project version source is unavailable") from error
    values: list[object] = []
    for node in tree.body:
        target, value = _assignment(node)
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        try:
            values.append(ast.literal_eval(value))
        except (ValueError, TypeError, SyntaxError) as error:
            raise ValueError("dynamic project version must be a literal string") from error
    return tuple(values)


def dynamic_attribute_version(root: Path, payload: dict[str, object]) -> str:
    """Return one non-empty top-level literal assigned to the configured attr."""
    source, name = _attribute_source(root, _version_attribute(payload))
    values = _literal_values(source, name)
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError("dynamic project version must be a single non-empty string")
    return values[0]


def declared_project_version(root: Path, payload: dict[str, object]) -> str:
    """Resolve the one configured project version source.

    A static version remains supported for generic fixture repositories, but it
    cannot coexist with setuptools dynamic metadata: choosing one silently
    would make release identity depend on which caller happened to read it.
    """
    project = _mapping(payload.get("project"), "pyproject is missing a project table")
    dynamic = project.get("dynamic", ())
    has_dynamic = isinstance(dynamic, (list, tuple)) and "version" in dynamic
    has_static = "version" in project
    if has_static and has_dynamic:
        raise ValueError("project version is declared both statically and dynamically")
    if has_dynamic:
        return dynamic_attribute_version(root, payload)
    value = project.get("version")
    if not isinstance(value, str) or not value:
        raise ValueError("project version must be a non-empty string")
    return value
