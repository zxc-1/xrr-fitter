"""Focused tests for the evaluation facade/implementation boundary."""

from __future__ import annotations

import ast

from tests.architecture.test_dependency_rules import (
    EVALUATION_FACADE_MODULE,
    EVALUATION_IMPLEMENTATION_MODULES,
    PACKAGE,
    _fixture_kinds,
    _internal_targets,
    _module_violations,
    _production_sources,
)


def test_geometry_implementation_is_consumed_only_by_the_facade() -> None:
    known = {"evaluation", "evaluation_geometry", "fit.search", "analysis.report"}
    source = "from xrr_fitter import evaluation_geometry\n"
    assert _module_violations("evaluation", source, known) == ()
    for consumer in ("fit.search", "analysis.report"):
        assert "evaluation-boundary" in _fixture_kinds(consumer, source, *known)


def test_production_geometry_reverse_consumers_are_exact() -> None:
    sources = _production_sources(PACKAGE)
    known = set(sources)
    consumers = {
        module
        for module, source in sources.items()
        if _internal_targets(ast.parse(source), module, known) & EVALUATION_IMPLEMENTATION_MODULES
    }
    assert consumers == {EVALUATION_FACADE_MODULE}
