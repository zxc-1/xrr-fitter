"""Focused tests for the evaluation facade/implementation boundary."""

from __future__ import annotations

import ast

from tests.architecture.evaluation_policy import (
    EVALUATION_BOUNDARY_MODULES,
    EVALUATION_FACADE_MODULE,
    EVALUATION_IMPLEMENTATION_MODULES,
)
from tests.architecture.test_dependency_rules import (
    PACKAGE,
    _fixture_kinds,
    _internal_targets,
    _module_violations,
    _production_sources,
)


def test_evaluation_implementation_is_consumed_only_inside_the_boundary() -> None:
    known = {
        *EVALUATION_BOUNDARY_MODULES,
        "fit.search",
        "analysis.report",
    }
    source = "from xrr_fitter import evaluation_geometry\n"
    assert _module_violations("evaluation", source, known) == ()
    assert _module_violations("evaluation_model", source, known) == ()
    for consumer in ("fit.search", "analysis.report"):
        assert "evaluation-boundary" in _fixture_kinds(consumer, source, *known)


def test_production_implementation_reverse_consumers_stay_inside_boundary() -> None:
    sources = _production_sources(PACKAGE)
    known = set(sources)
    consumers = {
        module
        for module, source in sources.items()
        if _internal_targets(ast.parse(source), module, known) & EVALUATION_IMPLEMENTATION_MODULES
    }
    assert EVALUATION_FACADE_MODULE in consumers
    assert consumers <= EVALUATION_BOUNDARY_MODULES


def test_implementation_modules_do_not_import_the_facade() -> None:
    sources = _production_sources(PACKAGE)
    known = set(sources)
    for module in EVALUATION_IMPLEMENTATION_MODULES:
        targets = _internal_targets(ast.parse(sources[module]), module, known)
        assert EVALUATION_FACADE_MODULE not in targets
