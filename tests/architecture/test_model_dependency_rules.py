from __future__ import annotations

import pytest
from tests.architecture.test_dependency_rules import (
    _fixture_kinds,
    _module_violations,
)

MODEL_KNOWN = {
    "model.analysis",
    "model.constraint_expression",
    "model.constraint_resolution",
    "model.fitting",
    "model.parameters",
    "model.project",
    "model.project_parameter_graph",
    "model.provenance",
}

ALLOWED_MODEL_EDGES = (
    ("model.analysis", "from xrr_fitter.model import fitting"),
    ("model.constraint_expression", "from xrr_fitter.model import parameters"),
    (
        "model.constraint_resolution",
        "from xrr_fitter.model import constraint_expression, parameters",
    ),
    ("model.project_parameter_graph", "from xrr_fitter.model import parameters"),
    ("model.project", "from xrr_fitter.model import project_parameter_graph"),
    ("model.provenance", "from xrr_fitter.model import fitting"),
)

FORBIDDEN_MODEL_EDGES = (
    ("model.parameters", "from xrr_fitter.model import constraint_expression"),
    (
        "model.constraint_expression",
        "from xrr_fitter.model import constraint_resolution",
    ),
    ("model.parameters", "from xrr_fitter.model import project_parameter_graph"),
    ("model.fitting", "from xrr_fitter.model import analysis"),
    ("model.fitting", "from xrr_fitter.model import provenance"),
)


@pytest.mark.parametrize(("source", "statement"), ALLOWED_MODEL_EDGES)
def test_fixture_checker_allows_declared_model_edges(
    source: str,
    statement: str,
) -> None:
    assert _module_violations(source, statement, MODEL_KNOWN) == ()


@pytest.mark.parametrize(("source", "statement"), FORBIDDEN_MODEL_EDGES)
def test_fixture_checker_rejects_reverse_model_edges(
    source: str,
    statement: str,
) -> None:
    assert "model-edge" in _fixture_kinds(source, statement, *MODEL_KNOWN)
