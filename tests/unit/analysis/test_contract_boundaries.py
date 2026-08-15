"""Focused analysis type and provenance boundary contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from typing import get_type_hints

import pytest
from tests.unit.analysis.test_report import _candidate, _problem

from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.provenance import bootstrap_provenance_sha256


@pytest.mark.parametrize(
    ("module_name", "function_names"),
    (
        (
            "binary_profiles",
            (
                "binary_derived_profiles",
                "decode_binary_coordinate",
                "build_binary_profile",
            ),
        ),
        ("bootstrap", ("bootstrap_problem_local",)),
        ("classification", ("classify_result_with_evidence",)),
        (
            "derivatives",
            (
                "objective_gradient",
                "objective_information",
                "physical_parameter_jacobian",
            ),
        ),
        ("diagnostics", ("ordered_fit_residuals", "diagnose_residual_patterns")),
        ("mcmc", ("map_problem_samples", "mcmc_boundary_hits", "run_problem_mcmc")),
        (
            "profiles",
            ("build_problem_profile", "select_profile_names", "recover_profile_basin"),
        ),
        ("report", ("build_uncertainty_report", "analyze_search_result")),
    ),
)
def test_public_analysis_entries_require_the_typed_context(
    module_name: str,
    function_names: tuple[str, ...],
) -> None:
    module = import_module(f"xrr_fitter.analysis.{module_name}")

    for name in function_names:
        assert get_type_hints(getattr(module, name))["problem"] is FitEvaluationContext


@dataclass(frozen=True)
class _UnsupportedBootstrapPayload:
    opaque: object


def test_provenance_rejects_unsupported_payload_values() -> None:
    problem = _problem()
    candidate = _candidate(problem, "E-0")

    with pytest.raises(TypeError, match="unsupported provenance value"):
        bootstrap_provenance_sha256(
            problem,
            candidate,
            _UnsupportedBootstrapPayload(object()),
        )


def test_provenance_omits_default_constraint_marker_but_binds_driven_definitions() -> None:
    provenance = import_module("xrr_fitter.model.provenance")
    definition = _problem().parameter_definitions[0]

    baseline = provenance._identity_value(definition)
    constrained = provenance._identity_value(replace(definition, constrained=True))

    assert "constrained" not in baseline
    assert constrained["constrained"] is True
