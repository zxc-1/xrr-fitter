from __future__ import annotations

import numpy as np
import pytest
from tests.support.drift_cases import drift_case

from xrr_fitter import evaluation
from xrr_fitter.evaluation import encode_physical_vector, values_and_jacobians
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.physics.geometry import expand_geometry, expand_structure_with_jacobian
from xrr_fitter.physics.stack import expand_structure, rebuild_structure


def _drift_problem():
    """Compile the shared linear thickness-drift block (repeats=3, amount=0.1)."""
    return compile_fit_problem(*drift_case())


@pytest.mark.parametrize(
    "name",
    [
        "component.0.drift_scale",
        "component.0.layer.0.thickness_a",
        "component.0.layer.1.thickness_a",
    ],
)
def test_thickness_drift_jacobian_matches_finite_differences(name: str) -> None:
    problem = _drift_problem()
    index = next(pos for pos, coord in enumerate(problem.variables) if coord.name == name)
    unit = encode_physical_vector(problem, {})
    stack = evaluation.expanded_structure_jacobian(problem, unit)
    # repeats=3 * 2 layers + fronting + backing pads = 8 media.
    assert stack.stack.thickness_a.size == 8
    # A drifted block breaks bit-identical repetition: no matrix-power fast path.
    assert stack.stack.periodic_spans == ()

    step = 1e-6
    forward = unit.copy()
    forward[index] += step
    backward = unit.copy()
    backward[index] -= step
    high = evaluation.expanded_structure_jacobian(problem, forward).stack
    low = evaluation.expanded_structure_jacobian(problem, backward).stack

    # Per-copy thickness tangents must track base*(1 + scale*coeff), so copies
    # 1..2 differ from the base cell. A tangent read from the base source name
    # would collapse them and diverge from the finite-difference slope.
    np.testing.assert_allclose(
        stack.thickness_jacobian[:, index],
        (high.thickness_a - low.thickness_a) / (2.0 * step),
        rtol=1e-6,
        atol=1e-9,
    )
    # Thickness drift never perturbs per-copy SLD.
    np.testing.assert_allclose(
        stack.sld_jacobian[:, index],
        (high.sld_a2 - low.sld_a2) / (2.0 * step),
        rtol=1e-6,
        atol=1e-16,
    )


def test_drift_expansion_aligns_across_all_three_paths() -> None:
    problem = _drift_problem()
    unit = encode_physical_vector(problem, {})
    values, value_jacobians = values_and_jacobians(problem, unit)
    rebuilt = rebuild_structure(problem.structure, values)
    wavelength = problem.data.beam.effective_wavelength_a

    primal = expand_structure(rebuilt, wavelength, values)
    differentiable = expand_structure_with_jacobian(
        problem.structure,
        values,
        value_jacobians,
        wavelength,
        len(problem.variables),
    )
    geometry = expand_geometry(rebuilt, len(problem.variables), value_jacobians, values)

    # Linear thickness drift, amount ~= 0.1, coeffs (0, 1, 2): copies scale the cell.
    np.testing.assert_allclose(primal.thickness_a, [0.0, 20.0, 500.0, 22.0, 550.0, 24.0, 600.0, 0.0])
    # All three expanders must consume the identical per-copy value map.
    np.testing.assert_array_equal(primal.thickness_a, differentiable.stack.thickness_a)
    np.testing.assert_array_equal(primal.thickness_a, geometry.thickness_a)
    assert primal.periodic_spans == ()
    assert len(geometry.interface_names) == primal.thickness_a.size - 1
