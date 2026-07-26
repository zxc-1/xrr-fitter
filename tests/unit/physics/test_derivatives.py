from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.model.structure import PeriodicSpan, SlabStack
from xrr_fitter.physics.derivatives import parratt_reflectivity_jacobian, smear_with_widths_jacobian


def _tangents(stack: SlabStack, q: np.ndarray, count: int = 3):
    q_jac = np.zeros((q.size, count)); q_jac[:, 0] = 1e-3
    thickness_jac = np.zeros((stack.thickness_a.size, count)); thickness_jac[1:-1:2, 1] = 0.7
    sld_jac = np.zeros((stack.sld_a2.size, count), complex); sld_jac[2:-1:2, 2] = 1e-7 + 2e-8j
    roughness_jac = np.zeros((stack.roughness_a.size, count)); roughness_jac[:, 0] = 0.03
    return q_jac, thickness_jac, sld_jac, roughness_jac


def test_periodic_mobius_jacobian_matches_expanded_recurrence() -> None:
    repeats = 8
    thickness = np.array([0] + [28, 42] * repeats + [0], float)
    sld = np.array([0j] + [55e-6 + 1.4e-6j, 20e-6 + 0.2e-6j] * repeats + [24e-6 + 0.3e-6j])
    roughness = np.array([2.5, 4] + [3, 4] * (repeats - 1) + [5], float)
    expanded = SlabStack(thickness, sld, roughness)
    periodic = SlabStack(thickness, sld, roughness, (PeriodicSpan(1, 2, repeats),))
    q = np.r_[np.geomspace(2e-4, 0.03, 20), np.linspace(0.03, 0.8, 30)]
    tangents = _tangents(expanded, q)
    expected = parratt_reflectivity_jacobian(q, expanded, *tangents)
    actual = parratt_reflectivity_jacobian(q, periodic, *tangents)
    np.testing.assert_allclose(actual[0], expected[0], rtol=5e-12, atol=2e-14)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-9, atol=2e-10)


def test_periodic_jacobian_falls_back_for_nonrepeating_cell_tangents() -> None:
    from xrr_fitter.physics.parratt import parratt_reflectivity

    repeats = 4
    thickness = np.array([0] + [28, 42] * repeats + [0], float)
    sld = np.array([0j] + [55e-6 + 1.4e-6j, 20e-6 + 0.2e-6j] * repeats + [24e-6 + 0.3e-6j])
    roughness = np.array([2.5, 4] + [3, 4] * (repeats - 1) + [5], float)
    expanded = SlabStack(thickness, sld, roughness)
    periodic = SlabStack(thickness, sld, roughness, (PeriodicSpan(1, 2, repeats),))
    q = np.linspace(0.01, 0.5, 60)
    q_jacobian = np.zeros((q.size, 3))
    thickness_jacobian = np.zeros((thickness.size, 3)); thickness_jacobian[5, 0] = 0.7
    sld_jacobian = np.zeros((sld.size, 3), complex); sld_jacobian[7, 1] = 1.1e-7 + 0.4e-8j
    roughness_jacobian = np.zeros((roughness.size, 3)); roughness_jacobian[6, 2] = 0.03
    tangents = q_jacobian, thickness_jacobian, sld_jacobian, roughness_jacobian

    expected = parratt_reflectivity_jacobian(q, expanded, *tangents)
    actual = parratt_reflectivity_jacobian(q, periodic, *tangents)

    np.testing.assert_array_equal(actual[0], parratt_reflectivity(q, periodic))
    np.testing.assert_allclose(actual[0], expected[0], rtol=5e-12, atol=2e-14)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-9, atol=2e-10)


def test_parratt_jacobian_supports_zero_parameters() -> None:
    from xrr_fitter.physics.parratt import parratt_reflectivity

    thickness = [0, 28, 42, 28, 42, 0]
    sld = [0j, 55e-6 + 1.4e-6j, 20e-6 + 0.2e-6j, 55e-6 + 1.4e-6j, 20e-6 + 0.2e-6j, 24e-6 + 0.3e-6j]
    roughness = [2.5, 4, 3, 4, 5]
    stacks = (
        SlabStack(thickness, sld, roughness),
        SlabStack(thickness, sld, roughness, (PeriodicSpan(1, 2, 2),)),
    )
    q = np.linspace(0.01, 0.5, 20)
    for stack in stacks:
        primal, jacobian = parratt_reflectivity_jacobian(
            q,
            stack,
            np.empty((q.size, 0)),
            np.empty((stack.thickness_a.size, 0)),
            np.empty((stack.sld_a2.size, 0), complex),
            np.empty((stack.roughness_a.size, 0)),
        )
        np.testing.assert_array_equal(primal, parratt_reflectivity(q, stack))
        assert jacobian.shape == (q.size, 0)


def test_analytic_tangent_matches_centered_difference() -> None:
    from xrr_fitter.physics.parratt import parratt_reflectivity
    q = np.linspace(0.01, 0.3, 80)
    stack = SlabStack([0, 80, 0], [0j, 25e-6 + 0.2e-6j, 20e-6 + 0.1e-6j], [2, 3])
    tangents = (np.zeros((q.size, 1)), np.array([[0], [1], [0]], float), np.zeros((3, 1), complex), np.zeros((2, 1)))
    primal, jacobian = parratt_reflectivity_jacobian(q, stack, *tangents)
    eps = 1e-4
    plus = SlabStack([0, 80 + eps, 0], stack.sld_a2, stack.roughness_a)
    minus = SlabStack([0, 80 - eps, 0], stack.sld_a2, stack.roughness_a)
    finite = (parratt_reflectivity(q, plus) - parratt_reflectivity(q, minus)) / (2 * eps)
    np.testing.assert_allclose(primal, parratt_reflectivity(q, stack), rtol=1e-13)
    np.testing.assert_allclose(jacobian[:, 0], finite, rtol=2e-7, atol=2e-10)


def test_quadrature_jacobian_uses_primal_order_selection_before_tangent_pass() -> None:
    samples = np.linspace(0.02, 0.4, 80)
    primal_calls, tangent_calls = [], []
    def primal(query):
        primal_calls.append(query.shape)
        return query**2
    def tangent(query, query_jacobian):
        tangent_calls.append(query.shape)
        return query**2, 2 * query[..., None] * query_jacobian
    values, jacobian = smear_with_widths_jacobian(samples, np.ones((samples.size, 1)), np.full(samples.size, 0.002), np.zeros((samples.size, 1)), tangent, primal_function=primal)
    np.testing.assert_allclose(values, samples**2 + 0.002**2)
    np.testing.assert_allclose(jacobian[:, 0], 2 * samples)
    assert {shape[1] for shape in primal_calls} == {17, 33}
    assert {shape[1] for shape in tangent_calls} == {33}


def test_adaptive_jacobian_evaluates_sixty_five_nodes_only_for_unresolved_points() -> None:
    calls = []
    def needle(query, query_jacobian):
        calls.append(query.shape)
        values = np.exp(-4e6 * (query - 0.02137) ** 2)
        derivative = -8e6 * (query - 0.02137) * values
        return values, derivative[..., None] * query_jacobian
    smear_with_widths_jacobian(
        np.array([0.02, 0.2]),
        np.ones((2, 1)),
        np.full(2, 0.011),
        np.zeros((2, 1)),
        needle,
    )
    assert calls == [(2, 17), (2, 33), (1, 65)]


def test_quadrature_jacobian_chunks_large_query_grids() -> None:
    samples = np.linspace(0.02, 0.4, 600)
    calls = []
    def smooth(query, query_jacobian):
        calls.append(query.shape)
        return query**2, 2 * query[..., None] * query_jacobian
    values, jacobian = smear_with_widths_jacobian(
        samples,
        np.ones((samples.size, 1)),
        np.full(samples.size, 0.002),
        np.zeros((samples.size, 1)),
        smooth,
    )
    np.testing.assert_allclose(values, samples**2 + 0.002**2, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(jacobian[:, 0], 2 * samples, rtol=2e-13, atol=2e-15)
    assert max(np.prod(shape) for shape in calls) <= 1024


@pytest.mark.parametrize(
    "widths",
    [np.zeros(2), np.array([0.0, 0.002])],
    ids=("all-zero", "mixed"),
)
@pytest.mark.parametrize(
    "invalid",
    ("value-shape", "tangent-shape", "value-nonfinite", "tangent-nonfinite"),
)
def test_zero_width_jacobian_validates_differentiable_callback(
    widths: np.ndarray,
    invalid: str,
) -> None:
    samples = np.array([0.1, 0.2])
    sample_jacobian = np.ones((2, 1))

    def callback(query, query_jacobian):
        if query.ndim > 1:
            return query**2, 2 * query[..., None] * query_jacobian
        values = query**2
        tangent = 2 * query[:, None] * query_jacobian
        if invalid == "value-shape":
            values = np.array(1.0)
        elif invalid == "tangent-shape":
            tangent = np.array(1.0)
        elif invalid == "value-nonfinite":
            values[0] = np.nan
        else:
            tangent[0, 0] = np.inf
        return values, tangent

    with pytest.raises(ValueError, match="differentiable function returned invalid values"):
        smear_with_widths_jacobian(
            samples,
            sample_jacobian,
            widths,
            np.zeros_like(sample_jacobian),
            callback,
        )


def test_parratt_jacobian_tracks_q_complex_sld_and_roughness_tangents() -> None:
    from xrr_fitter.physics.parratt import parratt_reflectivity
    q = np.linspace(0.01, 0.3, 80)
    stack = SlabStack([0, 80, 0], [0j, 25e-6 + 0.2e-6j, 20e-6 + 0.1e-6j], [2, 3])
    q_jacobian = np.zeros((q.size, 3)); q_jacobian[:, 0] = 0.4
    thickness_jacobian = np.zeros((3, 3))
    sld_jacobian = np.zeros((3, 3), complex); sld_jacobian[1, 1] = 0.7e-6 + 0.2e-6j
    roughness_jacobian = np.zeros((2, 3)); roughness_jacobian[0, 2] = 0.3
    _, actual = parratt_reflectivity_jacobian(q, stack, q_jacobian, thickness_jacobian, sld_jacobian, roughness_jacobian)
    eps = 1e-6
    expected = np.empty_like(actual)
    expected[:, 0] = (parratt_reflectivity(q + eps * 0.4, stack) - parratt_reflectivity(q - eps * 0.4, stack)) / (2 * eps)
    eps = 1e-4
    delta_sld = eps * (0.7e-6 + 0.2e-6j)
    plus_sld = stack.sld_a2.copy(); plus_sld[1] += delta_sld
    minus_sld = stack.sld_a2.copy(); minus_sld[1] -= delta_sld
    expected[:, 1] = (parratt_reflectivity(q, SlabStack(stack.thickness_a, plus_sld, stack.roughness_a)) - parratt_reflectivity(q, SlabStack(stack.thickness_a, minus_sld, stack.roughness_a))) / (2 * eps)
    plus_roughness = stack.roughness_a.copy(); plus_roughness[0] += eps * 0.3
    minus_roughness = stack.roughness_a.copy(); minus_roughness[0] -= eps * 0.3
    expected[:, 2] = (parratt_reflectivity(q, SlabStack(stack.thickness_a, stack.sld_a2, plus_roughness)) - parratt_reflectivity(q, SlabStack(stack.thickness_a, stack.sld_a2, minus_roughness))) / (2 * eps)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-10)
