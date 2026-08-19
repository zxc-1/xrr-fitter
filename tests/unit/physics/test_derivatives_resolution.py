from __future__ import annotations

from tests.unit.physics.derivative_cases import *


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


def test_compose_batches_periodic_tangent_product_rule(monkeypatch) -> None:
    module = import_module("xrr_fitter.physics.derivatives")
    rng = np.random.default_rng(20240607)

    def complex_values(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    left = complex_values(20, 2, 2)
    right = complex_values(20, 2, 2)
    left_tangent = complex_values(20, 2, 2, 6)
    right_tangent = complex_values(20, 2, 2, 6)
    expected_matrix = np.empty_like(left)
    expected_tangent = np.empty_like(left_tangent)
    for row in range(2):
        for column in range(2):
            expected_matrix[:, row, column] = (
                left[:, row, 0] * right[:, 0, column] + left[:, row, 1] * right[:, 1, column]
            )
            expected_tangent[:, row, column] = (
                left_tangent[:, row, 0] * right[:, 0, column, None]
                + left[:, row, 0, None] * right_tangent[:, 0, column]
                + left_tangent[:, row, 1] * right[:, 1, column, None]
                + left[:, row, 1, None] * right_tangent[:, 1, column]
            )
    expected = module._normalize(expected_matrix, expected_tangent)
    calls: list[str] = []
    original_einsum = np.einsum

    def audited_einsum(subscripts, *operands, **kwargs):
        calls.append(subscripts)
        return original_einsum(subscripts, *operands, **kwargs)

    monkeypatch.setattr(np, "einsum", audited_einsum)
    actual = module._compose(left, left_tangent, right, right_tangent)

    assert calls == ["qijp,qjk->qikp", "qij,qjkp->qikp"]
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-14, atol=1e-16)


def test_quadrature_jacobian_reuses_fine_tangent_values_for_order_selection() -> None:
    samples = np.linspace(0.02, 0.4, 80)
    primal_calls, tangent_calls = [], []

    def primal(query):
        primal_calls.append(query.shape)
        return query**2

    def tangent(query, query_jacobian):
        tangent_calls.append(query.shape)
        return query**2, 2 * query[..., None] * query_jacobian

    values, jacobian = smear_with_widths_jacobian(
        samples,
        np.ones((samples.size, 1)),
        np.full(samples.size, 0.002),
        np.zeros((samples.size, 1)),
        tangent,
        primal_function=primal,
    )
    np.testing.assert_allclose(values, samples**2 + 0.002**2)
    np.testing.assert_allclose(jacobian[:, 0], 2 * samples)
    assert {shape[1] for shape in primal_calls} == {17}
    assert {shape[1] for shape in tangent_calls} == {33}


def test_reflected_quadrature_has_zero_sample_derivative_at_the_origin() -> None:
    values, jacobian = smear_with_widths_jacobian(
        np.array([0.0]),
        np.ones((1, 1)),
        np.array([0.02]),
        np.zeros((1, 1)),
        lambda query, query_jacobian: (query, query_jacobian),
        primal_function=lambda query: query,
    )

    assert values[0] > 0.0
    assert jacobian[0, 0] == pytest.approx(0.0, abs=1e-15)


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
    assert len(calls) <= 8
    assert max(np.prod(shape) for shape in calls) <= 4096
