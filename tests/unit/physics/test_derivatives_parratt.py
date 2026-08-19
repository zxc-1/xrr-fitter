from __future__ import annotations

from tests.unit.physics.derivative_cases import *


def test_quadrature_jacobian_rejects_nonfinite_queries_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="query"):
            smear_with_widths_jacobian(
                np.array([1e308]),
                np.zeros((1, 1)),
                np.array([1e308]),
                np.zeros((1, 1)),
                lambda query, query_jacobian: (
                    np.ones_like(query),
                    np.zeros(query.shape + (1,)),
                ),
            )

    assert not any(item.category is RuntimeWarning for item in caught)


def test_quadrature_jacobian_normalizes_constant_extreme_tangent_without_overflow() -> None:
    constant = 1.7e308

    def extreme_tangent(query, _query_jacobian):
        return np.ones_like(query), np.full(query.shape + (1,), constant)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values, jacobian = smear_with_widths_jacobian(
            np.array([0.1]),
            np.zeros((1, 1)),
            np.array([0.01]),
            np.zeros((1, 1)),
            extreme_tangent,
        )

    np.testing.assert_array_equal(values, np.ones(1))
    assert np.all(np.isfinite(jacobian))
    assert jacobian[0, 0] == pytest.approx(constant, rel=2e-16)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_all_zero_width_jacobian_reuses_validated_input_arrays() -> None:
    samples = np.linspace(0.02, 0.4, 80)
    sample_jacobian = np.ones((samples.size, 1))
    observed: list[tuple[bool, bool]] = []

    def exact(query, query_jacobian):
        observed.append(
            (
                np.shares_memory(query, samples),
                np.shares_memory(query_jacobian, sample_jacobian),
            )
        )
        return query**2, 2 * query[..., None] * query_jacobian

    smear_with_widths_jacobian(
        samples,
        sample_jacobian,
        np.zeros(samples.size),
        np.zeros_like(sample_jacobian),
        exact,
    )

    assert observed == [(True, True)]


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
    q_jacobian = np.zeros((q.size, 3))
    q_jacobian[:, 0] = 0.4
    thickness_jacobian = np.zeros((3, 3))
    sld_jacobian = np.zeros((3, 3), complex)
    sld_jacobian[1, 1] = 0.7e-6 + 0.2e-6j
    roughness_jacobian = np.zeros((2, 3))
    roughness_jacobian[0, 2] = 0.3
    _, actual = parratt_reflectivity_jacobian(
        q, stack, q_jacobian, thickness_jacobian, sld_jacobian, roughness_jacobian
    )
    eps = 1e-6
    expected = np.empty_like(actual)
    expected[:, 0] = (parratt_reflectivity(q + eps * 0.4, stack) - parratt_reflectivity(q - eps * 0.4, stack)) / (
        2 * eps
    )
    eps = 1e-4
    delta_sld = eps * (0.7e-6 + 0.2e-6j)
    plus_sld = stack.sld_a2.copy()
    plus_sld[1] += delta_sld
    minus_sld = stack.sld_a2.copy()
    minus_sld[1] -= delta_sld
    expected[:, 1] = (
        parratt_reflectivity(q, SlabStack(stack.thickness_a, plus_sld, stack.roughness_a))
        - parratt_reflectivity(q, SlabStack(stack.thickness_a, minus_sld, stack.roughness_a))
    ) / (2 * eps)
    plus_roughness = stack.roughness_a.copy()
    plus_roughness[0] += eps * 0.3
    minus_roughness = stack.roughness_a.copy()
    minus_roughness[0] -= eps * 0.3
    expected[:, 2] = (
        parratt_reflectivity(q, SlabStack(stack.thickness_a, stack.sld_a2, plus_roughness))
        - parratt_reflectivity(q, SlabStack(stack.thickness_a, stack.sld_a2, minus_roughness))
    ) / (2 * eps)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-10)
