from __future__ import annotations

from tests.unit.physics.derivative_cases import *


def test_parratt_jacobian_rejects_nonfinite_nevot_croce_path() -> None:
    stack = SlabStack(
        [0, 1, 1, 0],
        [
            0j,
            0.09009274 + 5.25772738e-05j,
            0.08972989 + 3.62374192e-04j,
            0j,
        ],
        [0, 30, 0],
    )
    qz = np.asarray([0.011774154985086273])
    with pytest.raises(FloatingPointError, match="finite|Nevot-Croce"):
        parratt_reflectivity_jacobian(
            qz,
            stack,
            np.zeros((1, 1)),
            np.zeros((4, 1)),
            np.zeros((4, 1), complex),
            np.zeros((3, 1)),
        )


def test_parratt_jacobian_keeps_finite_large_interface_cancellation() -> None:
    stack = SlabStack(
        [0.0, 7.50344687, 1259.19309769, 3332.72786241, 0.0],
        [
            1.13524689e-05j,
            1.43978976e-04 + 1.47957432e-05j,
            4.03322613e-05 + 6.32290605e-06j,
            7.81663285e-05 + 2.01335130e-06j,
            -1.46866469e-04 + 7.41519866e-06j,
        ],
        [1.92383095, 3.32467999, 595.24373189, 1101.54876673],
    )
    values, jacobian = parratt_reflectivity_jacobian(
        np.array([7.14770265e-06]),
        stack,
        np.array([[0.05762203]]),
        np.array([[-1488.34246521], [1626.3788172], [-725.50776185], [-109.54522831], [479.25181699]]),
        np.zeros((5, 1), complex),
        np.array([[54.6641023], [977.02279696], [2227.28592803], [-1308.22801134]]),
    )

    np.testing.assert_allclose(values, [0.99974673], atol=5e-9)
    np.testing.assert_allclose(jacobian, [[-2.19246616]], atol=5e-9)


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
    thickness_jacobian = np.zeros((thickness.size, 3))
    thickness_jacobian[5, 0] = 0.7
    sld_jacobian = np.zeros((sld.size, 3), complex)
    sld_jacobian[7, 1] = 1.1e-7 + 0.4e-8j
    roughness_jacobian = np.zeros((roughness.size, 3))
    roughness_jacobian[6, 2] = 0.03
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


@pytest.mark.parametrize("periodic", (False, True))
def test_parratt_jacobian_evaluates_only_active_tangent_columns(monkeypatch, periodic) -> None:
    module = import_module("xrr_fitter.physics.derivatives")
    stack = SlabStack(
        [0, 28, 42, 28, 42, 0],
        [
            0j,
            55e-6 + 1.4e-6j,
            20e-6 + 0.2e-6j,
            55e-6 + 1.4e-6j,
            20e-6 + 0.2e-6j,
            24e-6 + 0.3e-6j,
        ],
        [2.5, 4, 3, 4, 5],
        (PeriodicSpan(1, 2, 2),) if periodic else (),
    )
    q = np.linspace(0.01, 0.3, 80)
    q_jacobian = np.zeros((q.size, 4))
    q_jacobian[:, 0] = 0.4
    thickness_jacobian = np.zeros((6, 4))
    thickness_jacobian[[1, 3], 2] = 1.0
    sld_jacobian = np.zeros((6, 4), complex)
    roughness_jacobian = np.zeros((5, 4))
    observed_counts: list[int] = []
    target = "_periodic_jacobian" if periodic else "_standard_jacobian"
    original = getattr(module, target)

    def audited(stack_value, inputs):
        observed_counts.append(inputs.parameter_count)
        return original(stack_value, inputs)

    monkeypatch.setattr(module, target, audited)
    primal, jacobian = module.parratt_reflectivity_jacobian(
        q,
        stack,
        q_jacobian,
        thickness_jacobian,
        sld_jacobian,
        roughness_jacobian,
    )

    assert observed_counts == [2]
    assert primal.shape == q.shape
    assert jacobian.shape == (q.size, 4)
    np.testing.assert_array_equal(jacobian[:, (1, 3)], 0.0)


def test_periodic_tangents_reuse_identical_expanded_optics() -> None:
    module = import_module("xrr_fitter.physics.derivatives")
    stack = SlabStack(
        [0, 28, 42, 28, 42, 0],
        [
            0j,
            55e-6 + 1.4e-6j,
            20e-6 + 0.2e-6j,
            55e-6 + 1.4e-6j,
            20e-6 + 0.2e-6j,
            24e-6 + 0.3e-6j,
        ],
        [2.5, 4, 3, 4, 5],
        (PeriodicSpan(1, 2, 2),),
    )
    q = np.linspace(0.01, 0.3, 80)
    tangents = _tangents(stack, q)
    inputs = module._inputs(q, stack, *tangents)
    optics = module._PeriodicTangents(stack, inputs)

    first_kz = optics.kz_at(1)
    repeated_kz = optics.kz_at(3)
    first_interface = optics.interface_at(1)
    repeated_interface = optics.interface_at(3)

    assert repeated_kz[0] is first_kz[0]
    assert repeated_kz[1] is first_kz[1]
    assert repeated_interface[0] is first_interface[0]
    assert repeated_interface[1] is first_interface[1]
