from __future__ import annotations

from tests.unit.fit.objective_cases import *


@pytest.mark.parametrize(
    ("model", "observed", "floor"),
    [
        ([1.0, np.nan], [1.0, 1.0], 1e-8),
        ([1.0, 1.0], [1.0, np.inf], 1e-8),
        ([1.0], [1.0], np.nan),
    ],
    ids=("model-nan", "observed-infinity", "floor-nan"),
)
def test_log_residuals_reject_nonfinite_inputs(model, observed, floor) -> None:
    with pytest.raises(ValueError, match="finite"):
        log_residuals(np.asarray(model), np.asarray(observed), floor)


@pytest.mark.parametrize(
    ("delta", "weights", "c"),
    [
        ([np.nan], [1.0], 0.05),
        ([0.0], [np.inf], 0.05),
        ([0.0], [1.0], np.nan),
        ([], [], 0.05),
    ],
    ids=("delta-nan", "weights-infinity", "threshold-nan", "empty"),
)
def test_robust_log_cost_returns_infinity_for_nonfinite_numeric_inputs(delta, weights, c) -> None:
    assert np.isinf(robust_log_cost(np.asarray(delta), np.asarray(weights), c))


@pytest.mark.parametrize(
    ("scale", "estimate", "tau", "count"),
    [
        (np.nan, 1.0, 0.1, 10),
        (1.0, np.inf, 0.1, 10),
        (1.0, 1.0, np.nan, 10),
        (1.0, 1.0, 0.1, 0),
    ],
    ids=("scale-nan", "estimate-infinity", "tau-nan", "count-zero"),
)
def test_scale_prior_rejects_nonfinite_numeric_inputs(scale, estimate, tau, count) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        scale_prior_penalty(scale, estimate, tau, count)


def test_evaluate_jacobian_matches_richardson_central_differences() -> None:
    problem = _problem()
    unit = np.full(len(problem.variables), 0.4)

    analytic = evaluate_jacobian(problem, unit)
    reference = _richardson(problem, unit)

    np.testing.assert_allclose(analytic, reference, rtol=1e-6, atol=5e-11)


def test_evaluate_jacobian_is_accurate_near_the_critical_edge() -> None:
    problem = _problem(size=160)
    unit = np.full(len(problem.variables), 0.5)
    fitted = problem.data.qz_a_inv <= 0.08
    critical = compile_fit_problem(
        replace(problem.data, fit_mask=fitted, fit_ready=True),
        problem.structure,
        problem.instrument,
        problem.config,
    )

    analytic = evaluate_jacobian(critical, unit)
    reference = _richardson(critical, unit)

    np.testing.assert_allclose(analytic, reference, rtol=1e-6, atol=5e-11)


def test_evaluate_jacobian_output_is_read_only() -> None:
    problem = _problem()
    jacobian = evaluate_jacobian(problem, np.full(len(problem.variables), 0.5))

    assert not jacobian.flags.writeable


def test_evaluate_jacobian_does_not_recompute_the_primal_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    unit = np.full(len(problem.variables), 0.5)
    monkeypatch.setattr(
        evaluation_module,
        "evaluate_model",
        lambda *_args, **_kwargs: pytest.fail("primal model was recomputed"),
    )

    jacobian = evaluate_jacobian(problem, unit)

    assert jacobian.shape == (np.count_nonzero(problem.data.fit_mask), len(unit))


def test_periodic_jacobian_expansion_reuses_shared_layer_sld_tangent() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (
            replace(film, name="a", thickness_a=20.0),
            replace(film, name="b", thickness_a=30.0),
        ),
        repeats=5,
        top_roughness_a=1.0,
    )
    problem = compile_fit_problem(
        prepared_data(size=64),
        StructureSpec(base.fronting, (block,), base.backing),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=9), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.5)

    differentiable = evaluation_module.expanded_structure_jacobian(problem, unit)

    for layer_offset in (0, 1):
        rows = differentiable.sld_jacobian[1 + layer_offset : 11 : 2]
        expected = np.repeat(rows[:1], rows.shape[0], axis=0)
        np.testing.assert_array_equal(rows, expected)
