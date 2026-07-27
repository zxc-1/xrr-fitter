from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import numpy as np

from xrr_fitter.model.instrument import InstrumentSpec


def _api():
    return import_module("xrr_fitter.analysis.diagnostics")


def _residual_case(
    residuals: np.ndarray,
    *,
    footprint_mode: str = "fit",
    background_kind: str = "constant",
):
    values = np.asarray(residuals, dtype=float)
    qz = np.linspace(0.01, 1.0, values.size)
    problem = SimpleNamespace(
        data=SimpleNamespace(
            qz_a_inv=qz,
            two_theta_deg=np.linspace(0.1, 5.0, values.size),
            fit_mask=np.ones(values.size, dtype=bool),
        ),
        instrument=InstrumentSpec(
            footprint_mode=footprint_mode,
            background_kind=background_kind,
        ),
    )
    return problem, SimpleNamespace(log_residuals_decades=values)


def test_residual_patterns_emit_actionable_model_diagnostics() -> None:
    module = _api()
    size = 400
    footprint_residual = np.zeros(size)
    footprint_residual[:60] = np.linspace(0.20, 0.0, 60)
    footprint = module.diagnose_residual_patterns(
        *_residual_case(footprint_residual, footprint_mode="none")
    )

    diffuse_residual = np.zeros(size)
    diffuse_residual[-100:] = np.linspace(0.20, 0.0, 100)
    diffuse = module.diagnose_residual_patterns(
        *_residual_case(diffuse_residual, background_kind="constant")
    )

    qz = np.linspace(0.01, 1.0, size)
    surface_residual = np.zeros(size)
    surface_residual[size // 2 :] = 0.10 * np.sin(40.0 * qz[size // 2 :])
    surface = module.diagnose_residual_patterns(*_residual_case(surface_residual))

    assert "suspected_unmodeled_footprint" in {item.code for item in footprint}
    assert "suspected_diffuse_background" in {item.code for item in diffuse}
    assert "surface_thin_layer_residual" in {item.code for item in surface}
    assert all(item.point_indices for item in (*footprint, *diffuse, *surface))
    assert next(
        item for item in footprint if item.code == "suspected_unmodeled_footprint"
    ).point_indices == tuple(range(60))
    assert next(
        item for item in diffuse if item.code == "suspected_diffuse_background"
    ).point_indices == tuple(range(320, 400))

    guarded_footprint = module.diagnose_residual_patterns(
        *_residual_case(footprint_residual, footprint_mode="fit")
    )
    guarded_diffuse = module.diagnose_residual_patterns(
        *_residual_case(diffuse_residual, background_kind="linear")
    )
    assert "suspected_unmodeled_footprint" not in {
        item.code for item in guarded_footprint
    }
    assert "suspected_diffuse_background" not in {
        item.code for item in guarded_diffuse
    }


def test_residual_acf_flags_two_significant_lags_but_not_white_noise() -> None:
    module = _api()
    rng = np.random.default_rng(994)
    innovations = rng.normal(size=800)
    correlated = np.empty_like(innovations)
    correlated[0] = innovations[0]
    for index in range(1, innovations.size):
        correlated[index] = 0.92 * correlated[index - 1] + innovations[index]

    assert module.residual_autocorrelation_flag(correlated)
    assert not module.residual_autocorrelation_flag(1e-14 * correlated)
    assert not module.residual_autocorrelation_flag(1e-9 * correlated)

    innovations = rng.normal(size=802)
    negative_at_two_lags = innovations[2:] - 0.8 * innovations[1:-1] - 0.5 * innovations[:-2]
    assert module.residual_autocorrelation_flag(negative_at_two_lags)
    assert not module.residual_autocorrelation_flag(rng.normal(size=800))


def test_report_computes_residual_acf_in_q_sorted_order() -> None:
    module = _api()
    rng = np.random.default_rng(306)
    qz = np.linspace(0.01, 1.0, 800)
    permutation = rng.permutation(qz.size)
    innovations = np.random.default_rng(305).normal(size=qz.size)
    sorted_residual = np.empty_like(innovations)
    sorted_residual[0] = innovations[0]
    for index in range(1, sorted_residual.size):
        sorted_residual[index] = 0.92 * sorted_residual[index - 1] + innovations[index]
    problem = SimpleNamespace(
        data=SimpleNamespace(
            qz_a_inv=qz[permutation],
            fit_mask=np.ones(qz.size, dtype=bool),
        )
    )
    residual = np.empty_like(sorted_residual)
    residual[np.argsort(problem.data.qz_a_inv, kind="stable")] = sorted_residual
    candidate = SimpleNamespace(log_residuals_decades=residual)

    ordered = module.ordered_fit_residuals(problem, candidate)

    np.testing.assert_array_equal(ordered, sorted_residual)
    assert module.residual_autocorrelation_flag(ordered)
