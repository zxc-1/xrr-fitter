from __future__ import annotations

from dataclasses import replace
from pathlib import Path, PurePosixPath

import numpy as np

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.fit.screening import (
    fringe_count_screen,
    fringe_extrema_qz,
    reliable_fringe_window,
)
from xrr_fitter.io.project_codec import project_from_bytes
from xrr_fitter.io.xy import read_xy_bytes
from xrr_fitter.model.fitting import FitConfig

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "examples"


def _problem(*, data=None):
    project = project_from_bytes((REFERENCE / "single-layer.xrrproj.json").read_bytes())
    dataset = project.datasets[0]
    prepared = data or read_xy_bytes(
        (REFERENCE / "single-layer.xy").read_bytes(),
        source_path=PurePosixPath("xrr_fitter/examples/single-layer.xy"),
        beam=dataset.beam,
        import_angle_offset_deg=dataset.import_angle_offset_deg,
        column_mapping=dataset.column_mapping,
    )
    return compile_fit_problem(
        prepared,
        dataset.structure,
        dataset.instrument,
        FitConfig.fast(master_seed=78),
        dataset.parameter_settings,
    )


def _candidate(problem, thickness_a: float):
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical["component.0.thickness_a"] = thickness_a
    unit = encode_physical_vector(problem, physical)
    evaluation = evaluate_vector(problem, unit)
    assert evaluation.valid
    return candidate_from_evaluation(
        problem,
        unit,
        evaluation,
        candidate_id=f"thickness-{thickness_a:g}",
        seed_index=-1,
        stop_reason="fringe-screen fixture",
        nfev=1,
    )


def test_fringe_count_screen_rejects_thickness_off_by_a_fringe() -> None:
    problem = _problem()
    good = _candidate(problem, 173.0)
    doubled = _candidate(problem, 346.0)

    result = fringe_count_screen(problem, (good, doubled))

    assert good in result.candidates
    assert doubled not in result.candidates
    assert not result.disabled


def test_fringe_extrema_scales_extreme_finite_q_before_qz4_transform() -> None:
    qz = np.linspace(1e80, 2e80, 256)
    normalized = 1e-8 * (1.0 + 0.2 * np.cos(np.linspace(0.0, 16.0 * np.pi, qz.size)))

    with np.errstate(over="raise", invalid="raise"):
        extrema = fringe_extrema_qz(qz, normalized, 1e-10)

    assert extrema.size >= 4
    assert np.all(np.isfinite(extrema))


def test_fringe_count_screen_disables_itself_without_reliable_fringes() -> None:
    base = _problem()
    featureless = np.geomspace(1.0, 1e-8, base.data.qz_a_inv.size)
    data = replace(
        base.data,
        intensity_raw=featureless,
        intensity_normalized=featureless,
    )
    problem = _problem(data=data)
    candidates = (_candidate(problem, 120.0), _candidate(problem, 240.0))

    result = fringe_count_screen(problem, candidates)

    assert result.candidates == candidates
    assert result.disabled


def test_reliable_fringe_window_excludes_uninformative_curve_tails() -> None:
    problem = _problem()

    window = reliable_fringe_window(problem)

    assert window is not None
    q_min, q_max, extrema_count = window
    fitted_q = problem.data.qz_a_inv[problem.data.fit_mask]
    assert q_min > fitted_q[50]
    assert q_max < fitted_q[-10]
    assert extrema_count >= 4


def test_fringe_count_screen_ignores_noisy_tails_outside_reliable_window() -> None:
    base = _problem()
    intensity = base.data.intensity_normalized.copy()
    intensity[:60] *= 1.0 + 0.6 * np.sin(np.linspace(0.0, 18.0 * np.pi, 60))
    intensity[-30:] *= 1.0 + 0.6 * np.sin(np.linspace(0.0, 12.0 * np.pi, 30))
    intensity = np.maximum(intensity, base.data.r_floor)
    problem = _problem(
        data=replace(
            base.data,
            intensity_raw=intensity,
            intensity_normalized=intensity,
        )
    )
    good = _candidate(problem, 173.0)
    doubled = _candidate(problem, 346.0)

    result = fringe_count_screen(problem, (good, doubled))

    assert good in result.candidates
    assert doubled not in result.candidates


def test_fringe_count_screen_does_not_undercount_on_feature_selected_grid() -> None:
    base = _problem()
    data = base.data
    fit_indices = np.flatnonzero(data.fit_mask)
    fitted_q = data.qz_a_inv[data.fit_mask]
    full = fringe_extrema_qz(
        fitted_q,
        data.intensity_normalized[data.fit_mask],
        data.r_floor,
    )
    extrema_indices = fit_indices[np.searchsorted(fitted_q, full)]
    selected = np.unique(
        np.concatenate(
            (
                np.arange(0, data.qz_a_inv.size, 8),
                extrema_indices,
                [data.qz_a_inv.size - 1],
            )
        )
    )
    optional = {
        name: None if getattr(data, name) is None else getattr(data, name)[selected]
        for name in (
            "intensity_sigma_raw",
            "resolution_raw",
            "intensity_sigma_normalized",
            "sigma_q_a_inv",
        )
    }
    coarse_data = replace(
        data,
        source_row_groups=tuple(data.source_row_groups[index] for index in selected),
        two_theta_deg=data.two_theta_deg[selected],
        intensity_raw=data.intensity_raw[selected],
        qz_a_inv=data.qz_a_inv[selected],
        intensity_normalized=data.intensity_normalized[selected],
        validation_mask=data.validation_mask[selected],
        fit_mask=data.fit_mask[selected],
        **optional,
    )
    problem = _problem(data=coarse_data)
    good = _candidate(problem, 173.0)
    doubled = _candidate(problem, 346.0)

    result = fringe_count_screen(problem, (good, doubled))

    assert not result.disabled
    assert result.candidates == (good,)


def test_fit_dataset_applies_fringe_screen_before_stage_a_deduplication() -> None:
    problem = _problem()
    good = _candidate(problem, 173.0)
    doubled = _candidate(problem, 346.0)

    screened = fringe_count_screen(problem, (good, doubled))
    deduplicated = tuple(dict.fromkeys(item.candidate_id for item in screened.candidates))

    assert deduplicated == (good.candidate_id,)


def test_fit_dataset_records_when_fringe_screen_is_disabled() -> None:
    base = _problem()
    featureless = np.geomspace(1.0, 1e-8, base.data.qz_a_inv.size)
    problem = _problem(
        data=replace(
            base.data,
            intensity_raw=featureless,
            intensity_normalized=featureless,
        )
    )

    result = fringe_count_screen(problem, (_candidate(problem, 173.0),))

    assert result.warnings == ("fringe_count_screen_disabled",)


def test_fit_dataset_stops_when_enabled_fringe_screen_rejects_every_start() -> None:
    problem = _problem()
    wrong = _candidate(problem, 346.0)

    result = fringe_count_screen(problem, (wrong,))

    assert not result.disabled
    assert result.candidates == ()
    assert result.stop_reason == "stage_a_all_candidates_rejected"
