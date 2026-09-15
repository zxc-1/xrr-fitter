"""Pure numerical diagnostic owners ignore analysis presentation, not data."""

from __future__ import annotations

import pickle
from dataclasses import fields, replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from tests.support.model_cases import fit_candidate, prepared_data, simple_structure
from tests.unit.model.test_diagnostic_calibration import _available, _evaluation, _statistic, _unavailable

from xrr_fitter.model.fitting import FitConfig, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec, PhysicsDiagnostic
from xrr_fitter.model.parameters import ParameterCoordinate, ParameterDefinition


def _function(name):
    module = import_module("xrr_fitter.model.provenance")
    assert hasattr(module, name), f"pure diagnostic provenance function {name} must exist"
    return getattr(module, name)


def _problem():
    data = prepared_data()
    definition = ParameterDefinition("scale", "Scale", "", "instrument", 1.0, 0.5, 1.5, "linear", False)
    return FitEvaluationContext(
        data=data,
        structure=simple_structure(),
        instrument=InstrumentSpec(),
        config=replace(FitConfig.fast(7), noise_model="poisson"),
        parameter_definitions=(definition,),
        variables=(ParameterCoordinate(0, "scale", "linear"),),
        region_labels=np.zeros(32, dtype=int),
        weights=np.ones(32),
        scale_prior_center=None,
    )


def test_residual_owner_is_pure_deterministic_and_bound_to_dataset_identity() -> None:
    owner = _function("residual_owner_sha256")
    problem, candidate = _problem(), fit_candidate()

    first = owner(problem, candidate, "curve")
    restored = pickle.loads(pickle.dumps((problem, candidate)))
    assert len(first) == 64
    assert first == owner(*restored, "curve")
    assert first != owner(problem, candidate, "other")
    assert first != owner(problem, candidate)


@pytest.mark.parametrize(
    "change",
    ["mask", "normalization", "counts", "q", "config", "definitions", "weights", "denominator", "sampling"],
)
def test_residual_owner_binds_exact_context_including_sampling(change) -> None:
    owner = _function("residual_owner_sha256")
    problem, candidate = _problem(), fit_candidate()
    mask = problem.data.fit_mask.copy()
    mask[-1] = False
    changes = {
        "mask": {"data": replace(problem.data, fit_mask=mask)},
        "normalization": {"data": replace(problem.data, normalization=1234.0)},
        "counts": {"data": replace(problem.data, intensity_raw=problem.data.intensity_raw + 1.0)},
        "q": {"data": replace(problem.data, qz_a_inv=problem.data.qz_a_inv * 1.001)},
        "config": {"config": replace(problem.config, master_seed=8)},
        "definitions": {"parameter_definitions": (replace(problem.parameter_definitions[0], initial=1.1),)},
        "weights": {"weights": np.full(32, 2.0)},
        "denominator": {"objective_point_count": 64},
        "sampling": {"sampling_multipliers": np.full(32, 2.0)},
    }

    assert owner(problem, candidate) != owner(replace(problem, **changes[change]), candidate)


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_id": "different-winner"},
        {"unit_vector": np.array([0.6])},
        {"model_normalized": np.full(4, 0.1)},
        {"residuals": np.full(4, 0.1)},
        {"objective": 2.0},
        {"noise_model": "poisson"},
        {"valid": False},
    ],
)
def test_residual_owner_binds_winner_numeric_projection(changes) -> None:
    owner = _function("residual_owner_sha256")
    problem, candidate = _problem(), fit_candidate()
    assert owner(problem, candidate) != owner(problem, replace(candidate, **changes))


def test_residual_owner_ignores_relocation_warning_and_analysis_derived_diagnostics() -> None:
    owner = _function("residual_owner_sha256")
    problem, candidate = _problem(), fit_candidate()
    changed_problem = replace(
        problem,
        data=replace(problem.data, source_path=Path("relocated/curve.xy"), warnings=("display warning",)),
        warnings=("analysis stage",),
    )
    changed_candidate = replace(
        candidate,
        seed_index=3,
        nfev=999,
        stop_reason="analysis_enriched",
        diagnostics=(PhysicsDiagnostic("suspected_diffuse_background", "derived screen"),),
    )

    assert owner(problem, candidate) == owner(changed_problem, changed_candidate)


def test_residual_owner_keeps_independent_physical_diagnostics() -> None:
    owner = _function("residual_owner_sha256")
    problem, candidate = _problem(), fit_candidate()
    changed = replace(candidate, diagnostics=(PhysicsDiagnostic("nevot_croce_applicability_exceeded", "physical"),))
    assert owner(problem, candidate) != owner(problem, changed)


def test_joint_owner_binds_ordered_members_global_unit_and_layout() -> None:
    owner = _function("joint_residual_owner_sha256")
    problems = (_problem(), replace(_problem(), config=replace(FitConfig.fast(8), noise_model="poisson")))
    evaluations = (_evaluation(), _evaluation(objective=2.0))
    unit = np.array([0.5, 0.25])
    first = owner(problems, ("first", "second"), unit, evaluations, "a" * 64)

    assert first == owner(*pickle.loads(pickle.dumps((problems, ("first", "second"), unit, evaluations, "a" * 64))))
    assert first != owner(problems[::-1], ("second", "first"), unit, evaluations[::-1], "a" * 64)
    assert first != owner(problems, ("first", "second"), np.array([0.5, 0.3]), evaluations, "a" * 64)
    assert first != owner(problems, ("first", "second"), unit, evaluations, "b" * 64)


@pytest.mark.parametrize(
    "changes",
    [
        {"model_normalized": np.full(4, 0.2)},
        {"fit_residuals": np.full(4, 0.2)},
        {"objective": 2.0},
        {"noise_model": "robust_log"},
        {"valid": False},
        {"diagnostics": (PhysicsDiagnostic("ideal_reflectivity_above_one", "physical"),)},
    ],
)
def test_joint_owner_binds_local_evaluation_numeric_evidence(changes) -> None:
    owner = _function("joint_residual_owner_sha256")
    problem, evaluation = _problem(), _evaluation()
    baseline = owner((problem,), ("curve",), np.array([0.5]), (evaluation,), "a" * 64)

    assert baseline != owner((problem,), ("curve",), np.array([0.5]), (replace(evaluation, **changes),), "a" * 64)


def test_joint_owner_ignores_analysis_advisories_but_not_sampling() -> None:
    owner = _function("joint_residual_owner_sha256")
    problem, evaluation = _problem(), _evaluation()
    enriched = replace(
        evaluation,
        reason="analysis_enriched",
        diagnostics=(PhysicsDiagnostic("surface_thin_layer_residual", "derived screen"),),
    )
    baseline = owner((problem,), ("curve",), np.array([0.5]), (evaluation,), "a" * 64)

    assert baseline == owner((problem,), ("curve",), np.array([0.5]), (enriched,), "a" * 64)
    sampled = replace(problem, sampling_multipliers=np.full(32, 2.0))
    assert baseline != owner((sampled,), ("curve",), np.array([0.5]), (evaluation,), "a" * 64)


@pytest.mark.parametrize("case", ["missing-member", "missing-evaluation", "duplicate-id", "empty-id"])
def test_joint_owner_rejects_incoherent_member_axes(case) -> None:
    owner = _function("joint_residual_owner_sha256")
    problem, evaluation = _problem(), _evaluation()
    examples = {
        "missing-member": ((problem,), ("a", "b"), (evaluation,)),
        "missing-evaluation": ((problem, problem), ("a", "b"), (evaluation,)),
        "duplicate-id": ((problem, problem), ("a", "a"), (evaluation, evaluation)),
        "empty-id": ((problem,), ("",), (evaluation,)),
    }
    problems, ids, evaluations = examples[case]
    with pytest.raises((TypeError, ValueError)):
        owner(problems, ids, np.array([0.5]), evaluations, "a" * 64)


def test_calibration_seal_uses_every_declared_field_except_its_own_seal() -> None:
    seal = _function("diagnostic_calibration_sha256")
    evidence = _available()
    provenance = import_module("xrr_fitter.model.provenance")
    expected = provenance._identity_sha256(
        {item.name: getattr(evidence, item.name) for item in fields(evidence) if item.name != "provenance_sha256"}
    )

    assert seal(evidence) == expected
    assert seal(evidence) == seal(replace(evidence, provenance_sha256="c" * 64))


@pytest.mark.parametrize(
    "changes",
    [
        {"child_seed": 8},
        {"owner_sha256": "d" * 64},
        {"tie_count": 2},
        {"observed_score": 3.0},
        {"null_statistics_sha256": "d" * 64},
        {"refit_discrepancy": (0.0, 0.0, 0.0)},
    ],
)
def test_calibration_seal_binds_replay_and_refit_evidence(changes) -> None:
    seal = _function("diagnostic_calibration_sha256")
    evidence = _available()
    assert seal(evidence) != seal(replace(evidence, **changes))


def test_calibration_seal_binds_a_consistent_change_in_charged_refit_work() -> None:
    seal = _function("diagnostic_calibration_sha256")
    evidence = _available()
    path = evidence.observed_work.paths[0]
    stages = (*path.stages[:2], replace(path.stages[2], nfev=path.stages[2].nfev + 1))
    work = replace(evidence.observed_work, paths=(replace(path, stages=stages),))
    changed = replace(evidence, observed_work=work, refit_nfev=work.nfev + evidence.null_work.nfev)

    assert changed.refit_nfev > evidence.refit_nfev
    assert seal(evidence) != seal(changed)


def test_a_refit_counter_only_change_is_rejected_before_sealing() -> None:
    evidence = _available()
    with pytest.raises(ValueError, match="work"):
        replace(evidence, refit_nfev=evidence.refit_nfev + 1)


def test_calibration_seal_binds_member_statistics_and_unavailable_failure_reasons() -> None:
    seal = _function("diagnostic_calibration_sha256")
    available = _available()
    changed = replace(available, statistics=(_statistic(observed=3.0, adjusted_p_value=0.008),))
    failed = _unavailable(failure_reasons=((-1, "diagnostic_refit_failed"),))

    assert seal(available) != seal(changed)
    assert seal(failed) != seal(replace(failed, failure_reasons=((-1, "diagnostic_refit_mismatch"),)))


def test_calibration_construction_checks_seal_syntax_without_recomputing_it() -> None:
    seal = _function("diagnostic_calibration_sha256")
    evidence = _available(provenance_sha256="0" * 64)
    assert evidence.provenance_sha256 != seal(evidence)
    assert pickle.loads(pickle.dumps(evidence)).provenance_sha256 == "0" * 64
