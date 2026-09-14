"""Independent joint context owners and complete bootstrap content seals."""

import pickle
from dataclasses import fields, replace
from importlib import import_module, util
from pathlib import Path

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import joint_scale_searches

from xrr_fitter.analysis.bootstrap_samples import bootstrap_local


def _api():
    name = "xrr_fitter.model.joint_bootstrap_provenance"
    assert util.find_spec(name) is not None, "joint bootstrap must have independent context and content seals"
    return import_module(name)


@pytest.fixture(scope="module")
def case():
    problem, searches = joint_scale_searches("gaussian")
    identifier = searches[0].best_candidate.candidate_id
    candidates = tuple(
        next(item for item in search.candidates if item.candidate_id == identifier) for search in searches
    )
    return problem, candidates, candidates[0].unit_vector, 2**63 + 7


def _sampling():
    return bootstrap_local(
        lambda _rng, index: np.array([float(index)]), ("shared-scale",), sample_count=10, child_seed=7
    )


def _owned(case):
    api = _api()
    owner = api.joint_bootstrap_owner_sha256(*case)
    return api.seal_joint_bootstrap(_sampling(), case[1][0].candidate_id, owner)


def test_owner_is_deterministic_pickle_safe_and_distinct_from_residual_domain(case):
    api = _api()
    owner = api.joint_bootstrap_owner_sha256(*case)
    assert len(owner) == 64
    assert owner == api.joint_bootstrap_owner_sha256(*pickle.loads(pickle.dumps(case)))
    from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector
    from xrr_fitter.model.provenance import joint_residual_owner_sha256

    problem, _candidates, vector, _seed = case
    evaluation = evaluate_joint_vector(problem, vector)
    assert owner != joint_residual_owner_sha256(
        problem.problems, problem.dataset_ids, vector, evaluation.local_evaluations, problem.layout_fingerprint
    )


@pytest.mark.parametrize("field", ("counts", "normalization", "mask", "weights", "sampling", "denominator", "config"))
def test_owner_binds_every_members_complete_context(case, field):
    api = _api()
    problem, candidates, vector, seed = case
    local = problem.problems[1]
    mask = local.data.fit_mask.copy()
    mask[-1] = False
    updates = {
        "counts": {"data": replace(local.data, intensity_raw=local.data.intensity_raw + 1.0)},
        "normalization": {"data": replace(local.data, normalization=2.0)},
        "mask": {"data": replace(local.data, fit_mask=mask)},
        "weights": {"weights": local.weights * 1.01},
        "sampling": {"sampling_multipliers": local.sampling_multipliers * 2.0},
        "denominator": {"objective_point_count": local.objective_point_count + 1},
        "config": {"config": replace(local.config, master_seed=18)},
    }
    changed = replace(problem, problems=(problem.problems[0], replace(local, **updates[field])))
    assert api.joint_bootstrap_owner_sha256(*case) != api.joint_bootstrap_owner_sha256(
        changed, candidates, vector, seed
    )


@pytest.mark.parametrize("field", ("layout_fingerprint", "sharing_rules", "scatter_maps", "global_variables"))
def test_owner_binds_full_layout_not_just_a_claimed_fingerprint(case, field):
    api = _api()
    problem, candidates, vector, seed = case
    updates = {
        "layout_fingerprint": "b" * 64,
        "sharing_rules": (),
        "scatter_maps": ((0,), (-1,)),
        "global_variables": (replace(problem.global_variables[0], transform="linear"),),
    }
    changed = replace(problem, **{field: updates[field]})
    assert api.joint_bootstrap_owner_sha256(*case) != api.joint_bootstrap_owner_sha256(
        changed, candidates, vector, seed
    )


@pytest.mark.parametrize("field", ("objective", "model_normalized", "residuals", "parameters", "unit_vector"))
def test_same_label_winner_changes_are_bound(case, field):
    api = _api()
    problem, candidates, vector, seed = case
    candidate = candidates[1]
    changes = {
        "objective": candidate.objective + 0.1,
        "model_normalized": candidate.model_normalized * 1.01,
        "residuals": candidate.residuals + 0.01,
        "parameters": (
            replace(candidate.parameters[0], value=candidate.parameters[0].value * 1.01),
            *candidate.parameters[1:],
        ),
        "unit_vector": candidate.unit_vector + 0.001,
    }
    modified = (candidates[0], replace(candidate, **{field: changes[field]}))
    assert api.joint_bootstrap_owner_sha256(*case) != api.joint_bootstrap_owner_sha256(problem, modified, vector, seed)


def test_owner_binds_member_order_global_vector_and_full_uint64_seed(case):
    api = _api()
    problem, candidates, vector, seed = case
    expected = api.joint_bootstrap_owner_sha256(*case)
    assert expected != api.joint_bootstrap_owner_sha256(problem, candidates, vector + 0.001, seed)
    assert expected != api.joint_bootstrap_owner_sha256(problem, candidates, vector, seed + 1)
    reversed_problem = replace(problem, problems=problem.problems[::-1], dataset_ids=problem.dataset_ids[::-1])
    assert expected != api.joint_bootstrap_owner_sha256(reversed_problem, candidates[::-1], vector, seed)


def test_source_relocation_does_not_change_owner(case):
    api = _api()
    problem, candidates, vector, seed = case
    moved = tuple(
        replace(local, data=replace(local.data, source_path=Path("moved") / f"{index}.xy"))
        for index, local in enumerate(problem.problems)
    )
    assert api.joint_bootstrap_owner_sha256(*case) == api.joint_bootstrap_owner_sha256(
        replace(problem, problems=moved), candidates, vector, seed
    )


def test_content_seal_covers_every_field_except_itself(case):
    api = _api()
    result = _owned(case)
    api.validate_joint_bootstrap(
        result, case[1][0].candidate_id, api.joint_bootstrap_owner_sha256(*case), result.parameter_names
    )
    from xrr_fitter.model.provenance import _identity_sha256

    payload = {field.name: getattr(result, field.name) for field in fields(result) if field.name != "provenance_sha256"}
    expected = _identity_sha256({"schema": "xrr-joint-bootstrap-content-v1", "bootstrap": payload})
    assert result.provenance_sha256 == expected
    api.validate_joint_bootstrap_content(pickle.loads(pickle.dumps(result)))


@pytest.mark.parametrize("field", ("samples", "method", "candidate_id", "joint_owner_sha256", "diagnostic"))
def test_content_tampering_is_rejected_even_if_interval_summary_is_unchanged(case, field):
    api = _api()
    original = _owned(case)
    samples = original.samples.copy()
    samples[5, 0] += 0.5
    updates = {
        "samples": {"samples": samples},
        "method": {"method": "other_custom_resampling"},
        "candidate_id": {"candidate_id": "E-other"},
        "joint_owner_sha256": {"joint_owner_sha256": "c" * 64},
        "diagnostic": {"diagnostic_unavailable_reason": "residual_diagnostics_failed"},
    }
    changed = replace(original, **updates[field])
    assert changed.intervals == original.intervals
    with pytest.raises(ValueError, match="provenance|content"):
        api.validate_joint_bootstrap_content(changed)


def test_complete_other_context_bootstrap_is_not_accepted_under_same_label(case):
    api = _api()
    result = _owned(case)
    with pytest.raises(ValueError, match="owner|context"):
        api.validate_joint_bootstrap(result, result.candidate_id, "d" * 64, result.parameter_names)


def test_resealing_changes_content_not_numerical_owner(case):
    api = _api()
    result = _owned(case)
    changed = replace(result, diagnostic_unavailable_reason="residual_diagnostics_failed")
    resealed = api.seal_joint_bootstrap(changed, result.candidate_id, result.joint_owner_sha256)
    assert resealed.joint_owner_sha256 == result.joint_owner_sha256
    assert resealed.provenance_sha256 != result.provenance_sha256
    api.validate_joint_bootstrap_content(resealed)
