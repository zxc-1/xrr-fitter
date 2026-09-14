"""Joint bootstrap sources cannot be exchanged across equal-looking reports."""

from dataclasses import replace

import pytest
from tests.support.model_cases import dataset_project, project
from tests.unit.analysis.covariance_cases import joint_scale_searches

from xrr_fitter.analysis.residual_calibration import qualify_poisson_bootstrap
from xrr_fitter.io.codec_inference import bootstrap_from_dict, bootstrap_to_dict
from xrr_fitter.io.project_codec import project_from_dict, project_to_dict
from xrr_fitter.model.inference import ResidualEvidence
from xrr_fitter.model.joint_bootstrap_provenance import (
    joint_bootstrap_owner_sha256,
    seal_joint_bootstrap,
    validate_joint_bootstrap_content,
)
from xrr_fitter.services import fitting


@pytest.fixture(scope="module")
def fitted():
    problem, searches = joint_scale_searches("gaussian")
    results = fitting._analyze_joint_searches(problem, searches, ((), ()), bootstrap_enabled=True)
    return problem, searches, results


def _sealed(fitted):
    problem, searches, results = fitted
    report = results[0].uncertainty
    candidates = tuple(
        next(candidate for candidate in search.candidates if candidate.candidate_id == report.candidate_id)
        for search in searches
    )
    from xrr_fitter.analysis.report import uncertainty_seed

    owner = joint_bootstrap_owner_sha256(
        problem, candidates, candidates[0].unit_vector, uncertainty_seed(problem.problems[0].config)
    )
    return seal_joint_bootstrap(report.bootstrap_evidence, report.candidate_id, owner)


def _project_with_sampling(fitted, sampling):
    problem, _searches, results = fitted
    report = replace(results[0].uncertainty, bootstrap_evidence=sampling)
    datasets = tuple(
        replace(
            dataset_project(identifier, result=replace(result, uncertainty=report)),
            fit_mask=(True,) * local.data.fit_mask.size,
        )
        for identifier, result, local in zip(problem.dataset_ids, results, problem.problems, strict=True)
    )
    base = project(*datasets)
    return replace(base, batch_mode="joint", fit_config=replace(base.fit_config, noise_model="gaussian"))


def test_generated_joint_evidence_has_independent_owner_and_content(fitted):
    result = fitted[2][0].uncertainty.bootstrap_evidence
    assert result.joint_owner_sha256 is not None, "generated joint bootstrap must own its full numerical context"
    assert result.candidate_id == fitted[2][0].uncertainty.candidate_id
    validate_joint_bootstrap_content(result)


@pytest.mark.parametrize("damage", ("foreign_owner", "payload", "unsealed"))
def test_joint_analysis_rejects_bad_callback_before_publishing(fitted, monkeypatch, damage):
    problem, searches, _results = fitted
    result = _sealed(fitted)
    if damage == "foreign_owner":
        result = seal_joint_bootstrap(result, result.candidate_id, "f" * 64)
    elif damage == "payload":
        result = replace(result, samples=result.samples + 1.0)
    else:
        result = replace(result, candidate_id=None, provenance_sha256=None, joint_owner_sha256=None)
    monkeypatch.setattr(fitting, "bootstrap_joint_local", lambda *_args, **_kwargs: result)
    with pytest.raises(ValueError, match="joint bootstrap|provenance|owner"):
        fitting._analyze_joint_searches(problem, searches, ((), ()), bootstrap_enabled=True)


def test_poisson_requalification_verifies_then_reseals_content(fitted):
    sampling = _sealed(fitted)
    locals_ = tuple(replace(local, config=replace(local.config, noise_model="poisson")) for local in fitted[0].problems)
    residual = ResidualEvidence(None, False, None, None, 80, (), "diagnostic_budget_insufficient")
    qualified = qualify_poisson_bootstrap(sampling, locals_, (residual, residual))
    assert qualified.joint_owner_sha256 == sampling.joint_owner_sha256
    assert qualified.provenance_sha256 != sampling.provenance_sha256, (
        "changed qualification requires a new content seal"
    )
    validate_joint_bootstrap_content(qualified)
    damaged = replace(sampling, samples=sampling.samples + 1.0)
    with pytest.raises(ValueError, match="provenance"):
        qualify_poisson_bootstrap(damaged, locals_, (residual, residual))


def test_codec_roundtrip_keeps_joint_owner_and_rejects_payload_change(fitted):
    sampling = _sealed(fitted)
    payload = bootstrap_to_dict(sampling)
    assert payload.get("joint_owner_sha256") == sampling.joint_owner_sha256
    restored = bootstrap_from_dict(payload)
    validate_joint_bootstrap_content(restored)
    payload["samples"][3][0] += 1.0
    with pytest.raises(ValueError, match="provenance|content"):
        bootstrap_from_dict(payload)
    with pytest.raises(ValueError, match="provenance|content"):
        bootstrap_to_dict(replace(sampling, samples=sampling.samples + 1.0))


@pytest.mark.parametrize("mode", ("joint", "independent"))
def test_joint_parameter_members_cannot_be_downgraded_to_unowned_bootstrap(fitted, mode):
    unsealed = replace(_sealed(fitted), joint_owner_sha256=None, candidate_id=None, provenance_sha256=None)
    with pytest.raises(ValueError, match="joint bootstrap|owner"):
        value = _project_with_sampling(fitted, unsealed)
        replace(value, batch_mode=mode)


def test_joint_graph_rejects_forked_but_individually_valid_content(fitted):
    sampling = _sealed(fitted)
    value = _project_with_sampling(fitted, sampling)
    altered = seal_joint_bootstrap(
        replace(sampling, samples=sampling.samples + 1.0), sampling.candidate_id, sampling.joint_owner_sha256
    )
    second = value.datasets[1]
    report = replace(second.last_valid_result.uncertainty, bootstrap_evidence=altered)
    changed = replace(second, last_valid_result=replace(second.last_valid_result, uncertainty=report))
    with pytest.raises(ValueError, match="joint.*(bootstrap|uncertainty|evidence)"):
        replace(value, datasets=(value.datasets[0], changed))


def test_pure_project_codec_checks_forged_objects_without_reading_sources(fitted):
    value = _project_with_sampling(fitted, _sealed(fitted))
    payload = project_to_dict(value)
    roundtrip = project_from_dict(payload)
    assert roundtrip.datasets[0].last_valid_result.uncertainty.bootstrap_evidence.joint_owner_sha256 is not None
    bad = replace(
        value.datasets[1].last_valid_result.uncertainty.bootstrap_evidence, samples=_sealed(fitted).samples + 2.0
    )
    object.__setattr__(value.datasets[1].last_valid_result.uncertainty, "bootstrap_evidence", bad)
    with pytest.raises(ValueError, match="provenance|joint"):
        project_to_dict(value)
