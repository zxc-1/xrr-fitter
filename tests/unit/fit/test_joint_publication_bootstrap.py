"""Memberwise resampling is not evidence for a shared-coordinate bootstrap."""

from dataclasses import replace

import pytest
from tests.support.model_cases import dataset_project, project
from tests.unit.fit.test_joint_pipeline import _asymmetric_joint_problem, _joint_problem

from xrr_fitter.analysis.report import analyze_search_result
from xrr_fitter.fit.joint_pipeline import JointFitRequest, run_joint_fit
from xrr_fitter.fit.problem import recompile_resampled_problem


@pytest.mark.parametrize("build_joint", (_joint_problem, _asymmetric_joint_problem))
def test_joint_project_rejects_memberwise_bootstrap_as_global_evidence(build_joint) -> None:
    joint = build_joint()
    searches = run_joint_fit(JointFitRequest(joint))
    results = tuple(
        analyze_search_result(problem, search, profile_names=(), recompile=recompile_resampled_problem)
        for problem, search in zip(joint.problems, searches, strict=True)
    )
    assert all(result.uncertainty.bootstrap_evidence is not None for result in results)
    assert all(result.uncertainty.bootstrap_evidence.joint_owner_sha256 is None for result in results)
    datasets = tuple(
        dataset_project(dataset_id, result=result)
        for dataset_id, result in zip(joint.dataset_ids, results, strict=True)
    )

    with pytest.raises(ValueError, match="joint bootstrap requires complete shared uncertainty"):
        replace(project(*datasets), batch_mode="joint")
