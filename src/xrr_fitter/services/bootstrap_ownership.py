"""Reconstruct manual joint bootstrap owners at public persistence boundaries.

Source validation remains with projects. Here valid source bytes are compiled
through the same preparation path and service seed as the original joint fit.
Saved candidates, not a fresh evaluation or fit, supply the numerical winner.
"""

from xrr_fitter.model.joint_bootstrap_provenance import joint_bootstrap_owner_sha256, validate_joint_bootstrap
from xrr_fitter.model.project import validate_project
from xrr_fitter.services.datasets import service_seed_branches


def _joint_sampling(report):
    if report is None or report.bootstrap_evidence is None:
        return False
    sampling = report.bootstrap_evidence
    return (
        report.parameter_members is not None
        or sampling.joint_owner_sha256 is not None
        or sampling.method.startswith("joint_")
    )


def _recompile(project, prepare_dataset, compile_joint_problem):
    seed = service_seed_branches(project)[1]
    prepared = tuple(prepare_dataset(project, dataset.dataset_id, seed) for dataset in project.datasets)
    return compile_joint_problem(
        tuple(item.dataset_id for item in prepared),
        tuple(item.problem for item in prepared),
        project.sharing_rules,
        project.constraint_rules,
    )


def _reconstructed_owner(problem, results, identifier, joint_candidate_vectors, uncertainty_seed):
    candidates = tuple(
        next(candidate for candidate in result.candidates if candidate.candidate_id == identifier) for result in results
    )
    vector = joint_candidate_vectors(problem, tuple(result.candidates for result in results), (identifier,))[0]
    return joint_bootstrap_owner_sha256(problem, candidates, vector, uncertainty_seed(problem.problems[0].config))


def _validate_reports(problem, reports, owner, identifier):
    names = tuple(variable.name for variable in problem.global_variables)
    members = tuple(variable.members for variable in problem.global_variables)
    for report in reports:
        if report.parameter_members != members:
            raise ValueError("joint bootstrap parameter members do not match the reconstructed layout")
        validate_joint_bootstrap(report.bootstrap_evidence, identifier, owner, names)


def _reports(project):
    results = tuple(dataset.last_valid_result for dataset in project.datasets)
    return results, tuple(None if result is None else result.uncertainty for result in results)


def validate_project_bootstrap_ownership(
    project,
    *,
    prepare_dataset,
    compile_joint_problem,
    joint_candidate_vectors,
    uncertainty_seed,
) -> None:
    """Reject foreign/mutated evidence rather than clear it and pretend success."""
    validate_project(project)
    results, reports = _reports(project)
    if not any(_joint_sampling(report) for report in reports):
        return
    if project.batch_mode != "joint":
        raise ValueError("joint bootstrap cannot reconstruct automatic/independent history")
    problem = _recompile(project, prepare_dataset, compile_joint_problem)
    identifier = reports[0].candidate_id
    owner = _reconstructed_owner(problem, results, identifier, joint_candidate_vectors, uncertainty_seed)
    _validate_reports(problem, reports, owner, identifier)
