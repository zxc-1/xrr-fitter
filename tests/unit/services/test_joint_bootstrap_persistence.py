"""Public persistence must reconstruct joint owners, not trust opaque hashes."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import scale_problem

import xrr_fitter.api as api
from xrr_fitter.analysis.report import uncertainty_seed
from xrr_fitter.fit.joint_pipeline import _project_candidate, _summary
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_sharing import initial_joint_vector
from xrr_fitter.fit.joint_solvers import solve_joint
from xrr_fitter.io.project_codec import save_project as raw_save_project
from xrr_fitter.model.fitting import FitSearchResult
from xrr_fitter.model.joint_bootstrap_provenance import joint_bootstrap_owner_sha256, seal_joint_bootstrap
from xrr_fitter.services import fitting
from xrr_fitter.services.datasets import service_seed_branches


def _project_inputs(root):
    config = replace(api.FitConfig.fast(713), noise_model="gaussian", scale_prior_enabled=False)
    config = replace(config, budget=replace(config.budget, bootstrap_samples=8))
    value = api.set_fit_config(api.new_project(), config)
    for index in range(2):
        local = scale_problem("gaussian", seed=17 + index)
        source = root / f"member-{index}.xy"
        np.savetxt(
            source,
            np.column_stack((local.data.two_theta_deg, local.data.intensity_raw, local.data.intensity_sigma_raw)),
            fmt="%.17g",
        )
        value = api.add_dataset(
            value,
            source,
            local.instrument,
            column_mapping=api.DataColumnMapping(intensity_sigma=2),
            beam=local.data.beam,
        )
        identifier = value.datasets[-1].dataset_id
        value = api.set_structure(value, identifier, local.structure)
        prepared = fitting.prepare_dataset_fit(value, identifier, 17)
        settings = tuple(
            api.ParameterSetting(
                item.name,
                item.initial,
                item.lower,
                item.upper,
                freedom=api.ParameterFreedom.from_locked(item.name != "instrument.scale"),
            )
            for item in prepared.problem.parameter_definitions
        )
        value = api.set_parameter_settings(value, identifier, settings)
    value = api.set_batch_mode(value, "joint")
    rule = api.SharingRule(
        "shared-scale", tuple(api.ParameterReference(item.dataset_id, "instrument.scale") for item in value.datasets)
    )
    return api.set_sharing_rules(value, (rule,))


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("joint-public-source")
    project = _project_inputs(root)
    seed = service_seed_branches(project)[1]
    prepared = tuple(fitting.prepare_dataset_fit(project, item.dataset_id, seed) for item in project.datasets)
    problem = compile_joint_problem(
        tuple(item.dataset_id for item in prepared),
        tuple(item.problem for item in prepared),
        project.sharing_rules,
        project.constraint_rules,
    )
    initial = initial_joint_vector(problem)
    solved = solve_joint(problem, initial, 80, None)
    assert solved.evaluation.valid
    candidates = _project_candidate(problem, solved, "E-0", 0)
    summary = _summary("E", (candidates,))
    searches = tuple(
        FitSearchResult(
            local.parameter_definitions, (candidate,), 0, (), (1,), (summary,), local.region_labels, local.weights
        )
        for local, candidate in zip(problem.problems, candidates, strict=True)
    )
    results = fitting._analyze_joint_searches(problem, searches, ((), ()), bootstrap_enabled=True)
    project = replace(
        project,
        datasets=tuple(
            replace(item.updated_dataset, last_valid_result=result)
            for item, result in zip(prepared, results, strict=True)
        ),
    )
    return project, problem, candidates, solved.unit_vector


def _with_sampling(project, sampling):
    return replace(
        project,
        datasets=tuple(
            replace(
                item,
                last_valid_result=replace(
                    item.last_valid_result,
                    uncertainty=replace(item.last_valid_result.uncertainty, bootstrap_evidence=sampling),
                ),
            )
            for item in project.datasets
        ),
    )


def _foreign_sampling(fitted):
    project, problem, candidates, vector = fitted
    second = problem.problems[1]
    changed = replace(second, data=replace(second.data, intensity_raw=second.data.intensity_raw + 1.0))
    foreign = replace(problem, problems=(problem.problems[0], changed))
    owner = joint_bootstrap_owner_sha256(foreign, candidates, vector, uncertainty_seed(problem.problems[0].config))
    sampling = project.datasets[0].last_valid_result.uncertainty.bootstrap_evidence
    return seal_joint_bootstrap(sampling, "E-0", owner)


@pytest.mark.parametrize("operation", ("save", "load"))
def test_public_persistence_rejects_whole_foreign_bootstrap_with_same_label_and_summaries(fitted, tmp_path, operation):
    project = fitted[0]
    bad = _with_sampling(project, _foreign_sampling(fitted))
    path = tmp_path / "project.xrrproj.json"
    if operation == "load":
        raw_save_project(bad, path)
        with pytest.raises(ValueError, match="joint bootstrap.*(owner|context)"):
            api.load_project(path)
    else:
        api.save_project(project, path)
        original = sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="joint bootstrap.*(owner|context)"):
            api.save_project(bad, path)
        assert sha256(path.read_bytes()).hexdigest() == original


def _forbidden(*_args, **_kwargs):
    pytest.fail("persistence ownership validation must not execute physics, fitting or resampling")


def test_public_roundtrip_and_relocation_only_recompile_declarations(fitted, tmp_path, monkeypatch):
    project = fitted[0]
    for name in ("evaluate_joint_vector", "bootstrap_joint_local", "refit_resampled_joint", "run_joint_fit"):
        monkeypatch.setattr(fitting, name, _forbidden)
    monkeypatch.setattr(np.random, "default_rng", _forbidden)
    path = tmp_path / "project.xrrproj.json"
    api.save_project(project, path)
    loaded = api.load_project(path)
    original = project.datasets[0].last_valid_result.uncertainty.bootstrap_evidence
    assert (
        loaded.datasets[0].last_valid_result.uncertainty.bootstrap_evidence.provenance_sha256
        == original.provenance_sha256
    )
    moved = []
    for index, item in enumerate(loaded.datasets):
        source = tmp_path / f"moved-{index}.xy"
        from pathlib import Path

        source.write_bytes(Path(item.source_path).read_bytes())
        moved.append(replace(item, source_path=str(source)))
    relocated = replace(loaded, datasets=tuple(moved))
    api.save_project(relocated, tmp_path / "moved.xrrproj.json")
    restored = api.load_project(tmp_path / "moved.xrrproj.json")
    assert (
        restored.datasets[1].last_valid_result.uncertainty.bootstrap_evidence.joint_owner_sha256
        == original.joint_owner_sha256
    )


def test_public_persistence_rejects_changed_saved_winner_numerics(fitted, tmp_path):
    value = fitted[0]
    second = value.datasets[1]
    result = second.last_valid_result
    changed = replace(result.candidates[0], model_normalized=result.candidates[0].model_normalized * 1.001)
    dataset = replace(second, last_valid_result=replace(result, candidates=(changed,)))
    value = replace(value, datasets=(value.datasets[0], dataset))
    with pytest.raises(ValueError, match="joint bootstrap.*(owner|context)"):
        api.save_project(value, tmp_path / "changed.xrrproj.json")


def test_public_persistence_checks_parameter_members_against_recompiled_layout(fitted, tmp_path):
    project = fitted[0]
    datasets = []
    for item in project.datasets:
        result = item.last_valid_result
        report = result.uncertainty
        changed = replace(report, parameter_members=(report.parameter_members[0][::-1],))
        datasets.append(replace(item, last_valid_result=replace(result, uncertainty=changed)))
    bad = replace(project, datasets=tuple(datasets))
    with pytest.raises(ValueError, match="joint bootstrap.*(layout|member)"):
        api.save_project(bad, tmp_path / "wrong-layout.xrrproj.json")


def test_public_persistence_does_not_guess_automatic_joint_history(fitted, tmp_path):
    value = replace(fitted[0], batch_mode="independent")
    with pytest.raises(ValueError, match="joint bootstrap"):
        api.save_project(value, tmp_path / "automatic.xrrproj.json")
