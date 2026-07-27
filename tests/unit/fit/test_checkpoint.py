from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, FitStageSummary, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting


def _api():
    return import_module("xrr_fitter.fit.checkpoint")


def _problem(*, seed: int = 751, source_path: str = "synthetic.xy"):
    config = replace(
        FitConfig.fast(seed),
        final_seed_count=4,
        budget=SearchBudget(0, 0, 5, 1, 1),
        local_workers=1,
        scale_prior_enabled=False,
    )
    data = replace(prepared_data(size=40), source_path=Path(source_path))
    return compile_fit_problem(
        data,
        simple_structure(),
        InstrumentSpec(footprint_mode="none", instrument_id="lab"),
        config,
    )


def _candidate(problem, candidate_id: str, seed_index: int):
    unit = np.full(len(problem.variables), 0.5)
    evaluation = evaluate_vector(problem, unit)
    return candidate_from_evaluation(
        problem,
        unit,
        evaluation,
        candidate_id,
        seed_index,
        "converged",
        4,
    )


def test_checkpoint_identity_is_canonical_and_ignores_source_location() -> None:
    api = _api()
    first = api.checkpoint_identity(_problem(source_path="one/synthetic.xy"))
    second = api.checkpoint_identity(_problem(source_path="moved/synthetic.xy"))

    assert first == second
    assert first.data_sha256 == "a" * 64
    for value in (
        first.structure_fingerprint,
        first.instrument_fingerprint,
        first.config_fingerprint,
        first.parameter_settings_fingerprint,
    ):
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")


def test_checkpoint_identity_changes_on_each_resume_relevant_axis() -> None:
    api = _api()
    problem = _problem()
    baseline = api.checkpoint_identity(problem)
    layer = problem.structure.components[0]
    changed_structure = replace(
        problem,
        structure=replace(
            problem.structure,
            components=(replace(layer, thickness_a=layer.thickness_a + 1.0),),
        ),
    )
    changed_instrument = replace(
        problem,
        instrument=replace(problem.instrument, instrument_id="other-lab"),
    )
    changed_config = replace(problem, config=replace(problem.config, master_seed=752))
    definition = problem.parameter_definitions[0]
    changed_parameters = replace(
        problem,
        parameter_definitions=(
            replace(definition, upper=definition.upper + 1.0),
            *problem.parameter_definitions[1:],
        ),
    )

    assert api.checkpoint_identity(changed_structure).structure_fingerprint != baseline.structure_fingerprint
    assert api.checkpoint_identity(changed_instrument).instrument_fingerprint != baseline.instrument_fingerprint
    assert api.checkpoint_identity(changed_config).config_fingerprint != baseline.config_fingerprint
    assert (
        api.checkpoint_identity(changed_parameters).parameter_settings_fingerprint
        != baseline.parameter_settings_fingerprint
    )


def test_build_checkpoint_preserves_candidate_seed_warning_and_history_order() -> None:
    api = _api()
    problem = _problem(seed=757)
    first = _candidate(problem, "B-0", 0)
    second = _candidate(problem, "B-1", 1)
    summary = FitStageSummary(
        "B",
        ("B-0", "B-1"),
        min(first.objective, second.objective),
        first.nfev + second.nfev,
        (first.stop_reason, second.stop_reason),
    )

    checkpoint = api.build_checkpoint(
        problem,
        stage="B",
        candidates=(first, second),
        child_seeds=(101, 102),
        runtime_warnings=("first warning", "second warning"),
        stage_summaries=(summary,),
    )

    assert checkpoint.stage == "B"
    assert checkpoint.candidates == (first, second)
    assert checkpoint.child_seeds == (101, 102)
    assert checkpoint.runtime_warnings == ("first warning", "second warning")
    assert checkpoint.stage_summaries == (summary,)
    assert checkpoint.joint_layout_fingerprint == ""


def test_checkpoint_parameter_fingerprint_binds_lock_and_bounds() -> None:
    api = _api()
    problem = _problem(seed=761)
    target = problem.parameter_definitions[0]
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.initial if definition.name == target.name else definition.lower,
            definition.initial if definition.name == target.name else definition.upper,
            locked=definition.name == target.name or definition.locked,
        )
        for definition in problem.parameter_definitions
    )
    locked = compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        settings,
    )

    assert (
        api.checkpoint_identity(locked).parameter_settings_fingerprint
        != api.checkpoint_identity(problem).parameter_settings_fingerprint
    )
