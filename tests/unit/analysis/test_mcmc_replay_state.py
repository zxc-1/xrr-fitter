from __future__ import annotations

from tests.support.model_cases import prepared_data, simple_structure
from tests.unit.analysis.test_mcmc import _candidate, run_problem_mcmc

from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.analysis import McmcConfig
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterDefinition, ParameterSetting

LOCKED_NAME = "component.0.thickness_a"


def _locked_setting(definition: ParameterDefinition) -> ParameterSetting:
    locked_value = 50.0 if definition.name == LOCKED_NAME else definition.initial
    return ParameterSetting(
        definition.name,
        locked_value,
        locked_value if definition.name == LOCKED_NAME else definition.lower,
        locked_value if definition.name == LOCKED_NAME else definition.upper,
        locked=definition.name != "component.0.density_scale",
    )


def _locked_replay_problem():
    initial = compile_fit_problem(
        prepared_data(size=40),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        FitConfig.fast(929),
    )
    return compile_fit_problem(
        initial.data,
        initial.structure,
        initial.instrument,
        initial.config,
        tuple(_locked_setting(definition) for definition in initial.parameter_definitions),
    )


def test_problem_mcmc_preserves_locked_parameter_values_for_replay() -> None:
    problem = _locked_replay_problem()

    report = run_problem_mcmc(
        problem,
        _candidate(problem),
        McmcConfig(walkers=6, burn_in=0, production_steps=4, thin=2),
        child_seed=441,
    )

    fixed = dict(report.fixed_parameter_values)
    assert fixed[LOCKED_NAME] == 50.0
    assert LOCKED_NAME not in report.parameter_names
    assert LOCKED_NAME not in report.derived_parameter_names
