"""Exposed development null paths, not new holdout or statistical acceptance.

The frozen product coordinates are the actual precision-v2 winners that supplied
these original null draws. They generate data only: the diagnostic estimator
still starts from its declaration and three fixed Sobol points. Rebuilding the
case through the current API avoids a saved-format compatibility fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.analysis.bootstrap_generation import bootstrap_source, draw_replicates
from xrr_fitter.analysis.residual_calibration import diagnostic_seed
from xrr_fitter.fit.diagnostic_refit import refit_diagnostic_single
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import recompile_resampled_problem
from xrr_fitter.services.datasets import service_seed_branches
from xrr_fitter.services.fitting import prepare_dataset_fit

DEVELOPMENT_NULLS = (
    ("replay_low", 119, 322, 0.6003929841455483, 4796801050170636519),
    ("replay_low", 148, 493, 0.6020883050738053, 953685659577371635),
    ("replay_low", 119, 7, 0.6003929841455483, 4796801050170636519),
    ("footprint", 37, 0, 0.5161287490090988, 4932508153149414292),
    ("surface", 37, 0, 0.8040366287961467, 4932508153149414292),
)


def _development_null(load_tool_module, tmp_path, group, seed, index, source_unit, expected_seed):
    tool = load_tool_module("check_poisson_diagnostics")
    builder = tool.cases.build_case if group == "replay_low" else tool.cases.build_observations
    fixture = builder(tmp_path, group, seed)
    project = fixture.project
    seeds, _, _ = service_seed_branches(project)
    dataset_id = project.datasets[0].dataset_id
    problem = prepare_dataset_fit(project, dataset_id, seeds[dataset_id]).problem
    assert diagnostic_seed(problem.config) == expected_seed
    product = evaluate_vector(problem, np.asarray([source_unit]))
    source = bootstrap_source(problem, product.model_normalized, product.fit_residuals)
    draws = draw_replicates(
        (source,), index + 1, np.random.default_rng(expected_seed), recompile_resampled_problem, None
    )
    assert not isinstance(draws[-1], str)
    return draws[-1][0]


@pytest.mark.parametrize("group,seed,index,source_unit,expected_seed", DEVELOPMENT_NULLS)
def test_declared_refitter_completes_exposed_development_null(
    load_tool_module, tmp_path, group, seed, index, source_unit, expected_seed
):
    problem = _development_null(load_tool_module, tmp_path, group, seed, index, source_unit, expected_seed)
    result = refit_diagnostic_single(problem)
    assert result.failure_reason is None, (group, seed, index, result.failure_reason, result.nfev)
    assert result.attempted_paths == 4
    budget = problem.config.budget
    limit = max(budget.local_min_nfev, budget.local_nfev_per_parameter * len(problem.variables))
    assert 0 < result.nfev <= 4 * limit
    assert np.all((result.unit_vector >= 0.0) & (result.unit_vector <= 1.0))
    assert len(result.evaluations) == 1 and result.evaluations[0].valid
