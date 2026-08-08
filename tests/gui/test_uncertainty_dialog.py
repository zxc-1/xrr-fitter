"""Placement of the MCMC sampling controls.

MCMC is an opt-in deep dive rather than part of the fit loop, so its seven
inputs must not occupy the analysis column for every project. The panel keeps
ownership of the controls, which is what lets candidate configuration and
operation state keep tracking the selection, but an on-demand dialog holds them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

import xrr_fitter.api as api


def _uncertainty(candidate_id: str = "candidate-a") -> api.UncertaintyReport:
    return api.UncertaintyReport(
        correlation_names=("scale",),
        correlation_matrix=np.array([[1.0]]),
        profiles=(),
        bootstrap_intervals=(("scale", 0.8, 1.2),),
        bootstrap_failure_rate=0.125,
        boundary_hits=("scale",),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate_id,
    )


def _result():
    first = replace(
        fit_candidate("candidate-a", 0.2),
        ranking_objective=0.4,
        unit_vector=np.zeros(17),
    )
    second = replace(
        fit_candidate("candidate-b", 0.3),
        ranking_objective=0.8,
        unit_vector=np.zeros(3),
    )
    return replace(final_fit_result(first, second), uncertainty=_uncertainty())


def _panel(qtbot, *, expert: bool):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    value = project(dataset_project(result=_result()))
    value = replace(value, base_directory="/private/tmp")
    value = api.select_active_dataset(value, "curve")
    if expert:
        value = api.set_expert_mode(value, True)
    panel = ResultsPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    panel.show()
    return panel


def test_mcmc_controls_stay_out_of_the_panel_layout(qtbot) -> None:
    panel = _panel(qtbot, expert=True)

    assert panel.mcmc_group.isVisibleTo(panel) is False
    for control in (panel.walkers, panel.burn_in, panel.production, panel.thin):
        assert control.isVisibleTo(panel) is False


def test_uncertainty_dialog_button_is_expert_only(qtbot) -> None:
    """One entry point replaces the inline group, and only in expert mode."""
    expert = _panel(qtbot, expert=True)
    assert expert.uncertainty_button.isVisibleTo(expert) is True

    plain = _panel(qtbot, expert=False)
    assert plain.uncertainty_button.isVisibleTo(plain) is False


def test_uncertainty_dialog_hosts_the_mcmc_controls(qtbot) -> None:
    """Opening the entry point reveals the same owned controls, not copies."""
    panel = _panel(qtbot, expert=True)

    dialog = panel.open_uncertainty_dialog()
    qtbot.addWidget(dialog)

    assert panel.mcmc_group.isVisibleTo(dialog) is True
    assert panel.mcmc_group.window() is dialog
    assert panel.mcmc_config() == api.McmcConfig.standard(17)
