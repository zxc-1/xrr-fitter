"""Read-only inspection must not discard a completed sampling result.

The worker still owns its original scientific inputs and selected candidate.
Only workspace navigation may change while that immutable snapshot is running;
publication must retain the newer view without accepting a stale fit input.
"""

from dataclasses import replace

import pytest
from tests.gui.test_mcmc_interaction import _click, _deliver
from tests.gui.test_results import _mcmc_report

import xrr_fitter.api as api


def _completed(source):
    dataset = source.datasets[0]
    uncertainty = replace(dataset.last_valid_result.uncertainty, mcmc=_mcmc_report())
    result = replace(dataset.last_valid_result, uncertainty=uncertainty)
    return replace(source, datasets=(replace(dataset, last_valid_result=result),))


@pytest.mark.parametrize(
    "view_changes",
    (
        {"workspace_splitter_sizes": (280, 650, 350)},
        {"plot_tab_index": 1},
        {"expert_mode": False},
        {"active_dataset_id": None},
    ),
)
def test_mcmc_publication_preserves_readonly_view_changes(qtbot, sampling, view_changes):
    panel, _dialog, job, starts = sampling
    source = panel.document.project
    completed = []
    panel.mcmc_completed.connect(completed.append)
    _click(qtbot, panel.mcmc_button)
    current = api.set_workspace_state(source, replace(source.ui_state, **view_changes))
    panel.document.replace_project(current)
    result = _completed(source)

    _deliver(panel, job, api.OperationEvent(1, "mcmc_result", mcmc_result=result), api.OperationEvent(2, "stopped"))

    assert starts[0][0] is source
    assert panel.status_text() == "MCMC 完成"
    assert panel.document.project.datasets is result.datasets
    assert panel.document.project.ui_state is current.ui_state
    assert completed == [panel.document.project]


@pytest.mark.parametrize("change", ("candidate", "config", "dataset", "base_directory"))
def test_mcmc_publication_still_rejects_changed_sampling_inputs(qtbot, sampling, change):
    panel, _dialog, job, _starts = sampling
    source = panel.document.project
    _click(qtbot, panel.mcmc_button)
    concurrent = {
        "candidate": lambda: api.select_candidate(source, "curve", "candidate-b"),
        "config": lambda: replace(source, fit_config=replace(source.fit_config, master_seed=9)),
        "dataset": lambda: replace(source, datasets=(replace(source.datasets[0]),)),
        "base_directory": lambda: replace(source, base_directory="/private/var/tmp"),
    }[change]()
    panel.document.replace_project(concurrent)

    _deliver(
        panel,
        job,
        api.OperationEvent(1, "mcmc_result", mcmc_result=_completed(source)),
        api.OperationEvent(2, "stopped"),
    )

    assert panel.document.project is concurrent
    assert "stale" in panel.status_text()
    assert not panel.controller.is_running
