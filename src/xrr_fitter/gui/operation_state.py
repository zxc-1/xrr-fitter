"""Project fit, MCMC, export, and external-operation action projection."""

from __future__ import annotations

from xrr_fitter.gui.chrome import refresh_status


def operation_controllers(window: object) -> tuple[object, ...]:
    values = (
        window.fit_panel.controller,
        window.result_panel.controller,
        window._operation_controller,
    )
    unique: list[object] = []
    for controller in values:
        if controller is not None and all(controller is not item for item in unique):
            unique.append(controller)
    return tuple(unique)


def operation_is_running(window: object) -> bool:
    return any(bool(controller.is_running) for controller in operation_controllers(window))


def has_exportable_results(window: object) -> bool:
    datasets = window.document.project.datasets
    return bool(datasets) and all(
        dataset.last_valid_result is not None for dataset in datasets
    )


def refresh_operation_state(window: object) -> None:
    fit_running = window.fit_panel.controller.is_running
    mcmc_running = window.result_panel.controller.is_running
    external = window._operation_controller
    external_running = external is not None and bool(external.is_running)
    running = fit_running or mcmc_running or external_running
    readiness = window._fit_readiness()
    window._workflow_actions["startFitAction"].setEnabled(
        readiness.ready and not running
    )
    window._workflow_actions["cancelFitAction"].setEnabled(fit_running)
    export_enabled = has_exportable_results(window) and not running
    window._workflow_actions["exportResultsAction"].setEnabled(export_enabled)
    window.export_button.setEnabled(export_enabled)
    window.fit_panel.setEnabled(not mcmc_running and not external_running)
    window.result_panel.setEnabled(not fit_running and not external_running)
    if not running:
        window.fit_panel.start_button.setEnabled(readiness.ready)
    refresh_status(window, readiness)
