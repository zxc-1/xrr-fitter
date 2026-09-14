"""Project fit, MCMC, export, and external-operation action projection."""

from __future__ import annotations

from xrr_fitter.gui.chrome import refresh_status, set_command_bar_running


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
    return bool(datasets) and all(dataset.last_valid_result is not None for dataset in datasets)


def running_kind(*, fit: bool, mcmc: bool, external: bool) -> str | None:
    """Name the live operation for the status bar, or ``None`` when idle.

    All three flags disable 开始拟合, so the button alone cannot say which one is
    holding the window. Reporting the fit first matches how the panels are
    interlocked: a fit and an MCMC run cannot both be live.
    """
    if fit:
        return "fit"
    if mcmc:
        return "mcmc"
    if external:
        return "external"
    return None


def refresh_operation_state(window: object) -> None:
    fit_running = window.fit_panel.controller.is_running
    mcmc_running = window.result_panel.controller.is_running
    external = window._operation_controller
    external_running = external is not None and bool(external.is_running)
    running = fit_running or mcmc_running or external_running
    readiness = window._fit_readiness()
    window._workflow_actions["startFitAction"].setEnabled(readiness.ready and not running)
    window._workflow_actions["cancelFitAction"].setEnabled(fit_running)
    export_enabled = has_exportable_results(window) and not running
    window._workflow_actions["exportResultsAction"].setEnabled(export_enabled)
    window.export_button.setEnabled(export_enabled)
    window.fit_panel.setEnabled(not mcmc_running and not external_running)
    window.result_panel.setEnabled(not fit_running and not external_running)
    if not running:
        window.fit_panel.start_button.setEnabled(readiness.ready)
    # 帧④：跑起来之后命令栏换成 ⏸ 暂停 / ⏹ 停止，灰着的 一键拟合/导出 让位。
    set_command_bar_running(window, fit_running)
    # 帧⑤ 左栏第四行：链在采时钉成「N walkers · 采样中」，停了交还给报告。两头都在这里推，
    # 因为「在跑」不是报告里的字段——报告要等采完才有 ``mcmc``，只推起来那一头的话，链停了
    # 半小时左栏还在报采样中。walkers 数读 spin box：运行期间它是禁用的，所以就是这条活链的数。
    window.pipeline_nav.set_sampling_walkers(window.result_panel.walkers.value() if mcmc_running else None)
    refresh_status(
        window,
        readiness,
        running_kind(fit=fit_running, mcmc=mcmc_running, external=external_running),
    )
