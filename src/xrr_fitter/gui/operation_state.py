"""Project fit, MCMC, export, and external-operation action projection."""

from __future__ import annotations

from PySide6.QtWidgets import QAbstractButton

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


def _window_action(window: object, object_name: str) -> object | None:
    """Find an action installed by either the project panel or the chrome."""
    for action in (*window.actions(), *window.chrome_actions.values()):
        if action.objectName() == object_name:
            return action
    return None


def _window_button(window: object, object_name: str) -> QAbstractButton | None:
    """Find a named homepage button, including plot empty-state buttons."""
    return window.findChild(QAbstractButton, object_name)


# 运行中关掉的入口：凡是能改项目或数据源的都在这里。按「怎么找到它」分组，因为按钮可能还没
# 装上（空状态页）、动作分别由项目面板与外壳装，找不到的一律跳过。
MUTATING_BUTTONS = (
    "newProjectButton",
    "openProjectButton",
    "saveProjectButton",
    "emptyStateImportButton",
    "emptyStateImportFolderButton",
)
MUTATING_ACTIONS = (
    "newProjectAction",
    "openProjectAction",
    "saveProjectAction",
    "saveAsProjectAction",
    "importFilesMenuAction",
    "importFolderMenuAction",
    "undoAction",
    "redoAction",
)
SOURCE_ACTIONS = ("reloadSourceAction", "relinkSourceAction")


def _set_all_enabled(items: object, enabled: bool) -> None:
    for item in items:
        if item is not None:
            item.setEnabled(enabled)


def _set_mutating_routes_enabled(window: object, *, enabled: bool) -> None:
    """Gate every homepage entrance that can change the project or its sources."""
    data_panel = window.data_panel
    _set_all_enabled(
        (
            window.structure_panel,
            window.parameters_panel,
            window.parameter_disposition,
            window.selected_layer_panel,
            window.batch_mode_group,
            data_panel.dataset_add_button,
            data_panel.import_folder_quick_button,
            data_panel.import_files_button,
            data_panel.import_folder_button,
            data_panel.change_preset_button,
        ),
        enabled,
    )
    window.plot_panel.set_editing_enabled(enabled)
    _set_all_enabled((_window_button(window, name) for name in MUTATING_BUTTONS), enabled)
    _set_all_enabled((_window_action(window, name) for name in MUTATING_ACTIONS), enabled)
    active_dataset = window.document.active_dataset_id is not None
    _set_all_enabled((_window_action(window, name) for name in SOURCE_ACTIONS), enabled and active_dataset)


def _restore_idle_projection(window: object, *, running: bool, fit_enabled: bool) -> None:
    """Restore idle-only controls without touching the live operation controls."""
    if running:
        return
    window.fit_panel.start_button.setEnabled(fit_enabled)
    for object_name, enabled in (
        ("undoAction", window.document.can_undo),
        ("redoAction", window.document.can_redo),
    ):
        action = _window_action(window, object_name)
        if action is not None:
            action.setEnabled(enabled)


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
    running = any((fit_running, mcmc_running, external_running))
    readiness = window._fit_readiness()
    fit_enabled = readiness.ready and not window.guidance_is_visible()
    window._workflow_actions["startFitAction"].setEnabled(fit_enabled and not running)
    window._workflow_actions["cancelFitAction"].setEnabled(window.fit_panel.can_request_stop())
    export_enabled = has_exportable_results(window) and not running
    window._workflow_actions["exportResultsAction"].setEnabled(export_enabled)
    window.export_button.setEnabled(export_enabled)
    window.fit_panel.setEnabled(not mcmc_running and not external_running)
    window.result_panel.setEnabled(not fit_running and not external_running)
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
    # ``refresh_status`` restores source actions from the active-dataset state, so
    # apply the operation gate after all status projections have run.
    _set_mutating_routes_enabled(window, enabled=not running)
    _restore_idle_projection(window, running=running, fit_enabled=fit_enabled)
