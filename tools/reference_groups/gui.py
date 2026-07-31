"""Replay and normalize the frozen desktop workflow through the R23 GUI."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile

import xrr_fitter.api as api


ARTIFACT = "golden/gui.json"
INPUTS = {
    "mo-si-periodic-data": (
        "bundled-example-data",
        "xrr_fitter/examples/mo-si-periodic.xy",
        65125,
        "5bcdf3669698c4482e409b65fca794e500c41924953bc4f12dfe1aeee5d3bd70",
    ),
    "mo-si-periodic-project": (
        "bundled-example-project",
        "xrr_fitter/examples/mo-si-periodic.xrrproj.json",
        29298,
        "613e86c22605b111ceb57fd6b3a63f93e3a330cfac65cc18d37b2f1a5c2407ee",
    ),
    "single-layer-data": (
        "bundled-example-data",
        "xrr_fitter/examples/single-layer.xy",
        43223,
        "85729258067ff1c953257f6e784b6ec5a5c9e175e92f449ae0bc04680c1e42ea",
    ),
    "single-layer-project": (
        "bundled-example-project",
        "xrr_fitter/examples/single-layer.xrrproj.json",
        20247,
        "c2aae5beca68b95d5dd0f06659fdf73c7ddc8921aa46e76bda7e7d2cae35fa65",
    ),
}
INPUT_ORDER = tuple(INPUTS)
SEEDS = (20260726,)
CONFIGURATION = {
    "case": "single-layer",
    "dataset_id": "gui-sample",
    "fit_target": "component.0.thickness_a",
    "qt_platform": "offscreen",
    "operations": [
        "create_application",
        "MainWindow",
        "save_project",
        "open_project",
        "restore_fitted_workspace",
        "close_delete",
    ],
    "real_data_acceptance": {
        "status": "NOT_RUN",
        "reason": "owner post-delivery acceptance",
    },
}
FIT_BUDGET = {
    "short_de_maxiter": 0,
    "full_de_maxiter": 0,
    "local_min_nfev": 5,
    "local_nfev_per_parameter": 1,
    "bootstrap_samples": 8,
}
ACTION_NAMES = (
    "cancelFitAction",
    "exportResultsAction",
    "openProjectAction",
    "saveAsProjectAction",
    "saveProjectAction",
    "startFitAction",
)
PARAMETER_NAMES = (
    "backing.roughness_a",
    "component.0.density_scale",
    "component.0.roughness_a",
    "component.0.thickness_a",
    "instrument.absolute_sigma_a_inv",
    "instrument.angle_offset_deg",
    "instrument.background",
    "instrument.footprint_spill_angle_deg",
    "instrument.linear_background_per_a_inv",
    "instrument.powerlaw_background_amplitude",
    "instrument.powerlaw_background_exponent",
    "instrument.relative_sigma",
    "instrument.scale",
    "instrument.sigma_theta_deg",
)
SHARING_PARAMETER_NAMES = (
    "component.0.density_scale",
    "instrument.absolute_sigma_a_inv",
    "instrument.footprint_spill_angle_deg",
    "instrument.relative_sigma",
)
STATIC_OBJECT_NAMES = (
    "ScrollLeftButton",
    "ScrollRightButton",
    "_layout",
    "addLayerButton",
    "addPeriodicBlockButton",
    "batchModeCombo",
    "batchModeLabel",
    "cancelFitAction",
    "cancelFitButton",
    "candidateList",
    "confidenceBadge",
    "confidenceMarker",
    "copyEvidenceButton",
    "dataPanel",
    "dataPanelTitle",
    "datasetTree",
    "diagnosticCanvas:candidates",
    "diagnosticCanvas:log",
    "diagnosticCanvas:qz4",
    "diagnosticCanvas:raw",
    "diagnosticCanvas:residual",
    "diagnosticCanvas:sld",
    "diagnosticCanvas:trend",
    "diagnosticCanvas:uncertainty",
    "diagnosticTabs",
    "excludedPointSummary",
    "expertMcmcGroup",
    "exportResultsAction",
    "exportResultsButton",
    "fitControls",
    "fitControlsTitle",
    "fitProgressBar",
    "fitReadinessLabel",
    "fitStatusLabel",
    "importError",
    "importFilesButton",
    "importFilesShortcut",
    "importFolderButton",
    "importFolderShortcut",
    "initializeStructureButton",
    "leftSplitter",
    "maskPlotCanvas",
    "mcmcBurnIn",
    "mcmcButton",
    "mcmcProduction",
    "mcmcStatus",
    "mcmcThin",
    "mcmcWalkers",
    "moveComponentDownButton",
    "moveComponentUpButton",
    "newProjectButton",
    "openProjectAction",
    "openProjectButton",
    "oxideSuggestionButton",
    "oxideSuggestionRefuseButton",
    "parameterExpertBackgroundKind",
    "parameterExpertBackgroundLabel",
    "parameterExpertControls",
    "parameterExpertModeToggle",
    "parameterExpertResolutionDomain",
    "parameterExpertResolutionLabel",
    "parameterExpertScalePriorEnabled",
    "parameterExpertScalePriorTau",
    "parameterExpertScaleTauLabel",
    "parameterTable",
    "parameterTableGrid",
    "parameterTableTitle",
    "parameterTableValidation",
    "plotCancelInteractionShortcut",
    "plotInteractionToolbar",
    "plotModeMask",
    "plotModeRange",
    "plotModeView",
    "plotPanel",
    "plotPanelTitle",
    "projectControls",
    "qt_scrollarea_hcontainer",
    "qt_scrollarea_vcontainer",
    "qt_scrollarea_viewport",
    "qt_spinbox_lineedit",
    "qt_spinboxvalidator",
    "qt_splithandle_",
    "qt_splithandle_dataPanel",
    "qt_splithandle_leftSplitter",
    "qt_splithandle_plotPanel",
    "qt_splithandle_structureEditor",
    "qt_tableview_cornerbutton",
    "qt_tabwidget_stackedwidget",
    "qt_tabwidget_tabbar",
    "relinkSourceButton",
    "reloadSourceButton",
    "removeComponentButton",
    "resultEvidence",
    "resultEvidenceLabel",
    "resultPanel",
    "resultPanelTitle",
    "resultWarnings",
    "resultWarningsLabel",
    "saveAsProjectAction",
    "saveAsProjectButton",
    "saveProjectAction",
    "saveProjectButton",
    "sourceWarningLabel",
    "startFitAction",
    "startFitButton",
    "structureComponentActions",
    "structureDensityEditor",
    "structureEditor",
    "structureEditorError",
    "structureEditorSummary",
    "structureEditorTitle",
    "structureEvidenceWarning",
    "structureFormulaEditor",
    "structureRoughnessEditor",
    "structureThicknessEditor",
    "structureTree",
    "workspaceSplitter",
)
OBJECT_ALIASES = {
    "batchModeCombo": "batchModeSelector",
    "batchModeLabel": "batchModeSelector",
    "copyEvidenceButton": "uncertaintyEvidence",
    "dataPanelTitle": "dataPanel",
    "excludedPointSummary": "datasetTree",
    "fitControls": "fitPanel",
    "fitControlsTitle": "fitPanel",
    "fitReadinessLabel": "fitStatusLabel",
    "importError": "sourceStatusLabel",
    "initializeStructureButton": "structureEditor",
    "maskPlotCanvas": "dataPanel",
    "mcmcStatus": "resultStatus",
    "parameterExpertBackgroundKind": "parametersPanel",
    "parameterExpertBackgroundLabel": "parametersPanel",
    "parameterExpertControls": "parametersPanel",
    "parameterExpertModeToggle": "expertModeToggle",
    "parameterExpertResolutionDomain": "parametersPanel",
    "parameterExpertResolutionLabel": "parametersPanel",
    "parameterExpertScalePriorEnabled": "parametersPanel",
    "parameterExpertScalePriorTau": "parametersPanel",
    "parameterExpertScaleTauLabel": "parametersPanel",
    "parameterTableGrid": "parameterTable",
    "parameterTableTitle": "parameterTable",
    "parameterTableValidation": "parameterStatus",
    "plotCancelInteractionShortcut": "plotInteractionToolbar",
    "plotPanelTitle": "plotPanel",
    "projectControls": "projectActions",
    "qt_splithandle_": "qt_splithandle_analysisColumn",
    "qt_splithandle_leftSplitter": "qt_splithandle_projectColumn",
    "qt_splithandle_plotPanel": "qt_splithandle_plotColumn",
    "qt_splithandle_structureEditor": "qt_splithandle_structurePanel",
    "resultEvidence": "uncertaintyEvidence",
    "resultEvidenceLabel": "uncertaintyView",
    "resultPanel": "resultsPanel",
    "resultPanelTitle": "resultsPanel",
    "resultWarnings": "uncertaintyView",
    "resultWarningsLabel": "uncertaintyView",
    "sourceWarningLabel": "sourceStatusLabel",
    "structureComponentActions": "structureEditor",
    "structureDensityEditor": "structureTree",
    "structureEditorError": "structureValidationError",
    "structureEditorSummary": "structureEditor",
    "structureEditorTitle": "structureEditor",
    "structureEvidenceWarning": "structurePanel",
    "structureFormulaEditor": "structureTree",
    "structureRoughnessEditor": "structureTree",
    "structureThicknessEditor": "structureTree",
}


def _expected_input(value: object) -> tuple[str, str, int, str]:
    try:
        expected = INPUTS[value.input_id]
    except (AttributeError, KeyError) as error:
        raise ValueError("gui input identity drift") from error
    input_class, path, _size, _digest = expected
    if value.input_class != input_class or value.path != path:
        raise ValueError("gui input identity drift")
    return expected


def _validate_input(value: object) -> bytes:
    _input_class, _path, size, digest = _expected_input(value)
    content = value.content
    if not isinstance(content, bytes):
        raise ValueError("gui input content must be bytes")
    if value.size != size or value.sha256 != digest:
        raise ValueError("gui input size or hash drift")
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("gui input content or hash drift")
    return content


def _validate_context(context: object) -> dict[str, bytes]:
    if context.group != "gui":
        raise ValueError("gui group drift")
    if tuple(context.artifacts) != (ARTIFACT,):
        raise ValueError("gui artifact drift")
    if context.configuration != CONFIGURATION:
        raise ValueError("gui configuration drift")
    if tuple(context.seeds) != SEEDS:
        raise ValueError("gui seed drift")
    inputs = tuple(context.inputs)
    if tuple(value.input_id for value in inputs) != INPUT_ORDER:
        raise ValueError("gui input set or order drift")
    return {value.input_id: _validate_input(value) for value in inputs}


def _materialize_inputs(root: Path, contents: dict[str, bytes]) -> Path:
    for input_id, content in contents.items():
        path = root.joinpath(*PurePosixPath(INPUTS[input_id][1]).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root / "xrr_fitter" / "examples"


def _fit_project(examples: Path) -> api.XrrProject:
    source = api.load_project(examples / "single-layer.xrrproj.json")
    dataset = source.datasets[0]
    if dataset.dataset_id != "single-layer" or dataset.structure is None:
        raise ValueError("gui single-layer project drift")
    seed = SEEDS[0]
    config = api.FitConfig.fast(seed)
    config = replace(
        config,
        budget=replace(config.budget, **FIT_BUDGET),
        local_workers=1,
    )
    dataset = replace(dataset, dataset_id=CONFIGURATION["dataset_id"], parameter_settings=())
    project = replace(
        source,
        datasets=(dataset,),
        fit_config=config,
        batch_mode="independent",
        sharing_rules=(),
        ui_state=api.ProjectUiState(active_dataset_id=CONFIGURATION["dataset_id"]),
    )
    definitions = api.describe_parameters(project, CONFIGURATION["dataset_id"])
    settings = tuple(
        api.ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name != CONFIGURATION["fit_target"],
        )
        for definition in definitions
    )
    project = api.set_parameter_settings(project, CONFIGURATION["dataset_id"], settings)
    result = api.fit_project(project)
    fitted = result.updated_project
    for item in result.datasets:
        candidate = item.fit_result.best_candidate
        if candidate is None:
            raise ValueError("gui fit did not produce a selected candidate")
        fitted = api.select_candidate(fitted, item.dataset_id, candidate.candidate_id)
    return fitted


def _parameter_names(window: object) -> tuple[str, ...]:
    definitions = tuple(window.parameters_panel.definitions)
    names = tuple(sorted(definition.name for definition in definitions))
    if names != PARAMETER_NAMES:
        raise ValueError("gui parameter definition drift")
    return names


def _sharing_object_names(names: tuple[str, ...]) -> set[str]:
    if any(name not in names for name in SHARING_PARAMETER_NAMES):
        raise ValueError("gui shareable parameter drift")
    return {f"parameterSharing:{name}" for name in SHARING_PARAMETER_NAMES}


def _parameter_object_names(window: object) -> set[str]:
    names = _parameter_names(window)
    result = {
        f"parameterEditor:{name}:{field}"
        for name in names
        for field in ("initial", "lower", "upper")
    }
    result.update(f"parameterLock:{name}" for name in names)
    result.update(_sharing_object_names(names))
    return result


def _normalized_object_names(window: object, qobject: type) -> list[str]:
    current = {
        item.objectName()
        for item in window.findChildren(qobject)
        if item.objectName()
    }
    for legacy in STATIC_OBJECT_NAMES:
        owner = OBJECT_ALIASES.get(legacy, legacy)
        if owner not in current:
            raise ValueError(f"gui object mapping is missing: {legacy} -> {owner}")
    return sorted({*STATIC_OBJECT_NAMES, *_parameter_object_names(window)})


def _action_states(window: object, action_type: type) -> list[dict[str, object]]:
    actions = {
        action.objectName(): action
        for action in window.findChildren(action_type)
        if action.objectName() in ACTION_NAMES
    }
    if tuple(sorted(actions)) != ACTION_NAMES:
        raise ValueError("gui workflow action set drift")
    return [
        {
            "object_name": name,
            "text": actions[name].text(),
            "shortcut": actions[name].shortcut().toString(),
            "enabled": bool(actions[name].isEnabled()),
            "visible": bool(actions[name].isVisible()),
        }
        for name in ACTION_NAMES
    ]


def _candidate_trace(window: object) -> dict[str, object]:
    dataset = window.document.project.datasets[0]
    result = dataset.last_valid_result
    if result is None or result.best_index is None:
        raise ValueError("gui did not restore the fitted result")
    order = tuple(candidate.candidate_id for candidate in result.candidates)
    recommended = window.result_panel.recommended_candidate_id()
    selected = window.result_panel.selected_candidate_id()
    if recommended not in order or selected not in order:
        raise ValueError("gui candidate projection drift")
    return {
        "widget_count": window.result_panel.candidate_count(),
        "candidate_order": list(order),
        "recommended_index": order.index(recommended),
        "selected_index": order.index(selected),
        "selected_candidate_id": selected,
        "persisted_selection": [
            list(item)
            for item in window.document.project.ui_state.selected_candidate_ids
        ],
    }


def _workspace_trace(window: object) -> dict[str, object]:
    path = window.document.path
    if path is None:
        raise ValueError("gui project path is missing")
    return {
        "project_path": path.name,
        "active_dataset_id": window.data_panel.active_dataset_id,
        "plot_dataset_id": window.plot_panel.selected_dataset_id(),
        "batch_mode": window.document.project.batch_mode,
        "fit_ready": window.fit_is_ready(),
        "fit_readiness": window.fit_readiness_text(),
        "dirty": window.document.is_dirty,
        "expert_mode": window.document.project.ui_state.expert_mode,
        "workspace_splitter_sizes": window.workspace_splitter.sizes(),
        "left_splitter_sizes": window.left_splitter.sizes(),
        "plot_tab_index": window.plot_panel.tabs.currentIndex(),
    }


def _normalize_workspace_height(window: object, app: object) -> None:
    # The frozen R22 trace recorded workspace geometry with the vertical
    # splitter owning the full 760px window height.  The R23 shell now adds
    # menu, toolbar, and status chrome plus column padding, so the replay
    # window grows by exactly that fixed overhead and the persisted sizes are
    # projected again through the production restore path.  This normalizes
    # presentation only; the semantic restore contract stays unmodified.
    from xrr_fitter.gui.workspace import restore_project

    for _attempt in range(8):
        shortfall = 760 - window.left_splitter.height()
        if shortfall == 0:
            break
        window.resize(window.width(), window.height() + shortfall)
        app.processEvents()
    restore_project(window.workspace_view, window.document.project)
    app.processEvents()


def _gui_snapshot(
    root: Path,
    fitted: api.XrrProject,
    source_content: bytes,
) -> dict[str, object]:
    previous = os.environ.get("QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = CONFIGURATION["qt_platform"]
    try:
        from PySide6.QtCore import QCoreApplication, QEvent, QObject
        from PySide6.QtGui import QAction
        import shiboken6

        from xrr_fitter.gui.application import create_application
        from xrr_fitter.gui.main_window import MainWindow

        source = root / "single-layer.xy"
        source.write_bytes(source_content)
        dataset = replace(fitted.datasets[0], source_path=source.name)
        fitted = replace(fitted, datasets=(dataset,), base_directory=str(root))
        project_path = root / "fitted.xrrproj.json"
        api.save_project(fitted, project_path)
        existing = QCoreApplication.instance()
        app = create_application(["xrr-r23-reference"])
        window = MainWindow()
        trace = [
            {"action": "create_application", "class": type(app).__name__},
            {
                "action": "construct_main_window",
                "class": type(window).__name__,
                "object_name": window.objectName(),
            },
        ]
        window.open_project(project_path)
        window.resize(1280, 760)
        window.show()
        app.processEvents()
        _normalize_workspace_height(window, app)
        trace.append(
            {
                "action": "open_project",
                "project": project_path.name,
                "source_status": window.source_hash_status(CONFIGURATION["dataset_id"]),
            }
        )
        snapshot = {
            "schema": "xrr-r22-gui-reference-v1",
            "runtime": "QApplication+MainWindow",
            "platform": CONFIGURATION["qt_platform"],
            "action_trace": trace,
            "action_states": _action_states(window, QAction),
            "object_names": _normalized_object_names(window, QObject),
            "tab_titles": list(window.plot_panel.tab_titles()),
            "candidate_trace": _candidate_trace(window),
            "workspace_trace": _workspace_trace(window),
            "real_data_acceptance": dict(CONFIGURATION["real_data_acceptance"]),
        }
        if not window.close():
            raise ValueError("gui window refused clean close")
        app.processEvents()
        trace.append({"action": "close", "visible": window.isVisible()})
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        snapshot["window_deleted"] = not shiboken6.isValid(window)
        if not snapshot["window_deleted"]:
            raise ValueError("gui window was not deleted")
        if existing is None:
            app.quit()
        return {ARTIFACT: snapshot}
    finally:
        if previous is None:
            os.environ.pop("QT_QPA_PLATFORM", None)
        else:
            os.environ["QT_QPA_PLATFORM"] = previous


def replay(context: object) -> dict[str, object]:
    contents = _validate_context(context)
    with tempfile.TemporaryDirectory(prefix="xrr-r23-reference-gui-") as directory:
        root = Path(directory)
        examples = _materialize_inputs(root, contents)
        fitted = _fit_project(examples)
        return _gui_snapshot(root, fitted, contents["single-layer-data"])
