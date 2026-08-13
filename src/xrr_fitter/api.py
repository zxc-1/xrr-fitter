"""The complete supported Python API for XRR project workflows."""

from __future__ import annotations

from functools import wraps

from xrr_fitter.model.analysis import (
    ConfidenceClass,
    FitResult,
    McmcConfig,
    McmcReport,
    ParameterProfile,
    SldUncertaintyBands,
    StructureEvidence,
    UncertaintyReport,
)
from xrr_fitter.model.automation import (
    AutomaticDatasetSummary,
    AutomaticLayerResult,
    AutomaticResultSummary,
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    ImportBatchPreview,
    ImportFailure,
    ImportFilePreview,
    LayerUniformitySummary,
    MeasurementPreset,
)
from xrr_fitter.model.data import BeamSpec, DataColumnMapping, PreparedData
from xrr_fitter.model.export import ExportManifest
from xrr_fitter.model.fitting import FitConfig, FitProgress
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.operations import (
    FitReadiness,
    OperationError,
    OperationEvent,
    ProjectFitResult,
    ProjectImportResult,
)
from xrr_fitter.model.parameters import (
    JointFitLayout,
    ParameterDefinition,
    ParameterPrior,
    ParameterReference,
    ParameterSetting,
    PriorSpec,
    SharingRule,
)
from xrr_fitter.model.project import (
    DatasetProject,
    OxideDecision,
    ProjectUiState,
    ProjectValidation,
    ScalePriorState,
    SourceUpdatePreview,
    ValidationIssue,
    XrrProject,
)
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    InterfaceTransition,
    LayerSpec,
    MaterialSpec,
    OxideSuggestion,
    PeriodicBlock,
    StructureSpec,
    TransitionBranch,
)
from xrr_fitter.services.datasets import (
    add_dataset,
    import_data,
    import_dataset_batch,
    preview_import_batch,
    preview_source_update,
    remove_dataset,
)
from xrr_fitter.services.datasets import (
    set_fit_mask as _set_fit_mask,
)
from xrr_fitter.services.datasets import (
    set_instrument as _set_instrument,
)
from xrr_fitter.services.exports import export_result
from xrr_fitter.services.fitting import (
    _reconcile_parameter_sidecars,
    fit_automatically,
    fit_project,
    preflight_automatic_fit,
    preflight_fit,
    run_mcmc,
    sld_uncertainty_bands,
)
from xrr_fitter.services.parameters import (
    accept_source_update,
    describe_parameters,
    set_parameter_priors,
    set_parameter_settings,
    set_sharing_rules,
    validate_parameter_priors,
    validate_parameter_settings,
    validate_sharing_rules,
)
from xrr_fitter.services.projects import (
    clear_fit_results,
    describe_joint_layout,
    inspect_sources,
    load_project,
    new_project,
    save_project,
    select_active_dataset,
    select_candidate,
    set_batch_mode,
    set_dock_state,
    set_expert_mode,
    set_workspace_state,
)
from xrr_fitter.services.results import summarize_automatic_results
from xrr_fitter.services.structures import (
    accept_oxide_suggestion,
    analyze_structure,
    record_oxide_decision,
    set_structure,
    suggest_oxide_layers,
    validate_structure,
)
from xrr_fitter.services.workers import (
    OperationJob,
    start_automatic_fit_job,
    start_fit_job,
    start_mcmc_job,
)


@wraps(_set_fit_mask)
def set_fit_mask(project, dataset_id, mask):
    """Persist a fit mask and reconcile all declaration-bound sidecars."""
    return _reconcile_parameter_sidecars(
        _set_fit_mask(project, dataset_id, mask),
        dataset_id,
    )


def set_instrument(
    project: XrrProject,
    dataset_id: str,
    instrument: InstrumentSpec,
) -> XrrProject:
    """Persist an instrument and reconcile all declaration-bound sidecars."""
    return _reconcile_parameter_sidecars(
        _set_instrument(project, dataset_id, instrument),
        dataset_id,
    )


__all__ = (
    "AutomaticDatasetSummary",
    "AutomaticLayerResult",
    "AutomaticResultSummary",
    "AutomaticRole",
    "AutomaticStatus",
    "BeamSpec",
    "ConfidenceClass",
    "DataColumnMapping",
    "DatasetAutomation",
    "DatasetProject",
    "ExportManifest",
    "FitConfig",
    "FitProgress",
    "FitReadiness",
    "FitResult",
    "GradientLayerSpec",
    "ImportBatchPreview",
    "ImportFilePreview",
    "ImportFailure",
    "InstrumentSpec",
    "InterfaceTransition",
    "JointFitLayout",
    "LayerSpec",
    "LayerUniformitySummary",
    "MaterialSpec",
    "MeasurementPreset",
    "McmcConfig",
    "McmcReport",
    "OperationError",
    "OperationEvent",
    "OperationJob",
    "OxideDecision",
    "OxideSuggestion",
    "ParameterDefinition",
    "ParameterPrior",
    "ParameterProfile",
    "ParameterReference",
    "ParameterSetting",
    "PeriodicBlock",
    "PreparedData",
    "PriorSpec",
    "ProjectFitResult",
    "ProjectImportResult",
    "ProjectUiState",
    "ProjectValidation",
    "ScalePriorState",
    "SharingRule",
    "SldUncertaintyBands",
    "SourceUpdatePreview",
    "StructureEvidence",
    "StructureSpec",
    "TransitionBranch",
    "UncertaintyReport",
    "ValidationIssue",
    "XrrProject",
    "accept_oxide_suggestion",
    "accept_source_update",
    "add_dataset",
    "analyze_structure",
    "clear_fit_results",
    "describe_joint_layout",
    "describe_parameters",
    "export_result",
    "fit_project",
    "fit_automatically",
    "import_data",
    "import_dataset_batch",
    "inspect_sources",
    "load_project",
    "new_project",
    "preflight_fit",
    "preflight_automatic_fit",
    "preview_import_batch",
    "preview_source_update",
    "record_oxide_decision",
    "remove_dataset",
    "run_mcmc",
    "save_project",
    "select_active_dataset",
    "select_candidate",
    "set_batch_mode",
    "set_dock_state",
    "set_expert_mode",
    "set_fit_mask",
    "set_instrument",
    "set_parameter_priors",
    "set_parameter_settings",
    "set_sharing_rules",
    "set_structure",
    "set_workspace_state",
    "sld_uncertainty_bands",
    "start_fit_job",
    "start_automatic_fit_job",
    "start_mcmc_job",
    "suggest_oxide_layers",
    "summarize_automatic_results",
    "validate_parameter_priors",
    "validate_parameter_settings",
    "validate_sharing_rules",
    "validate_structure",
)
