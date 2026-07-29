from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest


PUBLIC_NAMES = (
    "BeamSpec",
    "DataColumnMapping",
    "DatasetProject",
    "ExportManifest",
    "FitConfig",
    "FitProgress",
    "FitReadiness",
    "FitResult",
    "GradientLayerSpec",
    "InstrumentSpec",
    "LayerSpec",
    "MaterialSpec",
    "McmcConfig",
    "McmcReport",
    "OperationError",
    "OperationEvent",
    "OperationJob",
    "OxideDecision",
    "OxideSuggestion",
    "ParameterDefinition",
    "ParameterProfile",
    "ParameterReference",
    "ParameterSetting",
    "PeriodicBlock",
    "PreparedData",
    "ProjectFitResult",
    "ProjectUiState",
    "ProjectValidation",
    "ScalePriorState",
    "SharingRule",
    "SourceUpdatePreview",
    "StructureEvidence",
    "StructureSpec",
    "UncertaintyReport",
    "ValidationIssue",
    "XrrProject",
    "accept_oxide_suggestion",
    "accept_source_update",
    "add_dataset",
    "analyze_structure",
    "clear_fit_results",
    "describe_parameters",
    "export_result",
    "fit_project",
    "import_data",
    "inspect_sources",
    "load_project",
    "new_project",
    "preflight_fit",
    "preview_source_update",
    "record_oxide_decision",
    "remove_dataset",
    "run_mcmc",
    "save_project",
    "select_active_dataset",
    "select_candidate",
    "set_batch_mode",
    "set_expert_mode",
    "set_fit_mask",
    "set_instrument",
    "set_parameter_settings",
    "set_sharing_rules",
    "set_structure",
    "set_workspace_state",
    "start_fit_job",
    "start_mcmc_job",
    "suggest_oxide_layers",
    "validate_parameter_settings",
    "validate_sharing_rules",
    "validate_structure",
)


SIGNATURES = {
    "accept_oxide_suggestion": "(project: 'XrrProject', dataset_id: 'str', suggestion: 'OxideSuggestion') -> 'XrrProject'",
    "accept_source_update": "(project: 'XrrProject', preview: 'SourceUpdatePreview') -> 'XrrProject'",
    "analyze_structure": "(project: 'XrrProject', dataset_id: 'str') -> 'StructureEvidence'",
    "clear_fit_results": "(project: 'XrrProject', dataset_ids: 'Sequence[str]') -> 'XrrProject'",
    "describe_parameters": "(project: 'XrrProject', dataset_id: 'str') -> 'tuple[ParameterDefinition, ...]'",
    "import_data": "(path: 'str | Path', beam: 'BeamSpec', import_angle_offset_deg: 'float' = 0.0, column_mapping: 'DataColumnMapping | None' = None) -> 'PreparedData'",
    "inspect_sources": "(project: 'XrrProject') -> 'ProjectValidation'",
    "load_project": "(path: 'str | Path') -> 'XrrProject'",
    "new_project": "() -> 'XrrProject'",
    "add_dataset": "(project: 'XrrProject', source_path: 'str | Path', instrument: 'InstrumentSpec', display_name: 'str | None' = None, column_mapping: 'DataColumnMapping | None' = None, import_angle_offset_deg: 'float' = 0.0, beam: 'BeamSpec | None' = None) -> 'XrrProject'",
    "preview_source_update": "(project: 'XrrProject', dataset_id: 'str', new_path: 'str | Path | None' = None) -> 'SourceUpdatePreview'",
    "record_oxide_decision": "(project: 'XrrProject', dataset_id: 'str', decision: 'OxideDecision') -> 'XrrProject'",
    "remove_dataset": "(project: 'XrrProject', dataset_id: 'str') -> 'XrrProject'",
    "save_project": "(project: 'XrrProject', path: 'str | Path') -> 'None'",
    "select_active_dataset": "(project: 'XrrProject', dataset_id: 'str | None') -> 'XrrProject'",
    "select_candidate": "(project: 'XrrProject', dataset_id: 'str', candidate_id: 'str | None') -> 'XrrProject'",
    "set_batch_mode": "(project: 'XrrProject', mode: \"Literal['independent', 'joint']\") -> 'XrrProject'",
    "set_expert_mode": "(project: 'XrrProject', enabled: 'bool') -> 'XrrProject'",
    "set_fit_mask": "(project: 'XrrProject', dataset_id: 'str', mask: 'np.ndarray') -> 'XrrProject'",
    "set_instrument": "(project: 'XrrProject', dataset_id: 'str', instrument: 'InstrumentSpec') -> 'XrrProject'",
    "set_parameter_settings": "(project: 'XrrProject', dataset_id: 'str', settings: 'Sequence[ParameterSetting]') -> 'XrrProject'",
    "set_sharing_rules": "(project: 'XrrProject', rules: 'Sequence[SharingRule]') -> 'XrrProject'",
    "set_structure": "(project: 'XrrProject', dataset_id: 'str', structure: 'StructureSpec') -> 'XrrProject'",
    "set_workspace_state": "(project: 'XrrProject', state: 'ProjectUiState') -> 'XrrProject'",
    "suggest_oxide_layers": "(structure: 'StructureSpec') -> 'tuple[OxideSuggestion, ...]'",
    "validate_parameter_settings": "(definitions: 'Sequence[ParameterDefinition]', settings: 'Sequence[ParameterSetting]') -> 'tuple[ParameterSetting, ...]'",
    "validate_sharing_rules": "(project: 'XrrProject', rules: 'Sequence[SharingRule]') -> 'tuple[SharingRule, ...]'",
    "validate_structure": "(structure: 'StructureSpec', beam: 'BeamSpec') -> 'None'",
    "preflight_fit": "(project: 'XrrProject') -> 'FitReadiness'",
    "fit_project": "(project: 'XrrProject', progress_callback: 'ProgressCallback | None' = None, checkpoint_callback: 'CheckpointCallback | None' = None) -> 'ProjectFitResult'",
    "run_mcmc": "(project: 'XrrProject', dataset_id: 'str', candidate_id: 'str', config: 'McmcConfig', progress_callback: 'ProgressCallback | None' = None) -> 'XrrProject'",
    "start_fit_job": "(project: 'XrrProject', checkpoint_path: 'str | Path | None' = None) -> 'OperationJob'",
    "start_mcmc_job": "(project: 'XrrProject', dataset_id: 'str', candidate_id: 'str', config: 'McmcConfig') -> 'OperationJob'",
    "export_result": "(result: 'XrrProject | ProjectFitResult', output_dir: 'str | Path') -> 'ExportManifest'",
}


GUI_USE_CASES = {
    "project_files": ("new_project", "load_project", "save_project"),
    "datasets": ("add_dataset", "remove_dataset"),
    "sources": ("preview_source_update", "accept_source_update"),
    "mask": ("set_fit_mask",),
    "instrument": ("set_instrument",),
    "structure": ("set_structure",),
    "oxide": ("accept_oxide_suggestion", "record_oxide_decision"),
    "parameters": ("set_parameter_settings",),
    "sharing": ("validate_sharing_rules", "set_sharing_rules"),
    "batch": ("set_batch_mode",),
    "selection": ("select_active_dataset", "select_candidate"),
    "workspace": ("set_expert_mode", "set_workspace_state"),
    "results": ("clear_fit_results",),
    "fit": ("preflight_fit", "fit_project"),
    "fit_job": ("start_fit_job",),
    "mcmc": ("run_mcmc", "start_mcmc_job"),
    "export": ("export_result",),
}


def _gui_sources() -> tuple[tuple[Path, str], ...]:
    root = Path(__file__).resolve().parents[2] / "src/xrr_fitter/gui"
    return tuple(
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.py"))
    )


def _api_aliases(tree: ast.AST) -> set[str]:
    return {
        alias.asname or alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "xrr_fitter.api"
    }


def test_api_exports_only_the_complete_supported_surface() -> None:
    import xrr_fitter.api as api

    assert tuple(api.__all__) == PUBLIC_NAMES
    assert all(hasattr(api, name) for name in PUBLIC_NAMES)
    assert len({id(getattr(api, name)) for name in PUBLIC_NAMES}) == len(PUBLIC_NAMES)


def test_every_public_operation_has_an_exact_signature() -> None:
    import xrr_fitter.api as api

    operations = {
        name for name in PUBLIC_NAMES if inspect.isfunction(getattr(api, name))
    }
    assert set(SIGNATURES) == operations


@pytest.mark.parametrize(("name", "expected"), SIGNATURES.items())
def test_public_use_case_signatures_are_exact(name: str, expected: str) -> None:
    import xrr_fitter.api as api

    assert str(inspect.signature(getattr(api, name))) == expected


def test_operation_job_has_only_one_concrete_process_owner() -> None:
    import xrr_fitter.api as api
    from xrr_fitter.services import workers

    assert api.OperationJob is workers.OperationJob
    assert inspect.isclass(api.OperationJob)
    assert api.OperationJob.pid.fget is not None
    for name in ("poll", "cancel", "force_stop", "close"):
        assert callable(getattr(api.OperationJob, name))


def test_gui_use_case_fixture_resolves_domain_calls_only_through_public_api() -> None:
    mapped = tuple(name for names in GUI_USE_CASES.values() for name in names)
    source = "from xrr_fitter.api import " + ", ".join(mapped)
    tree = ast.parse(source)
    imports = tuple(node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))

    assert len(imports) == 1
    assert imports[0].module == "xrr_fitter.api"
    assert tuple(alias.name for alias in imports[0].names) == mapped
    assert set(mapped) <= set(PUBLIC_NAMES)


def test_package_initializers_remain_empty_and_do_not_reexport_api() -> None:
    root = Path(__file__).resolve().parents[2] / "src/xrr_fitter"

    assert (root / "__init__.py").read_bytes() == b""
    assert (root / "services/__init__.py").read_bytes() == b""
    import xrr_fitter

    assert not hasattr(xrr_fitter, "fit_project")
