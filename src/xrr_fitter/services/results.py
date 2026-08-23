"""Automatic per-layer projections and batch uniformity summaries."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt

from xrr_fitter.model.automation import (
    AutomaticDatasetSummary,
    AutomaticLayerResult,
    AutomaticResultSummary,
    AutomaticRole,
    LayerUniformitySummary,
)
from xrr_fitter.model.project import DatasetProject, XrrProject
from xrr_fitter.model.structure import LayerSpec
from xrr_fitter.physics.materials import (
    CLASSICAL_ELECTRON_RADIUS_A,
    material_sld,
)

UNKNOWN_DENSITY_NOTE = "配比未知，无法换算"


def _automatic_datasets(
    project: XrrProject,
    import_batch_id: str | None,
) -> tuple[DatasetProject, ...]:
    return tuple(
        dataset
        for dataset in project.datasets
        if dataset.automation.role is not AutomaticRole.MANUAL
        and (import_batch_id is None or dataset.automation.import_batch_id == import_batch_id)
    )


def _parameter_map(dataset: DatasetProject) -> dict[str, float] | None:
    result = dataset.last_valid_result
    candidate = None if result is None else result.best_candidate
    if candidate is None:
        return None
    return {parameter.name: parameter.value for parameter in candidate.parameters}


def _layer_value(
    dataset: DatasetProject,
    parameters: dict[str, float],
    name: str,
) -> float:
    if name in parameters:
        return parameters[name]
    setting = next(
        (value for value in dataset.parameter_settings if value.name == name and value.locked),
        None,
    )
    if setting is None:
        raise ValueError(f"automatic result candidate is missing parameter: {name}")
    return setting.initial


def _direct_sld(
    dataset: DatasetProject,
    layer: LayerSpec,
    parameters: dict[str, float],
    prefix: str,
) -> complex:
    declared = layer.material.sld_override_a2
    assert declared is not None
    real_name = f"{prefix}.sld_real_a2"
    imag_name = f"{prefix}.sld_imag_a2"
    real = parameters.get(real_name)
    imag = parameters.get(imag_name)
    if real is None:
        real = _layer_value(dataset, parameters, real_name)
    if imag is None:
        locked = next(
            (value.initial for value in dataset.parameter_settings if value.name == imag_name and value.locked),
            declared.imag,
        )
        imag = locked
    return complex(real, imag)


def _layer_result(
    dataset: DatasetProject,
    layer: LayerSpec,
    layer_index: int,
    parameters: dict[str, float],
) -> AutomaticLayerResult:
    prefix = f"component.{layer_index}"
    thickness = _layer_value(dataset, parameters, f"{prefix}.thickness_a")
    roughness = _layer_value(dataset, parameters, f"{prefix}.roughness_a")
    material = layer.material
    if material.sld_override_a2 is None:
        density_scale = _layer_value(
            dataset,
            parameters,
            f"{prefix}.density_scale",
        )
        sld = material_sld(
            material,
            density_scale,
            dataset.beam.effective_wavelength_a,
        )
        nominal_density = material.bulk_density_g_cm3
        assert nominal_density is not None
        fitted_density = nominal_density * density_scale
        density_note = None
    else:
        density_scale = 1.0
        sld = _direct_sld(dataset, layer, parameters, prefix)
        nominal_density = None
        fitted_density = None
        density_note = UNKNOWN_DENSITY_NOTE
    return AutomaticLayerResult(
        dataset_id=dataset.dataset_id,
        layer_index=layer_index,
        material_name=material.name,
        thickness_a=thickness,
        roughness_a=roughness,
        sld_real_a2=sld.real,
        sld_imag_a2=sld.imag,
        electron_density_a3=sld.real / CLASSICAL_ELECTRON_RADIUS_A,
        nominal_density_g_cm3=nominal_density,
        density_scale=density_scale,
        fitted_density_g_cm3=fitted_density,
        density_note=density_note,
    )


def _dataset_summary(dataset: DatasetProject) -> AutomaticDatasetSummary:
    structure = dataset.structure
    if structure is None:
        raise ValueError(f"automatic result requires a structure: {dataset.dataset_id}")
    # Per-layer rows assume the flat filename-derived layers the automatic route
    # builds. Anything else still deserves a status row: this runs on every
    # results refresh, so raising here would tear down the panel from a Qt slot.
    flat = all(isinstance(component, LayerSpec) for component in structure.components)
    parameters = _parameter_map(dataset)
    layers = (
        ()
        if parameters is None or not flat
        else tuple(_layer_result(dataset, layer, index, parameters) for index, layer in enumerate(structure.components))
    )
    return AutomaticDatasetSummary(
        dataset_id=dataset.dataset_id,
        status=dataset.automation.status,
        statistics_member=dataset.automation.statistics_member,
        reason=dataset.automation.reason,
        layers=layers,
    )


def _uniformity_row(
    key: tuple[str, int, str],
    values: Iterable[float],
) -> LayerUniformitySummary:
    thicknesses = tuple(values)
    mean = sum(thicknesses) / len(thicknesses)
    population_std = sqrt(sum((value - mean) ** 2 for value in thicknesses) / len(thicknesses))
    cv_percent = 0.0 if mean == 0.0 else population_std / mean * 100.0
    relative_range = 0.0 if mean == 0.0 else (max(thicknesses) - min(thicknesses)) / mean * 100.0
    fit_group_id, layer_index, material_name = key
    return LayerUniformitySummary(
        fit_group_id=fit_group_id,
        layer_index=layer_index,
        material_name=material_name,
        count=len(thicknesses),
        mean_thickness_a=mean,
        minimum_thickness_a=min(thicknesses),
        maximum_thickness_a=max(thicknesses),
        population_std_a=population_std,
        cv_percent=cv_percent,
        relative_range_percent=relative_range,
    )


def _uniformity(
    datasets: tuple[DatasetProject, ...],
    summaries: tuple[AutomaticDatasetSummary, ...],
) -> tuple[LayerUniformitySummary, ...]:
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for dataset, summary in zip(datasets, summaries, strict=True):
        group_id = dataset.automation.fit_group_id
        if not summary.statistics_member or group_id is None:
            continue
        for layer in summary.layers:
            key = (group_id, layer.layer_index, layer.material_name)
            grouped.setdefault(key, []).append(layer.thickness_a)
    return tuple(_uniformity_row(key, values) for key, values in grouped.items())


def summarize_automatic_results(
    project: XrrProject,
    import_batch_id: str | None = None,
) -> AutomaticResultSummary:
    """Project automatic results into per-layer values and batch statistics."""
    datasets = _automatic_datasets(project, import_batch_id)
    summaries = tuple(_dataset_summary(dataset) for dataset in datasets)
    return AutomaticResultSummary(
        import_batch_id=import_batch_id,
        datasets=summaries,
        uniformity=_uniformity(datasets, summaries),
    )
