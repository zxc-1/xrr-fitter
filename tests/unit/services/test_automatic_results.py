from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    fit_candidate,
    project,
)

from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
)
from xrr_fitter.model.parameters import ParameterValue
from xrr_fitter.services.materials import automatic_structure
from xrr_fitter.services.results import summarize_automatic_results


def _parameter(name: str, value: float) -> ParameterValue:
    if name.endswith("sld_real_a2"):
        return ParameterValue(name, value, -150e-6, 150e-6)
    if name.endswith("sld_imag_a2"):
        return ParameterValue(name, value, 0.0, 20e-6)
    if name.endswith("density_scale"):
        return ParameterValue(name, value, 0.2, 1.2)
    if name.endswith("roughness_a"):
        return ParameterValue(name, value, 0.0, 100.0)
    return ParameterValue(name, value, 2.0, 2e5)


def _point(
    dataset_id: str,
    *,
    thicknesses: tuple[float, ...],
    direct_sld: float | None = None,
    passed: bool,
):
    tokens = ("Si3N4", "CrSiC") if len(thicknesses) == 2 else ("Zr",)
    structure, settings = automatic_structure(tokens, "Al2O3")
    parameters = []
    for index, (layer, thickness) in enumerate(
        zip(structure.components, thicknesses, strict=True)
    ):
        prefix = f"component.{index}"
        parameters.extend(
            (
                _parameter(f"{prefix}.thickness_a", thickness),
                _parameter(f"{prefix}.roughness_a", 3.0),
                _parameter(
                    f"{prefix}.density_scale",
                    1.0 if layer.material.sld_override_a2 is not None else 0.9,
                ),
            )
        )
        if layer.material.sld_override_a2 is not None:
            parameters.extend(
                (
                    _parameter(
                        f"{prefix}.sld_real_a2",
                        20e-6 if direct_sld is None else direct_sld,
                    ),
                    _parameter(f"{prefix}.sld_imag_a2", 0.0),
                )
            )
    candidate = replace(
        fit_candidate(f"{dataset_id}-candidate"),
        parameters=tuple(parameters),
    )
    status = AutomaticStatus.PASSED if passed else AutomaticStatus.REVIEW
    automation = DatasetAutomation(
        import_batch_id="batch-1",
        fit_group_id="group-1",
        role=AutomaticRole.JOINT,
        status=status,
        statistics_member=passed,
        reason=None if passed else "synthetic quality failure",
    )
    return replace(
        dataset_project(dataset_id, result=final_fit_result(candidate)),
        structure=structure,
        parameter_settings=settings,
        automation=automation,
    )


def _fitted_project(*points):
    return project(*points)


def test_known_and_unknown_material_results_do_not_confuse_mass_density() -> None:
    value = _fitted_project(
        _point("p1", thicknesses=(90.0, 100.0), direct_sld=24e-6, passed=True),
    )

    summary = summarize_automatic_results(value, "batch-1")

    known, unknown = summary.datasets[0].layers
    assert (
        known.fitted_density_g_cm3,
        unknown.nominal_density_g_cm3,
        unknown.fitted_density_g_cm3,
        unknown.density_note,
        unknown.electron_density_a3,
    ) == (
        pytest.approx(known.nominal_density_g_cm3 * known.density_scale),
        None,
        None,
        "配比未知，无法换算",
        pytest.approx(unknown.sld_real_a2 / 2.8179403262e-5),
    )


def test_uniformity_uses_only_passed_members_and_population_standard_deviation() -> None:
    value = _fitted_project(
        _point("p1", thicknesses=(90.0,), passed=True),
        _point("p2", thicknesses=(100.0,), passed=True),
        _point("p3", thicknesses=(500.0,), passed=False),
    )

    item = summarize_automatic_results(value, "batch-1").uniformity[0]

    assert (
        item.count,
        item.mean_thickness_a,
        item.population_std_a,
        item.cv_percent,
        item.relative_range_percent,
    ) == (
        2,
        95.0,
        5.0,
        pytest.approx(5.0 / 95.0 * 100.0),
        pytest.approx(10.0 / 95.0 * 100.0),
    )
