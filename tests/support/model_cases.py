from __future__ import annotations

from pathlib import Path

import numpy as np

from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.data import (
    BeamSpec,
    DataColumnMapping,
    PreparedData,
    fit_ready,
    qz_from_two_theta,
)
from xrr_fitter.model.fitting import (
    FitCandidate,
    FitConfig,
    FitSearchResult,
)
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterValue
from xrr_fitter.model.project import DatasetProject, XrrProject
from xrr_fitter.model.structure import LayerSpec, MaterialSpec, StructureSpec


def prepared_data(
    size: int = 32,
    *,
    two_theta_deg: np.ndarray | None = None,
    intensity_raw: np.ndarray | None = None,
    validation_mask: np.ndarray | None = None,
    fit_mask: np.ndarray | None = None,
    beam: BeamSpec | None = None,
) -> PreparedData:
    angles = (
        np.linspace(0.1, 3.2, size)
        if two_theta_deg is None
        else np.asarray(two_theta_deg, dtype=float)
    )
    intensities = (
        np.linspace(1000.0, 10.0, size)
        if intensity_raw is None
        else np.asarray(intensity_raw, dtype=float)
    )
    beam_value = beam or BeamSpec("monochromatic")
    qz, positive = qz_from_two_theta(
        angles,
        beam_value.effective_wavelength_a,
        0.0,
    )
    valid = (
        positive & np.isfinite(intensities)
        if validation_mask is None
        else np.asarray(validation_mask, dtype=bool)
    )
    selected = valid.copy() if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    normalization = float(np.nanmax(intensities[valid])) if np.any(valid) else 1.0
    normalized = intensities / normalization
    groups = tuple((index,) for index in range(size))
    return PreparedData(
        source_path=Path("curve.xy"),
        source_sha256="a" * 64,
        raw_rows=tuple(f"{angle} {value}" for angle, value in zip(angles, intensities)),
        raw_parse_status=("numeric",) * size,
        source_row_groups=groups,
        beam=beam_value,
        import_angle_offset_deg=0.0,
        two_theta_deg=angles,
        intensity_raw=intensities,
        intensity_sigma_raw=None,
        resolution_raw=None,
        qz_a_inv=qz,
        intensity_normalized=normalized,
        intensity_sigma_normalized=None,
        sigma_q_a_inv=None,
        validation_mask=valid,
        fit_mask=selected,
        normalization=normalization,
        r_floor=1e-8,
        fit_ready=fit_ready(angles, qz, selected),
        warnings=(),
        column_mapping=DataColumnMapping(),
    )


def simple_structure() -> StructureSpec:
    air = MaterialSpec("Air", None, None, 0.0j)
    silicon = MaterialSpec("Si", "Si", 2.329)
    silica = MaterialSpec("SiO2", "SiO2", 2.2)
    return StructureSpec(
        fronting=air,
        components=(LayerSpec("film", silica, 20.0, roughness_a=2.0),),
        backing=silicon,
        backing_roughness_a=3.0,
    )


def fit_candidate(candidate_id: str = "candidate-0", objective: float = 1.0) -> FitCandidate:
    qz = np.linspace(0.01, 0.2, 4)
    return FitCandidate(
        candidate_id=candidate_id,
        seed_index=0,
        unit_vector=np.array([0.5]),
        parameters=(ParameterValue("scale", 1.0, 0.5, 1.5),),
        objective=objective,
        valid=True,
        stop_reason="converged",
        nfev=12,
        qz_a_inv=qz,
        model_normalized=np.linspace(1.0, 0.1, 4),
        log_residuals_decades=np.zeros(4),
        weighted_residuals=np.zeros(4),
        expanded_stack=None,
        sld_depth_a=np.array([0.0, 20.0]),
        sld_profile_a2=np.array([0.0, 2e-5]),
        diagnostics=(),
    )


def fit_result(*candidates: FitCandidate) -> FitSearchResult:
    values = candidates or (fit_candidate(),)
    return FitSearchResult(
        parameter_definitions=(),
        candidates=values,
        best_index=0,
        warnings=(),
        child_seeds=(101,),
        stage_summaries=(),
        region_labels=np.zeros(4, dtype=int),
        region_weights=np.ones(4),
    )


def final_fit_result(*candidates: FitCandidate) -> FitResult:
    return FitResult.from_search(
        fit_result(*candidates),
        confidence=ConfidenceClass.TRUSTED,
        uncertainty=None,
        classification_evidence=(),
    )


def dataset_project(
    dataset_id: str = "curve",
    *,
    result: FitResult | None = None,
) -> DatasetProject:
    return DatasetProject(
        dataset_id=dataset_id,
        source_path="curve.xy",
        source_sha256="a" * 64,
        beam=BeamSpec("monochromatic"),
        import_angle_offset_deg=0.0,
        column_mapping=DataColumnMapping(),
        fit_mask=(True,) * 32,
        fit_range_two_theta_deg=(0.1, 3.2),
        structure=simple_structure(),
        instrument=InstrumentSpec(instrument_id="lab"),
        last_valid_result=result,
    )


def project(*datasets: DatasetProject) -> XrrProject:
    values = datasets or (dataset_project(),)
    return XrrProject.new(tuple(values), master_seed=1201)
