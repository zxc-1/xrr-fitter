"""Physical synthetic observations and bounded public-API fitting experiments.

This private tool helper owns observation generation and save/load replay, not
interval mathematics or statistical calibration. All nuisance parameters are
known; the only free parameter is thickness in angstroms. Normalized amplitudes
are member-local and cannot be shared across different reader normalizations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

import xrr_fitter.api as api
from interval_coverage_evidence import _interval, bootstrap_interval, profile_interval
from xrr_fitter.physics.reflectivity import instrument_reflectivity
from xrr_fitter.physics.stack import expand_structure

SEEDS = tuple(range(200))
GROUPS = ("gaussian_regular", "poisson_low", "poisson_regular", "shared_gaussian")
ROBUST_GROUP = "robust_correlated"
TARGET = "component.0.thickness_a"
TRUTH = 100.0
INITIAL = 95.0
BOUNDS = (75.0, 125.0)
THETA = (0.15, 1.8, 80)
BEAM = api.BeamSpec("monochromatic", wavelength_a=1.5406)
INSTRUMENT = api.InstrumentSpec(instrument_id="interval-coverage", footprint_mode="none")
MODES = {
    "gaussian_regular": "gaussian",
    "poisson_low": "poisson",
    "poisson_regular": "poisson",
    "shared_gaussian": "gaussian",
    ROBUST_GROUP: "robust_log",
}
AMPLITUDES = {
    "gaussian_regular": (1.0,),
    "poisson_low": (400.0,),
    "poisson_regular": (40000.0,),
    "shared_gaussian": (1.0, 1.4),
    ROBUST_GROUP: (1.0,),
}


@dataclass(frozen=True, slots=True)
class CoverageCase:
    project: api.XrrProject
    observations: tuple[dict, ...]
    parameter_name: str = TARGET
    truth: float = TRUTH


def _structure() -> api.StructureSpec:
    return api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", api.MaterialSpec("film", None, None, 40e-6 + 0.0j), TRUTH, roughness_a=3.0),),
        api.MaterialSpec("substrate", None, None, 20e-6 + 0.0j),
        backing_roughness_a=3.0,
    )


def _config(group: str, seed: int) -> api.FitConfig:
    base = api.FitConfig.fast(seed)
    samples = 200 if group in {"shared_gaussian", ROBUST_GROUP} else 1
    budget = replace(
        base.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=80,
        local_nfev_per_parameter=20,
        bootstrap_samples=samples,
    )
    return replace(
        base,
        budget=budget,
        noise_model=MODES[group],
        local_workers=1,
        profile_steps=21,
        scale_prior_enabled=False,
    )


def _correlated_noise(rng, size: int) -> np.ndarray:
    rho, sigma = 0.75, 0.04
    values = np.empty(size)
    values[0] = rng.normal(0.0, sigma)
    innovations = rng.normal(0.0, sigma * sqrt(1.0 - rho**2), size - 1)
    for index, innovation in enumerate(innovations, start=1):
        values[index] = rho * values[index - 1] + innovation
    return values


def _observed(mode: str, mean: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray | None, dict]:
    if mode == "gaussian":
        sigma = 0.04 * mean + 2e-5
        return rng.normal(mean, sigma), sigma, {"kind": "independent_gaussian", "sigma": sigma.tolist()}
    if mode == "poisson":
        return rng.poisson(mean), None, {"kind": "independent_poisson", "mean_counts": mean.tolist()}
    noise = _correlated_noise(rng, mean.size)
    return mean * 10.0**noise, None, {"kind": "stationary_ar1_log10", "rho": 0.75, "stationary_sigma_decades": 0.04}


def _source(
    root: Path, group: str, seed: int, member: int, amplitude: float
) -> tuple[Path, dict, api.DataColumnMapping]:
    theta = np.linspace(*THETA)
    mean = instrument_reflectivity(theta, expand_structure(_structure(), BEAM.wavelength_a), BEAM, scale=amplitude)
    rng = np.random.default_rng(np.random.SeedSequence([seed, GROUPS.index(group) if group in GROUPS else 4, member]))
    values, sigma, noise = _observed(MODES[group], mean, rng)
    columns = (2.0 * theta, values) if sigma is None else (2.0 * theta, values, sigma)
    mapping = api.DataColumnMapping(intensity_sigma=None if sigma is None else 2)
    source = root / f"member-{member}.xy"
    np.savetxt(source, np.column_stack(columns), fmt="%.17g")
    data = api.import_data(source, BEAM, column_mapping=mapping, noise_model=MODES[group])
    metadata = {
        "member_index": member,
        "source_sha256": data.source_sha256,
        "normalization": data.normalization,
        "raw_amplitude": amplitude,
        "point_count": theta.size,
        "two_theta_deg": (2.0 * theta).tolist(),
        "raw_mean": mean.tolist(),
        "raw_observations": values.tolist(),
        "zero_observations": int(np.count_nonzero(values == 0.0)),
        "noise": noise,
    }
    return source, metadata, mapping


def _settings(project: api.XrrProject, dataset_id: str, scale: float) -> tuple[api.ParameterSetting, ...]:
    settings = []
    for definition in api.describe_parameters(project, dataset_id):
        if definition.name == TARGET:
            setting = api.ParameterSetting(TARGET, INITIAL, *BOUNDS)
        else:
            initial = scale if definition.name == "instrument.scale" else definition.initial
            setting = api.ParameterSetting(definition.name, initial, initial, initial, locked=True)
        settings.append(setting)
    return tuple(settings)


def _shared(project: api.XrrProject) -> api.XrrProject:
    rule = api.SharingRule(
        "coverage-thickness",
        tuple(api.ParameterReference(dataset.dataset_id, TARGET) for dataset in project.datasets),
    )
    return api.set_batch_mode(api.set_sharing_rules(project, (rule,)), "joint")


def build_case(root: Path, group: str, seed: int) -> CoverageCase:
    """Generate a physical curve, then declare a thickness-only public problem.

    All non-target parameters are known and locked. The reader's 95th percentile
    of leading valid positive intensities divides both data and sigma; locking
    scale to raw_amplitude / that same factor preserves the observation model.
    Shared thickness is
    in angstroms; different member scales are never shared.
    """
    if group not in MODES or seed not in SEEDS:
        raise ValueError("case must use a declared group and a fixed seed in 0..199")
    root.mkdir(parents=True, exist_ok=True)
    project = api.set_fit_config(api.new_project(), _config(group, seed))
    observations = []
    for member, amplitude in enumerate(AMPLITUDES[group]):
        source, metadata, mapping = _source(root, group, seed, member, amplitude)
        project = api.add_dataset(project, source, INSTRUMENT, column_mapping=mapping, beam=BEAM)
        dataset_id = project.datasets[-1].dataset_id
        project = api.set_structure(project, dataset_id, _structure())
        settings = _settings(project, dataset_id, amplitude / metadata["normalization"])
        project = api.set_parameter_settings(project, dataset_id, settings)
        observations.append(metadata)
    if group == "shared_gaussian":
        project = _shared(project)
    return CoverageCase(project, tuple(observations))


def _target_axis(report: api.UncertaintyReport, dataset_id: str) -> str:
    """Resolve only explicit member identity; global labels are not parameters."""
    members = report.parameter_members
    if members is None:
        raise ValueError("joint_parameter_members_unavailable")
    target = api.ParameterReference(dataset_id, TARGET)
    names = [name for name, references in zip(report.correlation_names, members, strict=True) if target in references]
    if len(names) != 1:
        raise ValueError("joint_target_axis_missing_or_ambiguous")
    return names[0]


def _candidate(dataset: api.DatasetProject):
    result = dataset.last_valid_result
    if result is None or result.best_candidate is None:
        raise ValueError(f"missing_fit_result:{dataset.dataset_id}")
    candidate = result.best_candidate
    if not candidate.valid or not isfinite(candidate.objective):
        raise ValueError(f"invalid_candidate:{dataset.dataset_id}:{candidate.stop_reason}")
    return result, candidate


def _saved_interval(project: api.XrrProject, result: api.FitResult, group: str) -> dict:
    report = result.uncertainty
    if report is None:
        return _interval("inference", TARGET, "uncertainty_not_performed")
    if group == "shared_gaussian":
        name = _target_axis(report, project.datasets[0].dataset_id)
        return bootstrap_interval(report.bootstrap_evidence, name)
    if group == ROBUST_GROUP:
        return bootstrap_interval(report.bootstrap_evidence, TARGET)
    profile = next((value for value in report.profiles if value.name == TARGET), None)
    return _interval("profile", TARGET, "target_profile_missing") if profile is None else profile_interval(profile)


def _diagnostics(report: api.UncertaintyReport | None) -> dict:
    if report is None:
        return {"unavailable_reason": "uncertainty_not_performed"}
    covariance = report.covariance_evidence
    return {
        "boundary_hits": list(report.boundary_hits),
        "systematic_residual": report.systematic_residual,
        "residual_autocorrelation": report.residual_autocorrelation,
        "diagnostics": [asdict(value) for value in report.diagnostics],
        "member_residuals": [asdict(value) for value in report.member_residuals],
        "covariance_method": None if covariance is None else covariance.method,
        "covariance_unavailable_reason": None if covariance is None else covariance.unavailable_reason,
        "profiles": [profile_interval(value) for value in report.profiles],
    } | _member_axis(report)


def _member_axis(report: api.UncertaintyReport) -> dict:
    members = report.parameter_members
    return {
        "correlation_names": list(report.correlation_names),
        "parameter_members": None if members is None else [[asdict(member) for member in group] for group in members],
    }


def _fit_summary(project: api.XrrProject, results: list) -> dict:
    result, candidate = results[0]
    estimate = next(value.value for value in candidate.parameters if value.name == TARGET)
    return {
        "estimate": estimate,
        "fit_available": True,
        "candidate_id": candidate.candidate_id,
        "objective": candidate.objective,
        "confidence": result.confidence.value,
        "objective_point_count": sum(sum(dataset.fit_mask) for dataset in project.datasets),
        "stage_nfev_by_dataset": [
            {value.stage: value.total_nfev for value in local_result.stage_summaries} for local_result, _ in results
        ],
        "child_seeds_by_dataset": [list(local_result.child_seeds) for local_result, _ in results],
    }


def _roundtrip_evidence(project: api.XrrProject, root: Path, group: str) -> dict:
    path = root / "fitted.xrrproj.json"
    api.save_project(project, path)
    saved = api.load_project(path)
    result, _candidate_value = _candidate(saved.datasets[0])
    return {
        "interval": _saved_interval(saved, result, group),
        "evidence_source": "api.fit_project -> api.save_project -> api.load_project",
        "diagnostics": _diagnostics(result.uncertainty),
        "failure_stage": None,
    }


def _fit_payload(fixture: CoverageCase, root: Path, group: str) -> dict:
    started = monotonic()
    fitted = api.fit_project(fixture.project)
    timing = {"fit_seconds": monotonic() - started, "fit_warnings": list(fitted.warnings)}
    try:
        if fitted.cancelled:
            raise InterruptedError("public_fit_cancelled")
        results = [_candidate(dataset) for dataset in fitted.updated_project.datasets]
        summary = _fit_summary(fitted.updated_project, results) | timing
    except ValueError as error:
        return failed_outcome(ValueError(f"{error}; public fit warnings: {fitted.warnings}")) | timing
    try:
        evidence = _roundtrip_evidence(fitted.updated_project, root, group)
    except Exception as error:
        evidence = {
            "interval": _interval("persistence", TARGET, f"{type(error).__name__}: {error}"),
            "failure_reason": f"{type(error).__name__}: {error}",
            "failure_stage": "evidence_roundtrip",
            "evidence_source": "public fit succeeded; saved evidence unavailable",
        }
    return summary | evidence


def _structure_definition() -> dict:
    structure = _structure()
    layer = structure.components[0]
    return {
        "fronting_sld_real_a2": structure.fronting.sld_override_a2.real,
        "film_sld_real_a2": layer.material.sld_override_a2.real,
        "backing_sld_real_a2": structure.backing.sld_override_a2.real,
        "all_sld_imag_a2": 0.0,
        "film_thickness_a": layer.thickness_a,
        "film_roughness_a": layer.roughness_a,
        "backing_roughness_a": structure.backing_roughness_a,
    }


def experiment_definition() -> dict:
    """Freeze the observation recipe independently of individual fit success."""
    return {
        "target": TARGET,
        "true_value": TRUTH,
        "unit": "angstrom",
        "initial_value": INITIAL,
        "parameter_bounds": list(BOUNDS),
        "theta_deg_start_stop_count": list(THETA),
        "structure": _structure_definition(),
        "beam": asdict(BEAM),
        "instrument": asdict(INSTRUMENT),
        "all_non_target_parameters_known": True,
        "normalization": (
            "95th percentile of the first min(n_positive, max(20, ceil(0.10 * n_positive))) "
            "valid positive intensities in reader order (1 if none); "
            "lock each scale to raw_amplitude / its own normalization"
        ),
        "observation_seed": "numpy.default_rng(SeedSequence([seed, group_index, member_index]))",
        "groups": {
            group: {
                "group_index": index,
                "noise_model": MODES[group],
                "raw_amplitudes": list(AMPLITUDES[group]),
                "fit_config_seed_0": asdict(_config(group, 0)),
            }
            for index, group in enumerate((*GROUPS, ROBUST_GROUP))
        },
    }


def failed_outcome(error: Exception) -> dict:
    reason = f"{type(error).__name__}: {error}"
    return {
        "estimate": None,
        "fit_available": False,
        "interval": _interval("fit", TARGET, reason),
        "failure_reason": reason,
        "failure_stage": "fit",
        "fit_warnings": [],
    }


def run_case(group: str, seed: int) -> dict:
    """Run one seed once; retain exact failures and remove only owned sources."""
    started = monotonic()
    with TemporaryDirectory(prefix=f"xrr-coverage-{group}-{seed}-") as directory:
        root = Path(directory)
        fixture = build_case(root, group, seed)
        try:
            result = _fit_payload(fixture, root, group)
        except Exception as error:
            result = failed_outcome(error)
        return result | {
            "elapsed_seconds": monotonic() - started,
            "observations": list(fixture.observations),
            "fit_config": asdict(fixture.project.fit_config),
            "parameter_settings_by_dataset": [
                [asdict(setting) for setting in dataset.parameter_settings] for dataset in fixture.project.datasets
            ],
        }
