"""Physical Poisson generators and current public-API save/load experiments.

Only generators live here: product search, refitting, calibration and interval
eligibility are never reimplemented. Observation files and input/result projects
remain under the caller's exclusively reserved per-seed directory, including
failed cases. Non-holdout generator unit tests use seed numbers outside every
registered range; the experiment entry point checks the complete fixed list.
"""

from __future__ import annotations

import traceback
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic

import numpy as np

import interval_coverage_cases as coverage
import poisson_diagnostic_evidence as evidence
import poisson_diagnostic_protocol as protocol
import xrr_fitter.api as api
from xrr_fitter.physics.reflectivity import instrument_reflectivity, qz_from_theta_deg
from xrr_fitter.physics.stack import expand_structure

TARGET = coverage.TARGET


def truth_structure(group: str) -> api.StructureSpec:
    structure = coverage._structure()
    if group != "surface":
        return structure
    cap = protocol.INJECTIONS["surface"]
    layer = api.LayerSpec(
        "cap",
        api.MaterialSpec("cap", None, None, cap["sld_real_a2"] + 0.0j),
        cap["thickness_a"],
        roughness_a=cap["roughness_a"],
    )
    return replace(structure, components=(layer, *structure.components))


def mean_counts(theta: np.ndarray, amplitude: float, group: str) -> np.ndarray:
    stack = expand_structure(truth_structure(group), coverage.BEAM.wavelength_a)
    spill = protocol.INJECTIONS["footprint"]["spill_angle_deg"] if group == "footprint" else 0.0
    mean = instrument_reflectivity(theta, stack, coverage.BEAM, scale=amplitude, footprint_spill_angle_deg=spill)
    if group == "background":
        q = qz_from_theta_deg(theta, coverage.BEAM.wavelength_a)
        mean = mean + protocol.INJECTIONS["background"]["slope_q_over_qmax"] * q / q.max()
    return mean


def _block_counts(mean: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    block_size = protocol.INJECTIONS["acf"]["block_size"]
    fraction = protocol.INJECTIONS["acf"]["shared_fraction"]
    output = np.empty(mean.size, dtype=np.int64)
    common_means = []
    for start in range(0, mean.size, block_size):
        block = mean[start : start + block_size]
        common = fraction * float(block.min())
        output[start : start + block_size] = rng.poisson(common) + rng.poisson(block - common)
        common_means.append(common)
    return output, {
        "kind": "shared_poisson_blocks",
        "block_size": block_size,
        "shared_fraction": fraction,
        "block_common_means": common_means,
        "marginal_distribution": "Poisson(mu_i)",
    }


def draw_counts(
    mean: np.ndarray, group: str, *, scenario_id: int, seed: int, member_id: int
) -> tuple[np.ndarray, dict]:
    mean = np.asarray(mean, dtype=float)
    if mean.ndim != 1 or mean.size == 0 or np.any(~np.isfinite(mean)) or np.any(mean < 0.0):
        raise ValueError("Poisson generation requires nonempty finite nonnegative means")
    sequence = [20260912, scenario_id, seed, member_id]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(sequence)))
    if group == "acf":
        values, recipe = _block_counts(mean, rng)
    else:
        values, recipe = rng.poisson(mean), {"kind": "independent_poisson", "marginal_distribution": "Poisson(mu_i)"}
    return values, recipe | {"rng": "numpy.Generator(PCG64)", "seed_sequence": sequence}


def _source(root: Path, group: str, seed: int, member: int, amplitude: float) -> tuple[Path, dict]:
    declaration = protocol.scenario(group)
    theta = np.linspace(*declaration["theta_deg_start_stop_count"])
    mean = mean_counts(theta, amplitude, group)
    values, noise = draw_counts(mean, group, scenario_id=declaration["scenario_id"], seed=seed, member_id=member)
    source = root / f"member-{member}.xy"
    with source.open("x", encoding="utf-8") as stream:
        np.savetxt(stream, np.column_stack((2.0 * theta, values)), fmt="%.17g")
    data = api.import_data(source, coverage.BEAM, column_mapping=api.DataColumnMapping(), noise_model="poisson")
    metadata = {
        "member_index": member,
        "source_sha256": data.source_sha256,
        "normalization": data.normalization,
        "raw_amplitude": amplitude,
        "point_count": theta.size,
        "two_theta_deg": (2.0 * theta).tolist(),
        "raw_mean": mean.tolist(),
        "raw_observations": values.tolist(),
        "zero_observations": int(np.count_nonzero(values == 0)),
        "noise": noise,
    }
    return source, metadata


def _settings(project, dataset_id: str, amplitude: float, normalization: float, group: str) -> tuple:
    settings = coverage._settings(project, dataset_id, amplitude / normalization)
    if group != "null_joint":
        return settings
    factors = protocol.scenario(group)
    scale = amplitude / normalization
    value = api.ParameterSetting(
        "instrument.scale",
        factors["scale_initial_factor"] * scale,
        *(factor * scale for factor in factors["scale_bound_factors"]),
    )
    return tuple(value if item.name == "instrument.scale" else item for item in settings)


def build_observations(root: Path, group: str, seed: int) -> coverage.CoverageCase:
    """Generator primitive; experiment callers must use the checked build_case."""
    declaration = protocol.scenario(group)
    root.mkdir(parents=True, exist_ok=True)
    project = api.set_fit_config(api.new_project(), protocol.config(group, seed))
    observations = []
    for member, amplitude in enumerate(declaration["raw_amplitudes"]):
        source, metadata = _source(root, group, seed, member, amplitude)
        project = api.add_dataset(
            project, source, coverage.INSTRUMENT, column_mapping=api.DataColumnMapping(), beam=coverage.BEAM
        )
        dataset_id = project.datasets[-1].dataset_id
        project = api.set_structure(project, dataset_id, coverage._structure())
        settings = _settings(project, dataset_id, amplitude, metadata["normalization"], group)
        project = api.set_parameter_settings(project, dataset_id, settings)
        observations.append(metadata)
    if group == "null_joint":
        project = coverage._shared(project)
    return coverage.CoverageCase(project, tuple(observations))


def build_case(root: Path, group: str, seed: int) -> coverage.CoverageCase:
    declaration = protocol.scenario(group)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed not in declaration["seeds"]:
        raise ValueError("case must use its preregistered fixed seed")
    if group == "replay_low":
        return coverage.build_case(root, "poisson_low", seed)
    return build_observations(root, group, seed)


def _save_inputs(fixture, root: Path) -> None:
    with (root / "observations.json").open("xb") as stream:
        stream.write(
            protocol.canonical(
                {
                    "observations": list(fixture.observations),
                    "fit_config": asdict(fixture.project.fit_config),
                    "parameter_settings_by_dataset": [
                        [asdict(value) for value in dataset.parameter_settings] for dataset in fixture.project.datasets
                    ],
                }
            )
        )
    api.save_project(fixture.project, root / "input.xrrproj.json")


def _fit_case(fixture, root: Path, group: str) -> dict:
    started = monotonic()
    try:
        fitted = api.fit_project(fixture.project)
    except Exception as error:
        return evidence.failed_outcome(error) | {
            "fit_seconds": monotonic() - started,
            "traceback": traceback.format_exc(),
        }
    timing = {"fit_seconds": monotonic() - started, "fit_warnings": list(fitted.warnings)}
    if fitted.cancelled:
        return evidence.failed_outcome(InterruptedError("public_fit_cancelled")) | timing
    try:
        results = [coverage._candidate(dataset) for dataset in fitted.updated_project.datasets]
        summary = coverage._fit_summary(fitted.updated_project, results)
    except ValueError as error:
        failure = ValueError(f"{error}; public fit warnings: {fitted.warnings}")
        return evidence.failed_outcome(failure) | timing
    try:
        target = root / "fitted.xrrproj.json"
        api.save_project(fitted.updated_project, target)
        saved = api.load_project(target)
        record = evidence.saved_result(saved, target=protocol.scenario(group)["target_column"])
    except Exception as error:
        record = evidence.failed_outcome(error, stage="evidence_roundtrip") | summary
        record["traceback"] = traceback.format_exc()
    return record | timing


def run_case(root: Path, group: str, seed: int) -> dict:
    """Attempt exactly once; the caller seals even failures, never resampling."""
    started = monotonic()
    stage = "generation"
    try:
        fixture = build_case(root / "input", group, seed)
        stage = "input_persistence"
        _save_inputs(fixture, root)
        stage = "fit"
        result = _fit_case(fixture, root, group)
    except Exception as error:
        result = evidence.failed_outcome(error, stage=stage) | {"traceback": traceback.format_exc()}
    return result | {"elapsed_seconds": monotonic() - started}
