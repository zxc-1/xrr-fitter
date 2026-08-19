"""Deterministic, bounded automatic-fit recovery fixtures.

Builders stop before fitting so benchmarks time only the public fit call. Run
helpers share one callback-driven work ledger that preserves prefits, joint
attempts, isolated retries, and analysis work without counting projections twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import xrr_fitter.api as api
from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.physics.reflectivity import instrument_reflectivity
from xrr_fitter.physics.stack import expand_structure

BEAM = api.BeamSpec("monochromatic", wavelength_a=1.5406)
INSTRUMENT = api.InstrumentSpec(instrument_id="automatic-recovery", footprint_mode="none")
PRESET = api.MeasurementPreset("automatic-recovery", BEAM, INSTRUMENT)
MASTER_SEED = 1701
AIR = api.MaterialSpec("Air", None, None, 0.0j)
SILICON = api.MaterialSpec("Si", "Si", 2.329)
SILICA = api.MaterialSpec("SiO2", "SiO2", 2.20)
ZIRCONIUM = api.MaterialSpec("Zr", "Zr", 6.52)
MOLYBDENUM = api.MaterialSpec("Mo", "Mo", 10.28)
DIRECT_SLD = api.MaterialSpec("CrSiC", None, None, 24e-6 + 0.0j)

FreeSettings = Mapping[str, tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class RecoveryTarget:
    dataset_id: str
    parameters: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        names = tuple(name for name, _truth in parameters)
        if not self.dataset_id or not parameters:
            raise ValueError("recovery target must identify a dataset and parameters")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("recovery target parameter names must be unique")
        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True, slots=True)
class RecoveryFixture:
    case_id: str
    project: api.XrrProject
    import_batch_id: str
    targets: tuple[RecoveryTarget, ...]

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        if not self.case_id or not self.import_batch_id:
            raise ValueError("recovery fixture must identify its case and import batch")
        if tuple(target.dataset_id for target in targets) != tuple(
            dataset.dataset_id for dataset in self.project.datasets
        ):
            raise ValueError("recovery fixture targets must align with project datasets")
        object.__setattr__(self, "targets", targets)


@dataclass(slots=True)
class _WorkUnit:
    stage_nfev: dict[str, int] = field(default_factory=dict)
    uncertainty: object | None = None
    authoritative: bool = False


class AutomaticWorkLedger:
    """Collect automatic-fit work from ordered checkpoint/result snapshots."""

    def __init__(self) -> None:
        self._units: dict[tuple[object, ...], _WorkUnit] = {}
        self._seen_results: list[object] = []
        self._latest_joint: dict[str, tuple[object, ...]] = {}

    @staticmethod
    def _stage_totals(summaries) -> dict[str, int]:
        totals: dict[str, int] = {}
        for summary in summaries:
            totals[summary.stage] = max(
                totals.get(summary.stage, 0),
                summary.total_nfev,
            )
        return totals

    def _record(
        self,
        key: tuple[object, ...],
        summaries,
        uncertainty,
        *,
        authoritative: bool,
        joint_checkpoint: bool = False,
    ) -> None:
        unit = self._units.get(key)
        if unit is None:
            unit = _WorkUnit()
            self._units[key] = unit
        if unit.authoritative and not authoritative:
            return
        values = self._stage_totals(summaries)
        if joint_checkpoint:
            values["A"] = max(values.get("A", 0), 1)
        for stage, count in values.items():
            unit.stage_nfev[stage] = max(unit.stage_nfev.get(stage, 0), count)
        if uncertainty is not None:
            unit.uncertainty = uncertainty
        if authoritative:
            unit.authoritative = True

    def _observe_checkpoint(self, dataset) -> None:
        checkpoint = dataset.checkpoint
        if checkpoint is None:
            return
        layout = getattr(checkpoint, "joint_layout_fingerprint", "")
        if layout:
            group = dataset.automation.fit_group_id or dataset.dataset_id
            key = ("joint", group, layout)
            self._latest_joint[group] = key
            self._record(
                key,
                checkpoint.stage_summaries,
                None,
                authoritative=False,
                joint_checkpoint=True,
            )
            return
        self._record(
            ("dataset", dataset.dataset_id),
            checkpoint.stage_summaries,
            None,
            authoritative=False,
        )

    def _result_seen(self, result: object) -> bool:
        return any(result is previous for previous in self._seen_results)

    def _remember_result(self, result: object) -> None:
        self._seen_results.append(result)

    def _observe_result(self, dataset) -> None:
        result = dataset.last_valid_result
        if result is None or self._result_seen(result):
            return
        role = dataset.automation.role
        status = dataset.automation.status
        if role is api.AutomaticRole.JOINT and status is api.AutomaticStatus.REFINING:
            key = ("dataset", dataset.dataset_id)
        elif role is api.AutomaticRole.SINGLE:
            key = ("dataset", dataset.dataset_id)
        elif role is api.AutomaticRole.JOINT:
            group = dataset.automation.fit_group_id or dataset.dataset_id
            key = self._latest_joint.get(group, ("joint-result", group))
        elif role is api.AutomaticRole.ISOLATED_RETRY:
            key = ("isolated", dataset.dataset_id)
        else:
            self._remember_result(result)
            return
        self._record(
            key,
            result.stage_summaries,
            result.uncertainty,
            authoritative=True,
        )
        self._remember_result(result)

    def observe(self, project) -> None:
        for dataset in project.datasets:
            self._observe_checkpoint(dataset)
        for dataset in project.datasets:
            self._observe_result(dataset)

    def metrics(self) -> tuple[tuple[tuple[str, int], ...], int, int]:
        stages: dict[str, int] = {}
        bootstrap = 0
        profiles = 0
        for unit in self._units.values():
            for stage, count in unit.stage_nfev.items():
                stages[stage] = stages.get(stage, 0) + count
            if unit.uncertainty is not None:
                bootstrap += int(unit.uncertainty.bootstrap_performed)
                profiles += len(unit.uncertainty.profiles)
        return tuple(sorted(stages.items())), bootstrap, profiles


@dataclass(frozen=True, slots=True)
class RecoveryRun:
    case_id: str
    project: api.XrrProject
    summary: api.AutomaticResultSummary
    stage_nfev: tuple[tuple[str, int], ...]
    bootstrap_count: int
    profile_count: int

    @property
    def work_signature(self) -> tuple[object, ...]:
        return self.stage_nfev, self.bootstrap_count, self.profile_count


def parameter_value(dataset: api.DatasetProject, name: str) -> float:
    result = dataset.last_valid_result
    candidate = None if result is None else result.best_candidate
    if candidate is None:
        raise ValueError(f"dataset has no publishable candidate: {dataset.dataset_id}")
    try:
        return next(value.value for value in candidate.parameters if value.name == name)
    except StopIteration as error:
        raise ValueError(f"candidate parameter is missing: {name}") from error


def _structure(
    material: api.MaterialSpec,
    thickness_a: float,
    *,
    density_scale: float = 1.0,
    roughness_a: float = 3.0,
) -> api.StructureSpec:
    return api.StructureSpec(
        AIR,
        (
            api.LayerSpec(
                material.name,
                material,
                thickness_a,
                density_scale=density_scale,
                roughness_a=roughness_a,
            ),
            api.LayerSpec("SiO2 native oxide", SILICA, 10.0, roughness_a=3.0),
        ),
        SILICON,
    )


def _write_curve(
    path: Path,
    structure: api.StructureSpec,
    *,
    seed: int,
    noise_decades: float,
    systematic_decades: float = 0.0,
    point_count: int = 140,
    theta_min_deg: float = 0.04,
    theta_max_deg: float = 3.2,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    theta_deg = np.linspace(theta_min_deg, theta_max_deg, point_count)
    intensity = instrument_reflectivity(
        theta_deg,
        expand_structure(structure, BEAM.wavelength_a),
        BEAM,
    )
    noise = np.random.default_rng(seed).normal(0.0, noise_decades, intensity.size)
    if systematic_decades:
        noise += systematic_decades * np.sin(np.linspace(0.0, 8.0 * np.pi, intensity.size))
    sampled = np.maximum(intensity * np.power(10.0, noise), 1e-14)
    path.write_bytes(xy_bytes(2.0 * theta_deg, sampled))
    return path


def _import_project(
    paths: tuple[Path, ...],
    batch_id: str,
) -> api.XrrProject:
    preview = api.preview_import_batch(paths, PRESET, batch_id)
    imported = api.import_dataset_batch(api.new_project(), preview)
    if imported.failures or len(imported.imported_dataset_ids) != len(paths):
        raise AssertionError(f"automatic recovery fixture import failed: {imported.failures}")
    return imported.updated_project


def _settings(
    project: api.XrrProject,
    dataset_id: str,
    free: FreeSettings,
) -> tuple[api.ParameterSetting, ...]:
    values = []
    for definition in api.describe_parameters(project, dataset_id):
        override = free.get(definition.name)
        if override is None:
            initial, lower, upper = (
                definition.initial,
                definition.lower,
                definition.upper,
            )
        else:
            initial, lower, upper = override
        values.append(
            api.ParameterSetting(
                definition.name,
                initial,
                lower,
                upper,
                locked=override is None,
            )
        )
    return tuple(values)


def _configure(
    project: api.XrrProject,
    free: FreeSettings,
    *,
    workers: int,
) -> api.XrrProject:
    updated = project
    for dataset in project.datasets:
        updated = api.set_parameter_settings(
            updated,
            dataset.dataset_id,
            _settings(updated, dataset.dataset_id, free),
        )
    budget = replace(
        updated.fit_config.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=100,
        local_nfev_per_parameter=20,
        bootstrap_samples=1,
    )
    config = replace(
        api.FitConfig.fast(MASTER_SEED),
        budget=budget,
        local_workers=workers,
        scale_prior_enabled=False,
    )
    return replace(updated, fit_config=config)


def _completed_run(
    case_id: str,
    project: api.XrrProject,
    batch_id: str,
) -> RecoveryRun:
    ledger = AutomaticWorkLedger()
    result = api.fit_automatically(
        project,
        batch_id,
        checkpoint_callback=ledger.observe,
    )
    updated = result.updated_project
    ledger.observe(updated)
    stage_nfev, bootstrap_count, profile_count = ledger.metrics()
    return RecoveryRun(
        case_id,
        updated,
        api.summarize_automatic_results(updated, batch_id),
        stage_nfev,
        bootstrap_count,
        profile_count,
    )


def build_direct_sld_project(root: Path) -> api.XrrProject:
    """Build the bounded public-API fixture used by recovery and spawn tests."""
    return build_direct_sld_fixture(root).project


def build_direct_sld_fixture(root: Path) -> RecoveryFixture:
    path = _write_curve(
        root / "D1 CrSiC.xy",
        _structure(DIRECT_SLD, 130.0),
        seed=1,
        noise_decades=0.015,
        point_count=160,
    )
    batch_id = "automatic-recovery-direct-sld"
    project = _configure(
        _import_project((path,), batch_id),
        {
            "component.0.thickness_a": (100.0, 80.0, 180.0),
            "component.0.sld_real_a2": (20e-6, 15e-6, 30e-6),
        },
        workers=1,
    )
    return RecoveryFixture(
        "direct-sld",
        project,
        batch_id,
        (
            RecoveryTarget(
                "D1",
                (
                    ("component.0.thickness_a", 130.0),
                    ("component.0.sld_real_a2", 24e-6),
                ),
            ),
        ),
    )


def run_direct_sld_recovery(root: Path) -> RecoveryRun:
    fixture = build_direct_sld_fixture(root)
    return _completed_run(fixture.case_id, fixture.project, fixture.import_batch_id)


def build_ambiguous_low_angle_fixture(root: Path) -> RecoveryFixture:
    path = _write_curve(
        root / "A1 Mo.xy",
        _structure(MOLYBDENUM, 220.0, roughness_a=5.0),
        seed=16000,
        noise_decades=0.01,
        point_count=300,
        theta_min_deg=0.03,
        theta_max_deg=0.22,
    )
    batch_id = "automatic-recovery-ambiguous-low-angle"
    project = _configure(
        _import_project((path,), batch_id),
        {
            "component.0.thickness_a": (180.0, 80.0, 320.0),
            "component.0.roughness_a": (4.0, 0.0, 20.0),
        },
        workers=1,
    )
    return RecoveryFixture(
        "ambiguous-low-angle",
        project,
        batch_id,
        (
            RecoveryTarget(
                "A1",
                (
                    ("component.0.thickness_a", 220.0),
                    ("component.0.roughness_a", 5.0),
                ),
            ),
        ),
    )


def run_ambiguous_low_angle_recovery(root: Path) -> RecoveryRun:
    fixture = build_ambiguous_low_angle_fixture(root)
    return _completed_run(fixture.case_id, fixture.project, fixture.import_batch_id)


def build_shared_local_fixture(root: Path) -> RecoveryFixture:
    thicknesses = (90.0, 100.0, 110.0, 120.0)
    paths = tuple(
        _write_curve(
            root / f"P{index} Zr.xy",
            _structure(ZIRCONIUM, thickness, density_scale=0.93),
            seed=100 + index,
            noise_decades=0.01,
        )
        for index, thickness in enumerate(thicknesses, 1)
    )
    batch_id = "automatic-recovery-shared-local"
    project = _configure(
        _import_project(paths, batch_id),
        {
            "component.0.thickness_a": (100.0, 60.0, 150.0),
            "component.0.density_scale": (0.90, 0.70, 1.10),
        },
        workers=min(4, len(paths)),
    )
    return RecoveryFixture(
        "shared-local",
        project,
        batch_id,
        tuple(
            RecoveryTarget(
                f"P{index}",
                (
                    ("component.0.thickness_a", thickness),
                    ("component.0.density_scale", 0.93),
                ),
            )
            for index, thickness in enumerate(thicknesses, 1)
        ),
    )


def run_shared_local_recovery(root: Path) -> RecoveryRun:
    fixture = build_shared_local_fixture(root)
    return _completed_run(fixture.case_id, fixture.project, fixture.import_batch_id)


def build_two_point_joint_project(root: Path) -> api.XrrProject:
    thicknesses = (90.0, 110.0)
    paths = tuple(
        _write_curve(
            root / f"J{index} Zr.xy",
            _structure(ZIRCONIUM, thickness, density_scale=0.93),
            seed=300 + index,
            noise_decades=0.01,
        )
        for index, thickness in enumerate(thicknesses, 1)
    )
    return _configure(
        _import_project(paths, "automatic-recovery-joint-spawn"),
        {
            "component.0.thickness_a": (100.0, 60.0, 150.0),
            "component.0.density_scale": (0.90, 0.70, 1.10),
        },
        workers=len(paths),
    )


def build_isolated_outlier_fixture(root: Path) -> RecoveryFixture:
    paths = tuple(
        _write_curve(
            root / f"P{index} Zr.xy",
            _structure(ZIRCONIUM, 100.0),
            seed=199 + index,
            noise_decades=0.01,
            systematic_decades=0.04 if index == 4 else 0.0,
        )
        for index in range(1, 5)
    )
    batch_id = "automatic-recovery-isolated-outlier"
    project = _configure(
        _import_project(paths, batch_id),
        {"component.0.thickness_a": (100.0, 60.0, 150.0)},
        workers=min(4, len(paths)),
    )
    return RecoveryFixture(
        "isolated-outlier",
        project,
        batch_id,
        tuple(
            RecoveryTarget(
                f"P{index}",
                (("component.0.thickness_a", 100.0),),
            )
            for index in range(1, 5)
        ),
    )


def run_isolated_outlier_recovery(root: Path) -> RecoveryRun:
    fixture = build_isolated_outlier_fixture(root)
    return _completed_run(fixture.case_id, fixture.project, fixture.import_batch_id)


def build_roughness_release_fixture(root: Path) -> RecoveryFixture:
    roughnesses = (2.0, 3.0, 8.0, 9.0)
    paths = tuple(
        _write_curve(
            root / f"R{index} Zr.xy",
            _structure(ZIRCONIUM, 100.0, roughness_a=roughness),
            seed=index,
            noise_decades=0.015,
        )
        for index, roughness in enumerate(roughnesses, 1)
    )
    batch_id = "automatic-recovery-roughness-release"
    project = _configure(
        _import_project(paths, batch_id),
        {"component.0.roughness_a": (3.0, 0.0, 40.0)},
        workers=min(4, len(paths)),
    )
    return RecoveryFixture(
        "roughness-release",
        project,
        batch_id,
        tuple(
            RecoveryTarget(
                f"R{index}",
                (("component.0.roughness_a", roughness),),
            )
            for index, roughness in enumerate(roughnesses, 1)
        ),
    )


def run_roughness_release_recovery(root: Path) -> RecoveryRun:
    fixture = build_roughness_release_fixture(root)
    return _completed_run(fixture.case_id, fixture.project, fixture.import_batch_id)
